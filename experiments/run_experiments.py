"""
Main runner: load data, build index, run B1-B6, compute metrics, adversarial, update main.tex.
CPU-only. Single-node prototype.
"""
import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd

# Ensure project root on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Use Hugging Face token when set to avoid unauthenticated-request warnings
_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _hf_token:
    try:
        from huggingface_hub import login
        login(token=_hf_token)
    except Exception as e:
        print(f"[run_experiments] HF login skipped: {e}")

from experiments.load_data import load_config, load_nq, load_enron, get_corpus_for_retrieval, get_qa_prompts
from experiments.tagging import doc_label, segment_label, build_token_min_clearance, write_manual_review_sample
from experiments.retrieval import build_index, retrieve
from experiments.baselines import run_all_baselines
from experiments.metrics import (
    restricted_content_exposure_rate,
    policy_violation_rate,
    latency_stats,
    blocked_rate,
)
from experiments.adversarial import get_adversarial_prompts, run_adversarial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/config.yaml")
    parser.add_argument("--quick", action="store_true", help="Use small sample sizes for a fast test run")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.quick:
        config["nq_samples"] = min(config.get("nq_samples", 400), 30)
        config["enron_samples"] = min(config.get("enron_samples", 1500), 50)
        config["k_runs"] = 1
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    output_dir = config.get("output_dir", "experiments/outputs")
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    # Data
    nq = load_nq(config)
    enron = load_enron(config)
    if not nq and not enron:
        print("No data loaded. Set nq_samples > 0 or enron_csv_path / enron_samples.")
        sys.exit(1)
    corpus = get_corpus_for_retrieval(nq, enron)
    n_nq = len(nq)
    n_enron = len(enron)
    max_prompts = min(20 if args.quick else 150, max(len(nq), len(enron)) or 50)
    prompts = get_qa_prompts(nq, enron, max_prompts, seed)

    # Tagging
    doc_labels = {}
    tenant_ids = {}
    for c in corpus:
        doc_labels[c["doc_id"]] = doc_label(c["text"])
        tenant_ids[c["doc_id"]] = hash(c["doc_id"]) % config.get("tenants", 3)
    docs_for_review = [{"doc_id": c["doc_id"], "text": c["text"], "label": doc_labels[c["doc_id"]]} for c in corpus[:500]]
    write_manual_review_sample(docs_for_review, os.path.join(output_dir, "metrics", "manual_review_sample.csv"), n_per_level=8)

    dist = {}
    for c in corpus:
        l = doc_labels[c["doc_id"]]
        dist[l] = dist.get(l, 0) + 1
    total = sum(dist.values())
    dist_pct = {k: round(100 * v / total, 1) for k, v in dist.items()} if total else {}

    # Model (CPU, GPU, or TPU when available, e.g. Colab)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif os.environ.get("COLAB_TPU_ADDR"):
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        except Exception:
            pass
    model_name = config.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=_hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32, token=_hf_token
    )
    model.eval()
    if hasattr(model, "to"):
        model = model.to(device)
    print(f"Using device: {('TPU' if 'xla' in str(type(device)) else ('GPU' if device == 'cuda' else 'CPU'))} ({device})")
    token_min_clearance = build_token_min_clearance(tokenizer)

    # Index
    retrieval_index = build_index(corpus, doc_labels, tenant_ids) if corpus else None

    # Run baselines
    roles = config.get("roles", ["analyst", "support", "engineer"])
    clearances = config.get("clearances", ["Public", "Internal", "Confidential", "Restricted"])
    tenants = config.get("tenants", 3)
    all_results = run_all_baselines(
        config, model, tokenizer, retrieval_index, docs_for_review, prompts,
        tagger=segment_label, token_min_clearance=token_min_clearance,
        roles=roles, clearances=clearances, tenants=tenants, output_dir=output_dir,
    )

    # Metrics
    def agg(responses, clearances_list, roles_list, latencies, blocked_list=None):
        exp = restricted_content_exposure_rate(responses, clearances_list)
        viol = policy_violation_rate(responses, roles_list)
        lat = latency_stats(latencies)
        base_lat = latencies[0] if latencies else 1.0
        rel_lat = lat["mean"] / base_lat if base_lat else 1.0
        blk = blocked_rate(blocked_list or []) if blocked_list else 0.0
        return exp, viol, rel_lat, blk

    results_list = []
    base_latencies = [r["latency"] for r in all_results["B1"]]
    for bname in ["B1", "B2", "B3", "B4", "B5", "B6"]:
        R = all_results[bname]
        responses = [r.get("response", "") for r in R]
        clearances_list = [r.get("clearance", "Public") for r in R]
        roles_list = [r.get("role", "analyst") for r in R]
        latencies = [r.get("latency", 0) for r in R]
        blocked_list = [r.get("blocked", False) for r in R] if bname in ("B2", "B5") else None
        exp, viol, rel_lat, blk = agg(responses, clearances_list, roles_list, latencies, blocked_list)
        base_mean = sum(base_latencies) / len(base_latencies) if base_latencies else 1.0
        rel_lat = (sum(latencies) / len(latencies)) / base_mean if base_mean and latencies else 1.0
        results_list.append({"baseline": bname, "exposure": round(exp, 3), "violation": round(viol, 3), "latency_rel": round(rel_lat, 3), "blocked_rate": round(blk, 3)})

    df = pd.DataFrame(results_list)
    df.to_csv(os.path.join(output_dir, "metrics", "results.csv"), index=False)

    # Adversarial (simplified: use same model functions)
    def model_b12(q, role, clearance, tenant_id):
        from experiments.baselines import run_b1
        r, _ = run_b1(model, tokenizer, q, retrieval_index, config.get("top_k", 3),
                      config.get("max_new_tokens", 128), config.get("do_sample", False), config.get("temperature", 0.2))
        return r

    def model_b6(q, role, clearance, tenant_id):
        from experiments.baselines import run_b6
        r, _ = run_b6(model, tokenizer, q, retrieval_index, config.get("top_k", 3),
                      config.get("max_new_tokens", 128), config.get("do_sample", False), config.get("temperature", 0.2),
                      role, clearance, tenant_id, token_min_clearance)
        return r

    adv_results = run_adversarial(model_b12, model_b6, segment_label, n_runs=1)
    adv_table = {}
    for cat in ["injection", "override", "paraphrase", "canary"]:
        adv_table[cat.replace(" ", "_")] = adv_results[cat]

    summary = {
        "config": {k: v for k, v in config.items() if k != "output_dir"},
        "data": {"n_nq": n_nq, "n_enron": n_enron, "n_prompts": len(prompts), "n_corpus": len(corpus), "distribution": dist_pct},
        "results": results_list,
        "adversarial": adv_results,
        "model_name": model_name,
        "hardware": "GPU" if device == "cuda" else ("TPU" if "xla" in str(type(device)) else "CPU"),
    }
    with open(os.path.join(output_dir, "metrics", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Update main.tex
    try:
        from experiments.update_latex import update_latex
        update_latex(
            main_tex_path=os.path.join(PROJECT_ROOT, "main.tex"),
            summary=summary,
            results_df=df,
            adv_table=adv_table,
        )
        print("Updated main.tex with measured values.")
    except Exception as e:
        print(f"update_latex failed: {e}")

    print("Done. Results in", os.path.join(output_dir, "metrics"))


if __name__ == "__main__":
    main()
