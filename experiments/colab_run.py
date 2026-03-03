"""
Colab-compatible experiment runner: NQ_N=400, Enron_N=1500, K=3, TF-IDF retrieval.
TPU/GPU/CPU transparent. Writes results.csv (mean ± std), summary.json, adversarial.csv.
Do not fabricate numbers; all metrics computed from actual outputs.
"""
from __future__ import print_function
import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _hf_token:
    try:
        from huggingface_hub import login
        login(token=_hf_token)
    except Exception as e:
        print("HF login skipped:", e)

# Optional: try natural_questions_open (HF); fallback to google-research-datasets/nq_open
def _load_nq_with_fallback(config):
    from experiments.load_data import load_nq
    nq = load_nq(config)
    if nq:
        return nq
    if config.get("nq_samples", 400) <= 0:
        return []
    try:
        import datasets as hf_datasets
        for name in ["natural_questions_open", "google-research-datasets/nq_open"]:
            try:
                ds = hf_datasets.load_dataset(name, split="train", token=_hf_token)
                rng = random.Random(config.get("seed", 42))
                n = min(config.get("nq_samples", 400), len(ds))
                indices = rng.sample(range(len(ds)), n)
                out = []
                for i in indices:
                    row = ds[int(i)]
                    q = row.get("question", row.get("question_text", ""))
                    ans = row.get("answer", row.get("answers", ""))
                    if isinstance(ans, list): ans = ans[0] if ans else ""
                    ctx = row.get("context", row.get("context_text", ""))
                    if isinstance(ctx, list): ctx = ctx[0] if ctx else ""
                    out.append({"question": str(q), "answer": str(ans), "context": str(ctx), "doc_id": f"nq_{i}"})
                return out
            except Exception:
                continue
    except ImportError:
        pass
    return nq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/config.yaml")
    parser.add_argument("--quick", action="store_true", help="Small N and K=1 for fast test")
    args = parser.parse_args()

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

    config = load_config(args.config)
    config.setdefault("nq_samples", 400)
    config.setdefault("enron_samples", 1500)
    config.setdefault("k_runs", 3)
    config.setdefault("seed", 42)
    if args.quick:
        config["nq_samples"] = min(config["nq_samples"], 30)
        config["enron_samples"] = min(config["enron_samples"], 50)
        config["k_runs"] = 1

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    k_runs = config["k_runs"]
    output_dir = config.get("output_dir", "experiments/outputs")
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    # Data: NQ (try natural_questions_open then nq_open), Enron (HF or CSV)
    nq = _load_nq_with_fallback(config)
    if not nq:
        nq = load_nq(config)
    enron = load_enron(config)
    if not nq and not enron:
        print("No data loaded. Set nq_samples > 0 or enron_csv_path / enron_samples.")
        sys.exit(1)

    corpus = get_corpus_for_retrieval(nq, enron)
    n_nq = len(nq)
    n_enron = len(enron)
    max_prompts = min(150, max(len(nq), len(enron)) or 50)
    if args.quick:
        max_prompts = min(20, max_prompts)
    prompts = get_qa_prompts(nq, enron, max_prompts, seed)

    # Tagging and observed distribution
    doc_labels = {}
    tenant_ids = {}
    for c in corpus:
        doc_labels[c["doc_id"]] = doc_label(c["text"])
        tenant_ids[c["doc_id"]] = hash(c["doc_id"]) % config.get("tenants", 3)
    docs_for_review = [{"doc_id": c["doc_id"], "text": c["text"], "label": doc_labels[c["doc_id"]]} for c in corpus[:500]]
    write_manual_review_sample(docs_for_review, os.path.join(output_dir, "metrics", "manual_review_sample.csv"), n_per_level=8)

    dist_counts = {}
    for c in corpus:
        l = doc_labels[c["doc_id"]]
        dist_counts[l] = dist_counts.get(l, 0) + 1
    total_docs = sum(dist_counts.values())
    dist_pct = {k: round(100 * v / total_docs, 1) for k, v in dist_counts.items()} if total_docs else {}
    print("Observed class distribution (counts):", dist_counts)
    print("Observed class distribution (%):", dist_pct)

    # Device: TPU > GPU > CPU
    import torch
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif os.environ.get("COLAB_TPU_ADDR"):
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        except Exception:
            pass
    hardware = "TPU" if "xla" in str(type(device)) else ("GPU" if device == "cuda" else "CPU")
    print("Device:", hardware, device)

    model_name = config.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=_hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32, token=_hf_token
    )
    model.eval()
    if hasattr(model, "to"):
        model = model.to(device)
    token_min_clearance = build_token_min_clearance(tokenizer)
    retrieval_index = build_index(corpus, doc_labels, tenant_ids) if corpus else None

    roles = config.get("roles", ["analyst", "support", "engineer"])
    clearances = config.get("clearances", ["Public", "Internal", "Confidential", "Restricted"])
    tenants = config.get("tenants", 3)

    # Single pass: run_all_baselines already runs k_runs times (same prompts, different role/clearance per run)
    random.seed(seed)
    np.random.seed(seed)
    all_results = run_all_baselines(
        config, model, tokenizer, retrieval_index, docs_for_review, prompts,
        tagger=segment_label, token_min_clearance=token_min_clearance,
        roles=roles, clearances=clearances, tenants=tenants, output_dir=output_dir,
    )
    n_p = len(prompts)
    n_p = len(prompts)
    k_runs = config["k_runs"]

    # Chunk by run: each baseline has k_runs * n_p responses; chunk into k_runs parts
    run_metrics = []
    for run_id in range(k_runs):
        start = run_id * n_p
        end = start + n_p
        run_record = {"run_id": run_id}
        base_latencies = [r["latency"] for r in all_results["B1"][start:end]]
        base_mean = sum(base_latencies) / len(base_latencies) if base_latencies else 1.0
        for bname in ["B1", "B2", "B3", "B4", "B5", "B6"]:
            R = all_results[bname][start:end]
            responses = [r.get("response", "") for r in R]
            clearances_list = [r.get("clearance", "Public") for r in R]
            roles_list = [r.get("role", "analyst") for r in R]
            latencies = [r.get("latency", 0) for r in R]
            blocked_list = [r.get("blocked", False) for r in R] if bname in ("B2", "B5") else None
            exp = restricted_content_exposure_rate(responses, clearances_list)
            viol = policy_violation_rate(responses, roles_list)
            rel_lat = (sum(latencies) / len(latencies)) / base_mean if base_mean and latencies else 1.0
            blk = blocked_rate(blocked_list or []) if blocked_list else 0.0
            run_record[bname] = {"exposure": exp, "violation": viol, "latency_rel": rel_lat, "blocked_rate": blk}
        run_metrics.append(run_record)

    # Write raw JSONL (last run only for readability)
    for bname in ["B1", "B2", "B3", "B4", "B5", "B6"]:
        path = os.path.join(output_dir, "raw", f"{bname}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in all_results[bname][(k_runs - 1) * n_p : k_runs * n_p]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Aggregate mean ± std
    results_list = []
    for bname in ["B1", "B2", "B3", "B4", "B5", "B6"]:
        exposures = [m[bname]["exposure"] for m in run_metrics]
        violations = [m[bname]["violation"] for m in run_metrics]
        latencies = [m[bname]["latency_rel"] for m in run_metrics]
        blocked_rates = [m[bname]["blocked_rate"] for m in run_metrics]
        results_list.append({
            "baseline": bname,
            "exposure": round(float(np.mean(exposures)), 4),
            "exposure_std": round(float(np.std(exposures)), 4) if len(exposures) > 1 else 0.0,
            "violation": round(float(np.mean(violations)), 4),
            "violation_std": round(float(np.std(violations)), 4) if len(violations) > 1 else 0.0,
            "latency_rel": round(float(np.mean(latencies)), 4),
            "latency_std": round(float(np.std(latencies)), 4) if len(latencies) > 1 else 0.0,
            "blocked_rate": round(float(np.mean(blocked_rates)), 4),
        })
    df = pd.DataFrame(results_list)
    df.to_csv(os.path.join(output_dir, "metrics", "results.csv"), index=False)

    # Adversarial
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
        adv_table[cat] = adv_results[cat]

    adv_rows = []
    for cat in ["injection", "override", "paraphrase", "canary"]:
        d = adv_results[cat]
        adv_rows.append({
            "attack": cat,
            "B1/B2": d.get("B1/B2", 0),
            "B6 (Proposed)": d.get("B6 (Proposed)", 0),
        })
    pd.DataFrame(adv_rows).to_csv(os.path.join(output_dir, "metrics", "adversarial.csv"), index=False)

    summary = {
        "config": {k: v for k, v in config.items() if k != "output_dir"},
        "data": {
            "n_nq": n_nq,
            "n_enron": n_enron,
            "n_prompts": len(prompts),
            "n_corpus": len(corpus),
            "distribution": dist_pct,
            "distribution_counts": dist_counts,
        },
        "results": [{"baseline": r["baseline"], "exposure": r["exposure"], "exposure_std": r["exposure_std"],
                     "violation": r["violation"], "violation_std": r["violation_std"],
                     "latency_rel": r["latency_rel"], "latency_std": r["latency_std"], "blocked_rate": r["blocked_rate"]}
                    for _, r in df.iterrows()],
        "adversarial": adv_results,
        "model_name": model_name,
        "hardware": hardware,
    }
    with open(os.path.join(output_dir, "metrics", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Update main.tex if update_latex is available
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
        print("update_latex skipped:", e)

    print("Done. Outputs:", os.path.join(output_dir, "metrics"))
    print("results.csv (mean ± std):", df.to_string())


if __name__ == "__main__":
    main()
