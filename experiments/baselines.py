"""
Baselines B1--B6. CPU-only generation with TinyLlama.
"""
import time
import json
import os
from tqdm import tqdm

from .tagging import doc_label, segment_label, build_token_min_clearance
from .retrieval import build_index, retrieve, retrieve_unfiltered
from .masking import get_processor
from .metrics import restricted_content_exposure_rate, policy_violation_rate, latency_stats, blocked_rate


def _generate(model, tokenizer, prompt: str, max_new_tokens: int, do_sample: bool, temperature: float,
              logits_processor=None, **kwargs):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if hasattr(inputs, "to"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature,
                  pad_token_id=tokenizer.eos_token_id, **kwargs)
    if logits_processor:
        gen_kw["logits_processor"] = [logits_processor]
    out = model.generate(**inputs, **gen_kw)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_b1(model, tokenizer, query: str, retrieval_index, top_k: int, max_new_tokens: int, do_sample: bool, temperature: float):
    """Standard LLM: no PCPE, unfiltered retrieval, no mask."""
    ctx_docs = retrieve_unfiltered(retrieval_index, query, top_k) if retrieval_index else []
    context = " ".join(d["text"][:200] for d in ctx_docs)
    prompt = f"{context}\n\nQuestion: {query}\n\nAnswer:" if context else f"Question: {query}\n\nAnswer:"
    t0 = time.perf_counter()
    resp = _generate(model, tokenizer, prompt, max_new_tokens, do_sample, temperature)
    return resp, time.perf_counter() - t0


def run_b2(model, tokenizer, query: str, retrieval_index, top_k: int, max_new_tokens: int, do_sample: bool, temperature: float,
           role: str, clearance: str, tagger):
    """Post-hoc RBAC: generate like B1; if tagger(response) > clearance -> exposure and blocked."""
    resp, lat = run_b1(model, tokenizer, query, retrieval_index, top_k, max_new_tokens, do_sample, temperature)
    lbl = tagger(resp)
    from .tagging import LEVEL_INDEX
    blocked = LEVEL_INDEX.get(lbl, 0) > LEVEL_INDEX.get(clearance, 0)
    return resp, lat, blocked


def run_b3(model, tokenizer, query: str, retrieval_index, top_k: int, max_new_tokens: int, do_sample: bool, temperature: float,
           role: str, clearance: str, tenant_id: int):
    """CAES-only: retrieval filtered by tenant+clearance; no PCPE, no mask."""
    if not retrieval_index:
        prompt = f"Question: {query}\n\nAnswer:"
    else:
        ctx_docs = retrieve(retrieval_index, query, top_k, tenant_id, clearance)
        context = " ".join(d["text"][:200] for d in ctx_docs)
        prompt = f"{context}\n\nQuestion: {query}\n\nAnswer:" if context else f"Question: {query}\n\nAnswer:"
    t0 = time.perf_counter()
    resp = _generate(model, tokenizer, prompt, max_new_tokens, do_sample, temperature)
    return resp, time.perf_counter() - t0


def run_b4(model, tokenizer, query: str, retrieval_index, top_k: int, max_new_tokens: int, do_sample: bool, temperature: float,
           role: str, clearance: str):
    """PCPE-only: prefix [ROLE] [CLEARANCE]; unfiltered retrieval; no mask."""
    prefix = f"[ROLE: {role}] [CLEARANCE: {clearance}]\n\n"
    ctx_docs = retrieve_unfiltered(retrieval_index, query, top_k) if retrieval_index else []
    context = " ".join(d["text"][:200] for d in ctx_docs)
    body = f"{context}\n\nQuestion: {query}\n\nAnswer:" if context else f"Question: {query}\n\nAnswer:"
    prompt = prefix + body
    t0 = time.perf_counter()
    resp = _generate(model, tokenizer, prompt, max_new_tokens, do_sample, temperature)
    return resp, time.perf_counter() - t0


def run_b5(model, tokenizer, query: str, retrieval_index, top_k: int, max_new_tokens: int, do_sample: bool, temperature: float,
           role: str, clearance: str, tagger):
    """Output-only guardrail: generate like B3; if tagger > clearance -> BLOCKED."""
    resp, lat = run_b3(model, tokenizer, query, retrieval_index, top_k, max_new_tokens, do_sample, temperature, role, clearance, 0)
    lbl = tagger(resp)
    from .tagging import LEVEL_INDEX
    if LEVEL_INDEX.get(lbl, 0) > LEVEL_INDEX.get(clearance, 0):
        return "BLOCKED", lat, True
    return resp, lat, False


def run_b6(model, tokenizer, query: str, retrieval_index, top_k: int, max_new_tokens: int, do_sample: bool, temperature: float,
           role: str, clearance: str, tenant_id: int, token_min_clearance: dict):
    """Full: PCPE + CAES + decode-time mask."""
    prefix = f"[ROLE: {role}] [CLEARANCE: {clearance}]\n\n"
    if retrieval_index:
        ctx_docs = retrieve(retrieval_index, query, top_k, tenant_id, clearance)
        context = " ".join(d["text"][:200] for d in ctx_docs)
        body = f"{context}\n\nQuestion: {query}\n\nAnswer:" if context else f"Question: {query}\n\nAnswer:"
    else:
        body = f"Question: {query}\n\nAnswer:"
    prompt = prefix + body
    proc = get_processor(clearance, token_min_clearance)
    t0 = time.perf_counter()
    resp = _generate(model, tokenizer, prompt, max_new_tokens, do_sample, temperature, logits_processor=proc)
    return resp, time.perf_counter() - t0


def run_all_baselines(config: dict, model, tokenizer, retrieval_index, corpus_with_labels, prompts: list,
                     tagger, token_min_clearance: dict, roles: list, clearances: list, tenants: int, output_dir: str):
    """Run B1-B6 over prompts; collect responses and metrics. K runs per config."""
    import random
    k_runs = config.get("k_runs", 3)
    max_new_tokens = config.get("max_new_tokens", 128)
    do_sample = config.get("do_sample", False)
    temperature = config.get("temperature", 0.2)
    top_k = config.get("top_k", 3)
    seed = config.get("seed", 42)
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    baseline_runners = {
        "B1": lambda q, r, c, t: run_b1(model, tokenizer, q, retrieval_index, top_k, max_new_tokens, do_sample, temperature),
        "B2": lambda q, r, c, t: run_b2(model, tokenizer, q, retrieval_index, top_k, max_new_tokens, do_sample, temperature, r, c, tagger),
        "B3": lambda q, r, c, t: run_b3(model, tokenizer, q, retrieval_index, top_k, max_new_tokens, do_sample, temperature, r, c, t),
        "B4": lambda q, r, c, t: run_b4(model, tokenizer, q, retrieval_index, top_k, max_new_tokens, do_sample, temperature, r, c),
        "B5": lambda q, r, c, t: run_b5(model, tokenizer, q, retrieval_index, top_k, max_new_tokens, do_sample, temperature, r, c, tagger),
        "B6": lambda q, r, c, t: run_b6(model, tokenizer, q, retrieval_index, top_k, max_new_tokens, do_sample, temperature, r, c, t, token_min_clearance),
    }

    all_results = {b: [] for b in baseline_runners}
    for run_id in range(k_runs):
        rng = random.Random(seed + run_id)
        for i, p in enumerate(tqdm(prompts, desc=f"Run {run_id+1}/{k_runs}", leave=False)):
            query = p["query"]
            role = rng.choice(roles)
            clearance = rng.choice(clearances)
            tenant_id = rng.randint(0, tenants - 1) if tenants else 0
            for bname, runner in baseline_runners.items():
                try:
                    out = runner(query, role, clearance, tenant_id)
                    if bname == "B2":
                        resp, lat, blocked = out
                        all_results[bname].append({"response": resp, "latency": lat, "blocked": blocked, "role": role, "clearance": clearance})
                    elif bname == "B5":
                        resp, lat, blocked = out
                        all_results[bname].append({"response": resp, "latency": lat, "blocked": blocked, "role": role, "clearance": clearance})
                    else:
                        resp, lat = out[:2]
                        all_results[bname].append({"response": resp, "latency": lat, "role": role, "clearance": clearance})
                except Exception as e:
                    all_results[bname].append({"response": "", "latency": 0, "error": str(e), "role": role, "clearance": clearance})

    # Write raw JSONL per baseline
    for bname in baseline_runners:
        path = os.path.join(output_dir, "raw", f"{bname}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in all_results[bname]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return all_results
