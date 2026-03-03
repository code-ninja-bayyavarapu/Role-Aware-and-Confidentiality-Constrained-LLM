"""
Load Natural Questions and Enron (or fallback) per config.
CPU-friendly; no GPU required.
Uses HuggingFace 'datasets' package (imported as hf_datasets to avoid name clash).
"""
import os
import random
import pandas as pd
import yaml

# HuggingFace datasets package (avoid shadowing by this module's name)
try:
    import datasets as hf_datasets
except ImportError:
    hf_datasets = None

def _hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

CLEARANCES_ORDER = ["Public", "Internal", "Confidential", "Restricted"]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_nq(config: dict) -> list:
    """Load Natural Questions subset. Returns list of dicts with keys: question, answer, context, doc_id."""
    n_samples = config.get("nq_samples", 400)
    seed = config.get("seed", 42)
    if n_samples <= 0:
        return []
    if hf_datasets is not None:
        try:
            # NQ Open: google-research-datasets/nq_open (free; use HF_TOKEN for higher rate limits)
            ds = hf_datasets.load_dataset(
                "google-research-datasets/nq_open", split="train",
                token=_hf_token()
            )
            rng = random.Random(seed)
            indices = rng.sample(range(len(ds)), min(n_samples, len(ds)))
            out = []
            for i in indices:
                row = ds[int(i)]
                q = row.get("question", row.get("question_text", ""))
                ans = row.get("answer", row.get("answers", ""))
                if isinstance(ans, list):
                    ans = ans[0] if ans else ""
                ctx = row.get("context", row.get("context_text", ""))
                if isinstance(ctx, list):
                    ctx = ctx[0] if ctx else ""
                out.append({"question": str(q), "answer": str(ans), "context": str(ctx), "doc_id": f"nq_{i}"})
            return out
        except Exception as e:
            print(f"[load_data] NQ load failed: {e}. Using synthetic QA fallback.")
    # Synthetic fallback for testing without HF datasets
    rng = random.Random(seed)
    out = []
    for i in range(min(n_samples, 500)):
        out.append({
            "question": f"What is the main topic of document {i}?",
            "answer": f"Document {i} discusses general information and internal process for workflow.",
            "context": f"Context for document {i}. Internal strategy and roadmap are mentioned.",
            "doc_id": f"nq_synth_{i}",
        })
    return out


def load_enron(config: dict) -> list:
    """Load Enron subset. Returns list of dicts with keys: text, doc_id."""
    n_samples = config.get("enron_samples", 1500)
    seed = config.get("seed", 42)
    csv_path = config.get("enron_csv_path")
    text_col = config.get("csv_text_column", "text")
    if csv_path and os.path.isfile(csv_path):
        df = pd.read_csv(csv_path, nrows=n_samples * 2)
        if text_col not in df.columns:
            text_col = df.columns[0]
        df = df.sample(n=min(n_samples, len(df)), random_state=seed)
        return [{"text": str(row[text_col]), "doc_id": f"enron_{i}"} for i, row in df.iterrows()]
    if hf_datasets is not None:
        try:
            # Enron-style emails (free; use HF_TOKEN for higher rate limits)
            ds = hf_datasets.load_dataset(
                "LLM-PBE/enron-email", split="train",
                token=_hf_token()
            )
            rng = random.Random(seed)
            n_avail = min(n_samples, len(ds))
            indices = rng.sample(range(len(ds)), n_avail) if len(ds) > n_avail else list(range(n_avail))
            out = []
            for idx, i in enumerate(indices):
                row = ds[int(i)]
                text = row.get("body", row.get("text", row.get("content", row.get("message", str(row)))))
                if isinstance(text, list):
                    text = " ".join(str(x) for x in text)
                out.append({"text": str(text)[:5000], "doc_id": f"enron_{idx}"})
            return out
        except Exception as e:
            print(f"[load_data] Enron load failed: {e}. Using synthetic fallback.")
    # Synthetic fallback
    rng = random.Random(seed)
    out = []
    for i in range(min(n_samples, 800)):
        out.append({
            "text": f"Internal email document {i}. Strategy and roadmap for Q4. Budget and revenue figures are confidential. Contact support for SLA.",
            "doc_id": f"enron_synth_{i}",
        })
    return out


def get_corpus_for_retrieval(nq_list: list, enron_list: list) -> list:
    """Single corpus: each item has text, doc_id, source (nq/enron)."""
    corpus = []
    for d in nq_list:
        text = (d.get("context", "") or "") + " " + (d.get("answer", "") or "")
        if not text.strip():
            text = d.get("question", "")
        corpus.append({"text": text.strip() or d.get("doc_id", ""), "doc_id": d.get("doc_id", ""), "source": "nq"})
    for d in enron_list:
        corpus.append({"text": d.get("text", ""), "doc_id": d.get("doc_id", ""), "source": "enron"})
    return corpus


def get_qa_prompts(nq_list: list, enron_list: list, max_prompts: int, seed: int) -> list:
    """Prompts for evaluation: mix of NQ questions and Enron-based prompts."""
    rng = random.Random(seed)
    prompts = []
    for d in nq_list[:max_prompts]:
        prompts.append({"query": d["question"], "doc_id": d["doc_id"], "source": "nq"})
    for d in enron_list[:max(0, max_prompts - len(prompts))]:
        topic = (d.get("text", "") or "")[:120].replace("\n", " ")
        prompts.append({"query": f"Summarize or explain: {topic}", "doc_id": d.get("doc_id", ""), "source": "enron"})
    rng.shuffle(prompts)
    return prompts[:max_prompts]
