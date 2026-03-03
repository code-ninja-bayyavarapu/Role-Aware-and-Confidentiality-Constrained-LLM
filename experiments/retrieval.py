"""
TF-IDF retrieval with tenant and clearance (CAES) filtering.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def build_index(corpus: list, doc_labels: dict, tenant_ids: dict):
    """
    corpus: list of {text, doc_id, source}
    doc_labels: doc_id -> clearance label
    tenant_ids: doc_id -> tenant_id (int 0..T-1)
    """
    texts = [c["text"] for c in corpus]
    doc_ids = [c["doc_id"] for c in corpus]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    return {
        "vectorizer": vectorizer,
        "X": X,
        "doc_ids": doc_ids,
        "doc_labels": doc_labels,
        "tenant_ids": tenant_ids,
        "corpus": corpus,
    }


def retrieve(index: dict, query: str, top_k: int, tenant_id: int, clearance: str):
    """
    CAES: filter by tenant_id and doc_label <= clearance.
    """
    from .tagging import LEVEL_INDEX
    doc_ids = index["doc_ids"]
    doc_labels = index["doc_labels"]
    tenant_ids = index["tenant_ids"]
    X = index["X"]
    vectorizer = index["vectorizer"]
    q_vec = vectorizer.transform([query])
    sim = cosine_similarity(q_vec, X).ravel()
    clearance_idx = LEVEL_INDEX.get(clearance, 0)
    candidates = []
    for i in range(len(doc_ids)):
        tid = tenant_ids.get(doc_ids[i], 0)
        if tid != tenant_id:
            continue
        lbl = doc_labels.get(doc_ids[i], "Public")
        if LEVEL_INDEX.get(lbl, 0) > clearance_idx:
            continue
        candidates.append((i, float(sim[i])))
    candidates.sort(key=lambda x: -x[1])
    top = candidates[:top_k]
    return [index["corpus"][i] for i, _ in top]


def retrieve_unfiltered(index: dict, query: str, top_k: int):
    """No CAES; for B1/B2/B4."""
    vectorizer = index["vectorizer"]
    X = index["X"]
    q_vec = vectorizer.transform([query])
    sim = cosine_similarity(q_vec, X).ravel()
    top_indices = np.argsort(-sim)[:top_k]
    return [index["corpus"][i] for i in top_indices]
