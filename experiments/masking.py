"""
Decode-time mask (RPEL): set logits to -inf for tokens above clearance.
"""
from transformers import LogitsProcessor

from .tagging import LEVEL_INDEX, CLEARANCES_ORDER


class ClearanceLogitsProcessor(LogitsProcessor):
    def __init__(self, clearance: str, token_min_clearance: dict):
        self.clearance_idx = LEVEL_INDEX.get(clearance, 0)
        self.token_min_clearance = token_min_clearance

    def __call__(self, input_ids, scores):
        for idx, sc in enumerate(scores[0]):
            min_level = self.token_min_clearance.get(idx, "Public")
            min_idx = LEVEL_INDEX.get(min_level, 0)
            if min_idx > self.clearance_idx:
                scores[0][idx] = float("-inf")
        return scores


def build_allowed_token_ids(tokenizer, clearance: str, token_min_clearance: dict) -> set:
    """Set of token ids allowed at this clearance. For caching."""
    c_idx = LEVEL_INDEX.get(clearance, 0)
    allowed = set()
    for tok_id, min_level in token_min_clearance.items():
        if LEVEL_INDEX.get(min_level, 0) <= c_idx:
            allowed.add(tok_id)
    return allowed


def get_processor(clearance: str, token_min_clearance: dict) -> ClearanceLogitsProcessor:
    return ClearanceLogitsProcessor(clearance, token_min_clearance)
