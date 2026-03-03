"""
Compute metrics per baseline: exposure, violation, latency, blocked rate.
"""
from .tagging import segment_label, LEVEL_INDEX, clearance_allows

# Simple policy: role -> disallowed content labels
ROLE_RULES = {
    "support": ["Confidential", "Restricted"],  # support cannot disclose confidential/restricted
    "analyst": ["Restricted"],
    "engineer": ["Restricted"],
}


def restricted_content_exposure_rate(responses: list, clearances: list) -> float:
    """Fraction of responses where segment_label(response) > clearance."""
    if not responses:
        return 0.0
    n = 0
    for r, c in zip(responses, clearances):
        lbl = segment_label(r)
        if not clearance_allows(c, lbl):
            n += 1
    return n / len(responses)


def policy_violation_rate(responses: list, roles: list) -> float:
    """Fraction where response contains content disallowed for role."""
    if not responses:
        return 0.0
    n = 0
    for r, role in zip(responses, roles):
        lbl = segment_label(r)
        disallowed = ROLE_RULES.get(role, ["Restricted"])
        if lbl in disallowed:
            n += 1
    return n / len(responses)


def clearance_mismatch_rate(responses: list, clearances: list) -> float:
    """Same as exposure: max segment label exceeds clearance."""
    return restricted_content_exposure_rate(responses, clearances)


def latency_stats(timings: list) -> dict:
    """Mean, p50, p95 in seconds."""
    if not timings:
        return {"mean": 0, "p50": 0, "p95": 0}
    s = sorted(timings)
    n = len(s)
    return {
        "mean": sum(s) / n,
        "p50": s[int(0.5 * n)] if n else 0,
        "p95": s[int(0.95 * n)] if n else 0,
    }


def blocked_rate(blocked_flags: list) -> float:
    if not blocked_flags:
        return 0.0
    return sum(1 for b in blocked_flags if b) / len(blocked_flags)
