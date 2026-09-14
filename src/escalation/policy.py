"""
Decide auto_handle vs escalate, with a human-readable reason.

Design (decision_log.md #9): escalation is a SEPARATE decision from intent
and from generation confidence — it can fire even when the reply looks
fine, because some things (legal threats, fraud, self-harm mentions)
should never be auto-sent regardless of how good the drafted text is.
Order of checks matters: hard triggers > low retrieval confidence > risky
intent-specific rules > default auto-handle.
"""
from src.config import HARD_ESCALATION_TRIGGERS

# Calibrated on a 300-row held-out DEV sample (NOT the golden set) using
# src/evaluation/calibrate_thresholds.py — see that file's docstring for
# why we don't just eyeball these. Values are top-1 cosine-similarity
# percentiles from that calibration run: p25≈0.22, p50≈0.27.
#
# Below this similarity, we don't trust the grounding evidence enough to
# auto-send anything, even if the drafted text reads fine.
LOW_CONFIDENCE_THRESHOLD = 0.22
# billing/account_access are the two intents most likely to touch money or
# account security — a wrong auto-reply there is costlier than in e.g.
# feature_request, so we hold them to a stricter (higher) bar than the
# general threshold above.
SENSITIVE_INTENTS = {"billing_subscription", "account_access"}
SENSITIVE_INTENT_THRESHOLD = 0.30


def decide_escalation(text: str, intent: str, top_similarity: float, generation_method: str) -> dict:
    text_lower = (text or "").lower()

    for trigger in HARD_ESCALATION_TRIGGERS:
        if trigger in text_lower:
            return _result(True, f"hard_trigger_matched:'{trigger}'")

    if generation_method == "offline_generic":
        return _result(True, "no_grounding_evidence_found")

    if intent in SENSITIVE_INTENTS and top_similarity < SENSITIVE_INTENT_THRESHOLD:
        return _result(
            True,
            f"sensitive_intent_low_confidence:intent={intent},similarity={top_similarity:.2f}<{SENSITIVE_INTENT_THRESHOLD}",
        )

    if top_similarity < LOW_CONFIDENCE_THRESHOLD:
        return _result(
            True,
            f"low_retrieval_confidence:similarity={top_similarity:.2f}<{LOW_CONFIDENCE_THRESHOLD}",
        )

    return _result(False, f"confident_match:intent={intent},similarity={top_similarity:.2f}")


def _result(escalate: bool, reason: str) -> dict:
    return {"escalate": escalate, "action": "escalate" if escalate else "auto_handle", "reason": reason}


if __name__ == "__main__":
    print(decide_escalation("I want a refund, this is fraud, get me a lawyer", "billing_subscription", 0.6, "offline_extractive_top1"))
    print(decide_escalation("songs keep skipping on my phone", "playback_technical", 0.6, "offline_extractive_top1"))
