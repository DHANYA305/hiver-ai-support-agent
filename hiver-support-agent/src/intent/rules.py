"""
Keyword rules for intent tagging.

Two jobs (see decision_log.md #4):
1. This IS the "simple baseline" required by the assignment (vs. the
   trivial "always predict the majority class" baseline, and vs. the
   trained TF-IDF classifier which is the actual system).
2. It also generates weak/distant labels over the ~12k unlabelled pairs so
   we have something to train the TF-IDF classifier on, since we don't
   have hand labels for more than 200 examples (the golden set).

Rules are intentionally simple substring/regex checks: this needs to be
weak-but-fast, not competitive.
"""
import re

from src.config import DEFAULT_INTENT, INTENTS

_PATTERNS: dict[str, list[str]] = {
    "playback_technical": [
        r"\bskip(ping|s|ped)?\b", r"\bbuffer(ing|s)?\b", r"\bcrash(ing|es|ed)?\b",
        r"\bfreez(e|ing|es)\b", r"\bwon'?t (play|load|open|start)\b", r"\bstop(s|ped|ping)? (playing|working)\b",
        r"\bads? (playing|keep)\b.*\bpremium\b", r"\baudio quality\b", r"\blag(gy|ging)?\b",
        r"\bnot loading\b", r"\bglitch(y|ing)?\b", r"\bpaused? (cos|because|randomly)\b",
        r"\bdownload(s|ed)? (unavailable|removed|missing|gone)\b", r"\bapp (keeps? )?(closing|crashing)\b",
    ],
    "account_access": [
        r"\blog\s?in\b", r"\bpassword\b", r"\b2fa\b", r"\btwo.factor\b", r"\blocked out\b",
        r"\bcan'?t access\b", r"\bhacked\b", r"\bverify(ing)? my (account|email)\b",
        r"\breset (my )?password\b", r"\baccount.*(suspend|disabl|ban)",
    ],
    "billing_subscription": [
        r"\bpremium\b.*\b(charge|pay|bill|activat|cancel)\b", r"\brefund\b", r"\bcancel(l?ing|led)? (my )?(membership|subscription|premium|account)\b",
        r"\bbill(ed|ing)?\b", r"\bcharge(d)? (me|twice|again)\b", r"\bpayment (failed|declined|method)\b",
        r"\bfree trial\b", r"\bdowngrade\b", r"\bnot activated\b", r"\bmoney\b.*\btaken\b",
    ],
    "content_availability": [
        r"(not|isn'?t|n'?t) (on|available on) spotify\b", r"\bmissing from\b", r"\bremoved? from (spotify|catalog)\b",
        r"\bregion(al)? (lock|restrict)\b", r"\bcan'?t find (this )?(song|album|podcast|artist)\b",
        r"\bunavailable in my country\b", r"\blicensing\b",
    ],
    "feature_request": [
        r"\bwish (spotify|you) (had|would|could)\b", r"\bplease add\b", r"\bfeature request\b",
        r"\bwhy (doesn'?t|isn'?t there)\b.*\b(feature|option|button)\b", r"\bcould you add\b",
        r"\bsuggestion\b", r"\bidea(s)?\b.*\bvote\b", r"\bwould be (great|nice|cool) if\b",
    ],
    "praise_other": [
        r"\bthank(s| you)\b", r"\blove (spotify|this|you)\b", r"\bappreciate\b", r"\bgreat (job|work|app)\b",
        r"\bno worries\b", r"\byou'?re the best\b",
    ],
}

_COMPILED = {intent: [re.compile(p, re.IGNORECASE) for p in pats] for intent, pats in _PATTERNS.items()}

# Order matters: technical/billing/access complaints should win over a
# trailing "thanks" in the same message, so praise_other is checked last.
_PRIORITY = ["account_access", "billing_subscription", "playback_technical",
             "content_availability", "feature_request", "praise_other"]


def rule_based_intent(text: str) -> str:
    if not text:
        return DEFAULT_INTENT
    for intent in _PRIORITY:
        for pattern in _COMPILED[intent]:
            if pattern.search(text):
                return intent
    return DEFAULT_INTENT


def majority_class_intent(_text: str) -> str:
    """Trivial baseline: always predict the single most frequent intent
    (computed empirically on the SpotifyCares pairs — see report.md)."""
    return "playback_technical"


assert all(i in INTENTS for i in _PRIORITY), "rule intents must be in taxonomy"
