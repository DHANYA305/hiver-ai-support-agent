"""
Sanity tests, not a replacement for the golden-set evaluation harness.
Run with: pytest tests/ -q
"""
import pandas as pd
import pytest

from src.config import INTENTS
from src.escalation.policy import decide_escalation
from src.intent.rules import majority_class_intent, rule_based_intent
from src.text_utils import clean_tweet, is_low_signal


def test_clean_tweet_strips_mentions_and_urls():
    raw = "@SpotifyCares check this out https://t.co/abc123 &amp; help"
    cleaned = clean_tweet(raw)
    assert "@SpotifyCares" not in cleaned
    assert "http" not in cleaned
    assert "&amp;" not in cleaned
    assert "&" in cleaned  # unescaped


def test_is_low_signal():
    assert is_low_signal("@x")
    assert not is_low_signal("my songs keep skipping on iOS")


@pytest.mark.parametrize("text,expected", [
    ("I forgot my password and can't log in", "account_access"),
    ("you charged me twice, please refund", "billing_subscription"),
    ("the app keeps crashing every time I open it", "playback_technical"),
    ("this song isn't available on spotify in my country", "content_availability"),
    ("please add a sleep timer feature", "feature_request"),
    ("thanks so much for the quick help!", "praise_other"),
])
def test_rule_based_intent_covers_obvious_cases(text, expected):
    assert rule_based_intent(text) == expected


def test_rule_based_intent_returns_valid_label_for_gibberish():
    assert rule_based_intent("asdkjfh aksjdhf") in INTENTS


def test_majority_class_is_constant():
    assert majority_class_intent("anything") == majority_class_intent("something else")


def test_hard_trigger_always_escalates_regardless_of_confidence():
    decision = decide_escalation(
        "get me a lawyer, this is fraud", "billing_subscription",
        top_similarity=0.99, generation_method="offline_extractive_top1",
    )
    assert decision["escalate"] is True
    assert "hard_trigger" in decision["reason"]


def test_confident_non_sensitive_match_auto_handles():
    decision = decide_escalation(
        "please add dark mode", "feature_request",
        top_similarity=0.9, generation_method="offline_extractive_top1",
    )
    assert decision["escalate"] is False


def test_low_confidence_billing_escalates_even_above_general_threshold():
    # 0.25 clears the general LOW_CONFIDENCE_THRESHOLD but not the
    # stricter SENSITIVE_INTENT_THRESHOLD for billing/account_access.
    decision = decide_escalation(
        "why hasn't my premium activated", "billing_subscription",
        top_similarity=0.25, generation_method="offline_extractive_top1",
    )
    assert decision["escalate"] is True


def test_golden_set_schema():
    df = pd.read_csv("data/golden/golden_set.csv")
    assert len(df) >= 150
    required_cols = {"customer_text", "gold_intent", "gold_escalate", "gold_escalation_reason"}
    assert required_cols.issubset(df.columns)
    assert set(df["gold_intent"].unique()).issubset(set(INTENTS))
