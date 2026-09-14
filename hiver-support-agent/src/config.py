"""
Central config for the Hiver support-agent take-home.

Brand chosen: SpotifyCares (@SpotifyCares on Twitter).
Why: it's one of the largest single-brand slices in the Customer Support on
Twitter dataset (~43k tweets), the domain is consumer-facing and easy to
reason about without specialised knowledge (unlike e.g. airlines/banks),
and the failure modes (playback bugs, billing, account access) are varied
enough to make intent classification non-trivial without being a legal/
safety minefield.
"""

BRAND_HANDLE = "SpotifyCares"

RAW_TWCS_PATH = "data/raw/twcs.csv"
PAIRS_PATH = "data/processed/pairs.csv"
GOLDEN_PATH = "data/golden/golden_set.csv"
GOLDEN_GUIDE_PATH = "data/golden/labeling_guide.md"
RESULTS_DIR = "results"

# Intent taxonomy, defined bottom-up after reading ~300 raw SpotifyCares
# customer messages (see decision_log.md #1-2). Kept small (7 classes) on
# purpose: a support team acts on these, not on 77 fine-grained banking77
# style intents that don't map to different playbooks for this brand.
INTENTS = [
    "playback_technical",   # skipping, buffering, crashes, ads-on-premium, audio quality
    "account_access",       # login, password reset, 2FA, locked/hacked account
    "billing_subscription", # payment failed, refund, premium not applied, cancel/downgrade
    "content_availability", # song/podcast/album missing, region lock, removed content
    "feature_request",      # asking for a feature, "why doesn't spotify do X"
    "general_complaint",    # venting / bad experience without a clear actionable bug
    "praise_other",         # compliments, small talk, or anything that doesn't fit above
]

DEFAULT_INTENT = "general_complaint"

# Words/phrases that should always push a ticket to a human regardless of
# intent or retrieval confidence — see decision_log.md #9.
HARD_ESCALATION_TRIGGERS = [
    "lawyer", "legal action", "sue", "gdpr", "ccpa", "data request",
    "delete my data", "hacked", "unauthorized charge", "fraud",
    "suicide", "self harm", "kill myself", "self-harm",
    "chargeback", "class action", "reporter", "journalist", "press inquiry",
]

RANDOM_SEED = 13
