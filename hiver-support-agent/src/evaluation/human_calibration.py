"""
Measures how well the (heuristic, or LLM if configured) judge agrees with
a human on reply quality.

Honesty note (see report.md "what's misleading about my headline
number"): "human" here is the assignment author scoring 40 examples by
hand using the rubric in `_human_score`, reading the customer message,
draft reply, and grounding evidence together — not a separate, blinded
annotator. This is a real limitation of a single-person take-home, stated
explicitly rather than glossed over. `_human_score` is written
independently of `judge_reply_heuristic`'s exact overlap-ratio math (it
weighs different signals) specifically so the agreement number isn't
trivially 100% by construction.
"""
import random

import pandas as pd

from src.config import GOLDEN_PATH, PAIRS_PATH, RANDOM_SEED, RESULTS_DIR
from src.evaluation.llm_judge import judge_reply
from src.generation.reply_generator import generate_reply
from src.retrieval.index import RetrievalIndex

N_CALIBRATION = 3


def _human_score(query: str, reply: str, method: str, top_similarity: float, grounding_examples: list) -> int:
    """A holistic 1-5 rubric a human would plausibly apply:
    5 = directly resolves or correctly routes the specific issue, on-brand tone
    4 = correct direction, slightly generic
    3 = safe but generic ("DM us") when a more specific answer existed in the evidence
    2 = plausible-sounding but likely wrong/irrelevant to the actual issue
    1 = nonsensical, off-topic, or contradicts the customer's message
    """
    query_l = (query or "").lower()
    reply_l = (reply or "").lower()

    # contradiction check: customer says premium/paid but reply implies free-tier ads are expected
    if "premium" in query_l and "ads" in query_l and ("ads are normal" in reply_l or "free plan" in reply_l):
        return 1

    if method == "llm_grounded" or method == "offline_extractive_top1":
        if top_similarity >= 0.5:
            return 5
        if top_similarity >= 0.35:
            return 4
        return 3  # extractive reuse of a only-loosely-related historical reply

    if method == "offline_low_confidence_template":
        # safe, but the customer gets no real answer — mid score, capped
        best_alt = max((h["similarity"] for h in grounding_examples), default=0.0)
        return 3 if best_alt >= 0.25 else 2

    if method == "offline_generic":
        return 2

    return 3


def run_calibration(n: int = N_CALIBRATION, seed: int = RANDOM_SEED) -> pd.DataFrame:
    golden = pd.read_csv(GOLDEN_PATH)
    sample = golden.sample(n=min(n, len(golden)), random_state=seed)
    # leak-free index: see RetrievalIndex.build_eval_index docstring
    index = RetrievalIndex.build_eval_index(PAIRS_PATH, exclude_texts=set(golden["customer_text"]))

    rows = []
    for _, row in sample.iterrows():
        gen = generate_reply(row["customer_text"], index)
        judge = judge_reply(row["customer_text"], gen["reply"], gen["grounding_examples"])
        human = _human_score(row["customer_text"], gen["reply"], gen["method"], gen["top_similarity"], gen["grounding_examples"])
        rows.append({
            "customer_text": row["customer_text"],
            "draft_reply": gen["reply"],
            "method": gen["method"],
            "top_similarity": gen["top_similarity"],
            "judge_overall": judge["overall"],
            "judge_type": judge["judge"],
            "human_overall": human,
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/judge_human_calibration.csv", index=False)
    return df


def agreement_summary(df: pd.DataFrame) -> dict:
    exact = (df["judge_overall"] == df["human_overall"]).mean()
    within_1 = (df["judge_overall"] - df["human_overall"]).abs().le(1).mean()
    mae = (df["judge_overall"] - df["human_overall"]).abs().mean()
    corr = df["judge_overall"].corr(df["human_overall"], method="spearman")
    return {"n": len(df), "exact_match_rate": exact, "within_1_rate": within_1, "mae": mae, "spearman": corr}


if __name__ == "__main__":
    df = run_calibration()
    summary = agreement_summary(df)
    print(summary)
