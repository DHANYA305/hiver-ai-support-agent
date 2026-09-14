"""
Full evaluation harness. Run with:  python -m src.evaluation.run_eval

IMPORTANT — LLM-as-judge status: this harness uses `judge_reply()`
(src/evaluation/llm_judge.py). It uses Gemini when `GEMINI_API_KEY`
is configured and available; otherwise it falls back to a heuristic
scorer.

Every output file records how the score was actually produced via a
`judge_type` / `judge_backend` field. Never assume that results were
produced by a real LLM unless the output explicitly indicates the LLM
backend was used.

The committed results/report were generated with
`judge_backend: "heuristic"` because no API key was available in the
build environment.


Produces, all under results/:
  - metrics.json           headline numbers for intent, escalation, reply quality
  - intent_predictions.csv per-example predictions for all 3 intent methods
  - reply_quality.csv      per-example judge scores for the system's replies
  - judge_human_calibration.csv  (from human_calibration.py)

Baselines required by the assignment:
  - trivial:  majority_class_intent (intent), always-auto-handle (escalation)
  - simple:   rule_based_intent (intent), the same rule-derived hard-trigger
              check only, no similarity signal (escalation)
  - system:   TF-IDF classifier (intent), full retrieval+rules policy (escalation)
"""
import json
import sys
from pathlib import Path

import pandas as pd

from src.config import GOLDEN_PATH, INTENTS, PAIRS_PATH, RESULTS_DIR
from src.escalation.policy import decide_escalation
from src.evaluation.human_calibration import agreement_summary, run_calibration
from src.evaluation.llm_judge import judge_reply
from src.evaluation.metrics import escalation_metrics, intent_metrics
from src.generation.llm_client import llm_available
from src.generation.reply_generator import generate_reply
from src.intent.classifier import load_classifier, predict_intent, train_from_pairs
from src.intent.rules import majority_class_intent, rule_based_intent
from src.retrieval.index import RetrievalIndex


def simple_baseline_escalation(text: str) -> dict:
    """Escalation baseline that only checks hard triggers — no retrieval
    confidence, no intent sensitivity. Everything else auto-handled."""
    from src.config import HARD_ESCALATION_TRIGGERS
    t = (text or "").lower()
    for trigger in HARD_ESCALATION_TRIGGERS:
        if trigger in t:
            return {"escalate": True, "reason": f"hard_trigger_matched:'{trigger}'"}
    return {"escalate": False, "reason": "default_auto_handle"}


def trivial_baseline_escalation(_text: str) -> dict:
    return {"escalate": False, "reason": "trivial_always_auto_handle"}


def main():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    golden = pd.read_csv(GOLDEN_PATH)
    golden["gold_escalate"] = (
    golden["gold_escalate"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"true": True, "false": False}))
    if golden["gold_escalate"].isna().any():
        raise ValueError(
        "gold_escalate contains invalid values. "
        "Use only True or False in golden_set.csv."
    )

    judge_backend = "llm" if llm_available() else "heuristic"
    banner = (
    f"\n{'='*70}\nJUDGE BACKEND FOR THIS RUN: {judge_backend.upper()}\n"
    + ("Real LLM (Gemini) will be used for generation + judging.\n"
       if judge_backend == "llm"
       else "No usable GEMINI_API_KEY found -- using heuristic fallback for\n"
            "reply-quality judging. Any 'LLM-as-judge' claim in report.md\n"
            "is INVALID for this run.\n")
    + f"{'='*70}\n"
)
    print(banner, file=sys.stderr)

    # ---- retrain classifier EXCLUDING golden rows (fixes a real leakage bug:
    # every golden customer_text is literally a row in pairs.csv — see
    # decision_log.md #16) + build a leak-free retrieval index for eval ----
    eval_model_path = f"{RESULTS_DIR}/intent_classifier_eval.pkl"
    classifier = train_from_pairs(PAIRS_PATH, model_path=eval_model_path,
                                   exclude_texts=set(golden["customer_text"]))
    eval_index = RetrievalIndex.build_eval_index(PAIRS_PATH, exclude_texts=set(golden["customer_text"]))

    results = {"n_golden": len(golden), "brand": "SpotifyCares", "judge_backend": judge_backend}

    # ================= INTENT =================
    y_true = golden["gold_intent"].tolist()
    preds = {
        "trivial_majority_class": golden["customer_text"].map(majority_class_intent).tolist(),
        "simple_rule_based": golden["customer_text"].map(rule_based_intent).tolist(),
        "system_tfidf_classifier": [predict_intent(classifier, t) for t in golden["customer_text"]],
    }
    intent_results = {name: intent_metrics(y_true, p, INTENTS) for name, p in preds.items()}
    results["intent"] = {name: {"accuracy": m["accuracy"], "macro_f1": m["report"]["macro avg"]["f1-score"]}
                          for name, m in intent_results.items()}

    pred_df = golden[["customer_text", "gold_intent"]].copy()
    for name, p in preds.items():
        pred_df[name] = p
    pred_df.to_csv(f"{RESULTS_DIR}/intent_predictions.csv", index=False)
    with open(f"{RESULTS_DIR}/intent_full_report.json", "w") as f:
        json.dump(intent_results, f, indent=2, default=float)

    # ================= ESCALATION =================
    system_gen = [generate_reply(t, eval_index) for t in golden["customer_text"]]
    system_intent = preds["system_tfidf_classifier"]
    system_escalate = [
        decide_escalation(t, i, g["top_similarity"], g["method"])["escalate"]
        for t, i, g in zip(golden["customer_text"], system_intent, system_gen)
    ]
    simple_escalate = [simple_baseline_escalation(t)["escalate"] for t in golden["customer_text"]]
    trivial_escalate = [trivial_baseline_escalation(t)["escalate"] for t in golden["customer_text"]]

    y_true_esc = golden["gold_escalate"].tolist()
    results["escalation"] = {
        "trivial_never_escalate": escalation_metrics(y_true_esc, trivial_escalate),
        "simple_hard_triggers_only": escalation_metrics(y_true_esc, simple_escalate),
        "system_full_policy": escalation_metrics(y_true_esc, system_escalate),
    }

    # ================= REPLY QUALITY (system only, judged) =================
    quality_rows = []
    for t, g in zip(golden["customer_text"], system_gen):
        j = judge_reply(t,g["reply"],g["grounding_examples"], require_llm=(judge_backend == "llm"))
        quality_rows.append({
            "customer_text": t, "draft_reply": g["reply"], "method": g["method"],
            "top_similarity": g["top_similarity"], **{f"judge_{k}": v for k, v in j.items()},
        })
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(f"{RESULTS_DIR}/reply_quality.csv", index=False)
    results["reply_quality"] = {
        "judge_backend": judge_backend,
        "mean_overall": quality_df["judge_overall"].mean(),
        "mean_faithfulness": quality_df["judge_faithfulness"].mean(),
        "mean_relevance": quality_df["judge_relevance"].mean(),
        "mean_tone": quality_df["judge_tone"].mean(),
        "judge_type": quality_df["judge_judge"].iloc[0] if len(quality_df) else None,
        "pct_by_method": quality_df["method"].value_counts(normalize=True).to_dict(),
    }

    # ================= JUDGE vs HUMAN AGREEMENT =================
    calib_df = run_calibration()
    results["judge_human_agreement"] = {"judge_backend": judge_backend, **agreement_summary(calib_df)}

    # ================= LEAKY-INDEX DEMO (for the "misleading number" section) =================
    leaky_index = RetrievalIndex.build_from_csv(PAIRS_PATH)  # NOT excluding golden rows
    leaky_gen = [generate_reply(t, leaky_index) for t in golden["customer_text"].head(50)]
    leaky_sims = [g["top_similarity"] for g in leaky_gen]
    clean_sims = [g["top_similarity"] for g in system_gen[:50]]
    results["leakage_demo"] = {
        "note": "top-1 retrieval similarity on first 50 golden examples: leaky index (golden rows left in the retrieval pool) vs. the clean eval index used everywhere else above",
        "mean_top_similarity_leaky_index": sum(leaky_sims) / len(leaky_sims),
        "mean_top_similarity_clean_index": sum(clean_sims) / len(clean_sims),
    }

    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(json.dumps(results, indent=2, default=float))
    print(banner, file=sys.stderr)


if __name__ == "__main__":
    main()
