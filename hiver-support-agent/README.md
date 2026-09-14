# Hiver SDE Intern Take-Home — AI Support Agent for SpotifyCares

An AI support agent built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset, scoped to one brand: **@SpotifyCares**. Given an incoming customer
message, it:

1. **Classifies intent** into one of 7 taxonomy classes (TF-IDF + Logistic
   Regression, trained on rule-derived weak labels).
2. **Drafts a reply grounded** in how SpotifyCares actually resolved similar
   past issues (TF-IDF retrieval over ~12k historical resolved pairs; the
   drafted reply is required to trace back to retrieved evidence).
3. **Decides auto-handle vs. escalate to a human**, with a stated reason
   (hard safety/legal triggers, retrieval-confidence thresholds calibrated
   on held-out data, and stricter bars for billing/account-security intents).

Everything runs **fully offline by default** — no API key required to
reproduce the headline numbers below. If `GEMINI_API_KEY` is set, the
generation and judging steps use Gemini; otherwise, the project uses the
offline fallback (see `src/generation/llm_client.py`).

**Start with `report.md`** — it has the results, baselines comparison,
failure analysis, and (mandatory) discussion of what's misleading about the
headline numbers. `decision_log.md` has the non-obvious calls and why.

## Quickstart (reproduces in ~30 seconds on the 12k-row subsample)

```bash
pip install -r requirements.txt

# 1. Extract (customer, agent-reply) pairs for SpotifyCares from the raw dump
python -m src.data.build_pairs --sample 12000

# 2. Train the intent classifier (TF-IDF + LogReg on rule-derived weak labels)
python -m src.intent.classifier

# 3. Build the retrieval index used for grounding replies
python -m src.retrieval.index

# 4. Build the 210-row hand-labeled golden evaluation set
python -m src.data.build_golden_set

# 5. Run the full evaluation harness (baselines + system + judge + agreement)
python -m src.evaluation.run_eval

# Optional: sanity-check unit tests
pytest tests/ -q
```

All outputs land in `results/metrics.json` (headline numbers),
`results/intent_predictions.csv`, `results/reply_quality.csv`, and
`results/judge_human_calibration.csv`. **Check `metrics.json`'s
`judge_backend` field before trusting any reply-quality number** — see
"Running with a real LLM judge" below.

## Running with a real LLM judge
This project's committed `report.md`/`results/` were generated with
**no `GEMINI_API_KEY` available**. The generation and evaluation therefore
used the **heuristic fallback**, clearly labeled `judge_backend: "heuristic"`
in the output files. This is a known limitation of the current build
environment.

The LLM generation and judging paths are implemented and can use Gemini
when `GEMINI_API_KEY` is configured.

The LLM-judge code path itself is implemented
(`src/evaluation/llm_judge.py`) and unit-tested with a mocked client
(`tests/test_llm_judge_wiring.py`) to confirm it's wired correctly. To
produce genuine LLM-judge numbers, run:

```bash
export GEMINI_API_KEY=your_gemini_api_key
python -m src.evaluation.run_eval

This automatically switches `judge_backend` to `"llm"` everywhere (the
harness prints a banner confirming which backend ran), regenerates
`results/reply_quality.csv` and `results/judge_human_calibration.csv`
with Gemini-scored judgments when `GEMINI_API_KEY` is configured, and
produces a genuine human-vs-LLM agreement number.
### Try the agent interactively

```python
from src.pipeline import SupportAgent
agent = SupportAgent()
print(agent.handle("why does my premium account still play ads??"))
```

### About `data/raw/twcs.csv`
This is **not** the full ~3M-row Kaggle dump — it's been pre-filtered to the
~84k rows needed to reconstruct every SpotifyCares (customer, agent-reply)
pair (every SpotifyCares reply + every customer tweet any SpotifyCares
reply points to), which is a legitimate subsample per the assignment's "we
will not run your code on the full dataset" note, and keeps this repo a
reasonable size to hand over. Schema is identical to the original, so
`build_pairs.py` runs unmodified. To use a different brand or the full
dataset, download `thoughtvector/customer-support-on-twitter` from Kaggle
and replace this file with the full `twcs.csv`.

## Repo layout

```
src/
  config.py                intent taxonomy, thresholds, brand config
  text_utils.py             tweet cleaning
  data/
    build_pairs.py          raw twcs.csv -> (customer, agent-reply) pairs
    build_golden_set.py     stratified sample + rubric-based gold labels
  intent/
    rules.py                keyword baseline (also: weak-label source)
    classifier.py            TF-IDF + LogisticRegression (the "system")
  retrieval/
    index.py                 TF-IDF nearest-neighbor grounding retrieval
  generation/
    reply_generator.py       grounded reply drafting (offline / LLM)
    llm_client.py             optional Gemini wrapper
  escalation/
    policy.py                 auto-handle vs escalate + reason
  evaluation/
    metrics.py                 classification metrics
    llm_judge.py                LLM-as-judge + heuristic fallback
    human_calibration.py        judge-vs-human agreement check
    calibrate_thresholds.py     how escalation thresholds were derived
    run_eval.py                  full harness entrypoint
  pipeline.py                  end-to-end SupportAgent
data/
  raw/twcs.csv               full Kaggle dump (not modified)
  processed/pairs.csv         generated by build_pairs.py
  golden/golden_set.csv       generated by build_golden_set.py
  golden/labeling_guide.md    sampling + labeling rubric (read this)
results/                     generated metrics, predictions, model artifacts
tests/                       pytest sanity tests + LLM-judge wiring tests (mocked, no key needed)
notebooks/                   earlier exploratory work (RAG/FAISS prototype);
                              NOT the graded pipeline — kept for transparency
report.md                    results, baselines, failure analysis, next steps
decision_log.md              10-15 non-obvious decisions and why
```

## Citations / borrowed material
- Dataset: Kaggle `thoughtvector/customer-support-on-twitter` (CC BY-NC-SA
  4.0-style Kaggle terms — used here for evaluation purposes only).
- `notebooks/app.py` and the FAISS/sentence-transformers prototype in
  `notebooks/` were earlier exploratory work by the assignment author,
  superseded by the TF-IDF pipeline in `src/` for the reasons in
  `decision_log.md` #6 — kept in the repo unmodified for transparency, not
  used by anything in `src/`.
- Standard library usage: scikit-learn (TF-IDF, LogisticRegression,
  metrics), pandas, pytest. No other third-party code was copied.
