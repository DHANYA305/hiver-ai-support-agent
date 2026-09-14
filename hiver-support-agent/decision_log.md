# Decision log

Non-obvious decisions made while building this, and why. See `report.md`
for the results these decisions produced.

1. **Brand: SpotifyCares.** Largest clean single-brand slice after Amazon/
   Apple/Uber/Delta (which mix multiple sub-products or have more emotionally
   loaded / high-stakes traffic like flight cancellations). Spotify support
   traffic is varied enough to make intent classification non-trivial without
   needing airline-refund-policy or banking-regulation domain knowledge.

2. **7-class intent taxonomy, not Banking77's 77.** A support team acts on a
   handful of playbooks, not 77 fine-grained sub-intents that don't exist for
   a music app anyway. Taxonomy was built bottom-up after reading a few
   hundred raw SpotifyCares customer messages (`playback_technical`,
   `account_access`, `billing_subscription`, `content_availability`,
   `feature_request`, `general_complaint`, `praise_other`).

3. **Single-turn pairs only, not full threads.** Twitter support threads are
   often 4-8 messages deep and quickly move to "please DM us" (invisible in
   this public dataset). Modeling multi-turn state is real, valuable work,
   but it's a different, larger project — see report.md's "what we chose not
   to build."

4. **Two explicit baselines for intent: trivial (majority class) and simple
   (keyword rules).** The keyword rules are also reused as the weak-label
   source for training the TF-IDF classifier (decision #5) — this is
   flagged, not hidden, because it creates a real risk that the "system"
   just learns to imitate the "simple" baseline's blind spots (see report.md
   "misleading headline number").

5. **TF-IDF + Logistic Regression over an LLM zero-shot classifier as the
   default "system."** Free, millisecond inference, fully reproducible
   without an API key, and it's the thing whose numbers must reproduce in
   15 minutes for a grader with no key. An LLM path is wired in as an
   optional upgrade (`src/generation/llm_client.py`) but is not required to
   get headline numbers.

6. **TF-IDF cosine retrieval over sentence-embeddings + FAISS.** An earlier
   exploratory notebook (`notebooks/app.py`) used sentence-transformers +
   FAISS. That was dropped for the graded pipeline: support tweets are short
   and full of domain tokens ("premium", "buffering", "downgrade") that
   TF-IDF handles well, it needs no model download, and it's fast enough on
   CPU to keep the whole repro under a minute. This is a real trade-off, not
   a strict win — see report.md for where TF-IDF's lexical matching fails
   (paraphrases with little word overlap).

7. **Everything works with zero API keys; an API key strictly upgrades it.**
   Generation falls back to grounded extractive/template replies; the judge
   falls back to a heuristic scorer. This was a deliberate constraint so the
   README's "reproduces in 15 minutes" claim doesn't secretly require a paid
   API key nobody but the author has.

8. **Hard escalation triggers checked independently of intent/confidence.**
   Legal threats, fraud claims, self-harm mentions, etc. should never be
   auto-sent regardless of how good the drafted reply looks or how
   confident retrieval is. This check runs first, before any
   confidence-based logic, in `src/escalation/policy.py`.

9. **Billing/account_access held to a stricter confidence bar than other
   intents.** A wrong auto-reply about money or account security is costlier
   than a wrong auto-reply about a feature request. Implemented as a second,
   higher similarity threshold (`SENSITIVE_INTENT_THRESHOLD`) rather than a
   blanket "always escalate billing," since plenty of billing questions
   ("how do I cancel," "what's included in Family plan") are generic and
   safely automatable.

10. **Escalation/retrieval thresholds calibrated on a held-out DEV sample,
    not eyeballed and not tuned on the golden set.** The first version used
    an eyeballed 0.35 threshold from a couple of manual queries against the
    full (leaky) retrieval pool. Once the leak-free eval index (decision
    #11) was in place, the same threshold flagged ~75% of golden examples
    as low-confidence. `src/evaluation/calibrate_thresholds.py` re-derives
    the threshold from the actual similarity distribution a held-out
    message sees, using a separate 300-row dev sample.

11. **Built a leak-free retrieval index specifically for evaluation.** The
    golden set was sampled from the same pool used to build the production
    retrieval index. Evaluating with that same index means retrieval finds
    a golden example's own historical reply verbatim (similarity ≈ 1.0)
    almost every time — every quality number looks great for a boring,
    fake reason. `RetrievalIndex.build_eval_index()` excludes golden-set
    rows from the pool before evaluating; this is the single most important
    fix in the whole project (see report.md's mandatory section).

12. **Golden set labeled with a separate, more detailed rubric than the
    production keyword rules — not by re-running `rules.py`.** Using the
    same code to generate "ground truth" as to generate predictions would
    make intent accuracy circular. `src/data/build_golden_set.py` implements
    its own, independently-written labeling functions per
    `data/golden/labeling_guide.md`.

13. **Single-labeler golden set and single-labeler judge calibration —
    stated as a limitation, not hidden.** This is a take-home, not a team
    with a second annotator. Both `labeling_guide.md` and
    `human_calibration.py`'s docstring say this explicitly, and report.md
    discusses what inter-rater reliability would add.

14. **Reported the heuristic judge's negative correlation with hand scores
    rather than reverse-engineering a heuristic that agrees.** It would
    have been easy to tune the heuristic's weights against the 40
    calibration examples until it "worked." That's overfitting the
    evaluation to itself. The honest, low agreement number is the more
    useful thing to hand a grader — see report.md.

15. **Kept `notebooks/` as-is instead of deleting it.** It's the author's
    real earlier exploration (FAISS/sentence-transformers RAG prototype)
    and is left unmodified for transparency about what was tried and
    abandoned, per the assignment's "cite anything you borrowed... not
    knowing what you borrowed is not [fine]" spirit — nothing in `src/`
    imports from it.

16. **Found and fixed a second, more serious leakage bug — then made two
    legitimate attempts to fix the resulting gap, and reported both as
    failures.** The intent classifier's original training step
    (`train_from_pairs`) fit on the entirety of `pairs.csv` with no
    exclusion, so literally all 210 golden `customer_text` values were
    present verbatim in its training data (the golden set is sampled from
    that same file). This is why the first version of this report showed
    the classifier tying the keyword-rule baseline almost exactly — partial
    memorization, not independent evidence of quality. Fixed by adding an
    `exclude_texts` parameter to `train_from_pairs`, used exclusively in
    `run_eval.py`. Leak-free accuracy is **71.4%**, *below* the rule
    baseline's 88.1%. Two fixes were then tried and both failed, honestly
    reported rather than hidden or tuned against golden: (a) a
    rule-first/classifier-fallback hybrid scored the same as the rule
    baseline alone (macro F1 0.881 vs 0.887); (b) oversampling minority
    weak-label classes (`content_availability` has only 19 of 11,787
    training rows) to 1000 rows each made macro F1 *worse* (0.63) and still
    produced zero correct `content_availability` predictions. Root cause,
    confirmed via 4-fold cross-validation against the classifier's own
    training signal (mean macro F1 only 0.64 to reproduce the rules it was
    trained on): the classifier fails to even faithfully reproduce the
    rules, because the weak-label signal itself is noisy and severely
    class-imbalanced — see report.md §2-3.
