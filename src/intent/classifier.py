"""
Trained intent classifier (TF-IDF + multinomial Logistic Regression).

Why this over an LLM zero-shot classifier as the "main" system (see
decision_log.md #5): it's free, runs in milliseconds, is fully
reproducible without an API key, and — as shown in report.md — beats the
keyword-rule baseline on the golden set. An LLM classifier is offered as
an optional swap-in (src/intent/llm_classifier.py-equivalent lives inside
generation/llm_client.py's classify_intent_llm) for anyone with an API key,
but it is not required to reproduce headline numbers.
"""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.intent.rules import rule_based_intent

MODEL_PATH = "results/intent_classifier.pkl"


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.9,
            sublinear_tf=True, stop_words="english",
        )),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=4.0)),
    ])


def train_from_pairs(pairs_csv: str, model_path: str = MODEL_PATH, exclude_texts: set | None = None) -> Pipeline:
    """Train the classifier on weak (rule-derived) labels.

    `exclude_texts`: customer_text values to drop before fitting. This
    MUST be set to the golden set's customer_text values whenever this
    function is used to produce numbers that get reported against the
    golden set — see decision_log.md #16. Without it, the classifier is
    literally fit on the exact rows it's later "tested" on (every golden
    example was sampled from this same pairs.csv), which is leakage, not
    generalization. The production model (no exclude_texts) uses all
    available data, since a shipped model should use everything once
    evaluation is done — that's a separate, later step from evaluation.
    """
    df = pd.read_csv(pairs_csv)
    if exclude_texts:
        before = len(df)
        df = df[~df["customer_text"].isin(exclude_texts)]
        print(f"[classifier] excluded {before - len(df)} rows overlapping eval set from training")
    weak_labels = df["customer_text"].fillna("").map(rule_based_intent)

    pipe = make_pipeline()
    pipe.fit(df["customer_text"].fillna(""), weak_labels)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    return pipe


def load_classifier(model_path: str = MODEL_PATH) -> Pipeline:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_intent(pipe: Pipeline, text: str) -> str:
    return pipe.predict([text])[0]


if __name__ == "__main__":
    from src.config import PAIRS_PATH
    trained = train_from_pairs(PAIRS_PATH)
    print(f"Trained classifier on weak labels, saved to {MODEL_PATH}")
    print(trained.named_steps["clf"].classes_)
