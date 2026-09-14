"""
Retrieval over historical (customer_text -> agent_text) pairs, used to
ground generated replies in how the brand actually resolved similar issues.

Why TF-IDF cosine similarity over sentence-embeddings/FAISS (see
decision_log.md #6): support tweets are short and heavy on
domain-specific tokens ("premium", "buffering", "downgrade") that TF-IDF
handles well; it needs no model download, runs on CPU in milliseconds for
12k documents, and is trivially reproducible in the 15-minute repro
budget. A prior scratch notebook (notebooks/) tried
sentence-transformers + FAISS; we intentionally did not carry that into
the graded pipeline — see report.md.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = "results/retrieval_index.pkl"


class RetrievalIndex:
    def __init__(self, pairs_df: pd.DataFrame):
        self.pairs = pairs_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.9, sublinear_tf=True,
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.pairs["customer_text"].fillna(""))

    def search(self, query: str, k: int = 3) -> list[dict]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix)[0]
        top_idx = np.argsort(-sims)[:k]
        results = []
        for idx in top_idx:
            row = self.pairs.iloc[idx]
            results.append({
                "customer_text": row["customer_text"],
                "agent_text": row["agent_text"],
                "similarity": float(sims[idx]),
            })
        return results

    def save(self, path: str = INDEX_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = INDEX_PATH) -> "RetrievalIndex":
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def build_from_csv(pairs_csv: str) -> "RetrievalIndex":
        df = pd.read_csv(pairs_csv)
        return RetrievalIndex(df)

    @staticmethod
    def build_eval_index(pairs_csv: str, exclude_texts: set) -> "RetrievalIndex":
        """Build the index used DURING EVALUATION ONLY, with any customer
        message that appears in the golden set removed from the retrieval
        pool first.

        Why this exists (decision_log.md #11 / report.md's mandatory
        "misleading headline number" section): the golden set was sampled
        FROM the same pairs.csv used to build the production retrieval
        index. If we evaluate with the leaky index, retrieval finds the
        golden example's own historical reply verbatim (similarity=1.0)
        almost every time, and every quality metric looks close to
        perfect — for a boring, meaningless reason. This method is the
        fix; `src/evaluation/run_eval.py` uses it exclusively."""
        df = pd.read_csv(pairs_csv)
        df = df[~df["customer_text"].isin(exclude_texts)].reset_index(drop=True)
        return RetrievalIndex(df)


if __name__ == "__main__":
    from src.config import PAIRS_PATH
    idx = RetrievalIndex.build_from_csv(PAIRS_PATH)
    idx.save()
    demo = idx.search("my premium subscription keeps playing ads", k=3)
    for r in demo:
        print(round(r["similarity"], 3), "|", r["customer_text"][:70], "->", r["agent_text"][:70])
