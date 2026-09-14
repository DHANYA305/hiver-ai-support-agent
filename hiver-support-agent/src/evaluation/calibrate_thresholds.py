"""
Calibrates the retrieval-similarity thresholds used in
src/escalation/policy.py, on a DEV sample drawn from pairs.csv — NOT the
golden set, to avoid tuning thresholds on the same data used to report
final metrics.

Why this exists at all (decision_log.md #12): the first version of
policy.py had thresholds picked by eyeballing 3-4 example queries against
the full (leaky) retrieval pool. Once we built the leak-free eval index
(RetrievalIndex.build_eval_index — see report.md's "misleading number"
section), similarity scores dropped across the board and the eyeballed
0.35 threshold flagged ~75% of golden examples as low-confidence. This
script re-derives thresholds from the actual similarity distribution a
held-out message sees, so the policy's aggressiveness is a deliberate
choice, not an accident of where the number came from.

Run: python -m src.evaluation.calibrate_thresholds
"""
import pandas as pd

from src.config import PAIRS_PATH, RANDOM_SEED
from src.retrieval.index import RetrievalIndex

DEV_SAMPLE_SIZE = 300


def calibrate(dev_size: int = DEV_SAMPLE_SIZE, seed: int = RANDOM_SEED + 1) -> pd.Series:
    from src.config import GOLDEN_PATH
    pairs = pd.read_csv(PAIRS_PATH)
    try:
        golden_texts = set(pd.read_csv(GOLDEN_PATH)["customer_text"])
        pairs = pairs[~pairs["customer_text"].isin(golden_texts)]
    except FileNotFoundError:
        pass  # calibration can still run before the golden set exists
    dev = pairs.sample(n=dev_size, random_state=seed)
    index = RetrievalIndex.build_eval_index(PAIRS_PATH, exclude_texts=set(dev["customer_text"]))
    sims = [index.search(t, k=1)[0]["similarity"] for t in dev["customer_text"]]
    return pd.Series(sims)


if __name__ == "__main__":
    dist = calibrate()
    print(dist.describe())
    print(dist.quantile([0.25, 0.4, 0.5, 0.6, 0.75]))
    print(
        "\nChosen: LOW_CONFIDENCE_THRESHOLD=0.22 (~p25 of held-out top-1 similarity: "
        "below this, a held-out message usually has no genuinely similar historical case), "
        "SENSITIVE_INTENT_THRESHOLD=0.30 (~p50: for billing/account we want at least a "
        "typical-quality match before trusting an auto-reply)."
    )
