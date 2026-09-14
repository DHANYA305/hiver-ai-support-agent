"""
Build (customer_message -> agent_reply) pairs for one brand from the raw
Customer Support on Twitter dump (twcs.csv).

Scope decision (see decision_log.md #3): we only keep the FIRST
customer message in a thread paired with the brand's FIRST reply to it.
Twitter support threads are often 4-8 turns deep and later turns lean on
private DMs ("please DM us") which aren't in this dataset — modelling
multi-turn state is out of scope for a take-home; see report.md "what we
chose not to build".

Usage:
    python -m src.data.build_pairs [--max-rows N] [--sample N]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import BRAND_HANDLE, PAIRS_PATH, RAW_TWCS_PATH, RANDOM_SEED
from src.text_utils import clean_tweet, is_low_signal


def build_pairs(raw_path: str, brand: str, max_rows: int | None = None) -> pd.DataFrame:
    usecols = ["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id", "created_at"]
    dtype = {"tweet_id": "int64", "in_response_to_tweet_id": "float64"}
    df = pd.read_csv(raw_path, usecols=usecols, dtype=dtype, nrows=max_rows)

    df["inbound"] = df["inbound"].astype(str).str.lower() == "true"

    # Agent replies made by the brand
    agent_msgs = df[(df["author_id"] == brand) & (~df["inbound"])].copy()
    # Customer messages (anyone inbound)
    cust_msgs = df[df["inbound"]].set_index("tweet_id")

    records = []
    for _, row in agent_msgs.iterrows():
        parent_id = row["in_response_to_tweet_id"]
        if pd.isna(parent_id):
            continue
        parent_id = int(parent_id)
        if parent_id not in cust_msgs.index:
            continue
        cust_row = cust_msgs.loc[parent_id]
        # a tweet_id can theoretically map to multiple rows if duplicated ids exist; guard it
        if isinstance(cust_row, pd.DataFrame):
            cust_row = cust_row.iloc[0]

        cust_text = clean_tweet(cust_row["text"])
        agent_text = clean_tweet(row["text"])
        if is_low_signal(cust_text) or is_low_signal(agent_text):
            continue

        records.append({
            "customer_tweet_id": parent_id,
            "customer_text": cust_text,
            "agent_tweet_id": int(row["tweet_id"]),
            "agent_text": agent_text,
            "created_at": row["created_at"],
        })

    out = pd.DataFrame.from_records(records)
    out = out.drop_duplicates(subset=["customer_text", "agent_text"])
    out = out.sort_values("created_at").reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-path", default=RAW_TWCS_PATH)
    ap.add_argument("--brand", default=BRAND_HANDLE)
    ap.add_argument("--max-rows", type=int, default=None,
                     help="Only read the first N rows of twcs.csv (fast dev loop / subsample).")
    ap.add_argument("--sample", type=int, default=None,
                     help="Randomly downsample the resulting pairs to N rows.")
    ap.add_argument("--out", default=PAIRS_PATH)
    args = ap.parse_args()

    pairs = build_pairs(args.raw_path, args.brand, args.max_rows)
    print(f"Built {len(pairs)} raw pairs for {args.brand}")

    if args.sample and len(pairs) > args.sample:
        pairs = pairs.sample(n=args.sample, random_state=RANDOM_SEED).sort_values("created_at")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out, index=False)
    print(f"Wrote {len(pairs)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
