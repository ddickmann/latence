"""Write every candidate pair of an estate with the first-stage score and the truth, and the
normalised records, so that later scoring ideas can be tried without blocking again.

Usage: python realdata/pairs_dump.py --path data/real_gleif_ch_small.parquet --out out/pairs_small
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_a import ORG_PASSES, _codes, block, normalise_org, record_numbers, score, vetoes  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    df = normalise_org(pd.read_parquet(a.path))
    rid, _ = record_numbers(df.unique_id.to_numpy())
    df["rid"] = rid
    by_record = df.iloc[np.argsort(rid, kind="stable")].reset_index(drop=True)
    left, right, stats = block(df, ORG_PASSES)
    print("blocked", len(left), {k: (v["pairs"], round(v["pair_completeness"], 4)) for k, v in stats.items()}, round(time.time() - t0), flush=True)
    p = score(df, left, right, "org")
    print("scored", round(time.time() - t0), flush=True)
    veto = vetoes(by_record, left, right, "org")
    gold = _codes(by_record.cluster.astype(str).to_numpy())
    y = gold[left] == gold[right]
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"l": left, "r": right, "p": p, "veto": veto, "y": y}).to_parquet(a.out / "pairs.parquet")
    by_record.to_parquet(a.out / "recs.parquet")
    print("gold pairs", stats["_union"]["gold_pairs"], "candidate true", int(y.sum()), "done", round(time.time() - t0))


if __name__ == "__main__":
    main()
