"""How few reviewer labels the label-fitted scorer needs: the same estates at smaller budgets.

Usage: python realdata/label_curve.py out/label_curve.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_a import run  # noqa: E402

ESTATES = [("estate", "data/real_gleif_ch_small.parquet"), ("estate", "data/estate_100k.parquet"), ("historical", "data/historical_50k.parquet")]
BUDGETS = [(150, 450), (300, 600), (500, 1000), (1000, 1500), (2000, 3000)]  # (to fit, to test)
rows = []
for kind, path in ESTATES:
    for n_cal, n_ltt in BUDGETS:
        for seed in (7, 8, 9):
            r = run(kind, Path(path), n_cal, n_ltt, seed, baseline_max=0, learned=True)
            b, mo = r["bands"], r["clustering"]["machine_only"]
            row = {"estate": Path(path).stem, "fit": n_cal, "test": n_ltt, "labels": n_cal + n_ltt, "seed": seed,
                   "accept_at": b["accept_at"], "reject_below": b["reject_below"], **b["counts"],
                   "accept_precision": b["auto_accept_precision_realised"], "P": mo["pairwise_precision"], "R": mo["pairwise_recall"], "F": mo["pairwise_f1"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
            Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
