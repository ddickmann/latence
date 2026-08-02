#!/usr/bin/env python3
"""Summarise the gamma-calibration judgments: the raw material an isotonic fit consumes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def main() -> int:
    base = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "musique-full")
    sample = {r["hyperedge_id"]: r for r in map(json.loads, (base / "calibration-sample.jsonl").open())}
    rows = []
    for j in map(json.loads, (base / "calibration-judgments.jsonl").open()):
        s = sample[j["hyperedge_id"]]
        rows.append({**s, "judgment": j["judgment"]})
    rows.sort(key=lambda r: r["gamma"])
    n = len(rows)

    def block(label: str, sel: list[dict]) -> str:
        if not sel:
            return f"| {label} | 0 | – | – | – | – |"
        c = sum(1 for r in sel if r["judgment"] == "correct")
        p = sum(1 for r in sel if r["judgment"] == "partial")
        i = sum(1 for r in sel if r["judgment"] == "incorrect")
        acc = sum(SCORE[r["judgment"]] for r in sel) / len(sel)
        return (
            f"| {label} | {len(sel)} | {c} ({c/len(sel):.0%}) | {p} ({p/len(sel):.0%}) "
            f"| {i} ({i/len(sel):.0%}) | {acc:.3f} |"
        )

    print("### gamma quintiles (50 Hyperedges each, sample sorted by gamma)\n")
    print("| gamma range | n | correct | partial | incorrect | score |")
    print("|---|---|---|---|---|---|")
    for q in range(5):
        sel = rows[q * n // 5 : (q + 1) * n // 5]
        lo, hi = sel[0]["gamma"], sel[-1]["gamma"]
        print(block(f"Q{q+1} {lo:.3f}–{hi:.3f}", sel))
    print()
    print("### by structure\n")
    print("| slice | n | correct | partial | incorrect | score |")
    print("|---|---|---|---|---|---|")
    print(block("all", rows))
    print(block("has_relations=False (weak)", [r for r in rows if not r["has_relations"]]))
    print(block("has_relations=True", [r for r in rows if r["has_relations"]]))
    print(block("arity 1", [r for r in rows if r["arity"] == 1]))
    print(block("arity 2-4", [r for r in rows if 2 <= r["arity"] <= 4]))
    print(block("arity >=5", [r for r in rows if r["arity"] >= 5]))
    print(block("support > 1", [r for r in rows if r["support"] > 1]))
    print(block("arity-split sibling group", [r for r in rows if r["chunk_siblings"] >= 3]))
    print()
    print("### relation-bearing decile detail (the 80 with relations)\n")
    rel = [r for r in rows if r["has_relations"]]
    for k in range(4):
        sel = rel[k * len(rel) // 4 : (k + 1) * len(rel) // 4]
        print(block(f"rel-Q{k+1} {sel[0]['gamma']:.3f}–{sel[-1]['gamma']:.3f}", sel))
    print()
    tri = sum(len(r["relations"]) for r in rel)
    print(f"relation triples in sample: {tri} over {len(rel)} relation-bearing Hyperedges")
    print(f"weak (entities-only) Hyperedges: {n - len(rel)} / {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
