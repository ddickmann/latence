#!/usr/bin/env python3
"""Assemble a multi-hop-hard RFC corpus + an INDEPENDENT cross-reference graph.

The corpus is the pipeline's input; ``_refs.jsonl`` is the human-authored (RFC-editor-authored)
cross-reference graph, kept strictly separate so it can serve as the gold signal the benchmark's
question builder mines. It is NEVER fed into the pipeline — otherwise the KG-augmented retriever
would be evaluated against a graph it was handed.

Authored for the RFC-retrieval-proof tutorial (docs/TUTORIAL-rfc-retrieval-proof.md §1). Runs on the
pod (needs network to rfc-editor.org). Stdlib only — no pipeline import, no heavy dep.

Usage:
  python benchmark/rfc/build_corpus.py --out /workspace/corpus-rfc \
      --seeds 9110,9111,9112,9113,9114 --expand obsoletes,updates,references --max-docs 80
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import deque
from pathlib import Path

_BASE = "https://www.rfc-editor.org/rfc/rfc{n}.txt"
# Header lines look like: "Obsoletes: 2818, 7230, 7231"  /  "Updates: 3864".
_HDR = {
    "obsoletes": re.compile(r"^Obsoletes:\s*([\d,\s]+)", re.MULTILINE),
    "updates": re.compile(r"^Updates:\s*([\d,\s]+)", re.MULTILINE),
}
# Body references: "[RFC7231]" or "RFC 7231" or "RFC7231".
_REF = re.compile(r"\bRFC[\s-]?(\d{3,5})\b")
_LICENSE = """# Corpus license — IETF RFCs

RFC documents are published by the RFC Editor under the IETF Trust's terms
(https://trustee.ietf.org/documents/trust-legal-provisions/), which permit free reproduction and
distribution of the RFC text. Each file here is the verbatim RFC text as retrieved from
https://www.rfc-editor.org/. This corpus is used only as retrieval input; no RFC text is modified.
"""


def _fetch(n: int, timeout: float = 30.0) -> str | None:
    try:
        with urllib.request.urlopen(_BASE.format(n=n), timeout=timeout) as r:  # noqa: S310 (fixed host)
            return r.read().decode("utf-8", errors="replace")
    except Exception as exc:  # network / 404 / decode — reported, never fatal to the whole build
        print(f"  ! rfc{n}: {type(exc).__name__} {exc}", file=sys.stderr)
        return None


def _parse_refs(text: str, self_n: int) -> dict[str, list[int]]:
    def _nums(pat: re.Pattern[str]) -> list[int]:
        out: list[int] = []
        for m in pat.finditer(text):
            out += [int(x) for x in re.findall(r"\d+", m.group(1))]
        return sorted(set(out))

    obsoletes = _nums(_HDR["obsoletes"])
    updates = _nums(_HDR["updates"])
    references = sorted({int(x) for x in _REF.findall(text)} - {self_n} - set(obsoletes) - set(updates))
    return {"obsoletes": obsoletes, "updates": updates, "references": references}


def build(out: Path, seeds: list[int], expand: set[str], max_docs: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    refs: dict[int, dict[str, list[int]]] = {}
    seen: set[int] = set()
    q: deque[int] = deque(seeds)

    while q and len(refs) < max_docs:
        n = q.popleft()
        if n in seen:
            continue
        seen.add(n)
        text = _fetch(n)
        if text is None:
            continue
        (out / f"rfc{n}.txt").write_text(text, encoding="utf-8")
        r = _parse_refs(text, n)
        refs[n] = r
        print(f"  + rfc{n}: obsoletes={len(r['obsoletes'])} updates={len(r['updates'])} refs={len(r['references'])}")
        for edge in expand:
            for m in r.get(edge, []):
                if m not in seen:
                    q.append(m)

    # Write the independent cross-reference graph (gold signal — never fed to the pipeline).
    with (out / "_refs.jsonl").open("w", encoding="utf-8") as fh:
        for n in sorted(refs):
            fh.write(json.dumps({"rfc": n, **refs[n]}) + "\n")
    (out / "LICENSE-CORPUS.md").write_text(_LICENSE, encoding="utf-8")

    # Connectivity stats — the hardness gate from the tutorial's §1 ✓ look-for.
    out_degrees = sorted(len(set(r["obsoletes"] + r["updates"] + r["references"]) & set(refs)) for r in refs.values())
    median = out_degrees[len(out_degrees) // 2] if out_degrees else 0
    # depth from seeds (BFS over intra-corpus edges)
    depth: dict[int, int] = {s: 0 for s in seeds if s in refs}
    dq: deque[int] = deque(depth)
    while dq:
        n = dq.popleft()
        for m in set(refs[n]["obsoletes"] + refs[n]["updates"] + refs[n]["references"]) & set(refs):
            if m not in depth:
                depth[m] = depth[n] + 1
                dq.append(m)
    deep = sum(1 for d in depth.values() if d >= 2)

    print(f"\ncorpus: {len(refs)} RFCs  median_out_degree={median}  docs>=2hops_from_seed={deep}")
    if median < 3 or deep < 15:
        print(
            "WARNING: corpus is too shallow to prove multi-hop (need median_out_degree>=3 and "
            ">=15 docs >=2 hops deep). Widen --max-docs or add seeds.",
            file=sys.stderr,
        )
    return len(refs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", required=True, help="comma-separated RFC numbers")
    ap.add_argument("--expand", default="obsoletes,updates,references")
    ap.add_argument("--max-docs", type=int, default=80)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    expand = {x.strip() for x in a.expand.split(",") if x.strip()}
    n = build(a.out, seeds, expand, a.max_docs)
    print(f"done: {n} RFCs written to {a.out}")


if __name__ == "__main__":
    main()
