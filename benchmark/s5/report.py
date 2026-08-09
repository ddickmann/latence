# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Render the S5 matrix as ``REPORT.md`` + ``metrics.json`` — deltas and control checks included.

Two things the report does that a plain table does not, both because the ablation is the point:

* **Cumulative deltas.** Each leg's row carries ``Δ recall@10`` against the leg below it, so the
  attribution ("+bm25 bought 4.1 points, +hyperedges bought 0.3") is read off the table rather
  than computed by the reader.
* **The control section.** ViDoSeek, OHR-Bench and UDA are single-hop; a hypergraph or CoE leg
  *should* be flat there. The report states the observed control deltas next to that expectation
  so a large control gain reads as the leakage warning it is, not as a headline.

Gold-resolution rates are printed per dataset and a dataset below
:data:`~benchmark.s5.goldmap.TRUST_THRESHOLD` is marked ``FLAGGED`` in place of an implied
endorsement of its numbers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from benchmark.s5.datasets import CONTROL_DATASETS
from benchmark.s5.goldmap import TRUST_THRESHOLD

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

_COLUMNS = (
    ("recall@1", "R@1"),
    ("recall@5", "R@5"),
    ("recall@10", "R@10"),
    ("ndcg@10", "nDCG@10"),
    ("chain_coverage@10", "chain cov"),
    ("full_chain_hit@10", "full chain"),
    ("latency_ms_median", "lat med ms"),
    ("latency_ms_p95", "lat p95 ms"),
)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _dataset_table(cells: Mapping[str, Any], legs: Sequence[str]) -> list[str]:
    header = "| leg | " + " | ".join(label for _key, label in _COLUMNS) + " | Δ R@10 |"
    lines = [header, "|" + "---|" * (len(_COLUMNS) + 2)]
    previous: float | None = None
    for leg in legs:
        cell = cells.get(leg)
        if not isinstance(cell, dict):
            continue
        if cell.get("status") != "ok":
            missing = ", ".join(cell.get("missing", []))
            lines.append(f"| {leg} | *missing inputs: {missing}* |" + " |" * len(_COLUMNS))
            continue
        metrics = cell["metrics"]
        row = [_fmt(metrics.get(key)) for key, _label in _COLUMNS]
        current = metrics.get("recall@10")
        delta = "—" if previous is None or current is None else f"{current - previous:+.4f}"
        if isinstance(current, (int, float)):
            previous = float(current)
        lines.append(f"| {leg} | " + " | ".join(row) + f" | {delta} |")
    return lines


def write_report(
    out_dir: Path,
    matrix: Mapping[str, Mapping[str, Any]],
    *,
    legs: Sequence[str],
    limit: int | None,
) -> None:
    """Write ``REPORT.md`` and ``metrics.json`` for ``matrix`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines: list[str] = [
        "# S5 — Retrieval feature matrix (dense → +bm25 → +KG → +hyperedges → +CoE → +packer)",
        "",
        f"Query cap for this invocation: **{limit if limit is not None else 'none (full set)'}**"
        " — each dataset section states the cap its own cells were run with, since cells are "
        "written once and resumed. Gold resolution is always measured over the FULL query set. "
        "Legs are CUMULATIVE — leg *n* runs signals 1..*n* and fuses them with RRF, so every "
        "`Δ R@10` is the increment that one signal bought.",
        "",
        "Dense retrieval is exact brute-force cosine over the exported 768-d Granite vectors "
        "(no ANN index exists — the S5 ticket's recorded choice); BM25 is the exported corpus "
        "statistics scored by the shipped `Bm25Rescorer`, which is exactly a full-corpus first "
        "stage because a document with no query term scores 0.",
        "",
        "### How to read the columns",
        "",
        "* `R@k` is **set** recall over gold CHUNKS: `|gold ∩ top-k| / |gold|`. Where a dataset's "
        "gold unit is a whole document (MultiHop-RAG articles, 2Wiki paragraphs, "
        "MuSiQue/HotpotQA supporting paragraphs), that document may have split into several "
        "chunks, so `R@k` is strictly harder than the published document-level number.",
        "* `chain cov` **is** the document-level recall@10 for those datasets — each chain slot "
        "is one gold document, so the fraction of slots covered is the fraction of gold "
        "documents retrieved. Compare published document-level baselines against this column, "
        "not against `R@10`.",
        "* `full chain` is the fraction of queries whose EVERY gold document made the top 10 — "
        "the number that separates a multi-hop retriever from a lucky single-hop one.",
        "* The `packer` leg does not rank, it **selects**: its output is the token-budgeted "
        "subset that fits the budget, typically ~15 chunks. Its recall is therefore a different "
        "quantity from the legs above it (what survives into the prompt, not what the retriever "
        "could see) and a drop there is expected, not a regression.",
        "",
        "## Gold resolution",
        "",
        "`granularity` is the mix the gold **actually** resolved at, not the mix the dataset "
        "advertises: a page reference whose document has no reconstructable S2 page map keeps "
        "coarser *document*-level gold rather than being dropped, so the two counts are stated "
        "separately instead of implying uniform page precision. `excluded` counts queries whose "
        "evidence is absent from this corpus entirely — they are removed from the denominator "
        "with their reason, because scoring them as misses would report a corpus-coverage fact "
        "as a retrieval failure.",
        "",
        "| dataset | refs resolved | rate | granularity (page / doc) | queries usable | "
        "excluded | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for dataset in sorted(matrix):
        resolution = _resolution_for(out_dir, dataset, matrix[dataset])
        if resolution is None:
            lines.append(f"| {dataset} | — | — | — | — | — | not run |")
            continue
        rate = float(resolution["resolution_rate"])
        verdict = "ok" if rate >= TRUST_THRESHOLD else "**FLAGGED — mapping, not recall**"
        lines.append(
            f"| {dataset} | {resolution['refs_resolved']}/{resolution['refs_total']} | "
            f"{rate:.1%} | {_granularity(resolution)} | "
            f"{resolution['queries_usable']}/{resolution['queries_total']} | "
            f"{_exclusions(resolution)} | {verdict} |"
        )

    for dataset in sorted(matrix):
        control = " (single-hop CONTROL)" if dataset in CONTROL_DATASETS else ""
        lines += ["", f"## {dataset}{control}", "", _cap_note(matrix[dataset]), ""]
        lines += _dataset_table(matrix[dataset], legs)

    lines += [
        "",
        "## Control check — where the hypergraph/CoE legs should buy nothing",
        "",
        "ViDoSeek, OHR-Bench and UDA are single-hop: their gold is one page or one sentence, "
        "with no chain to assemble. A `+hyperedges` or `+coe` gain there is not a win — it "
        "indicates the leg is reaching gold through a path the question did not require "
        "(leakage), so it is reported as a warning, not a headline.",
        "",
        "| control dataset | Δ R@10 from +hyperedges | Δ R@10 from +coe | reading |",
        "|---|---|---|---|",
    ]
    for dataset in sorted(d for d in matrix if d in CONTROL_DATASETS):
        hyper = _delta(matrix[dataset], "kg", "hyperedges")
        coe = _delta(matrix[dataset], "hyperedges", "coe")
        gains = [x for x in (hyper, coe) if x is not None]
        # Only a POSITIVE jump is the leakage signal. A drop is the ordinary dilution of a
        # strong dense+bm25 pool by a weak extra list under RRF, which is a finding, not a leak.
        best = max(gains, default=None)
        reading = (
            "—"
            if best is None
            else (
                "**investigate — possible leakage**"
                if best > 0.02
                else "as expected (no gain on single-hop)"
            )
        )
        lines.append(
            f"| {dataset} | {_signed(hyper)} | {_signed(coe)} | {reading} |"
        )

    lines += [
        "",
        "## Scope",
        "",
        "SPLADE, MaxSim/MUVERA and rerank/fusion ablations are OUT of this run (S5 ticket). "
        "Hop depth is 2–4 native only; 5–7-hop chains are untested and no claim is made about "
        "them.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _resolution_for(
    out_dir: Path, dataset: str, cells: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """``resolution/<dataset>.json`` if it exists, else the resolution a result cell embedded.

    The standalone file is authoritative because it is written over the dataset's FULL query set
    by ``--resolution-only``, whereas a cell's copy only covers that run's ``--limit`` sample.
    """
    path = out_dir / "resolution" / f"{dataset}.json"
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return loaded
    for cell in cells.values():
        if isinstance(cell, dict) and cell.get("status") == "ok":
            resolution = cell.get("resolution")
            if isinstance(resolution, dict):
                return resolution
    return None


def _cap_note(cells: Mapping[str, Any]) -> str:
    """What the cells below were actually run over — never inherited from the current CLI flags."""
    for cell in cells.values():
        if isinstance(cell, dict) and cell.get("status") == "ok":
            cap = cell.get("limit")
            return f"Legs run over **{cap if cap is not None else 'the full query set'}** queries."
    return "*No leg has been run for this dataset yet — gold resolution only.*"


def _granularity(resolution: Mapping[str, Any]) -> str:
    mix = resolution.get("granularity")
    if not isinstance(mix, dict):
        return "—"
    return f"{mix.get('page', 0)} / {mix.get('document', 0)}"


def _exclusions(resolution: Mapping[str, Any]) -> str:
    total = resolution.get("queries_excluded") or 0
    reasons = resolution.get("exclusions")
    if not total:
        return "0"
    detail = (
        ", ".join(f"{reason} {count}" for reason, count in sorted(reasons.items()))
        if isinstance(reasons, dict)
        else ""
    )
    return f"{total} ({detail})" if detail else str(total)


def _delta(cells: Mapping[str, Any], lower: str, upper: str) -> float | None:
    low, high = cells.get(lower), cells.get(upper)
    if not isinstance(low, dict) or not isinstance(high, dict):
        return None
    if low.get("status") != "ok" or high.get("status") != "ok":
        return None
    a, b = low["metrics"].get("recall@10"), high["metrics"].get("recall@10")
    if a is None or b is None:
        return None
    return float(b) - float(a)


def _signed(value: float | None) -> str:
    return "—" if value is None else f"{value:+.4f}"


__all__ = ["write_report"]
