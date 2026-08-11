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
* **The ablation section.** Ablation legs (``dense_content_only``) are kept OUT of the ladder
  table: they are not increments, they are the same rung on a different input, and putting them
  in the Δ column would turn the cumulative chain into nonsense. They get their own table, which
  subtracts each against the ladder rung it is the counterfactual of.
* **The weighted-fusion section.** Every number in the ladder table is EQUAL-weight RRF. The
  optional tuned variant (:mod:`benchmark.s5.tuning`) is rendered in its own section, against its
  own equal-weight number recomputed on its own test split — never against the full-set ladder
  cell, which is a different query set. The section states, in the report itself, that those
  numbers are dev-tuned and therefore not comparable to published baselines that did no tuning.
* **The chain-coverage divergence.** ``recall@10`` and ``chain_coverage@10`` measure different
  things (chunks retrieved vs gold DOCUMENTS covered), and on the graph legs they move in
  opposite directions. A step where they disagree in sign is called out explicitly, because that
  divergence — not the headline recall drop — is what says the graph signals carry information
  the equal-weight fusion is spending badly.

Gold-resolution rates are printed per dataset and a dataset below
:data:`~benchmark.s5.goldmap.TRUST_THRESHOLD` is marked ``FLAGGED`` in place of an implied
endorsement of its numbers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from benchmark.s5.datasets import CONTROL_DATASETS
from benchmark.s5.goldmap import TRUST_THRESHOLD
from benchmark.s5.legs import (
    ABLATION_BASELINE,
    ABLATION_LEG_NAMES,
    LEG_NAMES,
    RESCUE_LEG_NAMES,
    leg_signals,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

_COLUMNS = (
    ("recall@1", "R@1"),
    ("recall@5", "R@5"),
    ("recall@10", "R@10"),
    # Document-level, so the published comparison line (HippoRAG-2's passage recall@5) has a
    # like-for-like column instead of being set against a chunk-level number. Cells written before
    # this column existed simply render "—" — an absent measurement, not a zero.
    ("doc_recall@5", "docR@5"),
    ("doc_recall@10", "docR@10"),
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
    weighted: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Write ``REPORT.md`` and ``metrics.json`` for ``matrix`` into ``out_dir``.

    ``matrix`` is always the EQUAL-weight baseline — the ladder table, the deltas and the control
    check are all read off it. ``weighted`` is the optional dev-tuned variant; when it carries
    cells they are written to ``metrics-weighted.json`` and rendered as an extra section, and when
    it is empty the report is exactly the report the baseline alone produces.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if weighted:
        (out_dir / "metrics-weighted.json").write_text(
            json.dumps(weighted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
        "**Every number in the ladder tables below is EQUAL-weight RRF** — each leg's signals "
        "enter the fusion with the same pull. That is the assumption-free baseline, and it is "
        "also why a graph leg can lower `R@10`: RRF sums `w / (k + rank)`, so appending a noisier "
        "list at equal weight to a strong `dense+bm25` pool necessarily drags the fused ranking "
        "toward the noise. The dev-tuned variant (its own section, if it was run) is the "
        "controlled answer to \"does the signal help when fused properly?\"; it is NOT a "
        "replacement for these numbers.",
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

    ladder = [leg for leg in legs if leg in LEG_NAMES]
    for dataset in sorted(matrix):
        control = " (single-hop CONTROL)" if dataset in CONTROL_DATASETS else ""
        lines += ["", f"## {dataset}{control}", "", _cap_note(matrix[dataset]), ""]
        lines += _dataset_table(matrix[dataset], ladder)

    lines += _ablation_section(matrix, legs)
    lines += _rescue_section(matrix, legs)
    lines += _divergence_section(matrix, ladder)
    lines += _weighted_section(weighted or {}, legs)

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


def _ablation_section(
    matrix: Mapping[str, Mapping[str, Any]], legs: Sequence[str]
) -> list[str]:
    """The ``dense (+context-header)`` vs ``dense (content-only)`` table, per dataset.

    Emitted only when an ablation leg was actually requested, so a plain ladder run's report does
    not carry an empty promise. Rows the ablation could not run (no sidecar built yet) are shown
    as the refusal they are, because an absent content-only number must not read as "no
    difference".
    """
    wanted = [leg for leg in legs if leg in ABLATION_LEG_NAMES]
    if not wanted:
        return []
    lines = [
        "",
        "## Ablation — what the W13 context header bought the dense leg",
        "",
        "The `dense` rung ranks the vectors the S3 Export wrote, which embed each chunk's "
        "content **plus** its rendered ContextHeader (the chunk's KG entities and neighbour "
        "triples). `dense (content-only)` re-runs that same rung against vectors built from the "
        "content alone. Only the CORPUS side differs — queries carry no header, so both legs "
        "rank identical query vectors. A positive Δ is what the enrichment is worth; it also "
        "says how much of every leg above `dense` is header, not retrieval. This matters "
        "unevenly: the S3 corpora carry populated headers while the reused M6 exports carry "
        "`context_header: null` on every row, so without this table a single `dense` label "
        "would silently mean two different things across datasets.",
        "",
        "| dataset | leg | baseline R@10 | ablated R@10 | Δ (header − content-only) |",
        "|---|---|---|---|---|",
    ]
    for dataset in sorted(matrix):
        cells = matrix[dataset]
        for leg in wanted:
            baseline_name = ABLATION_BASELINE.get(leg, "dense")
            cell = cells.get(leg)
            if not isinstance(cell, dict):
                continue
            if cell.get("status") != "ok":
                missing = ", ".join(cell.get("missing", []))
                lines.append(
                    f"| {dataset} | {leg} | — | *missing inputs: {missing}* | — |"
                )
                continue
            ablated = cell["metrics"].get("recall@10")
            base_cell = cells.get(baseline_name)
            baseline = (
                base_cell["metrics"].get("recall@10")
                if isinstance(base_cell, dict) and base_cell.get("status") == "ok"
                else None
            )
            delta = (
                None if baseline is None or ablated is None else float(baseline) - float(ablated)
            )
            lines.append(
                f"| {dataset} | {leg} | {_fmt(baseline)} | {_fmt(ablated)} | {_signed(delta)} |"
            )
    return lines


def _rescue_section(
    matrix: Mapping[str, Mapping[str, Any]], legs: Sequence[str]
) -> list[str]:
    """The graph-as-rescue family, against the ladder's graph rungs — plus the gate and the curve.

    Kept OUT of the ladder table for the same reason the ablations are: these legs are not
    increments, so a ``Δ`` column through them would turn the cumulative chain into nonsense. What
    they *are* is a second application of the same components, so the comparison that matters is
    against the ladder's own graph rungs — ``kg``, ``hyperedges``, ``coe``, ``packer`` — and that is
    the table below.
    """
    wanted = [leg for leg in legs if leg in RESCUE_LEG_NAMES]
    if not wanted:
        return []
    lines = [
        "",
        "## Graph as RESCUE, not as a fused list (S5b)",
        "",
        "The ladder applies the graph lane as a co-equal RRF list, which is the application the "
        "matrix above measures regressing. These legs apply the *same shipped components* the way "
        "the framework's seams (and `colsearch`'s shipped graph sidecar) describe: `dense+bm25` is "
        "the base ranking; `expand.Expander` rescores it from the KG columns already denormalised "
        "onto every chunk; the graph lanes only **append** candidates the base missed, "
        "deduplicated by id and never reordering the base; and a selector spends a token budget "
        "over the union. Entity resolution is seeded from the query **and** the top first-stage "
        "documents. Zero LLM calls.",
        "",
        "| dataset | leg | R@1 | R@5 | R@10 | docR@5 | docR@10 | chain cov | full chain | "
        "gate hit rate | lat med ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for dataset in sorted(matrix):
        cells = matrix[dataset]
        for leg in ("kg", "hyperedges", "coe", "packer", *wanted):
            cell = cells.get(leg)
            if not isinstance(cell, dict):
                continue
            if cell.get("status") != "ok":
                lines.append(
                    f"| {dataset} | {leg} | *{cell.get('status', 'not run')}* |" + " |" * 9
                )
                continue
            metrics = cell["metrics"]
            gate = cell.get("gate")
            rate = (
                f"{float(gate['hit_rate']):.1%}"
                if isinstance(gate, dict) and gate.get("hit_rate") is not None
                else "— (ladder leg: no gate)"
            )
            lines.append(
                f"| {dataset} | {leg} | "
                + " | ".join(
                    _fmt(metrics.get(key))
                    for key in (
                        "recall@1",
                        "recall@5",
                        "recall@10",
                        "doc_recall@5",
                        "doc_recall@10",
                        "chain_coverage@10",
                        "full_chain_hit@10",
                    )
                )
                + f" | {rate} | {_fmt(metrics.get('latency_ms_median'))} |"
            )
    curve = _budget_curve_rows(matrix, wanted)
    if curve:
        lines += [
            "",
            "### Recall over the token budget (the selection legs' native protocol)",
            "",
            "A selector optimises *best context within a budget*, so a bare `recall@10` is the "
            "wrong shape of number for it. The curve below settles every budget against the "
            "**identical** candidate union — one retrieval per query, five selections — so what "
            "varies is the budget and nothing else. The headline row above is the "
            "`select_budget` point of this curve.",
            "",
            "| dataset | leg | budget | R@10 | docR@5 | docR@10 | chain cov | "
            "tokens packed (mean) |",
            "|---|---|---|---|---|---|---|---|",
            *curve,
        ]
    return lines


def _budget_curve_rows(
    matrix: Mapping[str, Mapping[str, Any]], wanted: Sequence[str]
) -> list[str]:
    """One row per (dataset, leg, budget) — ascending budget, so the curve reads as a curve."""
    rows: list[str] = []
    for dataset in sorted(matrix):
        for leg in wanted:
            cell = matrix[dataset].get(leg)
            if not isinstance(cell, dict) or cell.get("status") != "ok":
                continue
            curve = cell.get("budget_curve")
            if not isinstance(curve, dict) or len(curve) < 2:
                continue
            for budget in sorted(curve, key=int):
                point = curve[budget]
                rows.append(
                    f"| {dataset} | {leg} | {budget} | "
                    + " | ".join(
                        _fmt(point.get(key))
                        for key in (
                            "recall@10",
                            "doc_recall@5",
                            "doc_recall@10",
                            "chain_coverage@10",
                            "tokens_packed_mean",
                        )
                    )
                    + " |"
                )
    return rows


def _divergence_section(
    matrix: Mapping[str, Mapping[str, Any]], ladder: Sequence[str]
) -> list[str]:
    """Per ladder step, ``Δ R@10`` beside ``Δ chain coverage@10`` — and where they disagree.

    The two columns are not the same measurement said twice. ``R@10`` is set recall over gold
    CHUNKS; ``chain cov`` is the fraction of gold DOCUMENTS (chain slots) with at least one chunk
    in the top 10. A step that lowers the first while raising the second has not lost information
    — it has reached *more of the gold documents* while spending top-10 slots that used to hold
    several chunks of one document. That is the single most informative pattern in this matrix,
    so it gets a table rather than a sentence someone has to reconstruct from two other tables.

    Only steps where BOTH deltas exist are rowed; a dataset with no chain (the single-hop
    controls) contributes nothing here rather than a row of dashes.
    """
    rows: list[str] = []
    for dataset in sorted(matrix):
        cells = matrix[dataset]
        for lower, upper in zip(ladder, ladder[1:], strict=False):
            recall = _delta(cells, lower, upper)
            chain = _delta(cells, lower, upper, metric="chain_coverage@10")
            if recall is None or chain is None:
                continue
            diverges = (recall < 0 < chain) or (chain < 0 < recall)
            reading = (
                "**diverges — the signal reaches more gold documents while the fusion "
                "loses chunks**"
                if diverges
                else "same direction"
            )
            rows.append(
                f"| {dataset} | {lower} → {upper} | {_signed(recall)} | {_signed(chain)} | "
                f"{reading} |"
            )
    if not rows:
        return []
    return [
        "",
        "## Chain coverage vs recall — where the two disagree",
        "",
        "`R@10` counts gold CHUNKS retrieved; `chain cov` counts gold DOCUMENTS covered. They "
        "can move in opposite directions, and on the graph legs they do. Where they diverge, the "
        "equal-weight recall drop is **not** evidence that the signal is worthless: the signal is "
        "surfacing documents the dense+bm25 pool missed, while the equal-weight fusion pays for "
        "them with top-10 slots. Whether that trade is worth making is exactly the question the "
        "dev-tuned variant below is there to answer.",
        "",
        "| dataset | step | Δ R@10 | Δ chain cov@10 | reading |",
        "|---|---|---|---|---|",
        *rows,
    ]


def _weighted_section(
    weighted: Mapping[str, Mapping[str, Any]], legs: Sequence[str]
) -> list[str]:
    """The dev-tuned fusion variant: both variants on the SAME test split, plus the fitted weights.

    Emitted only when tuned cells exist. Every comparison inside it is equal-vs-tuned on the
    identical held-out queries from the identical retrieved lists, because the alternative —
    subtracting the tuned test number from the full-set ladder cell — would mix a fusion change
    with a change of query set and report the sum as if it were the former.
    """
    if not weighted:
        return []
    lines = [
        "",
        "## Dev-tuned fusion weights (second variant — NOT the baseline)",
        "",
        "The ladder above fuses every leg's signals at **equal** RRF weight. This section fits a "
        "per-signal weight vector per (dataset, leg) and reports it on held-out queries:",
        "",
        "* **Split.** Each dataset's usable queries are partitioned by a stable keyed hash of the "
        "query id (`blake2b(salt:query_id) % 1000 < dev_permille`) — no RNG, no ordering "
        "dependence, and independent of `--limit`, so a query is on the same side in every run. "
        "Sizes are stated per row.",
        "* **Fit.** Coordinate ascent over a fixed weight grid, from all-ones, maximising mean "
        "**nDCG@10** (the same definition this report prints) on the DEV half only. The first "
        "signal is pinned at 1.0 because RRF weights are scale-invariant. A value is adopted only "
        "on a strict improvement, so the fit can never land below the equal-weight start.",
        "* **Report.** Metrics are computed on the TEST half only, and the equal-weight column is "
        "recomputed on those same test queries from the same retrieved lists — so `Δ` is a "
        "difference in **fusion** and nothing else.",
        "",
        "**These tuned numbers are not comparable to published baselines.** Those baselines "
        "report an untuned system; these were fitted on this corpus's own dev queries. The "
        "comparable number remains the equal-weight ladder above. What this table answers is "
        "narrower and internal: *given the same candidate lists, does weighting the graph signals "
        "properly recover what equal-weight fusion loses?*",
        "",
        "| dataset | leg | dev n | test n | R@10 equal | R@10 tuned | Δ R@10 | nDCG@10 equal | "
        "nDCG@10 tuned | Δ nDCG@10 | chain cov equal | chain cov tuned |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    weights_rows: list[str] = []
    for dataset in sorted(weighted):
        cells = weighted[dataset]
        for leg in legs:
            cell = cells.get(leg)
            if not isinstance(cell, dict):
                continue
            if cell.get("status") != "ok":
                lines.append(
                    f"| {dataset} | {leg} | — | — | *{cell.get('status', 'not run')}* |"
                    + " |" * 8
                )
                continue
            tuned = cell.get("metrics", {})
            equal = cell.get("metrics_equal_weight_on_test", {})
            split = cell.get("split", {})
            lines.append(
                f"| {dataset} | {leg} | {split.get('dev_queries_tuned_on', '—')} | "
                f"{split.get('test_queries', '—')} | "
                f"{_fmt(equal.get('recall@10'))} | {_fmt(tuned.get('recall@10'))} | "
                f"{_signed(_difference(tuned, equal, 'recall@10'))} | "
                f"{_fmt(equal.get('ndcg@10'))} | {_fmt(tuned.get('ndcg@10'))} | "
                f"{_signed(_difference(tuned, equal, 'ndcg@10'))} | "
                f"{_fmt(equal.get('chain_coverage@10'))} | "
                f"{_fmt(tuned.get('chain_coverage@10'))} |"
            )
            weights_rows.append(_weights_row(dataset, leg, cell))
    if weights_rows:
        lines += [
            "",
            "### Fitted weights (auditable — every one of them)",
            "",
            "A weight of `0.00` is the fit saying *drop this list*, which is a finding, not a "
            "failure — it is an answer the equal-weight run cannot express. `dev nDCG@10` shows "
            "what the fit bought **on the fitting set**; the honest number is the Δ in the table "
            "above, on the test set.",
            "",
            "| dataset | leg | fitted weights | grid evals | search | "
            "dev nDCG@10 equal → tuned |",
            "|---|---|---|---|---|---|",
            *weights_rows,
        ]
    return lines


def _weights_row(dataset: str, leg: str, cell: Mapping[str, Any]) -> str:
    """One fit, fully auditable: the weights, the cost, and whether the search actually settled."""
    tuning = cell.get("tuning", {})
    weights = tuning.get("weights", {})
    # Ladder order, not the alphabetical order the cell was serialised in: a weight vector reads
    # as "what each rung is worth relative to dense", which only makes sense in rung order.
    order = [name for name in leg_signals(leg) if name in weights]
    order += [name for name in weights if name not in order]
    rendered = ", ".join(f"{name} {float(weights[name]):.2f}" for name in order)
    if cell.get("tunable") is False:
        rendered = "— (single signal: the RRF weight is a pure scale, so there is nothing to fit)"
    objective = tuning.get("objective", "ndcg@10")
    before = tuning.get(f"dev_{objective}_equal_weight")
    after = tuning.get(f"dev_{objective}_tuned")
    # "stopped improving" and "ran out of passes" are different claims about the same weights.
    search = (
        "converged"
        if tuning.get("converged")
        else f"**hit the {tuning.get('max_passes', '?')}-pass cap**"
    )
    return (
        f"| {dataset} | {leg} | {rendered or '—'} | {tuning.get('grid_evaluations', '—')} | "
        f"{search} | {_fmt(before)} → {_fmt(after)} |"
    )


def _difference(
    tuned: Mapping[str, Any], equal: Mapping[str, Any], key: str
) -> float | None:
    a, b = tuned.get(key), equal.get(key)
    if a is None or b is None:
        return None
    return float(a) - float(b)


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


def _delta(
    cells: Mapping[str, Any], lower: str, upper: str, *, metric: str = "recall@10"
) -> float | None:
    """``upper − lower`` for one metric, or ``None`` unless both cells ran and both carry it."""
    low, high = cells.get(lower), cells.get(upper)
    if not isinstance(low, dict) or not isinstance(high, dict):
        return None
    if low.get("status") != "ok" or high.get("status") != "ok":
        return None
    a, b = low["metrics"].get(metric), high["metrics"].get(metric)
    if a is None or b is None:
        return None
    return float(b) - float(a)


def _signed(value: float | None) -> str:
    return "—" if value is None else f"{value:+.4f}"


__all__ = ["write_report"]
