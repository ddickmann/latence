#!/usr/bin/env python3
"""Assemble the measured results + charts into a decision-maker-facing ``PROOF.md`` (tutorial §7).

``run_benchmark.py`` (§6) produced the **measured** comparison (``rfc-results.json``) and
``stage_charts.py`` (§3.3) produced the **correctness** SVGs. This script is the last step of the
RFC-retrieval-proof: it reads those artifacts and writes a single ``PROOF.md`` a non-technical
decision-maker can read end-to-end, carrying exactly what §7's ``✓ look for`` demands:

* the **headline recall table** (dense / bm25 / splade / fused) with fused's lift over the best
  single signal, plus a self-contained **headline recall bar chart** this script renders itself;
* **3–5 worked example questions**, each showing *why a single signal misses and fusion recovers* —
  preferring the §6 "fused-only wins" (fused finds gold in top-k that no single signal does), joined
  with the question text + KG chain from the §5 benchmark JSON (``meta.questions``);
* the **§3.3 correctness charts** embedded from ``--charts``;
* an **honest limitations** section — the corpus ceiling (generated-vs-survived), any question no
  configuration answered, whether fused actually beat every single signal, and the
  **SPLADE-v3 non-commercial (CC-BY-NC-SA-4.0) caveat** (tutorial Appendix A).

**The claim it must support is the defensible one** — *"the fused stack unlocks a class of queries
single retrievers structurally cannot answer"* — not an inflated single number. When fused does
**not** beat every single signal, that is surfaced at the top as the most important finding, never
buried (tutorial §6/§7).

Everything is read and written with the **standard library only**; the one chart this script
emits is a hand-written, self-contained SVG (ADR-0016: no heavy chart library, no external asset).
It consumes the already-produced retrieval artifacts, so it needs **no pod, network, Qdrant, GPU,
or model** — and no ``latence_retrieval`` import.

Usage (the exact invocation from docs/TUTORIAL-rfc-retrieval-proof.md §7):
  python benchmark/rfc/make_report.py \
      --results /workspace/report/rfc-results.json \
      --charts /workspace/report/stage-charts \
      --out /workspace/report/PROOF.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as _xml_escape

# The four configs, in report order (mirrors run_benchmark.ALL_CONFIGS).
ALL_CONFIGS: tuple[str, ...] = ("dense", "bm25", "splade", "fused")
SINGLE_CONFIGS: tuple[str, ...] = ("dense", "bm25", "splade")
FUSED_CONFIG = "fused"

# The §3.3 correctness charts stage_charts.py writes, in the order they read best in the report.
STAGE_CHART_FILES: tuple[tuple[str, str], ...] = (
    ("chunk-size-distribution.svg", "Chunk-size distribution — mass at/under the token budget"),
    (
        "entities-relations-funnel.svg",
        "Entities / relations funnel — extract → disambiguate collapses duplicate entities",
    ),
    ("kg-degree-and-centrality.svg", "Knowledge-graph degree distribution + top-centrality nodes"),
    (
        "induced-vs-independent-overlap.svg",
        "Induced-vs-independent edge overlap — the correctness bar (§3.1)",
    ),
)

DEFAULT_MAX_EXAMPLES = 5
MIN_EXAMPLES = 3

# The SPLADE-v3 license posture the proof deliberately carries (tutorial Appendix A). Stated in the
# report verbatim so the exported-artifact terms travel with the numbers — never a silent default.
SPLADE_LICENSE_CAVEAT = (
    "The SPLADE signal in this proof uses the `naver/splade-v3` checkpoint, whose **weights are "
    "CC-BY-NC-SA-4.0 (non-commercial, share-alike)**. This is a deliberate, operator-accepted "
    "choice for a **private internal proof only**. The exported sparse vectors are a derivative "
    "and inherit the non-commercial + share-alike terms, so this configuration is **not shippable "
    "as a "
    "default**. To ship a permissive default, swap to an Apache-2.0 sparse encoder (OpenSearch "
    "neural-sparse or `prithivida/Splade_PP_en_v1`)."
)


# ---------------------------------------------------------------------------
# Typed shapes read from the artifacts.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerQuestion:
    """One question's measured recall@k per config, from ``rfc-results.json`` per_question."""

    qid: str
    qtype: str
    recall: dict[str, float]
    fused_only_win: bool


@dataclass(frozen=True)
class Results:
    """The measured comparison (``rfc-results.json``): aggregate + per-type + per-question."""

    configs: tuple[str, ...]
    top_k: int
    recall_at_k: dict[str, float]
    by_type: dict[str, dict[str, float]]
    per_question: list[PerQuestion]
    meta: dict[str, Any]


@dataclass(frozen=True)
class Hop:
    """One typed edge of a mined KG chain (mirrors build_questions.Hop): src --rel--> dst."""

    src: int
    rel: str
    dst: int


@dataclass(frozen=True)
class QuestionDetail:
    """The §5 benchmark JSON detail for a question, joined into worked examples by id."""

    question: str
    subject: str
    gold_chunk_ids: tuple[str, ...]
    kg_chain: tuple[Hop, ...]
    fused_only_gold_ids: tuple[str, ...]


@dataclass(frozen=True)
class QuestionsDoc:
    """The §5 benchmark JSON: the per-question detail map + its honest generated/survived meta."""

    meta: dict[str, Any]
    detail: dict[str, QuestionDetail] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Readers — standard library only, strict (a malformed artifact is a finding, not something to
# paper over — mirrors run_benchmark.load_questions's discipline).
# ---------------------------------------------------------------------------


def _as_float(value: object) -> float:
    """A JSON number (int or float, never bool) as ``float`` — else ``0.0``."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def load_results(path: Path) -> Results:
    """Load ``rfc-results.json`` into a typed :class:`Results` (raises on a malformed payload)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"results JSON must be an object, got {type(payload).__name__}."
        raise ValueError(msg)
    raw_meta = payload.get("meta")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}

    raw_configs = meta.get("configs")
    if isinstance(raw_configs, list) and raw_configs:
        configs = tuple(str(c) for c in raw_configs)
    else:
        # Fall back to whatever recall_at_k reports, in canonical order.
        recall_block = payload.get("recall_at_k")
        present = set(recall_block) if isinstance(recall_block, dict) else set()
        configs = tuple(c for c in ALL_CONFIGS if c in present)
    if not configs:
        msg = "results JSON names no configs (neither meta.configs nor recall_at_k)."
        raise ValueError(msg)

    top_k_raw = meta.get("top_k")
    top_k = top_k_raw if isinstance(top_k_raw, int) and not isinstance(top_k_raw, bool) else 10

    raw_recall = payload.get("recall_at_k")
    if not isinstance(raw_recall, dict):
        msg = "results JSON must carry a 'recall_at_k' object."
        raise ValueError(msg)
    recall_at_k = {c: _as_float(raw_recall.get(c)) for c in configs}

    raw_by_type = payload.get("by_type")
    by_type: dict[str, dict[str, float]] = {}
    if isinstance(raw_by_type, dict):
        for qtype, row in raw_by_type.items():
            if isinstance(row, dict):
                by_type[str(qtype)] = {c: _as_float(row.get(c)) for c in configs}

    raw_pq = payload.get("per_question")
    per_question: list[PerQuestion] = []
    if isinstance(raw_pq, list):
        for row in raw_pq:
            if not isinstance(row, dict):
                continue
            qid = row.get("id")
            qtype = row.get("type")
            if not isinstance(qid, str) or not qid:
                continue
            recall_row = row.get("recall")
            recall = (
                {c: _as_float(recall_row.get(c)) for c in configs}
                if isinstance(recall_row, dict)
                else {c: 0.0 for c in configs}
            )
            per_question.append(
                PerQuestion(
                    qid=qid,
                    qtype=str(qtype) if isinstance(qtype, str) and qtype else "unknown",
                    recall=recall,
                    fused_only_win=bool(row.get("fused_only_win")),
                )
            )
    return Results(
        configs=configs,
        top_k=top_k,
        recall_at_k=recall_at_k,
        by_type=by_type,
        per_question=per_question,
        meta=meta,
    )


def _coerce_chain(raw: object) -> tuple[Hop, ...]:
    """Coerce a question's ``kg_chain`` cell to typed :class:`Hop`\\ s (skipping malformed hops)."""
    if not isinstance(raw, list):
        return ()
    hops: list[Hop] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        src, rel, dst = item.get("src"), item.get("rel"), item.get("dst")
        if (
            isinstance(src, int)
            and not isinstance(src, bool)
            and isinstance(dst, int)
            and not isinstance(dst, bool)
            and isinstance(rel, str)
            and rel
        ):
            hops.append(Hop(src=src, rel=rel, dst=dst))
    return tuple(hops)


def _coerce_str_ids(raw: object) -> tuple[str, ...]:
    """A JSON array of ids as a tuple of non-empty strings (else ``()``)."""
    if not isinstance(raw, list):
        return ()
    return tuple(str(x) for x in raw if isinstance(x, str) and x)


def load_questions(path: Path | None) -> QuestionsDoc:
    """Load the §5 benchmark JSON's per-question detail (for worked examples); tolerant of absence.

    The worked examples need the question **text** + **KG chain**, which the results JSON does not
    carry — they live in the §5 benchmark JSON (``rfc-benchmark.json``), pointed to by the results'
    ``meta.questions``. When that file is absent the report still builds (the worked examples then
    fall back to the results' per-question recall alone, with a visible note), so a moved artifact
    degrades gracefully rather than breaking the report.
    """
    if path is None or not path.is_file():
        return QuestionsDoc(meta={})
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return QuestionsDoc(meta={})
    raw_meta = payload.get("meta")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    detail: dict[str, QuestionDetail] = {}
    raw_questions = payload.get("questions")
    if isinstance(raw_questions, list):
        for row in raw_questions:
            if not isinstance(row, dict):
                continue
            qid = row.get("id")
            if not isinstance(qid, str) or not qid:
                continue
            question = row.get("question")
            detail[qid] = QuestionDetail(
                question=question if isinstance(question, str) else "",
                subject=str(row.get("subject")) if isinstance(row.get("subject"), str) else "",
                gold_chunk_ids=_coerce_str_ids(row.get("gold_chunk_ids")),
                kg_chain=_coerce_chain(row.get("kg_chain")),
                fused_only_gold_ids=_coerce_str_ids(row.get("fused_only_gold_ids")),
            )
    return QuestionsDoc(meta=meta, detail=detail)


# ---------------------------------------------------------------------------
# Analysis — the numbers the narrative is built from.
# ---------------------------------------------------------------------------


def best_single_recall(
    recall_at_k: dict[str, float], configs: tuple[str, ...]
) -> tuple[str, float]:
    """The strongest single signal and its recall (``("", 0.0)`` when none is present)."""
    singles = [(c, recall_at_k.get(c, 0.0)) for c in configs if c in SINGLE_CONFIGS]
    if not singles:
        return ("", 0.0)
    return max(singles, key=lambda kv: kv[1])


def fused_beats_every_single(results: Results) -> bool:
    """True only when fused's mean recall strictly exceeds **every** single signal (§6 bar)."""
    if FUSED_CONFIG not in results.configs:
        return False
    fused = results.recall_at_k.get(FUSED_CONFIG, 0.0)
    singles = [results.recall_at_k.get(c, 0.0) for c in results.configs if c in SINGLE_CONFIGS]
    return bool(singles) and all(fused > s for s in singles)


def select_worked_examples(results: Results, limit: int) -> list[PerQuestion]:
    """Pick up to ``limit`` questions that best show single-signal miss vs fused recovery (§7).

    The proof event is a **fused-only win** (fused finds gold in top-k that no single signal does),
    so those are taken first, highest fused-recall first, **round-robin across question types** so
    the examples span multi-hop / lexical / semantic / compound rather than piling onto one type.
    If fewer than :data:`MIN_EXAMPLES` fused-only wins exist, the set is topped up with the
    questions where fused most **lifts** over the best single signal (still an honest "fusion helps"
    example), then by fused recall — never with filler that shows nothing.
    """
    if not results.per_question:
        return []
    ranked = _round_robin_by_type(
        [q for q in results.per_question if q.fused_only_win],
        key=lambda q: q.recall.get(FUSED_CONFIG, 0.0),
    )
    if len(ranked) < min(limit, MIN_EXAMPLES):
        chosen_ids = {q.qid for q in ranked}
        lift_candidates = sorted(
            (q for q in results.per_question if q.qid not in chosen_ids),
            key=lambda q: (_fused_lift(q), q.recall.get(FUSED_CONFIG, 0.0)),
            reverse=True,
        )
        ranked.extend(lift_candidates)
    return ranked[:limit]


def _fused_lift(q: PerQuestion) -> float:
    """A question's fused recall minus its best single-signal recall (its fusion lift)."""
    fused = q.recall.get(FUSED_CONFIG, 0.0)
    singles = [q.recall.get(c, 0.0) for c in SINGLE_CONFIGS if c in q.recall]
    return fused - (max(singles) if singles else 0.0)


def _round_robin_by_type(
    questions: list[PerQuestion], key: Any
) -> list[PerQuestion]:
    """Interleave questions across their types, each type's own list sorted by ``key`` desc."""
    buckets: dict[str, list[PerQuestion]] = {}
    for q in questions:
        buckets.setdefault(q.qtype, []).append(q)
    for qtype in buckets:
        buckets[qtype].sort(key=key, reverse=True)
    out: list[PerQuestion] = []
    while any(buckets.values()):
        for qtype in sorted(buckets):
            if buckets[qtype]:
                out.append(buckets[qtype].pop(0))
    return out


def no_answer_questions(results: Results) -> list[str]:
    """Ids of questions where **no requested config** found gold in top-k (the corpus ceiling)."""
    out: list[str] = []
    for q in results.per_question:
        if all(q.recall.get(c, 0.0) <= 0.0 for c in results.configs):
            out.append(q.qid)
    return out


# ---------------------------------------------------------------------------
# Dependency-free SVG — the one chart this script renders (ADR-0016).
# ---------------------------------------------------------------------------

_BG = "#ffffff"
_INK = "#1f2933"
_MUTE = "#7b8794"
_GRID = "#e4e7eb"
_FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"
_CONFIG_COLOR: dict[str, str] = {
    "dense": "#2f6f9f",
    "bm25": "#b8742f",
    "splade": "#7a4fa0",
    "fused": "#2e7d32",
}


def _esc(text: str) -> str:
    return _xml_escape(text)


def _svg_text(
    x: float, y: float, s: str, *, size: int, fill: str, weight: str, anchor: str
) -> str:
    return (
        f"<text x='{x:.1f}' y='{y:.1f}' {_FONT} font-size='{size}' font-weight='{weight}' "
        f"fill='{fill}' text-anchor='{anchor}'>{_esc(s)}</text>"
    )


def _svg_rect(x: float, y: float, w: float, h: float, fill: str) -> str:
    return (
        f"<rect x='{x:.1f}' y='{y:.1f}' width='{max(w, 0.0):.1f}' "
        f"height='{max(h, 0.0):.1f}' fill='{fill}'/>"
    )


def render_headline_chart(results: Results) -> str:
    """A self-contained horizontal-bar SVG of mean recall@k per config (fused highlighted).

    Rendered directly from the authoritative results numbers (not copied from run_benchmark's
    chart), so the headline visual and the headline table can never disagree. Pure string — no
    external asset, font, or chart library (ADR-0016).
    """
    configs = results.configs
    width = 720
    row_h = 46
    pad_t, pad_b, pad_l, pad_r = 70, 30, 90, 70
    plot_w = width - pad_l - pad_r
    height = pad_t + pad_b + row_h * max(len(configs), 1)

    body: list[str] = [_svg_rect(0, 0, width, height, _BG)]
    body.append(
        _svg_text(
            width / 2, 34, f"Mean recall@{results.top_k} per retrieval config",
            size=18, fill=_INK, weight="700", anchor="middle",
        )
    )
    # x grid 0.0 .. 1.0
    for i in range(6):
        frac = i / 5
        x = pad_l + plot_w * frac
        body.append(
            f"<line x1='{x:.1f}' y1='{pad_t - 6}' x2='{x:.1f}' y2='{pad_t + row_h * len(configs)}' "
            f"stroke='{_GRID}' stroke-width='1'/>"
        )
        body.append(
            _svg_text(
                x, pad_t - 12, f"{frac:.1f}", size=10, fill=_MUTE, weight="400", anchor="middle"
            )
        )
    for i, config in enumerate(configs):
        value = max(0.0, min(1.0, results.recall_at_k.get(config, 0.0)))
        cy = pad_t + i * row_h
        bar_h = row_h - 16
        bar_w = plot_w * value
        color = _CONFIG_COLOR.get(config, "#888")
        weight = "700" if config == FUSED_CONFIG else "400"
        body.append(_svg_text(pad_l - 10, cy + bar_h / 2 + 4, config, size=13, fill=_INK,
                              weight=weight, anchor="end"))
        body.append(_svg_rect(pad_l, cy, bar_w, bar_h, color))
        body.append(_svg_text(pad_l + bar_w + 6, cy + bar_h / 2 + 4, f"{value:.3f}", size=12,
                              fill=_INK, weight=weight, anchor="start"))
    joined = "".join(body)
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' role='img' aria-label='mean recall per config'>"
        f"{joined}</svg>\n"
    )


# ---------------------------------------------------------------------------
# Markdown assembly.
# ---------------------------------------------------------------------------


def _rel(target: Path, start: Path) -> str:
    """A POSIX relative link from the report dir ``start`` to ``target`` (for markdown images)."""
    return Path(os.path.relpath(target, start)).as_posix()


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _chain_str(chain: tuple[Hop, ...]) -> str:
    """A KG chain as ``RFC A —obsoletes→ RFC B —updates→ RFC C`` (empty for no chain)."""
    if not chain:
        return ""
    parts = [f"RFC {chain[0].src}"]
    for hop in chain:
        parts.append(f" —{hop.rel}→ RFC {hop.dst}")
    return "".join(parts)


def _headline_section(results: Results) -> list[str]:
    k = results.top_k
    lines: list[str] = [f"## Headline — recall@{k} by retrieval config", ""]
    lines.append(f"| Retrieval config | recall@{k} |")
    lines.append("| --- | --- |")
    for config in results.configs:
        value = _fmt(results.recall_at_k.get(config, 0.0))
        if config == FUSED_CONFIG:
            lines.append(f"| **{config}** | **{value}** |")
        else:
            lines.append(f"| {config} | {value} |")
    lines.append("")
    if FUSED_CONFIG in results.configs:
        best_name, best_val = best_single_recall(results.recall_at_k, results.configs)
        fused_val = results.recall_at_k.get(FUSED_CONFIG, 0.0)
        if best_name:
            delta = fused_val - best_val
            sign = "+" if delta >= 0 else ""
            verb = "exceeds" if delta > 0 else ("ties" if delta == 0 else "trails")
            lines.append(
                f"Fused {verb} the best single signal (`{best_name}` at {_fmt(best_val)}) by "
                f"**{sign}{_fmt(delta)}** recall@{results.top_k}."
            )
            lines.append("")
    return lines


def _finding_banner(results: Results) -> list[str]:
    """The top-of-report verdict — the §6/§7 honest headline, win or not-win."""
    wins = results.meta.get("fused_only_wins")
    n = results.meta.get("n_questions", len(results.per_question))
    k = results.top_k
    if FUSED_CONFIG not in results.configs:
        return [
            "> **Scope.** The `fused` configuration was not part of this run, so this report "
            "compares only the requested single signals — it does not evaluate the fusion claim.",
            "",
        ]
    if fused_beats_every_single(results):
        win_note = ""
        if isinstance(wins, int) and wins > 0:
            win_note = (
                f" On **{wins}** of **{n}** questions, fused surfaced gold in the top-"
                f"{k} that **no single signal** did."
            )
        return [
            f"> **Finding — the fused stack wins.** Fused mean recall@{k} exceeds every single "
            f"retriever.{win_note} The defensible claim holds: *fusion unlocks a class of queries "
            "single retrievers structurally cannot answer.*",
            "",
        ]
    return [
        "> **Finding — fused did NOT beat every single signal.** This is the most important "
        "result of the run and is stated up front, not buried: on this corpus the fusion stack "
        f"did not clear the bar of exceeding every single retriever's mean recall@{k}. See "
        "**Limitations** for the corpus ceiling and next steps.",
        "",
    ]


def _worked_examples_section(
    results: Results, questions: QuestionsDoc, limit: int
) -> list[str]:
    k = results.top_k
    examples = select_worked_examples(results, limit)
    lines: list[str] = ["## Worked examples — where single signals miss and fusion recovers", ""]
    if not examples:
        lines.append("_No per-question detail was available to build worked examples._")
        lines.append("")
        return lines
    lines.append(
        f"Each example shows one question's measured recall@{k} per signal. A **fused-only win** "
        f"is a question where fusion found a gold chunk in the top-{k} that no single retriever "
        "did — the structural gap the fused stack closes."
    )
    lines.append("")
    for idx, q in enumerate(examples, start=1):
        detail = questions.detail.get(q.qid)
        tag = " · **fused-only win**" if q.fused_only_win else ""
        lines.append(f"### Example {idx} — {q.qtype} (`{q.qid}`){tag}")
        lines.append("")
        if detail is not None and detail.question:
            lines.append(f"**Question.** {detail.question}")
            lines.append("")
        else:
            lines.append(
                "_Question text unavailable for this id (not in the §5 benchmark JSON); showing "
                "measured recall only._"
            )
            lines.append("")
        if detail is not None and detail.kg_chain:
            lines.append(f"**KG chain.** {_chain_str(detail.kg_chain)}")
            lines.append("")
        lines.append(f"| signal | recall@{k} |")
        lines.append("| --- | --- |")
        for config in results.configs:
            value = _fmt(q.recall.get(config, 0.0))
            if config == FUSED_CONFIG:
                lines.append(f"| **{config}** | **{value}** |")
            else:
                lines.append(f"| {config} | {value} |")
        lines.append("")
        lines.append(_example_narrative(q, detail, k))
        lines.append("")
    return lines


def _example_narrative(q: PerQuestion, detail: QuestionDetail | None, top_k: int) -> str:
    """One honest sentence explaining why single signals miss and fusion recovers this question."""
    if q.fused_only_win:
        gold = ""
        if detail is not None and detail.fused_only_gold_ids:
            ids = ", ".join(f"`{g}`" for g in detail.fused_only_gold_ids)
            gold = f" (gold chunk(s) {ids})"
        why = ""
        if q.qtype.startswith("multi-hop") and detail is not None and detail.kg_chain:
            why = " — the KG chain bridged documents no lexical or dense query reached alone"
        elif q.qtype.startswith("lexical"):
            why = " — the exact term match carried it where dense paraphrase-search missed"
        elif q.qtype.startswith("semantic"):
            why = " — dense semantics carried it where the lexical signals had no overlap"
        return (
            f"No single retriever surfaced gold in the top-{top_k}; **fusion recovered it**"
            f"{gold}{why}."
        )
    lift = _fused_lift(q)
    if lift > 0:
        return (
            f"Fusion improves recall@{top_k} by **+{_fmt(lift)}** over the best single signal on "
            "this question — the signals are complementary, and fusing them compounds their reach."
        )
    return (
        f"On this question fusion matched but did not beat the best single signal (recall@{top_k} "
        f"{_fmt(q.recall.get(FUSED_CONFIG, 0.0))}) — an honest example that fusion does not help "
        "everywhere."
    )


def _per_type_section(results: Results) -> list[str]:
    if not results.by_type:
        return []
    lines: list[str] = ["## Per-type breakdown", ""]
    header = "| question type | " + " | ".join(results.configs) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---" for _ in results.configs) + " |")
    for qtype in sorted(results.by_type):
        row = results.by_type[qtype]
        cells = " | ".join(_fmt(row.get(c, 0.0)) for c in results.configs)
        lines.append(f"| {qtype} | {cells} |")
    lines.append("")
    return lines


def _charts_section(charts_dir: Path, out_dir: Path, headline_svg: Path) -> list[str]:
    lines: list[str] = ["## Charts", "", "### Headline — recall by config", ""]
    lines.append(f"![Mean recall per config]({_rel(headline_svg, out_dir)})")
    lines.append("")
    lines.append("### Correctness charts (pipeline stages, §3.3)")
    lines.append("")
    any_chart = False
    for filename, caption in STAGE_CHART_FILES:
        chart_path = charts_dir / filename
        if chart_path.is_file():
            any_chart = True
            lines.append(f"**{caption}**")
            lines.append("")
            lines.append(f"![{caption}]({_rel(chart_path, out_dir)})")
            lines.append("")
        else:
            lines.append(
                f"- _Missing chart: `{filename}` ({caption}) — not found in `{charts_dir}`._"
            )
            lines.append("")
    if not any_chart:
        lines.append(
            f"_No §3.3 correctness charts were found under `{charts_dir}`; run "
            "`benchmark/rfc/stage_charts.py` first._"
        )
        lines.append("")
    return lines


def _limitations_section(results: Results, questions: QuestionsDoc) -> list[str]:
    lines: list[str] = ["## Limitations — read before quoting any number", ""]

    # Corpus ceiling: generated-vs-survived from the §5 benchmark meta (honest funnel), else n.
    generated = questions.meta.get("generated")
    survived = questions.meta.get("survived")
    n = results.meta.get("n_questions", len(results.per_question))
    if isinstance(generated, int) and isinstance(survived, int) and generated:
        lines.append(
            f"- **Corpus ceiling.** Of **{generated}** mined candidate questions, only "
            f"**{survived}** survived the hard fused-vs-single gate and are measured here. A small "
            "surviving set is the honest signal of this corpus's multi-hop depth — not padded."
        )
    else:
        lines.append(
            f"- **Corpus ceiling.** **{n}** gated questions are measured here. The set is only as "
            "deep as the corpus's real multi-hop structure allows."
        )

    # Questions no configuration answered.
    unanswered = no_answer_questions(results)
    if unanswered:
        preview = ", ".join(f"`{q}`" for q in unanswered[:8])
        more = "" if len(unanswered) <= 8 else f" (+{len(unanswered) - 8} more)"
        lines.append(
            f"- **Refuted / unanswered questions.** **{len(unanswered)}** question(s) were "
            f"answered by *no* configuration (not even fused) in the top-{results.top_k}: "
            f"{preview}{more}. These are surfaced, not hidden — they mark the retrieval ceiling."
        )
    else:
        lines.append(
            "- **Refuted / unanswered questions.** Every measured question was answered by at "
            f"least one configuration in the top-{results.top_k}."
        )

    # The fusion verdict, restated honestly.
    if FUSED_CONFIG in results.configs and not fused_beats_every_single(results):
        lines.append(
            "- **Fusion verdict.** On this corpus fused did **not** exceed every single signal's "
            "mean recall — the headline claim is not supported by this run and must not be quoted "
            "as if it were."
        )

    # The SPLADE-v3 non-commercial caveat (Appendix A) — always stated when splade was in the run.
    if "splade" in results.configs:
        lines.append(f"- **SPLADE-v3 license.** {SPLADE_LICENSE_CAVEAT}")
    lines.append("")
    return lines


def build_report(
    results: Results,
    questions: QuestionsDoc,
    *,
    charts_dir: Path,
    out: Path,
    headline_svg: Path,
    max_examples: int,
) -> str:
    """Assemble the whole ``PROOF.md`` markdown string from the loaded artifacts (tutorial §7)."""
    out_dir = out.parent
    lines: list[str] = []
    lines.append("# RFC fused-retrieval proof")
    lines.append("")
    lines.append(
        "*The claim under test:* **the fused retrieval stack (dense + BM25 + SPLADE + KG "
        "multi-hop) unlocks a class of queries that single retrievers structurally cannot "
        f"answer.** Every number below is a real recall@{results.top_k} measured against the "
        "run's real indexes (tutorial §6) — none is fabricated."
    )
    lines.append("")

    # Provenance.
    run = results.meta.get("run")
    collection = results.meta.get("collection")
    qdrant = results.meta.get("qdrant")
    prov: list[str] = []
    if isinstance(run, str) and run:
        prov.append(f"run `{run}`")
    if isinstance(collection, str) and collection:
        prov.append(f"collection `{collection}`")
    if isinstance(qdrant, str) and qdrant:
        prov.append(f"Qdrant `{qdrant}`")
    prov.append(f"{results.meta.get('n_questions', len(results.per_question))} gated questions")
    prov.append("configs " + ", ".join(f"`{c}`" for c in results.configs))
    lines.append("_" + " · ".join(prov) + "_")
    lines.append("")

    lines.extend(_finding_banner(results))
    lines.extend(_headline_section(results))
    lines.extend(_worked_examples_section(results, questions, max_examples))
    lines.extend(_per_type_section(results))
    lines.extend(_charts_section(charts_dir, out_dir, headline_svg))
    lines.extend(_limitations_section(results, questions))

    lines.append("---")
    lines.append("")
    lines.append(
        "_Generated by `benchmark/rfc/make_report.py` from the measured results and the §3.3 "
        "correctness charts. No number here is a model claim — each is a real measurement._"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _resolve_questions_path(results: Results, override: Path | None) -> Path | None:
    """The §5 benchmark JSON to join for worked-example text: override wins, else meta.questions."""
    if override is not None:
        return override
    questions = results.meta.get("questions")
    if isinstance(questions, str) and questions:
        return Path(questions)
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI entry (tutorial §7): read the measured results + §3.3 charts, write ``PROOF.md``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results", required=True, type=Path,
        help="the §6 measured results JSON (rfc-results.json)",
    )
    ap.add_argument(
        "--charts", required=True, type=Path,
        help="the §3.3 stage-charts directory (holds the correctness SVGs)",
    )
    ap.add_argument("--out", required=True, type=Path, help="output report path (PROOF.md)")
    ap.add_argument(
        "--questions", type=Path, default=None,
        help="the §5 benchmark JSON for worked-example text (default: results meta.questions)",
    )
    ap.add_argument(
        "--max-examples", dest="max_examples", type=int, default=DEFAULT_MAX_EXAMPLES,
        help="max worked examples to include (tutorial §7 asks for 3–5)",
    )
    args = ap.parse_args(argv)

    if not args.results.is_file():
        print(f"error: results JSON not found: {args.results}", file=sys.stderr)
        return 2

    try:
        results = load_results(args.results)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"error: could not read results JSON {args.results}: {exc}", file=sys.stderr)
        return 2

    questions_path = _resolve_questions_path(results, args.questions)
    questions = load_questions(questions_path)

    # The one chart this script owns is written next to the report so its image link resolves.
    headline_svg = args.out.parent / "recall-headline.svg"

    report = build_report(
        results,
        questions,
        charts_dir=args.charts,
        out=args.out,
        headline_svg=headline_svg,
        max_examples=max(args.max_examples, 1),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    headline_svg.write_text(render_headline_chart(results), encoding="utf-8")

    n_examples = len(select_worked_examples(results, max(args.max_examples, 1)))
    print(f"make_report: read {args.results}")
    if questions_path is not None and questions.detail:
        print(f"  joined worked-example detail from {questions_path}")
    elif questions_path is not None:
        print(f"  note: benchmark JSON {questions_path} not found — examples use recall only")
    print(f"  {len(results.per_question)} questions, {n_examples} worked examples")
    if FUSED_CONFIG in results.configs:
        verdict = "fused beats every single signal" if fused_beats_every_single(results) else (
            "fused did NOT beat every single signal (see Limitations)"
        )
        print(f"  verdict: {verdict}")
    print(f"  wrote {args.out} + {headline_svg}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
