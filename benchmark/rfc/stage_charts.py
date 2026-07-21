#!/usr/bin/env python3
"""Correctness charts for a finished pipeline run — dependency-free SVGs (tutorial §3.3).

Reads a run directory (``_latence/runs/<run_id>/``) and emits one **self-contained SVG per
correctness claim** the RFC-retrieval-proof tutorial makes at §3.3:

* ``chunk-size-distribution.svg`` — the chunk token-count histogram, with the chunk budget
  (default 640) drawn as a reference line, so an operator sees the mass sits at/under budget.
* ``entities-relations-funnel.svg`` — the extract → disambiguate collapse: raw entity **mentions**
  vs the canonical **graph nodes** they fold into, and raw relation **mentions** vs the emitted
  **graph edges** (duplicate entities collapse; the ratio is the legibility signal).
* ``kg-degree-and-centrality.svg`` — the knowledge-graph degree distribution plus the top nodes by
  PageRank centrality (a pure-Python PageRank over the emitted graph — no graph library).
* ``induced-vs-independent-overlap.svg`` — the §3.1 correctness bar: how the **induced**
  ``obsoletes``/``updates`` edges (extracted by the pipeline) overlap the corpus's **independent**
  ``_refs.jsonl`` cross-reference graph (the human-authored gold signal, never fed to the pipeline).

Everything is read with the **standard library only** and every chart is a hand-written SVG string
(ADR-0016: no heavy chart library in the pipeline; no PNG, no external CDN asset). Missing inputs
degrade gracefully — the chart is still written, carrying a visible "no data" note, so a downstream
report build never breaks on a partial run.

Usage (the exact invocation from docs/TUTORIAL-rfc-retrieval-proof.md §3.3):
  python benchmark/rfc/stage_charts.py \
      --run _latence/runs/<run_id> --out /workspace/report/stage-charts

Optional:
  --refs PATH     the corpus's independent ``_refs.jsonl`` (else auto-discovered from the run's
                  Provenance ``source_uri``, or from ``<run>/_refs.jsonl``).
  --budget N      the chunk token budget drawn as the reference line (default 640).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree
from xml.sax.saxutils import escape as _xml_escape

# ---------------------------------------------------------------------------
# Data model — the small, typed shapes the charts consume.
# ---------------------------------------------------------------------------

# An RFC surface inside a canonical entity name, e.g. "RFC 9110" / "RFC9110" / "rfc-7230".
_RFC_NUM = re.compile(r"\bRFC[\s-]?(\d{3,5})\b", re.IGNORECASE)
# The typed cross-reference relations the corpus authors explicitly (see build_corpus.py). These are
# the ones the §3.1 correctness bar compares induced-vs-independent on.
_TYPED_REL = ("obsoletes", "updates")


class Graph:
    """The emitted knowledge graph, read from ``export/graph.graphml`` (stdlib XML)."""

    def __init__(
        self,
        names: dict[str, str],
        types: dict[str, str],
        edges: list[tuple[str, str, str]],
    ) -> None:
        self.names = names  # node_id -> canonical_name
        self.types = types  # node_id -> entity type
        self.edges = edges  # (source_node_id, target_node_id, label)

    @property
    def node_ids(self) -> list[str]:
        return list(self.names)


# ---------------------------------------------------------------------------
# Readers — standard library only.
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Every non-blank line of a JSONL file as a dict (``[]`` when the file is absent)."""
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_chunk_tokens(run: Path) -> list[int]:
    """The ``token_count`` of every exported chunk (``export/records.jsonl``).

    The Export dumps each record's full contract (ADR-0006), so a :class:`ChunkRecord` row carries
    its ``token_count``; non-chunk rows (a document) have none and are skipped.
    """
    tokens: list[int] = []
    for row in _read_jsonl(run / "export" / "records.jsonl"):
        value = row.get("token_count")
        if isinstance(value, bool):  # bool is an int subclass — never a token count
            continue
        if isinstance(value, int) and value >= 0:
            tokens.append(value)
    return tokens


def load_mention_counts(run: Path) -> tuple[int, int]:
    """Distinct raw (entity, relation) mention counts across the run's stage checkpoints.

    Every Stage checkpoints its output as ``checkpoints/<stage>.jsonl``; entity/relation mentions
    are re-checkpointed by more than one Stage (extract, then redact/graph consume them), so records
    are de-duplicated by ``record_id`` to count each raw mention exactly once. A record is an entity
    mention when it carries a ``chunk_record_id`` + surface ``text`` + ``label`` (and is not a
    relation), and a relation mention when it carries ``head_mention_id``/``tail_mention_id``.
    """
    entity_ids: set[str] = set()
    relation_ids: set[str] = set()
    ckpt_dir = run / "checkpoints"
    if not ckpt_dir.is_dir():
        return (0, 0)
    for path in sorted(ckpt_dir.glob("*.jsonl")):
        for row in _read_jsonl(path):
            rid = row.get("record_id")
            if not isinstance(rid, str):
                continue
            if "head_mention_id" in row and "tail_mention_id" in row:
                relation_ids.add(rid)
            elif "chunk_record_id" in row and "text" in row and "label" in row:
                entity_ids.add(rid)
    return (len(entity_ids), len(relation_ids))


def load_graph(run: Path) -> Graph:
    """Read the emitted KG from ``export/graph.graphml`` (stdlib ``xml.etree``).

    GraphML is one of the three deterministic KG representations the graph Export writes; it is
    plain XML, so the graph loads with no parquet reader and no graph library. An absent/empty file
    yields an empty :class:`Graph` (the dependent charts then render their "no data" note).
    """
    path = run / "export" / "graph.graphml"
    names: dict[str, str] = {}
    types: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []
    if not path.is_file():
        return Graph(names, types, edges)
    ns = "{http://graphml.graphdrawing.org/xmlns}"
    root = ElementTree.parse(path).getroot()
    graph = root.find(f"{ns}graph")
    if graph is None:
        return Graph(names, types, edges)
    for node in graph.findall(f"{ns}node"):
        node_id = node.get("id")
        if node_id is None:
            continue
        label = ""
        etype = ""
        for data in node.findall(f"{ns}data"):
            if data.get("key") == "label":
                label = data.text or ""
            elif data.get("key") == "type":
                etype = data.text or ""
        names[node_id] = label or node_id
        types[node_id] = etype
    for edge in graph.findall(f"{ns}edge"):
        src = edge.get("source")
        tgt = edge.get("target")
        if src is None or tgt is None:
            continue
        rel = ""
        for data in edge.findall(f"{ns}data"):
            if data.get("key") == "rel":
                rel = data.text or ""
        edges.append((src, tgt, rel))
    return Graph(names, types, edges)


def discover_refs(run: Path, explicit: Path | None) -> Path | None:
    """Locate the corpus's independent ``_refs.jsonl`` (never fed into the pipeline).

    Preference order: an explicit ``--refs`` path; a copy sitting in the run dir; else the corpus
    directory recovered from the first exported record's Provenance ``source_uri`` (the pipeline's
    own record of where it read the corpus from). ``None`` when none is found — the overlap chart
    then renders its "independent graph not found" note rather than failing.
    """
    if explicit is not None:
        return explicit if explicit.is_file() else None
    in_run = run / "_refs.jsonl"
    if in_run.is_file():
        return in_run
    for row in _read_jsonl(run / "export" / "records.jsonl"):
        prov = row.get("provenance")
        if not isinstance(prov, dict):
            continue
        source_uri = prov.get("source_uri")
        if not isinstance(source_uri, str) or not source_uri:
            continue
        # source_uri may be a bare path or a file:// URI; strip a scheme and read its directory.
        local = source_uri.split("://", 1)[-1]
        candidate = Path(local).parent / "_refs.jsonl"
        if candidate.is_file():
            return candidate
    return None


def load_independent_edges(refs: Path | None) -> set[tuple[int, str, int]]:
    """The independent typed edges ``(src_rfc, relation, dst_rfc)`` from ``_refs.jsonl``.

    Only the explicitly-authored ``obsoletes``/``updates`` relations are taken — the ones §3.1
    expects the extractor to reproduce.
    """
    out: set[tuple[int, str, int]] = set()
    if refs is None:
        return out
    for row in _read_jsonl(refs):
        rfc = row.get("rfc")
        if not isinstance(rfc, int):
            continue
        for rel in _TYPED_REL:
            targets = row.get(rel)
            if isinstance(targets, list):
                for target in targets:
                    if isinstance(target, int):
                        out.add((rfc, rel, target))
    return out


def induced_typed_edges(graph: Graph) -> set[tuple[int, str, int]]:
    """The induced typed edges ``(src_rfc, relation, dst_rfc)`` recovered from the KG.

    A graph edge counts only when its label is a typed cross-reference relation and BOTH endpoints'
    canonical names carry a parseable RFC number — so the comparison is like-for-like with the
    independent ``_refs.jsonl`` graph.
    """
    out: set[tuple[int, str, int]] = set()
    for src, tgt, label in graph.edges:
        rel = label.strip().lower()
        if rel not in _TYPED_REL:
            continue
        src_rfc = _rfc_number(graph.names.get(src, ""))
        dst_rfc = _rfc_number(graph.names.get(tgt, ""))
        if src_rfc is not None and dst_rfc is not None:
            out.add((src_rfc, rel, dst_rfc))
    return out


def _rfc_number(name: str) -> int | None:
    match = _RFC_NUM.search(name)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Graph metrics — pure Python (no networkx).
# ---------------------------------------------------------------------------


def degree_distribution(graph: Graph) -> Counter[int]:
    """Undirected degree (incident edge count) of every node, tallied by degree value."""
    degree: Counter[str] = Counter()
    for src, tgt, _ in graph.edges:
        if src in graph.names:
            degree[src] += 1
        if tgt in graph.names:
            degree[tgt] += 1
    dist: Counter[int] = Counter()
    for node_id in graph.names:
        dist[degree[node_id]] += 1
    return dist


def pagerank(graph: Graph, damping: float = 0.85, iterations: int = 60) -> dict[str, float]:
    """PageRank over the directed KG edges (pure Python; standard power iteration).

    Handles dangling nodes (no out-edges) by redistributing their mass uniformly, so the scores sum
    to 1. An empty graph returns ``{}``.
    """
    node_ids = graph.node_ids
    n = len(node_ids)
    if n == 0:
        return {}
    out_links: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    node_set = set(node_ids)
    for src, tgt, _ in graph.edges:
        if src in out_links and tgt in node_set:
            out_links[src].append(tgt)
    dangling = [node_id for node_id in node_ids if not out_links[node_id]]
    rank = {node_id: 1.0 / n for node_id in node_ids}
    for _ in range(iterations):
        dangling_mass = damping * sum(rank[node_id] for node_id in dangling) / n
        base = (1.0 - damping) / n + dangling_mass
        nxt = {node_id: base for node_id in node_ids}
        for src in node_ids:
            targets = out_links[src]
            if targets:
                share = damping * rank[src] / len(targets)
                for tgt in targets:
                    nxt[tgt] += share
        rank = nxt
    return rank


# ---------------------------------------------------------------------------
# SVG primitives — hand-written, self-contained, theme-neutral.
# ---------------------------------------------------------------------------

_BG = "#ffffff"
_INK = "#1f2933"
_MUTE = "#7b8794"
_GRID = "#e4e7eb"
_BAR = "#2f6f9f"
_BAR2 = "#7bb0d6"
_ACCENT = "#c0392b"
_FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"


def _svg_doc(width: int, height: int, title: str, body: list[str]) -> str:
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' role='img' aria-label='{_esc(title)}'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='{_BG}'/>",
        _text(20, 32, title, size=18, weight="700"),
        *body,
        "</svg>\n",
    ]
    return "\n".join(parts)


def _esc(text: str) -> str:
    return _xml_escape(text)


def _text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 12,
    fill: str = _INK,
    weight: str = "400",
    anchor: str = "start",
) -> str:
    return (
        f"<text x='{x:.1f}' y='{y:.1f}' {_FONT} font-size='{size}' font-weight='{weight}' "
        f"fill='{fill}' text-anchor='{anchor}'>{_esc(text)}</text>"
    )


def _rect(x: float, y: float, w: float, h: float, fill: str) -> str:
    return (
        f"<rect x='{x:.1f}' y='{y:.1f}' width='{max(w, 0):.1f}' "
        f"height='{max(h, 0):.1f}' fill='{fill}'/>"
    )


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str, dash: str = "") -> str:
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return (
        f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
        f"stroke='{stroke}' stroke-width='1'{d}/>"
    )


def _no_data(title: str, note: str) -> str:
    body = [
        _rect(20, 60, 760, 120, "#f7f9fb"),
        _text(400, 128, note, size=14, fill=_MUTE, anchor="middle"),
    ]
    return _svg_doc(800, 220, title, body)


# ---------------------------------------------------------------------------
# Charts.
# ---------------------------------------------------------------------------


def chart_chunk_sizes(tokens: list[int], budget: int) -> str:
    title = "Chunk-size distribution (tokens)"
    if not tokens:
        return _no_data(title, "no chunks with token_count in export/records.jsonl")
    top = max(max(tokens), budget)
    upper = max(int(top * 1.1) + 1, budget + 1)
    bins = 16
    width = upper / bins
    counts = [0] * bins
    for value in tokens:
        idx = min(int(value / width), bins - 1)
        counts[idx] += 1
    at_or_under = sum(1 for value in tokens if value <= budget)
    pct = 100.0 * at_or_under / len(tokens)

    plot_x, plot_y, plot_w, plot_h = 60.0, 60.0, 700.0, 300.0
    max_count = max(counts) or 1
    body: list[str] = [
        _line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, _INK),
        _line(plot_x, plot_y, plot_x, plot_y + plot_h, _INK),
    ]
    slot = plot_w / bins
    for i, count in enumerate(counts):
        bh = plot_h * count / max_count
        bx = plot_x + i * slot + 2
        by = plot_y + plot_h - bh
        body.append(_rect(bx, by, slot - 4, bh, _BAR))
        if count:
            body.append(
                _text(bx + (slot - 4) / 2, by - 4, str(count), size=10, fill=_MUTE, anchor="middle")
            )
    # x axis ticks (bin edges)
    for i in range(0, bins + 1, 4):
        tx = plot_x + i * slot
        edge = int(i * width)
        body.append(_line(tx, plot_y + plot_h, tx, plot_y + plot_h + 4, _INK))
        body.append(
            _text(tx, plot_y + plot_h + 18, str(edge), size=10, fill=_MUTE, anchor="middle")
        )
    # budget reference line
    budget_x = plot_x + plot_w * min(budget / upper, 1.0)
    body.append(_line(budget_x, plot_y, budget_x, plot_y + plot_h, _ACCENT, dash="5,4"))
    body.append(_text(budget_x + 4, plot_y + 14, f"budget {budget}", size=11, fill=_ACCENT))
    body.append(_text(plot_x, plot_y + plot_h + 40, "token_count", size=11, fill=_MUTE))
    body.append(
        _text(
            plot_x,
            plot_y - 22,
            f"{len(tokens)} chunks — {pct:.1f}% at/under {budget} tokens",
            size=13,
            weight="600",
        )
    )
    return _svg_doc(800, 420, title, body)


def chart_funnel(entities: int, canonical: int, relations: int, edges: int) -> str:
    title = "Entities / relations funnel (extract → disambiguate)"
    if entities == 0 and relations == 0 and canonical == 0 and edges == 0:
        return _no_data(title, "no mentions or graph elements found")

    groups = [
        ("Entities", [("mentions", entities, _BAR), ("canonical nodes", canonical, _BAR2)]),
        ("Relations", [("mentions", relations, _BAR), ("graph edges", edges, _BAR2)]),
    ]
    overall_max = max([entities, canonical, relations, edges, 1])
    plot_x, plot_y, plot_h = 70.0, 80.0, 240.0
    col_w = 300.0
    bar_w = 90.0
    body: list[str] = [_line(plot_x, plot_y + plot_h, plot_x + 2 * col_w, plot_y + plot_h, _INK)]
    for gi, (gname, bars) in enumerate(groups):
        gx = plot_x + gi * col_w
        body.append(_text(gx + col_w / 2 - 30, plot_y - 30, gname, size=14, weight="700"))
        for bi, (label, value, color) in enumerate(bars):
            bh = plot_h * value / overall_max
            bx = gx + 40 + bi * (bar_w + 30)
            by = plot_y + plot_h - bh
            body.append(_rect(bx, by, bar_w, bh, color))
            body.append(
                _text(bx + bar_w / 2, by - 6, str(value), size=12, weight="600", anchor="middle")
            )
            body.append(
                _text(
                    bx + bar_w / 2,
                    plot_y + plot_h + 18,
                    label,
                    size=11,
                    fill=_MUTE,
                    anchor="middle",
                )
            )
        # collapse ratio annotation
        raw, canon = bars[0][1], bars[1][1]
        if canon:
            ratio = raw / canon
            body.append(
                _text(
                    gx + col_w / 2 - 30,
                    plot_y + plot_h + 42,
                    f"collapse {ratio:.2f}× ({raw}→{canon})",
                    size=11,
                    fill=_ACCENT,
                    anchor="middle",
                )
            )
    return _svg_doc(800, 420, title, body)


def chart_degree_centrality(graph: Graph) -> str:
    title = "KG degree distribution + top PageRank centrality"
    if not graph.names:
        return _no_data(title, "no graph nodes in export/graph.graphml")
    dist = degree_distribution(graph)
    rank = pagerank(graph)
    top = sorted(rank.items(), key=lambda kv: kv[1], reverse=True)[:8]

    body: list[str] = []
    # -- left panel: degree distribution histogram
    max_degree = max(dist) if dist else 0
    degrees = list(range(max_degree + 1))
    counts = [dist.get(d, 0) for d in degrees]
    max_count = max(counts) or 1
    px, py, pw, ph = 60.0, 90.0, 320.0, 260.0
    body.append(_text(60, 66, "Degree distribution", size=13, weight="700"))
    body.append(_line(px, py + ph, px + pw, py + ph, _INK))
    body.append(_line(px, py, px, py + ph, _INK))
    slots = max(len(degrees), 1)
    slot = pw / slots
    for i, count in enumerate(counts):
        bh = ph * count / max_count
        bx = px + i * slot + 1
        by = py + ph - bh
        body.append(_rect(bx, by, max(slot - 2, 1), bh, _BAR))
        if len(degrees) <= 16:
            body.append(
                _text(bx + slot / 2, py + ph + 16, str(i), size=9, fill=_MUTE, anchor="middle")
            )
    body.append(_text(px, py + ph + 34, "degree", size=11, fill=_MUTE))

    # -- right panel: top nodes by PageRank
    rx, ry, rw = 440.0, 90.0, 320.0
    body.append(_text(440, 66, "Top nodes by PageRank", size=13, weight="700"))
    row_h = 28.0
    max_rank = top[0][1] if top else 1.0
    for i, (node_id, score) in enumerate(top):
        y = ry + i * row_h
        bar = rw * (score / max_rank) if max_rank else 0.0
        name = graph.names.get(node_id, node_id)
        if len(name) > 26:
            name = name[:25] + "…"
        body.append(_rect(rx, y, bar, row_h - 8, _BAR2))
        body.append(_text(rx + 4, y + 14, f"{name}  ({score:.3f})", size=11, fill=_INK))
    body.append(
        _text(
            60,
            372,
            f"{len(graph.names)} nodes, {len(graph.edges)} edges",
            size=11,
            fill=_MUTE,
        )
    )
    return _svg_doc(800, 400, title, body)


def chart_edge_overlap(
    induced: set[tuple[int, str, int]],
    independent: set[tuple[int, str, int]],
    refs_found: bool,
) -> str:
    title = "Induced vs independent edge overlap (obsoletes/updates)"
    if not refs_found:
        return _no_data(title, "independent _refs.jsonl not found (pass --refs to enable)")
    if not induced and not independent:
        return _no_data(title, "no typed obsoletes/updates edges on either side")
    overlap = induced & independent
    induced_only = induced - independent
    independent_only = independent - induced
    recall = 100.0 * len(overlap) / len(independent) if independent else 0.0

    bars = [
        ("induced only", len(induced_only), _BAR),
        ("overlap (agree)", len(overlap), "#2e7d32"),
        ("independent only", len(independent_only), _ACCENT),
    ]
    overall_max = max([b[1] for b in bars] + [1])
    px, py, ph = 80.0, 90.0, 240.0
    slot = 200.0
    body: list[str] = [_line(px - 20, py + ph, px - 20 + 3 * slot, py + ph, _INK)]
    for i, (label, value, color) in enumerate(bars):
        bh = ph * value / overall_max
        bx = px + i * slot
        by = py + ph - bh
        body.append(_rect(bx, by, 120, bh, color))
        body.append(_text(bx + 60, by - 8, str(value), size=14, weight="700", anchor="middle"))
        body.append(_text(bx + 60, py + ph + 20, label, size=11, fill=_MUTE, anchor="middle"))
    body.append(
        _text(
            px - 20,
            py - 26,
            f"independent-edge recall {recall:.1f}%  "
            f"({len(overlap)}/{len(independent)} gold edges reproduced)",
            size=13,
            weight="600",
        )
    )
    return _svg_doc(800, 400, title, body)


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


def build_charts(run: Path, out: Path, refs: Path | None, budget: int) -> list[Path]:
    """Read the run dir and write every chart SVG to ``out``; return the written paths."""
    out.mkdir(parents=True, exist_ok=True)

    tokens = load_chunk_tokens(run)
    entities, relations = load_mention_counts(run)
    graph = load_graph(run)
    refs_path = discover_refs(run, refs)
    independent = load_independent_edges(refs_path)
    induced = induced_typed_edges(graph)

    charts: dict[str, str] = {
        "chunk-size-distribution.svg": chart_chunk_sizes(tokens, budget),
        "entities-relations-funnel.svg": chart_funnel(
            entities, len(graph.names), relations, len(graph.edges)
        ),
        "kg-degree-and-centrality.svg": chart_degree_centrality(graph),
        "induced-vs-independent-overlap.svg": chart_edge_overlap(
            induced, independent, refs_path is not None
        ),
    }
    written: list[Path] = []
    for name, svg in charts.items():
        path = out / name
        path.write_text(svg, encoding="utf-8")
        written.append(path)

    print(f"stage-charts: read run {run}")
    print(f"  chunks={len(tokens)} entities={entities} canonical_nodes={len(graph.names)}")
    print(f"  relations={relations} graph_edges={len(graph.edges)}")
    if refs_path is not None:
        overlap = len(induced & independent)
        print(
            f"  independent edges={len(independent)} induced typed edges={len(induced)} "
            f"overlap={overlap}"
        )
    else:
        print("  independent _refs.jsonl: not found (overlap chart is a placeholder)")
    print(f"wrote {len(written)} SVGs to {out}")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path, help="the run dir (_latence/runs/<run_id>)")
    ap.add_argument("--out", required=True, type=Path, help="output directory for the SVGs")
    ap.add_argument("--refs", type=Path, default=None, help="the corpus's independent _refs.jsonl")
    ap.add_argument("--budget", type=int, default=640, help="chunk token budget reference line")
    args = ap.parse_args(argv)
    if not args.run.is_dir():
        print(f"error: run dir not found: {args.run}", file=sys.stderr)
        return 2
    build_charts(args.run, args.out, args.refs, args.budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
