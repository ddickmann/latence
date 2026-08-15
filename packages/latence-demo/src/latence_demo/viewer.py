"""Self-contained web viewer + the ``export.demo_site`` Export Provider (S13, ADR-0011).

The showable demo's decisive artifact is a **thin, self-contained web viewer** that
renders the knowledge graph (nodes, edges, and per-edge Evidence) and the Quality
Report — with **no external CDN, no network** (the S13 AC). This module builds that
page as ONE HTML file with all CSS/JS/graph inlined, and registers a
:class:`DemoSiteExport` Export Provider so a Pipeline materialises the viewer as a
portable file on Storage, exactly like the KG Export writes portable graph files
(ADR-0017): the deliverable is files, handable to a client, loadable offline.

Two entry points:

* :class:`DemoSiteExport` — the entry-point Provider (``export.demo_site``). It sits
  downstream of Graph Assembly, receives the run's :class:`~latence_core.contracts.GraphRecord`s
  through the Export seam, and writes a self-contained ``index.html`` KG viewer plus a
  typed :class:`~latence_demo.contracts.DemoSiteManifest` sidecar. Provenance +
  Classification are read off every record it renders (validated at construction — the
  enforced boundary), and it is a pass-through sink returning its inputs unchanged so the
  Runner checkpoints them, mirroring the S1/S9 exporters.

* :func:`render_site` — the full-page renderer the demo CLI calls once the run has a
  finalised Quality Report: it folds the report + the corpus manifest into the same page,
  next to the KG. (The Export seam only carries records, not the post-run report, so the
  CLI composes the complete showable page; the Provider guarantees a valid standalone KG
  page even when used outside the demo.)

Determinism: nodes are laid out on a fixed circle in node-id order and every element is
emitted in a stable order, so a seeded run renders byte-identically (Baseline bar).
"""

from __future__ import annotations

import json
import math
from html import escape
from typing import TYPE_CHECKING, Any

from latence_core.contracts import GraphEdge, GraphNode, GraphRecord, GraphScope
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)

from latence_demo.contracts import DemoSiteManifest, GraphSummary

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import Record
    from latence_core.storage import Storage

# A small, fixed, colour-blind-safe palette for node types (inlined; no external CSS).
_TYPE_COLORS = (
    "#4e79a7",
    "#59a14f",
    "#e15759",
    "#f28e2b",
    "#b07aa1",
    "#76b7b2",
    "#edc948",
    "#ff9da7",
)


def _partition(records: Iterable[Record]) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Split the records flowing through the Export seam into graph nodes + edges.

    Only :class:`GraphRecord`s are rendered (an Export wired downstream of Graph Assembly
    sees only graph records); any other carrier is ignored. Every rendered node/edge carries
    Provenance + Classification (contract-guaranteed on the :class:`GraphRecord`), so the
    viewer only ever shows records that passed the Stage-boundary check.
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    for r in records:
        if not isinstance(r, GraphRecord):
            continue
        # Touch Provenance/Classification so a malformed record (missing either) would have
        # failed construction upstream; rendering only proceeds for validated carriers.
        _ = r.provenance.document_id
        _ = r.classification.language
        if r.scope is GraphScope.NODE and r.node is not None:
            nodes.append(r.node)
        elif r.scope is GraphScope.EDGE and r.edge is not None:
            edges.append(r.edge)
    return nodes, edges


def summarize_graph(nodes: list[GraphNode], edges: list[GraphEdge]) -> GraphSummary:
    """Roll the assembled nodes/edges into the typed :class:`GraphSummary` (counts only)."""
    nodes_by_type: dict[str, int] = {}
    for n in nodes:
        nodes_by_type[n.label] = nodes_by_type.get(n.label, 0) + 1
    edges_by_label: dict[str, int] = {}
    for e in edges:
        edges_by_label[e.label] = edges_by_label.get(e.label, 0) + 1
    return GraphSummary(
        nodes_total=len(nodes),
        edges_total=len(edges),
        nodes_linked=sum(1 for n in nodes if n.kb_id),
        edges_with_evidence=sum(1 for e in edges if e.evidence.mention_ids),
        nodes_by_type=dict(sorted(nodes_by_type.items())),
        edges_by_label=dict(sorted(edges_by_label.items())),
    )


# -- SVG knowledge-graph rendering (deterministic circular layout) -------------


def _node_positions(
    nodes: list[GraphNode], width: int, height: int
) -> dict[str, tuple[float, float]]:
    """Lay nodes out on a fixed circle in node-id order — deterministic, dependency-free.

    A force-directed layout would need a JS physics lib (a CDN dependency, forbidden here) and
    a random seed (non-reproducible). A fixed circle keeps the page self-contained and the
    output byte-stable across runs (Baseline bar), while still rendering a readable graph.
    """
    ordered = sorted(nodes, key=lambda n: n.node_id)
    cx, cy = width / 2, height / 2
    radius = min(width, height) / 2 - 90
    positions: dict[str, tuple[float, float]] = {}
    count = max(len(ordered), 1)
    for i, node in enumerate(ordered):
        angle = 2 * math.pi * i / count - math.pi / 2
        positions[node.node_id] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
    return positions


def _render_graph_svg(nodes: list[GraphNode], edges: list[GraphEdge]) -> str:
    """Render the KG as an inline SVG: edges as lines, nodes as coloured, labelled circles."""
    width, height = 720, 520
    if not nodes:
        return (
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="empty graph">'
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'fill="#888">No graph nodes assembled.</text></svg>'
        )
    positions = _node_positions(nodes, width, height)
    type_color: dict[str, str] = {}
    for label in sorted({n.label for n in nodes}):
        type_color[label] = _TYPE_COLORS[len(type_color) % len(_TYPE_COLORS)]

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="knowledge graph with {len(nodes)} nodes and {len(edges)} edges">'
    ]
    # Edges first (under the nodes), each labelled with its relation + evidence count.
    for e in sorted(edges, key=lambda x: x.edge_id):
        src = positions.get(e.source_node_id)
        tgt = positions.get(e.target_node_id)
        if src is None or tgt is None:
            continue
        x1, y1 = src
        x2, y2 = tgt
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ev = len(e.evidence.mention_ids)
        title = escape(f"{e.label} (confidence {e.confidence:.2f}, {ev} evidence mention(s))")
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#b0b0b0" stroke-width="1.5"><title>{title}</title></line>'
        )
        parts.append(
            f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" '
            f'font-size="10" fill="#666">{escape(e.label)}</text>'
        )
    # Nodes on top.
    for n in sorted(nodes, key=lambda x: x.node_id):
        x, y = positions[n.node_id]
        color = type_color[n.label]
        kb = f" · KB:{escape(n.kb_id)}" if n.kb_id else ""
        title = escape(
            f"{n.canonical_name} [{n.label}] "
            f"conf {n.confidence:.2f}, {len(n.member_mention_ids)} mention(s), "
            f"{len(n.source_document_ids)} document(s)"
        ) + kb
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="{color}" '
            f'stroke="#fff" stroke-width="2"><title>{title}</title></circle>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{y - 16:.1f}" text-anchor="middle" '
            f'font-size="11" fill="#222">{escape(n.canonical_name)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _render_evidence_table(edges: list[GraphEdge], nodes: list[GraphNode]) -> str:
    """A table of edges with their per-edge Evidence — the 'every edge is justified' proof."""
    name_by_id = {n.node_id: n.canonical_name for n in nodes}
    rows: list[str] = []
    for e in sorted(edges, key=lambda x: x.edge_id):
        src = escape(name_by_id.get(e.source_node_id, e.source_node_id))
        tgt = escape(name_by_id.get(e.target_node_id, e.target_node_id))
        ev = e.evidence
        snippet = escape(ev.snippet) if ev.snippet else f"{len(ev.mention_ids)} mention(s)"
        rows.append(
            f"<tr><td>{src}</td><td class='rel'>{escape(e.label)}</td><td>{tgt}</td>"
            f"<td>{e.confidence:.2f}</td><td>{len(ev.mention_ids)}</td>"
            f"<td>{len(ev.document_ids)}</td><td class='snip'>{snippet}</td></tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='7' class='muted'>No edges assembled.</td></tr>")
    return (
        "<table><thead><tr><th>Head</th><th>Relation</th><th>Tail</th>"
        "<th>Conf.</th><th>Evidence mentions</th><th>Docs</th><th>Justification</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _render_report_section(report: dict[str, Any] | None) -> str:
    """Render the Quality Report (already loaded as a dict) into readable HTML tables.

    ``report`` is the persisted ``quality-report.json`` parsed to a dict (or ``None`` when the
    Provider renders a standalone KG page with no report available). Counts only — the report
    itself never carries raw text/PII (S6/S9), so this is safe to embed in a shareable page.
    """
    if report is None:
        return "<p class='muted'>Quality Report not available in this render.</p>"
    comp = report.get("completeness", {})
    status = "PASS" if report.get("contracts_complete") else "FAIL"
    lines = [
        "<dl class='kv'>",
        f"<dt>Documents</dt><dd>{report.get('document_count', 0)}</dd>",
        f"<dt>Duration</dt><dd>{report.get('total_duration_seconds', 0.0):.3f}s</dd>",
        f"<dt>Contracts complete</dt><dd class='{status.lower()}'>{status}</dd>",
        f"<dt>Provenance</dt><dd>{comp.get('records_with_provenance', 0)}"
        f"/{comp.get('records_checked', 0)}</dd>",
        f"<dt>Classification</dt><dd>{comp.get('records_with_classification', 0)}"
        f"/{comp.get('records_checked', 0)}</dd>",
        "</dl>",
    ]
    stage_rows = []
    for m in report.get("stages", []):
        stage_rows.append(
            f"<tr><td>{escape(str(m.get('name', '')))}</td>"
            f"<td>{escape(str(m.get('capability', '')))}</td>"
            f"<td>{escape(str(m.get('provider', '')))}</td>"
            f"<td>{m.get('records_in', 0)}</td><td>{m.get('records_out', 0)}</td>"
            f"<td>{m.get('duration_seconds', 0.0):.3f}s</td></tr>"
        )
    stages = (
        "<table><thead><tr><th>Stage</th><th>Capability</th><th>Provider</th>"
        "<th>In</th><th>Out</th><th>Time</th></tr></thead><tbody>"
        + "".join(stage_rows)
        + "</tbody></table>"
    )
    screening = report.get("screening")
    screen_html = ""
    if screening:
        finding_rows = []
        for f in screening.get("findings", []):
            if f.get("verdict") == "pass":
                continue
            finding_rows.append(
                f"<tr><td>{escape(str(f.get('checkpoint', '')))}</td>"
                f"<td class='verdict-{escape(str(f.get('verdict', '')))}'>"
                f"{escape(str(f.get('verdict', '')))}</td>"
                f"<td>{escape(str(f.get('category', '')))}</td>"
                f"<td class='snip'>{escape(str(f.get('reason', '')))}</td></tr>"
            )
        screen_html = (
            f"<h3>Screening — {screening.get('quarantined', 0)} quarantined, "
            f"{screening.get('flagged', 0)} flagged</h3>"
            "<table><thead><tr><th>Checkpoint</th><th>Verdict</th><th>Category</th>"
            "<th>Reason</th></tr></thead><tbody>"
            + ("".join(finding_rows) or "<tr><td colspan='4' class='muted'>clean</td></tr>")
            + "</tbody></table>"
        )
    return "".join(lines) + "<h3>Stages</h3>" + stages + screen_html


_STYLE = """
:root{--bg:#f7f8fa;--card:#fff;--ink:#1c2733;--muted:#6b7785;--line:#e3e7ec;--accent:#4e79a7}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,
Arial,sans-serif;background:var(--bg);color:var(--ink)}
header{background:var(--ink);color:#fff;padding:22px 28px}
header h1{margin:0;font-size:20px}
header p{margin:4px 0 0;color:#b8c4d0;font-size:13px}
main{max-width:960px;margin:0 auto;padding:24px 20px 60px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px;
margin:0 0 22px}
.card h2{margin:0 0 14px;font-size:16px}
h3{font-size:14px;margin:18px 0 8px;color:var(--muted);text-transform:uppercase;
letter-spacing:.04em}
svg{width:100%;height:auto;display:block;background:#fbfcfd;border:1px solid var(--line);
border-radius:8px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}
td.rel{color:var(--accent);font-weight:600}
td.snip{color:var(--muted);max-width:320px}
.muted{color:var(--muted)}
.pass{color:#2e7d32;font-weight:600}.fail{color:#c62828;font-weight:600}
.verdict-quarantine{color:#c62828;font-weight:600}
.verdict-flag{color:#e6820b;font-weight:600}
.kv{display:grid;grid-template-columns:auto auto;gap:2px 18px;margin:0}
.kv dt{color:var(--muted)}.kv dd{margin:0;font-variant-numeric:tabular-nums}
.pills span{display:inline-block;background:#eef2f6;border-radius:12px;padding:2px 10px;
margin:0 6px 6px 0;font-size:12px}
.swap{background:#f4f7fb;border-left:3px solid var(--accent);padding:10px 14px;font-size:13px;
border-radius:0 6px 6px 0}
code{background:#eef2f6;padding:1px 5px;border-radius:4px;font-size:12px}
footer{color:var(--muted);font-size:12px;text-align:center;padding:8px 0 40px}
"""


def render_site(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    *,
    pipeline_name: str,
    report: dict[str, Any] | None = None,
    corpus: dict[str, str] | None = None,
) -> str:
    """Render the complete self-contained showable page (KG + Quality Report + corpus).

    All CSS/JS/graph is inlined; the page references **no external CDN or network resource**
    (the S13 AC), so it opens offline from a clean clone or from Storage. ``report`` is the
    parsed ``quality-report.json`` dict (folded in by the CLI); ``corpus`` optionally names the
    ingested files and their Screening outcome for a self-describing header. Deterministic.
    """
    summary = summarize_graph(nodes, edges)
    type_pills = "".join(
        f"<span>{escape(t)}: {c}</span>" for t, c in summary.nodes_by_type.items()
    )
    label_pills = "".join(
        f"<span>{escape(lbl)}: {c}</span>" for lbl, c in summary.edges_by_label.items()
    )
    corpus_html = ""
    if corpus:
        rows = "".join(
            f"<tr><td>{escape(name)}</td><td class='snip'>{escape(disp)}</td></tr>"
            for name, disp in sorted(corpus.items())
        )
        corpus_html = (
            "<div class='card'><h2>Bundled corpus</h2>"
            "<table><thead><tr><th>File</th><th>Screening</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
    swap = (
        "<div class='card'><h2>Model- and cloud-agnostic</h2>"
        "<p class='swap'>Swap one line — e.g. <code>provider: entity.gazetteer</code> → "
        "<code>provider: entity.gliner</code> (local model) or "
        "<code>provider: entity.endpoint</code> (a cloud/vLLM endpoint) — and rerun the "
        "<em>identical</em> Pipeline. Same Stages, same contracts, same viewer. See "
        "<code>demo-pipeline.swap.yaml</code>.</p></div>"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Latence demo — {escape(pipeline_name)}</title>"
        f"<style>{_STYLE}</style></head><body>"
        "<header><h1>Latence — AI-ready data from a messy corpus</h1>"
        "<p>Source → Screen → Parse → Chunk → Screen → Entity → Relation → Redaction → "
        "Profiling → Disambiguation → Graph Assembly → Export. CPU-only, offline.</p></header>"
        "<main>"
        f"<div class='card'><h2>Knowledge graph — {summary.nodes_total} nodes, "
        f"{summary.edges_total} edges</h2>"
        f"<div class='pills'>{type_pills}{label_pills}</div>"
        f"{_render_graph_svg(nodes, edges)}</div>"
        "<div class='card'><h2>Edges &amp; per-edge Evidence</h2>"
        f"{_render_evidence_table(edges, nodes)}</div>"
        f"<div class='card'><h2>Quality Report</h2>{_render_report_section(report)}</div>"
        f"{corpus_html}{swap}"
        "</main>"
        "<footer>Generated by latence-demo · self-contained · no external network</footer>"
        "</body></html>"
    )


class DemoSiteExport:
    """Export Provider that writes the self-contained KG web viewer (``export.demo_site``).

    Config:
        basename: str — output stem (default 'index'); writes ``<basename>.html`` +
            ``<basename>.demo.json`` (the typed :class:`DemoSiteManifest` sidecar).
        pipeline_name: str — shown in the page title (default 'latence-demo').

    Wired downstream of Graph Assembly, it renders the :class:`GraphRecord`s that reach it
    (nodes / edges / per-edge Evidence) into a standalone HTML page — no CDN, no network — and
    is a pass-through sink returning its inputs unchanged so the Runner checkpoints them
    (mirroring the S1/S9 exporters). The full showable page (with the Quality Report folded in)
    is composed post-run by the demo CLI via :func:`render_site`; this Provider guarantees a
    valid standalone KG viewer for any Pipeline that installs the package.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic,
        Apache-2.0.

        The demo-site Export renders a standalone HTML KG viewer with a deterministic circular
        layout (dependency-free, no CDN/network). It ships no model weights (``model_id=None``),
        is Apache-2.0 framework code (verified per ADR-0012 — weights-and-code trivially the
        same), deterministic, CPU-only, does not batch, and has no per-call cost.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=32,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._basename = str(config.get("basename", "index"))
        self._pipeline_name = str(config.get("pipeline_name", "latence-demo"))

    def export(self, records: Iterable[Record], storage: Storage, out_dir: str) -> list[str]:
        record_list = list(records)
        nodes, edges = _partition(record_list)
        base = out_dir.rstrip("/")
        storage.makedirs(base)
        html_uri = f"{base}/{self._basename}.html"
        manifest_uri = f"{base}/{self._basename}.demo.json"
        html = render_site(nodes, edges, pipeline_name=self._pipeline_name)
        storage.write_atomic(html_uri, html.encode("utf-8"))
        manifest = DemoSiteManifest(
            site_uri=html_uri,
            graph=summarize_graph(nodes, edges),
            self_contained=_is_self_contained(html),
        )
        storage.write_atomic(manifest_uri, manifest.model_dump_json(indent=2).encode("utf-8"))
        return [html_uri, manifest_uri]


# Hosts a self-contained page may legitimately reference (none — the page is fully inline).
# Any occurrence of these URL schemes with a remote host is an external-resource leak.
_EXTERNAL_MARKERS = ("http://", "https://", "//cdn", "src=\"//", "href=\"//")


def _is_self_contained(html: str) -> bool:
    """True when the page pulls no external CDN/network resource (the S13 AC).

    A conservative structural check: the page is self-contained iff it contains none of the
    remote-fetch markers. The rendered page inlines all CSS/JS/graph, so a clean render has
    zero. (URNs in embedded data — e.g. an ``xmlns`` SVG namespace — are not fetched, so the
    SVG/XHTML namespace URLs are excluded explicitly.)
    """
    haystack = html
    # The SVG/graphml/xhtml namespace URLs are declarations, never fetched — exclude them.
    for allowed in (
        "http://www.w3.org/",
        "http://graphml.graphdrawing.org/",
    ):
        haystack = haystack.replace(allowed, "")
    return not any(marker in haystack for marker in _EXTERNAL_MARKERS)


def load_graph_records(jsonl_text: str) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Load nodes/edges from a Graph-Assembly checkpoint (newline-delimited GraphRecord JSON).

    Used by the CLI to recover the assembled graph from the run's checkpoint so the full page
    can be composed post-run without re-running Graph Assembly. Validates each line through the
    contract, so Provenance/Classification are enforced on the way in.
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    for line in jsonl_text.splitlines():
        if not line.strip():
            continue
        record = GraphRecord.model_validate_json(line)
        if record.scope is GraphScope.NODE and record.node is not None:
            nodes.append(record.node)
        elif record.scope is GraphScope.EDGE and record.edge is not None:
            edges.append(record.edge)
    return nodes, edges


def load_report(report_json: str) -> dict[str, Any]:
    """Parse a persisted ``quality-report.json`` into a dict for :func:`render_site`."""
    data: dict[str, Any] = json.loads(report_json)
    return data
