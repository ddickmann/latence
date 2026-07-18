"""HTML rendering — one self-contained page, no network, no JSON dump.

Every page is a single HTML document with its CSS inlined. There is **no external
stylesheet, no CDN font, no analytics beacon, no script tag** — the console works with the
machine's network cable pulled out, and it cannot phone home because there is nothing in
the emitted bytes that would reach outward. (:func:`external_references` is the audit the
tests run against the rendered output.)

Everything user-derived is escaped through :func:`html.escape` before it reaches the page.
"""

from __future__ import annotations

import re
from html import escape
from typing import TYPE_CHECKING
from urllib.parse import quote, urlencode

from latence_console import verdicts as V
from latence_console.runs import RunStatus

if TYPE_CHECKING:
    from datetime import datetime

    from latence_core.quality import QualityReport

    from latence_console import browse as B
    from latence_console.runs import RunSummary

_STYLE = """
:root { color-scheme: light dark; --fg:#16181d; --bg:#f7f7f9; --card:#fff; --muted:#5d6470;
        --line:#e2e4ea; --good:#1a7f4b; --warn:#8a6100; --bad:#a52020; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8eaee; --bg:#15171c; --card:#1e2128; --muted:#a2a9b6; --line:#2c313a;
          --good:#5cc98c; --warn:#e0b04a; --bad:#f08a8a; }
}
* { box-sizing:border-box; }
body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
       font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
main { max-width:62rem; margin:0 auto; }
h1 { font-size:1.7rem; margin:0 0 .25rem; }
h2 { font-size:1.15rem; margin:2rem 0 .6rem; }
a { color:inherit; }
.sub { color:var(--muted); margin:0 0 1.5rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:1rem 1.1rem; margin:0 0 .75rem; }
.verdict { border-left:4px solid var(--line); }
.verdict.good { border-left-color:var(--good); }
.verdict.warn { border-left-color:var(--warn); }
.verdict.bad  { border-left-color:var(--bad); }
.verdict .t { font-size:.72rem; letter-spacing:.09em; text-transform:uppercase;
              color:var(--muted); }
.verdict .h { font-size:1.05rem; margin:.15rem 0 0; }
.verdict .e { color:var(--muted); margin:.35rem 0 0; font-size:.92rem; }
table { border-collapse:collapse; width:100%; font-size:.94rem; }
th,td { text-align:left; padding:.42rem .6rem; border-bottom:1px solid var(--line);
        vertical-align:top; }
th { color:var(--muted); font-weight:600; font-size:.8rem; text-transform:uppercase;
     letter-spacing:.05em; }
tr:last-child td { border-bottom:none; }
.pill { display:inline-block; padding:.1rem .55rem; border-radius:999px; font-size:.78rem;
        font-weight:600; border:1px solid currentColor; }
.pill.passed{color:var(--good);} .pill.attention{color:var(--warn);}
.pill.failed{color:var(--bad);} .pill.unfinished{color:var(--muted);}
.foot { color:var(--muted); font-size:.84rem; margin-top:2.5rem;
        border-top:1px solid var(--line); padding-top:1rem; }
code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.88em; }
pre { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.85rem;
      white-space:pre-wrap; word-break:break-word; margin:0; }
.nav { display:flex; flex-wrap:wrap; gap:.4rem; margin:0 0 1.25rem; }
.nav .navlink { text-decoration:none; border:1px solid var(--line); border-radius:999px;
                padding:.2rem .75rem; font-size:.85rem; background:var(--card); }
.nav .navlink.on { border-color:currentColor; font-weight:600; }
"""

_FOOTER = (
    "This console is read-only. It shows files that are already on your storage and writes "
    "nothing back. It runs on this machine only and makes no network requests."
)

# Anything that would reach off this machine when the page is opened. Used by the offline
# audit test: the rendered bytes must contain none of these.
_EXTERNAL = re.compile(r"(?i)\b(?:https?:)?//(?!127\.0\.0\.1|localhost)[a-z0-9.-]+")


def external_references(html: str) -> list[str]:
    """Every off-machine URL reference in ``html`` (empty ⇒ the page is fully offline)."""
    return _EXTERNAL.findall(html)


def _when(stamp: datetime | None) -> str:
    return "unknown" if stamp is None else stamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="referrer" content="no-referrer">'
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head>"
        f'<body><main>{body}<p class="foot">{escape(_FOOTER)}</p></main></body></html>\n'
    )


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _verdict_card(v: V.Verdict) -> str:
    evidence = f'<p class="e">{escape(v.evidence)}</p>' if v.evidence else ""
    return (
        f'<div class="card verdict {v.tone.value}">'
        f'<div class="t">{escape(v.title)}</div>'
        f'<p class="h">{escape(v.headline)}</p>{evidence}</div>'
    )


def render_run_list(storage_uri: str, summaries: list[RunSummary]) -> str:
    """The landing page: every run on this Storage, newest first, with status + timestamps."""
    if not summaries:
        body = (
            "<h1>Runs</h1>"
            f'<p class="sub">Reading <code>{escape(storage_uri)}</code></p>'
            '<div class="card"><p>No runs found here yet. Run a pipeline against this '
            "storage location (for example <code>latence process</code> or "
            "<code>latence-demo</code>) and reload this page.</p></div>"
        )
        return _page("Latence runs", body)
    rows: list[str] = []
    for s in summaries:
        kind = "Update" if s.is_delta else "Full run"
        rows.append(
            "<tr>"
            f'<td><a href="/run/{escape(s.run_id)}">{escape(s.run_id)}</a></td>'
            f"<td>{escape(s.pipeline_name)}</td>"
            f'<td><span class="pill {s.status.value}">{escape(s.status.label)}</span></td>'
            f"<td>{escape(kind)}</td>"
            f"<td>{s.document_count}</td>"
            f"<td>{escape(_when(s.started_at))}</td>"
            f"<td>{escape(_when(s.finished_at))}</td>"
            "</tr>"
        )
    table = (
        "<table><thead><tr><th>Run</th><th>Pipeline</th><th>Status</th><th>Kind</th>"
        "<th>Documents</th><th>Started</th><th>Finished</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    body = (
        "<h1>Runs</h1>"
        f'<p class="sub">{len(summaries)} run(s) on <code>{escape(storage_uri)}</code>. '
        "Select one to read its quality report.</p>"
        f'<div class="card">{table}</div>'
    )
    return _page("Latence runs", body)


def render_run(summary: RunSummary, report: QualityReport | None) -> str:
    """One run's Quality Report as an experience: verdict first, numbers underneath."""
    head = (
        f"<h1>{escape(summary.run_id)}</h1>"
        f'<p class="sub">Pipeline <code>{escape(summary.pipeline_name)}</code> · '
        f"started {escape(_when(summary.started_at))} · "
        f"finished {escape(_when(summary.finished_at))} · "
        f'<a href="/">back to all runs</a></p>'
        f'<div class="card verdict {_tone_for(summary.status)}">'
        f'<div class="t">Overall</div>'
        f'<p class="h"><span class="pill {summary.status.value}">'
        f'{escape(summary.status.label)}</span></p>'
        f'<p class="e">{escape(summary.status.blurb)}</p></div>'
    )
    head += render_run_nav(summary.run_id, active="overview")
    if report is None:
        return _page(f"Run {summary.run_id}", head)

    parts = [head, "<h2>What the checks say</h2>"]
    parts += [_verdict_card(v) for v in V.headline_verdicts(report)]

    parts.append("<h2>Pipeline steps</h2>")
    parts.append(
        '<div class="card">'
        + _table(
            ["Step", "What ran", "Records in → out", "Time", "Outcome"],
            [list(row) for row in V.stage_rows(report)],
        )
        + "</div>"
    )

    reasons = V.screening_reasons(report)
    if reasons:
        parts.append("<h2>Why input was held back or flagged</h2>")
        parts.append(
            '<div class="card">'
            + _table(
                ["Checkpoint", "Decision", "Category", "Reason"],
                [list(row) for row in reasons],
            )
            + "</div>"
        )

    tables = V.fact_tables(report)
    if tables:
        parts.append("<h2>The numbers behind the checks</h2>")
        for facts in tables:
            parts.append(
                '<div class="card"><div class="t">'
                f"{escape(facts.title)}</div>"
                + _table(["Measure", "Value"], [[label, value] for label, value in facts.rows])
                + "</div>"
            )
    return _page(f"Run {summary.run_id}", "".join(parts))


# -- #188: the debugging loop (quarantine · chunk inspector · KG with Evidence) ------------

_NAV = (
    ("overview", "Overview", ""),
    ("quarantine", "Held back & flagged", "/quarantine"),
    ("chunks", "Chunks", "/chunks"),
    ("graph", "Knowledge graph", "/graph"),
)


def run_href(run_id: str, suffix: str = "", **params: str) -> str:
    """A link into one run's console. Ids ride the QUERY string, percent-encoded.

    A chunk ``record_id`` (``sha256:…#chunk-0``) and a mention id are not path segments —
    they contain ``:`` and ``#``. Putting them in the query keeps every console path a plain
    segment (so the traversal refusal in :func:`latence_console.runs.is_run_id` keeps
    covering the whole path space), and the id is only ever *matched against records already
    read*, never concatenated into a Storage URI.
    """
    url = f"/run/{quote(run_id, safe='')}{suffix}"
    if params:
        url += "?" + urlencode(params)
    return url


def render_run_nav(run_id: str, active: str) -> str:
    """The one nav strip every run page carries, so the four views are reachable."""
    links = []
    for key, label, suffix in _NAV:
        href = escape(run_href(run_id, suffix))
        cls = "navlink on" if key == active else "navlink"
        links.append(f'<a class="{cls}" href="{href}">{escape(label)}</a>')
    return f'<p class="nav">{"".join(links)}</p>'


def _run_head(run_id: str, title: str, blurb: str, active: str) -> str:
    return (
        f"<h1>{escape(title)}</h1>"
        f'<p class="sub">Run <code>{escape(run_id)}</code> · '
        f'<a href="{escape(run_href(run_id))}">back to the run overview</a></p>'
        + render_run_nav(run_id, active)
        + f'<p class="sub">{escape(blurb)}</p>'
    )


def _empty(message: str) -> str:
    return f'<div class="card"><p>{escape(message)}</p></div>'


def render_quarantine(
    run_id: str, quarantined: list[B.QuarantineEntry], flagged: list[B.FlagEntry]
) -> str:
    """What Screening held back, and what it let through carrying a risk marker."""
    parts = [
        _run_head(
            run_id,
            "Held back & flagged",
            "Screening removes dangerous input from the pipeline entirely (quarantine) and "
            "marks risky input that still proceeds (a risk marker that rides into the "
            "corpus, so a RAG consumer can exclude it). Both are listed here with the "
            "reason the run recorded.",
            "quarantine",
        ),
        f"<h2>Quarantined — removed from the pipeline ({len(quarantined)})</h2>",
    ]
    if quarantined:
        parts.append(
            '<div class="card">'
            + _table(
                ["File", "Type", "Bytes held back", "Checkpoint", "Detector", "Category",
                 "Reason"],
                [
                    [
                        e.file_name,
                        e.file_type,
                        f"{e.size_bytes:,}",
                        e.checkpoint,
                        e.detector,
                        e.category,
                        e.reason,
                    ]
                    for e in quarantined
                ],
            )
            + "</div>"
        )
    else:
        parts.append(_empty("Screening held nothing back in this run."))

    parts.append(f"<h2>Flagged — kept, with a risk marker in the corpus ({len(flagged)})</h2>")
    if flagged:
        rows = []
        for e in flagged:
            link = (
                f'<a href="{escape(run_href(run_id, "/chunk", id=e.chunk_id))}">'
                f"{escape(e.file_name)}</a>"
            )
            rows.append(
                "<tr>"
                f"<td>{link}</td>"
                f"<td>{escape(e.checkpoint)}</td>"
                f"<td>{escape(e.category)}</td>"
                f"<td>{escape(e.detector)}</td>"
                f"<td>{e.score:.2f}</td>"
                f"<td>{escape(e.reason)}</td>"
                "</tr>"
            )
        parts.append(
            '<div class="card"><table><thead><tr><th>Chunk</th><th>Checkpoint</th>'
            "<th>Category</th><th>Detector</th><th>Score</th><th>Reason</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )
    else:
        parts.append(_empty("No record in this run's corpus carries a risk marker."))
    return _page(f"Held back & flagged — {run_id}", "".join(parts))


def render_chunks(run_id: str, rows: list[B.ChunkRow]) -> str:
    """The chunk list: find the chunk you want to interrogate."""
    parts = [
        _run_head(
            run_id,
            "Chunks",
            "Every retrieval-sized piece this run produced, with the source page and "
            "character offsets it occupies in its source document. Open one to see how it "
            "resolves back through Provenance.",
            "chunks",
        )
    ]
    if not rows:
        parts.append(_empty("This run checkpointed no chunks."))
        return _page(f"Chunks — {run_id}", "".join(parts))
    body = []
    for row in rows:
        marks = []
        if row.flagged:
            marks.append('<span class="pill attention">flagged</span>')
        if row.redacted:
            marks.append('<span class="pill passed">redacted</span>')
        href = escape(run_href(run_id, "/chunk", id=row.chunk_id))
        body.append(
            "<tr>"
            f'<td><a href="{href}">{escape(row.file_name)}</a></td>'
            f"<td>{row.index}</td>"
            f"<td>{escape(row.pages)}</td>"
            f"<td>{escape(row.char_span)}</td>"
            f"<td>{row.token_count}</td>"
            f"<td>{' '.join(marks)}</td>"
            "</tr>"
        )
    parts.append(
        f'<p class="sub">{len(rows)} chunk(s).</p><div class="card"><table><thead><tr>'
        "<th>Source file</th><th>#</th><th>Page(s)</th><th>Characters</th><th>Tokens</th>"
        f"<th></th></tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )
    return _page(f"Chunks — {run_id}", "".join(parts))


def render_chunk(run_id: str, view: B.ChunkInspection) -> str:
    """The chunk inspector: where this chunk came from, and what was done to it."""
    parts = [
        _run_head(
            run_id,
            f"Chunk {view.index} of {view.file_name}",
            "Resolved back through Provenance to its source document, page and character "
            "offsets.",
            "chunks",
        )
    ]

    doc_state = (
        f"parsed ({view.document_pages} page(s), {view.document_chars:,} characters)"
        if view.document_parsed and view.document_pages is not None
        and view.document_chars is not None
        else ("parsed" if view.document_parsed else "not in this run's checkpoints")
    )
    if view.document_error:
        doc_state += f" — parse error: {view.document_error}"
    parts.append("<h2>Where it came from</h2>")
    parts.append(
        '<div class="card">'
        + _table(
            ["Measure", "Value"],
            [
                ["Source file", view.file_name],
                ["Source URI", view.source_uri],
                ["Source system", view.source_system or "—"],
                ["Source document", doc_state],
                ["Document id", view.document_id],
                ["Source page(s) — Provenance", view.provenance_pages],
                ["Source page(s) — resolved from the offset + page map", view.resolved_pages],
                ["Character offsets — Provenance", view.provenance_span],
                ["Character offsets — resolved from the offset map", view.resolved_span],
                ["Chunk id", view.chunk_id],
                ["Chunk length", f"{view.character_count:,} characters, "
                                 f"{view.token_count} tokens"],
            ],
        )
        + "</div>"
    )

    parts.append("<h2>Classification</h2>")
    parts.append(
        '<div class="card">'
        + _table(
            ["Measure", "Value"],
            [
                ["Language", view.language],
                ["Category", view.category],
                ["Sensitivity", view.sensitivity],
            ],
        )
        + "</div>"
    )

    parts.append("<h2>Redaction disposition</h2>")
    tone = "bad" if view.redaction.control_disabled else (
        "good" if view.redaction.applied else "warn"
    )
    parts.append(
        f'<div class="card verdict {tone}"><div class="t">Redaction</div>'
        f'<p class="h">{escape(view.redaction.summary)}</p></div>'
    )
    if view.redaction.spans:
        parts.append(
            '<div class="card">'
            + _table(
                ["PII type", "Detector", "Action", "Placeholder", "In this chunk",
                 "In the source document", "Page(s)", "Confidence"],
                [
                    [
                        s.pii_type,
                        s.detector,
                        s.action,
                        s.placeholder,
                        s.chunk_span,
                        s.source_span,
                        s.pages,
                        f"{s.confidence:.2f}",
                    ]
                    for s in view.redaction.spans
                ],
            )
            + "</div>"
        )

    if view.risk_markers:
        parts.append("<h2>Risk markers on this chunk</h2>")
        parts.append(
            '<div class="card">'
            + _table(
                ["Category", "Detector", "Score", "Reason"],
                [
                    [m.category, m.detector, f"{m.score:.2f}", m.reason]
                    for m in view.risk_markers
                ],
            )
            + "</div>"
        )

    label = "Corpus text (PII-handled)" if view.excerpt_is_masked else "Chunk text"
    suffix = " — truncated" if view.excerpt_truncated else ""
    parts.append(f"<h2>{escape(label)}{escape(suffix)}</h2>")
    parts.append(f'<div class="card"><pre>{escape(view.excerpt)}</pre></div>')
    return _page(f"Chunk {view.index} of {view.file_name} — {run_id}", "".join(parts))


def _edge_card(run_id: str, edge: B.EdgeView) -> str:
    """One edge and — always — the Evidence that justifies it."""
    src = escape(run_href(run_id, "/entity", id=edge.source_node_id))
    tgt = escape(run_href(run_id, "/entity", id=edge.target_node_id))
    kind = " · inferred" if edge.inferred else ""
    mention_rows = "".join(
        "<tr>"
        f"<td>{escape(m.text)}</td>"
        f"<td>{escape(m.label)}</td>"
        f"<td>{escape(m.file_name)}</td>"
        f"<td>{escape(m.pages)}</td>"
        f"<td>{escape(m.char_span)}</td>"
        f"<td>{'—' if not m.resolved else f'{m.confidence:.2f}'}</td>"
        f"<td><code>{escape(m.mention_id)}</code></td>"
        "</tr>"
        for m in edge.mentions
    )
    if mention_rows:
        mentions = (
            "<table><thead><tr><th>Mention</th><th>Type</th><th>File</th><th>Page(s)</th>"
            "<th>Characters</th><th>Confidence</th><th>Mention id</th></tr></thead>"
            f"<tbody>{mention_rows}</tbody></table>"
        )
    else:
        mentions = (
            '<p class="e">This Evidence cites no individual mention — see the '
            "justification above.</p>"
        )
    return (
        '<div class="card">'
        f'<p class="h"><a href="{src}">{escape(edge.source_name)}</a> '
        f"<strong>—{escape(edge.label)}→</strong> "
        f'<a href="{tgt}">{escape(edge.target_name)}</a> '
        f'<span class="e">(confidence {edge.confidence:.2f}{escape(kind)})</span></p>'
        f'<p class="e">Evidence — confidence {edge.evidence_confidence:.2f} · '
        f"{edge.document_count} source document(s) · "
        f"{escape(edge.snippet) if edge.snippet else 'no justification text recorded'}</p>"
        f"{mentions}</div>"
    )


def render_graph(run_id: str, entities: list[B.EntityView], edges: list[B.EdgeView]) -> str:
    """The KG browser: entities to navigate, and every edge with its Evidence."""
    parts = [
        _run_head(
            run_id,
            "Knowledge graph",
            "The canonical entities this run resolved and the relations between them. "
            "Every edge is shown with the Evidence that justifies it — the source mentions, "
            "where they sit in their documents, and how strongly.",
            "graph",
        ),
        f"<h2>Entities ({len(entities)})</h2>",
    ]
    if entities:
        rows = "".join(
            "<tr>"
            f'<td><a href="{escape(run_href(run_id, "/entity", id=e.node_id))}">'
            f"{escape(e.canonical_name)}</a></td>"
            f"<td>{escape(e.label)}</td>"
            f"<td>{escape(e.kb_id or '—')}</td>"
            f"<td>{e.confidence:.2f}</td>"
            f"<td>{e.mention_count}</td>"
            f"<td>{e.document_count}</td>"
            f"<td>{e.edge_count}</td>"
            "</tr>"
            for e in entities
        )
        parts.append(
            '<div class="card"><table><thead><tr><th>Entity</th><th>Type</th><th>KB link</th>'
            "<th>Confidence</th><th>Mentions</th><th>Documents</th><th>Edges</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
    else:
        parts.append(_empty("This run assembled no graph nodes."))

    parts.append(f"<h2>Relations ({len(edges)})</h2>")
    if edges:
        parts.extend(_edge_card(run_id, edge) for edge in edges)
    else:
        parts.append(_empty("This run assembled no graph edges."))
    return _page(f"Knowledge graph — {run_id}", "".join(parts))


def render_entity(run_id: str, entity: B.EntityView, edges: list[B.EdgeView]) -> str:
    """One entity, and every edge touching it — each with its Evidence."""
    parts = [
        _run_head(
            run_id,
            entity.canonical_name,
            "One canonical entity and every relation that touches it.",
            "graph",
        ),
        '<div class="card">'
        + _table(
            ["Measure", "Value"],
            [
                ["Type", entity.label],
                ["External KB link", entity.kb_id or "not linked"],
                ["Confidence", f"{entity.confidence:.2f}"],
                ["Mentions folded into it", str(entity.mention_count)],
                ["Source documents", str(entity.document_count)],
                ["Node id", entity.node_id],
            ],
        )
        + "</div>",
        f"<h2>Relations ({len(edges)})</h2>",
    ]
    if edges:
        parts.extend(_edge_card(run_id, edge) for edge in edges)
    else:
        parts.append(_empty("No relation in this graph touches this entity."))
    return _page(f"{entity.canonical_name} — {run_id}", "".join(parts))


def render_not_found(what: str) -> str:
    body = (
        "<h1>Not found</h1>"
        f'<div class="card"><p>{escape(what)}</p>'
        '<p><a href="/">Back to all runs</a></p></div>'
    )
    return _page("Not found", body)


def render_method_not_allowed() -> str:
    body = (
        "<h1>Read-only</h1>"
        '<div class="card"><p>This console only reads. It does not accept changes of any '
        "kind, so that request was refused.</p>"
        '<p><a href="/">Back to all runs</a></p></div>'
    )
    return _page("Read-only", body)


def _tone_for(status: RunStatus) -> str:
    return {
        RunStatus.PASSED: "good",
        RunStatus.ATTENTION: "warn",
        RunStatus.FAILED: "bad",
        RunStatus.UNFINISHED: "warn",
    }[status]
