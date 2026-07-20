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
from typing import TYPE_CHECKING, TypeVar
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
#
# An authority-form reference: an OPTIONAL scheme, then ``//``, then the host. The scheme is
# optional *and reachable* — no leading ``\b``, because a protocol-relative reference is always
# preceded by a non-word character (``"``, ``(``, whitespace), so a word boundary there can never
# hold and the scheme-less branch would be dead. Requiring the double slash keeps same-origin
# paths (``/run/abc``) out, and the leading ``(?<!/)`` keeps an EMPTY authority out too, so a
# local ``file:///tmp/x`` is not misread as ``//`` + host ``tmp``. Loopback is excluded by host,
# below, not by a lookahead — a prefix lookahead would whitelist ``localhost.evil.example.com``.
_EXTERNAL = re.compile(r"(?i)(?<!/)(?:\b[a-z][a-z0-9+.-]*:)?//(?P<host>\[[0-9a-f:]+\]|[a-z0-9.-]+)")
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})


def external_references(html: str) -> list[str]:
    """Every off-machine URL reference in ``html`` (empty ⇒ the page is fully offline)."""
    return [
        m.group(0) for m in _EXTERNAL.finditer(html) if m.group("host").lower() not in _LOOPBACK
    ]


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
    body = "".join("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>" for row in rows)
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
        f"{escape(summary.status.label)}</span></p>"
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


def _empty(message: str, *, partial: str | None = None, truncated: bool = False) -> str:
    """The "nothing here" card.

    ``message`` asserts something about the WHOLE run ("no relation touches this entity"), and
    that claim is only true when the reader actually read the whole run. The list views' scan is
    capped (:data:`~latence_console.artifacts.MAX_SCAN_RECORDS`), so on a capped read an empty
    list means "none in the prefix the reader kept" — a different, much weaker statement. Where a
    call site supplies ``partial``, a capped read shows that instead: the same emptiness, without
    the affirmative claim the files on Storage may contradict (audit R5, ADR-0021/0031).
    """
    text = partial if truncated and partial is not None else message
    return f'<div class="card"><p>{escape(text)}</p></div>'


def _count(total: int, truncated: bool) -> str:
    """A heading's count, marked as a partial tally when the scan stopped short (audit R5).

    ``(0)`` reads as "there are none"; on a capped read the honest reading is "none among the
    records this view read", so the number is never left standing bare.
    """
    return f"{total} read so far" if truncated else str(total)


# -- pagination (audit R4) ---------------------------------------------------
#
# A run is unbounded; a browse page is not. Before this, every list view emitted one row per
# record into a single HTML document, so a realistically-sized run produced a multi-megabyte page
# the browser had to lay out in one go. Every list view now renders at most PAGE_ROWS rows and
# carries a pager, so the page size is a function of PAGE_ROWS and nothing else.

#: Rows (or edge cards) a single browse page renders.
PAGE_ROWS = 200

_T = TypeVar("_T")


def parse_page(raw: str | None) -> int:
    """A ``?page=`` value as a page number — anything unreadable is page 1, never an error."""
    if raw is None:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _paginate(items: list[_T], page: int) -> tuple[list[_T], int, int]:
    """``(this page's items, the clamped page number, the total page count)``."""
    total_pages = max(1, (len(items) + PAGE_ROWS - 1) // PAGE_ROWS)
    page = min(max(page, 1), total_pages)
    start = (page - 1) * PAGE_ROWS
    return items[start : start + PAGE_ROWS], page, total_pages


def _pager(
    run_id: str,
    suffix: str,
    *,
    param: str,
    page: int,
    total_pages: int,
    total_items: int,
    noun: str,
    other: dict[str, str] | None = None,
) -> str:
    """The "showing X–Y of Z" line plus previous/next links, or just the count on one page."""
    first = 0 if total_items == 0 else (page - 1) * PAGE_ROWS + 1
    last = min(page * PAGE_ROWS, total_items)
    shown = (
        f"{total_items} {noun}."
        if total_pages == 1
        else f"Showing {first}–{last} of {total_items} {noun} (page {page} of {total_pages})."
    )
    if total_pages == 1:
        return f'<p class="sub">{escape(shown)}</p>'

    def link(target: int, label: str) -> str:
        params = dict(other or {})
        params[param] = str(target)
        return f'<a class="navlink" href="{escape(run_href(run_id, suffix, **params))}">{label}</a>'

    links = []
    if page > 1:
        links.append(link(page - 1, "← Previous"))
    if page < total_pages:
        links.append(link(page + 1, "Next →"))
    return f'<p class="sub">{escape(shown)}</p><p class="nav">{"".join(links)}</p>'


def _truncation_notice(truncated: bool, unreadable: tuple[str, ...] = ()) -> str:
    """Say it when the reader stopped short — a partial list is never shown as a whole one.

    Two different things stop it short, and the notice must name the right one (audit R7). The
    cap (``truncated``) means the run is larger than one view; ``unreadable`` means the reader
    could not read a file *at all*, or lost the stream part-way through and kept only a prefix.
    The second was previously invisible: an unreadable checkpoint produced an empty list with no
    notice, which the page then stated as fact (ADR-0034, ADR-0017/0021).
    """
    if not truncated and not unreadable:
        return ""
    if unreadable:
        listed = ", ".join(escape(uri.rsplit("/", 1)[-1]) for uri in unreadable)
        head = (
            '<div class="card verdict warn"><div class="t">Partial view</div>'
            '<p class="h">The console could not read part of this run.</p>'
            f'<p class="e">These files on Storage would not read: {listed}. '
            "What is listed below is therefore only what could be read — an empty list or a "
            "count here is NOT a statement that the run holds nothing. The files on Storage "
            "remain the source of truth; check their permissions and query them directly."
        )
        if truncated:
            head += (
                " The reader also hit its per-view record limit, so even the readable files are "
                "shown only in part."
            )
        return head + "</p></div>"
    return (
        '<div class="card verdict warn"><div class="t">Partial view</div>'
        '<p class="h">This run holds more records than the console reads into one view.</p>'
        '<p class="e">The rows below are a prefix of the run, not all of it — every one of them '
        "is the most-downstream (post-Redaction) version, but records beyond the reader's limit "
        "are not listed. Query the corpus files directly to audit the run exhaustively.</p></div>"
    )


def render_quarantine(
    run_id: str,
    quarantined: list[B.QuarantineEntry],
    flagged: list[B.FlagEntry],
    *,
    page: int = 1,
    held_page: int = 1,
    truncated: bool = False,
    unreadable: tuple[str, ...] = (),
) -> str:
    """What Screening held back, and what it let through carrying a risk marker."""
    incomplete = truncated or bool(unreadable)
    shown_held, held_page, held_pages = _paginate(quarantined, held_page)
    shown_flagged, page, flag_pages = _paginate(flagged, page)
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
        _truncation_notice(truncated, unreadable),
        # The held-back list is capped too since audit R7 (the quarantine sidecar grows with the
        # run), so its tally is hedged on an incomplete read exactly like the flagged one — but
        # only when there are rows: a cap only ever drops records *after* keeping its full
        # allowance, so an empty list was never shortened by one, and the empty-state message
        # below carries the unreadable-file caveat in words instead of muddying the number.
        "<h2>Quarantined — removed from the pipeline "
        f"({_count(len(quarantined), incomplete and bool(quarantined))})</h2>",
    ]
    if quarantined:
        parts.append(
            _pager(
                run_id,
                "/quarantine",
                param="held",
                page=held_page,
                total_pages=held_pages,
                total_items=len(quarantined),
                noun="record(s)",
                other={"page": str(page)},
            )
        )
        parts.append(
            '<div class="card">'
            + _table(
                ["File", "Type", "Bytes held back", "Checkpoint", "Detector", "Category", "Reason"],
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
                    for e in shown_held
                ],
            )
            + "</div>"
        )
    else:
        parts.append(
            _empty(
                "Screening held nothing back in this run.",
                # The cap cannot shorten this list, but an unreadable quarantine sidecar can
                # empty it outright (audit R7) — and then "held nothing back" is a false
                # statement about a file the console never managed to read.
                partial=(
                    "Nothing held back appears here, but the console could not read part of this "
                    "run — this is not a statement that Screening held nothing back."
                ),
                truncated=bool(unreadable),
            )
        )

    # The flagged list is derived from the chunks, so it rides the CAPPED scan (audit R6): a
    # marker past the cap is simply absent from ``flagged``. The held-back list above rides the
    # capped sidecar read (audit R7) — both halves can therefore be shortened by the cap, and
    # both are shortened by an unreadable file. Only the *absence* claims differ: an empty list
    # can only ever mean "nothing was there" or "I could not look", never "the cap dropped it".
    parts.append(
        "<h2>Flagged — kept, with a risk marker in the corpus "
        f"({_count(len(flagged), incomplete)})</h2>"
    )
    if flagged:
        parts.append(
            _pager(
                run_id,
                "/quarantine",
                param="page",
                page=page,
                total_pages=flag_pages,
                total_items=len(flagged),
                noun="marker(s)",
                other={"held": str(held_page)},
            )
        )
        rows = []
        for e in shown_flagged:
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
        parts.append(
            _empty(
                "No record in this run's corpus carries a risk marker.",
                partial=(
                    "No risk marker was among the records this view read, and this view did not "
                    "read the whole run — so this is not a statement that the run's corpus "
                    "carries none. A RAG consumer must filter on the corpus files, not on this "
                    "page."
                ),
                truncated=incomplete,
            )
        )
    return _page(f"Held back & flagged — {run_id}", "".join(parts))


def render_chunks(
    run_id: str,
    rows: list[B.ChunkRow],
    *,
    page: int = 1,
    truncated: bool = False,
    unreadable: tuple[str, ...] = (),
) -> str:
    """The chunk list: find the chunk you want to interrogate. Paginated (audit R4)."""
    incomplete = truncated or bool(unreadable)
    parts = [
        _run_head(
            run_id,
            "Chunks",
            "Every retrieval-sized piece this run produced, with the source page and "
            "character offsets it occupies in its source document. Open one to see how it "
            "resolves back through Provenance.",
            "chunks",
        ),
        _truncation_notice(truncated, unreadable),
    ]
    if not rows:
        parts.append(
            _empty(
                "This run checkpointed no chunks.",
                partial=(
                    "No chunk was among the records this view read, and this view did not read "
                    "the whole run — so this is not a statement that the run checkpointed none."
                ),
                truncated=incomplete,
            )
        )
        return _page(f"Chunks — {run_id}", "".join(parts))
    shown, page, total_pages = _paginate(rows, page)
    body = []
    for row in shown:
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
    pager = _pager(
        run_id,
        "/chunks",
        param="page",
        page=page,
        total_pages=total_pages,
        total_items=len(rows),
        noun="chunk(s)",
    )
    parts.append(
        f'{pager}<div class="card"><table><thead><tr>'
        "<th>Source file</th><th>#</th><th>Page(s)</th><th>Characters</th><th>Tokens</th>"
        f"<th></th></tr></thead><tbody>{''.join(body)}</tbody></table></div>{pager}"
    )
    return _page(f"Chunks — {run_id}", "".join(parts))


def render_chunk(run_id: str, view: B.ChunkInspection) -> str:
    """The chunk inspector: where this chunk came from, and what was done to it."""
    parts = [
        _run_head(
            run_id,
            f"Chunk {view.index} of {view.file_name}",
            "Resolved back through Provenance to its source document, page and character offsets.",
            "chunks",
        )
    ]

    doc_state = (
        f"parsed ({view.document_pages} page(s), {view.document_chars:,} characters)"
        if view.document_parsed
        and view.document_pages is not None
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
                ["Chunk length", f"{view.character_count:,} characters, {view.token_count} tokens"],
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
    tone = (
        "bad" if view.redaction.control_disabled else ("good" if view.redaction.applied else "warn")
    )
    parts.append(
        f'<div class="card verdict {tone}"><div class="t">Redaction</div>'
        f'<p class="h">{escape(view.redaction.summary)}</p></div>'
    )
    if view.redaction.spans:
        parts.append(
            '<div class="card">'
            + _table(
                [
                    "PII type",
                    "Detector",
                    "Action",
                    "Placeholder",
                    "In this chunk",
                    "In the source document",
                    "Page(s)",
                    "Confidence",
                ],
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
                [[m.category, m.detector, f"{m.score:.2f}", m.reason] for m in view.risk_markers],
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


def render_graph(
    run_id: str,
    entities: list[B.EntityView],
    edges: list[B.EdgeView],
    *,
    page: int = 1,
    entity_page: int = 1,
    truncated: bool = False,
    unreadable: tuple[str, ...] = (),
) -> str:
    """The KG browser: entities to navigate, and every edge with its Evidence.

    The two lists paginate independently (``?entities=`` and ``?page=``) because an edge card is
    an order of magnitude larger than an entity row — one shared pager would make the page size
    depend on which of the two happened to be long.
    """
    incomplete = truncated or bool(unreadable)
    shown_entities, entity_page, entity_pages = _paginate(entities, entity_page)
    shown_edges, page, edge_pages = _paginate(edges, page)
    parts = [
        _run_head(
            run_id,
            "Knowledge graph",
            "The canonical entities this run resolved and the relations between them. "
            "Every edge is shown with the Evidence that justifies it — the source mentions, "
            "where they sit in their documents, and how strongly.",
            "graph",
        ),
        _truncation_notice(truncated, unreadable),
        f"<h2>Entities ({_count(len(entities), incomplete)})</h2>",
        _pager(
            run_id,
            "/graph",
            param="entities",
            page=entity_page,
            total_pages=entity_pages,
            total_items=len(entities),
            noun="entities",
            other={"page": str(page)},
        )
        if entities
        else "",
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
            for e in shown_entities
        )
        parts.append(
            '<div class="card"><table><thead><tr><th>Entity</th><th>Type</th><th>KB link</th>'
            "<th>Confidence</th><th>Mentions</th><th>Documents</th><th>Edges</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
    else:
        parts.append(
            _empty(
                "This run assembled no graph nodes.",
                partial=(
                    "No graph node was among the records this view read, and this view did not "
                    "read the whole run — so this is not a statement that the run assembled "
                    "none."
                ),
                truncated=incomplete,
            )
        )

    parts.append(f"<h2>Relations ({_count(len(edges), incomplete)})</h2>")
    if edges:
        edge_pager = _pager(
            run_id,
            "/graph",
            param="page",
            page=page,
            total_pages=edge_pages,
            total_items=len(edges),
            noun="relation(s)",
            other={"entities": str(entity_page)},
        )
        parts.append(edge_pager)
        parts.extend(_edge_card(run_id, edge) for edge in shown_edges)
        parts.append(edge_pager)
    else:
        parts.append(
            _empty(
                "This run assembled no graph edges.",
                partial=(
                    "No graph edge was among the records this view read, and this view did not "
                    "read the whole run — so this is not a statement that the run assembled "
                    "none."
                ),
                truncated=incomplete,
            )
        )
    return _page(f"Knowledge graph — {run_id}", "".join(parts))


def render_entity(
    run_id: str,
    entity: B.EntityView,
    edges: list[B.EdgeView],
    *,
    page: int = 1,
    truncated: bool = False,
    unreadable: tuple[str, ...] = (),
) -> str:
    """One entity, and every edge touching it — each with its Evidence.

    Carries the same ``Partial view`` notice as the graph list (audit R5): this page's relations
    come from the same capped scan, so a truncated run must not be shown here as a whole one.

    The notice alone was not enough. Under a capped scan an entity whose every edge sits beyond
    the cap still rendered ``Relations (0)`` over the flat sentence "No relation in this graph
    touches this entity" — an affirmative claim of isolation that the graph checkpoints
    contradict. Both the tally and the empty state are now qualified when ``truncated``.
    """
    incomplete = truncated or bool(unreadable)
    shown_edges, page, edge_pages = _paginate(edges, page)
    parts = [
        _run_head(
            run_id,
            entity.canonical_name,
            "One canonical entity and every relation that touches it.",
            "graph",
        ),
        _truncation_notice(truncated, unreadable),
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
        f"<h2>Relations ({_count(len(edges), incomplete)})</h2>",
    ]
    if edges:
        pager = _pager(
            run_id,
            "/entity",
            param="page",
            page=page,
            total_pages=edge_pages,
            total_items=len(edges),
            noun="relation(s)",
            other={"id": entity.node_id},
        )
        parts.append(pager)
        parts.extend(_edge_card(run_id, edge) for edge in shown_edges)
        parts.append(pager)
    else:
        parts.append(
            _empty(
                "No relation in this graph touches this entity.",
                partial=(
                    "No relation touching this entity was among the relations this view read, "
                    "and this view did not read the whole run, so this is NOT a statement that "
                    "the entity is isolated — relations the reader never reached are not "
                    "listed. Query the graph checkpoints directly to settle it."
                ),
                truncated=incomplete,
            )
        )
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


def render_wrong_host() -> str:
    body = (
        "<h1>Not this machine</h1>"
        '<div class="card"><p>This console answers only requests addressed to this machine '
        "(<code>localhost</code> or <code>127.0.0.1</code>, on the port it was started on). "
        "The request that reached it named a different host, so it was refused before anything "
        "was read.</p></div>"
    )
    return _page("Not this machine", body)


def _tone_for(status: RunStatus) -> str:
    return {
        RunStatus.PASSED: "good",
        RunStatus.ATTENTION: "warn",
        RunStatus.FAILED: "bad",
        RunStatus.UNFINISHED: "warn",
    }[status]
