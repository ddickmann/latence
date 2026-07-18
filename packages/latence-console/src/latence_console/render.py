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

from latence_console import verdicts as V
from latence_console.runs import RunStatus

if TYPE_CHECKING:
    from datetime import datetime

    from latence_core.quality import QualityReport

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
