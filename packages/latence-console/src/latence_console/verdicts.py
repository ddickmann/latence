"""Quality Report → headline verdicts a non-engineer can read.

The Quality Report (ADR-0008) is a rich, machine-readable artifact — nineteen schema
versions of counts, histograms, and per-Stage bookkeeping. Dumping it at a knowledge
manager is the failure mode this console exists to avoid. This module is the translation
layer: it turns the report's numbers into a short list of **sentences with a verdict
attached**, plus the drill-down tables behind them.

Every verdict is derived from the report's own counts. Nothing here reads a stored
boolean that merely *records* an intent — ``contracts_complete`` is a mirror of the
completeness detail, so the verdict is computed from the detail and can never disagree
with the numbers shown beneath it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from latence_core.quality import QualityReport


class Tone(Enum):
    """How a verdict should read: settled, worth a look, or a problem."""

    GOOD = "good"
    WARN = "warn"
    BAD = "bad"


@dataclass(frozen=True)
class Verdict:
    """One headline check: a plain-language claim, its tone, and the evidence for it."""

    title: str
    tone: Tone
    headline: str
    evidence: str = ""


@dataclass(frozen=True)
class Facts:
    """A titled drill-down table: label → already-formatted value."""

    title: str
    rows: list[tuple[str, str]] = field(default_factory=list)


def _pct(part: int, whole: int) -> str:
    if whole <= 0:
        return "n/a"
    return f"{100.0 * part / whole:.1f}%"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def completeness_verdict(report: QualityReport) -> Verdict:
    c = report.completeness
    if c.records_checked == 0:
        return Verdict(
            title="Traceability",
            tone=Tone.WARN,
            headline="This run produced no records to check.",
            evidence="Nothing reached the end of the pipeline, so there is nothing to trace.",
        )
    if c.passed:
        return Verdict(
            title="Traceability",
            tone=Tone.GOOD,
            headline=(
                f"Every one of the {c.records_checked} pieces of output can be traced back "
                "to the exact page it came from."
            ),
            evidence=(
                f"{c.records_with_provenance}/{c.records_checked} carry their source, "
                f"{c.records_with_classification}/{c.records_checked} carry a sensitivity "
                f"label, and {c.records_with_aligned_offsets}/{c.offset_bearing_records} "
                "point at a valid location in the original document."
            ),
        )
    missing = c.records_checked - min(c.records_with_provenance, c.records_with_classification)
    unaligned = c.offset_bearing_records - c.records_with_aligned_offsets
    return Verdict(
        title="Traceability",
        tone=Tone.BAD,
        headline=(
            "Some output cannot be traced back to its source document — do not treat this "
            "corpus as audit-ready."
        ),
        evidence=(
            f"{missing} of {c.records_checked} records are missing a source or a sensitivity "
            f"label, and {unaligned} point at a location that does not resolve."
        ),
    )


def stage_verdict(report: QualityReport) -> Verdict | None:
    failed = [s for s in report.stages if s.error_count]
    skipped = [s for s in report.stages if s.skipped]
    if failed:
        names = ", ".join(f"{s.name} ({s.error_category})" for s in failed)
        return Verdict(
            title="Processing steps",
            tone=Tone.BAD,
            headline=f"{_plural(len(failed), 'step')} of the pipeline failed.",
            evidence=f"Failed: {names}.",
        )
    if skipped:
        names = ", ".join(s.name for s in skipped)
        return Verdict(
            title="Processing steps",
            tone=Tone.WARN,
            headline=(
                f"{_plural(len(skipped), 'step')} was skipped because this machine could not "
                "run it."
            ),
            evidence=f"Skipped: {names}. The output is missing whatever those steps add.",
        )
    if not report.stages:
        return None
    return Verdict(
        title="Processing steps",
        tone=Tone.GOOD,
        headline=f"All {len(report.stages)} pipeline steps completed.",
        evidence=f"Total time {report.total_duration_seconds:.1f} seconds.",
    )


def screening_verdict(report: QualityReport) -> Verdict | None:
    s = report.screening
    if s is None:
        return None
    if s.quarantined == 0 and s.flagged == 0:
        return Verdict(
            title="Unsafe input",
            tone=Tone.GOOD,
            headline=(
                "No dangerous or suspicious files were found among "
                f"{s.total_screened} checks."
            ),
        )
    return Verdict(
        title="Unsafe input",
        tone=Tone.GOOD,
        headline=(
            f"{_plural(s.quarantined, 'file')} held back as unsafe, "
            f"{_plural(s.flagged, 'item')} flagged for review."
        ),
        evidence=(
            "Held-back files never reached the corpus. This is the screen working, "
            "not a failure — open the run to see why each one was held."
        ),
    )


def redaction_verdict(report: QualityReport) -> Verdict | None:
    r = report.redaction
    if r is None:
        return None
    if r.redaction_disabled_for_sensitive:
        return Verdict(
            title="Personal data",
            tone=Tone.BAD,
            headline=(
                f"{_plural(r.redaction_disabled_for_sensitive, 'sensitive document')} were "
                "processed with personal-data detection switched off."
            ),
            evidence=(
                "A zero personal-data count is NOT a clean bill of health while this is "
                "above zero — the check did not run on those documents."
            ),
        )
    if r.pii_total == 0:
        return Verdict(
            title="Personal data",
            tone=Tone.GOOD,
            headline=f"No personal data was detected across {r.documents_redacted} documents.",
        )
    return Verdict(
        title="Personal data",
        tone=Tone.GOOD,
        headline=(
            f"{r.pii_total} pieces of personal data were found in {r.documents_with_pii} "
            "documents and handled by policy."
        ),
        evidence="Actions taken: "
        + ", ".join(f"{action} ×{count}" for action, count in sorted(r.per_action.items()))
        + ".",
    )


def graph_verdict(report: QualityReport) -> Verdict | None:
    g = report.graph
    if g is None:
        return None
    if g.edges_total == 0:
        return Verdict(
            title="Knowledge graph",
            tone=Tone.WARN,
            headline="No relationships were found, so the knowledge graph is empty.",
            evidence=f"{g.nodes_total} things were identified but nothing connected them.",
        )
    coverage = _pct(g.edges_with_evidence, g.edges_total)
    tone = Tone.GOOD if g.edges_with_evidence == g.edges_total else Tone.WARN
    headline = (
        f"The knowledge graph holds {g.nodes_total} things connected by {g.edges_total} "
        f"relationships, {coverage} of which quote the sentence they came from."
    )
    return Verdict(
        title="Knowledge graph",
        tone=tone,
        headline=headline,
        evidence=(
            f"{g.nodes_linked} of {g.nodes_total} were matched to a known reference entry."
        ),
    )


def goldset_verdict(report: QualityReport) -> Verdict | None:
    g = report.goldset
    if g is None or not g.stages:
        return None
    parts = ", ".join(
        f"{s.stage} {100 * s.f1:.0f}% accurate" for s in sorted(g.stages, key=lambda s: s.stage)
    )
    worst = min(s.f1 for s in g.stages)
    return Verdict(
        title="Measured against your own answers",
        tone=Tone.GOOD if worst >= 0.8 else Tone.WARN,
        headline=f"Scored on your {g.documents_annotated} hand-annotated documents: {parts}.",
        evidence=(
            f"{g.documents_matched} of {g.documents_annotated} annotated documents produced "
            "output to score against."
        ),
    )


def delta_verdict(report: QualityReport) -> Verdict | None:
    d = report.delta
    if d is None:
        return None
    changed = d.documents_added + d.documents_updated + d.documents_deleted
    if changed == 0:
        return Verdict(
            title="What changed",
            tone=Tone.GOOD,
            headline="Nothing changed since the previous version — the corpus is up to date.",
        )
    return Verdict(
        title="What changed",
        tone=Tone.GOOD,
        headline=(
            f"{d.documents_added} documents added, {d.documents_updated} updated and "
            f"{d.documents_deleted} removed since the previous version."
        ),
        evidence=(
            f"{d.documents_extracted} documents were re-read and {d.documents_reused} were "
            f"reused unchanged; the knowledge graph gained {d.edges_added} relationships and "
            f"lost {d.edges_retracted}."
        ),
    )


def headline_verdicts(report: QualityReport) -> list[Verdict]:
    """The short list a stakeholder reads first — most consequential check first.

    Only checks the run actually made appear: a pipeline with no Redaction Stage gets no
    personal-data verdict rather than a reassuring "0 found" the run never substantiated.
    """
    candidates = (
        stage_verdict(report),
        completeness_verdict(report),
        redaction_verdict(report),
        screening_verdict(report),
        graph_verdict(report),
        goldset_verdict(report),
        delta_verdict(report),
    )
    verdicts = [v for v in candidates if v is not None]
    order = {Tone.BAD: 0, Tone.WARN: 1, Tone.GOOD: 2}
    return sorted(verdicts, key=lambda v: order[v.tone])


def stage_rows(report: QualityReport) -> list[tuple[str, str, str, str, str]]:
    """The per-Stage drill-down: (step, what it does, in→out, time, outcome)."""
    rows: list[tuple[str, str, str, str, str]] = []
    for s in report.stages:
        if s.error_count:
            outcome = f"failed ({s.error_category})"
        elif s.skipped:
            outcome = f"skipped ({s.skip_reason or 'unavailable on this machine'})"
        elif s.from_checkpoint:
            outcome = "reused from checkpoint"
        else:
            outcome = "ok"
        rows.append(
            (
                s.name,
                f"{s.capability} via {s.provider}",
                f"{s.records_in} → {s.records_out}",
                f"{s.duration_seconds:.2f}s",
                outcome,
            )
        )
    return rows


def fact_tables(report: QualityReport) -> list[Facts]:
    """The numbers behind the verdicts, grouped and pre-formatted (never raw JSON)."""
    tables: list[Facts] = []
    if report.parse is not None:
        p = report.parse
        tables.append(
            Facts(
                "Documents read",
                [
                    ("Documents parsed", str(p.documents_parsed)),
                    ("Documents that could not be read", str(p.parse_errors)),
                    ("Pages", str(p.pages_total)),
                    ("Page positions recovered by fuzzy match", str(p.page_drift_recovered)),
                ],
            )
        )
    if report.chunk is not None:
        c = report.chunk
        tables.append(
            Facts(
                "Retrieval corpus",
                [
                    ("Passages", str(c.chunks_total)),
                    ("Passages that resolve to a source page", f"{c.chunks_offset_preserving}"),
                    ("Tokens", str(c.tokens_total)),
                    ("Passages flagged as risky", str(c.chunks_flagged)),
                ],
            )
        )
    if report.entities is not None:
        e = report.entities
        top = sorted(e.per_type.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        tables.append(
            Facts(
                "Things found",
                [("Mentions", str(e.mentions_total))]
                + [(f"of type {name}", str(count)) for name, count in top],
            )
        )
    if report.relations is not None:
        r = report.relations
        top = sorted(r.per_type.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        tables.append(
            Facts(
                "Relationships found",
                [("Relationships", str(r.relations_total))]
                + [(f"of type {name}", str(count)) for name, count in top],
            )
        )
    if report.disambiguation is not None:
        d = report.disambiguation
        tables.append(
            Facts(
                "Duplicate resolution",
                [
                    ("Mentions considered", str(d.mentions_in)),
                    ("Distinct things after merging", str(d.entities_out)),
                    ("Merged as duplicates", str(d.entities_merged)),
                    ("Matched to a reference entry", str(d.entities_linked)),
                    ("Not matched to any reference entry", str(d.entities_unlinked)),
                ],
            )
        )
    if report.graph is not None:
        g = report.graph
        top_edges = sorted(g.edges_by_label.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        tables.append(
            Facts(
                "Knowledge-graph statistics",
                [
                    ("Nodes", str(g.nodes_total)),
                    ("Relationships", str(g.edges_total)),
                    (
                        "Relationships quoting their source",
                        _pct(g.edges_with_evidence, g.edges_total),
                    ),
                    ("Nodes matched to a reference entry", str(g.nodes_linked)),
                ]
                + [(f"relationship: {label}", str(count)) for label, count in top_edges],
            )
        )
    if report.screening is not None:
        s = report.screening
        tables.append(
            Facts(
                "Input screening",
                [
                    ("Checks run", str(s.total_screened)),
                    ("Held back as unsafe", str(s.quarantined)),
                    ("Flagged for review", str(s.flagged)),
                    ("Passed", str(s.passed)),
                ],
            )
        )
    if report.delta is not None:
        churn = report.delta
        tables.append(
            Facts(
                "What changed since the previous version",
                [
                    (
                        "Previous version",
                        "none" if churn.from_version is None else str(churn.from_version),
                    ),
                    ("This version", str(churn.to_version)),
                    ("Documents added", str(churn.documents_added)),
                    ("Documents updated", str(churn.documents_updated)),
                    ("Documents removed", str(churn.documents_deleted)),
                    ("Documents unchanged", str(churn.documents_unchanged)),
                    ("Relationships added", str(churn.edges_added)),
                    ("Relationships withdrawn", str(churn.edges_retracted)),
                    ("Records permanently erased", str(churn.records_purged)),
                    (
                        "Rebuild scope",
                        "full reconciliation"
                        if churn.reconciliation_triggered
                        else "incremental (only affected records recomputed)",
                    ),
                ],
            )
        )
    return tables


def screening_reasons(report: QualityReport) -> list[tuple[str, str, str, str]]:
    """Why input was held back or flagged: (when, decision, category, reason).

    Deduplicated and sorted so the table is deterministic. ``PASS`` findings are omitted —
    the reader wants the exceptions, and the pass count is already in the facts table. The
    document *id* is deliberately not shown here: it is a content hash, illegible to the
    audience this console serves, and per-document browsing is the follow-up slice (#188).
    """
    s = report.screening
    if s is None:
        return []
    seen = {
        (f.checkpoint.value, f.verdict.value, f.category, f.reason)
        for f in s.findings
        if f.verdict.value != "pass"
    }
    return sorted(seen)
