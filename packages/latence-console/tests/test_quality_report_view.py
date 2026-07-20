"""The Quality Report view: headline verdicts a non-engineer can read (#186 AC 3 + AC 5)."""

from __future__ import annotations

import json
import re

from latence_console import render
from latence_console.server import ConsoleApp
from latence_console.verdicts import Tone, fact_tables, headline_verdicts, stage_rows
from latence_core.storage import Storage
from reports import delta_report, failed_report, gapped_report, healthy_report


def _page(populated: str, run_id: str) -> str:
    app = ConsoleApp.over_storage(Storage.for_uri(populated), populated)
    response = app.respond("GET", f"/run/{run_id}")
    assert response.status == 200
    return response.body.decode("utf-8")


def test_the_page_is_not_a_json_dump(populated: str) -> None:
    """The report's raw JSON must not appear on the page — that is the whole point."""
    page = _page(populated, "run-0001")
    raw = healthy_report("run-0001").to_json()
    assert raw not in page
    # No bare report keys either: the reader sees sentences, not field names.
    for machine_key in ('"records_with_provenance"', '"edges_with_evidence"', '"schema_version"'):
        assert machine_key not in page


def test_headline_verdicts_are_plain_sentences_with_numbers(populated: str) -> None:
    page = _page(populated, "run-0001")
    assert "can be traced back to the exact page it came from" in page
    assert "knowledge graph holds 6 things connected by 33 relationships" in page
    assert "1 file held back as unsafe, 1 item flagged for review." in page


def test_per_stage_metrics_are_rendered(populated: str) -> None:
    page = _page(populated, "run-0001")
    assert "Pipeline steps" in page
    for expected in ("parse", "parser.plaintext", "4 → 4", "graph.builder"):
        assert expected in page


def test_knowledge_graph_statistics_are_rendered(populated: str) -> None:
    page = _page(populated, "run-0001")
    assert "Knowledge-graph statistics" in page
    assert "works_for" in page
    assert "100.0%" in page  # every relationship quotes its source


def test_completeness_checks_are_rendered(populated: str) -> None:
    page = _page(populated, "run-0001")
    assert "82" in page
    assert "carry a sensitivity" in page


def test_a_completeness_gap_reads_as_a_warning_not_a_pass(populated: str) -> None:
    page = _page(populated, "run-0002")
    assert "Needs attention" in page
    assert "do not treat this corpus as audit-ready" in page
    assert "Every one of the" not in page


def test_a_failed_stage_is_named_with_its_category(populated: str) -> None:
    page = _page(populated, "run-0003")
    assert "Failed" in page
    assert "graph (provider)" in page
    assert "1 step of the pipeline failed." in page


def test_delta_run_shows_what_changed(populated: str) -> None:
    page = _page(populated, "run-0004")
    assert "2 documents added, 1 updated and 1 removed since the previous version" in page
    assert "What changed since the previous version" in page
    assert "incremental (only affected records recomputed)" in page


def test_unfinished_run_says_so_instead_of_guessing(populated: str) -> None:
    page = _page(populated, "run-0005")
    assert "Unfinished" in page
    assert "never wrote a Quality Report" in page


def test_counts_read_grammatically_in_the_singular() -> None:
    """A one-file, one-item run must not read '1 files were ...' — this page is prose."""
    report = healthy_report()
    assert report.screening is not None
    one = report.model_copy(
        update={"screening": report.screening.model_copy(update={"quarantined": 1, "flagged": 1})}
    )
    verdict = next(v for v in headline_verdicts(one) if v.title == "Unsafe input")
    assert verdict.headline == "1 file held back as unsafe, 1 item flagged for review."

    many = report.model_copy(
        update={"screening": report.screening.model_copy(update={"quarantined": 3, "flagged": 2})}
    )
    verdict = next(v for v in headline_verdicts(many) if v.title == "Unsafe input")
    assert verdict.headline == "3 files held back as unsafe, 2 items flagged for review."


def test_worst_verdict_is_listed_first() -> None:
    verdicts = headline_verdicts(gapped_report())
    assert verdicts[0].tone is Tone.BAD
    tones = [v.tone for v in verdicts]
    assert tones == sorted(tones, key=lambda t: {Tone.BAD: 0, Tone.WARN: 1, Tone.GOOD: 2}[t])


def test_only_checks_the_run_actually_made_are_claimed() -> None:
    """A pipeline with no Redaction Stage must NOT get a reassuring 'no personal data' verdict."""
    report = healthy_report().model_copy(update={"redaction": None, "graph": None})
    titles = {v.title for v in headline_verdicts(report)}
    assert "Personal data" not in titles
    assert "Knowledge graph" not in titles
    assert "Traceability" in titles


def test_redaction_disabled_on_a_sensitive_document_is_loud() -> None:
    report = healthy_report()
    assert report.redaction is not None
    silenced = report.model_copy(
        update={
            "redaction": report.redaction.model_copy(
                update={"redaction_disabled_for_sensitive": 2}
            )
        }
    )
    verdict = next(v for v in headline_verdicts(silenced) if v.title == "Personal data")
    assert verdict.tone is Tone.BAD
    assert "NOT a clean bill of health" in verdict.evidence


def test_screening_reasons_never_leak_the_document_hash(populated: str) -> None:
    page = _page(populated, "run-0001")
    assert "zip compression ratio 966.9x exceeds the 50.0x cap" in page
    assert "sha256:" not in page


def test_pages_are_fully_offline(populated: str) -> None:
    """No CDN, no font host, no beacon — the console cannot phone home (#186 AC 4)."""
    app = ConsoleApp.over_storage(Storage.for_uri(populated), populated)
    for path in ("/", "/run/run-0001", "/run/run-0004", "/run/nope"):
        page = app.respond("GET", path).body.decode("utf-8")
        assert render.external_references(page) == [], path
        assert "<script" not in page.lower()


def test_everything_user_derived_is_escaped(storage: Storage, storage_uri: str) -> None:
    from reports import write_run  # noqa: PLC0415 - test helper

    hostile = healthy_report(run_id="run-x", pipeline_name='<img src=x onerror="alert(1)">')
    write_run(storage, storage_uri, "run-x", hostile)
    app = ConsoleApp.over_storage(storage, storage_uri)
    for path in ("/", "/run/run-x"):
        page = app.respond("GET", path).body.decode("utf-8")
        assert "<img src=x" not in page
        assert "&lt;img src=x" in page


def test_stage_rows_report_skips_honestly() -> None:
    report = healthy_report()
    skipped = report.model_copy(
        update={
            "stages": [
                report.stages[0].model_copy(
                    update={"skipped": True, "skip_reason": "no GPU on this host"}
                )
            ]
        }
    )
    assert stage_rows(skipped)[0][4] == "skipped (no GPU on this host)"
    verdict = next(v for v in headline_verdicts(skipped) if v.title == "Processing steps")
    assert verdict.tone is Tone.WARN


def test_fact_tables_are_preformatted_strings_not_raw_objects() -> None:
    for table in fact_tables(delta_report()):
        for label, value in table.rows:
            assert isinstance(label, str)
            assert isinstance(value, str)
            assert not re.search(r"[{}\[\]]", value)


def test_render_is_deterministic(populated: str) -> None:
    app = ConsoleApp.over_storage(Storage.for_uri(populated), populated)
    assert app.respond("GET", "/run/run-0001").body == app.respond(
        "GET", "/run/run-0001"
    ).body


def test_failed_report_json_is_still_readable_from_storage(populated: str) -> None:
    """The console renders, it never replaces: the machine-readable artifact is untouched."""
    storage = Storage.for_uri(populated)
    raw = storage.read_text(f"{populated}/_latence/runs/run-0003/quality-report.json")
    assert json.loads(raw) == json.loads(failed_report("run-0003").to_json())
