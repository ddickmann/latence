"""CLI tests for the incremental-delta commands: ``latence delta`` / ``retract`` / ``purge``.

The three commands expose the EXISTING :class:`~latence_core.delta_runner.DeltaRunner` (S11,
ADR-0029). These tests cover both ends:

* a real end-to-end delta / retract / purge over the in-core deterministic CPU Providers (the
  same harness the DeltaRunner e2e tests use), asserting the churn is printed and the Corpus
  Version advances — so the CLI is wired to the right DeltaRunner method and the report-printing
  path runs; and
* the argument surface with a monkeypatched DeltaRunner (``--doc`` repeatable + ``--docs-file``
  parse/merge/dedup, the auto run-id default, purge's ``--yes`` confirmation guard, empty-id and
  malformed-id usage errors, and typed-error → non-zero exit).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import latence_core.cli as cli
from latence_core.cli import app
from latence_core.contracts import DeletionMode
from latence_core.state.corpus_store import CorpusStore
from latence_core.storage import Storage
from typer.testing import CliRunner

runner = CliRunner()

_FAKE_DOC_A = "sha256:" + "a" * 64
_FAKE_DOC_B = "sha256:" + "b" * 64


def _delta_pipeline(store: str, docs_dir: Path) -> Path:
    """Write a full Source->…->Graph->Export pipeline YAML (the in-core CPU delta harness)."""
    path = docs_dir.parent / "pipeline.yaml"
    path.write_text(
        f"""
name: cli-delta
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs_dir}', sensitivity: internal, extensions: [txt]}}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: entities
    capability: entity_extraction
    provider: entity.gazetteer
    depends_on: [chunk]
    config:
      labels:
        ORG: {{terms: [Acme, International Business Machines, IBM]}}
        PERSON: {{terms: [Alice, Bob]}}
  - name: relations
    capability: relation_extraction
    provider: relation.pattern
    depends_on: [chunk, entities]
    config:
      relations:
        works_for: {{head: PERSON, tail: ORG, triggers: [works at, joined]}}
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [entities, relations]
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [graph]
""",
        encoding="utf-8",
    )
    return path


def _doc_id(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


# -- real end-to-end (the happy paths, exercising DeltaRunner + report printing) ---------------


def test_cli_delta_adds_and_commits_a_version(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    (docs / "a.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    pipeline = _delta_pipeline(store, docs)

    result = runner.invoke(app, ["delta", str(pipeline), "--run-id", "d0"])
    assert result.exit_code == 0, result.output
    # The churn is printed and the corpus advanced to its first Version.
    assert "Corpus Version none -> v0" in result.output
    assert "documents: +1" in result.output
    # The honest O(corpus) caveat is surfaced in the churn output.
    assert "issue #42" in result.output
    assert CorpusStore(Storage.for_uri(store), store).head_version() == 0

    # A second file -> a new Version committed by a second delta.
    (docs / "b.txt").write_text(
        "International Business Machines partnered with Acme.\n", encoding="utf-8"
    )
    result2 = runner.invoke(app, ["delta", str(pipeline), "--run-id", "d1"])
    assert result2.exit_code == 0, result2.output
    assert "-> v1" in result2.output
    assert CorpusStore(Storage.for_uri(store), store).head_version() == 1


def test_cli_retract_soft_deletes_and_commits(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    (docs / "a.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    (docs / "b.txt").write_text("Bob is here.\n", encoding="utf-8")
    pipeline = _delta_pipeline(store, docs)
    assert runner.invoke(app, ["delta", str(pipeline), "--run-id", "d0"]).exit_code == 0

    a_id = _doc_id(docs / "a.txt")
    result = runner.invoke(app, ["retract", str(pipeline), "--doc", a_id, "--run-id", "r0"])
    assert result.exit_code == 0, result.output
    assert "documents: +0 ~0 -1" in result.output
    assert "retracted" in result.output
    assert "(retraction)" in result.output
    assert CorpusStore(Storage.for_uri(store), store).head_version() == 1


def test_cli_purge_hard_deletes_with_yes(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    store = f"file://{tmp_path}/store"
    (docs / "a.txt").write_text("Alice works at Acme. IBM is here.\n", encoding="utf-8")
    (docs / "keep.txt").write_text("Bob is here.\n", encoding="utf-8")
    pipeline = _delta_pipeline(store, docs)
    assert runner.invoke(app, ["delta", str(pipeline), "--run-id", "d0"]).exit_code == 0

    a_id = _doc_id(docs / "a.txt")
    result = runner.invoke(app, ["purge", str(pipeline), "--doc", a_id, "--yes", "--run-id", "p0"])
    assert result.exit_code == 0, result.output
    assert "purged" in result.output
    assert "(purge)" in result.output
    # The physical GDPR erasure removed the on-disk source file; the co-resident file survives.
    assert not (docs / "a.txt").exists()
    assert (docs / "keep.txt").exists()


# -- argument surface (monkeypatched DeltaRunner — no real corpus needed) ----------------------


class _FakeRunner:
    """Records which DeltaRunner method the CLI called and with what arguments."""

    calls: list[tuple[str, str, list[str] | None]] = []

    def apply_delta(  # noqa: ANN401
        self, pipeline: Any, run_id: str, *, reinduce: bool = False
    ) -> Any:
        _FakeRunner.calls.append(("apply_delta", run_id, None))
        return object()

    def retract(self, pipeline: Any, run_id: str, document_ids: list[str]) -> Any:  # noqa: ANN401
        _FakeRunner.calls.append(("retract", run_id, list(document_ids)))
        return object()

    def purge(self, pipeline: Any, run_id: str, document_ids: list[str]) -> Any:  # noqa: ANN401
        _FakeRunner.calls.append(("purge", run_id, list(document_ids)))
        return object()


def _stub_runner(monkeypatch: Any) -> None:  # noqa: ANN401
    """Swap in the fake DeltaRunner + a no-op report emitter (isolate argument parsing)."""
    _FakeRunner.calls = []
    import latence_core.delta_runner as dr

    monkeypatch.setattr(dr, "DeltaRunner", _FakeRunner)
    monkeypatch.setattr(cli, "_emit_delta_report", lambda pipeline, run_id, report: None)


def _tiny_pipeline(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Acme is here.\n", encoding="utf-8")
    return _delta_pipeline(f"file://{tmp_path}/store", docs)


def test_cli_delta_default_run_id_is_fresh_timestamp(tmp_path: Path, monkeypatch: Any) -> None:  # noqa: ANN401
    _stub_runner(monkeypatch)
    pipeline = _tiny_pipeline(tmp_path)
    result = runner.invoke(app, ["delta", str(pipeline)])
    assert result.exit_code == 0, result.output
    assert len(_FakeRunner.calls) == 1
    method, run_id, _ = _FakeRunner.calls[0]
    assert method == "apply_delta"
    assert run_id.startswith("delta-")  # the auto UTC-timestamped id (a delta needs a fresh one)


def test_cli_retract_doc_flag_is_repeatable(tmp_path: Path, monkeypatch: Any) -> None:  # noqa: ANN401
    _stub_runner(monkeypatch)
    pipeline = _tiny_pipeline(tmp_path)
    result = runner.invoke(
        app,
        ["retract", str(pipeline), "--doc", _FAKE_DOC_A, "--doc", _FAKE_DOC_B, "--run-id", "r"],
    )
    assert result.exit_code == 0, result.output
    assert _FakeRunner.calls == [("retract", "r", [_FAKE_DOC_A, _FAKE_DOC_B])]


def test_cli_docs_file_is_parsed_and_merged_deduped(tmp_path: Path, monkeypatch: Any) -> None:  # noqa: ANN401
    _stub_runner(monkeypatch)
    pipeline = _tiny_pipeline(tmp_path)
    docs_file = tmp_path / "ids.txt"
    # blank lines + a comment + a duplicate of the --doc id are all handled.
    docs_file.write_text(
        f"# ids to purge\n{_FAKE_DOC_B}\n\n{_FAKE_DOC_A}\n", encoding="utf-8"
    )
    result = runner.invoke(
        app,
        ["purge", str(pipeline), "--doc", _FAKE_DOC_A, "--docs-file", str(docs_file), "--yes"],
    )
    assert result.exit_code == 0, result.output
    method, _, ids = _FakeRunner.calls[0]
    assert method == "purge"
    # --doc first, then file ids, de-duplicated preserving order.
    assert ids == [_FAKE_DOC_A, _FAKE_DOC_B]


def test_cli_purge_requires_confirmation_without_yes(tmp_path: Path, monkeypatch: Any) -> None:  # noqa: ANN401
    _stub_runner(monkeypatch)
    pipeline = _tiny_pipeline(tmp_path)
    # Answering "no" at the prompt aborts (non-zero) and never calls the DeltaRunner.
    result = runner.invoke(
        app, ["purge", str(pipeline), "--doc", _FAKE_DOC_A], input="n\n"
    )
    assert result.exit_code != 0
    assert _FakeRunner.calls == []
    # Answering "yes" proceeds to the purge.
    result_yes = runner.invoke(
        app, ["purge", str(pipeline), "--doc", _FAKE_DOC_A, "--run-id", "p"], input="y\n"
    )
    assert result_yes.exit_code == 0, result_yes.output
    assert _FakeRunner.calls == [("purge", "p", [_FAKE_DOC_A])]


def test_cli_retract_no_docs_is_a_usage_error(tmp_path: Path) -> None:
    pipeline = _tiny_pipeline(tmp_path)
    result = runner.invoke(app, ["retract", str(pipeline)])
    assert result.exit_code == 2
    assert "No document ids given" in result.output


def test_cli_retract_malformed_doc_id_is_a_usage_error(tmp_path: Path) -> None:
    pipeline = _tiny_pipeline(tmp_path)
    result = runner.invoke(app, ["retract", str(pipeline), "--doc", "not-a-doc-id"])
    assert result.exit_code == 2
    assert "do not look like document ids" in result.output


def test_cli_delta_missing_pipeline_is_a_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["delta", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2
    assert "Pipeline file not found" in result.output


def test_cli_retract_on_empty_corpus_is_a_typed_error(tmp_path: Path) -> None:
    # A real typed-error path: retracting before any Corpus Version exists raises inside the
    # DeltaRunner; the CLI turns it into a clean message + non-zero exit (never a traceback).
    pipeline = _tiny_pipeline(tmp_path)
    result = runner.invoke(
        app, ["retract", str(pipeline), "--doc", _FAKE_DOC_A, "--run-id", "r"]
    )
    assert result.exit_code == 1
    assert "retract failed" in result.output
    assert "empty corpus" in result.output


def test_delete_document_ids_helper_reuses_deletion_mode_enum() -> None:
    # A guard that the two deletion commands map to the DeletionMode the DeltaRunner distinguishes
    # (retraction is the soft default; purge is the hard erasure) — the churn output prints
    # ``deletion_mode.value``, so the strings the e2e tests assert on are these enum values.
    assert DeletionMode.RETRACTION.value == "retraction"
    assert DeletionMode.PURGE.value == "purge"
