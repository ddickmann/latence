"""The stack-validation harness (P3-F4, ADR-0036 §3): the committed stacks + ``latence stack validate``.

Covers the Gate-bar DoD for the slice:

* the committed ``stacks/default.yaml`` + ``stacks/gpu-sota.yaml`` are valid Pipeline configs;
* ``latence stack validate stacks/default.yaml`` runs the full spine end-to-end on the bundled
  corpus and every check PASSes (contracts/KG/RAG/report + determinism + resume + graceful-failure
  + device-honesty), deterministically (``LATENCE_CUDA=0``);
* the harness GENUINELY BITES — a deliberately-broken stack (wrong wiring; and a redaction-bypass
  that leaks PII into the RAG corpus) makes it FAIL with a non-zero CLI exit;
* on this CPU host ``gpu-sota.yaml`` reports its learned Stages skipped-with-flag (never faked),
  still validates the CPU-reachable spine, and exits 0.

Seeded/deterministic: ``LATENCE_CUDA`` is pinned so the device seam's decision is fixed, and the
repo-root ``conftest`` pins ``PYTHONHASHSEED``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from latence_core.capability import ProviderRegistry
from latence_core.cli import app
from latence_core.pipeline import Pipeline
from latence_core.stacks.validate import (
    CheckResult,
    _cpu_reachable_subpipeline,
    load_stack,
    validate_stack,
)
from typer.testing import CliRunner

# packages/latence-core/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_STACKS = _REPO_ROOT / "stacks"
_DEFAULT = _STACKS / "default.yaml"
_GPU_SOTA = _STACKS / "gpu-sota.yaml"
_GPU_OCR = _STACKS / "gpu-ocr.yaml"

cli = CliRunner()


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the device seam to CPU-only so the skip-with-flag posture is deterministic in CI."""
    monkeypatch.setenv("LATENCE_CUDA", "0")


def _checks_by_name(result_checks: list[CheckResult]) -> dict[str, CheckResult]:
    return {c.name: c for c in result_checks}


# -- The committed stacks are valid Pipeline configs -------------------------


def test_committed_stacks_are_valid_pipelines(tmp_path: Path) -> None:
    """Both committed stacks parse (with placeholders filled) into a valid Pipeline."""
    store = f"file://{tmp_path}/store"
    corpus = f"file://{tmp_path}/corpus"
    for stack in (_DEFAULT, _GPU_SOTA):
        pipeline = load_stack(stack, storage_uri=store, corpus_dir=corpus)
        assert isinstance(pipeline, Pipeline)
        assert pipeline.stages  # non-empty DAG
        # The placeholders were substituted (no residual template tokens).
        assert "{STORAGE_URI}" not in pipeline.storage_uri
        assert all("{CORPUS_DIR}" not in str(s.config) for s in pipeline.stages)


# The gliner-family extractors run on mdeberta-v3 (768-token window). A stack feeding one must cap
# chunk.markdown at <= 640 chunker-tokens (~768 mdeberta-tokens, ~1.2× chunker→model) so the model
# never silently truncates. The provider `max_len` guard (default 768) is the hard backstop (W12).
_GLINER_FAMILY_PROVIDERS = frozenset(
    {
        "entity.gliner",
        "fused_entity_relation.gliner2",
        "relation.gliner_relex",
        "redaction.gliner_pii",
        "redaction.gliner2",
    }
)
_GLINER_SAFE_CHUNK_MAX_TOKENS = 640


def test_gliner_stacks_keep_chunk_within_the_mdeberta_window() -> None:
    """W12 stack-lint: every shipped stack that feeds a gliner-family extractor caps ``chunk.markdown``
    ``max_tokens`` at <= 640 (chunker tokens), keeping mdeberta-v3's 768 window un-truncated.
    """
    gliner_stacks_seen = 0
    for stack in sorted(_STACKS.glob("*.yaml")):
        pipeline = load_stack(stack, storage_uri="file:///tmp/x", corpus_dir="file:///tmp/c")
        providers = {s.provider for s in pipeline.stages}
        if not providers & _GLINER_FAMILY_PROVIDERS:
            continue
        gliner_stacks_seen += 1
        for chunk_stage in (s for s in pipeline.stages if s.provider == "chunk.markdown"):
            max_tokens = int(chunk_stage.config.get("max_tokens", 512))
            assert max_tokens <= _GLINER_SAFE_CHUNK_MAX_TOKENS, (
                f"{stack.name}: chunk.markdown max_tokens={max_tokens} exceeds the gliner-safe cap "
                f"({_GLINER_SAFE_CHUNK_MAX_TOKENS} chunker-tokens ~ 768 mdeberta-tokens); a longer "
                "chunk is silently truncated by the mdeberta window."
            )
    assert gliner_stacks_seen >= 1, "expected at least one shipped stack feeding a gliner extractor"


def test_default_stack_uses_only_offline_permissive_providers() -> None:
    """``default.yaml`` sneaks in NO learned/endpoint Provider (fully offline + deterministic)."""
    pipeline = load_stack(_DEFAULT, storage_uri="file:///tmp/x", corpus_dir="file:///tmp/c")
    providers = {s.provider for s in pipeline.stages}
    # None of the GPU/endpoint Providers appears in the blessed default.
    forbidden = {
        "entity.gliner",
        "relation.gliner_relex",
        "redaction.gliner_pii",
        "embedding.endpoint",
        "parser.endpoint",
        "relation.llm",
        "entity.endpoint",
    }
    intruders = providers & forbidden
    assert not intruders, f"default stack pulled in a non-offline provider: {intruders}"


# -- End-to-end: the default stack validates green ---------------------------


def test_validate_default_stack_all_checks_pass(tmp_path: Path) -> None:
    result = validate_stack(_DEFAULT, storage_root=f"file://{tmp_path}/store")
    assert result.ok, result.render()
    assert result.document_count == 4  # 7 files staged; 3 dangerous quarantined at intake
    checks = _checks_by_name(result.checks)
    # Every substantive check genuinely PASSed (not skipped) on the full CPU spine.
    for name in ("contracts", "kg", "rag", "report", "graceful_failure", "determinism", "resume"):
        c = checks[name]
        assert c.passed and not c.skipped, f"{name} did not PASS: {c.detail}"
    assert not result.skipped_stages  # nothing skipped on an all-CPU stack
    # The signed-style summary was persisted.
    assert result.summary_uri is not None


def test_validate_default_stack_writes_summary(tmp_path: Path) -> None:
    from latence_core.storage import Storage

    root = f"file://{tmp_path}/store"
    result = validate_stack(_DEFAULT, storage_root=root)
    assert result.summary_uri is not None
    text = Storage.for_uri(root).read_text(result.summary_uri)
    assert "Stack validation" in text and "PASS" in text


# -- Device-honesty: gpu-sota on a CPU host ----------------------------------


def test_gpu_sota_skips_learned_stages_with_flag(tmp_path: Path) -> None:
    """On a CPU-only host the learned Stages are skipped-with-flag; the CPU spine still validates."""
    result = validate_stack(_GPU_SOTA, storage_root=f"file://{tmp_path}/store")
    assert result.ok, result.render()  # a skip is NOT a failure (device honesty)
    # The learned Providers were skipped-with-flag, never run, never faked. The SOTA path's
    # extraction is now ONE fused Stage (``extract``), not separate entity+relation Stages.
    assert "extract" in result.skipped_stages
    assert "redact" in result.skipped_stages
    checks = _checks_by_name(result.checks)
    # The CPU-reachable spine still proved contracts + graceful-failure.
    assert checks["contracts"].passed
    assert checks["graceful_failure"].passed
    # The KG/RAG/determinism/resume checks skip-clean (their producing Stages were skipped).
    assert checks["kg"].skipped
    assert checks["rag"].skipped


def test_gpu_ocr_skips_the_ocr_parser_with_flag(tmp_path: Path) -> None:
    """gpu-ocr's LightOn-OCR Parse (compute='gpu') is skipped-with-flag on a CPU host (W1-lighton).

    Parse is GPU-gated (``parser.lighton``, device: cuda) so on this CPU host it skips-with-flag and
    everything depending on it skips too; the CPU-reachable head (Source -> Intake Screening) still
    validates, and a skip is NOT a failure (device honesty — never a fabricated OCR run).
    """
    result = validate_stack(_GPU_OCR, storage_root=f"file://{tmp_path}/store")
    assert result.ok, result.render()
    assert "parse" in result.skipped_stages  # the OCR Parser skipped-with-flag, never run/faked
    checks = _checks_by_name(result.checks)
    assert checks["contracts"].passed  # the CPU-reachable head proved contracts
    # The RAG-corpus Export depends on the skipped Parse, so its check skips-clean.
    assert checks["rag"].skipped


def test_cpu_reachable_subpipeline_drops_gpu_and_dependents() -> None:
    """The reduction drops the learned Stages AND everything transitively depending on them."""
    declared = load_stack(_GPU_SOTA, storage_uri="file:///tmp/x", corpus_dir="file:///tmp/c")
    _reduced, dropped = _cpu_reachable_subpipeline(declared, ProviderRegistry())
    # Directly-skipped learned Stages (the single fused ``extract`` Stage + the learned Redactor):
    assert "extract" in dropped and "redact" in dropped
    # Transitively dropped (depend on a skipped Stage):
    assert "disambiguate" in dropped  # depends on the fused extract Stage
    assert "graph" in dropped  # depends on disambiguate
    assert "export_corpus" in dropped  # depends on redact
    # The CPU spine survives.
    assert "source" not in dropped
    assert "parse" not in dropped
    assert "chunk" not in dropped


# -- The harness BITES -------------------------------------------------------


def _broken_wiring_stack(tmp_path: Path) -> Path:
    """A stack with a Relation Stage that does NOT depend on an Entity Stage (wrong wiring)."""
    text = """
name: broken-wiring
storage_uri: "{STORAGE_URI}"
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {path: "{CORPUS_DIR}", sensitivity: internal, extensions: [txt, md, markdown, pdf, zip]}
  - name: intake_screen
    capability: intake_screening
    provider: screening.intake_signature
    depends_on: [source]
    config: {max_bytes: 2097152, max_zip_ratio: 50.0}
  - name: parse
    capability: parse
    provider: parser.pdfplumber
    depends_on: [intake_screen]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
    config: {max_tokens: 256, overlap_tokens: 32, min_tokens: 8}
  - name: content_screen
    capability: content_screening
    provider: screening.content_keyword
    depends_on: [chunk]
  - name: relations
    capability: relation_extraction
    provider: relation.pattern
    depends_on: [content_screen]
    config: {window_chars: 200, relations: {}}
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [relations]
    config: {basename: corpus}
"""
    p = tmp_path / "broken-wiring.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_broken_wiring_stack_fails_the_harness(tmp_path: Path) -> None:
    """A wrong-wiring stack makes the harness FAIL (not crash) — the harness bites."""
    stack = _broken_wiring_stack(tmp_path)
    result = validate_stack(stack, storage_root=f"file://{tmp_path}/store")
    assert not result.ok
    assert any(not c.passed for c in result.checks)


def _pii_leak_stack(tmp_path: Path) -> Path:
    """A stack that exports the corpus straight off ``chunk`` — bypassing Redaction — so the
    planted PII leaks into the RAG corpus and the ``rag`` check FAILs."""
    text = """
name: pii-leak
storage_uri: "{STORAGE_URI}"
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {path: "{CORPUS_DIR}", sensitivity: internal, extensions: [txt, md, markdown, pdf, zip]}
  - name: intake_screen
    capability: intake_screening
    provider: screening.intake_signature
    depends_on: [source]
    config: {max_bytes: 2097152, max_zip_ratio: 50.0}
  - name: parse
    capability: parse
    provider: parser.pdfplumber
    depends_on: [intake_screen]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
    config: {max_tokens: 256, overlap_tokens: 32, min_tokens: 8}
  - name: content_screen
    capability: content_screening
    provider: screening.content_keyword
    depends_on: [chunk]
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [content_screen]
    config: {basename: corpus}
"""
    p = tmp_path / "pii-leak.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_pii_leak_stack_fails_rag_check(tmp_path: Path) -> None:
    """A redaction-bypass stack leaks PII into the corpus → the ``rag`` check FAILs (a clean bite)."""
    stack = _pii_leak_stack(tmp_path)
    result = validate_stack(stack, storage_root=f"file://{tmp_path}/store")
    assert not result.ok
    rag = _checks_by_name(result.checks)["rag"]
    assert not rag.passed
    assert "PII" in rag.detail


# -- CLI: exit codes gate CI -------------------------------------------------


def test_cli_validate_default_exits_zero() -> None:
    result = cli.invoke(app, ["stack", "validate", str(_DEFAULT)])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_cli_validate_broken_exits_nonzero(tmp_path: Path) -> None:
    stack = _broken_wiring_stack(tmp_path)
    result = cli.invoke(app, ["stack", "validate", str(stack)])
    assert result.exit_code == 1, result.output


def test_cli_validate_missing_file_exits_two(tmp_path: Path) -> None:
    result = cli.invoke(app, ["stack", "validate", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2


# -- Hardening: PII scan covers the WHOLE export surface (not just JSONL) -----


def test_raw_pii_scan_catches_ttl_and_parquet_not_just_jsonl(tmp_path: Path) -> None:
    """The export PII scan must bite on KG artifacts (ttl/graphml) + Parquet, not only *.jsonl.

    Closes the audit gap: the KG branch bypasses Redaction, so a leak in an edge snippet (ttl) or a
    Parquet column would slip a JSONL-only scan. A clean export set returns None; a planted token in
    a *.ttl OR a *.parquet is found.
    """
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    from latence_core.stacks.validate import _PLANTED_PII, _raw_pii_in_exports
    from latence_core.storage import Storage

    token = _PLANTED_PII[0]
    root = f"file://{tmp_path}/store"
    storage = Storage.for_uri(root)
    export_dir = f"{root}/export"

    # Clean export set (masked surrogate only) → no leak.
    storage.write_atomic(f"{export_dir}/corpus.jsonl", b'{"provenance": {}, "text": "[EMAIL]"}\n')
    storage.write_atomic(f"{export_dir}/graph.ttl", b"<n1> <label> \"Acme\" .\n")
    assert _raw_pii_in_exports(storage, export_dir) is None

    # A leak in the KG ttl (text) is caught.
    storage.write_atomic(f"{export_dir}/graph.ttl", f'<n1> <snippet> "{token}" .\n'.encode())
    leak = _raw_pii_in_exports(storage, export_dir)
    assert leak is not None and token in leak and "graph.ttl" in leak

    # A leak hidden in a Parquet string column is caught (read via pyarrow, cells stringified).
    storage.write_atomic(f"{export_dir}/graph.ttl", b"<n1> <label> \"Acme\" .\n")  # clean again
    buf = io.BytesIO()
    pq.write_table(pa.table({"snippet": [f"contact {token}"]}), buf)  # type: ignore[no-untyped-call]
    storage.write_atomic(f"{export_dir}/graph-edges.parquet", buf.getvalue())
    leak2 = _raw_pii_in_exports(storage, export_dir)
    assert leak2 is not None and token in leak2 and "parquet" in leak2


# -- Hardening: a CPU provider that fails to load must FAIL, not skip-clean ---


def test_cpu_provider_load_failure_is_fatal_but_gpu_request_skips() -> None:
    """A load error on a CPU/auto stage returns None (Runner runs it → FAILs loud); only a stage that
    REQUESTED cuda/gpu is honestly skipped-with-flag. Prevents a broken default stack greening."""
    from types import SimpleNamespace

    from latence_core.stacks.validate import _skip_reason_for_stage

    class _RaisingRegistry:
        def names(self) -> list[str]:
            return ["entity.broken"]

        def profile(self, name: str) -> object:  # noqa: ARG002
            raise RuntimeError("simulated import/profile failure")

    reg = _RaisingRegistry()
    cpu_stage = SimpleNamespace(provider="entity.broken", config={"device": "auto"})
    gpu_stage = SimpleNamespace(provider="entity.broken", config={"device": "cuda"})
    # CPU/auto: NOT skipped (fatal) — the Runner will run it and fail loudly.
    assert _skip_reason_for_stage(cpu_stage, reg) is None  # type: ignore[arg-type]
    # cuda-requested: honestly skipped-with-flag (missing GPU dep).
    reason = _skip_reason_for_stage(gpu_stage, reg)  # type: ignore[arg-type]
    assert reason is not None and "entity.broken" in reason
