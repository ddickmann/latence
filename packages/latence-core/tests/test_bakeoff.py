"""The bake-off / Provider Matrix harness (P3-F5, ADR-0036 §4): the Wave-0 capstone.

Covers the Gate-bar DoD for the slice:

* the committed ``matrix/parse.yaml`` + ``matrix/chunk.yaml`` are valid, real CPU stage matrices;
* the harness holds every OTHER Stage at ``stacks/default.yaml`` and swaps ONLY the target Stage
  (a real controlled comparison), running the SAME bundled corpus through each candidate;
* per-candidate quality / latency / memory (declared-vs-measured) / cost / license / determinism /
  error-rate are recorded, reading ``ProviderProfile.memory_mb`` / ``cost_per_1k`` /
  ``deterministic`` for real (the audit's inert-fields observation is retired);
* the determinism cross-check BITES — a Provider that declares ``deterministic=True`` but is not is
  flagged (``determinism.mismatch``);
* GPU/endpoint candidates are skipped-with-flag with NO fabricated numbers;
* the bake-off degrades per-candidate — a crashing candidate is a failed row, not an aborted matrix;
* ``latence bake-off <matrix>`` writes ``MATRIX-RESULTS-<stage>.md`` + a JSON companion;
* quality reuses the S10 gold-set scorer (a gold-set-bearing base stack surfaces P/R/F1).

Seeded/deterministic: ``LATENCE_CUDA`` is pinned so the device seam is a fixed CPU-only host, and
the repo-root ``conftest`` pins ``PYTHONHASHSEED``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from latence_core.bakeoff.harness import (
    BakeOffHarness,
    _quality_of,
    _resolve_stage_name,
    _swap_target_stage,
    run_bakeoff,
)
from latence_core.bakeoff.matrix import MatrixCandidate, MatrixSpec
from latence_core.bakeoff.report import CandidateResult, MatrixResult
from latence_core.capability import ProviderRegistry
from latence_core.cli import app
from latence_core.pipeline import CapabilityKind
from latence_core.providers.profile import REPO_LICENSE_CODE, ProviderProfile
from latence_core.stacks.validate import load_stack
from typer.testing import CliRunner

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord

# packages/latence-core/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_STACKS = _REPO_ROOT / "stacks"
_MATRIX = _REPO_ROOT / "matrix"
_DEFAULT = _STACKS / "default.yaml"
_PARSE_MATRIX = _MATRIX / "parse.yaml"
_CHUNK_MATRIX = _MATRIX / "chunk.yaml"
_ENTITY_MATRIX = _MATRIX / "entity.yaml"
_EMBED_MATRIX = _MATRIX / "embed.yaml"

cli = CliRunner()


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the device seam to CPU-only so the skip-with-flag posture is deterministic in CI."""
    monkeypatch.setenv("LATENCE_CUDA", "0")


def _by_provider(result: MatrixResult) -> dict[str, CandidateResult]:
    return {c.provider: c for c in result.candidates}


# -- The committed matrices are valid specs ----------------------------------


def test_committed_matrices_are_valid_specs() -> None:
    """Both shipped CPU matrices parse into a valid :class:`MatrixSpec` with real candidates."""
    parse = MatrixSpec.from_yaml_text(_PARSE_MATRIX.read_text(encoding="utf-8"))
    assert parse.stage is CapabilityKind.PARSE
    assert [c.provider for c in parse.candidates] == [
        "parser.plaintext",
        "parser.pdfplumber",
        "parser.document",
        "parser.lighton",  # the GPU OCR candidate (W1-lighton) — skipped-with-flag on CPU
        "parser.glm",  # the second GPU OCR candidate (W1-glm, MIT) — skipped-with-flag on CPU
        "parser.render",  # the universal any-file parser (W5) — skipped-with-flag on CPU
    ]
    chunk = MatrixSpec.from_yaml_text(_CHUNK_MATRIX.read_text(encoding="utf-8"))
    assert chunk.stage is CapabilityKind.CHUNK
    assert [c.provider for c in chunk.candidates] == ["chunk.markdown", "chunk.sentence_window"]


def test_matrix_rejects_duplicate_candidates() -> None:
    """A matrix that lists the same candidate twice is a spec error (no double-baking)."""
    with pytest.raises(ValueError, match="Duplicate candidate"):
        MatrixSpec.from_mapping(
            {
                "stage": "parse",
                "candidates": [
                    {"provider": "parser.plaintext"},
                    {"provider": "parser.plaintext"},
                ],
            }
        )


def test_matrix_rejects_unknown_key() -> None:
    """Strict contract: an unknown top-level key is a hard error, not a silently-ignored typo."""
    with pytest.raises(ValueError):  # noqa: PT011 - pydantic ValidationError is a ValueError
        MatrixSpec.from_mapping(
            {"stage": "parse", "candidates": [{"provider": "parser.plaintext"}], "typo": 1}
        )


# -- The controlled comparison: only the target Stage is swapped -------------


def test_swap_target_stage_changes_only_the_target(tmp_path: Path) -> None:
    """Swapping a candidate leaves every OTHER Stage byte-identical (the controlled comparison)."""
    base = load_stack(_DEFAULT, storage_uri=f"file://{tmp_path}/s", corpus_dir=f"file://{tmp_path}/c")
    swapped = _swap_target_stage(base, "parse", "parser.plaintext", {})
    for original, new in zip(base.stages, swapped.stages, strict=True):
        if new.name == "parse":
            assert new.provider == "parser.plaintext"
            assert original.provider != "parser.plaintext"  # it really changed
        else:
            # Every other stage is untouched — provider, config, and DAG edges identical.
            assert new.model_dump() == original.model_dump()


def test_candidate_config_merges_onto_base(tmp_path: Path) -> None:
    """A candidate's config override merges onto the base stage's config (inherits the rest)."""
    base = load_stack(_DEFAULT, storage_uri=f"file://{tmp_path}/s", corpus_dir=f"file://{tmp_path}/c")
    swapped = _swap_target_stage(base, "chunk", "chunk.sentence_window", {"max_tokens": 999})
    chunk = next(s for s in swapped.stages if s.name == "chunk")
    assert chunk.config["max_tokens"] == 999  # overridden
    assert chunk.config["overlap_tokens"] == 32  # inherited from the base stage config


def test_resolve_stage_name_infers_unique_stage(tmp_path: Path) -> None:
    """A matrix with no explicit stage_name resolves to the unique Stage of the target kind."""
    base = load_stack(_DEFAULT, storage_uri=f"file://{tmp_path}/s", corpus_dir=f"file://{tmp_path}/c")
    spec = MatrixSpec(stage=CapabilityKind.PARSE, candidates=[MatrixCandidate(provider="x")])
    assert _resolve_stage_name(base, spec) == "parse"


def test_resolve_stage_name_ambiguous_export_requires_explicit(tmp_path: Path) -> None:
    """The default stack has TWO Export Stages, so an export matrix must name which to swap."""
    base = load_stack(_DEFAULT, storage_uri=f"file://{tmp_path}/s", corpus_dir=f"file://{tmp_path}/c")
    spec = MatrixSpec(stage=CapabilityKind.EXPORT, candidates=[MatrixCandidate(provider="x")])
    with pytest.raises(ValueError, match="multiple .* Stages"):
        _resolve_stage_name(base, spec)


# -- End-to-end through the local Runner: the parse matrix -------------------


def test_parse_bakeoff_runs_all_candidates(tmp_path: Path) -> None:
    """The parse matrix runs the three CPU parsers for real; the GPU OCR one is skipped-with-flag.

    W1-lighton/W1-glm/W5-render device honesty on the Parse Stage: the three CPU parsers RUN
    (measured); the two GPU OCR parsers ``parser.lighton`` / ``parser.glm`` (installed descriptors,
    pinned ``device: cuda``) are device-skipped-with-flag, and the universal ``parser.render`` (its
    heavy GPU/endpoint package is not installed on this CPU host) is not-installed-skipped-with-flag
    — every skip recorded, never a fabricated OCR quality/throughput number (all cite #64).
    """
    result = run_bakeoff(
        _PARSE_MATRIX, base_stack_path=_DEFAULT, storage_root=f"file://{tmp_path}/store"
    )
    assert result.stage == "parse"
    assert result.stage_name == "parse"
    assert result.document_count == 7  # the bundled corpus
    rows = _by_provider(result)
    assert set(rows) == {
        "parser.plaintext",
        "parser.pdfplumber",
        "parser.document",
        "parser.lighton",
        "parser.glm",
        "parser.render",
    }
    skipped = {"parser.lighton", "parser.glm", "parser.render"}
    gpu_ocr = {"parser.lighton", "parser.glm"}
    cpu_rows = {k: v for k, v in rows.items() if k not in skipped}
    for row in cpu_rows.values():
        assert row.ran  # every CPU candidate ran (all installed)
        # -- the ADR-0036 §4 columns are all populated for a run candidate --
        assert row.quality is not None and row.quality.kind == "metric"
        assert row.stage_seconds is not None and row.stage_seconds >= 0.0
        assert row.throughput_docs_per_s is not None
        assert row.memory is not None
        # memory_mb is READ (declared-vs-measured) — the audit's inert-field is retired.
        assert row.memory.declared_mb is not None
        assert row.memory.measured_mb > 0.0
        # license is READ off the declared profile. A pure-code CPU reference declares the
        # REPO's own licence (PolyForm-Noncommercial-1.0.0 since 2026-07-29); a model- or
        # library-bearing one declares that third party's permissive licence.
        assert row.license.spdx in {REPO_LICENSE_CODE, "Apache-2.0", "MIT"}
        assert row.license.verified is True
        # deterministic is READ + cross-checked (all three CPU parsers are byte-stable).
        assert row.determinism.declared is True
        assert row.determinism.observed_identical is True
        assert row.determinism.mismatch is False
        assert row.error_count == 0
        assert row.compute == "cpu"
    # Every skipped candidate is skipped-with-flag — no fabricated columns, cite the rig gap (#64).
    for name in skipped:
        row = rows[name]
        assert row.ran is False
        assert row.skipped is True
        assert row.skip_reason is not None
        assert "#64" in row.skip_reason
    # The two installed GPU OCR descriptors resolve their compute + license without a rig.
    for name in gpu_ocr:
        gpu = rows[name]
        assert gpu.compute == "gpu"
        # The license IS still read off the declared profile (the descriptor needs no rig).
        assert gpu.license.verified is True
    # LightOn's model is Apache-2.0; GLM's is MIT — both permissive, read off the descriptor.
    assert rows["parser.lighton"].license.spdx == "Apache-2.0"
    assert rows["parser.glm"].license.spdx == "MIT"
    # parser.render's package is not installed on this CPU host, so its row is a not-installed skip
    # (no resolved compute/license) — still flagged with #64, never a fabricated number.
    render = rows["parser.render"]
    # parser.render is skipped on a CPU host either way: not-installed (deps absent) OR
    # device-skipped (compute="either", cuda requested). Robust to the heavy-deps install state.
    assert render.compute in (None, "either")
    assert render.skip_reason


def test_parse_bakeoff_is_a_real_comparison(tmp_path: Path) -> None:
    """The candidates genuinely differ — a real bake-off, not three identical rows."""
    result = run_bakeoff(
        _PARSE_MATRIX, base_stack_path=_DEFAULT, storage_root=f"file://{tmp_path}/store"
    )
    rows = _by_provider(result)
    # plaintext reads .txt/.md passthrough (more chars) vs the PDF parsers extracting page text —
    # the quality metric distinguishes them (a genuine controlled comparison, not a mock).
    plaintext_chars = rows["parser.plaintext"].quality.metrics["chars"]  # type: ignore[union-attr]
    pdf_chars = rows["parser.pdfplumber"].quality.metrics["chars"]  # type: ignore[union-attr]
    assert plaintext_chars != pdf_chars


def test_chunk_bakeoff_runs_all_candidates(tmp_path: Path) -> None:
    """The chunk matrix runs both real CPU chunkers end-to-end with the chunk-specific metric."""
    result = run_bakeoff(
        _CHUNK_MATRIX, base_stack_path=_DEFAULT, storage_root=f"file://{tmp_path}/store"
    )
    rows = _by_provider(result)
    assert set(rows) == {"chunk.markdown", "chunk.sentence_window"}
    for row in rows.values():
        assert row.ran
        assert row.quality is not None and row.quality.kind == "metric"
        assert "chunks" in row.quality.metrics
        assert "offset_preserving_fraction" in row.quality.metrics


# -- Device honesty: a GPU/endpoint candidate is skipped-with-flag -----------


def test_entity_matrix_gpu_candidate_skipped_with_flag(tmp_path: Path) -> None:
    """entity.gliner (GPU/heavy) is skipped-with-flag on a CPU host — NO fabricated numbers."""
    result = run_bakeoff(
        _ENTITY_MATRIX, base_stack_path=_DEFAULT, storage_root=f"file://{tmp_path}/store"
    )
    rows = _by_provider(result)
    gliner = rows["entity.gliner"]
    assert gliner.skipped is True
    assert gliner.skip_reason is not None
    assert "#64" in gliner.skip_reason  # references the rig gap, never fabricated
    # NO measured columns on a skipped row (the honest boundary).
    assert gliner.quality is None
    assert gliner.stage_seconds is None
    assert gliner.memory is None
    assert gliner.determinism.checked is False
    # The CPU candidate in the same matrix still ran.
    assert rows["entity.gazetteer"].ran


def test_embed_matrix_is_a_valid_spec() -> None:
    """The committed embed matrix parses: an ``embedding`` target swapping the Export's embedder."""
    embed = MatrixSpec.from_yaml_text(_EMBED_MATRIX.read_text(encoding="utf-8"))
    assert embed.stage is CapabilityKind.EMBEDDING
    assert embed.stage_name == "export_corpus"
    assert [c.provider for c in embed.candidates] == [
        "embedding.hashing",
        "embedding.sentence_transformers",
    ]


def test_embed_matrix_hashing_runs_st_skipped_with_flag(tmp_path: Path) -> None:
    """The embed bake-off: the CPU hashing embedder RUNS (measured), the learned ST is skipped.

    W1-embed device honesty applied to the *nested* embedder (the opt-in Export augmentation is not
    a DAG Stage): ``embedding.hashing`` is swapped into the corpus Export and measured for real;
    ``embedding.sentence_transformers`` (pinned ``device: cuda``) is skipped-with-flag on this
    CPU-only host through the SAME ``select_device`` seam — NO fabricated numbers (references #64).
    """
    result = run_bakeoff(
        _EMBED_MATRIX, base_stack_path=_DEFAULT, storage_root=f"file://{tmp_path}/store"
    )
    rows = _by_provider(result)
    # The in-core hashing embedder ran for real (measured, deterministic). It is pure framework
    # code with no weights, so its licence column is the REPO's own, not a permissive default.
    hashing = rows["embedding.hashing"]
    assert hashing.ran
    assert hashing.skipped is False
    assert hashing.determinism.checked is True
    assert hashing.license is not None and hashing.license.spdx == REPO_LICENSE_CODE
    # The learned ST embedder is skipped-with-flag — no fabricated columns, cites the rig gap.
    st = rows["embedding.sentence_transformers"]
    assert st.skipped is True
    assert st.skip_reason is not None
    assert "#64" in st.skip_reason
    assert "embedder" in st.skip_reason  # the nested-embedder skip, not a top-level stage skip
    assert st.quality is None
    assert st.stage_seconds is None
    assert st.memory is None
    assert st.determinism.checked is False
    # Its license column is still surfaced from the declared profile (Granite r2 Apache-2.0 weights),
    # never faked.
    assert st.license is not None and st.license.spdx == "Apache-2.0"


# -- The determinism cross-check BITES: a lying declared=True is flagged ------


class _LyingChunker:
    """A chunk Provider that declares ``deterministic=True`` but emits a DIFFERENT chunk each run.

    The determinism cross-check must catch it: two runs produce different target-Stage checkpoints,
    so ``observed_identical`` is False against a ``declared=True`` — a ``determinism.mismatch``. The
    non-determinism is process-run-scoped (a module counter that advances every construction), so
    two runs in the same process differ.
    """

    _counter = 0

    @classmethod
    def profile(cls) -> ProviderProfile:
        return ProviderProfile(
            compute="cpu",
            memory_mb=8,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_source="LICENSE",
            license_verified_on="2026-07-08",
            deterministic=True,  # THE LIE — the two runs below diverge.
            batch=False,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        from latence_core.stages.chunk import MarkdownChunker

        type(self)._counter += 1
        self._nonce = type(self)._counter
        self._inner = MarkdownChunker(config)

    def chunk(self, documents: Iterable[Any]) -> Iterator[ChunkRecord]:
        for chunk in self._inner.chunk(documents):
            # Append the per-construction nonce to the chunk content so two runs diverge byte-wise
            # (the checkpoint is model_dump_json of the record) while staying a valid ChunkRecord.
            # This is the "declares deterministic but isn't" lie the cross-check must catch; the
            # perturbation is downstream-tolerant (content is free text, offsets untouched).
            yield chunk.model_copy(update={"content": f"{chunk.content} [{self._nonce}]"})


def _registry_with_fake(name: str, provider_cls: type) -> ProviderRegistry:
    """A real :class:`ProviderRegistry` with one extra in-memory Provider injected by name.

    Seeds the discovery cache (``_entry_points``) with a fake entry point that loads to
    ``provider_cls``, plus pre-populates the ``_loaded``/``_profiles`` caches, so both the harness
    (``registry.profile``/``.names``) AND the LocalRunner it drives (``registry.load``) see the fake
    Provider — without installing a package. Real entry points are discovered first so the rest of
    the stack (source/parse/…/export) resolves normally.
    """
    registry = ProviderRegistry()
    discovered = dict(registry._discover())  # noqa: SLF001 - test seeds the discovery cache

    class _FakeEntryPoint:
        def load(self) -> type:
            return provider_cls

    discovered[name] = _FakeEntryPoint()  # type: ignore[assignment]
    registry._entry_points = discovered  # noqa: SLF001
    registry._loaded[name] = provider_cls  # noqa: SLF001
    return registry


def test_determinism_check_bites_a_lying_provider(tmp_path: Path) -> None:
    """A candidate declaring deterministic=True but diverging across two runs is FLAGGED."""
    _LyingChunker._counter = 0  # noqa: SLF001 - reset the process-run nonce for a seeded test
    registry = _registry_with_fake("chunk.lying", _LyingChunker)
    harness = BakeOffHarness(
        _DEFAULT, storage_root=f"file://{tmp_path}/store", registry=registry
    )
    spec = MatrixSpec(
        stage=CapabilityKind.CHUNK, candidates=[MatrixCandidate(provider="chunk.lying")]
    )
    result = harness.run(spec)
    row = _by_provider(result)["chunk.lying"]
    assert row.ran  # it ran (did not crash) …
    assert row.determinism.declared is True
    assert row.determinism.observed_identical is False  # … but its two runs diverged …
    assert row.determinism.mismatch is True  # … so the lie is FLAGGED (the check bites).
    # The rendered table surfaces the mismatch prominently.
    assert "MISMATCH" in result.render()


# -- Degrade per-candidate: a crashing candidate is a failed row -------------


class _CrashingChunker:
    """A chunk Provider that raises mid-run — the bake-off records a failed row, not an abort."""

    @classmethod
    def profile(cls) -> ProviderProfile:
        return ProviderProfile(
            compute="cpu",
            memory_mb=8,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_source="LICENSE",
            license_verified_on="2026-07-08",
            deterministic=True,
            batch=False,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        pass

    def chunk(self, documents: Iterable[Any]) -> Iterator[ChunkRecord]:
        msg = "candidate exploded on purpose"
        raise RuntimeError(msg)
        yield  # pragma: no cover - unreachable, marks this a generator


def test_a_crashing_candidate_is_a_failed_row_not_an_aborted_matrix(tmp_path: Path) -> None:
    """One bad candidate degrades to a failed row; the good candidate in the same matrix still runs."""
    registry = _registry_with_fake("chunk.crashing", _CrashingChunker)
    harness = BakeOffHarness(
        _DEFAULT, storage_root=f"file://{tmp_path}/store", registry=registry
    )
    spec = MatrixSpec(
        stage=CapabilityKind.CHUNK,
        candidates=[
            MatrixCandidate(provider="chunk.crashing"),
            MatrixCandidate(provider="chunk.markdown"),
        ],
    )
    result = harness.run(spec)
    rows = _by_provider(result)
    assert rows["chunk.crashing"].failed is True
    assert rows["chunk.crashing"].failure_detail is not None
    assert "RuntimeError" in rows["chunk.crashing"].failure_detail
    # The matrix did NOT abort — the good candidate still ran + was measured.
    assert rows["chunk.markdown"].ran
    assert "FAILED" in result.render()


# -- Quality reuses the S10 gold-set scorer (a gold-set-bearing base stack) ---


def _gold_bearing_stack(tmp_path: Path) -> Path:
    """A base stack that declares a gold_set_uri so the Entity Stage gets an S10 P/R/F1 score.

    The gold set annotates the bundled corpus's ``memo.md`` / ``note.txt`` entities against the same
    labels the default stack's gazetteer extracts — so the bake-off's quality column is a real
    gold-set F1 read off the Quality Report's `goldset` rollup (NOT reinvented scoring).
    """
    gold = {
        "schema_version": 1,
        "match_min_overlap": 0.0,
        "documents": [
            {
                "file_name": "memo.md",
                "entities": [
                    # Alice (PERSON) lands at [20, 25) in the parsed memo.md markdown — the exact
                    # offset the default stack's gazetteer emits, so this is a true positive.
                    {"label": "PERSON", "char_start": 20, "char_end": 25, "text": "Alice"},
                ],
            },
        ],
    }
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(gold), encoding="utf-8")

    text = _DEFAULT.read_text(encoding="utf-8")
    # Inject the gold_set_uri as a top-level Pipeline key (after the storage_uri line).
    text = text.replace(
        'storage_uri: "{STORAGE_URI}"',
        'storage_uri: "{STORAGE_URI}"\ngold_set_uri: "file://' + str(gold_path) + '"',
    )
    stack_path = tmp_path / "gold-stack.yaml"
    stack_path.write_text(text, encoding="utf-8")
    return stack_path


def test_quality_reuses_goldset_scorer_when_base_stack_has_a_gold_set(tmp_path: Path) -> None:
    """An Entity bake-off over a gold-set-bearing base stack surfaces the S10 P/R/F1 (reused, not new)."""
    stack = _gold_bearing_stack(tmp_path)
    spec = MatrixSpec(
        stage=CapabilityKind.ENTITY_EXTRACTION,
        candidates=[MatrixCandidate(provider="entity.gazetteer")],
    )
    harness = BakeOffHarness(stack, storage_root=f"file://{tmp_path}/store")
    result = harness.run(spec)
    row = _by_provider(result)["entity.gazetteer"]
    assert row.ran
    assert row.quality is not None
    assert row.quality.kind == "goldset"  # the S10 scorer, not a stage metric
    assert row.quality.f1 is not None
    # The gazetteer extracts Alice (PERSON) in memo.md — a true positive, so recall > 0.
    assert row.quality.recall is not None and row.quality.recall > 0.0


def test_quality_of_falls_back_to_none_without_a_lens() -> None:
    """A Stage the bake-off has no quality lens for (no gold set, no stage metric) yields None."""
    from latence_core.quality import QualityReport

    report = QualityReport(pipeline_name="p", run_id="r")  # no parse/chunk/goldset sections
    spec = MatrixSpec(stage=CapabilityKind.REDACTION, candidates=[MatrixCandidate(provider="x")])
    assert _quality_of(report, spec, "redact") is None


# -- The CLI writes MATRIX-RESULTS-<stage>.md + a JSON companion --------------


def test_cli_bakeoff_writes_md_and_json(tmp_path: Path) -> None:
    """``latence bake-off matrix/parse.yaml`` writes the signed MD + the machine-readable JSON."""
    out = tmp_path / "out"
    result = cli.invoke(
        app,
        [
            "bake-off",
            str(_PARSE_MATRIX),
            "--base-stack",
            str(_DEFAULT),
            "--out-dir",
            str(out),
            "--storage-root",
            f"file://{tmp_path}/store",
        ],
    )
    assert result.exit_code == 0, result.output
    md = out / "MATRIX-RESULTS-parse.md"
    js = out / "MATRIX-RESULTS-parse.json"
    assert md.exists()
    assert js.exists()
    assert "Bake-off — `parse`" in md.read_text(encoding="utf-8")
    # The JSON companion round-trips to a MatrixResult (schema-valid, consumable numbers).
    parsed = MatrixResult.model_validate_json(js.read_text(encoding="utf-8"))
    assert parsed.stage == "parse"
    assert len(parsed.candidates) == 6  # 3 CPU parsers + 2 GPU OCR + parser.render (skipped on CPU)


def test_cli_bakeoff_missing_file_exits_two(tmp_path: Path) -> None:
    """A missing matrix file exits 2 (a usage error), never a traceback."""
    result = cli.invoke(app, ["bake-off", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2


# -- Determinism of the harness itself: a seeded bake-off is reproducible -----


def test_bakeoff_render_is_deterministic(tmp_path: Path) -> None:
    """Two bake-off runs of the same matrix render byte-identical tables modulo wall-clock.

    Wall-clock latency is machine-relative (it legitimately varies), so we assert the STRUCTURAL
    columns (candidates, quality, memory-declared, license, determinism, errors) are identical —
    the deterministic content the report rests on.
    """
    a = run_bakeoff(_CHUNK_MATRIX, base_stack_path=_DEFAULT, storage_root=f"file://{tmp_path}/a")
    b = run_bakeoff(_CHUNK_MATRIX, base_stack_path=_DEFAULT, storage_root=f"file://{tmp_path}/b")
    for ca, cb in zip(a.candidates, b.candidates, strict=True):
        assert ca.provider == cb.provider
        assert ca.quality == cb.quality
        assert ca.error_count == cb.error_count
        assert ca.determinism.observed_identical == cb.determinism.observed_identical
        assert ca.license == cb.license
        assert (ca.memory.declared_mb if ca.memory else None) == (
            cb.memory.declared_mb if cb.memory else None
        )
