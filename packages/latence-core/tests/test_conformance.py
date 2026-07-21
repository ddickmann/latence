"""Provider Conformance suite — the parametrized-over-every-Provider gate (P3-F3, ADR-0036 §5).

This is the enterprise-readiness gate: it parametrizes over **every** Provider the
:class:`~latence_core.capability.ProviderRegistry` discovers and asserts the C1–C6 contract
(:mod:`latence_core.conformance`). "Passes conformance" *is* the definition of enterprise-ready.

Deterministic + seeded (``PYTHONHASHSEED`` pinned by the repo-root conftest; ``LATENCE_CUDA=0``
pinned below so the device seam is reproducible). The CPU reference Providers run for real; the
endpoint/GPU Providers run the contract/graceful-failure/license/determinism-declaration subset
against a deterministic stub (their real numbers need a rig, #64 — flagged, never fabricated).

The suite also proves the checks **bite** (not merely assert): a deliberately-broken throwaway
Provider FAILs C2 (untyped crash) and C4 (declared-deterministic-but-not), and a synthetic
GPU-only Provider is skipped-with-flag by C5 — so a bad Provider would actually be caught.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from latence_core.capability import ProviderRegistry
from latence_core.conformance import (
    build_conformance_report,
    capability_of,
    case_for_provider,
    run_conformance,
)
from latence_core.conformance.cases import AdversarialOutcome, Capability
from latence_core.conformance.harness import (
    _check_c2_graceful,
    _check_c3_license,
    _check_c4_determinism,
    _check_c5_device,
    _run_conformance_checks_for,
)
from latence_core.conformance.report import CheckStatus
from latence_core.providers.profile import ProviderProfile

# Evidence-bearing verified profiles (H-B1) built in-test. A pure-code profile cites the repo
# LICENSE (weights None); a model-bearing profile records the weights SPDX + a model-card
# citation (headline license == weights). Typed builders (not ``**dict`` spreads) so mypy
# --strict can check every keyword against ProviderProfile's typed constructor.


def _purecode_profile(
    *, compute: str = "cpu", memory_mb: int = 1, deterministic: bool = True, batch: bool = False
) -> ProviderProfile:
    """A pure-code (model_id=None) evidence-bearing verified profile."""
    return ProviderProfile(
        compute=compute,  # type: ignore[arg-type]
        memory_mb=memory_mb,
        model_id=None,
        license="Apache-2.0",
        license_verified=True,
        license_code="Apache-2.0",
        license_weights=None,
        license_source="LICENSE",
        license_verified_on="2026-07-06",
        deterministic=deterministic,
        batch=batch,
    )


def _model_profile(
    *,
    model_id: str,
    spdx: str,
    source: str,
    compute: str = "either",
    memory_mb: int = 1,
    deterministic: bool = False,
    batch: bool = True,
) -> ProviderProfile:
    """A model-bearing evidence-bearing verified profile (weights SPDX + citation, headline==weights)."""
    return ProviderProfile(
        compute=compute,  # type: ignore[arg-type]
        memory_mb=memory_mb,
        model_id=model_id,
        license=spdx,
        license_verified=True,
        license_code="Apache-2.0",
        license_weights=spdx,
        license_source=source,
        license_verified_on="2026-07-06",
        deterministic=deterministic,
        batch=batch,
    )

# The exact roster the registry must discover — asserting real coverage, not a moving target.
# 50 Providers across 21 Capabilities (P3 landscape + the three index-time retrieval-signal
# reference generators sparse.hashing / multivector.hashing / fde.reference #160 + the first real
# Wave-1 adapter H-A1 + the
# learned sentence-transformers Embedder W1-embed + the real Presidio Redactor W1-pii + the real
# GPU LightOn-OCR Parser W1-lighton + its served-vLLM twin parser.lighton_vllm W10 + its MIT twin
# the GPU GLM-OCR Parser W1-glm + the OPTIONAL reference GraphCompleter W2-linkpred Slice 1 + the
# learned ULTRA/PyKEEN GraphCompleter W2-linkpred Slice 2 + the fastino GLiNER2 Redactor
# W3-pii-gliner2 + the OPTIONAL LLM schema-induction LabelInducer W4-schema-induction + the universal
# license-clean parser.render W5-universal-parser + the unified fused Entity+Relation gliner2
# extractor W9-gliner2-unified, the relex drop-in + the OPTIONAL reference ContextEnricher
# context.kg_header W13-context-enrichment + the enterprise GLinker Disambiguator
# disambiguation.glinker ADR-0046 — the neural L2→L3→L4→L0 linking + audited resolver, GPU + the two
# OPTIONAL reference GraphFeatureComputers graph_features.degree / graph_features.pagerank — degree/
# connected-components + PageRank/label-propagation centrality+community, ADR-0051 #161).
# If a Provider is added/removed, this list is updated deliberately, so the count check catches an
# accidental un-covered Provider.
_EXPECTED_PROVIDERS = frozenset(
    {
        "source.local_folder",
        # The SharePoint/OneDrive source-SYSTEM connector (#182, latence-source-sharepoint).
        "source.sharepoint",
        "parser.plaintext",
        "parser.document",
        "parser.pdfplumber",
        "parser.lighton",
        "parser.lighton_vllm",
        "parser.glm",
        "parser.render",
        "parser.endpoint",
        "chunk.markdown",
        "chunk.sentence_window",
        "screening.intake_signature",
        "screening.content_keyword",
        "screening.content_fuzzy",
        "label_inducer.llm",
        "entity.gazetteer",
        "entity.gliner",
        "entity.endpoint",
        "relation.pattern",
        "relation.llm",
        "relation.gliner_relex",
        "fused_entity_relation.gliner2",
        # The in-core deterministic structured-header cross-reference extractor (dogfood-3-inspect):
        # recovers self-anchored, bare-identifier metadata-header relations (RFC Obsoletes:/Updates:)
        # that entity-anchored RE cannot. Pure-Python, zero-dep, fulfils the fused seam.
        "fused_entity_relation.header_refs",
        # The in-core deterministic INLINE-CITATION cross-reference extractor (the dogfood measured
        # gap): recovers a document's INLINE body "RFC <n>" citations — the majority of a corpus's
        # gold cross-references — that the header-only header_refs sibling structurally cannot see.
        # Pure-Python, zero-dep, fulfils the fused seam.
        "fused_entity_relation.inline_refs",
        "redaction.hybrid_rule",
        "redaction.gliner_pii",
        "redaction.gliner2",
        "redaction.presidio",
        "profiling.statistical",
        "profiling.lightweight",
        "disambiguation.cascade",
        "disambiguation.exact_surface",
        "disambiguation.embedding",
        "disambiguation.glinker",
        "graph.canonical",
        "graph.weighted",
        "graph_completion.reference",
        "graph_completion.ultra",
        "graph_features.degree",
        "graph_features.pagerank",
        "context.kg_header",
        "export.jsonl_parquet",
        "export.knowledge_graph",
        "export.demo_site",
        "embedding.hashing",
        "embedding.endpoint",
        "embedding.sentence_transformers",
        "sparse.hashing",
        "sparse.splade",
        "multivector.hashing",
        "fde.reference",
        # The real, canonical MUVERA FDE (latence-muvera, #163) — pure-Python, training-free,
        # numpy-free, installed in the CPU-first dev env so it RUNS the default _fde_case() for
        # real, alongside the in-core fde.reference (ADR-0050).
        "fde.muvera",
        # The two in-core reference Tokenizers (#164) — the varying seam of the optional BM25
        # term-statistics artifact exporter (ADR-0048/0049). Pure-Python, deterministic, zero-dep.
        "tokenizer.regex",
        "tokenizer.whitespace",
        "delta.affected_set",
    }
)


@pytest.fixture(autouse=True)
def _pin_cpu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the device seam to a CPU-only host so the suite is reproducible (LATENCE_CUDA=0)."""
    monkeypatch.setenv("LATENCE_CUDA", "0")


def _registry() -> ProviderRegistry:
    return ProviderRegistry()


def _all_provider_names() -> list[str]:
    return _registry().names()


# ---------------------------------------------------------------------------
# Coverage: every registered Provider is parametrized, and every Capability has a case.
# ---------------------------------------------------------------------------


def test_registry_discovers_the_expected_provider_roster() -> None:
    """The registry discovers exactly the expected 56 Providers — real coverage, not a subset."""
    names = set(_all_provider_names())
    assert names == set(_EXPECTED_PROVIDERS), (
        f"unexpected roster; missing={_EXPECTED_PROVIDERS - names}, extra={names - _EXPECTED_PROVIDERS}"
    )
    assert len(names) == 56


def test_every_provider_maps_to_a_capability_with_a_case() -> None:
    """No silent coverage gap: every Provider's Capability has a ConformanceCase (else FAIL)."""
    for name in _all_provider_names():
        capability = capability_of(name)  # raises if the prefix names no Capability
        case = case_for_provider(name)  # raises if the Capability has no case
        assert case.capability is capability or case.stub_scoped


def test_a_capability_with_no_case_fails_loudly() -> None:
    """A Provider whose Capability has no case FAILs loudly (forces the case), never skips."""
    # ``capability_of`` rejects an unknown prefix — the anti-silent-skip guard.
    with pytest.raises(KeyError, match="unknown Capability prefix"):
        capability_of("mystery.provider")


# ---------------------------------------------------------------------------
# C1–C6 for every registered Provider (the gate itself).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_name", sorted(_EXPECTED_PROVIDERS))
def test_provider_passes_conformance(provider_name: str) -> None:
    """Every registered Provider passes C1–C6 (no check FAILs) — the enterprise-readiness gate."""
    result = run_conformance(_registry(), provider_name)
    failed = [c for c in result.checks if c.status is CheckStatus.FAIL]
    assert not failed, f"{provider_name} failed: {[(c.check, c.detail) for c in failed]}"
    # Every Provider is checked for all six concerns (some SKIPPED, none missing).
    checks = {c.check for c in result.checks}
    assert {"C1_valid_records", "C2_graceful_failure", "C3_license", "C4_determinism",
            "C5_device", "C6_no_pii_leak"} <= checks


def test_full_report_all_pass_and_flags_stub_scoped() -> None:
    """The rolled-up report: all pass, endpoint/GPU stub-flagged, the GPU OCR Parsers skipped."""
    report = build_conformance_report(_registry())
    assert report.providers_total == 56
    assert report.providers_failed == 0
    assert report.all_passed
    # The endpoint/model Providers are flagged stub-scoped ("needs a rig"), never fabricated.
    stub_scoped = {r.provider for r in report.results if r.stub_scoped}
    assert stub_scoped == {
        "parser.endpoint",
        "parser.lighton_vllm",
        "entity.endpoint",
        "entity.gliner",
        "label_inducer.llm",
        "relation.llm",
        "relation.gliner_relex",
        "fused_entity_relation.gliner2",
        "redaction.gliner_pii",
        "redaction.gliner2",
        "redaction.presidio",
        "embedding.endpoint",
        "embedding.sentence_transformers",
        "sparse.splade",
        "graph_completion.ultra",
        # source.sharepoint (#182) needs a tenant for live numbers, so it is verified against the
        # faithful offline Graph fake its package ships — flagged, never passed off as live.
        "source.sharepoint",
    }
    # The GPU-only Providers (compute="gpu"): the LightOn-OCR + MIT twin GLM-OCR Parsers and the
    # enterprise GLinker Disambiguator (its neural gliner-linker biencoder needs a GPU, ADR-0046)
    # are all skipped-with-flag on this CPU host (the G1 posture) — recorded as device-skipped, still
    # passing (a SKIP is never a FAIL), never faked.
    device_skipped = {r.provider for r in report.results if r.device_skipped}
    assert device_skipped == {"parser.lighton", "parser.glm", "disambiguation.glinker"}


def test_cpu_providers_run_for_real_not_stub() -> None:
    """The in-core CPU reference Providers are NOT stub-scoped — they run for real."""
    report = build_conformance_report(_registry())
    for res in report.results:
        if res.provider in {
            "chunk.markdown",
            "entity.gazetteer",
            "graph.canonical",
            "graph_completion.reference",
        }:
            assert not res.stub_scoped


# ---------------------------------------------------------------------------
# C3 license gate is genuine.
# ---------------------------------------------------------------------------


def test_c3_unverified_plus_verified_is_impossible() -> None:
    """The UNVERIFIED-never-verified invariant is enforced at the provider level (C3)."""
    # A ProviderProfile forbids constructing UNVERIFIED+verified, so C3 re-asserts on a would-be
    # bad profile via a stub object carrying the two fields (the invariant re-check).
    class _BadProfileHolder:
        license = "UNVERIFIED"
        license_verified = True

    outcome = _check_c3_license(_BadProfileHolder())  # type: ignore[arg-type]
    assert outcome.status is CheckStatus.FAIL


def test_c3_missing_profile_fails() -> None:
    """A Provider that declares no profile FAILs C3 (every provider must declare one)."""
    outcome = _check_c3_license(None)
    assert outcome.status is CheckStatus.FAIL


def test_c3_bites_verified_without_evidence() -> None:
    """H-B1 bite-proof: a provider claiming license_verified=True with NO evidence FAILs C3.

    The exact audit finding — ``license="Apache-2.0", license_verified=True`` with nothing
    checked (no source, no date, no code SPDX). The ProviderProfile validator makes this
    un-constructable, so the theater is reproduced via a fake profile-holder that bypasses it
    (a bare object carrying just the two self-attested fields). C3 must FAIL it, not pass —
    the evidence-bearing rule is the real enforcement point, not merely a construction-time hope.
    """

    class _SelfAttestedHolder:
        license = "Apache-2.0"
        license_verified = True
        # No license_source / license_verified_on / license_code / license_weights / model_id.

    outcome = _check_c3_license(_SelfAttestedHolder())  # type: ignore[arg-type]
    assert outcome.status is CheckStatus.FAIL
    assert "evidence" in outcome.detail


def test_c3_bites_model_verified_without_weights_spdx() -> None:
    """C3 FAILs a model-bearing provider that claims verified with a source/date but NO weights SPDX."""

    class _NoWeightsHolder:
        license = "Apache-2.0"
        license_verified = True
        license_code = "Apache-2.0"
        license_weights = None  # missing — ADR-0012 requires it for a model-bearing provider
        license_source = "https://huggingface.co/x/y"
        license_verified_on = "2026-07-06"
        model_id = "x/y"

    outcome = _check_c3_license(_NoWeightsHolder())  # type: ignore[arg-type]
    assert outcome.status is CheckStatus.FAIL
    assert "weights" in outcome.detail


def test_c3_bites_inconsistent_license_fields() -> None:
    """C3 FAILs a provider whose headline license disagrees with its weights SPDX."""

    class _InconsistentHolder:
        license = "MIT"  # headline disagrees with the weights SPDX below
        license_verified = True
        license_code = "Apache-2.0"
        license_weights = "Apache-2.0"
        license_source = "https://huggingface.co/x/y"
        license_verified_on = "2026-07-06"
        model_id = "x/y"

    outcome = _check_c3_license(_InconsistentHolder())  # type: ignore[arg-type]
    assert outcome.status is CheckStatus.FAIL
    assert "disagree" in outcome.detail


def test_c3_passes_evidence_bearing_verified_provider() -> None:
    """C3 PASSes a real evidence-bearing verified profile (the reconciled entity.gliner shape)."""
    p = _model_profile(
        model_id="urchade/gliner_multi-v2.1",
        spdx="Apache-2.0",
        source="https://huggingface.co/urchade/gliner_multi-v2.1",
        memory_mb=2048,
    )
    outcome = _check_c3_license(p)
    assert outcome.status is CheckStatus.PASS


def test_c3_restricted_unverified_license_fails() -> None:
    """A restricted (non-permissive) license that is not verified fails C3 (must be opt-in)."""
    restricted = ProviderProfile(
        compute="cpu",
        memory_mb=1,
        model_id="m",
        license="LicenseRef-Gemma",
        license_verified=False,
        deterministic=True,
        batch=False,
    )
    outcome = _check_c3_license(restricted)
    assert outcome.status is CheckStatus.FAIL


def test_c3_restricted_verified_license_is_opt_in_pass() -> None:
    """A restricted license that IS verified passes C3 as a flagged opt-in (ADR-0012)."""
    opt_in = _model_profile(
        model_id="google/embeddinggemma-300m",
        spdx="LicenseRef-Gemma",
        source="https://huggingface.co/google/embeddinggemma-300m",
        compute="gpu",
    )
    outcome = _check_c3_license(opt_in)
    assert outcome.status is CheckStatus.PASS
    assert "opt-in" in outcome.detail


# ---------------------------------------------------------------------------
# C5 device: a synthetic GPU-only Provider is skipped-with-flag on this CPU host.
# ---------------------------------------------------------------------------


def test_c5_gpu_provider_is_skipped_with_flag_on_cpu_host() -> None:
    """A compute="gpu" Provider on a CPU-only host is skipped-with-flag (never run, never faked)."""
    gpu_profile = _model_profile(
        model_id="some/gpu-only-model",
        spdx="Apache-2.0",
        source="https://huggingface.co/some/gpu-only-model",
        compute="gpu",
        memory_mb=4096,
    )
    outcome, skipped, reason = _check_c5_device(gpu_profile)
    assert skipped is True
    assert outcome.status is CheckStatus.SKIPPED
    assert reason is not None and "GPU" in reason.upper() or "cuda" in (reason or "").lower()


def test_c5_gpu_provider_end_to_end_skip_records_but_does_not_run() -> None:
    """A GPU-only Provider skipped by C5 records C1/C2/C4/C6 as SKIPPED (not run, not faked)."""
    # Register a synthetic GPU-only Provider inline and run the full harness path on it.
    class _GpuOnlyChunker:
        @classmethod
        def profile(cls) -> ProviderProfile:
            return _model_profile(
                model_id="gpu/chunker",
                spdx="Apache-2.0",
                source="https://huggingface.co/gpu/chunker",
                compute="gpu",
                memory_mb=4096,
            )

        def __init__(self, config: dict[str, Any] | None = None) -> None:
            self._config = config or {}

        def chunk(self, documents: Any) -> Any:  # pragma: no cover - must NOT be called
            raise AssertionError("a skipped GPU provider must never be run")

    result = _run_conformance_checks_for(
        provider_name="chunk.gpu_only",
        capability=Capability.CHUNK,
        provider_cls=_GpuOnlyChunker,
    )
    assert result.device_skipped is True
    statuses = {c.check: c.status for c in result.checks}
    assert statuses["C5_device"] is CheckStatus.SKIPPED
    for name in ("C1_valid_records", "C2_graceful_failure", "C4_determinism", "C6_no_pii_leak"):
        assert statuses[name] is CheckStatus.SKIPPED


# ---------------------------------------------------------------------------
# The checks BITE: a deliberately-broken throwaway Provider FAILs C2 and C4.
# ---------------------------------------------------------------------------


class _BrokenChunker:
    """A throwaway broken Provider: crashes untyped on bad input (C2) + non-deterministic (C4).

    It declares ``deterministic=True`` but returns a different result each call (C4 must catch
    that), and on an adversarial input it raises a **bare** ``RuntimeError`` (not a typed
    ``LatenceError``), which C2 must catch as an untyped crash.
    """

    _counter = 0

    @classmethod
    def profile(cls) -> ProviderProfile:
        # deterministic=True is a LIE (default) — C4 must catch it via the differing output.
        return _purecode_profile()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def chunk(self, documents: Any) -> Any:
        docs = list(documents)
        # A PARSE_ERROR document (the adversarial input) triggers the untyped crash C2 forbids.
        from latence_core.contracts import Disposition, DocumentRecord

        if any(
            isinstance(d, DocumentRecord) and d.disposition is Disposition.PARSE_ERROR
            for d in docs
        ):
            msg = "boom (untyped)"
            raise RuntimeError(msg)
        # Non-deterministic: a different (invalid) output each call, so two runs differ.
        type(self)._counter += 1
        base = list(_valid_chunks())
        # Mutate one field so the two runs are not byte-identical.
        if base:
            base[0] = base[0].model_copy(update={"token_count": type(self)._counter})
        return base


def _valid_chunks() -> Any:
    from latence_core.conformance import fixtures

    return list(fixtures.canonical_corpus().chunks)


def test_c2_bites_an_untyped_crash() -> None:
    """C2 FAILs a Provider that raises a bare (untyped) exception on the adversarial input."""
    outcome = _check_c2_graceful(case_for_provider("chunk.markdown"), _BrokenChunker)
    assert outcome.status is CheckStatus.FAIL
    assert "untyped" in outcome.detail


def test_c4_bites_a_declared_deterministic_that_is_not() -> None:
    """C4 FAILs a Provider that declares deterministic=True but yields different output twice."""
    outcome = _check_c4_determinism(
        case_for_provider("chunk.markdown"), _BrokenChunker, _BrokenChunker.profile()
    )
    assert outcome.status is CheckStatus.FAIL
    assert "differ" in outcome.detail


# ---------------------------------------------------------------------------
# C2 no-PII: a typed error / graceful path must never leak the adversarial PII.
# ---------------------------------------------------------------------------


class _PIILeakingScreener:
    """A Provider whose graceful path LEAKS the adversarial PII in its reason (C2 must catch it)."""

    @classmethod
    def profile(cls) -> ProviderProfile:
        return _purecode_profile()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def screen_content(self, chunks: Any) -> Any:
        from latence_core.capability import ScreeningOutcome
        from latence_core.conformance import fixtures
        from latence_core.contracts import (
            ScreeningCheckpoint,
            ScreeningFinding,
            ScreeningVerdict,
        )

        list(chunks)
        # A quarantine whose reason echoes the raw PII payload — the leak C2 forbids.
        finding = ScreeningFinding(
            checkpoint=ScreeningCheckpoint.CONTENT,
            verdict=ScreeningVerdict.QUARANTINE,
            document_id="adv-doc",
            detector="pii-leaker",
            category="prompt_injection",
            reason=f"blocked secret {fixtures.PII_ADVERSARIAL_SECRET}",
        )
        return ScreeningOutcome(passed=[], quarantined=[], findings=[finding])


def test_c2_bites_a_pii_leak_in_the_graceful_reason() -> None:
    """C2 FAILs a Provider that leaks the adversarial PII in its quarantine reason (no-PII gate)."""
    outcome = _check_c2_graceful(case_for_provider("screening.content_keyword"), _PIILeakingScreener)
    assert outcome.status is CheckStatus.FAIL
    assert "PII" in outcome.detail


def test_c2_passes_a_clean_typed_failure() -> None:
    """C2 PASSes the real keyword Content Screener (graceful, no PII in its surfaced reason)."""
    outcome = _check_c2_graceful(
        case_for_provider("screening.content_keyword"),
        _registry().load("screening.content_keyword"),
    )
    assert outcome.status is CheckStatus.PASS


# ---------------------------------------------------------------------------
# Determinism of the harness itself (a seeded run is byte-identical).
# ---------------------------------------------------------------------------


class _NonDeterministicEmbedder:
    """An Embedder that declares deterministic=True but returns different vectors each call.

    The archetypal C4 regression: a lying embedder. C4 must byte-compare the *actual vectors* the
    invoke returns (not a discarded empty list) and catch that the two runs differ — the signal
    the bake-off (P3-F5) trusts.
    """

    _counter = 0

    @classmethod
    def profile(cls) -> ProviderProfile:
        # deterministic=True is a LIE (default) — C4 must catch it via the returned vectors.
        return _purecode_profile()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._dimension = int((config or {}).get("dimension", 64))

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Any) -> Any:
        for _ in texts:
            type(self)._counter += 1
            # A correctly-shaped but non-deterministic vector: right width, different values.
            yield [float(type(self)._counter)] + [0.0] * (self._dimension - 1)


def test_c4_bites_a_non_deterministic_embedder_that_claims_determinism() -> None:
    """C4 FAILs a deterministic-declaring Embedder whose vectors differ across two runs.

    Regression guard: the embedding case must RETURN its vectors (not discard them), else C4
    compares empty-string to empty-string and a lying embedder passes vacuously.
    """
    outcome = _check_c4_determinism(
        case_for_provider("embedding.hashing"),
        _NonDeterministicEmbedder,
        _NonDeterministicEmbedder.profile(),
    )
    assert outcome.status is CheckStatus.FAIL
    assert "differ" in outcome.detail


def test_c4_passes_the_real_deterministic_embedder() -> None:
    """C4 PASSes the in-core hashing Embedder — byte-identical vectors over two real runs."""
    from latence_core.providers.profile import profile_of

    embedder = _registry().load("embedding.hashing")
    outcome = _check_c4_determinism(
        case_for_provider("embedding.hashing"),
        embedder,
        profile_of(embedder),
    )
    assert outcome.status is CheckStatus.PASS
    assert "byte-identical" in outcome.detail


def test_c1_bites_an_out_of_range_mention_offset() -> None:
    """C1's offset audit FAILs a mention whose char span runs past the document length.

    Regression guard: the in-range half of C1 must BOUND offsets against the document, not merely
    assert char_end >= char_start. The stronger page-map audit must see nonzero drift for an
    out-of-bounds mention (page_for_offset clamps + records drift rather than raising).
    """
    from latence_core.conformance import cases as conf_cases
    from latence_core.conformance import fixtures as conf_fixtures

    doc = conf_fixtures.canonical_corpus().documents[0]
    good_mention = conf_fixtures.canonical_corpus().mentions[0]
    total = doc.page_map.total_chars if doc.page_map is not None else len(conf_fixtures.CANONICAL_TEXT)
    # A mention whose span runs far past the assembled document — a broken/out-of-range offset.
    bad_prov = good_mention.provenance.model_copy(
        update={"char_start": total + 100, "char_end": total + 200}
    )
    bad_mention = good_mention.model_copy(update={"provenance": bad_prov})

    # The bounded in-range check FAILs it.
    assert conf_cases._offsets_in_range([bad_mention]) is False
    # The ADR-0031 round-trip audit sees nonzero drift (out-of-range clamps + records drift).
    assert conf_cases._page_offsets_resolve([bad_mention]) is False
    # And a faithful in-range mention still passes both.
    assert conf_cases._offsets_in_range([good_mention]) is True
    assert conf_cases._page_offsets_resolve([good_mention]) is True


def test_report_is_deterministic() -> None:
    """Two conformance sweeps over the same registry produce byte-identical reports (seeded)."""
    first = build_conformance_report(_registry()).model_dump_json()
    second = build_conformance_report(_registry()).model_dump_json()
    assert first == second


def test_adversarial_outcome_graceful_predicate() -> None:
    """An AdversarialOutcome is graceful iff it took any defined non-crash path."""
    assert AdversarialOutcome(raised=True).graceful
    assert AdversarialOutcome(quarantined=True).graceful
    assert AdversarialOutcome(parse_error=True).graceful
    assert AdversarialOutcome(empty=True).graceful
    assert AdversarialOutcome(noop=True).graceful
    assert not AdversarialOutcome().graceful


def test_conformance_env_is_cpu_pinned() -> None:
    """The suite runs with LATENCE_CUDA pinned to 0 (a reproducible CPU-only host)."""
    assert os.environ.get("LATENCE_CUDA") == "0"
