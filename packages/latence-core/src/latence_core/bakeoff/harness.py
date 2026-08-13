"""The bake-off harness — same corpus + gold-set, one Stage swapped (ADR-0036 §4, P3-F5).

Goal B made real: the per-Stage bake-off. Given a ``matrix/<stage>.yaml``, this holds every OTHER
Stage at ``stacks/default.yaml`` and swaps in each candidate for the target Stage, running the SAME
bundled corpus (+ gold-set where the Stage has one) through each candidate and recording the
ADR-0036 §4 columns per candidate: quality, latency/throughput, peak memory (measured vs declared),
cost, license, determinism (declared + cross-checked), error rate.

This is **composition** over the seams the earlier Wave-0 slices built — it reinvents NONE of them:

* the **F4 stack machinery** (:mod:`latence_core.stacks.corpus` for the bundled corpus,
  :func:`~latence_core.stacks.validate.load_stack` for the placeholder-filled base stack, and its
  device pre-flight seam) holds the other stages constant;
* the **S10 gold-set scorer** — the Quality Report the run already computes carries the
  :class:`~latence_core.quality.GoldSetEvaluation` when the base stack declares a ``gold_set_uri``;
  the bake-off READS the target Stage's score off it (no reinvented P/R/F1);
* the **H-E1 export/checkpoint equality** discipline cross-checks a candidate's declared
  determinism by comparing the target Stage's checkpoint bytes across two independent runs;
* the shared **``select_device`` seam** device-gates GPU/endpoint candidates (skipped-with-flag on a
  CPU-only host, never a fabricated number — #64).

The bake-off degrades **per candidate**: a candidate that crashes / fails conformance is recorded as
a failed row, never aborting the whole matrix. Heavy imports (runner/stages) are pulled in at call
time so a bare ``import latence_core`` stays thin.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING

from latence_core.bakeoff.matrix import MatrixSpec
from latence_core.bakeoff.report import (
    CandidateResult,
    DeterminismColumn,
    LicenseColumn,
    MatrixResult,
    MemoryMeasurement,
    QualityMetric,
)
from latence_core.capability import ProviderRegistry
from latence_core.pipeline import CapabilityKind, Pipeline, Stage
from latence_core.quality import QualityReport, StageMetrics
from latence_core.run_store import RunLayout
from latence_core.stacks.validate import _skip_reason_for_stage, load_stack
from latence_core.storage import Storage

if TYPE_CHECKING:
    from latence_core.providers.profile import ProviderProfile

# The default base stack whose OTHER stages are held constant — the blessed CPU-first stack (F4).
# The harness takes an explicit ``base_stack_path`` (repo-root resolution at runtime is fragile);
# this is only the file *name* the CLI joins to the repo's ``stacks/`` dir.
DEFAULT_BASE_STACK = "stacks/default.yaml"

# The two run ids a candidate uses: ``_RUN_A`` is the measured run (quality/latency/memory/errors
# are read from it); ``_RUN_B`` is the second, independent run whose target-Stage checkpoint must be
# byte-identical to ``_RUN_A``'s for the determinism cross-check (the H-E1 equality). Namespaced per
# candidate so two candidates never collide on Storage.
_RUN_A = "bakeoff-a"
_RUN_B = "bakeoff-b"


def _restricted_opt_in_license(profile: ProviderProfile | None) -> bool:
    """Whether a Provider's license is restricted-but-permitted (opt-in), per ADR-0012.

    A permissive SPDX (Apache-2.0/MIT/BSD/…) is not opt-in. A Gemma-family (or other
    source-available/restricted-but-permitted) license IS opt-in — the adopter accepts a flow-down
    obligation. We detect it conservatively from the declared SPDX substring rather than guessing;
    an ``UNVERIFIED`` license is not treated as opt-in (it is an honest gap, surfaced separately).
    """
    if profile is None:
        return False
    spdx = profile.license.lower()
    # Gemma is the one restricted-but-opt-in license the workspace ships (EmbeddingGemma, ADR-0012).
    return "gemma" in spdx


def _swap_target_stage(
    base: Pipeline,
    stage_name: str,
    candidate_provider: str,
    candidate_config: dict[str, object],
    *,
    embedder_swap: bool = False,
) -> Pipeline:
    """Return a copy of ``base`` with ONE Stage's Provider swapped + its config overridden.

    The controlled comparison: every other Stage (its Provider, config, and DAG edges) is left
    byte-identical; only the target Stage's ``provider`` and (shallow-merged) ``config`` change.
    The candidate's ``config`` is merged ONTO the base stage's config so a candidate overrides only
    the keys it names (e.g. ``device: cuda``) and inherits the rest — the base stack stays the
    control. A brand-new provider gets a fresh config from the base + the override.

    When ``embedder_swap`` is True (the EMBEDDING target, W1-embed), the candidate is the opt-in
    *Embedder* Provider, NOT the Export Provider: the Export node named by ``stage_name`` keeps its
    own Provider and every other config key, and only its ``config['embedder']`` block is replaced
    with ``{"provider": candidate_provider, "config": candidate_config}``. This makes the Embedder a
    proper bake-off dimension (each candidate a distinct embedder) without changing the Export
    Provider — a genuine controlled comparison of the two embedders (the Export augmentation is not
    a DAG Stage, ADR-0017, so it is swapped inside the Export config, not as a top-level node).
    """
    new_stages: list[Stage] = []
    for stage in base.stages:
        if stage.name == stage_name:
            if embedder_swap:
                merged = {
                    **stage.config,
                    "embedder": {"provider": candidate_provider, "config": candidate_config},
                }
                new_stages.append(stage.model_copy(update={"config": merged}))
            else:
                merged = {**stage.config, **candidate_config}
                new_stages.append(
                    stage.model_copy(update={"provider": candidate_provider, "config": merged})
                )
        else:
            new_stages.append(stage.model_copy())
    return base.model_copy(update={"stages": new_stages})


def _resolve_stage_name(base: Pipeline, spec: MatrixSpec) -> str:
    """The base-stack node to swap: the spec's ``stage_name``, or the unique Stage of the kind.

    When ``stage_name`` is given it must name a real Stage of the target Capability. When omitted it
    is inferred as the sole Stage of ``spec.stage`` in the base stack; an ambiguous base stack (two
    Stages of the kind — e.g. two Export nodes) requires an explicit ``stage_name`` and raises here
    rather than silently swapping the wrong one.
    """
    # The EMBEDDING target is special: the Embedder is an OPT-IN Export augmentation (ADR-0017),
    # not a DAG Stage, so a base stack has NO `embedding` node. The bake-off swaps the embedder
    # INSIDE an Export node's `config['embedder']`, so `stage_name` must name a real Export node
    # (required, since a stack typically has >1 Export). We validate it is an Export node here and
    # let the harness do the nested embedder swap (W1-embed).
    if spec.stage == CapabilityKind.EMBEDDING:
        if spec.stage_name is None:
            msg = (
                "an embedding matrix must set stage_name to the Export node whose embedder is "
                "swapped (the Embedder is an opt-in Export augmentation, not a DAG Stage)."
            )
            raise ValueError(msg)
        match = [s for s in base.stages if s.name == spec.stage_name]
        if not match:
            msg = f"matrix stage_name {spec.stage_name!r} is not a Stage in the base stack."
            raise ValueError(msg)
        if match[0].capability != CapabilityKind.EXPORT:
            msg = (
                f"an embedding matrix's stage_name {spec.stage_name!r} must name an Export node "
                f"(the embedder lives in its config), not a {match[0].capability.value!r} Stage."
            )
            raise ValueError(msg)
        return spec.stage_name
    if spec.stage_name is not None:
        match = [s for s in base.stages if s.name == spec.stage_name]
        if not match:
            msg = f"matrix stage_name {spec.stage_name!r} is not a Stage in the base stack."
            raise ValueError(msg)
        if match[0].capability != spec.stage:
            msg = (
                f"matrix stage_name {spec.stage_name!r} has capability "
                f"{match[0].capability.value!r}, not the target {spec.stage.value!r}."
            )
            raise ValueError(msg)
        return spec.stage_name
    of_kind = [s for s in base.stages if s.capability == spec.stage]
    if not of_kind:
        msg = f"base stack has no Stage of capability {spec.stage.value!r} to bake off."
        raise ValueError(msg)
    if len(of_kind) > 1:
        names = ", ".join(s.name for s in of_kind)
        msg = (
            f"base stack has multiple {spec.stage.value!r} Stages ({names}); "
            "set matrix.stage_name to disambiguate which to swap."
        )
        raise ValueError(msg)
    return of_kind[0].name


def _target_metrics(report: QualityReport, stage_name: str) -> StageMetrics | None:
    """The target Stage's :class:`StageMetrics` from a run's Quality Report, or ``None``."""
    for m in report.stages:
        if m.name == stage_name:
            return m
    return None


def _quality_of(report: QualityReport, spec: MatrixSpec, stage_name: str) -> QualityMetric | None:
    """The target Stage's quality — the S10 gold-set score if present, else a stage-specific metric.

    Reuses the Quality Report the run ALREADY produced (no reinvented scoring): for a gold-set Stage
    (entity/relation, when the base stack declares a ``gold_set_uri``) it reads the matching
    :class:`~latence_core.quality.StageScore` off ``report.goldset``; for a Stage with no gold set
    it pulls the stage-specific metric off the report's per-Stage rollup — parse: chars-extracted +
    offset-drift-recovered (``report.parse``); chunk: chunk count + offset-preserving fraction
    (``report.chunk``). Returns ``None`` only when the report carries neither (a Stage the bake-off
    has no quality lens for — the row still records latency/memory/etc.).
    """
    gold_stage_key = {
        CapabilityKind.ENTITY_EXTRACTION: "entities",
        CapabilityKind.RELATION_EXTRACTION: "relations",
        CapabilityKind.FUSED_ENTITY_RELATION: "entities",
    }.get(spec.stage)
    if gold_stage_key is not None and report.goldset is not None:
        for score in report.goldset.stages:
            if score.stage == gold_stage_key:
                return QualityMetric(
                    kind="goldset",
                    primary_name="f1",
                    primary=score.f1,
                    precision=score.precision,
                    recall=score.recall,
                    f1=score.f1,
                )
    if spec.stage == CapabilityKind.PARSE and report.parse is not None:
        p = report.parse
        return QualityMetric(
            kind="metric",
            primary_name="chars",
            primary=float(p.bytes_after_parse),
            metrics={
                "chars": float(p.bytes_after_parse),
                "documents_parsed": float(p.documents_parsed),
                "parse_errors": float(p.parse_errors),
                "offset_drift_recovered": float(p.page_drift_recovered),
            },
        )
    if spec.stage == CapabilityKind.CHUNK and report.chunk is not None:
        ch = report.chunk
        coverage = (
            ch.chunks_offset_preserving / ch.chunks_total if ch.chunks_total else 0.0
        )
        return QualityMetric(
            kind="metric",
            primary_name="chunks",
            primary=float(ch.chunks_total),
            metrics={
                "chunks": float(ch.chunks_total),
                "tokens": float(ch.tokens_total),
                "offset_preserving_fraction": round(coverage, 6),
            },
        )
    if spec.stage == CapabilityKind.DISAMBIGUATION and report.disambiguation is not None:
        # The enterprise ER lens (W1 rework, #94): reuse the DisambiguationQuality the run ALREADY
        # computed (no reinvented scoring). The primary metric is the cross-surface merge rate —
        # merges_applied / mentions_in — the signal the whole slice exists to move: a bare-surface
        # embedder over-merges (a HIGH but WRONG rate collapsing distinct entities), a
        # surface+context embedder merges genuine duplicates while merges_below_policy captures the
        # audited non-merges. A real precision/recall lands here when a Disambiguation gold-set is
        # declared; until then these honest counts are the comparison, never a fabricated P/R.
        dq = report.disambiguation
        merge_rate = (dq.merges_applied / dq.mentions_in) if dq.mentions_in else 0.0
        return QualityMetric(
            kind="metric",
            primary_name="merge_rate",
            primary=round(merge_rate, 6),
            metrics={
                "mentions_in": float(dq.mentions_in),
                "entities_out": float(dq.entities_out),
                "entities_merged": float(dq.entities_merged),
                "merges_applied": float(dq.merges_applied),
                "merges_below_policy": float(dq.merges_below_policy),
                "entities_linked": float(dq.entities_linked),
                "avg_entity_confidence": round(dq.avg_entity_confidence, 6),
            },
        )
    return None


class BakeOffHarness:
    """Runs one ``matrix/<stage>.yaml`` end-to-end and produces its :class:`MatrixResult`.

    Constructed with the base stack path + a Storage root it owns for the bake-off (state,
    checkpoints, exports, and the staged corpus live under it). :meth:`run` iterates the matrix's
    candidates, swapping each into the base stack's target Stage, measuring the ADR-0036 §4 columns,
    and degrading per-candidate (a crash is a failed row, not an aborted matrix).
    """

    def __init__(
        self,
        base_stack_path: Path,
        *,
        storage_root: str,
        registry: ProviderRegistry | None = None,
    ) -> None:
        self._base_stack_path = base_stack_path
        self._root = storage_root.rstrip("/")
        self._storage = Storage.for_uri(storage_root)
        # One registry shared by the device pre-flight AND every candidate run, so the
        # installed/CPU-runnable decision is the exact roster the Runner loads against.
        self._registry = registry or ProviderRegistry()

    def run(self, spec: MatrixSpec) -> MatrixResult:
        """Bake off every candidate for the target Stage; return the full :class:`MatrixResult`."""
        # Stage the bundled corpus once (deterministic bytes) — the SAME corpus every candidate.
        from latence_core.stacks.corpus import stage_validation_corpus

        corpus_dir = f"{self._root}/_bakeoff-corpus"
        manifest = stage_validation_corpus(self._storage, corpus_dir)
        document_count = len(manifest.documents)

        base = load_stack(self._base_stack_path, storage_uri=self._root, corpus_dir=corpus_dir)
        stage_name = _resolve_stage_name(base, spec)
        embedder_swap = spec.stage == CapabilityKind.EMBEDDING

        results: list[CandidateResult] = []
        for candidate in spec.candidates:
            results.append(
                self._bake_candidate(
                    base, spec, stage_name, candidate.provider, candidate.config,
                    embedder_swap=embedder_swap,
                )
            )

        return MatrixResult(
            stage=spec.stage.value,
            stage_name=stage_name,
            base_stack=self._base_stack_path.name,
            document_count=document_count,
            candidates=results,
        )

    def _bake_candidate(
        self,
        base: Pipeline,
        spec: MatrixSpec,
        stage_name: str,
        provider: str,
        config: dict[str, object],
        *,
        embedder_swap: bool = False,
    ) -> CandidateResult:
        """Measure one candidate — or record it skipped-with-flag / failed (degrade per-candidate).

        Steps: (1) build the swapped pipeline; (2) device pre-flight — a GPU/endpoint candidate not
        runnable on this host is skipped-with-flag (no fabricated numbers, #64); (3) run A under
        ``tracemalloc`` (quality/latency/memory/errors read from its Quality Report); (4) run B for
        the determinism cross-check (target-Stage checkpoint byte-identity, the H-E1 equality);
        (5) assemble the row. Any exception in the run path is caught and recorded as a failed row.
        """
        # Per-candidate Storage namespace so two candidates never collide on run state.
        cand_root = f"{self._root}/candidates/{provider.replace('.', '_')}"
        cand_storage = Storage.for_uri(cand_root)
        swapped = _swap_target_stage(
            base, stage_name, provider, config, embedder_swap=embedder_swap
        )
        # Re-point the swapped pipeline's storage at the per-candidate root so runs are isolated.
        swapped = swapped.model_copy(update={"storage_uri": cand_root})
        target_stage = next(s for s in swapped.stages if s.name == stage_name)

        profile = self._profile_or_none(provider)
        license_col = self._license_column(profile)

        # -- (2) device pre-flight: skip-with-flag a GPU/endpoint candidate on a CPU-only host --
        skip_reason = _skip_reason_for_stage(target_stage, self._registry)
        if skip_reason is not None:
            return CandidateResult(
                provider=provider,
                skipped=True,
                skip_reason=f"{skip_reason} (needs a rig, #64)",
                license=license_col,
                cost_per_1k=profile.cost_per_1k if profile is not None else None,
                compute=profile.compute if profile is not None else None,
                model_id=profile.model_id if profile is not None else None,
                determinism=DeterminismColumn(
                    declared=profile.deterministic if profile is not None else None,
                    checked=False,
                ),
            )

        # -- (3) measured run A under tracemalloc --
        try:
            from latence_core.runner import LocalRunner

            runner = LocalRunner(self._registry)
            tracemalloc.start()
            try:
                report_a = runner.run(swapped, run_id=_RUN_A)
            finally:
                _current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            # -- (4) run B for the determinism cross-check (fresh run id, independent state) --
            runner.run(swapped, run_id=_RUN_B)
            checkpoint_a = self._stage_checkpoint_bytes(cand_storage, swapped, _RUN_A, stage_name)
            checkpoint_b = self._stage_checkpoint_bytes(cand_storage, swapped, _RUN_B, stage_name)
        except Exception as exc:  # noqa: BLE001 - a bad candidate is a failed row, not an aborted matrix
            # Cap the detail: it lands in the (potentially repo-committed) MATRIX-RESULTS artifact,
            # and a native exception run against a real-PII base stack could otherwise echo input
            # content into it. Type name + a bounded message is enough to triage a bad candidate.
            detail = f"{type(exc).__name__}: {exc}"
            if len(detail) > 200:
                detail = detail[:197] + "..."
            return CandidateResult(
                provider=provider,
                failed=True,
                failure_detail=detail,
                license=license_col,
                compute=profile.compute if profile is not None else None,
                model_id=profile.model_id if profile is not None else None,
                determinism=DeterminismColumn(
                    declared=profile.deterministic if profile is not None else None,
                    checked=False,
                ),
            )

        metrics = _target_metrics(report_a, stage_name)
        stage_seconds = metrics.duration_seconds if metrics is not None else None
        error_count = metrics.error_count if metrics is not None else 0
        documents = report_a.document_count
        # Throughput is the TARGET stage's own rate, so divide by what the stage actually
        # processed (records_in, e.g. post-intake-quarantine), not the whole-corpus document
        # count — dividing by the corpus size overstates a stage that ran on fewer records.
        stage_records = metrics.records_in if metrics is not None else documents
        throughput = (
            round(stage_records / stage_seconds, 3)
            if stage_seconds is not None and stage_seconds > 0.0 and stage_records > 0
            else None
        )

        observed_identical = checkpoint_a == checkpoint_b
        determinism = DeterminismColumn(
            declared=profile.deterministic if profile is not None else None,
            observed_identical=observed_identical,
            checked=True,
        )

        memory = MemoryMeasurement(
            measured_mb=round(peak / (1024 * 1024), 6),
            declared_mb=profile.memory_mb if profile is not None else None,
        )

        return CandidateResult(
            provider=provider,
            quality=_quality_of(report_a, spec, stage_name),
            documents=documents,
            stage_seconds=stage_seconds,
            throughput_docs_per_s=throughput,
            memory=memory,
            cost_per_1k=profile.cost_per_1k if profile is not None else None,
            license=license_col,
            determinism=determinism,
            error_count=error_count,
            compute=profile.compute if profile is not None else None,
            model_id=profile.model_id if profile is not None else None,
        )

    def _profile_or_none(self, provider: str) -> ProviderProfile | None:
        """The candidate's declared profile, or ``None`` — never fabricated (a load error → None).

        Reading a profile loads the Provider class; a candidate whose heavy package is not installed
        would raise, but the device pre-flight already skips those, so this is only reached for an
        installed candidate. Guarded anyway so a malformed profile is a ``None`` here rather than a
        crash that aborts the whole matrix (the failure surfaces as the failed run row instead).
        """
        try:
            return self._registry.profile(provider)
        except Exception:  # noqa: BLE001 - a profile load error must not abort the whole matrix
            return None

    def _license_column(self, profile: ProviderProfile | None) -> LicenseColumn:
        """The license decision column read from the declared profile (never fabricated)."""
        if profile is None:
            return LicenseColumn()
        return LicenseColumn(
            spdx=profile.license,
            verified=profile.license_verified,
            opt_in=_restricted_opt_in_license(profile),
            source=profile.license_source,
        )

    @staticmethod
    def _stage_checkpoint_bytes(
        storage: Storage, pipeline: Pipeline, run_id: str, stage_name: str
    ) -> bytes:
        """The target Stage's checkpoint bytes for one run (the H-E1 byte-identity subject).

        The Stage checkpoint (``<run_dir>/checkpoints/<stage_name>.jsonl``) is the newline-delimited
        ``model_dump_json`` record set the Stage emitted — the deterministic G1 fingerprint. Two
        runs of a genuinely-deterministic Provider produce byte-identical checkpoints; a Provider
        declares ``deterministic=True`` but reorders/varies its output diverges here (the check that
        BITES). A Stage that wrote no checkpoint (all inputs quarantined) yields ``b""`` for both
        runs — identical, so an empty-but-stable Stage is honestly deterministic.
        """
        uri = RunLayout(pipeline.storage_uri, run_id).checkpoint_uri(stage_name)
        if not storage.exists(uri):
            return b""
        return storage.read_bytes(uri)


def run_bakeoff(
    matrix_path: Path,
    *,
    base_stack_path: Path,
    storage_root: str,
    registry: ProviderRegistry | None = None,
) -> MatrixResult:
    """Load a ``matrix/<stage>.yaml`` + run the bake-off, returning the :class:`MatrixResult`.

    The single functional entry point the CLI drives. ``base_stack_path`` is the stack holding the
    other stages constant (the blessed ``stacks/default.yaml`` by default, or the matrix's
    ``base_stack`` override). ``storage_root`` is a Storage URI the harness owns for the run.
    """
    spec = MatrixSpec.from_yaml_text(matrix_path.read_text(encoding="utf-8"))
    # The matrix may override which stack holds the other stages constant.
    resolved_base = base_stack_path
    if spec.base_stack is not None:
        resolved_base = (matrix_path.parent / ".." / spec.base_stack).resolve()
    harness = BakeOffHarness(resolved_base, storage_root=storage_root, registry=registry)
    return harness.run(spec)
