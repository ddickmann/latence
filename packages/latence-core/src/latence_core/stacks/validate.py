"""``latence stack validate <name>`` — the one-command stack-validation harness (ADR-0036 §3).

Goal A made real: a **blessed default stack** validated end-to-end by **one command**. This is
the G1 gate's E2E / reliability lanes generalized from a one-off gate into a **reusable** harness
— *composition* over the existing seams (``LocalRunner`` + ``QualityReportBuilder`` + the
device/skip seam + the P3-F3 conformance predicates), **not** new pipeline logic.

Given a stack config (a full-spine Pipeline YAML) it:

1. stages the small, bundled, deterministic messy corpus (:mod:`latence_core.stacks.corpus`);
2. runs the stack end-to-end on the local :class:`~latence_core.runner.LocalRunner`;
3. asserts the G1-style checks in one place, each a pass/fail :class:`CheckResult`:

   * **contracts** — every record carries Provenance + Classification and offsets align
     (reuses the ``QualityReportBuilder`` completeness check + the ADR-0031 zero-drift roll-up
     the run already computed — NOT a reinvented completeness/offset audit);
   * **kg** — a non-empty, content-addressed knowledge graph with per-edge Evidence is produced
     and exported (KG Parquet/TTL/GraphML on Storage) — skipped-clean when the stack declares no
     Graph-Assembly + KG-export stage (a stack may legitimately export only a RAG corpus);
   * **rag** — the redacted, chunked corpus is exported (JSONL/Parquet) and well-formed;
   * **report** — a schema-valid Quality Report is produced, persisted, counts-only (no PII),
     with per-stage metrics + the device/license/profile columns (P3-F1);
   * **determinism** — a second run under a *fresh* ``run_id`` produces a byte-identical export
     (reuses the H-E1 export-artifact equality discipline: compare the run-relative export bytes);
   * **resume** — re-running the *same* ``run_id`` resumes from checkpoints and reproduces a
     byte-identical export (kill-less resume is sufficient here);
   * **graceful-failure** — the planted malformed inputs are quarantined / PARSE_ERROR'd, never
     crashing the run (the run completed at all *is* the headline; the disposition counts confirm);
   * **device-honesty** — on a CPU-only host a GPU stack's learned stages are **skipped-with-flag**
     (recorded in the report, never faked), and the CPU-reachable spine still validates.

Output: a clear pass/fail per check + a signed-style Markdown summary (like ``G1-RESULTS.md``)
written to Storage, and a non-zero process exit on failure so it gates CI/publish.

The heavy imports (runner/stages) are pulled in here, at harness call time, so a bare
``import latence_core`` stays thin (the package ``__init__`` re-exports only the light contracts).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from latence_core.capability import ProviderRegistry
from latence_core.pipeline import CapabilityKind, Pipeline, Stage
from latence_core.providers.device import select_device
from latence_core.quality import QualityReport
from latence_core.storage import Storage

if TYPE_CHECKING:
    from collections.abc import Sequence

# The two run ids the harness uses. ``_RUN_PRIMARY`` is run twice (the resume check re-runs the
# same id); ``_RUN_DETERMINISM`` is a second, independent run under a fresh id whose export must
# be byte-identical to the primary's (the determinism check).
_RUN_PRIMARY = "stack-validate"
_RUN_DETERMINISM = "stack-validate-b"

# Placeholders a stack YAML uses so it is corpus- and storage-location-independent; the harness
# fills them with the concrete staging dir + storage root it chose (like the demo's EXAMPLES_DIR /
# STORAGE_URI templating). A stack with hard-coded absolute paths still runs, but using the
# placeholders is how the SAME committed stack validates in CI (tmp dirs) and on an adopter box.
_CORPUS_PLACEHOLDER = "{CORPUS_DIR}"
_STORAGE_PLACEHOLDER = "{STORAGE_URI}"

# The raw PII planted in the bundled corpus (mirrors the canonical fixture in stacks/corpus.py).
# NONE of these substrings may appear in ANY exported artifact — the masked corpus carries
# ``[EMAIL]``/``[PHONE]`` surrogates instead. Kept as one constant so every export-scan uses the
# same set (a drift here would silently narrow the PII gate).
_PLANTED_PII: tuple[str, ...] = ("alice@acme.example", "555-0142")


def _raw_pii_in_exports(storage: Storage, export_dir: str) -> str | None:
    """Scan EVERY exported artifact — not just the RAG JSONL — for any planted raw PII.

    A publish gate's PII check must cover the whole export surface: the RAG corpus (``.jsonl``),
    its Parquet twin, AND the KG artifacts (``.ttl`` / ``.graphml`` / ``*-nodes.parquet`` /
    ``*-edges.parquet``). The KG branch bypasses Redaction (entities→relations→graph→export), so a
    future Graph provider that inlined a raw source sentence into an edge would leak PII the
    JSONL-only scan never sees. Text artifacts are read as text; Parquet is read via pyarrow and
    every cell stringified, so a planted token cannot hide in a binary column. Returns a
    ``"<token> in <uri>"`` description on the first leak, else ``None``.
    """
    for uri in storage.walk_files(export_dir):
        haystack: str
        if uri.endswith(".parquet"):
            import io

            import pyarrow.parquet as pq

            table = pq.read_table(io.BytesIO(storage.read_bytes(uri)))  # type: ignore[no-untyped-call]
            haystack = " ".join(str(v) for col in table.columns for v in col.to_pylist())
        else:
            haystack = storage.read_text(uri)
        for token in _PLANTED_PII:
            if token in haystack:
                return f"{token!r} in {uri}"
    return None


@dataclass(frozen=True)
class CheckResult:
    """One named G1-style check's pass/fail outcome + a short human-readable detail.

    ``skipped`` marks a check that did not apply (e.g. the KG check on a stack with no
    Graph-Assembly stage, or a GPU stage skipped-with-flag on a CPU-only host) — a skip is
    NOT a failure (the device-honesty posture). ``passed`` is True for a genuine pass AND for a
    legitimate skip, so ``all(c.passed for c in checks)`` is the gate.
    """

    name: str
    passed: bool
    detail: str
    skipped: bool = False


@dataclass(frozen=True)
class StackValidationResult:
    """The full outcome of validating one stack: every check + the persisted summary URI.

    ``ok`` is the single gate the CLI turns into an exit code (0 / non-zero). ``skipped_stages``
    names the stages the device seam skipped-with-flag (the honest GPU-on-CPU gaps), surfaced so
    the summary reports them explicitly rather than papering over them.
    """

    stack_name: str
    ok: bool
    checks: list[CheckResult]
    document_count: int
    skipped_stages: list[str] = field(default_factory=list)
    summary_uri: str | None = None

    def render(self) -> str:
        """A signed-style Markdown summary (like ``G1-RESULTS.md``): pass/fail per check."""
        status = "PASS" if self.ok else "FAIL"
        lines = [
            f"# Stack validation — `{self.stack_name}` — **{status}**",
            "",
            f"- documents ingested: {self.document_count}",
        ]
        if self.skipped_stages:
            lines.append(
                "- device-honesty: stages skipped-with-flag on this host (no GPU / not faked): "
                + ", ".join(self.skipped_stages)
            )
        lines.append("")
        lines.append("| check | result | detail |")
        lines.append("|---|---|---|")
        for c in self.checks:
            mark = "SKIP" if c.skipped else ("PASS" if c.passed else "FAIL")
            lines.append(f"| {c.name} | {mark} | {c.detail} |")
        lines.append("")
        return "\n".join(lines)


def load_stack(stack_path: Path, storage_uri: str, corpus_dir: str) -> Pipeline:
    """Load a stack YAML, substituting the corpus/storage placeholders, into a validated Pipeline.

    ``corpus_dir`` is the Storage URI the harness staged the bundled corpus under;
    ``storage_uri`` is the run's state/artifact root. Substitution is a plain string replace of
    the two placeholders BEFORE parse, so the committed stack carries no machine-specific path.
    """
    text = stack_path.read_text(encoding="utf-8")
    text = text.replace(_CORPUS_PLACEHOLDER, corpus_dir).replace(_STORAGE_PLACEHOLDER, storage_uri)
    return Pipeline.from_yaml_text(text)


def _has_capability(pipeline: Pipeline, kind: CapabilityKind) -> bool:
    return any(s.capability == kind for s in pipeline.stages)


def _skip_reason_for_stage(stage: Stage, registry: ProviderRegistry) -> str | None:
    """Why this stage's provider is not runnable on THIS host (skip-with-flag), or None.

    The device-honesty posture (P3-F1 / the G1 skip-with-flag): a stack may name a
    learned/endpoint Provider that needs a GPU (or an inference endpoint) that a CPU-only host
    does not have. Two honest, non-faking skip conditions:

    * **not installed** — the Provider is not registered in this environment (its heavy package
      was not installed — ADR-0016 thin core: the ``gpu-sota`` learned Providers are NOT in the
      default CPU dev/CI env). Loading it would raise; the honest disposition is skip-with-flag,
      not a crash and never a fabricated number.
    * **GPU-only, no CUDA** — the Provider IS installed but its declared ``ProviderProfile`` +
      the stage's requested ``device`` resolve (through the SAME :func:`select_device` seam the
      Runner uses — not a weaker copy) to a skip on a CPU-only host.

    A CPU-runnable installed Provider returns ``None`` (runs normally). This never instantiates a
    Provider — it only reads the registry roster + the declared profile (a cheap class import).
    """
    if stage.provider not in registry.names():
        return (
            f"provider {stage.provider!r} not installed on this host "
            f"(needs its GPU/endpoint package)"
        )
    requested = str(stage.config.get("device", "auto"))
    try:
        profile = registry.profile(stage.provider)
    except Exception:  # noqa: BLE001 - triage a pre-flight load error by the requested device
        # The package is registered but importing it (or reading its profile) failed. Triage:
        # only a stage that *requested* a GPU/endpoint (``device: cuda|gpu``) is honestly
        # skipped-with-flag on a missing heavy dep — that is the gpu-sota posture. A CPU/auto stage
        # whose provider fails to load is a REAL breakage (a broken default stack), NOT a device
        # skip: return None so the Runner actually runs it and FAILs loudly. Silently skipping it
        # would green a broken blessed stack — the exact false-green this harness must prevent.
        if requested in ("cuda", "gpu"):
            return f"provider {stage.provider!r} failed to load on this host (missing GPU dep)"
        return None
    decision = select_device(requested, profile)
    if decision.skipped:
        return decision.reason
    # An Export stage may opt into a *nested* Embedder Provider (ADR-0017 opt-in augmentation) that
    # is itself GPU/endpoint-gated (e.g. embedding.sentence_transformers pinned device: cuda, or an
    # endpoint embedder whose package is absent). The embedder is not a top-level DAG node, so the
    # check above never sees it; inspect it through the SAME select_device seam so the embed
    # bake-off's ST candidate is honestly skipped-with-flag on a CPU host rather than run-and-fail
    # (W1-embed — the same device-honesty posture applied to the nested embedder).
    return _embedder_skip_reason(stage, registry)


def _embedder_skip_reason(stage: Stage, registry: ProviderRegistry) -> str | None:
    """Why this Export stage's nested ``config['embedder']`` Provider is not runnable here, or None.

    The Embedder is an opt-in Export sub-config (``{"provider": ..., "config": {...}}``), not a DAG
    Stage, so :func:`_skip_reason_for_stage`'s top-level check never inspects it. This applies the
    identical honest skip conditions to the nested embedder Provider: **not installed** (its heavy
    package is absent) OR **GPU-only, no CUDA** (its declared profile + the embedder config's
    requested ``device`` resolve to a skip through the SAME :func:`select_device` seam). A CPU/auto
    embedder whose package fails to load is a REAL breakage (return None ⇒ run + FAIL loudly), never
    silently skipped. A stage with no embedder returns None.
    """
    spec = stage.config.get("embedder")
    if not isinstance(spec, dict):
        return None
    provider = spec.get("provider")
    if not isinstance(provider, str) or not provider:
        return None
    if provider not in registry.names():
        return (
            f"embedder {provider!r} not installed on this host (needs its GPU/endpoint package)"
        )
    embedder_config = spec.get("config", {})
    requested = (
        str(embedder_config.get("device", "auto"))
        if isinstance(embedder_config, dict)
        else "auto"
    )
    try:
        profile = registry.profile(provider)
    except Exception:  # noqa: BLE001 - triage a nested-embedder load error by its requested device
        if requested in ("cuda", "gpu"):
            return f"embedder {provider!r} failed to load on this host (missing GPU dep)"
        return None
    decision = select_device(requested, profile)
    if decision.skipped:
        return f"embedder {provider!r}: {decision.reason}"
    return None


def _cpu_reachable_subpipeline(
    pipeline: Pipeline, registry: ProviderRegistry
) -> tuple[Pipeline, dict[str, str]]:
    """Reduce a stack to the sub-Pipeline whose Stages are runnable on THIS host + the skip map.

    On a CPU-only host a ``gpu-sota`` stack's learned Stages are skipped-with-flag (see
    :func:`_skip_reason_for_stage`); a Stage that *depends on* a skipped Stage is transitively
    dropped too (its input would be empty/absent). What remains is the **CPU-reachable spine** the
    harness still validates end-to-end — exactly the P3-F1 device posture: report the GPU Stages
    as skipped, never faked, and prove the CPU-reachable part.

    Returns the reduced :class:`Pipeline` (its ``depends_on`` edges pruned to the surviving Stages)
    and a ``{stage_name: skip_reason}`` map of every dropped Stage. A stack whose Stages are ALL
    CPU-runnable (the ``default`` stack) reduces to itself with an empty skip map.
    """
    dropped: dict[str, str] = {}
    for stage in pipeline.stages:
        reason = _skip_reason_for_stage(stage, registry)
        if reason is not None:
            dropped[stage.name] = reason

    # Transitively drop any Stage that depends (directly or indirectly) on a dropped Stage — its
    # input would be empty. Iterate to a fixed point over the dependency edges.
    changed = True
    while changed:
        changed = False
        for stage in pipeline.stages:
            if stage.name in dropped:
                continue
            if any(dep in dropped for dep in stage.depends_on):
                dropped[stage.name] = (
                    f"depends on skipped stage(s): "
                    f"{', '.join(d for d in stage.depends_on if d in dropped)}"
                )
                changed = True

    surviving = [s for s in pipeline.stages if s.name not in dropped]
    if not surviving:
        # Nothing runs (a fully-GPU stack on a CPU host) — return an empty-stage marker the caller
        # turns into an explicit device-honesty skip rather than constructing an invalid Pipeline
        # (which requires ≥1 stage).
        return pipeline.model_copy(), dropped
    reduced = pipeline.model_copy(
        update={
            "stages": [
                s.model_copy(update={"depends_on": [d for d in s.depends_on if d not in dropped]})
                for s in surviving
            ]
        }
    )
    return reduced, dropped


def _export_artifacts(storage: Storage, run_dir: str) -> dict[str, bytes]:
    """Every exported artifact of a run, keyed by its path relative to ``export/``.

    The H-E1 equality discipline (``tests/_resume_equality.py``): the exported corpus + KG bytes
    are the AI-ready output an adopter consumes; a corrupt resume / a nondeterministic run shows
    up here as different/dropped/reordered bytes. Keyed by the run-relative path so two runs under
    *different* run roots compare like-with-like (the determinism check). We deliberately exclude
    ``quality-report.json`` / ``wal.log`` / ``manifest.json`` — they legitimately differ across a
    resume (wall-clock durations, append-only log) and comparing them would make the check lie.
    """
    export_dir = f"{run_dir}/export"
    captured: dict[str, bytes] = {}
    if not storage.exists(export_dir):
        return captured
    prefix = _rel_key(export_dir) + "/"
    for uri in storage.walk_files(export_dir):
        bare = _rel_key(uri)
        rel = bare[len(prefix) :] if bare.startswith(prefix) else bare
        captured[rel] = storage.read_bytes(uri)
    return captured


def _rel_key(uri: str) -> str:
    """A scheme- and leading-slash-free key so file:// and object-store paths compare alike.

    Mirrors ``tests/_resume_equality._rel_key`` (different fsspec backends disagree on the leading
    slash of a bare path); kept here so the harness — production code — does not import a test
    helper.
    """
    bare = uri.split("://", 1)[1] if "://" in uri else uri
    return bare.lstrip("/")


def _pii_in_report(report_json: str) -> str | None:
    """Return a planted-PII token if it leaked into the persisted report, else None.

    The Quality Report is counts-only and safe to persist/share (S10). The bundled corpus plants
    a known email + phone in the injection document; neither must appear anywhere in the report
    bytes (a report that inlined record content would leak them). A cheap, decisive substring
    audit over the serialized report — no need to enumerate every field.
    """
    for token in ("alice@acme.example", "555-0142"):
        if token in report_json:
            return token
    return None


def _check_contracts(report: QualityReport) -> CheckResult:
    """Contracts/completeness: Provenance + Classification present + offsets align + zero drift.

    Reuses the completeness check the ``QualityReportBuilder`` already computed
    (``contracts_complete`` is ``completeness.passed``) and the ADR-0031 drift roll-up — no
    reinvented offset/completeness
    audit. A ``drift`` roll-up (present when the run parsed documents) must be ``clean`` (zero
    fuzzy recoveries — every chunk offset resolves exactly to its source page).
    """
    comp = report.completeness
    drift_clean = report.drift is None or report.drift.clean
    passed = report.contracts_complete and drift_clean
    detail = (
        f"completeness {comp.records_with_provenance}/{comp.records_checked} prov, "
        f"{comp.records_with_classification}/{comp.records_checked} class, "
        f"offsets_aligned={comp.offsets_aligned}, "
        f"drift={'clean' if drift_clean else 'RECOVERED'}"
    )
    return CheckResult(name="contracts", passed=passed, detail=detail)


def _check_kg(
    pipeline: Pipeline, report: QualityReport, storage: Storage, run_dir: str
) -> CheckResult:
    """KG: a non-empty content-addressed graph with per-edge Evidence is produced + exported.

    Skipped-clean when the stack declares no Graph-Assembly stage (a stack may export only a RAG
    corpus). When present: the report's ``graph`` roll-up must have nodes + edges with full
    Evidence coverage, and — if the stack also has a knowledge_graph Export — the KG artifacts
    (``*-nodes.parquet`` / ``*-edges.parquet`` / ``*.ttl`` / ``*.graphml``) must exist on Storage.
    """
    if not _has_capability(pipeline, CapabilityKind.GRAPH_ASSEMBLY):
        return CheckResult(
            name="kg", passed=True, skipped=True, detail="stack declares no Graph-Assembly stage"
        )
    graph = report.graph
    if graph is None or graph.nodes_total == 0 or graph.edges_total == 0:
        return CheckResult(
            name="kg",
            passed=False,
            detail=f"empty or missing KG (graph={graph})",
        )
    evidence_ok = graph.edges_with_evidence == graph.edges_total
    # If a knowledge_graph export exists, its files must be on Storage. Detect it by the presence
    # of any ``*-nodes.parquet`` under export/ (the KnowledgeGraphExport artifact stem).
    export_dir = f"{run_dir}/export"
    kg_files = []
    if storage.exists(export_dir):
        kg_files = [u for u in storage.walk_files(export_dir) if u.endswith("-nodes.parquet")]
    exported_ok = bool(kg_files)
    passed = evidence_ok and exported_ok
    detail = (
        f"nodes={graph.nodes_total}, edges={graph.edges_total}, "
        f"evidence_coverage={graph.evidence_coverage:.2f}, exported={exported_ok}"
    )
    return CheckResult(name="kg", passed=passed, detail=detail)


def _check_rag(pipeline: Pipeline, storage: Storage, run_dir: str) -> CheckResult:
    """RAG corpus: a chunked, (redacted-where-wired) corpus is exported JSONL/Parquet + well-formed.

    Asserts at least one ``*.jsonl`` corpus export exists under ``export/`` and its lines parse as
    JSON objects carrying Provenance (the AI-ready corpus a vector DB ingests). The paired Parquet
    is written by the same Export provider; JSONL well-formedness is the decisive, dependency-light
    check here (Parquet round-trip is separately covered by the export provider's own tests).

    Skipped-clean when the (device-reduced) pipeline has no Export Stage — e.g. a ``gpu-sota``
    stack on a CPU host whose only corpus Export depended on a skipped learned Redactor. That is
    the honest device posture: nothing to export in the CPU-reachable spine, not a failure.
    """
    if not _has_capability(pipeline, CapabilityKind.EXPORT):
        return CheckResult(
            name="rag",
            passed=True,
            skipped=True,
            detail="no Export stage in the CPU-reachable spine (export-feeding stages skipped)",
        )
    export_dir = f"{run_dir}/export"
    if not storage.exists(export_dir):
        return CheckResult(name="rag", passed=False, detail="no export/ directory produced")
    # Security invariant (whole export surface, not just the RAG JSONL): NO exported artifact —
    # corpus JSONL/Parquet OR the KG ttl/graphml/parquet — may carry the planted raw PII. A stack
    # that exported unredacted chunks (redaction skipped/mis-wired) OR a Graph provider that inlined
    # a raw source sentence into an edge leaks it, and MUST FAIL here (this is what makes Redaction
    # load-bearing and closes the KG-bypass gap). The masked corpus carries ``[EMAIL]``/``[PHONE]``.
    leak = _raw_pii_in_exports(storage, export_dir)
    if leak is not None:
        return CheckResult(
            name="rag",
            passed=False,
            detail=f"raw PII leaked into an exported artifact ({leak}) — redaction/KG bypass",
        )
    # A corpus JSONL is any exported .jsonl; the RAG corpus export writes ``<basename>.jsonl``.
    # Any well-formed record JSONL with Provenance qualifies as a RAG corpus export.
    jsonl_files = [u for u in storage.walk_files(export_dir) if u.endswith(".jsonl")]
    if not jsonl_files:
        return CheckResult(name="rag", passed=False, detail="no *.jsonl corpus exported")
    total_rows = 0
    for uri in jsonl_files:
        text = storage.read_text(uri)
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "provenance" not in row:
                return CheckResult(
                    name="rag",
                    passed=False,
                    detail=f"malformed RAG row (no provenance) in {uri}",
                )
            total_rows += 1
    passed = total_rows > 0
    return CheckResult(
        name="rag",
        passed=passed,
        detail=f"{total_rows} corpus rows across {len(jsonl_files)} JSONL export(s), no PII leak",
    )


def _check_report(report: QualityReport, report_json: str) -> CheckResult:
    """Quality Report: produced, schema-valid, counts-only (no PII), per-stage + profile columns.

    ``report_json`` is the persisted report bytes (already schema-valid — it round-tripped through
    ``QualityReport.model_validate_json`` to be read). The check asserts per-stage metrics carry
    the P3-F1 profile columns (at least one stage names its provider's compute/license) and that
    NO planted PII leaked into the report bytes (the counts-only, safe-to-share invariant).
    """
    leaked = _pii_in_report(report_json)
    if leaked is not None:
        return CheckResult(
            name="report", passed=False, detail=f"PII leaked into report: {leaked!r}"
        )
    if not report.stages:
        return CheckResult(
            name="report", passed=False, detail="report carries no per-stage metrics"
        )
    # At least one stage should surface a declared compute column (the P3-F1 device/profile
    # columns) — the in-core CPU references all declare compute="cpu".
    has_profile_column = any(m.compute is not None for m in report.stages)
    passed = has_profile_column
    detail = (
        f"schema v{report.schema_version}, {len(report.stages)} stage metrics, "
        f"profile_columns={has_profile_column}, no PII leak"
    )
    return CheckResult(name="report", passed=passed, detail=detail)


def _check_graceful_failure(report: QualityReport) -> CheckResult:
    """Graceful failure: the planted malformed inputs are quarantined / PARSE_ERROR'd, not crashed.

    The run completing at all (we reached this check) is the headline — a malformed input that
    crashed the run would have raised before here. The disposition counts confirm the fixtures were
    caught: the intake screener Quarantines the oversized / zip-bomb / type-spoof fixtures, so the
    report's screening roll-up must show quarantines. (A PARSE_ERROR path is the parser's graceful
    degradation for a corrupt file that slips past intake; either disposition is acceptable — the
    invariant is "handled, never crashed".)
    """
    screening = report.screening
    if screening is None:
        return CheckResult(
            name="graceful_failure",
            passed=False,
            detail="stack has no Screening stage — dangerous inputs unhandled",
        )
    quarantined = screening.quarantined
    parse_errors = 0 if report.parse is None else report.parse.parse_errors
    handled = quarantined + parse_errors
    passed = handled > 0
    detail = (
        f"run completed; {quarantined} quarantined + {parse_errors} PARSE_ERROR "
        f"(dangerous fixtures handled, never crashed)"
    )
    return CheckResult(name="graceful_failure", passed=passed, detail=detail)


def _check_determinism(
    pipeline: Pipeline, primary: dict[str, bytes], second: dict[str, bytes]
) -> CheckResult:
    """Determinism: a second run under a fresh run_id produced a byte-identical export.

    The H-E1 export-artifact equality: same input + config ⇒ byte-identical export. Compared by
    the run-relative export key so two independent run roots compare like-with-like. Skipped-clean
    when the device-reduced pipeline has no Export stage (nothing to compare — the CPU-reachable
    spine of a GPU stack produced no export).
    """
    if not _has_capability(pipeline, CapabilityKind.EXPORT):
        return CheckResult(
            name="determinism",
            passed=True,
            skipped=True,
            detail="no Export stage in the CPU-reachable spine — no artifacts to compare",
        )
    passed = primary == second and bool(primary)
    if not primary:
        detail = "no export artifacts captured to compare"
        passed = False
    elif passed:
        detail = f"{len(primary)} export artifact(s) byte-identical across two fresh runs"
    else:
        only_primary = sorted(set(primary) - set(second))
        only_second = sorted(set(second) - set(primary))
        diff_bytes = sorted(k for k in primary.keys() & second.keys() if primary[k] != second[k])
        detail = (
            f"export diverged — primary_only={only_primary}, second_only={only_second}, "
            f"byte_diff={diff_bytes}"
        )
    return CheckResult(name="determinism", passed=passed, detail=detail)


def _check_resume(
    pipeline: Pipeline, before: dict[str, bytes], after: dict[str, bytes]
) -> CheckResult:
    """Resume: re-running the same run_id reproduced a byte-identical export.

    Kill-less resume is sufficient here (the G1 chaos lane proves the SIGKILL path; this harness
    proves the resume-equality invariant cheaply). Every stage resumes from its checkpoint and the
    export bytes are unchanged. Skipped-clean when the device-reduced pipeline has no Export stage.
    """
    if not _has_capability(pipeline, CapabilityKind.EXPORT):
        return CheckResult(
            name="resume",
            passed=True,
            skipped=True,
            detail="no Export stage in the CPU-reachable spine — no artifacts to compare",
        )
    passed = before == after and bool(before)
    if not before:
        return CheckResult(name="resume", passed=False, detail="no export artifacts to compare")
    detail = (
        f"{len(before)} export artifact(s) byte-identical after same-run_id resume"
        if passed
        else "resumed export diverged from the first run"
    )
    return CheckResult(name="resume", passed=passed, detail=detail)


def validate_stack(
    stack_path: Path,
    *,
    storage_root: str,
    corpus_dir: str | None = None,
) -> StackValidationResult:
    """Run one stack end-to-end on the bundled corpus and assert the G1-style checks.

    ``storage_root`` is a Storage URI the harness owns for this validation (state, checkpoints,
    exports, and the staged corpus live under it). ``corpus_dir`` overrides where the bundled
    corpus is staged (defaults to ``<storage_root>/_validation-corpus``). Returns a
    :class:`StackValidationResult`; the caller (CLI) turns ``ok`` into a process exit code.

    Composition only: the run is a plain ``LocalRunner().run(...)`` and every check reads the
    Quality Report / exported artifacts the run already produced. No pipeline logic here.
    """
    from latence_core.runner import LocalRunner  # lazy: keep ``import latence_core`` thin

    storage = Storage.for_uri(storage_root)
    root = storage_root.rstrip("/")
    corpus_dir = corpus_dir or f"{root}/_validation-corpus"

    # Stage the bundled corpus (deterministic bytes) so the stack's Source reads it like a real
    # adopter folder.
    from latence_core.stacks.corpus import stage_validation_corpus

    stage_validation_corpus(storage, corpus_dir)

    declared = load_stack(stack_path, storage_uri=root, corpus_dir=corpus_dir)

    # One registry shared by the pre-flight AND the runner, so the "installed / CPU-runnable"
    # decision the harness makes is the exact roster the Runner would load against.
    registry = ProviderRegistry()

    # -- Device-honesty pre-flight (P3-F1 posture) ----------------------------
    # Reduce the stack to the sub-Pipeline whose Providers are installed + CPU-runnable on THIS
    # host; the learned/endpoint GPU Stages (and anything depending on them) are skipped-with-flag,
    # never faked, never crashed. The ``default`` stack (all in-core CPU Providers) reduces to
    # itself; a ``gpu-sota`` stack on a CPU-only host reduces to its CPU-reachable spine.
    pipeline, dropped = _cpu_reachable_subpipeline(declared, registry)
    skipped_stages = sorted(dropped)

    if all(s.name in dropped for s in declared.stages):
        # Nothing CPU-reachable (a fully-GPU stack on a CPU host). Not a failure: report every
        # Stage skipped-with-flag + a clean device-honesty pass, no run attempted (no fake numbers).
        checks = [
            CheckResult(
                name="device_honesty",
                passed=True,
                skipped=True,
                detail=f"all stages skipped-with-flag on this host: {', '.join(skipped_stages)}",
            )
        ]
        result = StackValidationResult(
            stack_name=Path(stack_path).name,
            ok=True,
            checks=checks,
            document_count=0,
            skipped_stages=skipped_stages,
        )
        summary_uri = f"{root}/_latence/stack-validation/{result.stack_name}.md"
        storage.write_atomic(summary_uri, result.render().encode("utf-8"))
        return StackValidationResult(
            stack_name=result.stack_name,
            ok=True,
            checks=checks,
            document_count=0,
            skipped_stages=skipped_stages,
            summary_uri=summary_uri,
        )

    runner = LocalRunner(registry)

    # -- Primary run (also the resume subject) --------------------------------
    # A broken stack (wrong wiring, a Provider that raises) must make the harness FAIL — not
    # crash it. Any run exception is caught and turned into a ``run`` FAIL check + a non-zero exit
    # (the harness bites). The run failure still persists a FAIL Quality Report on Storage, but we
    # do not depend on reading it here — the caught exception IS the signal.
    try:
        runner.run(pipeline, run_id=_RUN_PRIMARY)
        primary_run_dir = runner.run_dir(pipeline, _RUN_PRIMARY)
        primary_export = _export_artifacts(storage, primary_run_dir)
        report_json = storage.read_text(runner.report_uri(pipeline, _RUN_PRIMARY))
        # Round-trip to prove the persisted report is schema-valid (raises on a schema violation).
        persisted_report = QualityReport.model_validate_json(report_json)

        # -- Resume: re-run the SAME run_id; every stage resumes; export unchanged -
        runner.run(pipeline, run_id=_RUN_PRIMARY)
        resumed_export = _export_artifacts(storage, primary_run_dir)

        # -- Determinism: a second, independent run under a FRESH run_id ----------
        runner.run(pipeline, run_id=_RUN_DETERMINISM)
        determinism_run_dir = runner.run_dir(pipeline, _RUN_DETERMINISM)
        determinism_export = _export_artifacts(storage, determinism_run_dir)
    except Exception as exc:  # noqa: BLE001 - a broken stack must FAIL the harness, not crash it
        checks = [
            CheckResult(
                name="run",
                passed=False,
                detail=f"stack run raised {type(exc).__name__}: {exc}",
            )
        ]
        result = StackValidationResult(
            stack_name=Path(stack_path).name,
            ok=False,
            checks=checks,
            document_count=0,
            skipped_stages=skipped_stages,
        )
        summary_uri = f"{root}/_latence/stack-validation/{result.stack_name}.md"
        storage.write_atomic(summary_uri, result.render().encode("utf-8"))
        return StackValidationResult(
            stack_name=result.stack_name,
            ok=False,
            checks=checks,
            document_count=0,
            skipped_stages=skipped_stages,
            summary_uri=summary_uri,
        )

    # A stage the RUNNER itself skipped-with-flag (an installed GPU Provider on a CPU host) adds to
    # the harness pre-flight skips, so the device-honesty roll-up is complete either way.
    runner_skipped = [m.name for m in persisted_report.stages if m.skipped]
    skipped_stages = sorted(set(skipped_stages) | set(runner_skipped))

    checks = [
        _check_contracts(persisted_report),
        _check_kg(pipeline, persisted_report, storage, primary_run_dir),
        _check_rag(pipeline, storage, primary_run_dir),
        _check_report(persisted_report, report_json),
        _check_graceful_failure(persisted_report),
        _check_determinism(pipeline, primary_export, determinism_export),
        _check_resume(pipeline, primary_export, resumed_export),
    ]

    # Device-honesty: a run whose GPU stages were skipped-with-flag is a PASS with the skips
    # surfaced (never a failure, never faked). Recorded as its own check so the summary is explicit.
    device_detail = (
        "skipped-with-flag (no GPU / not installed, not faked): "
        + "; ".join(f"{name} [{dropped.get(name, 'runner-skipped')}]" for name in skipped_stages)
        if skipped_stages
        else "no GPU stages skipped (all providers CPU-reachable on this host)"
    )
    checks.append(
        CheckResult(
            name="device_honesty",
            passed=True,
            detail=device_detail,
            skipped=bool(skipped_stages),
        )
    )

    ok = all(c.passed for c in checks)
    result = StackValidationResult(
        stack_name=Path(stack_path).name,
        ok=ok,
        checks=checks,
        document_count=persisted_report.document_count,
        skipped_stages=skipped_stages,
    )

    # Persist a signed-style summary alongside the run (like G1-RESULTS.md), so a failing check is
    # inspectable after the fact and CI can archive it.
    summary_uri = f"{root}/_latence/stack-validation/{result.stack_name}.md"
    storage.write_atomic(summary_uri, result.render().encode("utf-8"))
    return StackValidationResult(
        stack_name=result.stack_name,
        ok=result.ok,
        checks=result.checks,
        document_count=result.document_count,
        skipped_stages=result.skipped_stages,
        summary_uri=summary_uri,
    )


def _describe_failures(checks: Sequence[CheckResult]) -> str:
    """One-line-per-failed-check description for the CLI's stderr on a non-zero exit."""
    return "\n".join(f"  - {c.name}: {c.detail}" for c in checks if not c.passed and not c.skipped)
