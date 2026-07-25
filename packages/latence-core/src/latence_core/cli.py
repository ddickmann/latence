"""``latence`` CLI (typer).

* ``latence run <pipeline.yaml>`` — execute a Pipeline on the local Runner (S1).
* ``latence stack validate <stack.yaml>`` — run a named stack end-to-end on the bundled
  corpus and assert the G1-style checks (contracts/KG/RAG/report + determinism + resume +
  graceful-failure + device-honesty), write a signed-style summary, and exit non-zero on
  failure so it gates CI/publish (ADR-0036 §3, P3-F4).
* ``latence bake-off <matrix.yaml>`` — run a per-Stage provider bake-off: hold every OTHER
  Stage at ``stacks/default.yaml`` and run the SAME bundled corpus through each candidate for
  ONE Stage, writing a signed-style ``MATRIX-RESULTS-<stage>.md`` + a JSON companion with
  per-candidate quality/latency/memory/cost/license/determinism/error-rate (ADR-0036 §4, P3-F5).
* ``latence tune <stack.yaml>`` — profile the device, sweep each heavy Provider's perf knobs
  over a small real sample under a HARD correctness gate (reject any setting that crashes —
  incl. inference-time ``BackendCompilerFailed`` — or changes output), pick the fastest eligible
  per Provider, and write a device-fingerprinted ``tuned/<device>.yaml`` + an honest report
  (winners AND rejected knobs). ``latence run --tuned tuned/<device>.yaml`` overlays it ONLY on
  a matching device_fingerprint, else warns + uses stack defaults (ADR-0036, W11-autotune).
* ``latence delta <stack.yaml>`` / ``retract`` / ``purge`` — incremental corpus updates over the
  EXISTING :class:`~latence_core.delta_runner.DeltaRunner` (S11, ADR-0029): ``delta`` adds/updates
  from the current source and commits the next Corpus Version; ``retract`` soft-deletes documents
  (auditable, rollback-able); ``purge`` is the hard GDPR erasure (guarded by ``--yes``). Since W17
  (ADR-0043) ``delta`` extraction is INCREMENTAL: the induced schema is frozen and each unchanged
  document's records are reused, so only new/changed documents are extracted (``--reinduce`` is the
  O(corpus) escape hatch). The *committed* corpus stays affected-set-scoped (issue #42, ADR-0018).
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from latence_core.errors import LatenceError, error_category
from latence_core.observability import (
    DEFAULT_LINEAGE_NAMESPACE,
    OpenLineageEmitter,
    build_lineage_emitter,
)
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

if TYPE_CHECKING:
    from latence_core.quality import QualityReport

app = typer.Typer(add_completion=False, help="Latence — AI-ready data pipelines.")

# The ``stack`` sub-command group (``latence stack validate ...``). A Typer sub-app keeps the
# stack tooling namespaced without cluttering the top-level command list.
stack_app = typer.Typer(add_completion=False, help="Full-stack Pipeline configs + validation.")
app.add_typer(stack_app, name="stack")

# The ``env`` sub-command group (``latence env verify|render ...``, issue #195). The reproducible
# environments under ``deploy/env/<name>/`` are specifications the build checks, not wiki pages.
env_app = typer.Typer(
    add_completion=False, help="Reproducible environment specs (deploy/env/<name>/)."
)
app.add_typer(env_app, name="env")

# The ``retrieval`` sub-command group (``latence retrieval splade-index ...``, tutorial §4.2). The
# stateless (ADR-0048) index-time tooling that builds a retrieval signal over an existing Export —
# today the SPLADE-v3 sparse leg. The heavy model stack lives in the ``sparse.splade`` Provider
# package (ADR-0016); this group resolves it through the Provider registry, so the thin core names
# the capability, not ``latence-splade``.
retrieval_app = typer.Typer(
    add_completion=False, help="Stateless retrieval-index tooling over an Export (ADR-0048)."
)
app.add_typer(retrieval_app, name="retrieval")

logger = logging.getLogger("latence")


@app.callback()
def _root() -> None:
    """Latence — turn messy enterprise data into AI-ready data."""
    # Presence of a callback forces a multi-command app so `latence run ...`
    # (rather than a collapsed single-command form) is the invocation.


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


# A content-addressed document id (the sha256-of-bytes id the Source stamps + `retract`/`purge`
# take). Kept lenient-but-shaped: a `sha256:` prefix over 64 lowercase hex chars.
_DOC_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _load_pipeline(pipeline_path: Path) -> Pipeline:
    """Load a Pipeline from a stack/pipeline YAML (the exact loader ``run`` uses).

    The YAML itself is read from local disk; its declared ``storage_uri`` can point anywhere
    fsspec supports. Exits 2 (a usage error) when the file is missing.
    """
    if not pipeline_path.exists():
        typer.echo(f"Pipeline file not found: {pipeline_path}", err=True)
        raise typer.Exit(code=2)
    text = pipeline_path.read_text(encoding="utf-8")
    pipeline = Pipeline.from_yaml_text(text)
    logger.info("loaded pipeline %r with %d stages", pipeline.name, len(pipeline.stages))
    return pipeline


def _default_run_id(prefix: str) -> str:
    """A fresh UTC-timestamped run id (a delta needs a new run id each time)."""
    return datetime.now(UTC).strftime(f"{prefix}-%Y%m%dT%H%M%SZ")


def _collect_document_ids(doc: list[str] | None, docs_file: Path | None) -> list[str]:
    """Gather + validate the document ids from ``--doc`` (repeatable) and ``--docs-file``.

    De-duplicates preserving order, skips blank / ``#`` comment lines in the file, and rejects
    anything that is not shaped like a content-addressed document id. Exits 2 (a usage error) when
    no ids are given or any id is malformed, with a message pointing at where to find one.
    """
    ids: list[str] = list(doc or [])
    if docs_file is not None:
        if not docs_file.exists():
            typer.echo(f"--docs-file not found: {docs_file}", err=True)
            raise typer.Exit(code=2)
        for line in docs_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                ids.append(stripped)
    seen: set[str] = set()
    unique: list[str] = []
    for doc_id in ids:
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc_id)
    if not unique:
        typer.echo(
            "No document ids given. Pass --doc <sha256:id> (repeatable) and/or "
            "--docs-file <path> (one id per line).",
            err=True,
        )
        raise typer.Exit(code=2)
    bad = [i for i in unique if not _DOC_ID_RE.match(i)]
    if bad:
        typer.echo(
            "These do not look like document ids (expected sha256:<64 hex chars>): "
            + ", ".join(bad),
            err=True,
        )
        typer.echo(
            "Find a document_id in the export's records.jsonl (the provenance.document_id "
            "field), or hash the file: printf 'sha256:%s\\n' \"$(shasum -a 256 FILE | "
            "cut -d' ' -f1)\".",
            err=True,
        )
        raise typer.Exit(code=2)
    return unique


def _emit_delta_report(pipeline: Pipeline, run_id: str, report: QualityReport) -> None:
    """Print the delta churn concisely and persist the human-readable rendering (mirrors ``run``).

    The DeltaRunner already persisted ``quality-report.json`` (with the churn); this writes the
    ``quality-report.md`` rendering alongside it — the same human/machine pair ``run`` writes — and
    prints the churn: documents ±, entities/edges churn, records retracted/purged, and the
    affected-set size vs the whole corpus-level set (the perf witness), plus the new Corpus Version.
    """
    storage = Storage.for_uri(pipeline.storage_uri)
    run_dir = f"{pipeline.storage_uri.rstrip('/')}/_latence/runs/{run_id}"
    storage.write_atomic(f"{run_dir}/quality-report.md", report.render().encode("utf-8"))
    dc = report.delta
    if dc is None:  # pragma: no cover - defensive; a delta run always attaches its churn
        typer.echo("warning: the delta run produced no churn", err=True)
        return
    recon = "reconciled" if dc.reconciliation_triggered else "incremental"
    frm = "none" if dc.from_version is None else f"v{dc.from_version}"
    typer.echo(f"Corpus Version {frm} -> v{dc.to_version} ({recon})")
    typer.echo(
        f"  documents: +{dc.documents_added} ~{dc.documents_updated} "
        f"-{dc.documents_deleted} (={dc.documents_unchanged} unchanged)"
    )
    typer.echo(
        f"  entities:  +{dc.entities_created} created, {dc.entities_merged} merged, "
        f"{dc.entities_split} split"
    )
    typer.echo(f"  edges:     +{dc.edges_added} -{dc.edges_retracted}")
    typer.echo(
        f"  records:   {dc.records_retracted} retracted, {dc.records_purged} purged "
        f"({dc.deletion_mode.value})"
    )
    typer.echo(
        f"  affected set: {dc.affected_set_size}/{dc.corpus_record_count} corpus-level records "
        "recomputed"
    )
    typer.echo(
        f"  extraction: {dc.documents_extracted} extracted, {dc.documents_reused} reused "
        "(unchanged documents' doc-level Stages skipped — issue #42 / ADR-0043)"
    )
    typer.echo(f"  drift: {dc.drift:.3f}")
    typer.echo(f"report: {run_dir}/quality-report.json", err=True)


@app.command()
def run(
    pipeline_path: Annotated[Path, typer.Argument(help="Path to the pipeline YAML.")],
    run_id: Annotated[str, typer.Option(help="Run id (state/checkpoint namespace).")] = "run-0001",
    tuned: Annotated[
        Path | None,
        typer.Option(
            "--tuned",
            help="A tuned/<device>.yaml (from `latence tune`). Its per-Provider knobs are overlaid "
            "onto the stages ONLY when its device_fingerprint matches this host; a mismatch "
            "warns + "
            "uses stack defaults (never silently applied).",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Execute a Pipeline declared in ``pipeline_path`` on the local Runner."""
    _configure_logging(verbose)
    _execute_pipeline_file(pipeline_path, run_id=run_id, tuned=tuned)


def _execute_pipeline_file(pipeline_path: Path, *, run_id: str, tuned: Path | None = None) -> None:
    """Load a stack/pipeline YAML from disk and run it on the local Runner (shared by run+process).

    The single execution path both ``latence run`` and ``latence process`` delegate to: load the
    YAML (via ``_load_pipeline`` — the SAME loader ``delta``/``retract``/``purge`` use), optionally
    overlay a tuned config, run, persist + echo the Quality Report. Logging is configured by the
    caller so each command owns its ``--verbose`` flag.
    """
    pipeline = _load_pipeline(pipeline_path)

    # W11: a tuned config's per-Provider knobs are overlaid ONLY on a device_fingerprint match
    # (else warn + stack defaults) — the overlay is config-only, no contract/offset change.
    if tuned is not None:
        if not tuned.exists():
            typer.echo(f"Tuned config not found: {tuned}", err=True)
            raise typer.Exit(code=2)
        from latence_core.tune.overlay import overlay_tuned_config
        from latence_core.tune.report import TuneReport

        tune_report = TuneReport.from_yaml_text(tuned.read_text(encoding="utf-8"))
        overlay = overlay_tuned_config(pipeline, tune_report)
        typer.echo(overlay.reason, err=True)
        pipeline = overlay.pipeline

    report = LocalRunner(lineage=_lineage_emitter_from_env()).run(pipeline, run_id=run_id)

    logger.info(
        "run complete: %d documents, contracts_complete=%s",
        report.document_count,
        report.contracts_complete,
    )
    _maybe_export_metrics(report)
    storage = Storage.for_uri(pipeline.storage_uri)
    run_dir = f"{pipeline.storage_uri.rstrip('/')}/_latence/runs/{run_id}"
    report_uri = f"{run_dir}/quality-report.json"
    # Persist the human-readable rendering alongside the machine-readable JSON (S10: the
    # Quality Report is both machine- and human-readable). stdout stays JSON (pipeable);
    # the rendered summary goes to Storage and to the log.
    storage.write_atomic(f"{run_dir}/quality-report.md", report.render().encode("utf-8"))
    logger.info("quality report:\n%s", report.render())
    typer.echo(report.to_json())
    if not storage.exists(report_uri):  # pragma: no cover - defensive
        typer.echo("warning: quality report was not persisted", err=True)


def _lineage_emitter_from_env() -> OpenLineageEmitter | None:
    """Build the OpenLineage emitter IFF ``LATENCE_OPENLINEAGE_URL`` opts in (#185).

    Off by default: with the variable unset this returns ``None`` and the Runner emits nothing,
    opens no socket, and behaves exactly as before. Set it to an OpenLineage endpoint
    (``http://localhost:5000`` for a local Marquez) to have each Stage show up in the catalog;
    ``LATENCE_OPENLINEAGE_NAMESPACE`` names the catalog namespace and
    ``LATENCE_OPENLINEAGE_API_KEY`` supplies a bearer token for a hosted collector — that token
    may only travel over ``https`` (or to loopback), and is never replayed to a redirect target. A
    misconfigured
    endpoint never breaks a run: the ConfigError is logged and lineage stays off.
    """
    url = os.environ.get("LATENCE_OPENLINEAGE_URL", "")
    if not url:
        return None
    try:
        return build_lineage_emitter(
            url,
            namespace=os.environ.get("LATENCE_OPENLINEAGE_NAMESPACE", DEFAULT_LINEAGE_NAMESPACE),
            api_key=os.environ.get("LATENCE_OPENLINEAGE_API_KEY") or None,
        )
    except Exception:  # pragma: no cover - observability must never break a run
        logger.warning(
            "OpenLineage endpoint %r is not usable; lineage disabled", url, exc_info=True
        )
        return None


def _maybe_export_metrics(report: QualityReport) -> None:
    """Export the run's Quality-Report metrics IFF ``LATENCE_METRICS`` opts in (ADR-0047).

    Off by default: ``LATENCE_METRICS`` unset (or ``none``) is a no-op, so a normal ``latence
    run`` behaves exactly as before and pulls no optional dependency. Set ``LATENCE_METRICS`` to
    ``prometheus`` or ``otel`` (with ``latence-core[observability]`` installed) to record the
    per-Stage durations/counts/errors + the ER-merge audit through
    :func:`~latence_core.observability.record_quality_report`. For the Prometheus batch case,
    ``LATENCE_METRICS_TEXTFILE`` dumps the registry for the node-exporter textfile collector.
    Observability never breaks a run: any failure here is logged, not raised.
    """
    kind = os.environ.get("LATENCE_METRICS", "none")
    if kind.lower() in {"", "none", "noop"}:
        return
    try:
        from latence_core.observability import (
            PrometheusMetricSink,
            build_metric_sink,
            record_quality_report,
        )

        sink = build_metric_sink(kind)
        record_quality_report(report, sink)
        textfile = os.environ.get("LATENCE_METRICS_TEXTFILE")
        if textfile and isinstance(sink, PrometheusMetricSink):
            sink.write_textfile(textfile)
            logger.info("metrics written to textfile: %s", textfile)
    except Exception:  # pragma: no cover - observability must never break a run
        logger.warning(
            "metrics export failed (LATENCE_METRICS=%s); continuing", kind, exc_info=True
        )


@stack_app.command("validate")
def stack_validate(
    stack_path: Annotated[
        Path, typer.Argument(help="Path to the stack YAML (e.g. stacks/default.yaml).")
    ],
    storage_root: Annotated[
        str | None,
        typer.Option(
            "--storage-root",
            help="Storage URI the harness owns for this validation (state/exports/corpus). "
            "Defaults to a fresh temp dir so a validation never collides with a real run.",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Run a named stack end-to-end on the bundled corpus and assert the G1-style checks.

    Exits 0 when every check passes (a skipped GPU stage on a CPU-only host is a PASS, not a
    failure — the device-honesty posture), non-zero otherwise, so this command gates CI/publish.
    """
    _configure_logging(verbose)
    if not stack_path.exists():
        typer.echo(f"Stack file not found: {stack_path}", err=True)
        raise typer.Exit(code=2)

    # Lazy import so a bare ``latence run`` (or ``import latence_core.cli``) does not pull the
    # harness + its runner/stage graph.
    from latence_core.stacks.validate import _describe_failures, validate_stack

    # A pinned ``--storage-root`` is honored (so an operator can inspect the artifacts after the
    # run); otherwise a dedicated temp dir keeps the harness hermetic + non-colliding by default.
    # The temp dir is torn down only after the checks + summary have been read out.
    if storage_root is not None:
        result = validate_stack(stack_path, storage_root=storage_root)
    else:
        with tempfile.TemporaryDirectory(prefix="latence-stack-validate-") as tmp:
            result = validate_stack(stack_path, storage_root=f"file://{tmp}")

    logger.info("stack validation summary:\n%s", result.render())
    typer.echo(result.render())
    if result.summary_uri is not None:
        typer.echo(f"summary written to: {result.summary_uri}", err=True)

    if not result.ok:
        typer.echo("stack validation FAILED:", err=True)
        typer.echo(_describe_failures(result.checks), err=True)
        raise typer.Exit(code=1)


@app.command("bake-off")
def bake_off(
    matrix_path: Annotated[
        Path, typer.Argument(help="Path to the matrix YAML (e.g. matrix/parse.yaml).")
    ],
    base_stack: Annotated[
        Path | None,
        typer.Option(
            "--base-stack",
            help="Stack whose OTHER stages are held constant. Defaults to the repo's "
            "stacks/default.yaml (resolved relative to the matrix file's repo root).",
        ),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            help="Directory to write MATRIX-RESULTS-<stage>.md + .json. Defaults to the matrix "
            "file's parent directory.",
        ),
    ] = None,
    storage_root: Annotated[
        str | None,
        typer.Option(
            "--storage-root",
            help="Storage URI the harness owns for the bake-off (state/exports/corpus). "
            "Defaults to a fresh temp dir so a bake-off never collides with a real run.",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Run a per-Stage provider bake-off and write ``MATRIX-RESULTS-<stage>.md`` + a JSON companion.

    Holds every OTHER Stage at the base stack and runs the SAME bundled corpus through each
    for the ONE target Stage (ADR-0036 §4). Degrades per-candidate — a candidate that crashes / is
    device-skipped is a row, never an aborted matrix — so this exits 0 when the harness itself
    ran (the enterprise reads the table; a failed/skipped candidate is a documented row).
    """
    _configure_logging(verbose)
    if not matrix_path.exists():
        typer.echo(f"Matrix file not found: {matrix_path}", err=True)
        raise typer.Exit(code=2)

    # Default base stack: the blessed stacks/default.yaml at the matrix file's repo root
    # (matrix/<stage>.yaml -> repo-root/stacks/default.yaml). An explicit --base-stack overrides.
    resolved_base = base_stack or (matrix_path.resolve().parent.parent / "stacks" / "default.yaml")
    if not resolved_base.exists():
        typer.echo(f"Base stack not found: {resolved_base}", err=True)
        raise typer.Exit(code=2)

    destination = (out_dir or matrix_path.resolve().parent).resolve()

    # Lazy import so a bare ``latence run`` does not pull the bake-off harness + its runner graph.
    from latence_core.bakeoff.harness import run_bakeoff

    if storage_root is not None:
        result = run_bakeoff(matrix_path, base_stack_path=resolved_base, storage_root=storage_root)
    else:
        with tempfile.TemporaryDirectory(prefix="latence-bakeoff-") as tmp:
            result = run_bakeoff(
                matrix_path, base_stack_path=resolved_base, storage_root=f"file://{tmp}"
            )

    destination.mkdir(parents=True, exist_ok=True)
    md_path = destination / f"MATRIX-RESULTS-{result.stage}.md"
    json_path = destination / f"MATRIX-RESULTS-{result.stage}.json"
    md_path.write_text(result.render(), encoding="utf-8")
    json_path.write_text(result.to_json() + "\n", encoding="utf-8")

    typer.echo(result.render())
    typer.echo(f"results written to: {md_path}", err=True)
    typer.echo(f"machine-readable companion: {json_path}", err=True)


@app.command()
def tune(
    stack_path: Annotated[
        Path, typer.Argument(help="Path to the stack YAML to tune (e.g. stacks/gpu-sota.yaml).")
    ],
    sample: Annotated[
        int, typer.Option("--sample", help="Records to time each candidate on (the small sample).")
    ] = 8,
    repeats: Annotated[
        int, typer.Option("--repeats", help="Timed runs per candidate after warmup (median → r/s).")
    ] = 3,
    seed: Annotated[int, typer.Option("--seed", help="Sweep seed (determinism).")] = 0,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Where to write the tuned config. Defaults to tuned/<device>.yaml at the stack's "
            "repo root (the device slug is derived from the fingerprint).",
        ),
    ] = None,
    storage_root: Annotated[
        str | None,
        typer.Option(
            "--storage-root",
            help="Storage URI the sweep owns (state/checkpoints/corpus). Defaults to a fresh temp "
            "dir so a tune never collides with a real run.",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Profile the device, sweep each heavy Provider's perf knobs correctness-gated, write a tune.

    For each GPU/heavy Provider in the stack it sweeps dtype/quantize/compile/batch_size/
    ocr_batch_size/max_concurrency over the small real sample, REJECTS any setting that crashes
    (including an inference-time ``BackendCompilerFailed`` — the W9 lesson) or changes output vs the
    fp32/plain baseline, picks the fastest ELIGIBLE per Provider, and writes a device-fingerprinted
    ``tuned/<device>.yaml`` + an honest report (winners AND rejected knobs). Never fabricates a
    number — every throughput is the median of the seeded timed runs on THIS device.
    """
    _configure_logging(verbose)
    if not stack_path.exists():
        typer.echo(f"Stack file not found: {stack_path}", err=True)
        raise typer.Exit(code=2)

    # Lazy import so a bare ``latence run`` does not pull the sweep's runner/stage graph.
    from latence_core.tune.fingerprint import probe_device_fingerprint
    from latence_core.tune.realprobe import RegistrySampleProbe
    from latence_core.tune.tuner import tune_stack

    fingerprint = probe_device_fingerprint()

    def _run_tune(root: str) -> object:
        probe = RegistrySampleProbe(stack_path, storage_root=root)
        return tune_stack(
            str(stack_path),
            probe=probe,
            sample_size=sample,
            repeats=repeats,
            seed=seed,
            fingerprint=fingerprint,
        )

    if storage_root is not None:
        report = _run_tune(storage_root)
    else:
        with tempfile.TemporaryDirectory(prefix="latence-tune-") as tmp:
            report = _run_tune(f"file://{tmp}")

    # Default output: tuned/<device>.yaml at the stack's repo root (matrix/<stage>.yaml → repo-
    # root).
    repo_root = stack_path.resolve().parent.parent
    destination = out or (repo_root / "tuned" / f"{fingerprint.slug()}.yaml")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report.to_yaml(), encoding="utf-8")  # type: ignore[attr-defined]

    typer.echo(report.render())  # type: ignore[attr-defined]
    typer.echo(f"tuned config written to: {destination}", err=True)


@app.command()
def setup(
    source: Annotated[
        str | None,
        typer.Option("--source", help="Folder (Storage URI or path) holding the files to process."),
    ] = None,
    storage: Annotated[
        str | None,
        typer.Option(
            "--storage", help="Output/state Storage URI. Default: file://<cwd>/latence-out."
        ),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            help="Setup tier: 'lite' (default — CPU-first, laptop-friendly) or 'enterprise-sota' "
            "(the pod-validated composed stack: served-vLLM OCR + LLM induce + GLinker ER + "
            "Granite + ULTRA). See stacks/gpu-sota-glinker.yaml.",
        ),
    ] = None,
    parser: Annotated[
        str | None,
        typer.Option(
            "--parser",
            help="Parser Provider. lite profile: parser.render (universal), parser.pdfplumber "
            "(PDF text), or parser.lighton (OCR). enterprise-sota profile: parser.lighton_vllm "
            "(served-vLLM OCR for scans/PDFs/images) or parser.plaintext (born-digital "
            ".txt/.md/.jsonl, CPU). Default: recommended from the detected file types.",
        ),
    ] = None,
    ocr_base_url: Annotated[
        str | None,
        typer.Option(
            "--ocr-base-url",
            help="enterprise-sota only: the served-vLLM OCR endpoint (OpenAI-compatible).",
        ),
    ] = None,
    ocr_model: Annotated[
        str | None,
        typer.Option("--ocr-model", help="enterprise-sota only: the served OCR model id."),
    ] = None,
    device: Annotated[
        str | None, typer.Option("--device", help="auto | cpu | cuda for the learned Stages.")
    ] = None,
    llm_base_url: Annotated[
        str | None,
        typer.Option("--llm-base-url", help="Schema-induction LLM base URL (OpenAI-compatible)."),
    ] = None,
    llm_model: Annotated[
        str | None, typer.Option("--llm-model", help="Schema-induction LLM model id.")
    ] = None,
    pii: Annotated[
        bool | None,
        typer.Option(
            "--pii/--no-pii",
            help="Include (default) or omit the PII redaction + guardrail screening stages.",
        ),
    ] = None,
    extract_on: Annotated[
        str | None,
        typer.Option(
            "--extract-on",
            help="Advanced (W18): 'unmasked' (default — the KG keeps real entities) or 'masked' "
            "(extraction reads REDACTED chunks so PII never reaches the KG). 'masked' needs --pii.",
        ),
    ] = None,
    domain: Annotated[
        str | None,
        typer.Option(
            "--domain",
            help="Optional free-text description of your corpus's subject matter. It steers schema "
            "induction (it rides `induce.config.domain`) and is recorded as a header note.",
        ),
    ] = None,
    predefined_labels: Annotated[
        Path | None,
        typer.Option(
            "--predefined-labels",
            help="PREDEFINED-ONLY mode: a JSON/YAML file with your FIXED label schema "
            "({entity_types, relation_types, pii_types}). Emits NO `induce` stage at all — zero "
            "induction LLM calls — and extracts every chunk against exactly these labels. For a "
            "corpus with a known taxonomy. The ADR-0044 financial-PII floor still applies.",
        ),
    ] = None,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to save the generated stack config.")
    ] = Path("latence.stack.yaml"),
    non_interactive: Annotated[
        bool,
        typer.Option(
            "--non-interactive",
            help="Take every answer from flags (no prompts). Same inputs → byte-identical config.",
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Guided, deterministic setup → a clean, validated, quality-baked stack config (W15, ADR-0041).

    Interactive by default: each prompt carries a sensible default + why it matters. With
    ``--non-interactive`` every answer comes from the flags (unset flags fall back to the SAME
    defaults), so identical inputs produce a **byte-identical** config for CI/repro. The generated
    stack is DRY-VALIDATED (structural + DAG + capability wiring — reusing the ``stack validate``
    machinery, no pipeline run) before it is saved.
    """
    _configure_logging(verbose)
    from latence_core.setup import (
        SetupAnswers,
        SetupConfigError,
        build_stack_config,
        detect_header_family,
        detect_inline_family,
        llm_key_env_var,
        parse_predefined_labels,
        recommend_parser,
        recommend_sota_parser,
        scan_source,
        validate_config_text,
    )
    from latence_core.setup.wizard import (
        _KNOWN_PARSERS,
        _KNOWN_PROFILES,
        _KNOWN_SOTA_PARSERS,
        DEFAULT_LLM_BASE_URL,
        DEFAULT_LLM_MODEL,
        DEFAULT_OCR_BASE_URL,
        DEFAULT_OCR_MODEL,
        INTAKE_MAX_BYTES_FLOOR,
        PARSER_OCR_VLLM,
        PARSER_PLAINTEXT,
        PROFILE_LITE,
        PROFILE_SOTA,
        SOTA_DEVICE,
        normalize_domain_hint,
    )

    interactive = not non_interactive

    def ask_text(flag: str | None, prompt: str, why: str, default: str) -> str:
        if flag is not None:
            return flag
        if not interactive:
            return default
        typer.echo(f"  ({why})", err=True)
        return str(typer.prompt(prompt, default=default))

    # a. Source folder — required, must exist. Detect the extensions present.
    default_cwd = Path.cwd()
    src = ask_text(
        source,
        "Source folder (where your files live)",
        "the folder Latence reads; validated to exist, its file types auto-detected",
        f"file://{default_cwd}/corpus",
    )
    src_path = Path(src.removeprefix("file://"))
    if not src_path.exists() or not src_path.is_dir():
        typer.echo(f"Source folder does not exist (or is not a directory): {src}", err=True)
        raise typer.Exit(code=2)
    # ONE walk answers both questions: which parser the corpus needs, and how big its documents
    # actually are (which sizes the intake oversize cap, so a real corpus is not quarantined).
    scan = scan_source(src)
    extensions = scan.extensions
    typer.echo(f"Detected file types under the source folder: {', '.join(extensions)}", err=True)
    if not scan.scanned:
        typer.echo(
            "WARNING: could not read any files under the source folder (empty, unreadable, or not "
            f"a local path), so the file types above are the broad default set and the intake "
            f"size cap stays at the {INTAKE_MAX_BYTES_FLOOR} byte default. If your documents are "
            "larger than that they will be quarantined at intake — check `--source`, or raise "
            "`intake_screen.config.max_bytes` in the generated config by hand.",
            err=True,
        )

    # b. Profile / tier — 'lite' (default) or 'enterprise-sota' (the pod-validated composed stack).
    chosen_profile = ask_text(
        profile,
        f"Profile ({PROFILE_LITE} | {PROFILE_SOTA})",
        "lite = CPU-first, laptop-friendly; enterprise-sota = the validated GPU stack (served-vLLM "
        "OCR + LLM induce + GLinker ER + Granite + ULTRA)",
        PROFILE_LITE,
    )
    if chosen_profile not in _KNOWN_PROFILES:
        typer.echo(
            f"Unknown profile {chosen_profile!r}; expected one of {sorted(_KNOWN_PROFILES)}.",
            err=True,
        )
        raise typer.Exit(code=2)
    is_sota = chosen_profile == PROFILE_SOTA

    # c. Output / storage URI.
    store = ask_text(
        storage,
        "Output / storage URI",
        "where state + the exported RAG corpus and KG are written",
        f"file://{default_cwd}/latence-out",
    )

    # d. Parser + OCR endpoint. enterprise-sota recommends a parse path from the detected file types
    # — the served-vLLM OCR parser for a scanned/PDF/image corpus, or the CPU plaintext parser for a
    # born-digital .txt/.md/.jsonl corpus (the OCR VLM has no text passthrough, so forcing it on a
    # text corpus makes the stack un-runnable) — and only prompts for the OCR endpoint when the OCR
    # path is chosen. lite recommends a parser from the detected file types (confirm/override).
    ocr_base = DEFAULT_OCR_BASE_URL
    ocr_mdl = DEFAULT_OCR_MODEL
    if is_sota:
        recommended = recommend_sota_parser(extensions)
        chosen_parser = ask_text(
            parser,
            f"Parser ({PARSER_PLAINTEXT} | {PARSER_OCR_VLLM})",
            f"recommended '{recommended}' from your file types; "
            f"plaintext=born-digital .txt/.md/.jsonl (CPU, no OCR), "
            f"lighton_vllm=served-vLLM OCR for scans/PDFs/images",
            recommended,
        )
        if chosen_parser not in _KNOWN_SOTA_PARSERS:
            typer.echo(
                f"Unknown enterprise-sota parser {chosen_parser!r}; expected one of "
                f"{sorted(_KNOWN_SOTA_PARSERS)}.",
                err=True,
            )
            raise typer.Exit(code=2)
        if chosen_parser == PARSER_OCR_VLLM:
            ocr_base = ask_text(
                ocr_base_url,
                "OCR endpoint base URL (served vLLM, OpenAI-compatible)",
                "the running vLLM OCR server the enterprise stack POSTs page-images to",
                DEFAULT_OCR_BASE_URL,
            )
            ocr_mdl = ask_text(
                ocr_model,
                "OCR model id",
                "the model your vLLM server registers (the LightOn-OCR checkpoint id)",
                DEFAULT_OCR_MODEL,
            )
    else:
        recommended = recommend_parser(extensions)
        chosen_parser = ask_text(
            parser,
            "Parser (parser.render | parser.pdfplumber | parser.lighton)",
            f"recommended '{recommended}' from your file types; render=universal, "
            f"pdfplumber=born-digital PDF, lighton=OCR for scans",
            recommended,
        )
        if chosen_parser not in _KNOWN_PARSERS:
            typer.echo(
                f"Unknown parser {chosen_parser!r}; expected one of {sorted(_KNOWN_PARSERS)}.",
                err=True,
            )
            raise typer.Exit(code=2)

    # e. Device. enterprise-sota is a GPU-pod stack — every learned Stage runs on CUDA (skip-with-
    # flag on a CPU host); lite lets the operator choose (default 'auto').
    if is_sota:
        dev = SOTA_DEVICE
    else:
        dev = ask_text(
            device,
            "Device (auto | cpu | cuda)",
            "learned Stages run here; 'auto' picks CUDA when present, else CPU (skip-with-flag)",
            "auto",
        )

    # e. Schema induction LLM (mandatory for quality) — endpoint + model; key from env at run.
    base_url = ask_text(
        llm_base_url,
        "Schema-induction LLM base URL",
        "the OpenAI-compatible endpoint that discovers your corpus's own schema (quality lever)",
        DEFAULT_LLM_BASE_URL,
    )
    model = ask_text(
        llm_model,
        "Schema-induction LLM model id",
        "a small, cheap, capable model is plenty (it induces label TYPES, not content)",
        DEFAULT_LLM_MODEL,
    )

    # f. The ONE optional toggle — PII redaction + guardrail screening.
    if pii is not None:
        pii_on = pii
    elif interactive:
        typer.echo(
            "  (adds content-screening + learned financial-PII detection/redaction; default ON)",
            err=True,
        )
        pii_on = typer.confirm("Enable PII redaction + guardrail screening?", default=True)
    else:
        pii_on = True

    # f2. Advanced (W18): mask-before-extract. Only meaningful with PII ON; default OFF (unmasked),
    # which keeps the config byte-identical to pre-W18. 'masked' feeds extraction the redacted chunk
    # stream so PII never reaches the KG (the stricter posture).
    if extract_on is not None:
        extract_mode = extract_on
    elif interactive and pii_on:
        typer.echo(
            "  (advanced: 'masked' feeds extraction the REDACTED chunks so PII never reaches the "
            "KG — stricter privacy; default 'unmasked' keeps the KG's real entities)",
            err=True,
        )
        extract_mode = (
            "masked"
            if typer.confirm(
                "Mask PII BEFORE extraction (keep it out of the knowledge graph)?", default=False
            )
            else "unmasked"
        )
    else:
        extract_mode = "unmasked"
    if extract_mode not in ("unmasked", "masked"):
        typer.echo(
            f"Unknown --extract-on {extract_mode!r}; expected 'unmasked' or 'masked'.", err=True
        )
        raise typer.Exit(code=2)
    if extract_mode == "masked" and not pii_on:
        typer.echo(
            "--extract-on masked requires --pii (there is no redact Stage to mask from otherwise).",
            err=True,
        )
        raise typer.Exit(code=2)

    # g. Optional domain hint.
    domain_hint = (
        domain
        if domain is not None
        else (
            str(typer.prompt("Domain / description hint (optional)", default=""))
            if interactive
            else ""
        )
    )

    # dogfood-3-inspect: enterprise-SOTA sniffs the corpus for a known structured metadata-header
    # cross-reference family (an RFC's `Request for Comments:` + `Obsoletes:`/`Updates:` lines) and,
    # when found, wires the deterministic `header_refs` extractor to recover those explicit edges
    # the entity-anchored learned extractor cannot see. No family detected (or the `lite` profile)
    # ⇒ None ⇒ no such Stage, byte-identical to before.
    header_refs_family = detect_header_family(src) if is_sota else None
    if header_refs_family is not None:
        typer.echo(
            "Detected structured cross-reference headers in the corpus: wiring the deterministic "
            "header-reference extractor (recovers e.g. RFC Obsoletes:/Updates: edges).",
            err=True,
        )

    # The companion of the above (the dogfood measured gap): the header-only extractor recovers a
    # document's `Obsoletes:`/`Updates:` HEADER edges but structurally cannot see its INLINE body
    # citations ("RFC 7230" scattered through the body) — the overwhelming majority of the corpus's
    # gold cross-references. When the same family is detected, wire the deterministic
    # inline-citation extractor beside it. No family (or `lite`) ⇒ None ⇒ no such Stage,
    # byte-identical to before.
    inline_refs_family = detect_inline_family(src) if is_sota else None
    if inline_refs_family is not None:
        typer.echo(
            "Wiring the deterministic inline-citation reference extractor (recovers the body "
            "'RFC <n>' references the header-only extractor cannot see) into the knowledge graph.",
            err=True,
        )

    # T1 predefined-only: an operator with a KNOWN schema (a compliance taxonomy, a fixed entity
    # set) supplies it as a file, and the generator emits NO `induce` stage at all — no Provider,
    # no client, no LLM call. Loaded + validated here (I/O is the CLI's job; the generator stays
    # pure) and failing LOUDLY, because a typo'd slot that silently fell back to the profile's
    # default labels is the worst outcome for exactly the corpora that opt in.
    fixed_labels: dict[str, list[str]] | None = None
    if predefined_labels is not None:
        if not predefined_labels.is_file():
            typer.echo(f"Predefined-labels file not found: {predefined_labels}", err=True)
            raise typer.Exit(code=2)
        import yaml  # YAML is a superset of JSON, so ONE loader reads either file format.

        try:
            fixed_labels = parse_predefined_labels(
                yaml.safe_load(predefined_labels.read_text(encoding="utf-8"))
            )
        except SetupConfigError as exc:
            typer.echo(f"Invalid --predefined-labels file: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        except yaml.YAMLError as exc:
            typer.echo(f"--predefined-labels file is not valid JSON/YAML: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        typer.echo(
            "Predefined-only mode: schema induction is OFF (no `induce` stage, zero LLM calls); "
            "extraction runs every chunk against your fixed labels.",
            err=True,
        )

    answers = SetupAnswers(
        source=src,
        storage=store,
        parser=chosen_parser,
        device=dev,
        llm_base_url=base_url,
        llm_model=model,
        profile=chosen_profile,
        ocr_base_url=ocr_base,
        ocr_model=ocr_mdl,
        pii=pii_on,
        extract_on=extract_mode,
        domain=domain_hint,
        largest_source_bytes=scan.largest_bytes,
        extensions=extensions,
        header_refs=header_refs_family,
        inline_refs=inline_refs_family,
        predefined_labels=fixed_labels,
    )

    # Confirm the LLM key env var is set (never store the key). Warn — never crash — if unset.
    # Predefined-only makes NO induction call, so warning about a key it will never read would be
    # noise that teaches operators to ignore the warning.
    key_var = llm_key_env_var(base_url)
    if fixed_labels is None and not os.environ.get(key_var):
        typer.echo(
            f"WARNING: ${key_var} is not set. Schema induction needs it at run time "
            f"(`latence process`). Export it before running, e.g. `export {key_var}=...`.",
            err=True,
        )

    # Generate → dry-validate → save. Fail loudly (no broken config saved) on a validation error.
    config_text = build_stack_config(answers)
    try:
        validate_config_text(config_text)
    except SetupConfigError as exc:
        typer.echo(f"Generated config failed validation: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    output.write_text(config_text, encoding="utf-8")

    # Clean summary: stages, toggles, where saved.
    from latence_core.pipeline import Pipeline

    pipeline = Pipeline.from_yaml_text(config_text)
    typer.echo("")
    typer.echo(f"Saved quality-baked stack to: {output}")
    typer.echo(f"  profile: {chosen_profile}")
    stage_flow = " -> ".join(s.name for s in pipeline.stages)
    typer.echo(f"  stages ({len(pipeline.stages)}): {stage_flow}")
    typer.echo(f"  parser: {chosen_parser}   device: {dev}")
    if is_sota:
        typer.echo(f"  OCR endpoint: {ocr_mdl} @ {ocr_base}")
    # The intake oversize cap: say what it is AND where the number came from, because a cap below
    # the corpus's own files is the difference between a full run and a silently empty one.
    cap = int(next(s for s in pipeline.stages if s.name == "intake_screen").config["max_bytes"])
    if scan.scanned:
        typer.echo(
            f"  intake size cap: {cap} bytes — sized from your corpus "
            f"(largest file {scan.largest_bytes} bytes)"
        )
    else:
        typer.echo(f"  intake size cap: {cap} bytes — the default (your corpus was not readable)")
    if scan.largest_bytes > cap:
        typer.echo(
            f"  WARNING: the largest file in your source folder is {scan.largest_bytes} bytes, "
            f"OVER the {cap} byte intake cap — files that big are quarantined at intake and will "
            "NOT reach the corpus. Split them, or raise `intake_screen.config.max_bytes` by hand "
            "(and expect the memory cost of screening a file that size)."
        )
    typer.echo(f"  schema induction: {model} @ {base_url}  (key from ${key_var})")
    hint = normalize_domain_hint(domain_hint)
    if hint:
        typer.echo(f"  domain hint: steering schema induction ({hint})")
    typer.echo(
        "  PII redaction + guardrail screening: "
        + ("ON (financial-PII; chunk-level masking + report)" if pii_on else "OFF")
    )
    if pii_on:
        typer.echo(
            "  NOTE: the exported RAG chunks are masked AND context-enriched (W16, ADR-0042): "
            "redact sets each chunk's masked_content, enrich reads that redacted stream, and "
            "export_corpus materializes it — a genuinely PII-safe, context-enriched corpus."
        )
    typer.echo("")
    typer.echo("Run `latence process` to start.")


@app.command()
def process(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="The saved stack config to run."),
    ] = Path("latence.stack.yaml"),
    run_id: Annotated[
        str, typer.Option("--run-id", help="Run id (state/checkpoint namespace).")
    ] = "run-0001",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Run a stack config saved by ``latence setup`` — one line to kick off the pipeline (W15).

    Loads ``--config`` (default ``./latence.stack.yaml``) and runs it on the local Runner, the SAME
    execution path as ``latence run``.
    """
    _configure_logging(verbose)
    if not config.exists():
        typer.echo(
            f"Config not found: {config}. Run `latence setup` first (or pass --config).", err=True
        )
        raise typer.Exit(code=2)
    _execute_pipeline_file(config, run_id=run_id)


# -- incremental corpus updates (S11, ADR-0029) --------------------------------
#
# These three commands expose the EXISTING DeltaRunner. They reuse ``run``'s pipeline loader
# (``_load_pipeline``) and its report-printing discipline (``_emit_delta_report`` persists the
# same human/machine report pair). Since W17 (ADR-0043) a delta's extraction is incremental —
# only new/changed documents are re-extracted (``--reinduce`` forces the full O(corpus) pass).

_RUN_ID_HELP = (
    "Run id for this operation (state/checkpoint namespace). A delta needs a FRESH run id every "
    "time; defaults to a UTC-timestamped id (e.g. delta-20260713T101500Z). Override to pin one."
)
_STACK_ARG_HELP = "Path to the stack/pipeline YAML (same file `latence run` takes)."


@app.command()
def delta(
    pipeline_path: Annotated[Path, typer.Argument(help=_STACK_ARG_HELP)],
    run_id: Annotated[str | None, typer.Option("--run-id", help=_RUN_ID_HELP)] = None,
    reinduce: Annotated[
        bool,
        typer.Option(
            "--reinduce",
            help=(
                "Full rebuild: re-induce EVERY chunk and re-canonicalize the type vocabulary "
                "from scratch, re-extracting every document (the O(corpus) escape hatch, and the "
                "one path that may re-elect canonical type labels). Default induces only the "
                "new/changed chunks and EXTENDS the corpus's canonical type vocabulary."
            ),
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Add/update the corpus from the current source and commit the next Corpus Version.

    The user added, changed, or removed files in the source folder; this detects them by
    content-hash against the last committed Corpus Version and commits Version N+1 with the
    affected-set recompute. Since W17 (ADR-0043) the extraction is INCREMENTAL: each UNCHANGED
    document's doc-level records are reused, so only NEW/CHANGED documents run
    parse/chunk/extract/redact (and only their chunks are induced) — the compute is O(changed
    documents), not O(corpus). Consistency with the base run comes from the corpus's canonical
    TYPE vocabulary, which the delta EXTENDS rather than recomputes (T4/ADR-0054): a new type
    aliases into an existing canonical entry or appends a new one, and existing canonical types
    never churn. ``--reinduce`` is the full rebuild — re-induce and re-canonicalize from scratch.
    """
    _configure_logging(verbose)
    pipeline = _load_pipeline(pipeline_path)
    rid = run_id or _default_run_id("delta")
    from latence_core.delta_runner import DeltaRunner

    try:
        report = DeltaRunner().apply_delta(pipeline, rid, reinduce=reinduce)
    except (LatenceError, ValueError) as exc:
        typer.echo(f"delta failed [{error_category(exc)}]: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _emit_delta_report(pipeline, rid, report)


@app.command()
def retract(
    pipeline_path: Annotated[Path, typer.Argument(help=_STACK_ARG_HELP)],
    doc: Annotated[
        list[str] | None,
        typer.Option("--doc", help="A document id to act on (sha256:...). Repeatable."),
    ] = None,
    docs_file: Annotated[
        Path | None,
        typer.Option("--docs-file", help="A file of document ids, one per line (# comments ok)."),
    ] = None,
    run_id: Annotated[str | None, typer.Option("--run-id", help=_RUN_ID_HELP)] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Soft-delete documents (Retraction — the auditable, rollback-able default).

    The documents' derived records are excluded from the live corpus but RETAINED in the prior
    Corpus Version for audit and rollback, and any cluster whose merge was justified solely by a
    retracted document is split. Commits the next Corpus Version. Give ids via ``--doc``
    (repeatable) and/or ``--docs-file``.
    """
    _configure_logging(verbose)
    document_ids = _collect_document_ids(doc, docs_file)
    pipeline = _load_pipeline(pipeline_path)
    rid = run_id or _default_run_id("retract")
    from latence_core.delta_runner import DeltaRunner

    try:
        report = DeltaRunner().retract(pipeline, rid, document_ids)
    except (LatenceError, ValueError) as exc:
        typer.echo(f"retract failed [{error_category(exc)}]: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _emit_delta_report(pipeline, rid, report)


@app.command()
def purge(
    pipeline_path: Annotated[Path, typer.Argument(help=_STACK_ARG_HELP)],
    doc: Annotated[
        list[str] | None,
        typer.Option("--doc", help="A document id to act on (sha256:...). Repeatable."),
    ] = None,
    docs_file: Annotated[
        Path | None,
        typer.Option("--docs-file", help="A file of document ids, one per line (# comments ok)."),
    ] = None,
    run_id: Annotated[str | None, typer.Option("--run-id", help=_RUN_ID_HELP)] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt (erasure is permanent)."),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Hard-delete documents (Purge — GDPR / right-to-be-forgotten erasure).

    DESTRUCTIVE and NON-RECOVERABLE: unlike a Retraction, NOTHING is retained — the on-disk source
    file(s), every derived record across ALL Corpus Versions, and the run artifacts are physically
    erased. Guarded by a confirmation prompt; pass ``--yes`` to skip it. Give ids via ``--doc``
    (repeatable) and/or ``--docs-file``.
    """
    _configure_logging(verbose)
    document_ids = _collect_document_ids(doc, docs_file)
    if not yes:
        # Destructive + non-recoverable: confirm first. `abort=True` exits non-zero on "no".
        typer.confirm(
            f"Purge PHYSICALLY and PERMANENTLY erases {len(document_ids)} document(s) — the source "
            "file(s), every derived record across every Corpus Version, and the run artifacts. "
            "This cannot be undone. Continue?",
            abort=True,
        )
    pipeline = _load_pipeline(pipeline_path)
    rid = run_id or _default_run_id("purge")
    from latence_core.delta_runner import DeltaRunner

    try:
        report = DeltaRunner().purge(pipeline, rid, document_ids)
    except (LatenceError, ValueError) as exc:
        typer.echo(f"purge failed [{error_category(exc)}]: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _emit_delta_report(pipeline, rid, report)


def _repo_root_for(spec_path: Path) -> Path:
    """The repo root a ``deploy/env/<name>/environment.yaml`` belongs to (three parents up)."""
    from latence_core.envspec import repo_root_for  # noqa: PLC0415 - keeps the CLI import lazy

    return repo_root_for(spec_path)


@env_app.command("verify")
def env_verify(
    spec_path: Annotated[
        Path, typer.Argument(help="Path to a deploy/env/<name>/environment.yaml.")
    ],
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Workspace root (default: three parents up from spec)."),
    ] = None,
    probe: Annotated[
        bool,
        typer.Option(
            "--probe",
            help="Additionally LOAD every declared Provider from the registry (imports the heavy "
            "stack). Only meaningful inside an installed environment.",
        ),
    ] = False,
) -> None:
    """Check a reproducible environment against the workspace it claims to stand up.

    Offline and import-free: compares the generated ``requirements.in``/``constraints.txt``, the
    solved lock, the Provider packages' declared dependencies and entry-points, and the smoke
    stack. Exits non-zero on any finding, so dependency drift fails the build instead of the pod.
    """
    from latence_core.envspec import load_environment, probe_providers, verify_environment

    try:
        spec = load_environment(spec_path)
        findings = verify_environment(spec, repo_root or _repo_root_for(spec_path)).findings
        if probe:
            findings += probe_providers(spec).findings
    except (LatenceError, OSError) as exc:
        typer.echo(f"env verify failed [{error_category(exc)}]: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    from latence_core.envspec import EnvironmentReport

    report = EnvironmentReport(spec.name, findings)
    typer.echo(report.describe())
    if not report.ok:
        raise typer.Exit(code=1)


@env_app.command("placement")
def env_placement(
    spec_path: Annotated[
        Path, typer.Argument(help="Path to a deploy/env/<name>/environment.yaml.")
    ],
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Workspace root (default: three parents up from spec)."),
    ] = None,
) -> None:
    """PROVE every model-loading Provider of a GPU environment really runs on the GPU (issue #198).

    Drives one real inference through each such Provider the environment's stack instantiates and
    reads torch's CUDA allocator around the call. A model that silently stayed on the CPU allocates
    zero CUDA bytes no matter how correct its output is — which is exactly the #191 failure this
    gate makes impossible to reintroduce. Meaningful only inside an installed GPU environment; a
    missing CUDA device is a FAILURE, never a skip.

    The probe set is derived from the workspace packages the environment installs (which is why
    ``--repo-root`` matters), unioned with its declared ``learned_providers``.
    """
    from latence_core.envspec import load_environment
    from latence_core.placement import probe_placement

    try:
        spec = load_environment(spec_path)
        result = probe_placement(spec, repo_root=repo_root or _repo_root_for(spec_path))
    except (LatenceError, OSError, ImportError) as exc:
        typer.echo(f"env placement failed [{error_category(exc)}]: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(result.describe())
    if not result.ok:
        raise typer.Exit(code=1)


@env_app.command("render")
def env_render(
    spec_path: Annotated[
        Path, typer.Argument(help="Path to a deploy/env/<name>/environment.yaml.")
    ],
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Workspace root (default: three parents up from spec)."),
    ] = None,
    check: Annotated[
        bool,
        typer.Option("--check", help="Do not write; exit non-zero if a generated file is stale."),
    ] = False,
) -> None:
    """Regenerate ``requirements.in`` + ``constraints.txt`` from the spec + the real pyprojects."""
    from latence_core.envspec import load_environment, render_constraints, render_requirements

    try:
        spec = load_environment(spec_path)
        root = repo_root or _repo_root_for(spec_path)
        rendered = {
            spec.requirements_path: render_requirements(spec, root),
            spec.constraints_path: render_constraints(spec, root),
        }
    except (LatenceError, OSError) as exc:
        typer.echo(f"env render failed [{error_category(exc)}]: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    stale = [
        path
        for path, text in rendered.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != text
    ]
    if check:
        for path in stale:
            typer.echo(f"stale: {path}", err=True)
        typer.echo(f"{len(stale)} stale generated file(s) for environment '{spec.name}'")
        if stale:
            raise typer.Exit(code=1)
        return
    for path, text in rendered.items():
        path.write_text(text, encoding="utf-8")
    typer.echo(f"rendered {len(rendered)} file(s) for environment '{spec.name}'")


@env_app.command("matrix")
def env_matrix(
    matrix_path: Annotated[
        Path,
        typer.Argument(help="Path to deploy/env/compatibility-matrix.yaml."),
    ] = Path("deploy/env/compatibility-matrix.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Workspace root (default: two parents up from the file)."),
    ] = None,
    render: Annotated[
        bool,
        typer.Option("--render", help="Write the generated Markdown document instead of checking."),
    ] = False,
    require_verified: Annotated[
        bool,
        typer.Option(
            "--require-verified",
            help="Also fail while ANY cell or the full-stack run is still UNVERIFIED. The switch "
            "a maintainer flips ON the Blackwell host to prove the map is fully measured.",
        ),
    ] = False,
) -> None:
    """Check the Blackwell/CUDA-13 compatibility matrix (issue #199).

    Offline and import-free. Every cell must be a measurement — date, hardware, command, observed
    output — or explicitly ``UNVERIFIED`` with the exact command that turns it into one. Coverage
    is derived from the committed ``deploy/env/*/environment.yaml`` specs, so a new enforced
    boundary or learned Provider fails this check until it is mapped.
    """
    from latence_core.compat import load_matrix, render_matrix, verify_matrix

    try:
        matrix = load_matrix(matrix_path)
        root = repo_root or matrix_path.resolve().parents[2]
        if render:
            doc = root / matrix.doc_path
            doc.parent.mkdir(parents=True, exist_ok=True)
            doc.write_text(render_matrix(matrix), encoding="utf-8")
            typer.echo(f"rendered {matrix.doc_path} from {matrix_path}")
            return
        report = verify_matrix(matrix, root, require_verified=require_verified)
    except (LatenceError, OSError) as exc:
        typer.echo(f"env matrix failed [{error_category(exc)}]: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(report.describe())
    if not report.ok:
        raise typer.Exit(code=1)


@retrieval_app.command("splade-index")
def splade_index(
    export: Annotated[
        Path,
        typer.Option(
            "--export",
            help="The Export directory to index (holds records.parquet + records.jsonl).",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out", help="Where to write the splade/ artifact (index.parquet + manifest.json)."
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="The multilingual SPLADE checkpoint id (e.g. naver/splade-v3). No default is "
            "pinned — the model is operator-selected and its license cleared (ADR-0053/0012).",
        ),
    ],
    accept_noncommercial_license: Annotated[
        bool,
        typer.Option(
            "--accept-noncommercial-license",
            help="Assert you accept the model's NON-COMMERCIAL weights license (e.g. "
            "naver/splade-v3 is CC-BY-NC-SA-4.0). MANDATORY: without it the command refuses "
            "rather than silently defaulting to permissive.",
        ),
    ] = False,
    weights_license: Annotated[
        str,
        typer.Option(
            "--weights-license",
            help="The non-commercial weights license the acceptance covers (recorded honestly in "
            "the manifest, license_verified=false — this tool does not read the model card).",
        ),
    ] = "CC-BY-NC-SA-4.0",
    max_length: Annotated[
        int, typer.Option("--max-length", help="Max tokens per chunk fed to the SPLADE model.")
    ] = 512,
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Chunks encoded per batch (O(batch) memory).")
    ] = 64,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Build a REAL SPLADE-v3 sparse index over an Export corpus (tutorial §4.2).

    Loads the operator-selected multilingual SPLADE checkpoint via the ``sparse.splade`` Provider
    (:class:`~latence_splade.provider.SpladeCompositionEmbedder`), encodes every corpus chunk's
    (redacted) content to a term-weight :class:`~latence_core.capability.SparseVector`, and writes a
    ``splade/`` artifact: ``index.parquet`` (per-chunk ``record_id`` + parallel ``indices`` /
    ``values`` for every chunk with a non-empty vector) plus a ``manifest.json`` recording the
    checkpoint and the honest ADR-0012 license provenance (``weights_license`` /
    ``license_verified=false`` / ``operator_accepted=true``).

    ``--accept-noncommercial-license`` is MANDATORY: naver/splade-v3's weights are CC-BY-NC-SA-4.0
    (non-commercial), so the command REFUSES without the operator's explicit acceptance rather than
    silently treating a non-commercial model as a permissive default.
    """
    _configure_logging(verbose)

    # License gate FIRST — before resolving or loading anything heavy, and independent of whether
    # ``latence-splade`` is installed. A non-commercial weights license is never a silent default:
    # the operator must accept it explicitly (ADR-0012 honesty).
    if not accept_noncommercial_license:
        typer.echo(
            "refused: --accept-noncommercial-license is required. The SPLADE checkpoint's weights "
            f"({model}) are a NON-COMMERCIAL license (e.g. naver/splade-v3 is CC-BY-NC-SA-4.0); "
            "this command will not default a non-commercial model to permissive. Re-run with "
            "--accept-noncommercial-license once you accept the model's non-commercial terms.",
            err=True,
        )
        raise typer.Exit(code=2)

    if not export.exists():
        typer.echo(f"Export directory not found: {export}", err=True)
        raise typer.Exit(code=2)

    from latence_core.capability import ProviderRegistry
    from latence_core.splade_index import SpladeIndexError, build_splade_index

    # Resolve the real sparse signal generator through the Provider registry (ADR-0004/0049) — the
    # thin core names the ``sparse.splade`` capability, not the provider package. A missing provider
    # package surfaces as a clear KeyError → exit 2 (install latence-splade), not a fake fallback.
    registry = ProviderRegistry()
    try:
        embedder_cls = registry.load("sparse.splade")
    except (KeyError, TypeError) as exc:
        typer.echo(
            f"sparse.splade Provider unavailable [{error_category(exc)}]: {exc}. Install the "
            "SPLADE Provider (e.g. `uv pip install -e packages/latence-splade`) so the real "
            "naver/splade-v3 sparse leg can be generated.",
            err=True,
        )
        raise typer.Exit(code=2) from exc
    embedder = embedder_cls({"model": model, "max_length": max_length})

    try:
        result = build_splade_index(
            export_dir=str(export),
            out_dir=str(out),
            embedder=embedder,
            model_id=model,
            weights_license=weights_license,
            batch_size=batch_size,
        )
    except (SpladeIndexError, LatenceError) as exc:
        typer.echo(f"splade-index failed [{error_category(exc)}]: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"splade-index: {result.chunks_indexed}/{result.chunks_total} chunks indexed "
        f"({result.chunks_empty} empty) → {result.index_uri}"
    )
    typer.echo(
        f"  model={result.model_id} vocab_size={result.vocab_size} "
        f"weights_license={result.weights_license}, "
        f"license_verified={str(result.license_verified).lower()}, "
        f"operator_accepted={str(result.operator_accepted).lower()}"
    )
    typer.echo(f"  manifest → {result.manifest_uri}")


def main() -> None:  # pragma: no cover - console entry shim
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
