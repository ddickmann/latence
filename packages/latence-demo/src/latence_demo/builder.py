"""DemoBuilder — run the whole showable demo end to end (S13, ADR-0011).

One call: stage the bundled messy corpus onto Storage, run the full-spine demo Pipeline on
the built-in local Runner (CPU-only, offline), then compose the self-contained showable page
(KG + Quality Report + corpus) and a typed :class:`DemoRunSummary`. This is the code
``latence-demo`` (the CLI) and ``docker compose up`` both call — the single reproducible path
from a clean clone to the artifact.

The DemoBuilder reuses the existing seams verbatim — the ``LocalRunner``, the ``Storage``
seam, the reference Providers, the persisted ``quality-report.json`` — and never forks them.
The only demo-specific work is corpus staging, folding the finalised Quality Report into the
viewer (the Export seam carries records, not the post-run report), and asserting the planted
dangerous fixtures were caught.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from latence_core.contracts import GraphEdge, GraphNode
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

if TYPE_CHECKING:
    from latence_core.capability import ProviderRegistry

from latence_demo.contracts import (
    DangerousFixtureOutcome,
    DemoRunSummary,
    GraphSummary,
)
from latence_demo.corpus import CorpusManifest, stage_corpus
from latence_demo.pipeline import build_pipeline
from latence_demo.viewer import (
    load_graph_records,
    load_report,
    render_site,
    summarize_graph,
)

# Default demo storage root — a local folder (the CPU-only, no-cloud default, ADR-0007).
# NOTE (hardening): this path is under a shared, world-writable ``/tmp`` and is predictable,
# which is fine for a throwaway local demo but NOT for a real deployment — a co-located user
# could pre-create or symlink the path. Adopters should pass a private, per-run ``--storage-uri``
# (or an ``s3://``/``gs://`` root) rather than rely on this default.
DEFAULT_STORAGE_URI = "file:///tmp/latence-demo"


@dataclass(frozen=True)
class DemoResult:
    """Everything the demo produced: the typed summary + where the artifacts landed."""

    summary: DemoRunSummary
    report: QualityReport
    manifest: CorpusManifest


class DemoBuilder:
    """Stage → run → compose the showable demo. Reuses the LocalRunner + Storage seams.

    Config:
        storage_uri: str — Storage root for the corpus + run outputs (default a local folder).
        run_id: str — run/checkpoint namespace (default 'demo').
        entity_provider / entity_config — the one-line Provider swap (ADR-0011): default the
            in-core CPU gazetteer; pass ``entity.gliner`` / ``entity.endpoint`` to rerun the
            identical Pipeline against a local model or a cloud endpoint.
    """

    def __init__(
        self,
        storage_uri: str = DEFAULT_STORAGE_URI,
        *,
        run_id: str = "demo",
        entity_provider: str = "entity.gazetteer",
        name: str = "latence-demo",
        registry: ProviderRegistry | None = None,
    ) -> None:
        self._storage_uri = storage_uri.rstrip("/")
        self._run_id = run_id
        self._entity_provider = entity_provider
        self._name = name
        # An optional Provider registry the LocalRunner uses. Left ``None`` (the default) the
        # Runner discovers Providers from the ``latence.providers`` entry points; a test (or an
        # adopter) can inject a registry carrying an extra Provider — e.g. to exercise the
        # one-line NER swap against a Provider whose heavy/endpoint deps are not installed.
        self._registry = registry

    def run(self) -> DemoResult:
        storage = Storage.for_uri(self._storage_uri)
        corpus_folder = f"{self._storage_uri}/corpus"
        manifest = stage_corpus(storage, corpus_folder)

        # The Source reads a Storage URI; the corpus folder is a URI on the same Storage.
        pipeline = build_pipeline(
            self._storage_uri,
            corpus_folder,
            name=self._name,
            entity_provider=self._entity_provider,
        )

        report = LocalRunner(registry=self._registry).run(pipeline, run_id=self._run_id)
        run_dir = f"{self._storage_uri}/_latence/runs/{self._run_id}"

        # Compose the full showable page: recover the assembled graph from the Graph-Assembly
        # checkpoint + fold in the finalised Quality Report + the corpus manifest, then write
        # index.html next to the KG export (the Export-seam page had no report yet).
        site_uri = self._compose_site(storage, run_dir, report, manifest)

        summary = self._summarize(storage, run_dir, site_uri, report, manifest)
        return DemoResult(summary=summary, report=report, manifest=manifest)

    def _compose_site(
        self,
        storage: Storage,
        run_dir: str,
        report: QualityReport,
        manifest: CorpusManifest,
    ) -> str:
        graph_ckpt = f"{run_dir}/checkpoints/graph.jsonl"
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        if storage.exists(graph_ckpt):
            nodes, edges = load_graph_records(storage.read_text(graph_ckpt))
        report_dict = load_report(report.to_json())
        corpus_view = {name: manifest.dangerous.get(name, "clean") for name in manifest.documents}
        html = render_site(
            nodes,
            edges,
            pipeline_name=self._name,
            report=report_dict,
            corpus=corpus_view,
        )
        site_uri = f"{run_dir}/export/index.html"
        storage.write_atomic(site_uri, html.encode("utf-8"))
        return site_uri

    def _summarize(
        self,
        storage: Storage,
        run_dir: str,
        site_uri: str,
        report: QualityReport,
        manifest: CorpusManifest,
    ) -> DemoRunSummary:
        graph_ckpt = f"{run_dir}/checkpoints/graph.jsonl"
        if storage.exists(graph_ckpt):
            nodes, edges = load_graph_records(storage.read_text(graph_ckpt))
            graph = summarize_graph(nodes, edges)
        else:  # pragma: no cover - the demo always assembles a graph
            graph = GraphSummary()

        quarantined = report.screening.quarantined if report.screening else 0
        flagged = report.screening.flagged if report.screening else 0
        dangerous = self._score_dangerous(report, manifest)
        return DemoRunSummary(
            pipeline_name=report.pipeline_name,
            run_id=report.run_id,
            storage_uri=self._storage_uri,
            run_dir=run_dir,
            site_uri=site_uri,
            documents_ingested=report.document_count,
            quarantined=quarantined,
            flagged=flagged,
            graph=graph,
            dangerous=dangerous,
            contracts_complete=report.contracts_complete,
        )

    @staticmethod
    def _score_dangerous(
        report: QualityReport, manifest: CorpusManifest
    ) -> list[DangerousFixtureOutcome]:
        """Prove each planted dangerous fixture got its expected Screening disposition.

        Binds every quarantine/flag finding in the Quality Report to its source file **by
        ``document_id``** — the content-addressed id the Source stamps and the finding carries —
        via the manifest's ``file_name -> document_id`` map, then compares the observed
        disposition against that file's planted intent. Because the match is per-file, a
        'right reason, wrong file' mislabel (e.g. the injection flag landing on the wrong
        document) is caught, not passed. A mismatch — a dangerous file that slipped through, or
        the expected disposition landing on a different file — surfaces as ``matched=False``, so
        the demo fails loudly rather than asserting a false "it's safe" (S13 AC).
        """
        # document_id -> "verdict:category" observed from the Screening findings. Each finding
        # carries the document_id of the exact record it screened, so we can attribute the
        # disposition to the specific planted file rather than to a coincidental pair.
        observed: dict[str, str] = {}
        screening = report.screening
        if screening is not None:
            for f in screening.findings:
                if f.verdict.value == "pass":
                    continue
                observed[f.document_id] = f"{f.verdict.value}:{f.category}"

        outcomes: list[DangerousFixtureOutcome] = []
        for file_name, expected in sorted(manifest.dangerous.items()):
            doc_id = manifest.document_ids.get(file_name)
            got = observed.get(doc_id, "not-detected") if doc_id is not None else "not-detected"
            outcomes.append(
                DangerousFixtureOutcome(
                    file_name=file_name,
                    expected=expected,
                    observed=got,
                    matched=(got == expected),
                )
            )
        return outcomes
