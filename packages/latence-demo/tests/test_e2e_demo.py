"""End-to-end demo through the local Runner + the one-line Provider swap (S13).

The headline S13 acceptance run: DemoBuilder stages the bundled messy corpus, runs the full
spine on the built-in LocalRunner (CPU-only, offline), and produces the KG + the Quality
Report + the self-contained web viewer. Asserts every S13 acceptance criterion end to end:

* a dangerous fixture is visibly Quarantined; a prompt-injection fixture is visibly flagged;
* the viewer renders the KG (nodes/edges/Evidence) + the Quality Report with no external CDN;
* a one-line Provider swap reruns the identical Pipeline and the rerun reflects it;
* the run is deterministic and contract-complete (Baseline bar).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from latence_core.capability import ProviderRegistry
from latence_core.contracts import ChunkRecord, EntityMention, Provenance
from latence_core.pipeline import CapabilityKind
from latence_core.storage import Storage
from latence_demo.builder import DemoBuilder
from latence_demo.pipeline import build_pipeline


def test_demo_end_to_end(tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    result = DemoBuilder(storage_uri=store, run_id="e2e").run()
    summary = result.summary
    storage = Storage.for_uri(store)

    # 1. Dangerous fixtures Quarantined; the prompt-injection fixture flagged.
    assert summary.quarantined >= 3  # type-spoof + zip-bomb + oversized
    assert summary.flagged >= 1  # prompt-injection
    assert summary.dangerous_all_caught
    by_name = {d.file_name: d for d in summary.dangerous}
    assert by_name["invoice.pdf"].observed == "quarantine:type_spoof"
    assert by_name["archive.zip"].observed == "quarantine:zip_bomb"
    assert by_name["dump.txt"].observed == "quarantine:oversized"
    assert by_name["support-ticket.md"].observed == "flag:prompt_injection"

    # 2. The knowledge graph assembled with cross-document merge + KB links + Evidence.
    assert summary.graph.nodes_total >= 4
    assert summary.graph.edges_total >= 1
    assert summary.graph.nodes_linked >= 1
    assert summary.graph.edges_with_evidence == summary.graph.edges_total  # every edge justified

    # 3. All KG deliverables + the viewer landed as portable files (ADR-0017).
    export = f"{store}/_latence/runs/e2e/export"
    for name in ("graph-nodes.parquet", "graph-edges.parquet", "graph.ttl", "graph.graphml",
                 "corpus.jsonl", "index.html"):
        assert storage.exists(f"{export}/{name}"), name

    # 4. The viewer renders the KG + Quality Report and is self-contained (no external CDN).
    html = storage.read_text(summary.site_uri)
    assert "<svg" in html
    assert "works_for" in html  # a relation edge label
    assert "Quality Report" in html
    assert "Screening" in html
    assert "https://" not in html.lower()
    assert "<script src" not in html and "<link " not in html

    # 5. Contract-completeness held over the exported records (Baseline bar).
    assert summary.contracts_complete
    assert result.report.contracts_complete


def test_demo_is_deterministic(tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    DemoBuilder(storage_uri=store, run_id="det-a").run()
    DemoBuilder(storage_uri=store, run_id="det-b").run()
    storage = Storage.for_uri(store)
    base = f"{store}/_latence/runs"
    # The KG deliverables are byte-identical across runs (deterministic ids + ordering,
    # Baseline bar). The viewer's index.html is NOT asserted byte-equal because it embeds the
    # Quality Report's per-Stage wall-clock durations, which legitimately vary run to run; the
    # viewer's *graph* rendering determinism is covered by test_viewer with a fixed report.
    for name in ("graph.ttl", "graph.graphml", "graph-nodes.parquet", "graph-edges.parquet"):
        a = storage.read_bytes(f"{base}/det-a/export/{name}")
        b = storage.read_bytes(f"{base}/det-b/export/{name}")
        assert a == b, name
    # The KG summaries the two runs report are identical (same nodes/edges/links).
    a_manifest = storage.read_text(f"{base}/det-a/export/index.demo.json")
    b_manifest = storage.read_text(f"{base}/det-b/export/index.demo.json")
    assert a_manifest.replace("det-a", "X") == b_manifest.replace("det-b", "X")


def test_demo_resumes_from_checkpoints(tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    DemoBuilder(storage_uri=store, run_id="resume").run()
    result2 = DemoBuilder(storage_uri=store, run_id="resume").run()
    # Every Stage resumed from its checkpoint on the second run (a no-op re-run).
    assert all(m.from_checkpoint for m in result2.report.stages)
    assert result2.summary.dangerous_all_caught


# -- The one-line Provider swap (the model- and cloud-agnostic proof, ADR-0011) ------


class _FakeModelNer:
    """A stand-in 'model' NER Provider taking a FLAT zero-shot label list (like a real model).

    Registered into an injected registry under a swapped name so the demo's one-line Provider
    swap is exercised end to end WITHOUT installing a heavy model/endpoint dependency (ADR-0007).
    It labels the same entities as the gazetteer so the swapped run still assembles a graph,
    proving the swap changes the Provider — not the Pipeline, the contracts, or the viewer.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        labels = config.get("labels", [])
        # Prove the swapped Provider received the flat zero-shot label list (nothing hardcoded).
        assert isinstance(labels, list)
        self._terms = {
            "Alice": "PERSON", "Bob": "PERSON", "Carol": "PERSON",
            "Acme": "ORG", "IBM": "ORG", "Globex": "ORG",
        }

    def extract(self, chunks: list[ChunkRecord]) -> list[EntityMention]:
        mentions: list[EntityMention] = []
        for chunk in chunks:
            ordinal = 0
            for surface, label in self._terms.items():
                idx = chunk.content.find(surface)
                if idx < 0:
                    continue
                base = chunk.provenance.char_start or 0
                prov = Provenance(
                    source_uri=chunk.provenance.source_uri,
                    file_name=chunk.provenance.file_name,
                    file_type=chunk.provenance.file_type,
                    file_size=chunk.provenance.file_size,
                    document_id=chunk.provenance.document_id,
                    char_start=base + idx,
                    char_end=base + idx + len(surface),
                    page_start=chunk.provenance.page_start or 1,
                    page_end=chunk.provenance.page_end or 1,
                )
                mentions.append(
                    EntityMention(
                        record_id=f"{chunk.record_id}#{surface}-{idx}",
                        document_record_id=chunk.document_record_id,
                        chunk_record_id=chunk.record_id,
                        index=ordinal,
                        provenance=prov,
                        classification=chunk.classification,
                        text=surface,
                        label=label,
                        confidence=0.95,
                    )
                )
                ordinal += 1
        return mentions


def test_pipeline_swap_is_a_one_line_change() -> None:
    # The swap is literally one field: build_pipeline(entity_provider=...) changes ONLY the
    # entities Stage's provider; every other Stage is identical.
    base = build_pipeline("file:///s", "file:///s/corpus")
    swapped = build_pipeline("file:///s", "file:///s/corpus", entity_provider="entity.custom_model")
    base_stages = {s.name: s.provider for s in base.stages}
    swap_stages = {s.name: s.provider for s in swapped.stages}
    diff = {k for k in base_stages if base_stages[k] != swap_stages[k]}
    assert diff == {"entities"}
    assert swap_stages["entities"] == "entity.custom_model"
    # The swapped entities Stage now gets a FLAT zero-shot label list (a model, not a gazetteer).
    ent = next(s for s in swapped.stages if s.name == "entities")
    assert ent.capability is CapabilityKind.ENTITY_EXTRACTION
    assert isinstance(ent.config["labels"], list)


def test_demo_runs_the_swapped_provider_and_reflects_it(tmp_path: Path) -> None:
    # Inject a registry carrying the swapped 'model' Provider, then rerun the IDENTICAL Pipeline
    # against it. The rerun still assembles a graph + catches the dangerous fixtures — the swap
    # changed the Provider, nothing else (ADR-0011).
    registry = ProviderRegistry()
    registry._loaded["entity.custom_model"] = _FakeModelNer  # type: ignore[attr-defined]

    store = f"file://{tmp_path}/store"
    result = DemoBuilder(
        storage_uri=store,
        run_id="swap",
        entity_provider="entity.custom_model",
        registry=registry,
    ).run()

    # The swapped run reports the swapped Provider on the entities Stage.
    entities_metrics = [m for m in result.report.stages if m.name == "entities"]
    assert entities_metrics and entities_metrics[0].provider == "entity.custom_model"
    # Same downstream: a graph assembled and the dangerous fixtures were still caught.
    assert result.summary.graph.nodes_total >= 4
    assert result.summary.dangerous_all_caught
    assert result.summary.contracts_complete
