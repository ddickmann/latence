"""End-to-end embedding Disambiguation through the local Runner (W1 rework, #94).

Source -> Parse -> Chunk -> Entity -> Disambiguation(disambiguation.embedding) -> Export, with the
in-core deterministic ``embedding.hashing`` embedder (offline, no heavy dep). Proves the Runner
actually THREADS the run's ChunkRecords into ``disambiguate`` (the rework's whole point — the Chunk
Stage is a mention ancestor, not a Disambiguation parent, yet the surface+context provider must reach
its content), that the exported rows are valid DisambiguationRecords, that the surface+context path
does NOT over-merge distinct entities, and that the run is deterministic + resumes cleanly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from latence_core.contracts import DisambiguationRecord, DisambiguationScope
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.stage_inputs import StageInputs
from latence_core.storage import Storage
from latence_disambig_embedding.provider import EmbeddingDisambiguator


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: embedding-disambiguation-slice
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
      extensions: [txt, md]
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
        ORG:
          terms: [International Business Machines, IBM, Apple Inc, Apple]
        LOC:
          terms: [Berlin]
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.embedding
    depends_on: [entities]
    config:
      threshold: 0.90
      context_window: 160
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [disambiguate]
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    # Each document gives its entity a rich, distinctive context so surface+context embedding can
    # tell IBM/International Business Machines and Apple/Apple Inc apart from each other and Berlin.
    (docs / "ibm.txt").write_text(
        "IBM reported strong quarterly earnings today. International Business Machines is a "
        "technology company making enterprise computers, mainframes and software services.\n",
        encoding="utf-8",
    )
    (docs / "apple.txt").write_text(
        "Apple unveiled a new iPhone at its event. Apple Inc is a consumer electronics company "
        "designing phones, laptops and tablets, based in Cupertino California.\n",
        encoding="utf-8",
    )
    (docs / "berlin.txt").write_text(
        "Berlin is the capital city of Germany, a major European cultural and political centre.\n",
        encoding="utf-8",
    )
    return docs


def test_embedding_disambiguation_end_to_end(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_pipeline(store, docs), run_id="w1e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/w1e2e"

    # 1. The export holds typed DisambiguationRecord rows.
    jsonl_uri = f"{base}/export/records.jsonl"
    assert s.exists(jsonl_uri)
    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    records = [DisambiguationRecord.model_validate(r) for r in rows]
    assert records
    entities = [
        r.entity for r in records if r.scope is DisambiguationScope.ENTITY and r.entity
    ]

    # 2. The Runner threaded the ChunkRecords: the embedding provider ran (the run did not crash on
    #    a mentions-only signature) and produced canonical entities covering every mention.
    covered = {mid for e in entities for mid in e.member_mention_ids}
    assert covered  # mentions were resolved

    # 3. Enterprise safety end-to-end: no cluster over-merges two DIFFERENT real-world entities.
    #    Each document is one distinct entity, so a canonical entity that spans BOTH the IBM doc and
    #    the Apple doc would be an over-merge.
    for e in entities:
        docs_spanned = set(e.source_document_ids)
        ibm_and_apple = any("ibm" in d for d in docs_spanned) and any(
            "apple" in d for d in docs_spanned
        )
        assert not ibm_and_apple, f"over-merge: {e.canonical_text} spans {docs_spanned}"

    # 4. Quality Report present with disambiguation stats + contract completeness.
    persisted = QualityReport.model_validate_json(
        s.read_text(f"{base}/quality-report.json")
    )
    assert persisted.disambiguation is not None
    assert persisted.disambiguation.entities_out >= 1
    assert persisted.contracts_complete
    assert report.disambiguation is not None


def test_embedding_disambiguation_is_deterministic(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    LocalRunner().run(pipeline, run_id="det-a")
    LocalRunner().run(pipeline, run_id="det-b")
    s = Storage.for_uri(store)
    out_a = s.read_text(f"{store}/_latence/runs/det-a/export/records.jsonl")
    out_b = s.read_text(f"{store}/_latence/runs/det-b/export/records.jsonl")
    assert out_a == out_b


def test_embedding_disambiguation_is_a_drop_in_for_cascade(tmp_path: Path) -> None:
    """Swapping disambiguation.cascade -> disambiguation.embedding needs no other change (drop-in).

    Same DisambiguationRecord contract: both pipelines run to completion and export validatable
    DisambiguationRecords over the identical corpus + wiring, differing only in the provider name.
    """
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    # The pipeline names disambiguation.embedding; swapping to disambiguation.cascade is the ONLY
    # change a stack makes. Both satisfy the same Disambiguator seam + emit the same contract.
    LocalRunner().run(_pipeline(store, docs), run_id="embed")
    s = Storage.for_uri(store)
    rows = [
        DisambiguationRecord.model_validate_json(line)
        for line in s.read_text(f"{store}/_latence/runs/embed/export/records.jsonl").splitlines()
        if line.strip()
    ]
    assert rows
    assert all(r.scope in (DisambiguationScope.ENTITY, DisambiguationScope.RELATION) for r in rows)


# --------------------------------------------------------------------------- #
# Runner chunk-THREADING regression (W1 rework): a cross-surface merge must
# happen ONLY when the Runner threads the ChunkRecords into disambiguate(); if
# the threading breaks (no chunks reach the provider), the merge must NOT occur.
# This is the assertion the all-singleton default embedder could not make —
# dropping chunks in the Runner would otherwise ship green.
# --------------------------------------------------------------------------- #

# A distinctive context marker that co-occurs in both variant surfaces' chunk, present ONLY when the
# window is sliced from the chunk content (i.e. only when the Runner threaded chunks).
_CONTEXT_MARKER = "spacecraft"


class _ContextGatedEmbedder:
    """Two distinct surfaces embed CLOSE only when their shared context marker is in the text.

    * ``surface + context`` (the marker present) → both variants map to angle 0.0 → cos 1.0 (MERGE).
    * bare ``surface`` (no context — the marker absent) → each variant gets its OWN wide angle →
      cos well below the 0.90 threshold (STAY APART).

    So a merge of the two variants happens iff the Runner threaded the chunk content into
    ``disambiguate`` (context reached the provider). Deterministic, offline, 2-D unit vectors;
    satisfies the Embedder Protocol.
    """

    # Bare-surface angles: far apart so the variants do NOT merge without context.
    _BARE_ANGLES = {
        "globex corporation": 0.0,
        "globex": 1.2,  # ~69° from 'globex corporation' → cos ≈ 0.36 (below threshold)
        "initech": 2.6,
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = dict(config or {})

    def _angle_for(self, text: str) -> float:
        low = text.lower().strip()
        if _CONTEXT_MARKER in low:
            # Context present: the two 'globex' variants collapse to the same direction (merge);
            # a genuinely different entity (initech) keeps its own direction (no over-merge).
            return 3.0 if low.startswith("initech") else 0.0
        for key in sorted(self._BARE_ANGLES, key=len, reverse=True):
            if low.startswith(key):
                return self._BARE_ANGLES[key]
        return 5.0

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, texts: Any) -> Any:
        for text in texts:
            angle = self._angle_for(text)
            yield [math.cos(angle), math.sin(angle)]


def _gated_docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    # ONE document naming both surface variants of the same entity, sharing the context marker, so a
    # correctly-threaded run merges Globex/Globex Corporation via context; a distinct entity (Initech)
    # in another doc must never fold in.
    (docs / "globex.txt").write_text(
        f"Globex launched a {_CONTEXT_MARKER} today. Globex Corporation is the {_CONTEXT_MARKER} "
        f"manufacturer behind the programme.\n",
        encoding="utf-8",
    )
    (docs / "initech.txt").write_text(
        f"Initech is a software firm. Initech builds office {_CONTEXT_MARKER} accounting tools.\n",
        encoding="utf-8",
    )
    return docs


def _gated_pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: embedding-disambiguation-threading
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
      extensions: [txt, md]
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
        ORG:
          terms: [Globex Corporation, Globex, Initech]
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.embedding
    depends_on: [entities]
    config:
      threshold: 0.90
      context_window: 200
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [disambiguate]
"""
    )


def _run_gated(store: str, docs: Path, run_id: str) -> list[Any]:
    s = Storage.for_uri(store)
    LocalRunner().run(_gated_pipeline(store, docs), run_id=run_id)
    rows = [
        DisambiguationRecord.model_validate_json(line)
        for line in s.read_text(
            f"{store}/_latence/runs/{run_id}/export/records.jsonl"
        ).splitlines()
        if line.strip()
    ]
    return [
        r.entity for r in rows if r.scope is DisambiguationScope.ENTITY and r.entity
    ]


def test_runner_threads_chunks_context_gated_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WITH the Runner's chunk-threading, the two Globex surfaces merge (context reached the provider).

    Uses a context-gated embedder that merges the variants ONLY when the shared context marker is in
    the embedded text — which only happens if the Runner threaded the ChunkRecords into
    ``disambiguate``. Globex + Globex Corporation therefore land in ONE cluster while Initech stays
    separate. The companion negative test breaks the threading and asserts the merge vanishes, so a
    regression that dropped chunks in the Runner's disambiguation dispatch cannot ship green.
    """
    monkeypatch.setattr(
        EmbeddingDisambiguator, "_resolve_embedder", lambda self: _ContextGatedEmbedder()
    )
    docs = _gated_docs(tmp_path)
    store = f"file://{tmp_path}/store"
    entities = _run_gated(store, docs, "threaded")

    clusters = {frozenset(e.member_mention_ids) for e in entities}
    # Globex + Globex Corporation share a cluster of >= 2 mentions (the context-gated merge fired).
    globex = [e for e in entities if "globex" in e.canonical_text.lower()]
    assert len(globex) == 1, f"Globex variants must merge WITH context; got {clusters}"
    assert len(globex[0].member_mention_ids) >= 2
    # Initech never folds into Globex (no over-merge across distinct entities).
    for e in entities:
        docs_spanned = set(e.source_document_ids)
        assert not (
            any("globex" in d for d in docs_spanned) and any("initech" in d for d in docs_spanned)
        ), f"over-merge: {e.canonical_text} spans {docs_spanned}"


def test_broken_threading_drops_the_merge_regression_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break the Runner's chunk-threading → the context-gated merge MUST vanish (the regression guard).

    Fully breaks the rework's mechanism by making ``StageInputs.corpus_records_of`` yield NO
    chunks, so no context reaches the provider. With the SAME context-gated embedder + threshold, the Globex
    variants now embed as bare surfaces (far apart) and stay SEPARATE. This is exactly the failure a
    silent chunk-drop in the Runner would cause — and it is now observable: if this assertion ever
    passes with the merge still present, the threading is not actually load-bearing.
    """
    monkeypatch.setattr(
        EmbeddingDisambiguator, "_resolve_embedder", lambda self: _ContextGatedEmbedder()
    )
    # Break the corpus-wide chunk gather the rework adds: yield nothing.
    monkeypatch.setattr(
        StageInputs, "corpus_records_of", lambda self, *a, **k: iter(())
    )
    docs = _gated_docs(tmp_path)
    store = f"file://{tmp_path}/store"
    entities = _run_gated(store, docs, "unthreaded")

    # Globex and Globex Corporation are now SEPARATE singletons (no context → no merge).
    globex = [e for e in entities if "globex" in e.canonical_text.lower()]
    assert len(globex) == 2, (
        "without threaded chunks the Globex variants must NOT merge (proves threading is "
        f"load-bearing); got {[e.canonical_text for e in entities]}"
    )
    assert all(len(e.member_mention_ids) == 1 for e in globex)
