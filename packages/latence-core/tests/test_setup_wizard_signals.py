"""The enterprise-sota profile's INDEX-TIME SIGNALS: graph features (#193) and BM25 (#194).

ADR-0041 says the wizard emits ONE opinionated, quality-baked config. Two signals whose machinery
the framework already owned were missing from it:

* **#193 / ADR-0051** — the emitted ``graph`` Stage carried no ``graph_features`` block, so
  centrality + community were never computed and the exported graph-nodes artifact had no
  ``centrality`` / ``community`` columns (the signals multi-hop expansion and the Knapsack packer
  read).
* **#194 / ADR-0049** — the emitted ``export_corpus`` Stage carried no ``bm25`` block, so the BM25
  term-statistics artifact was never produced and the query-side sparse rescorer had nothing to
  run on.

Each fix is covered TWICE, per its acceptance criteria: the emitted config carries the block, AND a
REAL run of the Stage the block lands on materializes the columns / the artifact. The run tests
drive the actual Stage implementations with the config the wizard emitted — read straight out of
the generated YAML, never hand-written — so a wiring regression in the wizard fails them.

Offline scope note: the enterprise-sota profile is a GPU-pod stack (served-vLLM OCR, GLiNER2,
GLinker, Granite r2). The two Stages under test here are pure-Python and CPU-only, so they run for
real. The corpus-Export run substitutes the emitted Granite ``embedder`` block (a multi-GB model
download) away; every other key — including the ``bm25`` block under test — is used verbatim.
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
from latence_core.conformance import fixtures
from latence_core.contracts import Classification, ChunkRecord, Provenance
from latence_core.pipeline import Pipeline
from latence_core.setup import SetupAnswers, build_stack_config
from latence_core.setup.wizard import PROFILE_LITE, PROFILE_SOTA
from latence_core.stages.export import JsonlParquetExport
from latence_core.stages.graph_assembly import CanonicalGraphAssembler
from latence_core.stages.graph_export import KnowledgeGraphExport
from latence_core.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# The emitted config (the single source of truth for every test below)
# ---------------------------------------------------------------------------


def _answers(tmp_path: Path, *, profile: str = PROFILE_SOTA) -> SetupAnswers:
    src = tmp_path / "corpus"
    src.mkdir(exist_ok=True)
    (src / "a.pdf").write_text("a", encoding="utf-8")
    return SetupAnswers(
        source=f"file://{src}",
        storage=f"file://{tmp_path}/out",
        parser="parser.render",
        device="auto",
        pii=True,
        extract_on="unmasked",
        profile=profile,
        extensions=("pdf", "txt"),
    )


def _stage_config(tmp_path: Path, name: str, *, profile: str = PROFILE_SOTA) -> dict[str, Any]:
    text = build_stack_config(_answers(tmp_path, profile=profile))
    stage = next(s for s in Pipeline.from_yaml_text(text).stages if s.name == name)
    return dict(stage.config)


# ---------------------------------------------------------------------------
# #193 — graph features on the emitted `graph` Stage
# ---------------------------------------------------------------------------


def test_sota_profile_emits_the_graph_feature_block(tmp_path: Path) -> None:
    """AC1: the emitted graph Stage names a graph-feature computer (the pagerank reference)."""
    block = _stage_config(tmp_path, "graph").get("graph_features")
    assert isinstance(block, dict)
    assert block["provider"] == "graph_features.pagerank"


def _export_nodes_table(records: list[Any], tmp_path: Path, sub: str = "kg") -> Any:
    store = f"file://{tmp_path}"
    storage = Storage.for_uri(store)
    KnowledgeGraphExport().export(records, storage, f"{store}/{sub}")
    return pq.read_table(  # type: ignore[no-untyped-call]
        io.BytesIO(storage.read_bytes(f"{store}/{sub}/graph-nodes.parquet"))
    )


def test_sota_graph_config_produces_centrality_and_community_columns(tmp_path: Path) -> None:
    """AC2 (the run): assembling with the EMITTED graph config and exporting the KG yields
    graph-nodes rows that actually carry centrality + community values."""
    config = _stage_config(tmp_path, "graph")
    records = list(CanonicalGraphAssembler(config).assemble(fixtures.graph_records()))

    table = _export_nodes_table(records, tmp_path)
    assert {"centrality", "community"} <= set(table.column_names)
    centralities = table.column("centrality").to_pylist()
    communities = table.column("community").to_pylist()
    assert centralities and communities  # the fixture graph has nodes
    # Real values on every row (the assembler stamps all-or-none), and PageRank is a probability
    # distribution over the nodes, so each score is a finite fraction.
    assert all(isinstance(c, float) and 0.0 <= c <= 1.0 for c in centralities)
    assert all(isinstance(c, str) and c for c in communities)


def test_sota_graph_features_are_deterministic_and_additive(tmp_path: Path) -> None:
    """AC4 (#161 discipline): the emitted config is deterministic run-to-run, and dropping the
    graph-feature block reproduces the plain graph export byte-for-byte."""
    config = _stage_config(tmp_path, "graph")
    first = list(CanonicalGraphAssembler(config).assemble(fixtures.graph_records()))
    second = list(CanonicalGraphAssembler(config).assemble(fixtures.graph_records()))
    assert [r.model_dump(mode="json") for r in first] == [
        r.model_dump(mode="json") for r in second
    ]

    off = {k: v for k, v in config.items() if k != "graph_features"}
    off_records = list(CanonicalGraphAssembler(off).assemble(fixtures.graph_records()))
    storage = Storage.for_uri(f"file://{tmp_path}")
    KnowledgeGraphExport().export(off_records, storage, f"file://{tmp_path}/off")
    plain = list(CanonicalGraphAssembler({"corpus_id": config["corpus_id"]}).assemble(
        fixtures.graph_records()
    ))
    KnowledgeGraphExport().export(plain, storage, f"file://{tmp_path}/plain")
    assert storage.read_bytes(f"file://{tmp_path}/off/graph-nodes.parquet") == storage.read_bytes(
        f"file://{tmp_path}/plain/graph-nodes.parquet"
    )


# ---------------------------------------------------------------------------
# #194 — the BM25 signal on the emitted corpus Export
# ---------------------------------------------------------------------------


def test_sota_profile_emits_the_bm25_block(tmp_path: Path) -> None:
    """AC1: the emitted export_corpus Stage names a BM25 tokenizer (the regex reference)."""
    block = _stage_config(tmp_path, "export_corpus").get("bm25")
    assert isinstance(block, dict)
    assert block["provider"] == "tokenizer.regex"


def _chunk(record_id: str, text: str, index: int) -> ChunkRecord:
    return ChunkRecord(
        record_id=record_id,
        provenance=Provenance(
            source_uri="file:///c/a.txt",
            file_name="a.txt",
            file_type="txt",
            file_size=100,
            document_id="sha256:d1",
            char_start=0,
            char_end=len(text),
            page_start=1,
            page_end=1,
        ),
        classification=Classification(language="de", sensitivity="internal"),
        document_record_id="d1",
        index=index,
        content=text,
        token_count=max(1, len(text.split())),
    )


def _cpu_export_config(tmp_path: Path) -> dict[str, Any]:
    """The emitted export_corpus config minus only the GPU-bound dense embedder (see module docs)."""
    config = _stage_config(tmp_path, "export_corpus")
    config.pop("embedder", None)
    return config


def test_sota_export_config_materializes_the_bm25_artifact(tmp_path: Path) -> None:
    """AC2 (the run): exporting with the EMITTED export_corpus config writes the term-statistics
    artifact alongside the corpus, with real per-term statistics in it."""
    records = [
        _chunk("c1", "Zolltarif Einreihung der Ware", 0),
        _chunk("c2", "Einreihung und Ausfuhrkontrolle", 1),
    ]
    store = f"file://{tmp_path}/out"
    storage = Storage.for_uri(store)
    uris = JsonlParquetExport(_cpu_export_config(tmp_path)).export(records, storage, store)

    stats_uri = next(u for u in uris if u.endswith("bm25-stats.json"))
    postings_uri = next(u for u in uris if u.endswith("bm25-postings.parquet"))
    assert storage.exists(stats_uri) and storage.exists(postings_uri)

    stats: dict[str, Any] = json.loads(storage.read_text(stats_uri))
    assert stats["artifact"] == "latence.bm25.termstats"
    assert stats["doc_count"] == 2
    # "einreihung" appears in BOTH documents; "zolltarif" in one — real corpus statistics, not a
    # placeholder file.
    df = {t["term"]: t["df"] for t in stats["terms"]}
    assert df["einreihung"] == 2
    assert df["zolltarif"] == 1

    with storage.open(postings_uri, "rb") as fh:
        postings = pq.read_table(fh)  # type: ignore[no-untyped-call]
    assert postings.num_rows == 2


def test_sota_bm25_is_additive_to_the_corpus_export(tmp_path: Path) -> None:
    """Dropping the bm25 block reproduces the plain corpus export byte-for-byte (ADR-0049)."""
    records = [_chunk("c1", "alpha beta", 0)]
    config = _cpu_export_config(tmp_path)
    off = {k: v for k, v in config.items() if k != "bm25"}

    on_store = f"file://{tmp_path}/on"
    off_store = f"file://{tmp_path}/off"
    on_storage, off_storage = Storage.for_uri(on_store), Storage.for_uri(off_store)
    JsonlParquetExport(config).export(records, on_storage, on_store)
    off_uris = JsonlParquetExport(off).export(records, off_storage, off_store)

    assert not any(u.rsplit("/", 1)[-1].startswith("bm25-") for u in off_uris)
    basename = str(config["basename"])
    assert on_storage.read_bytes(f"{on_store}/{basename}.parquet") == off_storage.read_bytes(
        f"{off_store}/{basename}.parquet"
    )


# ---------------------------------------------------------------------------
# The lite profile is untouched by either fix
# ---------------------------------------------------------------------------


def test_lite_profile_carries_neither_signal(tmp_path: Path) -> None:
    """Both signals are enterprise-sota-only: the CPU-first default is unchanged."""
    assert "graph_features" not in _stage_config(tmp_path, "graph", profile=PROFILE_LITE)
    assert "bm25" not in _stage_config(tmp_path, "export_corpus", profile=PROFILE_LITE)
