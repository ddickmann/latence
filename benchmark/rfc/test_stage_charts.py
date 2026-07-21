"""Exercise ``stage_charts.py`` end-to-end on a tiny synthetic run dir (no pod, no network).

The fixture hand-builds the exact artifacts the chart reader consumes — an ``export/records.jsonl``
(chunks with ``token_count`` + Provenance), stage checkpoints carrying entity/relation mentions, a
GraphML KG, and the corpus's independent ``_refs.jsonl`` — then asserts every claimed SVG is
written, is self-contained (no external asset / PNG), and carries the metric it proves.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_MODULE_PATH = Path(__file__).with_name("stage_charts.py")


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("stage_charts", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sc = _load_module()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


_GRAPHML = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="label" for="node" attr.name="label" attr.type="string"/>
  <key id="type" for="node" attr.name="type" attr.type="string"/>
  <key id="rel" for="edge" attr.name="label" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="n9110"><data key="label">RFC 9110</data><data key="type">protocol</data></node>
    <node id="n7230"><data key="label">RFC 7230</data><data key="type">protocol</data></node>
    <node id="n7231"><data key="label">RFC 7231</data><data key="type">protocol</data></node>
    <node id="n3864"><data key="label">RFC 3864</data><data key="type">protocol</data></node>
    <edge id="e1" source="n9110" target="n7230"><data key="rel">obsoletes</data></edge>
    <edge id="e2" source="n9110" target="n7231"><data key="rel">obsoletes</data></edge>
    <edge id="e3" source="n9110" target="n3864"><data key="rel">updates</data></edge>
  </graph>
</graphml>
"""


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus-rfc"
    corpus.mkdir()
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)

    # export/records.jsonl — three chunks, token_counts straddling the 640 budget, Provenance points
    # at the corpus dir so _refs.jsonl auto-discovers.
    _write_jsonl(
        run / "export" / "records.jsonl",
        [
            {
                "record_id": "c1",
                "token_count": 320,
                "content": "x",
                "provenance": {"source_uri": str(corpus / "rfc9110.txt")},
            },
            {
                "record_id": "c2",
                "token_count": 600,
                "content": "y",
                "provenance": {"source_uri": str(corpus / "rfc9110.txt")},
            },
            {
                "record_id": "c3",
                "token_count": 900,
                "content": "z",
                "provenance": {"source_uri": str(corpus / "rfc7230.txt")},
            },
        ],
    )

    # checkpoints — 5 entity mentions (c1 re-checkpointed by a 2nd stage to prove de-dup) that
    # collapse to 4 canonical graph nodes, and 3 relation mentions.
    entity_rows: list[dict[str, object]] = [
        {"record_id": f"m{i}", "chunk_record_id": "c1", "text": f"RFC {n}", "label": "protocol"}
        for i, n in enumerate([9110, 7230, 7231, 3864, 9110])
    ]
    relation_rows: list[dict[str, object]] = [
        {
            "record_id": f"r{i}",
            "head_mention_id": "m0",
            "tail_mention_id": f"m{i}",
            "label": "obsoletes",
        }
        for i in range(1, 4)
    ]
    _write_jsonl(run / "checkpoints" / "extract.jsonl", entity_rows + relation_rows)
    # A second stage re-checkpoints the same mention ids — must not double-count.
    _write_jsonl(run / "checkpoints" / "redact.jsonl", entity_rows)

    (run / "export" / "graph.graphml").write_text(_GRAPHML, encoding="utf-8")

    # Independent _refs.jsonl: 9110 obsoletes 7230,7231 (both agree) + updates 8888 (a gold edge the
    # extractor MISSED) → recall 3/4.
    _write_jsonl(
        corpus / "_refs.jsonl",
        [
            {"rfc": 9110, "obsoletes": [7230, 7231], "updates": [3864, 8888], "references": []},
        ],
    )
    return run


def test_stage_charts_writes_every_claimed_svg(run_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "stage-charts"
    rc = sc.main(["--run", str(run_dir), "--out", str(out)])
    assert rc == 0

    expected = {
        "chunk-size-distribution.svg",
        "entities-relations-funnel.svg",
        "kg-degree-and-centrality.svg",
        "induced-vs-independent-overlap.svg",
    }
    produced = {p.name for p in out.glob("*.svg")}
    assert expected == produced

    for svg in out.glob("*.svg"):
        text = svg.read_text(encoding="utf-8")
        assert text.startswith("<svg")
        # Self-contained (ADR-0016): no PNG, no external asset reference.
        assert "<image" not in text
        assert "http://" not in text.replace("http://www.w3.org/2000/svg", "").replace(
            "http://graphml.graphdrawing.org/xmlns", ""
        )
        assert "https://" not in text


def test_charts_carry_the_right_metrics(run_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "stage-charts"
    sc.main(["--run", str(run_dir), "--out", str(out)])

    # chunk sizes: 2 of 3 chunks are at/under budget 640 → 66.7%.
    chunk = (out / "chunk-size-distribution.svg").read_text(encoding="utf-8")
    assert "3 chunks" in chunk
    assert "66.7% at/under 640" in chunk

    # funnel: 5 raw entity mentions collapse to 4 canonical nodes.
    funnel = (out / "entities-relations-funnel.svg").read_text(encoding="utf-8")
    assert "5→4" in funnel

    # overlap: induced {obsoletes 7230, obsoletes 7231, updates 3864}; independent adds updates 8888
    # → recall 3/4 = 75.0%.
    overlap = (out / "induced-vs-independent-overlap.svg").read_text(encoding="utf-8")
    assert "recall 75.0%" in overlap
    assert "3/4 gold edges reproduced" in overlap


def test_direct_readers_and_metrics(run_dir: Path) -> None:
    assert sc.load_chunk_tokens(run_dir) == [320, 600, 900]
    assert sc.load_mention_counts(run_dir) == (5, 3)  # de-duplicated across two checkpoints

    graph = sc.load_graph(run_dir)
    assert len(graph.names) == 4
    assert len(graph.edges) == 3
    assert sc.induced_typed_edges(graph) == {
        (9110, "obsoletes", 7230),
        (9110, "obsoletes", 7231),
        (9110, "updates", 3864),
    }

    # PageRank: the hub RFC 9110 (all edges point away from it, so mass flows to its targets) — the
    # scores are all finite and sum to ~1.
    rank = sc.pagerank(graph)
    assert abs(sum(rank.values()) - 1.0) < 1e-6
    assert set(rank) == set(graph.node_ids)


def test_missing_refs_degrades_to_placeholder(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "export").mkdir(parents=True)
    _write_jsonl(run / "export" / "records.jsonl", [{"record_id": "c1", "token_count": 100}])
    (run / "export" / "graph.graphml").write_text(_GRAPHML, encoding="utf-8")
    out = tmp_path / "charts"
    rc = sc.main(["--run", str(run), "--out", str(out)])
    assert rc == 0
    overlap = (out / "induced-vs-independent-overlap.svg").read_text(encoding="utf-8")
    assert "independent _refs.jsonl not found" in overlap
    # The other charts still render from what IS present.
    assert "1 chunks" in (out / "chunk-size-distribution.svg").read_text(encoding="utf-8")
