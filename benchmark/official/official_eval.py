# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""M6 OFFICIAL eval (ticket 08's full design): per-tier curves, four A/Bs, recall@5.

Config axes (every one pre-registered here, tuned on DEV only, scored ONCE on test):

* **Entry tier** (ADR-0059): tier0 = hashing-embedder ANN over node descriptions (the
  zero-dep lexical floor — an adopter's learned embedder is strictly stronger; framed as
  such), tier1 = question n-grams vs the alias KV, tier2 = GLiNER2 query surfaces (from the
  pod's query-surfaces.jsonl) vs the alias KV, tier12 = union.
* **A/B heuristic-vs-exact token costs** (ADR-0058): the exact variant rebuilds the KV
  locally with tiktoken cl100k_base counts over tau_text (KV is derived + rebuildable).
* **A/B weak on/off** (ADR-0057): weak hyperedges filtered from the candidate signals.
* **A/B calibrated γ** (ticket 09): isotonic fit from judged samples, applied to gamma in
  solver fields before selection.
* **recall@5**: paragraph-level — baseline ranks paragraphs by best BM25 chunk; CoE ranks
  by best hyperedge ρ mapped through its primary chunk's source file.

Composition policy: the subset-gate winner (piecewise: baseline@1k, CoE@≥2k with dev-tuned
share) re-tuned per tier on dev.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "musique-subset"))
from run_eval_final import (  # noqa: E402
    BUDGETS,
    Arm,
    _norm,
    _recalled,
    gate,
    load_chunks,
    question_ngrams,
)

from latence_core.chunking import estimate_tokens  # noqa: E402
from latence_retrieval.hypergraph import (  # noqa: E402
    HyperedgeKV,
    HypergraphSignals,
    hypergraph_signals,
)

SHARES = [0.25, 0.5]


# -- tier-0 entry: hashing-embedder ANN over node descriptions -------------------


class Tier0Index:
    """Lexical-floor ANN: hashed bag-of-words vectors over node descriptions (numpy matmul —
    pure-Python cosine over 112k–349k nodes was the measured per-question wall)."""

    DIM = 256

    def __init__(self, nodes_parquet: str) -> None:
        import numpy as np
        import pyarrow.parquet as pq

        rows = pq.read_table(nodes_parquet).to_pylist()
        self.ids: list[str] = []
        vecs = []
        for r in rows:
            props = json.loads(r["properties"]) if r.get("properties") else {}
            text = props.get("description") or r["canonical_name"]
            self.ids.append(r["node_id"])
            vecs.append(self._embed(text))
        self.matrix = np.array(vecs, dtype="float32")

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        v = [0.0] * cls.DIM
        for tok in re.findall(r"\w+", text.lower()):
            v[hash(tok) % cls.DIM] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def topk(self, query: str, k: int = 10) -> list[str]:
        import numpy as np

        qv = np.array(self._embed(query), dtype="float32")
        scores = self.matrix @ qv
        idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [self.ids[i] for i in idx if scores[i] > 0]


# -- signal post-processors for the A/Bs -----------------------------------------


def filter_weak(signals: HypergraphSignals) -> HypergraphSignals:
    keep = {h for h, f in signals.fields.items() if f.get("has_relations")}
    return HypergraphSignals(
        rho={h: v for h, v in signals.rho.items() if h in keep},
        fields={h: f for h, f in signals.fields.items() if h in keep},
        anchors=signals.anchors,
        vertex_confidence=signals.vertex_confidence,
        expanded_candidates=signals.expanded_candidates,
        pruned_candidates=signals.pruned_candidates,
    )


def apply_calibration(signals: HypergraphSignals, calibrate) -> HypergraphSignals:
    """``calibrate(gamma, has_relations)`` — per-regime, because the judged sample showed
    gamma<=0.5 <=> weak BY CONSTRUCTION and an inverted-U global trend (a single global
    isotonic fit would flatten both regimes' monotone signals)."""
    fields = {
        h: {**f, "gamma": float(calibrate(f["gamma"], bool(f.get("has_relations"))))}
        for h, f in signals.fields.items()
    }
    return HypergraphSignals(
        rho=signals.rho, fields=fields, anchors=signals.anchors,
        vertex_confidence=signals.vertex_confidence,
        expanded_candidates=signals.expanded_candidates,
        pruned_candidates=signals.pruned_candidates,
    )


# -- exact-token-cost KV variant (ADR-0058 A/B) ----------------------------------


def build_exact_kv(hyperedges_parquet: str, nodes_parquet: str, out_path: str) -> str:
    """Rebuild the KV with tiktoken cl100k_base token costs (derived artifact, ADR-0056)."""
    import pyarrow.parquet as pq
    import tiktoken

    from latence_core.contracts import (
        Classification,
        Evidence,
        GraphHyperedge,
        GraphNode,
        HyperedgeRelation,
        Provenance,
    )
    from latence_core.hyperedge_store import build_store_bytes, store_entries

    enc = tiktoken.get_encoding("cl100k_base")
    prov = Provenance(
        source_uri="file:///rebuild", file_name="r", file_type="txt", file_size=1,
        document_id="rebuild", char_start=0, char_end=1, page_start=1, page_end=1,
    )
    cls = Classification(language="en", category="corpus", sensitivity="internal")
    hyperedges = []
    for r in pq.read_table(hyperedges_parquet).to_pylist():
        hyperedges.append(GraphHyperedge(
            record_id=f"hyperedge:{r['hyperedge_id']}",
            hyperedge_id=r["hyperedge_id"],
            entity_node_ids=json.loads(r["entity_node_ids"]),
            relations=[HyperedgeRelation(**x) for x in json.loads(r["relations"])],
            tau_text=r["tau_text"],
            token_cost=max(1, len(enc.encode(r["tau_text"]))),
            token_counter="tiktoken:cl100k_base",
            gamma=r["gamma"], support=r["support"], has_relations=r["has_relations"],
            evidence=Evidence(mention_ids=[], document_ids=[], confidence=r["gamma"]),
            provenance=prov, classification=cls,
        ))
    nodes = []
    for r in pq.read_table(nodes_parquet).to_pylist():
        nodes.append(GraphNode(
            record_id=f"node:{r['node_id']}", node_id=r["node_id"],
            entity_id=r["entity_id"], label=r["type"],
            canonical_name=r["canonical_name"], confidence=r["confidence"],
            evidence=Evidence(mention_ids=[], document_ids=[], confidence=r["confidence"]),
            provenance=prov, classification=cls,
        ))
    Path(out_path).write_bytes(build_store_bytes(store_entries(hyperedges, nodes)))
    return out_path


# -- recall@5 (paragraph level) --------------------------------------------------


def chunk_file_map(corpus_jsonl: Path) -> dict[str, str]:
    out = {}
    for line in corpus_jsonl.open():
        r = json.loads(line)
        if "content" in r and "index" in r and "provenance" in r:
            out[r["record_id"]] = r["provenance"]["file_name"]
    return out


def hyperedge_file_map(hyperedges_parquet: str, chunk_files: dict[str, str]) -> dict[str, str]:
    """hyperedge_id -> source paragraph file, via the parquet's primary chunk property
    (the KV's solver fields deliberately carry no provenance — the parquet is the
    durable/inspectable form, ADR-0056)."""
    import pyarrow.parquet as pq

    out = {}
    for r in pq.read_table(hyperedges_parquet).to_pylist():
        props = json.loads(r["properties"]) if r.get("properties") else {}
        f = chunk_files.get(str(props.get("primary_chunk_record_id", "")))
        if f:
            out[r["hyperedge_id"]] = f
    return out


def recall_at_5(
    questions: list[dict], arm: Arm, kv_path: str, he_files: dict[str, str],
    chunk_files: dict[str, str],
    surfaces: dict[str, list[str]] | None, tier: str, tier0: Tier0Index | None,
) -> dict:
    hits_base = hits_coe = total = 0
    with HyperedgeKV.open(kv_path) as kv:
        for q in questions:
            sup_files = {p["file"] for p in q["paragraphs"] if p["is_supporting"]}
            total += len(sup_files)
            ranked = arm.bm25.process(q["question"], arm.candidates)
            base_files: list[str] = []
            for c in ranked:
                f = chunk_files.get(c.id)
                if f and f not in base_files:
                    base_files.append(f)
                if len(base_files) == 5:
                    break
            hits_base += len(sup_files & set(base_files))
            sig = entry_signals(kv, q, surfaces, tier, tier0)
            coe_files: list[str] = []
            for hid in sorted(sig.rho, key=lambda h: -sig.rho[h]):
                f = he_files.get(hid)
                if f and f not in coe_files:
                    coe_files.append(f)
                if len(coe_files) == 5:
                    break
            hits_coe += len(sup_files & set(coe_files))
    return {
        "baseline_recall@5": round(hits_base / total, 4),
        "coe_recall@5": round(hits_coe / total, 4),
    }


def entry_signals(
    kv: HyperedgeKV, q: dict, surfaces: dict[str, list[str]] | None,
    tier: str, tier0: Tier0Index | None,
) -> HypergraphSignals:
    alias: list[str] = []
    ann: list[str] = []
    if tier in ("tier1", "tier12"):
        alias += question_ngrams(q["question"])
    if tier in ("tier2", "tier12") and surfaces:
        alias += surfaces.get(q["id"], [])
    if tier == "tier0" and tier0 is not None:
        ann = tier0.topk(q["question"])
    return hypergraph_signals(kv, alias_surfaces=alias, ann_node_ids=ann)
