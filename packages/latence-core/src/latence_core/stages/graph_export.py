"""Knowledge-Graph Export Provider — Parquet + TTL + GraphML (ADR-0017, S9).

The S9 KG Export materializes the canonical knowledge graph as **portable files on Storage,
never a live database** (ADR-0017): a graph-RAG / Neo4j / triple-store consumer loads the
files with no framework code in the loop. Given the run's
:class:`~latence_core.contracts.GraphRecord`s (NODE + EDGE scope, from the Graph Assembly
Stage) it writes, deterministically, three representations of the same graph:

* **Parquet** (``nodes.parquet`` + ``edges.parquet``) — loads as an Arrow / Delta table into
  a lakehouse or graph-RAG engine. Nested contract fields are JSON-encoded strings so the
  schema stays flat (ADR-0006), mirroring the S1 exporter. The node entity-type column is named
  ``type`` (W12) so it matches GraphML (``type``) and TTL (``lg:entityType``) — the same entity
  type under the same name in all three formats.
* **TTL** (``graph.ttl``) — well-formed RDF Turtle: each node is a subject with an
  ``rdfs:label``, each edge a triple ``:node_src :label :node_tgt`` plus a reified statement
  carrying its confidence and Evidence. Hand-emitted with strict IRI/literal escaping (no
  ``rdflib`` dependency — ADR-0016).
* **GraphML** (``graph.graphml``) — XML that loads in yEd / Gephi / networkx: ``<node>`` +
  ``<edge>`` elements with typed ``<data>`` keys (label, type, confidence, evidence count).
  Hand-emitted with XML-escaped text.

Every file is byte-stable (records are emitted in the order Graph Assembly produced them —
node-id then edge-id order — and every attribute is written in a fixed order), so a seeded
run is reproducible (Baseline bar). The Provider is a pass-through sink: it returns its input
records unchanged so the Runner can checkpoint them, exactly like the S1 exporter.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any
from xml.sax.saxutils import escape as xml_escape

import pyarrow as pa
import pyarrow.parquet as pq

from latence_core.contracts import GraphRecord, GraphScope
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import GraphEdge, GraphNode, Record
    from latence_core.storage import Storage

# The base IRI the TTL/GraphML node and predicate identifiers hang off. A non-dereferenceable
# ``urn:`` namespace keeps the export self-contained (no network) and still valid RDF.
_BASE_IRI = "urn:latence:graph:"

# XML 1.0 (§2.2) forbids almost every C0 control character even when entity-escaped: only
# tab (U+0009), LF (U+000A) and CR (U+000D) are legal in the Char production; the DEL/C1 block
# U+007F–U+0084 and U+0086–U+009F are also disallowed, as are the noncharacters U+FFFE/U+FFFF.
# Node canonical_name / edge label are verbatim spans sliced straight from messy source
# documents (the S9 threat model), so a stray form-feed, NUL or BEL would otherwise make the
# GraphML/TTL non-well-formed and unloadable in any standard tool. We drop these characters
# before serialization (both XML and Turtle) — a lossless-enough repair that keeps the visible
# label intact while guaranteeing the export parses.
_XML_ILLEGAL = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f￾￿]"
)


def _scrub(value: str) -> str:
    """Strip characters that XML 1.0 / strict Turtle forbid from a verbatim source span."""
    return _XML_ILLEGAL.sub("", value)


def _feature_float(value: Any) -> float | None:
    """A stamped graph-feature centrality as a float column cell (None if absent/malformed)."""
    return float(value) if isinstance(value, (int, float)) else None


def _feature_str(value: Any) -> str | None:
    """A stamped graph-feature community id as a string column cell (None if absent/malformed)."""
    return value if isinstance(value, str) else None


class KnowledgeGraphExport:
    """Export Provider writing the KG as Parquet + TTL + GraphML (ADR-0017).

    Config:
        basename: str  — output file stem prefix; the TTL/GraphML use it too. Default 'graph'.

    Only :class:`~latence_core.contracts.GraphRecord`s are materialized; any other record kind
    in the input is ignored (an Export wired downstream of Graph Assembly sees only graph
    records). An empty graph writes valid empty files (headers only), never an error.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic, Apache-2.0.

        The KG Export writes Parquet + TTL + GraphML files deterministically (ADR-0017).
        A pure-Python part of ``latence-core`` (Apache-2.0, verified per ADR-0012 — the
        package's own code, weights-and-code trivially the same), it ships no model weights
        (``model_id=None``), is deterministic, CPU-only, does not batch, and has no per-call
        cost, so a bake-off/conformance run reads honest, non-fabricated columns off it.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=32,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._basename = str(config.get("basename", "graph"))

    def export(self, records: Iterable[Record], storage: Storage, out_dir: str) -> list[str]:
        record_list = list(records)
        nodes: list[GraphNode] = [
            r.node
            for r in record_list
            if isinstance(r, GraphRecord) and r.scope is GraphScope.NODE and r.node is not None
        ]
        edges: list[GraphEdge] = [
            r.edge
            for r in record_list
            if isinstance(r, GraphRecord) and r.scope is GraphScope.EDGE and r.edge is not None
        ]
        base = out_dir.rstrip("/")
        stem = self._basename
        # Parquet is written through storage.open (a raw handle that does not create parent
        # dirs, unlike write_atomic), so ensure the export directory exists first.
        storage.makedirs(base)

        nodes_uri = f"{base}/{stem}-nodes.parquet"
        edges_uri = f"{base}/{stem}-edges.parquet"
        ttl_uri = f"{base}/{stem}.ttl"
        graphml_uri = f"{base}/{stem}.graphml"

        self._write_nodes_parquet(nodes, storage, nodes_uri)
        self._write_edges_parquet(edges, storage, edges_uri)
        storage.write_atomic(ttl_uri, self._render_ttl(nodes, edges).encode("utf-8"))
        storage.write_atomic(graphml_uri, self._render_graphml(nodes, edges).encode("utf-8"))

        return [nodes_uri, edges_uri, ttl_uri, graphml_uri]

    # -- Parquet ------------------------------------------------------------

    @staticmethod
    def _write_nodes_parquet(nodes: list[GraphNode], storage: Storage, uri: str) -> None:
        # ADR-0051: promote the graph-feature enrichment to FIRST-CLASS ``centrality`` (double) +
        # ``community`` (string) columns — the shape the Knapsack packer / a graph-aware ranker
        # reads directly (they also ride the ``properties`` JSON; this is the consumable view).
        # Added ONLY when a node carries them (the assembler stamped features), so the schema is
        # byte-identical to today when the enrichment is off (the genuinely-optional guarantee).
        has_features = any("centrality" in n.properties for n in nodes)
        flat = [
            {
                "node_id": n.node_id,
                "entity_id": n.entity_id,
                # W12: the entity-type column is named ``type`` (not ``label``) so the Parquet
                # export names it consistently with GraphML (``<data key="type">``) and TTL
                # (``lg:entityType``); a consumer reading ``type`` gets the entity type in all three
                # formats. (Export-schema note only — the GraphNode record contract is unchanged.)
                "type": n.label,
                "canonical_name": n.canonical_name,
                "kb_id": n.kb_id or "",
                "confidence": n.confidence,
                "member_mention_ids": json.dumps(n.member_mention_ids, sort_keys=True),
                "source_document_ids": json.dumps(n.source_document_ids, sort_keys=True),
                "properties": json.dumps(n.properties, sort_keys=True),
                "evidence": json.dumps(n.evidence.model_dump(mode="json"), sort_keys=True),
                **(
                    {
                        "centrality": _feature_float(n.properties.get("centrality")),
                        "community": _feature_str(n.properties.get("community")),
                    }
                    if has_features
                    else {}
                ),
            }
            for n in nodes
        ]
        fields: list[tuple[str, pa.DataType]] = [
            ("node_id", pa.string()),
            ("entity_id", pa.string()),
            ("type", pa.string()),
            ("canonical_name", pa.string()),
            ("kb_id", pa.string()),
            ("confidence", pa.float64()),
            ("member_mention_ids", pa.string()),
            ("source_document_ids", pa.string()),
            ("properties", pa.string()),
            ("evidence", pa.string()),
        ]
        if has_features:
            fields.append(("centrality", pa.float64()))
            fields.append(("community", pa.string()))
        table = pa.Table.from_pylist(flat, schema=pa.schema(fields))
        with storage.open(uri, "wb") as fh:
            pq.write_table(table, fh)  # type: ignore[no-untyped-call]

    @staticmethod
    def _write_edges_parquet(edges: list[GraphEdge], storage: Storage, uri: str) -> None:
        flat = [
            {
                "edge_id": e.edge_id,
                "relation_id": e.relation_id,
                "source_node_id": e.source_node_id,
                "target_node_id": e.target_node_id,
                "label": e.label,
                "confidence": e.confidence,
                "properties": json.dumps(e.properties, sort_keys=True),
                "evidence": json.dumps(e.evidence.model_dump(mode="json"), sort_keys=True),
            }
            for e in edges
        ]
        schema = pa.schema(
            [
                ("edge_id", pa.string()),
                ("relation_id", pa.string()),
                ("source_node_id", pa.string()),
                ("target_node_id", pa.string()),
                ("label", pa.string()),
                ("confidence", pa.float64()),
                ("properties", pa.string()),
                ("evidence", pa.string()),
            ]
        )
        table = pa.Table.from_pylist(flat, schema=schema)
        with storage.open(uri, "wb") as fh:
            pq.write_table(table, fh)  # type: ignore[no-untyped-call]

    # -- TTL (RDF Turtle) ---------------------------------------------------

    @classmethod
    def _render_ttl(cls, nodes: list[GraphNode], edges: list[GraphEdge]) -> str:
        lines: list[str] = [
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            f"@prefix lg: <{_BASE_IRI}> .",
            "",
        ]
        for n in nodes:
            subject = cls._iri("node", n.node_id)
            lines.append(f"{subject} a lg:Entity ;")
            lines.append(f'    rdfs:label {cls._literal(n.canonical_name)} ;')
            lines.append(f'    lg:entityType {cls._literal(n.label)} ;')
            if n.kb_id:
                lines.append(f'    lg:kbId {cls._literal(n.kb_id)} ;')
            lines.append(f'    lg:confidence "{cls._num(n.confidence)}"^^xsd:decimal .')
            lines.append("")
        for e in edges:
            subject = cls._iri("node", e.source_node_id)
            predicate = cls._iri("rel", e.label)
            obj = cls._iri("node", e.target_node_id)
            lines.append(f"{subject} {predicate} {obj} .")
            # A reified statement carries the edge's confidence + Evidence justification.
            stmt = cls._iri("edge", e.edge_id)
            lines.append(f"{stmt} a rdf:Statement ;")
            lines.append(f"    rdf:subject {subject} ;")
            lines.append(f"    rdf:predicate {predicate} ;")
            lines.append(f"    rdf:object {obj} ;")
            lines.append(f'    lg:confidence "{cls._num(e.confidence)}"^^xsd:decimal ;')
            lines.append(f"    lg:evidenceCount {len(e.evidence.mention_ids)} .")
            lines.append("")
        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _iri(kind: str, value: str) -> str:
        # Percent-escape everything an RFC 3987 IRI forbids: the fixed delimiter set below AND
        # every C0/C1 control character and '%' itself (a raw '%' not starting a valid escape
        # makes the IRI illegal, and a stray control char breaks a strict Turtle parser).
        safe = "".join(
            ch
            if (ch not in "<>\"{}|^`\\ %" and not (ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F))
            else f"%{ord(ch):02X}"
            for ch in value
        )
        return f"<{_BASE_IRI}{kind}:{safe}>"

    @staticmethod
    def _literal(value: str) -> str:
        # Scrub XML/Turtle-illegal control chars (verbatim source spans), then escape the
        # Turtle string-literal metacharacters. tab/LF/CR survive scrubbing and are escaped here.
        escaped = (
            _scrub(value)
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'

    @staticmethod
    def _num(value: float) -> str:
        return f"{value:.6f}"

    # -- GraphML (XML) ------------------------------------------------------

    @classmethod
    def _render_graphml(cls, nodes: list[GraphNode], edges: list[GraphEdge]) -> str:
        lines: list[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
            '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
            '  <key id="kb_id" for="node" attr.name="kb_id" attr.type="string"/>',
            '  <key id="confidence" for="node" attr.name="confidence" attr.type="double"/>',
            '  <key id="rel" for="edge" attr.name="label" attr.type="string"/>',
            '  <key id="econfidence" for="edge" attr.name="confidence" attr.type="double"/>',
            '  <key id="evidence" for="edge" attr.name="evidence_count" attr.type="int"/>',
            '  <graph id="G" edgedefault="directed">',
        ]
        for n in nodes:
            nid = xml_escape(_scrub(n.node_id), {'"': "&quot;"})
            lines.append(f'    <node id="{nid}">')
            lines.append(f'      <data key="label">{xml_escape(_scrub(n.canonical_name))}</data>')
            lines.append(f'      <data key="type">{xml_escape(_scrub(n.label))}</data>')
            if n.kb_id:
                lines.append(f'      <data key="kb_id">{xml_escape(_scrub(n.kb_id))}</data>')
            lines.append(f'      <data key="confidence">{cls._num(n.confidence)}</data>')
            lines.append("    </node>")
        for e in edges:
            eid = xml_escape(_scrub(e.edge_id), {'"': "&quot;"})
            src = xml_escape(_scrub(e.source_node_id), {'"': "&quot;"})
            tgt = xml_escape(_scrub(e.target_node_id), {'"': "&quot;"})
            lines.append(f'    <edge id="{eid}" source="{src}" target="{tgt}">')
            lines.append(f'      <data key="rel">{xml_escape(_scrub(e.label))}</data>')
            lines.append(f'      <data key="econfidence">{cls._num(e.confidence)}</data>')
            lines.append(
                f'      <data key="evidence">{len(e.evidence.mention_ids)}</data>'
            )
            lines.append("    </edge>")
        lines.append("  </graph>")
        lines.append("</graphml>")
        return "\n".join(lines) + "\n"
