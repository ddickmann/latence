"""What a run left behind, read back as the SAME typed records the run wrote (#188).

The run console already reads the two summary files a run writes (``manifest.json`` and
``quality-report.json``, :mod:`latence_console.runs`). A run leaves far more than a summary
on Storage, all of it JSONL, all of it written by the local Runner under the run directory
(ADR-0003/0010 — run state *is* files, there is no database):

* ``quarantine/<stage>.jsonl`` — the :class:`~latence_core.contracts.QuarantineRecord`\\ s
  Screening removed from the pipeline, retained for exactly this audit;
* ``screening/<stage>.jsonl`` — every :class:`~latence_core.contracts.ScreeningFinding`,
  including the ``FLAG`` verdicts whose risk markers then rode into the corpus;
* ``checkpoints/<stage>.jsonl`` — each Stage's own output records: the parsed
  :class:`~latence_core.contracts.DocumentRecord`\\ s, the
  :class:`~latence_core.contracts.ChunkRecord`\\ s, the
  :class:`~latence_core.contracts.EntityMention`\\ s a KG edge's Evidence cites, and the
  :class:`~latence_core.contracts.GraphRecord`\\ s holding the graph's nodes and edges.

:class:`RunArtifacts` is the one door onto all of it. Three properties are load-bearing:

* **The contracts do the parsing.** Every line is validated by the real Pydantic model the
  Stage serialised, so a hand-edited, truncated or foreign line is *rejected* here exactly
  as the pipeline would reject it — the reader cannot invent a record shape the framework
  would not accept, and cannot silently pass one through.
* **Read-only.** Only ``exists`` / ``ls`` / ``read_text`` are reached, like the rest of the
  console. Nothing is written, indexed or cached across requests: the app builds a fresh
  :class:`RunArtifacts` per request, so its memoisation lives and dies inside one answer.
* **Latest Stage wins.** A chunk is rewritten by later Stages (Redaction replaces a
  :class:`~latence_core.contracts.ChunkRecord` with the same ``record_id``, now carrying
  ``masked_content`` + ``pii_spans``). Checkpoints are therefore read in *pipeline
  dependency order* — taken from the manifest the run itself wrote — so the record the
  console shows is the most-downstream one, the one the corpus was exported from.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TypeVar

from latence_core.contracts import (
    ChunkRecord,
    DocumentRecord,
    EntityMention,
    GraphEdge,
    GraphNode,
    GraphRecord,
    GraphScope,
    QuarantineRecord,
    ScreeningFinding,
    ScreeningVerdict,
)
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from latence_core.storage import Storage

CHECKPOINTS_DIR = "checkpoints"
QUARANTINE_DIR = "quarantine"
SCREENING_DIR = "screening"

#: One of the four record carriers a Stage checkpoint line can hold.
Carrier = GraphRecord | EntityMention | ChunkRecord | DocumentRecord

# Order matters only as a tie-break: ``extra="forbid"`` on every contract makes the carriers
# mutually exclusive, so at most one of them accepts a given line. A line no carrier accepts
# (a relation mention, a profiling record, a torn write) is skipped — the console stays
# legible over whatever is on disk (the posture :meth:`RunIndex.report` takes for an
# unreadable report).
_CARRIERS: tuple[type[Carrier], ...] = (GraphRecord, EntityMention, ChunkRecord, DocumentRecord)

# Cheap key hints that pick the likely carrier first, so the common line costs one validation
# instead of four. Purely a shortcut: the chosen model still validates, and every other
# carrier is still tried if it refuses, so the hint can never let a bad line through.
_HINTS: tuple[tuple[str, type[Carrier]], ...] = (
    ("scope", GraphRecord),
    ("chunk_record_id", EntityMention),
    ("document_record_id", ChunkRecord),
    ("disposition", DocumentRecord),
)


class RunArtifacts:
    """Read-only, typed view over one run's JSONL sidecars and Stage checkpoints.

    Accepts the :class:`~latence_core.storage.Storage` it reads through and the run
    directory it reads under (it builds neither), so the same object serves a local folder,
    an S3 prefix, or an in-memory fixture — the seam :class:`~latence_console.runs.RunIndex`
    already uses.
    """

    def __init__(self, storage: Storage, run_dir: str) -> None:
        self._storage = storage
        self._run_dir = run_dir.rstrip("/")
        self._scanned: _Scan | None = None
        self._quarantine: list[QuarantineRecord] | None = None
        self._findings: list[ScreeningFinding] | None = None

    # -- the public surface ------------------------------------------------

    def quarantined(self) -> list[QuarantineRecord]:
        """Every record Screening held back, in the order the run recorded them."""
        if self._quarantine is None:
            self._quarantine = [
                record
                for _stage, payload in self._sidecar_payloads(QUARANTINE_DIR)
                for record in _validated(QuarantineRecord, payload)
            ]
        return self._quarantine

    def findings(self) -> list[ScreeningFinding]:
        """Every Screening finding the run recorded (``PASS`` verdicts included)."""
        if self._findings is None:
            self._findings = [
                finding
                for _stage, payload in self._sidecar_payloads(SCREENING_DIR)
                for finding in _validated(ScreeningFinding, payload)
            ]
        return self._findings

    def flag_checkpoints(self) -> dict[tuple[str, str], str]:
        """``(document_id, category) -> checkpoint`` for every FLAG finding.

        Lets a risk marker that survived onto a chunk be attributed to the checkpoint that
        raised it, without the chunk having to carry that (it does not).
        """
        return {
            (f.document_id, f.category): f.checkpoint.value
            for f in self.findings()
            if f.verdict is ScreeningVerdict.FLAG
        }

    def chunks(self) -> list[ChunkRecord]:
        """Every chunk, most-downstream version, ordered by source document then index."""
        chunks = list(self._scan().chunks.values())
        return sorted(chunks, key=lambda c: (c.provenance.file_name, c.index, c.record_id))

    def chunk(self, record_id: str) -> ChunkRecord | None:
        """One chunk by ``record_id``, or ``None``.

        Looked up in the scanned map — the id is never concatenated into a Storage URI, so a
        crafted id cannot address a file, only fail to match a record.
        """
        return self._scan().chunks.get(record_id)

    def document(self, document_id: str) -> DocumentRecord | None:
        """The parsed document a chunk's Provenance names, or ``None`` if not checkpointed."""
        return self._scan().documents.get(document_id)

    def mention(self, record_id: str) -> EntityMention | None:
        """One entity mention by ``record_id`` — what a KG edge's Evidence cites."""
        return self._scan().mentions.get(record_id)

    def graph_nodes(self) -> list[GraphNode]:
        """Every KG node, ordered by canonical name."""
        return sorted(
            self._scan().nodes.values(), key=lambda n: (n.canonical_name.casefold(), n.node_id)
        )

    def graph_edges(self) -> list[GraphEdge]:
        """Every KG edge, ordered by label then id (deterministic, like the export)."""
        return sorted(self._scan().edges.values(), key=lambda e: (e.label, e.edge_id))

    def node(self, node_id: str) -> GraphNode | None:
        return self._scan().nodes.get(node_id)

    # -- reading -----------------------------------------------------------

    def _uri(self, *parts: str) -> str:
        return "/".join((self._run_dir, *parts))

    def _read_jsonl(self, uri: str) -> Iterator[dict[str, Any]]:
        """Every well-formed JSON object in a JSONL file (blank/torn lines skipped)."""
        try:
            text = self._storage.read_text(uri)
        except (UnicodeDecodeError, OSError, ValueError):
            return
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict):
                yield payload

    def _jsonl_files(self, directory: str) -> list[tuple[str, str]]:
        """``(stage_name, uri)`` for every ``<stage>.jsonl`` in one run subdirectory."""
        root = self._uri(directory)
        if not self._storage.exists(root):
            return []
        found: list[tuple[str, str]] = []
        for entry in self._storage.ls(root):
            name = entry.rstrip("/").rsplit("/", 1)[-1]
            if name.endswith(".jsonl"):
                found.append((name[: -len(".jsonl")], self._uri(directory, name)))
        return sorted(found)

    def _sidecar_payloads(self, directory: str) -> Iterator[tuple[str, dict[str, Any]]]:
        for stage, uri in self._jsonl_files(directory):
            for payload in self._read_jsonl(uri):
                yield stage, payload

    def _stage_rank(self) -> dict[str, int]:
        """Pipeline dependency order of the run's Stages, from the manifest it wrote.

        A dependency-respecting order is what makes "latest Stage wins" correct: Redaction
        rewrites a chunk Chunk produced, so its checkpoint must be read *after*. Kahn's
        algorithm over the manifest's ``depends_on`` edges; a manifest that is missing,
        unreadable, or cyclic degrades to the declared order (and, for a Stage the manifest
        never mentions, to alphabetical after the known ones).
        """
        try:
            payload = json.loads(self._storage.read_text(self._uri("manifest.json")))
        except (ValueError, UnicodeDecodeError, OSError):
            return {}
        pipeline = payload.get("pipeline") if isinstance(payload, dict) else None
        stages = pipeline.get("stages") if isinstance(pipeline, dict) else None
        if not isinstance(stages, list):
            return {}
        declared: list[str] = []
        deps: dict[str, set[str]] = {}
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            name = stage.get("name")
            if not isinstance(name, str) or not name:
                continue
            declared.append(name)
            raw = stage.get("depends_on")
            deps[name] = {d for d in raw if isinstance(d, str)} if isinstance(raw, list) else set()
        ordered: list[str] = []
        remaining = dict(deps)
        while remaining:
            ready = [n for n in declared if n in remaining and not (remaining[n] - set(ordered))]
            if not ready:  # a cycle, or a dependency on a Stage that is not declared
                ordered.extend(n for n in declared if n in remaining)
                break
            for name in ready:
                ordered.append(name)
                del remaining[name]
        return {name: rank for rank, name in enumerate(ordered)}

    def _scan(self) -> _Scan:
        """Read every Stage checkpoint once, in dependency order, into typed maps."""
        if self._scanned is not None:
            return self._scanned
        rank = self._stage_rank()
        files = self._jsonl_files(CHECKPOINTS_DIR)
        files.sort(key=lambda item: (rank.get(item[0], len(rank)), item[0]))
        scan = _Scan()
        for _stage, uri in files:
            for payload in self._read_jsonl(uri):
                scan.absorb(_parse_record(payload))
        self._scanned = scan
        return scan


class _Scan:
    """The typed maps one pass over the checkpoints produces. Later Stages overwrite."""

    def __init__(self) -> None:
        self.chunks: dict[str, ChunkRecord] = {}
        self.documents: dict[str, DocumentRecord] = {}
        self.mentions: dict[str, EntityMention] = {}
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}

    def absorb(self, record: Carrier | None) -> None:
        if isinstance(record, ChunkRecord):
            self.chunks[record.record_id] = record
        elif isinstance(record, DocumentRecord):
            self.documents[record.provenance.document_id] = record
        elif isinstance(record, EntityMention):
            self.mentions[record.record_id] = record
        elif isinstance(record, GraphRecord):
            if record.scope is GraphScope.NODE and record.node is not None:
                self.nodes[record.node.node_id] = record.node
            elif record.edge is not None:
                self.edges[record.edge.edge_id] = record.edge


_M = TypeVar("_M", bound=BaseModel)


def _validated(model: type[_M], payload: dict[str, Any]) -> list[_M]:
    """``[record]`` if the real contract accepts ``payload``, else ``[]``."""
    try:
        return [model.model_validate(payload)]
    except ValidationError:
        return []


def _parse_record(payload: dict[str, Any]) -> Carrier | None:
    """The typed record a checkpoint line holds, or ``None`` if no carrier accepts it."""
    hinted = next((model for key, model in _HINTS if key in payload), None)
    candidates = (
        (hinted, *(c for c in _CARRIERS if c is not hinted)) if hinted else _CARRIERS
    )
    for model in candidates:
        try:
            return model.model_validate(payload)
        except ValidationError:
            continue
    return None
