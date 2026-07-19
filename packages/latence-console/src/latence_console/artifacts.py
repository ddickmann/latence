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
* **Read-only.** Only ``exists`` / ``ls`` / ``read_text`` / ``open(uri, "rb")`` are reached, like
  the rest of the console. Nothing is written, indexed or cached across requests: the app builds a
  fresh :class:`RunArtifacts` per request, so its memoisation lives and dies inside one answer.
* **Latest Stage wins.** A chunk is rewritten by later Stages (Redaction replaces a
  :class:`~latence_core.contracts.ChunkRecord` with the same ``record_id``, now carrying
  ``masked_content`` + ``pii_spans``). Checkpoints are therefore read in *pipeline
  dependency order* — taken from the manifest the run itself wrote — so the record the
  console shows is the most-downstream one, the one the corpus was exported from.
* **Bounded in the run's size** (audit R4). A real run's checkpoints are hundreds of megabytes;
  the console must not spend them to answer one request. Three rules keep it bounded:

  1. every file is *streamed* line by line — never decoded into one ``str`` (:meth:`_read_jsonl`);
  2. the four single-record lookups (:meth:`chunk`, :meth:`document`, :meth:`mention`,
     :meth:`node`) stream the checkpoints **most-downstream first** and retain only the match,
     so the chunk inspector costs one file pass and O(1) memory instead of the whole run. They
     read a list view's scan only where it is *authoritative*: a capped scan that dropped a
     record cannot be allowed to answer "the run never checkpointed it" (audit R5), so a miss in
     a truncated map falls through to the streaming lookup. :meth:`mentions_by_id` settles a
     whole page's Evidence citations that way in one pass rather than one pass each;
  3. the *list* views' scan retains at most :data:`MAX_SCAN_RECORDS` records of each kind and
     reports :meth:`truncated`. To make that truncation redaction-safe, the scan walks the
     checkpoints in **reverse** dependency order with downstream-wins semantics — so every
     record it keeps is already the most-downstream version, and a truncated scan can never
     surface a pre-Redaction chunk (ADR-0042/0044). With nothing truncated the result is
     identical to reading forwards and letting the last write win.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

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
    from collections.abc import Callable, Iterable, Iterator

    from latence_core.storage import Storage

CHECKPOINTS_DIR = "checkpoints"
QUARANTINE_DIR = "quarantine"
SCREENING_DIR = "screening"

#: How many records of each kind a *list* view's scan will hold at once (audit R4). A run is
#: unbounded; a browse page is not. Past this the scan stops absorbing that kind and says so
#: (:meth:`RunArtifacts.truncated`) rather than growing until the console is killed.
MAX_SCAN_RECORDS = 2000

#: How many bytes of a single JSONL line the reader will accumulate before abandoning it. A
#: checkpoint line is one record; a line orders of magnitude larger than any record is a torn or
#: foreign file, and reading it into memory is the very failure this module bounds.
MAX_LINE_BYTES = 16 * 1024 * 1024

#: One of the four record carriers a Stage checkpoint line can hold.
Carrier = GraphRecord | EntityMention | ChunkRecord | DocumentRecord

#: A record a capped map holds.
_R = TypeVar("_R")
#: The record a *read-only* view of a capped map yields. Covariant: a reader that only ever
#: takes records out of the map is safe to hand a map of a narrower record type.
_R_co = TypeVar("_R_co", covariant=True)
#: The carrier subtype :meth:`RunArtifacts._lookup` is asked for.
_C = TypeVar("_C", bound=Carrier)


class _KeptView(Protocol[_R_co]):
    """The read-only face of :class:`_Kept` — exactly what :meth:`RunArtifacts._from_scan` needs.

    :class:`_Kept` itself is **invariant** (it has :meth:`_Kept.put`, which consumes an ``_R``),
    so a selector returning ``_Kept[ChunkRecord]`` cannot satisfy a parameter written in terms of
    the optional ``ChunkRecord | None`` that ``_from_scan``'s own return type introduces. This
    view drops ``put`` and is therefore genuinely covariant, which both type-checks and states
    the real contract: ``_from_scan`` only ever *reads* the scan's maps and never writes one.
    """

    @property
    def truncated(self) -> bool: ...

    def get(self, key: str) -> _R_co | None: ...

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
        self._checkpoint_files: list[tuple[str, str]] | None = None
        self._quarantine: list[QuarantineRecord] | None = None
        self._findings: list[ScreeningFinding] | None = None

    def truncated(self) -> bool:
        """Did a list view hit :data:`MAX_SCAN_RECORDS` and leave records out?

        ``False`` until a list view has actually scanned — the single-record views never scan,
        and never claim anything about the size of the run.
        """
        return self._scanned is not None and self._scanned.truncated

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

        A **targeted, streaming** lookup (audit R4) — opening one chunk must not cost the whole
        run. The id is only ever matched against records already read, never concatenated into a
        Storage URI, so a crafted id cannot address a file, only fail to match a record.
        """
        answered, record = self._from_scan(lambda s: s.chunks, record_id)
        if answered:
            return record
        return self._lookup(ChunkRecord, lambda c: c.record_id, record_id)

    def document(self, document_id: str) -> DocumentRecord | None:
        """The parsed document a chunk's Provenance names, or ``None`` if not checkpointed."""
        answered, record = self._from_scan(lambda s: s.documents, document_id)
        if answered:
            return record
        return self._lookup(DocumentRecord, lambda d: d.provenance.document_id, document_id)

    def mention(self, record_id: str) -> EntityMention | None:
        """One entity mention by ``record_id`` — what a KG edge's Evidence cites."""
        answered, record = self._from_scan(lambda s: s.mentions, record_id)
        if answered:
            return record
        return self._lookup(EntityMention, lambda m: m.record_id, record_id)

    def mentions_by_id(self, record_ids: Iterable[str]) -> dict[str, EntityMention]:
        """Resolve many cited mentions at once — ``{record_id: mention}``, misses left out.

        The KG views resolve one mention per Evidence entry, and they always scan first (they
        need the node/edge maps). Whatever a truncated scan could not answer is therefore
        collected here and settled in **one** streaming pass over the checkpoints instead of one
        pass per mention (audit R5) — the answer of :meth:`mention` called per id, at the cost of
        a single read. Peak memory stays O(number of ids asked for), which is what the caller is
        already holding a rendered row for.
        """
        wanted = list(dict.fromkeys(record_ids))
        resolved: dict[str, EntityMention] = {}
        missing: list[str] = []
        for record_id in wanted:
            answered, record = self._from_scan(lambda s: s.mentions, record_id)
            if record is not None:
                resolved[record_id] = record
            elif not answered:
                missing.append(record_id)
        if missing:
            resolved.update(self._lookup_many(EntityMention, lambda m: m.record_id, missing))
        return resolved

    def graph_nodes(self) -> list[GraphNode]:
        """Every KG node, ordered by canonical name."""
        return sorted(
            self._scan().nodes.values(), key=lambda n: (n.canonical_name.casefold(), n.node_id)
        )

    def graph_edges(self) -> list[GraphEdge]:
        """Every KG edge, ordered by label then id (deterministic, like the export)."""
        return sorted(self._scan().edges.values(), key=lambda e: (e.label, e.edge_id))

    def node(self, node_id: str) -> GraphNode | None:
        answered, record = self._from_scan(lambda s: s.nodes, node_id)
        if answered:
            return record
        return self._lookup_node(node_id)

    def _from_scan(
        self, select: Callable[[_Scan], _KeptView[_R]], key: str
    ) -> tuple[bool, _R | None]:
        """``(answered, record)`` — the scan replies only where it is *authoritative*.

        A list view's scan is capped (:data:`MAX_SCAN_RECORDS`), so a miss in its map has two
        very different meanings: the run never checkpointed that record, or the cap dropped it.
        Only the first is an answer. Reporting the second as ``None`` made the KG views label a
        checkpointed Evidence mention "not in this run's checkpoints" — a *false provenance
        claim* (ADR-0021/0031, audit R5). Where the map is not authoritative the caller falls
        back to the targeted streaming lookup, which reads what is actually on disk.
        """
        scan = self._scanned
        if scan is None:
            return False, None
        kept = select(scan)
        record = kept.get(key)
        if record is not None:
            return True, record
        return (not kept.truncated) and not scan.stopped_early, None

    # -- reading -----------------------------------------------------------

    def _uri(self, *parts: str) -> str:
        return "/".join((self._run_dir, *parts))

    def _read_jsonl(self, uri: str, *, contains: str | None = None) -> Iterator[dict[str, Any]]:
        """Every well-formed JSON object in a JSONL file (blank/torn lines skipped).

        **Streamed** (audit R4): the file is read line by line through ``Storage.open`` and each
        line is decoded on its own, so peak memory is one record — not the file. A checkpoint is
        routinely hundreds of megabytes, and ``read_text`` would have materialised all of it (plus
        a second full copy from ``splitlines``) before the first record was parsed.

        ``contains`` is a cheap pre-filter for targeted lookups: a line whose raw bytes do not
        contain that substring cannot encode a record carrying it, so it is skipped without being
        decoded or validated. It is a superset filter over the same lines — never a different
        answer, only a faster one — and callers only pass a value that survives JSON escaping
        unchanged (:func:`_literal_in_json`).
        """
        needle = contains.encode("utf-8") if contains is not None else None
        try:
            handle = self._storage.open(uri, "rb")
        except (OSError, ValueError):
            return
        try:
            with handle:
                for raw in handle:
                    if len(raw) > MAX_LINE_BYTES:
                        continue
                    if needle is not None and needle not in raw:
                        continue
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped.decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(payload, dict):
                        yield payload
        except (OSError, ValueError):
            return

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

    def _checkpoints(self) -> list[tuple[str, str]]:
        """``(stage, uri)`` for every checkpoint, in pipeline dependency order."""
        if self._checkpoint_files is None:
            rank = self._stage_rank()
            files = self._jsonl_files(CHECKPOINTS_DIR)
            files.sort(key=lambda item: (rank.get(item[0], len(rank)), item[0]))
            self._checkpoint_files = files
        return self._checkpoint_files

    def _scan(self) -> _Scan:
        """Read the Stage checkpoints once into typed maps, bounded in the run's size.

        Walked **most-downstream first** with downstream-wins semantics, which is what makes the
        :data:`MAX_SCAN_RECORDS` cap safe: whatever the scan holds when it stops is already the
        version the corpus was exported from, so a truncated list can never show a chunk its
        Redaction Stage had already rewritten (ADR-0042/0044). With no truncation this is the same
        map reading forwards and overwriting would have produced.
        """
        if self._scanned is not None:
            return self._scanned
        scan = _Scan()
        for index, (_stage, uri) in enumerate(reversed(self._checkpoints())):
            if scan.full:
                scan.stopped_early = True
                break
            for payload in self._read_jsonl(uri):
                scan.absorb(_parse_record(payload), index)
        self._scanned = scan
        return scan

    def _lookup(
        self,
        wanted: type[_C],
        key_of: Callable[[_C], str],
        key: str,
    ) -> _C | None:
        """One record by key, streamed most-downstream first — never a whole-run scan.

        Reuses the list views' scan when this request already built one (the graph view resolves
        every edge's Evidence through :meth:`mention`, which would otherwise be one file pass per
        mention); otherwise it streams, keeping only the match. The first checkpoint file that
        supplies the key wins — that is the most-downstream Stage that wrote it — and within one
        file a later line still overrides an earlier one, exactly as reading forwards did.
        """
        contains = key if _literal_in_json(key) else None
        for _stage, uri in reversed(self._checkpoints()):
            found: _C | None = None
            for payload in self._read_jsonl(uri, contains=contains):
                record = _parse_record(payload)
                if isinstance(record, wanted) and key_of(record) == key:
                    found = record
            if found is not None:
                return found
        return None

    def _lookup_many(
        self,
        wanted: type[_C],
        key_of: Callable[[_C], str],
        keys: list[str],
    ) -> dict[str, _C]:
        """Many records by key in ONE streaming pass, most-downstream first.

        The batched sibling of :meth:`_lookup`, and it keeps that method's semantics per key:
        the first checkpoint file supplying a key wins (the most-downstream Stage that wrote it),
        and within a file a later line still overrides an earlier one. Only the requested keys
        are retained, so peak memory is O(``len(keys)``) — never the run. No pre-filter is used:
        one pass answering every key beats one filtered pass per key.
        """
        found: dict[str, _C] = {}
        outstanding = set(keys)
        for _stage, uri in reversed(self._checkpoints()):
            if not outstanding:
                break
            from_file: dict[str, _C] = {}
            for payload in self._read_jsonl(uri):
                record = _parse_record(payload)
                if isinstance(record, wanted):
                    key = key_of(record)
                    if key in outstanding:
                        from_file[key] = record
            found.update(from_file)
            outstanding -= from_file.keys()
        return found

    def _lookup_node(self, node_id: str) -> GraphNode | None:
        """A node lives *inside* a :class:`GraphRecord` carrier, so it needs its own unwrap."""
        contains = node_id if _literal_in_json(node_id) else None
        for _stage, uri in reversed(self._checkpoints()):
            found: GraphNode | None = None
            for payload in self._read_jsonl(uri, contains=contains):
                record = _parse_record(payload)
                if (
                    isinstance(record, GraphRecord)
                    and record.scope is GraphScope.NODE
                    and record.node is not None
                    and record.node.node_id == node_id
                ):
                    found = record.node
            if found is not None:
                return found
        return None


class _Kept(Generic[_R]):
    """One capped map of records, remembering which checkpoint file supplied each key.

    Records arrive most-downstream first, so an existing key was set by this file or a more
    downstream one. Only the *same* file may override it (a later line of a file still wins, as
    it did when the scan read forwards); a further-upstream Stage's version is dropped.
    """

    def __init__(self) -> None:
        self._items: dict[str, _R] = {}
        self._origin: dict[str, int] = {}
        self.truncated = False

    def get(self, key: str) -> _R | None:
        return self._items.get(key)

    def values(self) -> Iterable[_R]:
        return self._items.values()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def full(self) -> bool:
        return len(self._items) >= MAX_SCAN_RECORDS

    def put(self, key: str, value: _R, file_index: int) -> None:
        if key in self._items:
            if self._origin[key] == file_index:
                self._items[key] = value
            return
        if self.full:
            self.truncated = True
            return
        self._items[key] = value
        self._origin[key] = file_index


class _Scan:
    """The typed maps one pass over the checkpoints produces.

    Filled **most-downstream first** (see :meth:`RunArtifacts._scan`), each map capped at
    :data:`MAX_SCAN_RECORDS`, with :attr:`truncated` recording whether anything was left out.
    """

    def __init__(self) -> None:
        self.chunks: _Kept[ChunkRecord] = _Kept()
        self.documents: _Kept[DocumentRecord] = _Kept()
        self.mentions: _Kept[EntityMention] = _Kept()
        self.nodes: _Kept[GraphNode] = _Kept()
        self.edges: _Kept[GraphEdge] = _Kept()
        #: Did the scan stop before reading every checkpoint (every map already full)? Then a
        #: miss in *any* map may be a record in a file the scan never opened, not an absence.
        self.stopped_early = False

    @property
    def _maps(self) -> tuple[_Kept[Any], ...]:
        return (self.chunks, self.documents, self.mentions, self.nodes, self.edges)

    @property
    def full(self) -> bool:
        """Has every map hit the cap? (Then there is nothing left to read for.)"""
        return all(m.full for m in self._maps)

    @property
    def truncated(self) -> bool:
        """Did any map have to drop a record? (Stated on the page, never hidden.)"""
        return any(m.truncated for m in self._maps)

    def absorb(self, record: Carrier | None, file_index: int = 0) -> None:
        if isinstance(record, ChunkRecord):
            self.chunks.put(record.record_id, record, file_index)
        elif isinstance(record, DocumentRecord):
            self.documents.put(record.provenance.document_id, record, file_index)
        elif isinstance(record, EntityMention):
            self.mentions.put(record.record_id, record, file_index)
        elif isinstance(record, GraphRecord):
            if record.scope is GraphScope.NODE and record.node is not None:
                self.nodes.put(record.node.node_id, record.node, file_index)
            elif record.edge is not None:
                self.edges.put(record.edge.edge_id, record.edge, file_index)


def _literal_in_json(value: str) -> bool:
    """Does ``value`` appear verbatim inside a JSON string, unescaped?

    The gate on the :meth:`RunArtifacts._read_jsonl` pre-filter. A record id
    (``sha256:…#chunk-0``, a node id) survives JSON encoding unchanged, so a line that does not
    contain those bytes cannot carry that record. An id containing a quote, a backslash or a
    control character would be *escaped* in the file, so the substring test would wrongly skip
    the line — such an id simply gets no pre-filter, and the full scan of the file decides.
    """
    return bool(value) and json.dumps(value)[1:-1] == value


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
