"""Capability protocols and the entry-point Provider registry (ADR-0004).

A Capability is a narrow, model-agnostic interface a Stage depends on. A Provider
is a concrete implementation, registered as a plugin under the ``latence.providers``
entry-point group and discovered dynamically via ``importlib.metadata`` — the
framework never names a model, only Capabilities.

For S1 the Capabilities are the three spine Stages (Source, Parser, Export). Each
is a ``typing.Protocol`` so any object with the right shape satisfies it — no
base class or import of core required for a third-party Provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import (
        ChunkRecord,
        DocumentRecord,
        EntityMention,
        ParserInput,
        QuarantineRecord,
        Record,
        RelationMention,
        ScreeningFinding,
    )
    from latence_core.storage import Storage

ENTRY_POINT_GROUP = "latence.providers"

# The carrier type a Screening checkpoint operates on: ParserInput at intake,
# ChunkRecord at content. ``ScreeningOutcome`` is generic over it.
_Screened = TypeVar("_Screened")


@dataclass(frozen=True)
class ScreeningOutcome(Generic[_Screened]):
    """The result of a Screening checkpoint: what passed, what was Quarantined.

    ``passed`` are the records that proceed downstream — for Content Screening
    these may carry newly-attached risk markers. ``quarantined`` are the
    :class:`~latence_core.contracts.QuarantineRecord`s removed from the pipeline
    (retained for audit, never exported to the corpus). ``findings`` is the full
    per-record decision log Screening emits into the Quality Report — every
    Quarantine reason and every flag, so the disposition is auditable (S3 AC).
    """

    passed: list[_Screened] = field(default_factory=list)
    quarantined: list[QuarantineRecord] = field(default_factory=list)
    findings: list[ScreeningFinding] = field(default_factory=list)


@runtime_checkable
class Source(Protocol):
    """Lists and fetches documents from where they live, stamping Provenance."""

    def produce(self, storage: Storage) -> Iterator[ParserInput]:
        """Yield one raw document per source document, with initial Provenance.

        The yielded :class:`~latence_core.contracts.ParserInput` carries the
        undecoded source ``content`` (bytes) — decoding and the turn into
        markdown belong to the Parser (ADR-0019), not the Source.
        """
        ...


@runtime_checkable
class Parser(Protocol):
    """Turns raw source documents into parsed markdown records."""

    def parse(self, inputs: Iterable[ParserInput]) -> Iterator[DocumentRecord]:
        """Consume raw :class:`ParserInput`s and yield parsed markdown records."""
        ...


@runtime_checkable
class Chunker(Protocol):
    """Splits parsed documents into retrieval-sized ChunkRecords (CONTEXT ``Chunk``).

    Preserves offsets, page alignment, Provenance and Classification on every
    emitted chunk. A ``PARSE_ERROR`` DocumentRecord (no usable content) yields no
    chunks.
    """

    def chunk(self, documents: Iterable[DocumentRecord]) -> Iterator[ChunkRecord]:
        """Consume parsed :class:`DocumentRecord`s and yield :class:`ChunkRecord`s."""
        ...


@runtime_checkable
class IntakeScreener(Protocol):
    """Screens raw documents BEFORE Parse (CONTEXT ``Screening`` — intake checkpoint).

    Catches malware, zip bombs, file-type spoofing, oversized/corrupt files. A
    dangerous document is Quarantined (removed from every downstream Stage); a
    clean one passes through untouched.
    """

    def screen_intake(self, inputs: Iterable[ParserInput]) -> ScreeningOutcome[ParserInput]:
        """Partition raw inputs into passed vs quarantined, with findings."""
        ...


@runtime_checkable
class ContentScreener(Protocol):
    """Screens chunked content AFTER Chunk (CONTEXT ``Screening`` — content checkpoint).

    Catches prompt injection, harmful content, sensitivity escalation. A flagged
    chunk keeps flowing but carries a :class:`~latence_core.contracts.RiskMarker`
    that survives into the corpus; a chunk deemed dangerous is Quarantined.
    """

    def screen_content(self, chunks: Iterable[ChunkRecord]) -> ScreeningOutcome[ChunkRecord]:
        """Return chunks (some marked) plus any quarantined chunks and findings."""
        ...


@runtime_checkable
class EntityExtractor(Protocol):
    """Finds typed entity mentions in chunked text (CONTEXT ``Entity Extraction``).

    Zero-shot NER: the label set is configured per Pipeline (Stage ``config``), not
    hardcoded — the Capability names no types. Each emitted
    :class:`~latence_core.contracts.EntityMention` carries its label, confidence,
    and the char span it occupies **in the parent document's assembled markdown**,
    so a mention resolves back through the parent's page map to its source page(s).
    An empty or near-empty chunk yields no mentions (never an error).
    """

    def extract(self, chunks: Iterable[ChunkRecord]) -> Iterator[EntityMention]:
        """Consume :class:`ChunkRecord`s and yield typed :class:`EntityMention`s."""
        ...


@runtime_checkable
class RelationExtractor(Protocol):
    """Finds typed relations between entity mentions (CONTEXT ``Relation Extraction``).

    Relation Extraction relates *mentions within a document* (ADR-0022): given the
    document's :class:`~latence_core.contracts.EntityMention`s plus the chunk text
    that supplies the surrounding context, it yields directed, typed
    :class:`~latence_core.contracts.RelationMention`s (head → tail). The relation
    label set is configured per Pipeline (Stage ``config``), not hardcoded — the
    Capability names no relation types (zero-shot / prompt-driven), mirroring
    Entity Extraction. Each emitted relation carries the head/tail mention refs, its
    label, confidence, and a Provenance span covering both endpoints, so it resolves
    back through the parent's page map to its source page(s). A document with fewer
    than two mentions, or no relatable pair, yields no relations (never an error).
    """

    def relate(
        self, chunks: Iterable[ChunkRecord], mentions: Iterable[EntityMention]
    ) -> Iterator[RelationMention]:
        """Consume a run's chunks + mentions and yield typed :class:`RelationMention`s.

        Both carriers are supplied: ``mentions`` are what a relation connects, and
        ``chunks`` carry the text (and the parent page map) a text-based Provider
        needs for context and offset→page resolution. The Provider groups both by
        ``document_record_id`` internally — a relation only ever links two mentions
        of the *same* document.
        """
        ...


@dataclass(frozen=True)
class FusedExtraction:
    """The joint output of a :class:`FusedEntityRelationExtractor` — one pass, both Stages.

    A Fused Provider (ADR-0013: gliner-relex, the custom span-predictor) fulfils
    Entity Extraction AND Relation Extraction in a single pass over the chunks, so
    it returns both carriers together: the ``mentions`` it found and the
    ``relations`` between them. The Runner threads both downstream from the one
    fused Stage — S4's mentions and S5's relations — without invoking a separate
    Entity-Extraction Stage (no double-extraction).
    """

    mentions: list[EntityMention] = field(default_factory=list)
    relations: list[RelationMention] = field(default_factory=list)


@runtime_checkable
class FusedEntityRelationExtractor(Protocol):
    """One Provider that fulfils BOTH Entity + Relation Extraction in one pass (ADR-0013).

    The Pipeline does not force one Provider per Stage: a fused model (gliner-relex,
    the custom span-predictor) does joint NER + RE, so a single fused Stage produces
    both the :class:`~latence_core.contracts.EntityMention`s and the
    :class:`~latence_core.contracts.RelationMention`s. The Runner routes a fused
    Stage's :class:`FusedExtraction` to both downstream consumers without running a
    separate Entity-Extraction Stage — the S5 seam requirement.
    """

    def extract_fused(self, chunks: Iterable[ChunkRecord]) -> FusedExtraction:
        """Consume :class:`ChunkRecord`s and return mentions + relations together."""
        ...


@runtime_checkable
class Export(Protocol):
    """Materializes AI-ready outputs (JSONL + Parquet in S1)."""

    def export(self, records: Iterable[Record], storage: Storage, out_dir: str) -> list[str]:
        """Write records to ``out_dir`` on ``storage``; return the output URIs."""
        ...


class ProviderRegistry:
    """Discovers Providers from ``latence.providers`` entry points.

    Discovery is lazy and cached: entry points are enumerated once, then each is
    loaded (imported) only when its name is first requested. This keeps the
    registry cheap even when many heavy Provider packages are installed.
    """

    def __init__(self, group: str = ENTRY_POINT_GROUP) -> None:
        self._group = group
        self._entry_points: dict[str, metadata.EntryPoint] | None = None
        self._loaded: dict[str, type] = {}

    def _discover(self) -> dict[str, metadata.EntryPoint]:
        if self._entry_points is None:
            eps = metadata.entry_points(group=self._group)
            self._entry_points = {ep.name: ep for ep in eps}
        return self._entry_points

    def names(self) -> list[str]:
        return sorted(self._discover())

    def load(self, name: str) -> type:
        """Load and return the Provider class registered under ``name``."""
        if name in self._loaded:
            return self._loaded[name]
        eps = self._discover()
        if name not in eps:
            available = ", ".join(self.names()) or "<none>"
            msg = (
                f"No Provider registered as {name!r} in group "
                f"{self._group!r}. Available: {available}"
            )
            raise KeyError(msg)
        obj = eps[name].load()
        if not isinstance(obj, type):
            msg = f"Provider {name!r} did not resolve to a class (got {type(obj)!r})."
            raise TypeError(msg)
        self._loaded[name] = obj
        return obj
