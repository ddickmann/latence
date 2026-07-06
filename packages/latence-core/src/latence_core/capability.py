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
        ParserInput,
        QuarantineRecord,
        Record,
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
