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
        DisambiguationRecord,
        DocumentRecord,
        EntityMention,
        FeatureRecord,
        ParserInput,
        QuarantineRecord,
        Record,
        RedactionRecord,
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
class PIIDetector(Protocol):
    """Detects PII and produces masked/replaced text variants (CONTEXT ``Redaction``).

    The S6 Redaction Capability. Given the run's parsed
    :class:`~latence_core.contracts.DocumentRecord`s — each carrying the whole
    document's assembled markdown at its exact document offsets plus the page map — a
    Provider detects PII per document and returns one
    :class:`~latence_core.contracts.RedactionRecord` per document: the masked/replaced
    text variant and the structured list of detected
    :class:`~latence_core.contracts.PIISpan`s (type, document offset, action,
    placeholder), so the redaction is auditable and the corpus gets a PII-handled
    variant. Redaction is metadata-aware: the Provider keys its policy off the
    document's ``Classification.sensitivity`` (S6 AC). The PII type set is
    Provider/config-supplied — the Capability names no PII types. A document with no
    detected PII yields a RedactionRecord whose masked text equals the source and whose
    span list is empty (never an error).

    Redaction scans the **parsed document text** (not the markup-stripped chunk text),
    because the masked variant, every PIISpan offset, and the page resolution all live
    in the document's original-markdown coordinate system — the one the parent page map
    and the S6 round-trip use. Chunk text is markup-stripped with offsets remapped to
    the original, so its content length no longer equals its span width: scanning it
    would fabricate a broken coordinate system (offsets and pages would point at the
    wrong location). Redaction therefore depends on the Parse Stage.
    """

    def redact(self, documents: Iterable[DocumentRecord]) -> Iterator[RedactionRecord]:
        """Consume the run's parsed :class:`DocumentRecord`s; yield one RedactionRecord each.

        Each :class:`DocumentRecord` carries the whole document's assembled markdown
        (at original-document offsets) and its page map — exactly the text and
        coordinate system a text-based Provider scans. Records are emitted in first-seen
        document order for determinism. A ``PARSE_ERROR`` document (no usable content) is
        skipped, mirroring the Chunk Stage.
        """
        ...


@runtime_checkable
class Profiler(Protocol):
    """Computes corpus-level statistical + quality features (CONTEXT ``Profiling``).

    The S7 Profiling Capability. Profiling is a **corpus-level** Stage: it consumes the
    whole run's parsed :class:`~latence_core.contracts.DocumentRecord`s (and, when an
    Entity-Extraction Stage is upstream, the run's
    :class:`~latence_core.contracts.EntityMention`s) and emits
    :class:`~latence_core.contracts.FeatureRecord`s — one ``DOCUMENT``-scope record per
    parsed document carrying its per-document features (density, readability, Zipf α,
    compression ratio, structure) and exactly one ``CORPUS``-scope record carrying the
    cross-document aggregate features (entity frequency, co-occurrence, type consensus,
    source coverage).

    Both carriers are supplied so a fused corpus profile can relate document text to the
    entities extracted from it; a Provider that only wants document text ignores the
    mentions. The feature pass must be **streaming / spill-to-disk friendly** — it must
    not assume the whole corpus fits in RAM (S7 AC) — so a Provider folds documents in
    one at a time rather than materialising the corpus. A ``PARSE_ERROR`` document (no
    usable content) is skipped, mirroring the Chunk Stage. Records are emitted
    deterministically (document-scope in first-seen order, then the single corpus-scope
    record last), so a seeded run is byte-identical.
    """

    def profile(
        self, documents: Iterable[DocumentRecord], mentions: Iterable[EntityMention]
    ) -> Iterator[FeatureRecord]:
        """Consume the run's documents + mentions; yield per-document + corpus FeatureRecords."""
        ...


@runtime_checkable
class Disambiguator(Protocol):
    """Links mentions to canonical entities + merges duplicates (CONTEXT ``Disambiguation``).

    The S8 Disambiguation Capability — the algorithmic crown. Disambiguation is a
    **corpus-level** Stage: it consumes the whole run's
    :class:`~latence_core.contracts.EntityMention`s (required — the mentions it resolves and
    links) and, OPTIONALLY, the run's :class:`~latence_core.contracts.RelationMention`s (to
    normalise onto the canonical entities), and emits
    :class:`~latence_core.contracts.DisambiguationRecord`s at two scopes:

    * one ``ENTITY``-scope record per resolved
      :class:`~latence_core.contracts.CanonicalEntity` — a cross-document merged cluster with
      its member mentions, an audited, confidence-weighted merge log (**no silent
      over-merge** — a below-policy merge is logged not applied), an optional external-KB
      link (with a **graceful fallback** for unlinked mentions), and page-accurate
      Provenance + :class:`~latence_core.contracts.Evidence` back to the source mentions;
    * one ``RELATION``-scope record per :class:`~latence_core.contracts.NormalizedRelation` —
      a relation whose endpoints are remapped to canonical entities and whose label is L2
      normalised (fuzzy/alias, type-filtered, inverse-detected + direction-swapped).

    Both carriers are supplied so relations can be normalised onto the entities resolved in
    the same pass; a Provider that only disambiguates entities ignores the relations. The
    mention side is REQUIRED (unlike Profiling's optional mentions) — Disambiguation with no
    mentions has nothing to resolve. Records are emitted deterministically (all ENTITY
    records in canonical-text order, then RELATION records), so a seeded run is
    byte-identical (Baseline bar). The CPU reference path must work with no embedder/KB; a
    FAISS/GPU blocking Provider is a pluggable option, not required (S8 AC).
    """

    def disambiguate(
        self, mentions: Iterable[EntityMention], relations: Iterable[RelationMention]
    ) -> Iterator[DisambiguationRecord]:
        """Consume the run's mentions (+ optional relations); yield entities + relations."""
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
