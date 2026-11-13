"""ReuseOverlay — ADR-0061 §5's cross-run fold rule, written once (W17/ADR-0043 + ADR-0061).

A delta run excludes an unchanged document at Source (so its doc-level Stages never run) and
hands the Runner that document's persisted doc-level records — the **reuse overlay**. The
corpus-level Stages and the doc-level sinks below them must then see the whole corpus
(``fresh ∪ reused``) exactly as if every document had been re-extracted. ADR-0061 §5 states the
ONE rule that makes the equivalence hold for the *stream*, not just the set:

    the reuse overlay of carrier ``T`` folds into the FIRST document-level Stage stream
    producing ``T``, document-interleaved in source order.

Both halves of the rule live here:

* **where** — :func:`folds_at`: the stream of a DOCUMENT-level Stage (:func:`is_doc_level_stage`)
  that PRODUCES the carrier (:func:`stage_produces`). Never a corpus-level Stage's stream: a
  corpus-level re-emitter (Type Consolidation) already carries the union in
  its own checkpoint, so folding again would double-count every reused record. And only the
  FIRST such stream per enumeration, so the union reaches each consumer exactly once.
* **how** — :func:`merge_overlay_doc_ordered`: document-interleaved in source order (not
  appended), so every consumer sees ``fresh ∪ reused`` in the exact arrival order a full rebuild
  would deliver — the resolver's first-seen display spelling, the profiler's last-document
  fields, the chunk order and hence the export row order (and the embedder's batch
  composition) all match a from-scratch run byte-for-byte.

What VARIES between the Runner's consumers is only (a) the *enumeration* the fold walks — a
Stage's ``depends_on`` parents, the whole pipeline in topological order, or an Export's
nearest-producer walk — and (b) what happens to an overlay carrier the enumeration never gave a
fold target: the :class:`OverlayTrailer` policy. Those used to be six hand-synchronised
re-derivations of the rule inside ``runner.py``; each is now one call into this module, with its
enumeration passed in and its trailer policy named.

The overlay's within-document record order is EMISSION order, reconstructed at indexing time:
the doc-level reuse store persists records id-sorted LEXICOGRAPHICALLY (its canonical on-disk
form — ``CorpusStore._encode_doclevel``), which is not emission order (``…#chunk-10`` sorts
before ``…#chunk-2``). Every reference id scheme mints ``<parent>#<kind>-<counter>`` in emission
order, so the digit-aware :func:`natural_record_order` sort reconstructs each document's
original within-run order from the canonical store — including stores committed by older
versions (the fix is read-side, not a store rewrite). This is the divergence the ticket-03
real-data smoke caught (commit 4249620); the e2e regression runs a >10-chunk document through
the reuse path, and the direct unit tests here pin the sort itself.

Pure module: no I/O, no Storage — a total function of (records, stage, carrier, pipeline). The
Runner composes it with its own checkpoint readers; that is what makes the fold independently
unit-testable without a pipeline run.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING, Protocol, TypeVar

from latence_core.capability_descriptor import CORPUS_LEVEL_KINDS
from latence_core.contracts import EntityMention, Provenance, Record, RelationMention
from latence_core.run_store import stage_output_type

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from latence_core.pipeline import Pipeline, Stage


class ProvenanceCarrier(Protocol):
    """Any record the reuse-overlay merge can order — everything with a Provenance.

    The ADR-0061 document-interleaved fold (:func:`merge_overlay_doc_ordered`) reads exactly
    two fields: ``provenance.source_uri`` (the rebuild's arrival-order key) and
    ``provenance.document_id`` (the grouping key). Every inter-Stage carrier — including
    ``ParserInput`` and the ``Record`` base an Export materializes — carries a Provenance, so
    the merge is typed against this Protocol rather than the Runner's ``StageOutput`` union.
    """

    provenance: Provenance


#: A record type the merge/overlay helpers accept: bound by the Provenance the merge reads.
_TP = TypeVar("_TP", bound=ProvenanceCarrier)

#: Digit runs in a record id, for the natural (emission-order) sort of the reuse overlay.
_DIGIT_RUNS = re.compile(r"(\d+)")


def natural_record_order(record_id: str) -> tuple[str | int, ...]:
    """The digit-aware sort key that reconstructs emission order from an id-sorted store.

    Splits the id into alternating text/digit runs (``re.split`` with one capture group always
    alternates non-captured text and captured digits, starting with a possibly-empty text run),
    so two ids are compared text-run by text-run and COUNTER BY NUMBER: ``…#chunk-2`` orders
    before ``…#chunk-10``. The alternation guarantee makes the mixed tuple type-safe — an int
    and a str never meet at the same tuple position.
    """
    return tuple(int(tok) if tok.isdigit() else tok for tok in _DIGIT_RUNS.split(record_id))


def merge_overlay_doc_ordered(fresh: Iterable[_TP], overlay: Sequence[_TP]) -> Iterator[_TP]:
    """Fold the reuse overlay into a doc-level stream in the full rebuild's arrival order.

    (ADR-0061 §5, the **how** half.) Both sides are document-grouped subsequences of the same
    global order: a doc-level checkpoint holds its records per document in the Source's
    globally-sorted URI order (:meth:`~latence_core.storage.Storage.walk_files`), and the
    overlay's records are grouped per reused document with within-document order preserved from
    their original run. Merging the two by ``(provenance.source_uri, document_id)`` therefore
    reproduces exactly the arrival order a full rebuild of the whole corpus would deliver —
    which is what upgrades the ADR-0043 equivalence from "same set, ids stable" to "same
    stream" (see the module docstring for the order-dependences this pins).

    Memory: O(overlay) — the delta path already holds the overlay whole — plus O(1) fresh
    records; the fresh side stays a lazy stream. An empty overlay yields the fresh stream
    untouched (the non-delta no-op).

    Honest limit: with several Source Stages whose URI ranges interleave, a rebuild's arrival
    order is "Source by Source", not globally URI-sorted, so the byte-order guarantee holds
    for single-Source pipelines (every reference stack); the set/ranking equivalence holds
    regardless.
    """
    if not overlay:
        yield from fresh
        return
    groups: dict[str, list[_TP]] = {}
    order_key: dict[str, tuple[str, str]] = {}
    for record in overlay:
        doc_id = record.provenance.document_id
        groups.setdefault(doc_id, []).append(record)
        order_key.setdefault(doc_id, (record.provenance.source_uri, doc_id))
    pending = sorted(groups, key=order_key.__getitem__)
    idx = 0
    current_doc: str | None = None
    for record in fresh:
        doc_id = record.provenance.document_id
        if doc_id != current_doc:
            current_doc = doc_id
            key = (record.provenance.source_uri, doc_id)
            while idx < len(pending) and order_key[pending[idx]] < key:
                yield from groups[pending[idx]]
                idx += 1
        yield record
    while idx < len(pending):
        yield from groups[pending[idx]]
        idx += 1


def overlay_carriers(carrier: type[object]) -> tuple[type[Record], ...]:
    """The concrete overlay carrier(s) one checkpoint carrier delivers.

    A MIXED-carrier Stage (a fused extractor, Type Consolidation) declares the
    :class:`~latence_core.contracts.Record` sentinel and delivers BOTH mention carriers; every
    other Record carrier delivers itself; a non-Record carrier (a Source/Intake-Screening
    Stage's raw ``ParserInput``) delivers nothing — raw inputs are never reused.
    """
    if carrier is Record:
        return (EntityMention, RelationMention)
    if isinstance(carrier, type) and issubclass(carrier, Record):
        return (carrier,)
    return ()


def is_doc_level_stage(stage: Stage) -> bool:
    """True when ``stage``'s Capability is document-level (the descriptor partition, L2/C2)."""
    return stage.capability not in CORPUS_LEVEL_KINDS


def stage_produces(stage: Stage, pipeline: Pipeline, expected: type[object]) -> bool:
    """True when ``stage``'s checkpoint carrier includes the ``expected`` record type.

    Read off the Stage's declared carrier (:func:`~latence_core.run_store.stage_output_type`):
    an exact carrier match, or — for a MIXED-carrier Stage, whose carrier is the
    :class:`~latence_core.contracts.Record` sentinel — either mention carrier.
    """
    out = stage_output_type(stage, pipeline)
    if out is Record:
        return expected in (EntityMention, RelationMention)
    return out is expected


def folds_at(stage: Stage | None, pipeline: Pipeline, expected: type[object]) -> bool:
    """ADR-0061 §5's fold-target decision, the **where** half — one function, every consumer.

    The overlay of carrier ``expected`` folds into a Stage's stream iff that Stage is
    DOCUMENT-level AND produces the carrier. Corpus-level producers are excluded because their
    checkpoint already carries the ``fresh ∪ reused`` union (their own accumulator input got
    the overlay folded in), so folding again would deliver every reused record twice. The
    "FIRST such stream only" part of the rule is the enumerating caller's
    (:meth:`ReuseOverlay.fold_into` folds once then disarms; the Export dispatch has exactly
    one candidate — its nearest producer). ``None`` (an enumeration slot with no resolvable
    Stage) never folds.
    """
    return (
        stage is not None
        and stage_produces(stage, pipeline, expected)
        and is_doc_level_stage(stage)
    )


class OverlayTrailer(Enum):
    """What :meth:`ReuseOverlay.fold_into` does with an overlay no enumerated stream claimed.

    The fold rule names WHERE the overlay belongs; the trailer names what each consumer's
    contract says when the enumeration contains no fold target. The three policies are REAL
    behavioral differences between the Runner's consumers, preserved exactly:

    * ``APPEND_IF_CARRIER_UNSEEN`` — the accumulator gather over a Stage's ``depends_on``
      parents (``StageInputs.stream_of`` under an ACCUMULATOR plan). An accumulator's contract is
      union semantics: an
      overlay carrier NO parent produces (a degenerate DAG) is still appended last, so the
      union always reaches the accumulator exactly once. But a CORPUS-LEVEL parent producing
      the carrier means the union already arrived through that parent's checkpoint — appending
      would double-count — so "seen" counts producers of ANY level.
    * ``NEVER`` — a doc-level per-carrier input (``StageInputs.stream_of`` under a
      non-accumulator plan). A doc-level Stage's input contract is "its parents' records",
      never records no parent produced — so an unclaimed overlay is dropped, exactly as the
      overlay-blind eager gather would have.
    * ``ALWAYS`` — the corpus-wide gather over every Stage in topological order
      (``StageInputs.corpus_records_of``). Its contract is "every record of this carrier in the
      run"; the enumeration spans the whole pipeline, so an unclaimed overlay means no
      doc-level producer exists anywhere and the reused records ARE the corpus's records of
      that carrier — appended last, unconditionally.
    """

    APPEND_IF_CARRIER_UNSEEN = "append_if_carrier_unseen"
    NEVER = "never"
    ALWAYS = "always"


class ReuseOverlay:
    """The cross-run doc-level reuse overlay, indexed by carrier type (W17/ADR-0043).

    Constructed from the reused records a delta hands :meth:`LocalRunner.run
    <latence_core.runner.LocalRunner.run>` (``reuse=…``): grouped by concrete carrier type and
    sorted per carrier by the NATURAL record-id key (:func:`natural_record_order`), which
    reconstructs each document's emission order from the lexicographically-id-sorted reuse
    store (see the module docstring). Empty (``ReuseOverlay()``) every method is a no-op and
    :meth:`fold_into` yields the enumerated streams untouched — a non-delta run is
    byte-identical by construction.
    """

    def __init__(self, records: Iterable[Record] | None = None) -> None:
        by_carrier: dict[type[Record], list[Record]] = {}
        for record in records or []:
            by_carrier.setdefault(type(record), []).append(record)
        for group in by_carrier.values():
            group.sort(key=lambda r: natural_record_order(r.record_id))
        self._by_carrier = by_carrier

    def of_type(self, expected: type[_TP]) -> list[_TP]:
        """The overlay's records of one carrier type, emission-ordered per document.

        Empty when the carrier was never reused (including always, for a non-delta run).
        """
        return self._by_carrier.get(expected, [])  # type: ignore[arg-type,return-value]

    def has_carrier(self, carrier: type[object]) -> bool:
        """True when the overlay holds records whose concrete type IS ``carrier``.

        Deliberately an exact-type lookup, not an :func:`overlay_carriers` expansion: the one
        consumer (``StageInputs.carriers_present``'s wiring-check top-up) asks whether a parent
        whose FRESH output is empty — because every one of its documents was reused — still
        "produces" its declared carrier via the overlay, and a mixed-carrier declaration (the
        ``Record`` sentinel) is not a concrete overlay carrier.
        """
        return bool(self._by_carrier.get(carrier))  # type: ignore[arg-type]

    def records_for_carrier(self, carrier: type[object]) -> list[Record]:
        """Every overlay record a checkpoint carrier delivers (mixed → both mention sides).

        The Export fold's materialization: a fused ancestor's pass-through carrier is the
        mixed mention/relation stream, so both reused mention carriers are delivered — in
        carrier order (mentions then relations), each emission-ordered per document, exactly
        as :func:`merge_overlay_doc_ordered` expects its overlay side.
        """
        return [
            record
            for concrete in overlay_carriers(carrier)
            for record in self._by_carrier.get(concrete, [])
        ]

    def count_for_stage(
        self, stage: Stage, pipeline: Pipeline, *, already_counted: set[type[Record]]
    ) -> int:
        """Overlay records the fold rule delivers through this parent's stream — counted once.

        The counting twin of :meth:`fold_into` for an accumulator's ``records_in`` metric
        (``StageInputs.count`` under an ACCUMULATOR plan): a DOC-level parent tops the count up
        by its carrier(s)' overlay sizes; a corpus-level parent contributes nothing (its
        checkpoint's line count already contains the union). ``already_counted`` is the caller's
        cross-parent memory of the "first producer only" half of the rule — a carrier two parents
        produce is counted at the first, mirroring where the fold would deliver it.
        """
        if not is_doc_level_stage(stage):
            return 0
        total = 0
        for carrier in overlay_carriers(stage_output_type(stage, pipeline)):
            if carrier in already_counted:
                continue
            already_counted.add(carrier)
            total += len(self._by_carrier.get(carrier, []))
        return total

    def fold_into(
        self,
        expected: type[_TP],
        sources: Iterable[tuple[Stage | None, Iterator[_TP]]],
        pipeline: Pipeline,
        *,
        trailer: OverlayTrailer,
    ) -> Iterator[_TP]:
        """Yield the enumerated streams with the ``expected`` overlay folded in per ADR-0061 §5.

        ``sources`` is the caller's enumeration — ``(stage, records-of-``expected``)`` pairs in
        the caller's deterministic order (``depends_on`` order for a gather, topological order
        for the corpus-wide stream). The overlay folds document-interleaved
        (:func:`merge_overlay_doc_ordered`) into the FIRST stream whose Stage is a fold target
        (:func:`folds_at`); every other stream passes through untouched, and an overlay no
        stream claimed falls to the ``trailer`` policy (:class:`OverlayTrailer` — the one
        honest behavioral difference between the Runner's consumers). With an empty overlay
        this is exactly ``chain(*streams)`` — the non-delta byte-identity.
        """
        overlay = self.of_type(expected)
        overlay_pending = bool(overlay)
        carrier_seen = False
        for stage, stream in sources:
            carrier_seen = carrier_seen or (
                stage is not None and stage_produces(stage, pipeline, expected)
            )
            if overlay_pending and folds_at(stage, pipeline, expected):
                overlay_pending = False
                yield from merge_overlay_doc_ordered(stream, overlay)
            else:
                yield from stream
        if not overlay_pending:
            return
        if trailer is OverlayTrailer.ALWAYS or (
            trailer is OverlayTrailer.APPEND_IF_CARRIER_UNSEEN and not carrier_seen
        ):
            yield from overlay
