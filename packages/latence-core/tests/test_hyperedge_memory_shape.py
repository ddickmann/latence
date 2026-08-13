"""The memory SHAPE of :meth:`HyperedgeBuilder.build`, pinned structurally (no RSS thresholds).

Why this file exists: the S3 campaign's UDA dataset (220 129 chunks / 2 603 887 entity mentions
/ 586 415 disambiguation records) was OOM-killed during Graph Assembly inside a container with a
hard 29 GB cgroup cap. The cause was shape, not scale — the builder held the WHOLE run as
pydantic records at once (a measured ~3.5 kB per ``EntityMention``, each dragging its own
``Provenance`` + ``Classification``: 8.5 GB of mentions before a single chunk of text). A
Hyperedge is chunk-local, so almost none of that had to be resident.

The two properties that fix pins, asserted structurally so they cannot flake:

1. **No input record is retained.** ``ChunkRecord``/``EntityMention``/``RelationMention`` are
   projected to compact views on arrival and dropped; by the time the first chunk is being
   segmented, ZERO of them are alive. This is the 8.5 GB → ~0.3 GB term.
2. **Chunk-local state does not accumulate.** At the LAST chunk of a many-chunk run, only that
   chunk's compact state is still alive — not the whole run's. This is what makes a corpus 100×
   bigger cost the same per-chunk working set rather than 100× more.

Both are measured by counting live objects (``gc.get_objects()`` filtered by exact type and by
this test's own id prefixes), sampled from inside the builder via the ``token_counter`` seam at
two known points: the first chunk and the last chunk. Counting objects is deterministic under
CPython refcounting; a peak-RSS threshold would not be, which is why none is used.

A small constant of slack (``_SLACK``) absorbs the references the interpreter legitimately holds
across a loop iteration (the previous iteration's locals are not rebound until the next statement
executes). It is far smaller than the accumulation this test would catch: at 40 chunks the
"accumulating" reading is 40 views / 320 metas, the bounded reading is ~2 / 8.
"""

from __future__ import annotations

import gc
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest
from latence_core.contracts import (
    CanonicalEntity,
    ChunkRecord,
    Classification,
    EntityMention,
    Evidence,
    Provenance,
    RelationMention,
)
from latence_core.hyperedge import (
    HyperedgeBuilder,
    HyperedgeConfig,
    _ChunkView,
    _MentionMeta,
    _RelationMeta,
)

if TYPE_CHECKING:
    from latence_core.contracts import GraphHyperedge

_CLS = Classification(language="en", sensitivity="internal")
_CHUNKS = 40
_SENTENCES = 3
_MENTIONS_PER_CHUNK = 2 * _SENTENCES + 2  # two related per sentence + two unrelated
_RELATIONS_PER_CHUNK = _SENTENCES
_SLACK = 4  # loop-carried references; see the module docstring
_MARKER = re.compile(r"Kx(\d+)Kx")


def _prov(doc: str, cs: int, ce: int) -> Provenance:
    return Provenance(
        source_uri=f"file:///{doc}.txt",
        file_name=f"{doc}.txt",
        file_type="txt",
        file_size=4096,
        document_id=doc,
        char_start=cs,
        char_end=ce,
        page_start=1,
        page_end=1,
    )


def _sentences(ci: int) -> list[str]:
    """Every sentence carries the chunk marker, so ANY span slice identifies its chunk."""
    marker = f"Kx{ci}Kx"
    out = [
        f"{marker} Alpha{ci}s{si} works for Beta{ci}s{si} in the city of Springfield "
        f"throughout the whole of the fiscal year. "
        for si in range(_SENTENCES)
    ]
    out.append(
        f"{marker} Gamma{ci} and Delta{ci} both attended the review with no stated relation. "
    )
    return out


def _chunk_id(ci: int) -> str:
    return f"memchunk:{ci:05d}"


def _corpus_stream() -> Iterator[tuple[str, Any]]:
    """Yield every input record tagged by stream, building each one lazily.

    Nothing here keeps a strong reference to a record it has yielded: the builder is the only
    holder, which is what makes "zero input records alive" a meaningful assertion.
    """
    for ci in range(_CHUNKS):
        doc = f"memdoc:{ci // 4:03d}"
        base = 100_000 * ci
        sentences = _sentences(ci)
        content = "".join(sentences)
        yield (
            "chunk",
            ChunkRecord(
                record_id=_chunk_id(ci),
                document_record_id=doc,
                index=ci % 4,
                content=content,
                token_count=len(content) // 4,
                provenance=_prov(doc, base, base + len(content)),
                classification=_CLS,
            ),
        )
        surfaces: list[str] = []
        for si in range(_SENTENCES):
            surfaces.extend((f"Alpha{ci}s{si}", f"Beta{ci}s{si}"))
        surfaces.extend((f"Gamma{ci}", f"Delta{ci}"))
        for mi, surface in enumerate(surfaces):
            local = content.index(surface)
            yield (
                "mention",
                EntityMention(
                    record_id=f"memmention:{ci:05d}:{mi:03d}",
                    document_record_id=doc,
                    chunk_record_id=_chunk_id(ci),
                    index=mi,
                    label="ORG",
                    text=surface,
                    confidence=round(0.5 + 0.04 * ((ci + mi) % 10), 3),
                    provenance=_prov(doc, base + local, base + local + len(surface)),
                    classification=_CLS,
                ),
            )
        for si in range(_RELATIONS_PER_CHUNK):
            yield (
                "relation",
                RelationMention(
                    record_id=f"memrelation:{ci:05d}:{si:03d}",
                    document_record_id=doc,
                    index=si,
                    label="works_for",
                    head_mention_id=f"memmention:{ci:05d}:{2 * si:03d}",
                    tail_mention_id=f"memmention:{ci:05d}:{2 * si + 1:03d}",
                    head_text=f"Alpha{ci}s{si}",
                    tail_text=f"Beta{ci}s{si}",
                    confidence=round(0.4 + 0.05 * ((ci + si) % 8), 3),
                    provenance=_prov(doc, base, base + 200),
                    classification=_CLS,
                ),
            )


def _entities() -> list[CanonicalEntity]:
    """One canonical entity per surface — small, and the only genuinely cross-chunk input."""
    out: list[CanonicalEntity] = []
    for ci in range(_CHUNKS):
        surfaces: list[str] = []
        for si in range(_SENTENCES):
            surfaces.extend((f"Alpha{ci}s{si}", f"Beta{ci}s{si}"))
        surfaces.extend((f"Gamma{ci}", f"Delta{ci}"))
        for mi, surface in enumerate(surfaces):
            mid = f"memmention:{ci:05d}:{mi:03d}"
            out.append(
                CanonicalEntity(
                    record_id=f"mementity:{surface}",
                    canonical_text=surface,
                    entity_type="ORG",
                    member_mention_ids=[mid],
                    member_link_confidences={mid: round(0.6 + 0.03 * (mi % 8), 3)},
                    source_document_ids=[f"memdoc:{ci // 4:03d}"],
                    evidence=Evidence(mention_ids=[mid], confidence=0.9),
                    provenance=_prov(f"memdoc:{ci // 4:03d}", 0, 10),
                    classification=_CLS,
                )
            )
    return out


def _stream(kind: str) -> Iterator[Any]:
    for tag, record in _corpus_stream():
        if tag == kind:
            yield record


def _live_counts() -> dict[str, int]:
    """Live instances of each relevant type, restricted to THIS test's records."""
    gc.collect()
    counts = dict.fromkeys(
        ("ChunkRecord", "EntityMention", "RelationMention", "view", "mention", "relation"), 0
    )
    for obj in gc.get_objects():
        cls = type(obj)
        if cls is ChunkRecord:
            if obj.record_id.startswith("memchunk:"):
                counts["ChunkRecord"] += 1
        elif cls is EntityMention:
            if obj.record_id.startswith("memmention:"):
                counts["EntityMention"] += 1
        elif cls is RelationMention:
            if obj.record_id.startswith("memrelation:"):
                counts["RelationMention"] += 1
        elif cls is _ChunkView:
            if obj.record_id.startswith("memchunk:"):
                counts["view"] += 1
        elif cls is _MentionMeta:
            if obj.record_id.startswith("memmention:"):
                counts["mention"] += 1
        elif cls is _RelationMeta and obj.record_id.startswith("memrelation:"):
            counts["relation"] += 1
    return counts


class _Sampler:
    """A ``token_counter`` that also samples live-object counts at two known chunks.

    ``token_counter`` is the one seam the builder calls from *inside* per-chunk segmentation, and
    every span slice carries its chunk's ``Kx<i>Kx`` marker — so it doubles as a probe that knows
    which chunk is being processed without the test reaching into the builder.
    """

    def __init__(self) -> None:
        self.samples: dict[int, dict[str, int]] = {}

    def __call__(self, text: str) -> int:
        match = _MARKER.search(text)
        if match is not None:
            ci = int(match.group(1))
            if ci in (0, _CHUNKS - 1) and ci not in self.samples:
                self.samples[ci] = _live_counts()
        return max(1, len(text) // 4)


def _build(counter: Any = None) -> list[GraphHyperedge]:
    config = HyperedgeConfig(
        token_counter=counter, token_counter_id="test:sampler"
    ) if counter is not None else HyperedgeConfig()
    builder = HyperedgeBuilder("corpus-mem", config)
    return builder.build(
        _stream("chunk"),
        _stream("mention"),
        _stream("relation"),
        _entities(),
        lambda eid: f"node:{eid}",
    )


def test_build_retains_no_input_records_and_no_cross_chunk_pileup() -> None:
    """The two pinned properties, sampled from inside the run at the first and last chunk."""
    sampler = _Sampler()
    hyperedges = _build(sampler)

    assert hyperedges, "the fixture must actually produce hyperedges for the probe to be real"
    assert set(sampler.samples) == {0, _CHUNKS - 1}, (
        "the probe did not fire on both the first and the last chunk; the token_counter seam or "
        f"the chunk markers changed. Sampled: {sorted(sampler.samples)}"
    )
    first, last = sampler.samples[0], sampler.samples[_CHUNKS - 1]

    # (1) No pydantic input record survives the projection — the 8.5 GB term at UDA scale.
    for when, sample in (("first chunk", first), ("last chunk", last)):
        for kind in ("ChunkRecord", "EntityMention", "RelationMention"):
            assert sample[kind] == 0, (
                f"{sample[kind]} input {kind}(s) were still alive at the {when}. build() must "
                "project every input record to its compact view on arrival and drop the record; "
                "holding them is exactly what OOM-killed the UDA run under the 29 GB cap."
            )

    # (2) Chunk-local state is released per chunk, not accumulated for the run.
    assert first["view"] >= _CHUNKS - _SLACK, (
        f"only {first['view']} chunk views existed at the first chunk of {_CHUNKS}; the fixture "
        "is no longer exercising a many-chunk run."
    )
    assert last["view"] <= _SLACK, (
        f"{last['view']} chunk views were still alive at the LAST of {_CHUNKS} chunks — chunk "
        "text is accumulating across the run instead of being released after its hyperedges "
        "are emitted."
    )
    total_mentions = _CHUNKS * _MENTIONS_PER_CHUNK
    assert first["mention"] >= total_mentions - _SLACK
    assert last["mention"] <= _MENTIONS_PER_CHUNK + _SLACK, (
        f"{last['mention']} mention projections were alive at the LAST chunk; only one chunk's "
        f"worth (~{_MENTIONS_PER_CHUNK}) should be. Per-chunk mention state is accumulating."
    )
    assert last["relation"] <= _RELATIONS_PER_CHUNK + _SLACK, (
        f"{last['relation']} relation projections were alive at the LAST chunk; only one "
        f"chunk's worth (~{_RELATIONS_PER_CHUNK}) should be."
    )


def test_streaming_build_is_byte_identical_to_a_list_fed_build() -> None:
    """Bounded memory must not have cost a single byte of output.

    Same corpus, once through one-shot generators (what the Runner hands the assembler) and once
    through fully materialised lists — the serialized hyperedges must be identical, including
    order, ids, ``support``, ``merged_sources`` and ``tau_text``.
    """
    streamed = [h.model_dump_json() for h in _build()]

    records = list(_corpus_stream())
    builder = HyperedgeBuilder("corpus-mem")
    materialised = [
        h.model_dump_json()
        for h in builder.build(
            [r for tag, r in records if tag == "chunk"],
            [r for tag, r in records if tag == "mention"],
            [r for tag, r in records if tag == "relation"],
            _entities(),
            lambda eid: f"node:{eid}",
        )
    ]
    assert streamed == materialised
    assert streamed


class _OneShot:
    """Raises on a second pass — a generator would silently read an EMPTY corpus instead.

    Mirrors ``test_accumulator_memory_bound._OneShot``: the Runner hands Graph Assembly one-shot
    generators over checkpoints, so a builder that iterated an input twice would drop half the
    corpus with no error at all.
    """

    def __init__(self, records: list[Any]) -> None:
        self._records = records
        self.passes = 0

    def __iter__(self) -> Iterator[Any]:
        self.passes += 1
        if self.passes > 1:
            msg = (
                "an input stream was iterated more than once; the Runner hands Graph Assembly "
                "one-shot generators, so a second pass silently reads an EMPTY corpus"
            )
            raise AssertionError(msg)
        yield from self._records


@pytest.mark.parametrize("weak", [True, False])
def test_each_input_stream_is_consumed_exactly_once(weak: bool) -> None:
    """Every stream is single-pass — the property streaming the inputs at all depends on."""
    records = list(_corpus_stream())
    streams = {
        kind: _OneShot([r for tag, r in records if tag == kind])
        for kind in ("chunk", "mention", "relation")
    }
    builder = HyperedgeBuilder("corpus-mem", HyperedgeConfig(weak_hyperedges=weak))
    hyperedges = builder.build(
        streams["chunk"],
        streams["mention"],
        streams["relation"],
        _entities(),
        lambda eid: f"node:{eid}",
    )
    assert hyperedges
    assert all(s.passes == 1 for s in streams.values())
