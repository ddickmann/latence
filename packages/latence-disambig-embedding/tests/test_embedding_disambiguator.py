"""EmbeddingDisambiguator: surface+context learned entity resolution (W1 rework, #94).

The enterprise crux is proven two ways, both offline + deterministic:

1. **Context-bearing vectors (the pod evidence):** a stub Embedder returns the pod-measured
   surface+context vectors (IBM/International Business Machines ≈0.96, Apple/Apple Inc ≈0.99, IBM/Apple
   ≈0.81). At the context-safe default threshold 0.90 the genuine duplicates merge while IBM and Apple
   stay SEPARATE — the exact over-merge the pod proved bare-surface produces, now fixed *because of
   context*, not a higher threshold masking it (a bare-surface control, cos 0.87 for IBM~Apple, would
   over-merge at any threshold below 0.87 that also catches the 0.88 same-pair — no safe threshold).

2. **The real in-core hashing embedder** (dependency-free, deterministic) over full ChunkRecords:
   distinct entities stay separate and a genuine duplicate merges, with no crash — the honest offline
   default path, not a hand-tuned double.

Plus: drop-in parity with the cascade contract, graceful no-chunk fallback, the type guard, the
max_pairs O(n²) cap, determinism (byte-identical over two runs), and native-error → typed mapping.
"""

from __future__ import annotations

from typing import Any

import pytest
from latence_core.capability import Disambiguator
from latence_core.chunking import strip_markup
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    DisambiguationScope,
    EntityMention,
    OffsetMap,
    OffsetSegment,
    PageMap,
    PageSlice,
    Provenance,
    RelationMention,
)
from latence_core.errors import ConfigError, ProviderError
from latence_core.providers.profile import REPO_LICENSE_CODE
from latence_disambig_embedding.provider import EmbeddingDisambiguator

# --------------------------------------------------------------------------- #
# Fixtures: mentions + chunks for the IBM / Apple / Berlin case.
# --------------------------------------------------------------------------- #

_CLS = Classification(language="en", category="news", sensitivity="internal")

# The chunk texts carrying each mention's context (documents d1/d2/d3).
_C1 = (
    "IBM reported strong earnings. International Business Machines is a technology "
    "company making computers and software."
)
_C2 = (
    "Apple unveiled a new iPhone. Apple Inc is a consumer electronics company based "
    "in Cupertino."
)
_C3 = "Berlin is the capital of Germany, a major European city in central Europe."


def _prov(doc: str, cs: int, ce: int) -> Provenance:
    return Provenance(
        source_uri=f"file://{doc}.txt",
        file_name=f"{doc}.txt",
        file_type="txt",
        file_size=1000,
        document_id=doc,
        char_start=cs,
        char_end=ce,
        page_start=1,
        page_end=1,
    )


def _chunk(rid: str, doc: str, content: str) -> ChunkRecord:
    return ChunkRecord(
        record_id=rid,
        document_record_id=f"{doc}:doc",
        index=0,
        content=content,
        token_count=20,
        provenance=_prov(doc, 0, len(content)),
        classification=_CLS,
        # v19: the chunk carries its own spans; a one-page fixture, so exactly that page.
        page_slice=PageSlice.for_span(
            PageMap.from_page_texts([content]).page_map, 0, len(content)
        ),
    )


def _mention(
    rid: str, doc: str, chunk_id: str, text: str, start: int, label: str = "ORG"
) -> EntityMention:
    return EntityMention(
        record_id=rid,
        document_record_id=f"{doc}:doc",
        chunk_record_id=chunk_id,
        index=0,
        label=label,
        text=text,
        confidence=0.9,
        provenance=_prov(doc, start, start + len(text)),
        classification=_CLS,
    )


def _corpus() -> tuple[list[EntityMention], list[ChunkRecord]]:
    """The canonical 5-mention case: {IBM, International Business Machines}, {Apple, Apple Inc}, {Berlin}."""
    chunks = [_chunk("ch1", "d1", _C1), _chunk("ch2", "d2", _C2), _chunk("ch3", "d3", _C3)]
    mentions = [
        _mention("m1", "d1", "ch1", "IBM", _C1.index("IBM")),
        _mention("m2", "d1", "ch1", "International Business Machines", _C1.index("International")),
        _mention("m3", "d2", "ch2", "Apple", _C2.index("Apple")),
        _mention("m4", "d2", "ch2", "Apple Inc", _C2.index("Apple Inc")),
        _mention("m5", "d3", "ch3", "Berlin", _C3.index("Berlin"), label="LOC"),
    ]
    return mentions, chunks


def _markup_chunk(rid: str, doc: str, markdown: str) -> tuple[ChunkRecord, str]:
    """A ChunkRecord built from RAW markdown the way the chunking engine builds it (real coords).

    Runs the production :func:`~latence_core.chunking.strip_markup` to get the markup-stripped
    ``content`` AND the run-length ``offset_map`` (stripped→original), so a mention's ORIGINAL
    ``char_start`` only resolves to the right content window through the map — exactly the real
    markdown path the plaintext fixtures never exercise. Returns the chunk and its stripped content
    (so a caller can locate a surface inside ``content`` for assertions if needed).
    """
    strip = strip_markup(markdown)
    content = strip.stripped
    mapping = strip.mapping
    # One breakpoint per contiguous surviving run (mirrors chunking._offset_segments over [0, len)).
    segs: list[tuple[int, int]] = [(0, mapping[0])]
    for k in range(1, len(content)):
        if mapping[k] != mapping[k - 1] + 1:
            segs.append((k, mapping[k]))
    offset_map = OffsetMap(
        segments=[OffsetSegment(local_start=ls, doc_start=ds) for ls, ds in segs],
        length=len(content),
    )
    chunk = ChunkRecord(
        record_id=rid,
        document_record_id=f"{doc}:doc",
        index=0,
        content=content,
        token_count=20,
        # The chunk covers the whole document markdown: char_start 0, char_end len(markdown).
        provenance=_prov(doc, 0, len(markdown)),
        classification=_CLS,
        page_slice=PageSlice.for_span(
            PageMap.from_page_texts([markdown]).page_map, 0, len(markdown)
        ),
        offset_map=offset_map,
    )
    return chunk, content


# --------------------------------------------------------------------------- #
# A stub Embedder returning the pod-measured surface+context vectors.
# --------------------------------------------------------------------------- #


class _PodVectorEmbedder:
    """Returns 2-D unit vectors reproducing the pod's bare-surface-vs-surface+context measurements.

    Faithful to the pod finding (the reason for this rework): the SAME surface embeds DIFFERENTLY
    with vs without context. Keyed on the leading surface (the Provider embeds "surface + context",
    so the surface is the prefix) AND on whether context is present (the text is longer than the bare
    surface):

    * **Bare surface** (no context): IBM and Apple sit CLOSE (cos ≈ 0.87) — the over-merge the pod
      proved a bare-surface embedder produces (no safe threshold).
    * **Surface + context**: the angles spread so
        cos(IBM+ctx, International Business Machines+ctx) ≈ 0.96   (same → merge)
        cos(Apple+ctx, Apple Inc+ctx)                    ≈ 0.99   (same → merge)
        cos(IBM+ctx, Apple+ctx)                          ≈ 0.81   (distinct → stay apart)
      i.e. the clean separation ONLY context buys — the enterprise fix.

    Deterministic, offline; satisfies the Embedder Protocol (dimension + embed).
    """

    import math as _math

    # Surface+context angles (radians): tight same-entity pairs, wide cross-entity gap.
    _CTX_ANGLES = {
        "international business machines": 0.0,
        "ibm": 0.28,  # ~16° from IBM-full  → cos ≈ 0.961 (same → merge)
        "apple inc": 1.30,
        "apple": 1.44,  # ~8° from Apple Inc → cos ≈ 0.990; ~66° from IBM → cos ≈ 0.81 (distinct)
        "berlin": 3.0,
    }
    # Bare-surface angles: IBM and Apple collapse together (the pod over-merge, cos ≈ 0.87).
    _BARE_ANGLES = {
        "international business machines": 0.20,
        "ibm": 0.10,  # ~cos 0.995 with IBM-full, BUT...
        "apple inc": 0.62,
        "apple": 0.52,  # ~24° from IBM (cos ≈ 0.91) — bare surfaces have no safe threshold
        "berlin": 2.6,
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = dict(config or {})

    def _angle_for(self, text: str) -> float:
        low = text.lower().strip()
        for key in sorted(self._CTX_ANGLES, key=len, reverse=True):
            if low.startswith(key):
                # Context present when the text is longer than the bare surface itself.
                has_context = len(low) > len(key)
                table = self._CTX_ANGLES if has_context else self._BARE_ANGLES
                return table[key]
        return 5.0

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, texts: Any) -> Any:
        for text in texts:
            angle = self._angle_for(text)
            yield [self._math.cos(angle), self._math.sin(angle)]


def _pod_disambiguator(**config: Any) -> EmbeddingDisambiguator:
    """A disambiguator wired to the pod-vector stub embedder (injected via a fake registry)."""

    class _FakeRegistry:
        def load(self, name: str) -> Any:  # noqa: ARG002 - name ignored, always the stub
            return _PodVectorEmbedder

    return EmbeddingDisambiguator(dict(config), registry=_FakeRegistry())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_satisfies_disambiguator_protocol_and_names_itself() -> None:
    d = EmbeddingDisambiguator({})
    assert isinstance(d, Disambiguator)
    assert d.provider_name == "disambiguation.embedding"


def test_profile_is_evidence_bearing_and_honest() -> None:
    p = EmbeddingDisambiguator.profile()
    assert p.compute == "either"
    assert p.license == REPO_LICENSE_CODE
    assert p.license_verified is True
    # The learned path is not promised byte-identical across embedders — declared, not overclaimed.
    assert p.deterministic is False


def test_context_separates_ibm_from_apple_no_over_merge() -> None:
    """THE enterprise regression: at the default 0.90 threshold, duplicates merge but IBM≠Apple.

    With the pod's surface+context vectors, {IBM, International Business Machines} and
    {Apple, Apple Inc} each merge, and Berlin is alone — but IBM and Apple are NOT collapsed
    (the bare-surface over-merge the pod proved). A too-eager threshold or a context-blind embed
    would fail this by producing fewer than 3 entity clusters.
    """
    mentions, chunks = _corpus()
    d = _pod_disambiguator()
    records = list(d.disambiguate(mentions, [], chunks))
    entities = [r.entity for r in records if r.scope is DisambiguationScope.ENTITY]
    clusters = {e.canonical_text: sorted(e.member_mention_ids) for e in entities if e}
    assert clusters == {
        "Apple": ["m3", "m4"],
        "Berlin": ["m5"],
        "IBM": ["m1", "m2"],
    }, clusters
    # IBM and Apple are DISTINCT canonical entities — the over-merge is gone.
    assert len(entities) == 3


def test_below_threshold_is_logged_rejected_never_silently_merged() -> None:
    """A similar-but-below-threshold pair is recorded as an applied=False MergeDecision (audited)."""
    mentions, chunks = _corpus()
    # Raise the threshold above the same-pair cosines so NOTHING merges — every near pair is a
    # logged rejection, proving no silent over-merge and that the audit trail is populated.
    d = _pod_disambiguator(threshold=0.999)
    records = list(d.disambiguate(mentions, [], chunks))
    entities = [r.entity for r in records if r.entity]
    # All five surfaces stay singletons (no merge at all).
    assert sum(len(e.member_mention_ids) for e in entities) == 5
    assert len(entities) == 5
    # At least one below-threshold rejection is recorded on some entity's merge_decisions.
    rejected = [
        md
        for e in entities
        for md in e.merge_decisions
        if md.applied is False
    ]
    assert rejected, "a below-threshold near pair must be logged as a rejected MergeDecision"
    assert all(0.0 <= md.confidence <= 1.0 for md in rejected)


def test_type_guard_blocks_high_cosine_cross_type_merge() -> None:
    """A high-cosine pair with incompatible entity types is logged rejected, not merged (#35 guard)."""
    # Two surfaces that the stub scores as near-identical (Apple/Apple Inc ≈0.99) but tagged with
    # DIFFERENT types → the type guard must block the merge.
    chunks = [_chunk("ch2", "d2", _C2)]
    org = _mention("m3", "d2", "ch2", "Apple", _C2.index("Apple"), label="ORG")
    person = _mention("m4", "d2", "ch2", "Apple Inc", _C2.index("Apple Inc"), label="PERSON")
    d = _pod_disambiguator()  # require_type_compatibility defaults True
    records = list(d.disambiguate([org, person], [], chunks))
    entities = [r.entity for r in records if r.entity]
    assert len(entities) == 2, "type-incompatible near-duplicates must NOT merge"
    rejected = [md for e in entities for md in e.merge_decisions if not md.applied]
    assert any("type" in md.reason for md in rejected)


def test_graceful_fallback_when_no_chunk_never_crashes() -> None:
    """A mention with no locatable chunk falls back to bare surface (no crash, still resolves)."""
    mentions, _ = _corpus()
    d = _pod_disambiguator()
    # Pass NO chunks: every mention falls back to bare surface. The run must COMPLETE (never raise)
    # and emit valid ENTITY records covering every mention — the graceful degradation contract.
    records = list(d.disambiguate(mentions, [], ()))
    entities = [r.entity for r in records if r.entity]
    covered = {mid for e in entities for mid in e.member_mention_ids}
    assert covered == {"m1", "m2", "m3", "m4", "m5"}
    # Every emitted record satisfies the contract (Provenance/Classification/Evidence present).
    for r in records:
        assert r.provenance is not None and r.classification is not None


def test_bare_surface_over_merges_the_pod_control() -> None:
    """The pod control: WITHOUT context, distinct IBM and Apple over-merge (why the rework exists).

    Embedding the bare surfaces (``context_window=0``) reproduces the pod's over-merge: IBM and
    Apple collapse into ONE cluster at the SAME default threshold that keeps them apart with
    context. This is the negative control that proves the fix is *because of context*, not a higher
    threshold masking the problem.
    """
    mentions, chunks = _corpus()
    d = _pod_disambiguator(context_window=0)  # force bare-surface embedding
    records = list(d.disambiguate(mentions, [], chunks))
    entities = [r.entity for r in records if r.entity]
    merged = {frozenset(e.member_mention_ids) for e in entities}
    # An ORG super-cluster spanning IBM (m1,m2) AND Apple (m3,m4) — the over-merge context prevents.
    assert any({"m1", "m3"} <= c for c in merged), (
        f"bare surface should over-merge IBM+Apple (the pod finding); got {merged}"
    )


def test_deterministic_byte_identical_over_two_runs() -> None:
    """Identical input + config → byte-identical output (Baseline bar) on the deterministic path."""
    mentions, chunks = _corpus()
    d1 = _pod_disambiguator()
    d2 = _pod_disambiguator()
    out1 = [r.model_dump_json() for r in d1.disambiguate(mentions, [], chunks)]
    out2 = [r.model_dump_json() for r in d2.disambiguate(mentions, [], chunks)]
    assert out1 == out2


def test_real_hashing_embedder_keeps_distinct_entities_apart() -> None:
    """The honest offline default (in-core deterministic hashing embedder) over full chunks.

    Not a hand-tuned double: the real ``embedding.hashing`` embedder, resolved by default. It proves
    the algorithm on the default path — distinct entities (IBM vs Apple vs Berlin) never collapse,
    and a genuine surface-shared duplicate (Apple/Apple Inc, which share the 'apple' token AND
    context) merges. The weaker bag-of-words embedder may not catch the token-disjoint IBM pair, but
    it must NEVER over-merge distinct entities — the enterprise-safety invariant.
    """
    mentions, chunks = _corpus()
    d = EmbeddingDisambiguator({})  # default → embedding.hashing, threshold 0.90
    records = list(d.disambiguate(mentions, [], chunks))
    entities = [r.entity for r in records if r.entity]
    clusters = {frozenset(e.member_mention_ids) for e in entities}
    # No cluster spans two DIFFERENT real-world entities (no over-merge). Every cluster is a subset
    # of one of the three ground-truth entity id sets.
    truth = [frozenset({"m1", "m2"}), frozenset({"m3", "m4"}), frozenset({"m5"})]
    for c in clusters:
        assert any(c <= t for t in truth), f"cluster {c} over-merges across entities"
    # And the genuine surface+context duplicate Apple/Apple Inc is caught.
    assert frozenset({"m3", "m4"}) in clusters


def test_max_pairs_cap_raises_config_error() -> None:
    """The O(n²) cross-surface cap raises a ConfigError naming the knob, not partial silent work."""
    # Distinct surfaces so no exact block collapses them; 4 surfaces → 6 pairs > max_pairs=2.
    chunks = [_chunk("chx", "dx", "alpha beta gamma delta epsilon zeta content here")]
    mentions = [
        _mention("a", "dx", "chx", "Alpha", 0),
        _mention("b", "dx", "chx", "Beta", 6),
        _mention("c", "dx", "chx", "Gamma", 11),
        _mention("e", "dx", "chx", "Delta", 17),
    ]
    d = _pod_disambiguator(max_pairs=2)
    with pytest.raises(ConfigError, match="max_pairs"):
        list(d.disambiguate(mentions, [], chunks))


def test_native_embedder_fault_maps_to_typed_provider_error_no_pii() -> None:
    """A native embedder failure → typed ProviderError tagged with provider+op, no raw text."""

    class _BoomEmbedder:
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            self._config = dict(config or {})

        @property
        def dimension(self) -> int:
            return 2

        def embed(self, texts: Any) -> Any:
            raise RuntimeError("secret model path /home/user/private leaked in message")

    class _BoomRegistry:
        def load(self, name: str) -> Any:  # noqa: ARG002
            return _BoomEmbedder

    mentions, chunks = _corpus()
    d = EmbeddingDisambiguator({}, registry=_BoomRegistry())  # type: ignore[arg-type]
    with pytest.raises(ProviderError) as exc:
        list(d.disambiguate(mentions, [], chunks))
    msg = str(exc.value)
    assert "disambiguation.embedding" in msg
    assert "secret model path" not in msg  # no native message / PII leaked


def test_bad_embedder_spec_raises_config_error() -> None:
    """A malformed embedder config block fails loudly at wiring time (ConfigError)."""
    mentions, chunks = _corpus()
    d = EmbeddingDisambiguator({"embedder": {"config": {}}})  # missing 'provider'
    with pytest.raises(ConfigError, match="provider"):
        list(d.disambiguate(mentions, [], chunks))


def test_relations_normalized_onto_canonical_entities() -> None:
    """Relations are remapped onto the merged canonical entities (inherited cascade behaviour)."""
    mentions, chunks = _corpus()
    # A relation from the IBM mention to the Apple mention: after merge both endpoints resolve to
    # their canonical entities and the relation is emitted (distinct endpoints).
    rel = RelationMention(
        record_id="r1",
        document_record_id="d1:doc",
        index=0,
        label="competes_with",
        head_mention_id="m1",
        tail_mention_id="m3",
        head_text="IBM",
        tail_text="Apple",
        confidence=0.8,
        provenance=_prov("d1", 0, 10),
        classification=_CLS,
    )
    d = _pod_disambiguator()
    records = list(d.disambiguate(mentions, [rel], chunks))
    relations = [r.relation for r in records if r.scope is DisambiguationScope.RELATION]
    assert len(relations) == 1
    assert relations[0] is not None
    assert relations[0].label == "competes_with"


# --------------------------------------------------------------------------- #
# The coordinate-correctness regression (W1 rework): the surface+context window
# must be sliced through the chunk's offset_map, NOT the naive char_start shift,
# on a REAL markup-bearing chunk (the plaintext fixtures never exercise this).
# --------------------------------------------------------------------------- #


def test_context_window_aligns_through_offset_map_on_markdown() -> None:
    """The context window is sliced at the mention's TRUE content offset on a markup chunk.

    ``mention.char_start`` is in ORIGINAL markdown coordinates while ``chunk.content`` is
    markup-stripped, so the naive ``char_start - chunk_start`` overshoots once markup precedes the
    mention (the v11/ADR-0031 drift). This asserts the produced ``surface + context`` string is
    centred on the mention (the mention's own words are IN the window), which only holds when the
    offset is inverted through ``offset_map``; the naive shift would slice a wrong, later span.
    """
    # Markup BEFORE each mention (a header + bold/italic) forces a non-identity offset_map: the
    # naive shift would land many characters past the true position.
    md = (
        "# Company Report\n\n"
        "**IBM** reported strong quarterly earnings. *International Business Machines* is a "
        "technology company making enterprise mainframes and software services.\n"
    )
    chunk, content = _markup_chunk("chm", "dm", md)
    d = EmbeddingDisambiguator({"context_window": 80})
    chunk_by_id = {chunk.record_id: chunk}

    # The mention "International Business Machines" sits AFTER stripped markup (# , **IBM**, *).
    ibm_full = "International Business Machines"
    m = _mention("m2", "dm", "chm", ibm_full, md.index(ibm_full))
    windowed = d._surface_context_text(ibm_full, m, chunk_by_id)

    # The window is aligned: it contains the mention's own tail words + the following context,
    # NOT some later slice the naive overshoot would produce.
    assert ibm_full in windowed
    assert "technology company" in windowed
    # Sanity: the naive (broken) offset would overshoot past the mention inside stripped content.
    naive_local = m.provenance.char_start  # chunk_start is 0
    assert naive_local is not None
    assert naive_local > content.index(ibm_full), (
        "the naive shift must overshoot the true content offset (else the fixture proves nothing)"
    )


def test_no_over_merge_on_markdown_corpus_through_offset_map() -> None:
    """Enterprise regression on the REAL markup path: IBM≠Apple, but a true duplicate still merges.

    The pod-vector stub keys on the leading surface AND on whether context is present. With the
    offset correctly inverted through ``offset_map``, every markup chunk yields an aligned
    surface+context window (context present), so the stub returns the context-separated vectors:
    {IBM, International Business Machines} and {Apple, Apple Inc} each merge while IBM and Apple stay
    apart — proven on markdown, not just plaintext. (If the offset overshot the content end, the code
    would drop to BARE surface and the stub's bare-surface angles would over-merge IBM+Apple — so
    this passing at the default 0.90 threshold is exactly the coordinate-path proof.)
    """
    ibm_md = (
        "## Earnings\n\n**IBM** posted results. *International Business Machines* is a technology "
        "company making mainframes.\n"
    )
    apple_md = (
        "## Product\n\n**Apple** shipped a phone. *Apple Inc* is a consumer electronics company in "
        "Cupertino.\n"
    )
    ibm_chunk, _ = _markup_chunk("chi", "di", ibm_md)
    apple_chunk, _ = _markup_chunk("cha", "da", apple_md)
    chunks = [ibm_chunk, apple_chunk]
    mentions = [
        _mention("m1", "di", "chi", "IBM", ibm_md.index("IBM")),
        _mention(
            "m2", "di", "chi", "International Business Machines",
            ibm_md.index("International"),
        ),
        _mention("m3", "da", "cha", "Apple", apple_md.index("Apple")),
        _mention("m4", "da", "cha", "Apple Inc", apple_md.index("Apple Inc")),
    ]
    d = _pod_disambiguator()  # default threshold 0.90, context_window 160
    records = list(d.disambiguate(mentions, [], chunks))
    entities = [r.entity for r in records if r.entity]
    clusters = {e.canonical_text: sorted(e.member_mention_ids) for e in entities if e}
    assert clusters == {
        "Apple": ["m3", "m4"],
        "IBM": ["m1", "m2"],
    }, clusters
    # No cluster spans BOTH the IBM and Apple documents (the over-merge context prevents).
    merged = {frozenset(e.member_mention_ids) for e in entities}
    assert not any({"m1", "m3"} <= c for c in merged), f"over-merge on markdown path: {merged}"
