"""Shared offline fixtures + faithful fakes for the GLinker Disambiguator tests.

The heavy gliner/torch stack is NOT installed in the CPU-first dev env (ADR-0016), so the tests
exercise the ported ALGORITHM through:

* :class:`FakeGLinkerBackend` — a faithful in-memory stand-in for the neural
  :class:`~latence_disambig_glinker.backends.GLinkerBackend` that RETURNS/REJECTS exactly the way
  the real biencoder does (a mention whose surface is in its ``link_map`` links to the mapped KB
  entry; an unmapped surface returns ``None`` -> the direct-KB fallback), so a green test is not a
  false green.
* :class:`FakeLabelEncoder` — a deterministic STS label encoder (crafted unit vectors) standing in
  for the resolved framework Embedder (Granite r2), so the resolver's embedding-by-label merge rung
  runs offline + deterministically.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest
from latence_core.contracts import (
    Classification,
    EntityMention,
    PageMap,
    PageSlice,
    Provenance,
    RelationMention,
)
from latence_disambig_glinker.backends import InferenceBackend, LinkedEntity

if TYPE_CHECKING:
    from collections.abc import Iterator

    from latence_core.contracts import ChunkRecord

_CLS = Classification(language="en", category="news", sensitivity="internal")


@pytest.fixture(autouse=True)
def _reset_fake_backend_state() -> Iterator[None]:
    """Reset the FakeGLinkerBackend class-level link tables before AND after every test.

    ``link_map`` / ``context_rules`` are class attributes (the tests set them declaratively), so a
    stray value must never leak across tests — otherwise a later test could silently link on a
    previous test's rule (a false green). Runs first (conftest scope), so a test-module fixture or
    body that then sets the tables wins.
    """
    FakeGLinkerBackend.link_map = {}
    FakeGLinkerBackend.context_rules = []
    yield
    FakeGLinkerBackend.link_map = {}
    FakeGLinkerBackend.context_rules = []


# --------------------------------------------------------------------------- #
# Framework contract fixtures.
# --------------------------------------------------------------------------- #


def prov(doc: str, cs: int, ce: int) -> Provenance:
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


def chunk(rid: str, doc: str, content: str) -> ChunkRecord:
    from latence_core.contracts import ChunkRecord

    return ChunkRecord(
        record_id=rid,
        document_record_id=f"{doc}:doc",
        index=0,
        content=content,
        token_count=20,
        provenance=prov(doc, 0, len(content)),
        classification=_CLS,
        # v19: the chunk carries its own spans; a one-page fixture, so exactly that page.
        page_slice=PageSlice.for_span(
            PageMap.from_page_texts([content]).page_map, 0, len(content)
        ),
    )


def mention(
    rid: str, doc: str, chunk_id: str, text: str, start: int, label: str = "organization"
) -> EntityMention:
    return EntityMention(
        record_id=rid,
        document_record_id=f"{doc}:doc",
        chunk_record_id=chunk_id,
        index=0,
        label=label,
        text=text,
        confidence=0.9,
        provenance=prov(doc, start, start + len(text)),
        classification=_CLS,
    )


def relation(
    rid: str, doc: str, label: str, head: EntityMention, tail: EntityMention
) -> RelationMention:
    return RelationMention(
        record_id=rid,
        document_record_id=f"{doc}:doc",
        index=0,
        label=label,
        head_mention_id=head.record_id,
        tail_mention_id=tail.record_id,
        head_text=head.text,
        tail_text=tail.text,
        confidence=0.8,
        provenance=prov(doc, 0, 10),
        classification=_CLS,
    )


# --------------------------------------------------------------------------- #
# Faithful fake neural backend.
# --------------------------------------------------------------------------- #


class FakeGLinkerBackend(InferenceBackend):
    """Faithful stand-in for the neural GLinker backend (returns/rejects like the real biencoder).

    Mirrors the real backend's inputs: the biencoder reads the mention's ``context`` (the real
    :meth:`GLinkerBackend.link_mentions` feeds ``m.get("context", m["text"])`` as the encoded text
    and ``m["text"]`` as the span), so this fake ALSO reads ``m["context"]`` — a link decision is
    genuinely context-conditional, and a regression that stopped passing context to the backend would
    change the fake's output and fail the context tests (anti-false-green, C-F2/C-F3/A-F5).

    Two linking channels, context first:

    * ``context_rules`` — ``(surface_lower, context_needle, (entity_id, label))``. When the mention's
      surface matches and ``context_needle`` occurs in the mention's context, link to that entry. This
      models the biencoder disambiguating a **polysemous surface by its surrounding text** (the same
      surface in different contexts links to different entities).
    * ``link_map`` — ``surface_lower -> (entity_id, label)``, the context-independent fallback (e.g. an
      unambiguous acronym mention linked to its full-name entry).

    A surface matched by neither returns ``None`` -> the mention flows to the direct-KB fallback,
    exactly as the real backend does. Set the maps before monkeypatching.
    """

    LINKER_MODEL = "fake/gliner-linker-large"
    RERANKER_MODEL = "fake/gliner-linker-rerank"
    link_map: dict[str, tuple[str, str]] = {}
    context_rules: list[tuple[str, str, tuple[str, str]]] = []

    def __init__(self, **_kwargs: Any) -> None:
        self._warmed = False

    @property
    def name(self) -> str:
        return "fake_glinker"

    def warmup(self, entity_kb: list[dict[str, Any]]) -> None:
        self._warmed = True

    def link_mentions(
        self,
        mentions: list[dict[str, Any]],
        entity_kb: list[dict[str, Any]],
        threshold: float = 0.3,
        batch_size: int = 32,
    ) -> list[LinkedEntity | None]:
        results: list[LinkedEntity | None] = []
        for m in mentions:
            surface = str(m["text"]).strip().lower()
            # READ the context exactly as the real backend does (m.get("context", m["text"])).
            context = str(m.get("context", m["text"]))
            mapped = self._link_one(surface, context)
            if mapped is None:
                results.append(None)
                continue
            entity_id, label = mapped
            results.append(
                LinkedEntity(
                    entity_id=entity_id,
                    label=label,
                    score=0.95,
                    entity_type=label,
                    source="fake_glinker",
                )
            )
        return results

    def _link_one(self, surface: str, context: str) -> tuple[str, str] | None:
        """Context-conditional link decision (context_rules first, then the surface link_map)."""
        for rule_surface, needle, target in self.context_rules:
            if rule_surface == surface and needle in context:
                return target
        return self.link_map.get(surface)


# --------------------------------------------------------------------------- #
# Faithful fake STS label encoder.
# --------------------------------------------------------------------------- #


class FakeLabelEncoder:
    """Deterministic STS encoder: 2-D unit vectors from a per-keyword angle map (cosine == dot).

    ``angles`` maps a lowercased substring -> angle (radians). A text's angle is the first keyword
    it contains, else a stable hash-derived angle (near-orthogonal to the named ones). So two names
    the test wants to merge get near-identical angles (cos ≈ 1) and distinct names get wide gaps.
    """

    def __init__(self, angles: dict[str, float]) -> None:
        self._angles = angles

    def _angle(self, text: str) -> float:
        low = text.lower()
        for kw, ang in self._angles.items():
            if kw in low:
                return ang
        # Stable, spread-out fallback angle for an unnamed text (near-orthogonal to named anchors).
        return 0.5 + (hash(low) % 997) / 997.0 * math.pi

    def embed_labels(self, texts: list[str]) -> list[list[float]]:
        return [[math.cos(self._angle(t)), math.sin(self._angle(t))] for t in texts]
