"""Fused Entity + Relation Extraction Provider over gliner-relex (CPU path, ADR-0013).

The S5 *fused* reference Provider: ``knowledgator/gliner-relex-multi-v1.0`` does
joint zero-shot NER **and** Relation Extraction in one pass, so a single Provider
fulfils BOTH the Entity Extraction (S4) and Relation Extraction (S5) Capabilities
(ADR-0013: the Pipeline is not forced to one Provider per Stage). It wraps the
``gliner`` library (plain ``transformers``/ONNX, no vLLM), so it runs in-process on
a laptop CPU (ADR-0007) — first contact is ``pip install latence-relation-gliner``
→ fused NER+RE, no CUDA.

Licensing (ADR-0012, verified 2026-07-06):
    * ``gliner`` library (PyPI + github.com/urchade/GLiNER): **Apache-2.0** (code).
    * Default checkpoint ``knowledgator/gliner-relex-multi-v1.0``: **Apache-2.0**
      (weights, per the model card).
    Both weights AND code are permissive; nothing is downloaded unless a Pipeline
    actually configures this Provider.

Zero-shot (S4/S5 seam): both the entity label set and the relation label set come
entirely from Stage ``config`` — the same checkpoint tags ``person``/``organization``
and relates them with ``works_for`` for one Pipeline, and tags ``gene``/``drug`` and
relates them with ``targets`` for another, with no retraining and no labels hardcoded
in core.

Both the emitted :class:`~latence_core.contracts.EntityMention`s and
:class:`~latence_core.contracts.RelationMention`s carry Provenance whose char span is
resolved from the model's chunk-local offsets into the parent document's assembled
markdown through the chunk's ``offset_map`` (the stripped-``content``→original offset
map carried since v11) via :class:`~latence_core.offsetmap.OffsetIndex` — so a mention's
offset is EXACT even when markdown markup was stripped before it (``char_start + local``
is only a lower bound; ADR-0022 corrected, ADR-0031; a pre-v11 chunk degrades to that
shift). A relation's covering span is the min/max of its resolved endpoints, so it
inherits the correction. Each page span is re-resolved through the parent's page map
(carried on the chunk) via :class:`~latence_core.pagemap.PageOffsetIndex` — the S4/S5
round-trip. Because a fused
Stage emits both carriers together, the Runner threads mentions AND relations
downstream from one Stage without running a separate Entity-Extraction Stage (no
double-extraction).

The ``gliner`` import is deferred to first use so importing this module (for Provider
discovery) never requires torch, and a missing optional dep raises a clear error only
when the Provider is run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.capability import FusedExtraction
from latence_core.contracts import EntityMention, RelationMention
from latence_core.offsetmap import OffsetIndex
from latence_core.pagemap import PageOffsetIndex
from latence_core.providers.profile import ProviderProfile

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import ChunkRecord, Provenance

# Apache-2.0 weights (verified 2026-07-06). Multilingual fused NER+RE.
_DEFAULT_MODEL = "knowledgator/gliner-relex-multi-v1.0"


class GlinerRelexExtractor:
    """Fused zero-shot Entity+Relation Extractor backed by a gliner-relex checkpoint.

    Config:
        labels: list[str]  — REQUIRED. The zero-shot entity types to extract (e.g.
            ``["person", "organization"]``). GLiNER works best with lower/title case.
        relation_labels: list[str]  — REQUIRED. The zero-shot relation types to
            extract between the entities (e.g. ``["works_for", "located_in"]``).
        model: str  — the checkpoint id (default ``knowledgator/gliner-relex-multi-v1.0``,
            Apache-2.0).
        threshold: float  — score threshold; entities/relations below it are dropped
            (default 0.5).
        min_chars: int  — a chunk whose stripped length is below this is skipped
            without invoking the model (default 1).

    Fulfils BOTH the EntityExtractor and the FusedEntityRelationExtractor seam: it
    also exposes ``extract`` so it can stand in for a plain Entity-Extraction Stage
    when a Pipeline wants only its NER half. The model is loaded lazily and cached on
    first use, so constructing the Provider never downloads weights.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2) for the default gliner-relex checkpoint.

        ``compute="either"``: gliner-relex runs in-process on a laptop CPU (ADR-0007 — first
        contact is ``pip install`` → learned fused NER+RE, no CUDA) and accelerates on a GPU
        when present, so the device seam routes it rather than skipping it on a CPU-only host.

        License **verified** per ADR-0012 (recorded 2026-07-06): the ``gliner`` library is
        Apache-2.0 (code) and the default checkpoint ``knowledgator/gliner-relex-multi-v1.0``
        is Apache-2.0 (weights) — both permissive, so ``license_verified=True``.

        ``deterministic=False``: learned float inference is not guaranteed byte-identical
        across hardware/library versions, so the profile declares it non-deterministic rather
        than overclaim reproducibility. ``memory_mb`` is the approximate resident footprint of
        the checkpoint + its transformers stack (declared, not measured — descriptor-only).
        """
        return ProviderProfile(
            compute="either",
            memory_mb=2048,
            model_id=_DEFAULT_MODEL,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_weights="Apache-2.0",
            license_source=(
                "https://huggingface.co/knowledgator/gliner-relex-multi-v1.0 "
                "(weights, Apache-2.0); "
                "https://github.com/urchade/GLiNER/blob/main/LICENSE (library, Apache-2.0)"
            ),
            license_verified_on="2026-07-06",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        labels = config.get("labels")
        if not labels or not isinstance(labels, list):
            msg = (
                "GlinerRelexExtractor requires a non-empty 'labels' list in its config "
                "(zero-shot NER: the entity label set is supplied per Pipeline)."
            )
            raise ValueError(msg)
        relation_labels = config.get("relation_labels")
        if not relation_labels or not isinstance(relation_labels, list):
            msg = (
                "GlinerRelexExtractor requires a non-empty 'relation_labels' list in its "
                "config (zero-shot RE: the relation label set is supplied per Pipeline)."
            )
            raise ValueError(msg)
        self._labels = [str(label) for label in labels]
        self._relation_labels = [str(label) for label in relation_labels]
        self._model_id = str(config.get("model", _DEFAULT_MODEL))
        self._threshold = float(config.get("threshold", 0.5))
        self._min_chars = int(config.get("min_chars", 1))
        self._model: Any = None

    # -- lazy model load ----------------------------------------------------

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from gliner import GLiNER
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'gliner' package is required to run GlinerRelexExtractor. "
                "Install it with: pip install latence-relation-gliner"
            )
            raise ImportError(msg) from exc
        self._model = GLiNER.from_pretrained(self._model_id)
        return self._model

    # -- fused extraction ---------------------------------------------------

    def extract_fused(self, chunks: Iterable[ChunkRecord]) -> FusedExtraction:
        """One pass over the chunks yielding both mentions and relations."""
        model = None
        mentions: list[EntityMention] = []
        relations: list[RelationMention] = []
        for chunk in chunks:
            if len(chunk.content.strip()) < self._min_chars:
                continue
            if model is None:
                model = self._load_model()
            chunk_mentions, chunk_relations = self._extract_one(model, chunk)
            mentions.extend(chunk_mentions)
            relations.extend(chunk_relations)
        return FusedExtraction(mentions=mentions, relations=relations)

    # -- plain Entity-Extraction half (so it also fits the EntityExtractor seam) --

    def extract(self, chunks: Iterable[ChunkRecord]) -> list[EntityMention]:
        """The NER half alone (for a Pipeline that wants only entities from this model)."""
        return self.extract_fused(chunks).mentions

    def _extract_one(
        self, model: Any, chunk: ChunkRecord
    ) -> tuple[list[EntityMention], list[RelationMention]]:
        # The REAL gliner-relex API (introspected + run on the pod, W1-relex FIX): the
        # checkpoint exposes ``predict_relations`` (NOT ``predict_with_relations`` — that
        # method never existed and was the false-green trap) returning a **TUPLE**
        # ``(entities, relations)``, NOT a dict. Entity dicts carry chunk-local
        # 'start'/'end', 'text', 'label', 'score' (identical shape to ``predict_entities``).
        # Relation dicts carry 'head'/'tail' sub-dicts ('start'/'end'/'text'/'type'/
        # 'entity_idx') plus 'relation' + 'score'. We resolve the relations back to the
        # EntityMentions we stamp by their (start, end) span.
        raw_entities, raw_relations = model.predict_relations(
            chunk.content,
            self._labels,
            self._relation_labels,
            threshold=self._threshold,
        )
        raw_entities = list(raw_entities)
        raw_relations = list(raw_relations)
        raw_entities.sort(key=lambda e: (int(e["start"]), str(e["label"])))

        # Resolve chunk-local model offsets to TRUE original offsets through the
        # chunk's stripped→original offset map (markup stripping makes char_start+local
        # only a lower bound); a pre-v11 chunk with no map degrades to that shift.
        offset_index = OffsetIndex(chunk.offset_map) if chunk.offset_map is not None else None
        chunk_char_start = chunk.provenance.char_start or 0
        page_index = PageOffsetIndex(chunk.page_map) if chunk.page_map is not None else None

        mentions: list[EntityMention] = []
        # Map the model's chunk-local (start, end) to the stamped mention, so a
        # relation referencing those offsets resolves to a real mention record_id.
        by_span: dict[tuple[int, int], EntityMention] = {}
        for i, ent in enumerate(raw_entities):
            local_start = int(ent["start"])
            local_end = int(ent["end"])
            if offset_index is not None:
                doc_start, doc_end = offset_index.resolve_span(local_start, local_end)
            else:
                doc_start, doc_end = chunk_char_start + local_start, chunk_char_start + local_end
            provenance = _mention_provenance(
                chunk.provenance,
                doc_start,
                doc_end,
                page_index,
            )
            mention = EntityMention(
                record_id=f"{chunk.record_id}#entity-{i}",
                document_record_id=chunk.document_record_id,
                chunk_record_id=chunk.record_id,
                index=i,
                label=str(ent["label"]),
                text=chunk.content[local_start:local_end],
                confidence=min(max(float(ent["score"]), 0.0), 1.0),
                provenance=provenance,
                classification=chunk.classification,
            )
            mentions.append(mention)
            by_span[(local_start, local_end)] = mention

        relations: list[RelationMention] = []
        emitted = 0
        for rel in sorted(raw_relations, key=_relation_sort_key):
            head = _resolve_endpoint(rel, "head", by_span)
            tail = _resolve_endpoint(rel, "tail", by_span)
            label = str(rel.get("relation", rel.get("label", "")))
            if head is None or tail is None or head.record_id == tail.record_id or not label:
                continue
            span_start = min(head.provenance.char_start or 0, tail.provenance.char_start or 0)
            span_end = max(head.provenance.char_end or 0, tail.provenance.char_end or 0)
            rel_prov = _relation_provenance(head, tail, span_start, span_end, page_index)
            relations.append(
                RelationMention(
                    record_id=f"{chunk.record_id}#relation-{emitted}",
                    document_record_id=chunk.document_record_id,
                    index=emitted,
                    label=label,
                    head_mention_id=head.record_id,
                    tail_mention_id=tail.record_id,
                    head_text=head.text,
                    tail_text=tail.text,
                    confidence=min(max(float(rel.get("score", 1.0)), 0.0), 1.0),
                    provenance=rel_prov,
                    classification=chunk.classification,
                )
            )
            emitted += 1
        return mentions, relations


def _relation_sort_key(rel: dict[str, Any]) -> tuple[int, int, str]:
    head = rel.get("head", {})
    start = int(head.get("start", 0)) if isinstance(head, dict) else 0
    end = int(head.get("end", 0)) if isinstance(head, dict) else 0
    label = str(rel.get("relation", rel.get("label", "")))
    return (start, end, label)


def _resolve_endpoint(
    rel: dict[str, Any], role: str, by_span: dict[tuple[int, int], EntityMention]
) -> EntityMention | None:
    endpoint = rel.get(role)
    if not isinstance(endpoint, dict):
        return None
    try:
        key = (int(endpoint["start"]), int(endpoint["end"]))
    except (KeyError, TypeError, ValueError):
        return None
    return by_span.get(key)


def _mention_provenance(
    chunk_prov: Provenance,
    doc_char_start: int,
    doc_char_end: int,
    page_index: PageOffsetIndex | None,
) -> Provenance:
    update: dict[str, int] = {"char_start": doc_char_start, "char_end": doc_char_end}
    if page_index is not None:
        update["page_start"] = page_index.page_for_offset(doc_char_start)
        update["page_end"] = page_index.page_for_offset(max(doc_char_end - 1, doc_char_start))
    return chunk_prov.model_copy(update=update)


def _relation_provenance(
    head: EntityMention,
    tail: EntityMention,
    span_start: int,
    span_end: int,
    page_index: PageOffsetIndex | None,
) -> Provenance:
    base = head.provenance
    update: dict[str, int] = {"char_start": span_start, "char_end": span_end}
    if page_index is not None:
        update["page_start"] = page_index.page_for_offset(span_start)
        update["page_end"] = page_index.page_for_offset(max(span_end - 1, span_start))
    else:
        hp, tp = head.provenance, tail.provenance
        starts = [p for p in (hp.page_start, tp.page_start) if p is not None]
        ends = [p for p in (hp.page_end, tp.page_end) if p is not None]
        if starts:
            update["page_start"] = min(starts)
        if ends:
            update["page_end"] = max(ends)
    return base.model_copy(update=update)
