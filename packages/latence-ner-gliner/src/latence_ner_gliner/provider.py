"""Learned zero-shot EntityExtractor Provider over a GLiNER checkpoint (CPU path).

The S4 Entity Extraction *learned* reference Provider (ADR-0004; ADR-0007 CPU-first;
ADR-0016 heavy-dep isolation — torch/gliner ship HERE, not in the thin core). It
wraps a permissively-licensed GLiNER checkpoint via the ``gliner`` library (plain
``transformers``/ONNX, no vLLM), so it runs in-process on a laptop CPU — first
contact is ``pip install latence-ner-gliner`` → learned NER, no CUDA.

Licensing (ADR-0012, verified 2026-07-06):
    * ``gliner`` library (PyPI + github.com/urchade/GLiNER): **Apache-2.0** (code).
    * Default checkpoint ``urchade/gliner_multi-v2.1``: **Apache-2.0** (weights).
      (v2.1 relicensed from the earlier CC-BY-NC-4.0 to Apache-2.0; earlier v0/v1
      checkpoints are CC-BY-NC and are NOT the default — a commercial-safe default.)
    Both weights AND code are permissive; nothing is downloaded unless a Pipeline
    actually configures this Provider.

Zero-shot (S4 AC): the label set comes entirely from Stage ``config`` — GLiNER is a
zero-shot NER model, so the same checkpoint tags ``PERSON``/``ORG`` for one Pipeline
and ``GENE``/``DRUG`` for another with no retraining, and core hardcodes no labels.

Each emitted :class:`~latence_core.contracts.EntityMention` carries its label,
GLiNER's confidence, and the mention's char span **in the parent document's
assembled markdown** — the model returns chunk-local offsets, which are shifted by
the chunk's own ``provenance.char_start`` (the chunk records its span in the parent
document, S3) so a mention resolves back to the exact source offsets and, through
the parent's page map (carried on the chunk) via
:class:`~latence_core.pagemap.PageOffsetIndex`, to its original source page(s) (the
S4 round-trip). The page span is re-resolved from the mention's *own* document
offset — a mention on a later page of a multi-page chunk cites that page, not the
chunk's first page — degrading to the chunk's inherited range only when no page map
is present. An empty/near-empty chunk yields no mentions (S4 AC).

The ``gliner`` import is deferred to first use so importing this module (for
Provider discovery) never requires torch to be installed, and so a missing optional
dep raises a clear, actionable error only when the Provider is actually run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.contracts import EntityMention
from latence_core.pagemap import PageOffsetIndex

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord, Provenance

# Apache-2.0 weights (verified). A commercial-safe zero-shot multilingual default.
_DEFAULT_MODEL = "urchade/gliner_multi-v2.1"


class GlinerEntityExtractor:
    """Learned zero-shot EntityExtractor Provider backed by a GLiNER checkpoint.

    Config:
        labels: list[str]  — REQUIRED. The zero-shot entity types to extract (e.g.
            ``["person", "organization", "location"]``). GLiNER works best with
            lower/title-case labels. Nothing is hardcoded in core (S4 AC).
        model: str  — the GLiNER checkpoint id (default
            ``urchade/gliner_multi-v2.1``, Apache-2.0). A CC-BY-NC checkpoint is a
            deliberate opt-in, never the default.
        threshold: float  — GLiNER's score threshold; mentions below it are dropped
            (default 0.5).
        min_chars: int  — a chunk whose stripped length is below this is treated as
            near-empty and skipped without invoking the model (default 1).

    The model is loaded lazily and cached on first ``extract`` call, so constructing
    the Provider (which the Runner does for every Stage) never downloads weights.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        labels = config.get("labels")
        if not labels or not isinstance(labels, list):
            msg = (
                "GlinerEntityExtractor requires a non-empty 'labels' list in its "
                "config (zero-shot NER: the label set is supplied per Pipeline)."
            )
            raise ValueError(msg)
        self._labels = [str(label) for label in labels]
        self._model_id = str(config.get("model", _DEFAULT_MODEL))
        self._threshold = float(config.get("threshold", 0.5))
        self._min_chars = int(config.get("min_chars", 1))
        self._model: Any = None

    # -- lazy model load ----------------------------------------------------

    def _load_model(self) -> Any:
        """Import ``gliner`` and load the checkpoint on first use (cached).

        Deferred so Provider discovery (importing this module) never needs torch,
        and a missing optional dependency surfaces as a clear install hint only when
        the Provider is actually run — not at registry enumeration time.
        """
        if self._model is not None:
            return self._model
        try:
            from gliner import GLiNER
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'gliner' package is required to run GlinerEntityExtractor. "
                "Install it with: pip install latence-ner-gliner"
            )
            raise ImportError(msg) from exc
        self._model = GLiNER.from_pretrained(self._model_id)
        return self._model

    # -- extraction ---------------------------------------------------------

    def extract(self, chunks: Iterable[ChunkRecord]) -> Iterator[EntityMention]:
        model = None
        for chunk in chunks:
            if len(chunk.content.strip()) < self._min_chars:
                # Empty / near-empty chunk: no mentions, no model call (S4 AC).
                continue
            if model is None:
                model = self._load_model()
            yield from self._extract_one(model, chunk)

    def _extract_one(self, model: Any, chunk: ChunkRecord) -> Iterator[EntityMention]:
        # GLiNER returns dicts with chunk-local 'start'/'end', 'label', 'score'.
        raw: list[dict[str, Any]] = model.predict_entities(
            chunk.content, self._labels, threshold=self._threshold
        )
        # Deterministic ordering: by chunk-local start, then label.
        raw.sort(key=lambda e: (int(e["start"]), str(e["label"])))
        chunk_char_start = chunk.provenance.char_start or 0
        # Re-resolve each mention's OWN page from the parent's page map (carried on
        # the chunk), not just inherit the chunk's page range. Built once per chunk.
        page_index = PageOffsetIndex(chunk.page_map) if chunk.page_map is not None else None
        for i, ent in enumerate(raw):
            local_start = int(ent["start"])
            local_end = int(ent["end"])
            provenance = self._mention_provenance(
                chunk.provenance,
                doc_char_start=chunk_char_start + local_start,
                doc_char_end=chunk_char_start + local_end,
                page_index=page_index,
            )
            yield EntityMention(
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

    @staticmethod
    def _mention_provenance(
        chunk_prov: Provenance,
        doc_char_start: int,
        doc_char_end: int,
        page_index: PageOffsetIndex | None,
    ) -> Provenance:
        """Provenance for a mention: chunk lineage + the mention's own span AND page.

        The char span is narrowed to the mention's own ``[start, end)`` in the parent
        document's assembled markdown. The page span is re-resolved from that offset
        through the parent's page map (``page_index``) so a mention on a later page of
        a multi-page chunk cites its own page (``page_end`` from the last included
        character so a boundary-straddling mention records both pages). Absent a page
        map, the mention degrades to inheriting the chunk's page range.
        """
        update: dict[str, int] = {"char_start": doc_char_start, "char_end": doc_char_end}
        if page_index is not None:
            update["page_start"] = page_index.page_for_offset(doc_char_start)
            update["page_end"] = page_index.page_for_offset(max(doc_char_end - 1, doc_char_start))
        return chunk_prov.model_copy(update=update)
