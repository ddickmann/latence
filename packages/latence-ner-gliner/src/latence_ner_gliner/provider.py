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
assembled markdown** — the model returns chunk-local offsets, which are resolved to
their TRUE original offsets through the chunk's stripped-``content``→original
``offset_map`` (carried since v11) via :class:`~latence_core.offsetmap.OffsetIndex`,
so a mention resolves back to its EXACT source offset even when markdown markup was
stripped before it within the chunk. (Chunk ``content`` is markup-stripped, so the
naive ``char_start + local`` shift is only a lower bound — ADR-0022 corrected,
ADR-0031.) A pre-v11 chunk with no offset map degrades to that shift. From the
resolved offset the page span is re-resolved through the parent's page map (carried
on the chunk) via :class:`~latence_core.pagemap.PageOffsetIndex` to its original
source page(s) (the S4 round-trip) — a mention on a later page of a multi-page chunk
cites that page, not the chunk's first page, degrading to the chunk's inherited
range only when no page map is present. An empty/near-empty chunk yields no mentions
(S4 AC).

The ``gliner`` import is deferred to first use so importing this module (for
Provider discovery) never requires torch to be installed, and so a missing optional
dep raises a clear, actionable error only when the Provider is actually run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.contracts import EntityMention
from latence_core.induced import LabelSlot, effective_labels
from latence_core.offsetmap import OffsetIndex
from latence_core.pagemap import PageOffsetIndex
from latence_core.providers.adapter import validate_batch_size
from latence_core.providers.profile import ProviderProfile

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord, Provenance

# Apache-2.0 weights (verified). A commercial-safe zero-shot multilingual default.
_DEFAULT_MODEL = "urchade/gliner_multi-v2.1"

# The default number of chunks whose text shares one ``batch_predict_entities`` forward pass
# (W7). A sensible GPU value (the P8 autotuner overrides per-device); purely memory-shaping —
# outputs are record-for-record identical to a per-chunk loop for any positive batch size
# (ADR-0033: the in-flight tensor is bounded to ``batch_size``, never the whole corpus).
_DEFAULT_BATCH_SIZE = 16


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
        batch_size: int  — how many chunks share ONE ``batch_predict_entities`` forward
            pass (default 16; the P8 autotuner overrides per-device). Purely memory-shaping:
            the emitted mentions (offsets/labels/order) are record-for-record identical for
            any positive batch size (W7). Must be ``>= 1``.

    The model is loaded lazily and cached on first ``extract`` call, so constructing
    the Provider (which the Runner does for every Stage) never downloads weights.

    Batching (W7, ADR-0036 §1 / ADR-0033): these are ENCODERS, so the GPU is under-fed by a
    per-chunk ``predict_entities`` loop. This Provider accumulates up to ``batch_size`` chunks
    that share the same effective label set and issues ONE ``model.batch_predict_entities(
    list_of_texts, labels, threshold=…)`` call per group (``ceil(N/batch_size)`` calls, not N),
    then maps each result list back to its chunk **by index** — the real batch dimension +
    SDPA/bf16 win, no server (vLLM is irrelevant for an encoder). The per-chunk offset/page
    resolution and contract stamping below are UNCHANGED; only how many chunks share a forward
    pass changes. The in-flight batch is bounded to ``batch_size`` (never the whole corpus), so
    the streaming Runner (ADR-0033) stays memory-bounded.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2) for the default GLiNER checkpoint.

        ``compute="either"``: GLiNER runs in-process on a laptop CPU (the ADR-0007 CPU
        path — first contact is ``pip install`` → learned NER, no CUDA) and accelerates on
        a GPU when one is present, so the device seam routes it to ``auto``/``cpu``/``cuda``
        rather than skipping it on a CPU-only host.

        License is **verified** per ADR-0012, recorded evidence-bearingly in the structured
        fields (H-B1 — the claim is inseparable from its proof, not a bare bool): weights and
        code are recorded **separately** (``license_code="Apache-2.0"`` for the ``gliner``
        library; ``license_weights="Apache-2.0"`` for the default checkpoint
        ``urchade/gliner_multi-v2.1``), with a ``license_source`` citing the HF model-card +
        the GitHub LICENSE where a human checked, and ``license_verified_on="2026-07-06"``.
        (v2.1 was relicensed from the earlier CC-BY-NC-4.0 to Apache-2.0; earlier CC-BY-NC
        checkpoints are a deliberate config opt-in, never this default — the profile describes
        the default.) Both weights AND code permissive, so ``license="Apache-2.0"`` with
        ``license_verified=True``.

        ``deterministic=False``: learned float inference is not guaranteed byte-identical
        across hardware/library versions (unlike the pure-code gazetteer), so the profile
        declares it non-deterministic rather than overclaim reproducibility. ``memory_mb``
        is the approximate resident footprint of the multi-v2.1 checkpoint + its transformers
        stack (declared, not measured here — P3-F1 is descriptor-only). ``model_id`` is the
        default checkpoint; a Pipeline that overrides ``config['model']`` changes what runs,
        but the declared default descriptor is what the registry/bake-off reads.
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
                "https://huggingface.co/urchade/gliner_multi-v2.1 (weights, Apache-2.0); "
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
                "GlinerEntityExtractor requires a non-empty 'labels' list in its "
                "config (zero-shot NER: the label set is supplied per Pipeline)."
            )
            raise ValueError(msg)
        self._labels = [str(label) for label in labels]
        self._model_id = str(config.get("model", _DEFAULT_MODEL))
        self._threshold = float(config.get("threshold", 0.5))
        self._min_chars = int(config.get("min_chars", 1))
        self._batch_size = validate_batch_size(
            config.get("batch_size", _DEFAULT_BATCH_SIZE),
            provider_name="entity.gliner",
        )
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
        """Batch chunks through ``batch_predict_entities`` and emit mentions in input order (W7).

        Chunks whose stripped length is below ``min_chars`` are skipped without a model call
        (S4 AC). The remaining chunks are accumulated up to ``batch_size`` — grouped so a single
        batch shares ONE effective label set (the W4 induced-label seam is per-document, so
        consecutive same-document chunks batch together; a label-set change flushes the batch,
        never a mis-labelled call) — and each group is one ``model.batch_predict_entities(texts,
        labels, threshold=…)`` forward pass. Results map back to their chunk **by index**. The
        overall emission order is input (chunk) order, and each chunk's mentions are byte-identical
        to the per-chunk ``predict_entities`` path (the equivalence invariant, W7). The in-flight
        batch is bounded to ``batch_size`` — the whole corpus is never accumulated (ADR-0033).
        """
        model = None
        batch: list[ChunkRecord] = []
        batch_labels: tuple[str, ...] | None = None

        def flush() -> Iterator[EntityMention]:
            nonlocal model, batch, batch_labels
            if not batch:
                return
            if model is None:
                model = self._load_model()
            texts = [c.content for c in batch]
            labels = list(batch_labels or ())
            # ONE forward pass over the whole group; results[i] is the entity list for texts[i].
            results: list[list[dict[str, Any]]] = model.batch_predict_entities(
                texts, labels, threshold=self._threshold
            )
            for chunk, raw in zip(batch, results, strict=True):
                yield from self._mentions_for(chunk, list(raw))
            batch = []
            batch_labels = None

        for chunk in chunks:
            if len(chunk.content.strip()) < self._min_chars:
                # Empty / near-empty chunk: no mentions, no model call (S4 AC).
                continue
            # W4 label-source seam (ADR-0038): the effective label set is the config labels UNIONed
            # with any entity types the OPTIONAL schema_induction Stage induced for this chunk's
            # document, deduped + order-stable. Absent (the default, every existing stack) this is
            # exactly ``self._labels``. GLiNER's batch call takes ONE label set, so a chunk whose
            # effective labels differ from the accumulating batch flushes it first.
            labels = tuple(effective_labels(self._labels, chunk.induced_labels, LabelSlot.ENTITY))
            if batch and labels != batch_labels:
                yield from flush()
            batch_labels = labels
            batch.append(chunk)
            if len(batch) >= self._batch_size:
                yield from flush()
        yield from flush()

    def _mentions_for(
        self, chunk: ChunkRecord, raw: list[dict[str, Any]]
    ) -> Iterator[EntityMention]:
        # ``raw`` is this chunk's entity list (chunk-local 'start'/'end', 'label', 'score') — the
        # same shape ``predict_entities`` returned per chunk, now sourced from the batched call and
        # mapped back by index. The offset/page resolution + contract stamping below are UNCHANGED.
        # Deterministic ordering: by chunk-local start, then label.
        raw.sort(key=lambda e: (int(e["start"]), str(e["label"])))
        # Resolve chunk-local model offsets to TRUE original offsets through the chunk's
        # stripped→original offset map (markup stripping makes char_start+local only a
        # lower bound — ADR-0031); a pre-v11 chunk with no map degrades to that shift.
        offset_index = OffsetIndex(chunk.offset_map) if chunk.offset_map is not None else None
        chunk_char_start = chunk.provenance.char_start or 0
        # Re-resolve each mention's OWN page from the parent's page map (carried on
        # the chunk), not just inherit the chunk's page range. Built once per chunk.
        page_index = PageOffsetIndex(chunk.page_map) if chunk.page_map is not None else None
        for i, ent in enumerate(raw):
            local_start = int(ent["start"])
            local_end = int(ent["end"])
            if offset_index is not None:
                doc_char_start, doc_char_end = offset_index.resolve_span(local_start, local_end)
            else:
                doc_char_start = chunk_char_start + local_start
                doc_char_end = chunk_char_start + local_end
            provenance = self._mention_provenance(
                chunk.provenance,
                doc_char_start=doc_char_start,
                doc_char_end=doc_char_end,
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
