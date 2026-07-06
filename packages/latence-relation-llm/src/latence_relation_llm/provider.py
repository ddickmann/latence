"""LLM-based RelationExtractor Provider over any OpenAI-compatible endpoint.

The S5 Relation Extraction *LLM* reference Provider (ADR-0013: the choice of RE
Provider is empirical; an LLM extractor over any OpenAI-compatible endpoint —
including a **local Ollama** — is one of the two shipped candidates). It runs
behind the OpenAI-compatible chat-completions API, so the same Provider drives
OpenAI, vLLM, Together, or a laptop-local Ollama with only a ``base_url`` change
(ADR-0007: the endpoint path is the performance path; nothing GPU is forced into
first-run).

Given a document's :class:`~latence_core.contracts.EntityMention`s plus the chunk
text for context, it prompts the model to emit typed directed relations over the
mention set, constrained to a per-Pipeline relation label set (zero-shot — the
Capability names no relation types; the label set is Stage ``config``). The model
returns JSON referencing mentions by an index into the numbered mention list; the
Provider resolves those indices back to real mention ``record_id``s and stamps a
:class:`~latence_core.contracts.RelationMention` whose Provenance span covers both
endpoints (resolving to the source page via the parent's page map — the S5
round-trip). A relation naming an out-of-range index, a self-relation, or an
unknown label is dropped, so a hallucinated reference never yields an
un-resolvable record.

Licensing: this Provider is framework code (Apache-2.0) and calls a user-supplied
endpoint; it bundles no model weights. The ``openai`` client is Apache-2.0 (code).
No key or endpoint is embedded — both come from config/environment.

The ``openai`` import is deferred to first use so importing this module (for
Provider discovery) never requires the client to be installed, and a missing
optional dep raises a clear, actionable error only when the Provider is run.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from latence_core.contracts import RelationMention
from latence_core.pagemap import PageOffsetIndex

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord, EntityMention, PageMap, Provenance

_DEFAULT_MODEL = "gpt-4o-mini"
_SYSTEM_PROMPT = (
    "You are a precise relation-extraction engine. Given a document's text and a "
    "numbered list of entity mentions found in it, identify the typed, directed "
    "relations between those mentions. Only use the provided relation labels. "
    "Refer to mentions by their number. Respond ONLY with a JSON object of the form "
    '{"relations": [{"head": <int>, "tail": <int>, "label": "<relation>", '
    '"confidence": <0..1>}]}. Emit no relation you cannot support from the text.'
)


class LLMRelationExtractor:
    """RelationExtractor Provider backed by an OpenAI-compatible chat endpoint.

    Config:
        labels: list[str]  — REQUIRED. The zero-shot relation types to extract (e.g.
            ``["works_for", "located_in"]``). Nothing is hardcoded in core (S5 seam).
        model: str  — the chat model id (default ``gpt-4o-mini``). For Ollama use e.g.
            ``"llama3.1"`` with ``base_url="http://localhost:11434/v1"``.
        base_url: str | None  — the OpenAI-compatible endpoint. ``None`` uses the
            client default (OpenAI). Set it to a vLLM/Ollama URL to swap substrate
            with no code change.
        api_key: str | None  — the key; falls back to ``$OPENAI_API_KEY``. A local
            Ollama accepts any placeholder (e.g. ``"ollama"``).
        temperature: float  — sampling temperature (default 0.0 for determinism).
        max_mentions: int  — cap on mentions sent per document to bound the prompt
            (default 100).
        default_confidence: float  — used when the model omits a confidence (default 0.8).

    The client is constructed lazily on first ``relate`` so building the Provider
    (which the Runner does for every Stage) never needs the endpoint reachable.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        labels = config.get("labels")
        if not labels or not isinstance(labels, list):
            msg = (
                "LLMRelationExtractor requires a non-empty 'labels' list in its config "
                "(zero-shot RE: the relation label set is supplied per Pipeline)."
            )
            raise ValueError(msg)
        self._labels = [str(label) for label in labels]
        self._label_set = set(self._labels)
        self._model = str(config.get("model", _DEFAULT_MODEL))
        self._base_url = config.get("base_url")
        self._api_key = config.get("api_key")
        self._temperature = float(config.get("temperature", 0.0))
        self._max_mentions = int(config.get("max_mentions", 100))
        self._default_confidence = float(config.get("default_confidence", 0.8))
        self._client: Any = None

    # -- lazy client --------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'openai' package is required to run LLMRelationExtractor. "
                "Install it with: pip install latence-relation-llm"
            )
            raise ImportError(msg) from exc
        api_key = self._api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self._client = OpenAI(base_url=self._base_url, api_key=api_key)
        return self._client

    # -- extraction ---------------------------------------------------------

    def relate(
        self, chunks: Iterable[ChunkRecord], mentions: Iterable[EntityMention]
    ) -> Iterator[RelationMention]:
        chunk_list = list(chunks)
        doc_text = _document_texts(chunk_list)
        doc_page_map = _document_page_maps(chunk_list)

        by_doc: dict[str, list[EntityMention]] = {}
        order: list[str] = []
        for m in mentions:
            if m.document_record_id not in by_doc:
                by_doc[m.document_record_id] = []
                order.append(m.document_record_id)
            by_doc[m.document_record_id].append(m)

        client = None
        for doc_id in order:
            doc_mentions = by_doc[doc_id]
            if len(doc_mentions) < 2:
                continue
            if client is None:
                client = self._get_client()
            yield from self._relate_one_document(
                client, doc_id, doc_mentions, doc_text.get(doc_id, ""), doc_page_map.get(doc_id)
            )

    def _relate_one_document(
        self,
        client: Any,
        doc_id: str,
        mentions: list[EntityMention],
        text: str,
        page_map: PageMap | None,
    ) -> Iterator[RelationMention]:
        # Order mentions in document position so the numbered list is stable and the
        # model sees them in reading order. Cap the count to bound the prompt.
        ordered = sorted(mentions, key=lambda m: (m.provenance.char_start or 0, m.record_id))[
            : self._max_mentions
        ]
        raw = self._call_model(client, text, ordered)
        page_index = PageOffsetIndex(page_map) if page_map is not None else None
        for emitted, rel in enumerate(self._valid_relations(raw, ordered)):
            head, tail, label, confidence = rel
            span_start = min(head.provenance.char_start or 0, tail.provenance.char_start or 0)
            span_end = max(head.provenance.char_end or 0, tail.provenance.char_end or 0)
            provenance = _relation_provenance(head, tail, span_start, span_end, page_index)
            yield RelationMention(
                record_id=f"{doc_id}#relation-{emitted}",
                document_record_id=doc_id,
                index=emitted,
                label=label,
                head_mention_id=head.record_id,
                tail_mention_id=tail.record_id,
                head_text=head.text,
                tail_text=tail.text,
                confidence=confidence,
                provenance=provenance,
                classification=head.classification,
            )

    def _call_model(
        self, client: Any, text: str, mentions: list[EntityMention]
    ) -> list[dict[str, Any]]:
        numbered = "\n".join(
            f"{i}: {m.text} ({m.label})" for i, m in enumerate(mentions)
        )
        user = (
            f"Relation labels: {', '.join(self._labels)}\n\n"
            f"Document text:\n{text}\n\n"
            f"Entity mentions:\n{numbered}\n\n"
            "Return the JSON object of relations now."
        )
        response = client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # A non-JSON reply yields no relations rather than crashing the run.
            return []
        relations = parsed.get("relations", []) if isinstance(parsed, dict) else []
        return [r for r in relations if isinstance(r, dict)]

    def _valid_relations(
        self, raw: list[dict[str, Any]], mentions: list[EntityMention]
    ) -> Iterator[tuple[EntityMention, EntityMention, str, float]]:
        """Resolve model output to valid (head, tail, label, confidence) tuples.

        Drops any relation with an out-of-range mention index, an unknown label, or a
        head == tail self-reference — so a hallucinated reference never becomes an
        un-resolvable record. Deterministic order (as returned, then de-duplicated on
        head/tail/label).
        """
        n = len(mentions)
        seen: set[tuple[str, str, str]] = set()
        for rel in raw:
            label = str(rel.get("label", ""))
            if label not in self._label_set:
                continue
            try:
                head_idx = int(rel["head"])
                tail_idx = int(rel["tail"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= head_idx < n) or not (0 <= tail_idx < n) or head_idx == tail_idx:
                continue
            head = mentions[head_idx]
            tail = mentions[tail_idx]
            key = (head.record_id, tail.record_id, label)
            if key in seen:
                continue
            seen.add(key)
            try:
                confidence = float(rel.get("confidence", self._default_confidence))
            except (TypeError, ValueError):
                confidence = self._default_confidence
            confidence = min(max(confidence, 0.0), 1.0)
            yield head, tail, label, confidence


# -- shared text/page-map/provenance helpers (mirror the core provider) ------


def _document_texts(chunks: list[ChunkRecord]) -> dict[str, str]:
    """Reconstruct each document's assembled text from its chunks' document spans."""
    by_doc: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.document_record_id, []).append(chunk)
    texts: dict[str, str] = {}
    for doc_id, doc_chunks in by_doc.items():
        end = 0
        for chunk in doc_chunks:
            start = chunk.provenance.char_start or 0
            end = max(end, start + len(chunk.content))
        buf = [" "] * end
        for chunk in doc_chunks:
            start = chunk.provenance.char_start or 0
            buf[start : start + len(chunk.content)] = chunk.content
        texts[doc_id] = "".join(buf)
    return texts


def _document_page_maps(chunks: list[ChunkRecord]) -> dict[str, PageMap | None]:
    maps: dict[str, PageMap | None] = {}
    for chunk in chunks:
        if chunk.document_record_id not in maps and chunk.page_map is not None:
            maps[chunk.document_record_id] = chunk.page_map
    return maps


def _relation_provenance(
    head: EntityMention,
    tail: EntityMention,
    span_start: int,
    span_end: int,
    page_index: PageOffsetIndex | None,
) -> Provenance:
    """Provenance for a relation: shared lineage + covering span AND its page(s)."""
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
