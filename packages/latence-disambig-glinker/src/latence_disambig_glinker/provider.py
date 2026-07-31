"""Enterprise GLinker Disambiguation Provider (``disambiguation.glinker``) — ADR-0046.

The ENTERPRISE entity-linking + resolution engine ported from the private
``latenceai-dataset-intelligence`` repo, wired behind the framework's single
:class:`~latence_core.capability.Disambiguator` seam as a drop-in for ``disambiguation.cascade`` /
``disambiguation.embedding``. It fuses the two ported halves into one Stage:

1. **Corpus-internal KB + neural GLinker linking** (:mod:`.kb_builder`, :mod:`.backends`,
   :mod:`.linker`) — builds a self-referencing KB from the run's mentions (dominant-label vote +
   aliases + context descriptions), then runs the full GLinker **L2->L3->L4->L0** pipeline
   (``knowledgator/gliner-linker-large-v1.0`` + ``gliner-linker-rerank-v1.0``, FlashDeBERTa) over
   ±window-context inputs to link each mention to its canonical KB entry, with a **direct-KB
   fallback** for what the neural pass misses and provisional entities for the unresolved.
2. **Audited resolver** (:mod:`.entity_resolver`) — fuses the linked canonical entities by
   exact / alias / acronym / substring / **fuzzy edit-distance** (the embedding-by-label rung is
   opt-in, OFF by default — it over-merges short names on real data), with transitive merge + cycle
   detection and a :class:`~.entity_resolver.MergeDecision` audit log (§19: **no silent over-merge**
   — every merge carries reason + confidence + provenance).

The resolved clusters are lifted onto the framework's ``ResolvedCluster`` so ALL the inherited
:class:`~latence_core.stages.disambiguation.CascadeDisambiguator` machinery — record building,
external-KB linking (``kb_id``), relation normalisation, page-accurate Provenance/Evidence, and
deterministic emission order — is reused unchanged. A stack swapping ``disambiguation.embedding`` ->
``disambiguation.glinker`` needs no other change; the ``DisambiguationRecord`` contract holds.

Enterprise honesty: ProviderProfile ``compute="gpu"`` (the neural gliner-linker needs a GPU) with
``deterministic=False`` (the neural biencoder + FA2 is not bitwise-reproducible; the string-cascade
rungs ARE deterministic). Robust fallbacks throughout — a backend load/inference failure degrades to
the direct-KB path, and the FAISS-GPU embedding-merge path falls back to brute-force when ``faiss``
is absent — so the Stage never crashes. The vLLM serving backend is a documented, seam-compatible
follow-on (:class:`~.backends.VLLMGLiNERBackend`); the transformers path is the complete default.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from latence_core.capability import Embedder, ProviderRegistry
from latence_core.disambiguation import MergeAudit, ResolvedCluster, Strategy, normalize_text
from latence_core.errors import ConfigError
from latence_core.offsetmap import OffsetIndex
from latence_core.providers.adapter import AdapterBase
from latence_core.providers.profile import ProviderProfile
from latence_core.stages.disambiguation import CascadeDisambiguator

from latence_disambig_glinker.backends import GLinkerBackend
from latence_disambig_glinker.embedder import LabelEmbedder
from latence_disambig_glinker.entity_resolver import EntityResolver, MergeDecision
from latence_disambig_glinker.linker import EntityLinker
from latence_disambig_glinker.models import CanonicalEntityRecord, MentionRecord

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import ChunkRecord, EntityMention

logger = logging.getLogger(__name__)

# The in-core, dependency-free, DETERMINISTIC Embedder used for the resolver's STS merge when no
# embedder is configured (offline / CI). A learned embedder — the gpu-sota-glinker stack names IBM
# Granite Embedding r2 via 'embedding.sentence_transformers' — is a one-line opt-in.
_DEFAULT_EMBEDDER = "embedding.hashing"

# The stable dataset id used for the ported engine's internal (provisional) entity ids. It does NOT
# leak into the framework CanonicalEntity id (that is content-addressed from the canonical surface),
# so it only affects internal clustering — kept stable for determinism.
_DATASET_ID = "latence-corpus"

# ±window (characters) of chunk context taken around a mention for the neural linker input — the
# source used ±200. Bigger context = better biencoder disambiguation, bounded by the model window.
_DEFAULT_CONTEXT_WINDOW = 200

# Map each ported resolver merge reason to the framework engine Strategy (the contract MergeStrategy
# maps 1:1 by value), so an audited MergeDecision surfaces under the right strategy in the report.
_REASON_TO_STRATEGY: dict[str, Strategy] = {
    "exact_name_match": Strategy.EXACT,
    "incremental_exact_name": Strategy.EXACT,
    "alias_overlap": Strategy.ALIAS,
    "incremental_alias_overlap": Strategy.ALIAS,
    "acronym_expansion": Strategy.ACRONYM,
    "incremental_acronym": Strategy.ACRONYM,
    "substring_containment": Strategy.SUBSTRING,
    "incremental_substring": Strategy.SUBSTRING,
    # The fuzzy edit-distance rung (FIX-3) is a deterministic LEXICAL rung (a normalized char-
    # similarity match, not a vector one), so it surfaces under the SUBSTRING strategy bucket — the
    # closest lexical kin in the contract MergeStrategy — while its distinct technique stays legible
    # in the audited MergeDecision ``reason`` ("fuzzy_edit_distance") + the Evidence snippet.
    "fuzzy_edit_distance": Strategy.SUBSTRING,
    "embedding_similarity": Strategy.EMBEDDING,
    "embedding_similarity_faiss": Strategy.EMBEDDING,
    "incremental_embedding": Strategy.EMBEDDING,
}


class GLinkerDisambiguator(CascadeDisambiguator, AdapterBase):
    """Disambiguator Provider: neural GLinker linking + audited resolver (the enterprise crown).

    Config (beyond the cascade's ``aliases`` / ``kb`` / relation-label keys, all still honoured):
        embedder: dict | None  — the STS Embedder for the resolver's embedding-by-label merge, a
            ``{"provider": <entry-point name>, "config": {...}}`` block. Absent ⇒ the in-core
            deterministic ``embedding.hashing`` (offline). The gpu-sota-glinker stack names Granite.
        context_window: int  — ±chars of chunk context around a mention for the linker input
            (default ``200``); ``0`` feeds the bare surface.
        device: str  — the neural backend device (default ``"cuda"``). On a non-GPU host it
            load fails and the Stage degrades to the deterministic direct-KB + resolver path.
        threshold: float  — neural linking confidence threshold (default ``0.3``).
        use_reranker: bool  — enable the L4 reranker (default ``True``).
        use_flash_deberta: bool  — enable FlashDeBERTa when its package is present (default True).
        embedding_merge_threshold: float  — cosine cutoff for the STS embedding-merge rung. Default
            ``0.93`` (safe floor); the rung REQUIRES a discriminative (non-retrieval) STS embedder.
        use_embedding_merge: bool  — enable the Granite-STS embedding-by-label rung. Default
            ``False`` (ER audit FIX-1: it over-merges short names with a retrieval embedder).
        acronym_expand: bool  — toggle the acronym rung (default True).
        use_fuzzy_merge: bool  — enable the conservative fuzzy edit-distance rung (default True: the
            precision-safe OCR/spelling-variant recall the string + embedding rungs cannot supply).
        min_similarity: float  — char-similarity cutoff for the fuzzy rung (default 0.84; the
            distinguishing-token guard, not the threshold, blocks numeric/leading-code confusions).
        fuzzy_shared_doc_guard: bool  — also require a shared doc for a fuzzy merge (default off).
        max_candidates / max_length / precompute_batch_size / l2_search_mode / reranker_max_labels:
            GLinker pipeline pass-throughs (see :class:`~.backends.GLinkerBackend`).
        batch_size / device: — AdapterBase batching + guard knobs.
    """

    #: A native ValueError/KeyError from a resolved embedder means a bad config -> ConfigError.
    config_exceptions = (ValueError, KeyError)

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: ProviderRegistry | None = None,
    ) -> None:
        config = dict(config or {})
        # The cascade half owns record building, external-KB linking + relation normalisation.
        CascadeDisambiguator.__init__(self, config)
        # The AdapterBase half owns batching/guard + the config accessor.
        AdapterBase.__init__(self, config)

        self._dataset_id = str(config.get("dataset_id", _DATASET_ID))
        self._context_window = int(config.get("context_window", _DEFAULT_CONTEXT_WINDOW))
        if self._context_window < 0:
            msg = f"{self.provider_name}: context_window must be >= 0, got {self._context_window}."
            raise ConfigError(msg)
        self._device = str(config.get("device", "cuda"))
        self._threshold = float(config.get("threshold", 0.3))
        self._use_reranker = bool(config.get("use_reranker", True))
        self._use_flash_deberta = bool(config.get("use_flash_deberta", True))
        self._embedding_merge_threshold = float(config.get("embedding_merge_threshold", 0.93))
        # PRECISION default (ER audit FIX-1): the embedding-by-label rung is OFF by default. On real
        # short entity names Granite's retrieval embeddings collapse distinct persons/dates/orgs (no
        # clean threshold). Recall comes from the neural linker + string cascade + fuzzy rung.
        self._use_embedding_merge = bool(config.get("use_embedding_merge", False))
        self._acronym_expand = bool(config.get("acronym_expand", True))
        self._use_fuzzy_merge = bool(config.get("use_fuzzy_merge", True))
        self._min_similarity = float(config.get("min_similarity", 0.84))
        self._fuzzy_shared_doc_guard = bool(config.get("fuzzy_shared_doc_guard", False))
        self._max_candidates = int(config.get("max_candidates", 100))
        self._max_length = int(config.get("max_length", 512))
        self._precompute_batch_size = int(config.get("precompute_batch_size", 16))
        self._l2_search_mode = str(config.get("l2_search_mode", "exact"))
        self._reranker_max_labels = int(config.get("reranker_max_labels", 20))

        self._embedder_spec = config.get("embedder")
        self._registry = registry
        self._embedder: Embedder | None = None

    # -- profile ------------------------------------------------------------

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2 / ADR-0046): neural GLinker, GPU, Apache-2.0.

        ``compute="gpu"``: the L3/L4 gliner-linker biencoder needs a GPU (the device seam
        skips-with-flag on a CPU-only host — enterprise honesty). ``model_id`` is the L3
        linker checkpoint ``knowledgator/gliner-linker-large-v1.0`` — Apache-2.0 WEIGHTS (card,
        verified 2026-07-14); the ``glinker`` (Knowledgator, provides ``ProcessorFactory``) +
        ``flashDeBERTa`` LIBRARY code is Apache-2.0 and the optional ``faiss`` is MIT
        (THIRD-PARTY-LICENSES + ADR-0046). ``deterministic=False``: the
        neural biencoder + FlashDeBERTa is not bitwise-reproducible across BLAS/hardware (declared
        honestly, not overclaimed); the string-cascade rungs of the resolver ARE deterministic.
        """
        return ProviderProfile(
            compute="gpu",
            memory_mb=2048,
            model_id="knowledgator/gliner-linker-large-v1.0",
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_weights="Apache-2.0",
            license_source="https://huggingface.co/knowledgator/gliner-linker-large-v1.0",
            license_verified_on="2026-07-14",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
            perf_note="FlashDeBERTa when present; graceful standard-attention fallback",
        )

    @property
    def provider_name(self) -> str:
        """The registered entry-point name, for error tags (never carries content)."""
        return "disambiguation.glinker"

    # -- the overridden blocking/merge seam ---------------------------------

    def _resolve_clusters(
        self,
        mention_list: list[EntityMention],
        chunks: Iterable[ChunkRecord],
    ) -> list[ResolvedCluster]:
        """Neural GLinker linking + audited resolver -> canonical clusters (the enterprise path).

        Projects EntityMentions into the ported engine's MentionRecords (±window chunk context),
        runs :class:`~.linker.EntityLinker` (corpus KB + neural GLinker + direct-KB fallback), fuses
        the linked canonical entities with :class:`~.entity_resolver.EntityResolver` (Granite STS,
        FAISS-optional), and lifts the result onto ``ResolvedCluster`` (with the audited
        MergeDecisions mapped to engine MergeAudits) so the inherited record building is unchanged.
        Deterministic in the cluster ORDER (sorted by canonical text); the neural link content is
        declared non-deterministic in the profile.
        """
        if not mention_list:
            return []

        chunk_by_id = {c.record_id: c for c in chunks}
        records = [self._to_mention_record(m, chunk_by_id) for m in mention_list]

        with self.guard("disambiguate"):
            backend = GLinkerBackend(
                model_name=GLinkerBackend.LINKER_MODEL,
                device=self._device,
                reranker_model=(GLinkerBackend.RERANKER_MODEL if self._use_reranker else None),
                reranker_max_labels=self._reranker_max_labels,
                use_flash_deberta=self._use_flash_deberta,
                threshold=self._threshold,
                max_length=self._max_length,
                precompute_batch_size=self._precompute_batch_size,
                max_candidates=self._max_candidates,
                l2_search_mode=self._l2_search_mode,
            )
            linker = EntityLinker(
                backend=backend, threshold=self._threshold, batch_size=self.batch_size()
            )
            linked_mentions, canonical_entities = linker.link_dataset(records, self._dataset_id)

            resolver = EntityResolver(
                acronym_expand=self._acronym_expand,
                use_fuzzy_merge=self._use_fuzzy_merge,
                min_similarity=self._min_similarity,
                fuzzy_shared_doc_guard=self._fuzzy_shared_doc_guard,
                embedding_merge_threshold=self._embedding_merge_threshold,
                use_embedding_merge=self._use_embedding_merge,
                label_encoder=(
                    LabelEmbedder(self._resolve_embedder()) if self._use_embedding_merge else None
                ),
                # Inject the framework's own surface normaliser so the resolver's exact-name/alias
                # clusters agree BY CONSTRUCTION with the framework's content-addressed record_id
                # (``entity:{normalize_text(canonical_text)}``): a confusable surface never survives
                # as a second cluster that collapses onto an existing id, losing provenance (A-F1).
                name_normalizer=normalize_text,
            )
            resolved_entities, _resolved_mentions = resolver.resolve(
                canonical_entities, linked_mentions, self._dataset_id
            )

        return self._to_resolved_clusters(resolved_entities, resolver.merge_log)

    # -- EntityMention -> MentionRecord (±window chunk context) -------------

    def _to_mention_record(
        self,
        mention: EntityMention,
        chunk_by_id: dict[str, ChunkRecord],
    ) -> MentionRecord:
        """Project an EntityMention into the ported engine's MentionRecord (±window context)."""
        return MentionRecord(
            mention_id=mention.record_id,
            dataset_id=self._dataset_id,
            document_id=mention.provenance.document_id,
            chunk_id=mention.chunk_record_id,
            text=mention.text,
            label=mention.label,
            start=mention.provenance.char_start or 0,
            end=mention.provenance.char_end or 0,
            score=mention.confidence,
            context=self._mention_context(mention, chunk_by_id),
        )

    def _mention_context(
        self,
        mention: EntityMention,
        chunk_by_id: dict[str, ChunkRecord],
    ) -> str:
        """A ±``context_window`` window of the mention's chunk content (coordinate-correct).

        Mirrors the source linker's ±200 window, but sourced from the run's ChunkRecords: the
        mention's ORIGINAL-markdown offset is inverted to a content-local offset through the chunk's
        ``offset_map`` (the ADR-0031 inverse resolver), so the window indexes markup-STRIPPED
        ``content`` correctly. A missing chunk / unlocatable offset / ``context_window == 0`` falls
        to the bare surface — never a crash.
        """
        if self._context_window <= 0:
            return mention.text
        chunk = chunk_by_id.get(mention.chunk_record_id)
        if chunk is None or not chunk.content:
            return mention.text
        content = chunk.content
        mention_start = mention.provenance.char_start
        chunk_start = chunk.provenance.char_start
        if mention_start is None or chunk_start is None:
            return mention.text
        if chunk.offset_map is not None:
            local = OffsetIndex(chunk.offset_map).resolve_original(mention_start)
        else:
            local = mention_start - chunk_start
        if local < 0 or local > len(content):
            return mention.text
        start = max(0, local - self._context_window)
        end = min(len(content), local + len(mention.text) + self._context_window)
        window = content[start:end].strip()
        return window or mention.text

    # -- CanonicalEntityRecord -> ResolvedCluster (with audited merges) -----

    def _to_resolved_clusters(
        self,
        entities: list[CanonicalEntityRecord],
        merge_log: list[MergeDecision],
    ) -> list[ResolvedCluster]:
        """Lift resolved canonical entities onto ``ResolvedCluster`` with mapped merge audits.

        Each surviving entity becomes one cluster; the audited MergeDecisions are attached to the
        surviving entity that ABSORBED the merge — located by surface name, since ``_apply_entity_
        merges`` folds every merged entity's canonical_name + aliases into the survivor (so both the
        merge's source_name and target_name appear on exactly one surviving entity). Attaching by
        name (not by the merge-map root id, which can differ from the survivor's evidence-chosen
        ``entity_id``) is robust — no audited merge is silently lost (§19). The
        cluster's fields come straight from the resolved entity; clusters are sorted by canonical
        text for deterministic emission order.
        """
        # Map every surface (canonical + aliases, normalised) of each surviving entity -> its id, so
        # a MergeDecision can be attached to the entity that absorbed either endpoint's surface.
        entity_by_surface: dict[str, str] = {}
        for e in entities:
            for surface in (e.canonical_name, *e.aliases):
                entity_by_surface.setdefault(normalize_text(surface), e.entity_id)

        audits_by_entity: dict[str, list[MergeAudit]] = {e.entity_id: [] for e in entities}
        for d in merge_log:
            target_name = d.target_name or d.target_entity_id
            source_name = d.source_name or d.source_entity_id
            owner = entity_by_surface.get(normalize_text(target_name)) or entity_by_surface.get(
                normalize_text(source_name)
            )
            if owner is None:
                continue  # both endpoints vanished (should not happen); never fabricate an owner
            strategy = _REASON_TO_STRATEGY.get(d.reason)
            if strategy is None:
                # An unmapped reason means a resolver rung was added without extending
                # _REASON_TO_STRATEGY — log it loudly (rather than silently mislabelling the audit
                # as EMBEDDING) and fall back to EMBEDDING so the merge is still surfaced (B-F5).
                logger.warning(
                    "%s: unmapped resolver merge reason %r — auditing as EMBEDDING; extend "
                    "_REASON_TO_STRATEGY to classify it correctly.",
                    self.provider_name,
                    d.reason,
                )
                strategy = Strategy.EMBEDDING
            audits_by_entity[owner].append(
                MergeAudit(
                    source_text=source_name,
                    target_text=target_name,
                    strategy=strategy,
                    confidence=_clamp01(d.confidence),
                    applied=True,
                    reason=d.reason,
                )
            )

        clusters: list[ResolvedCluster] = []
        for e in entities:
            audits = sorted(
                audits_by_entity.get(e.entity_id, []),
                key=lambda a: (a.strategy.value, a.target_text, a.source_text),
            )
            clusters.append(
                ResolvedCluster(
                    canonical_text=e.canonical_name,
                    entity_type=e.label,
                    member_mention_ids=sorted(e.mention_ids),
                    document_ids=sorted(e.document_ids),
                    merge_audits=audits,
                    confidence=_clamp01(e.confidence),
                    link_confidences={
                        m: _clamp01(v) for m, v in sorted(e.link_confidences.items())
                    },
                )
            )
        clusters.sort(key=lambda c: (c.canonical_text, normalize_text(c.canonical_text)))
        return clusters

    # -- embedder resolution (mirrors EmbeddingDisambiguator) ---------------

    def _resolve_embedder(self) -> Embedder:
        """Resolve (once) the STS Embedder from ``config['embedder']`` (default hashing).

        ``spec`` is ``None`` (⇒ the in-core deterministic ``embedding.hashing``) or a
        ``{"provider", "config"}`` mapping naming any Embedder Provider (the gpu-sota-glinker stack
        names Granite r2). A mis-named / non-Embedder Provider fails loudly at wiring time.
        """
        if self._embedder is not None:
            return self._embedder
        spec = self._embedder_spec
        registry = self._registry or ProviderRegistry()
        if spec is None:
            provider_name = _DEFAULT_EMBEDDER
            embedder_config: dict[str, Any] = {}
        elif isinstance(spec, dict):
            name = spec.get("provider")
            if not isinstance(name, str) or not name:
                msg = (
                    f"{self.provider_name}: config 'embedder' requires a non-empty string "
                    f"'provider' entry-point name."
                )
                raise ConfigError(msg)
            provider_name = name
            raw_config = spec.get("config", {})
            if not isinstance(raw_config, dict):
                msg = f"{self.provider_name}: 'embedder.config' must be a mapping if present."
                raise ConfigError(msg)
            embedder_config = dict(raw_config)
        else:
            msg = (
                f"{self.provider_name}: config 'embedder' must be a mapping "
                f"{{'provider': <name>, 'config': {{...}}}} or absent; got {type(spec).__name__}."
            )
            raise ConfigError(msg)
        embedder_cls = registry.load(provider_name)
        embedder = embedder_cls(embedder_config)
        if not isinstance(embedder, Embedder):
            msg = (
                f"{self.provider_name}: Provider {provider_name!r} named in config 'embedder' does "
                f"not satisfy the Embedder capability (needs 'dimension' + 'embed(texts)')."
            )
            raise ConfigError(msg)
        self._embedder = embedder
        return embedder


def _clamp01(value: float) -> float:
    """Clamp a confidence into the contract's ``[0.0, 1.0]`` bound."""
    return max(0.0, min(1.0, value))
