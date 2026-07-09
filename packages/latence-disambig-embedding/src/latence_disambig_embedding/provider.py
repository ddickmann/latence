"""Learned (embedding-based) Disambiguation Provider — surface+context ER (W1 rework, #94).

The first **learned** Disambiguation Provider: semantic entity resolution beyond the algorithmic
``disambiguation.cascade`` (surface/acronym/substring + union-find). It embeds each entity mention's
**surface plus its surrounding context** and merges semantically-equivalent entities the surface
cascade misses ("IBM" ↔ "International Business Machines" in one document's context) while keeping
distinct entities apart — anticipated by the design ("a FAISS/GPU blocking Provider is a pluggable
option", the :class:`~latence_core.capability.Disambiguator` docstring, S8 AC).

Why surface+context (the pod finding that drove this rework, supersedes PR #127):
    The pod validation proved a **bare-surface** embedding disambiguator OVER-MERGES distinct
    entities, because short entity surfaces aren't discriminative with a sentence embedder. Measured
    (multilingual-e5-small, cosine):

    ============================================  ============  =====================
    pair                                          bare surface  surface + context
    ============================================  ============  =====================
    IBM ~ International Business Machines (same)   0.883         0.964
    Apple ~ Apple Inc (same)                       0.933         0.988
    IBM ~ Apple (DISTINCT)                          0.870         0.808
    IBM ~ Berlin (distinct)                         0.818         0.764
    ============================================  ============  =====================

    Bare surfaces have NO robust threshold (a distinct pair 0.870 outranks a same pair's margin);
    WITH context the separation is huge and clean (same ≈0.96–0.99, distinct ≈0.76–0.81, any ~0.90
    threshold works). So this Provider embeds ``surface + context`` — reaching the context through
    the run's :class:`~latence_core.contracts.ChunkRecord`s (threaded to ``disambiguate`` by the
    Runner, W1 rework) — and defaults the threshold to the context-safe ``0.90``. A mention with no
    locatable chunk gracefully falls back to bare surface (never crashes).

Drop-in for ``disambiguation.cascade``:
    It subclasses :class:`~latence_core.stages.disambiguation.CascadeDisambiguator`, overriding ONLY
    the blocking/merge step (:meth:`_resolve_clusters`) — the record building, KB linking, relation
    normalisation, page-accurate Provenance/Evidence carry, and deterministic emission order are all
    inherited, so it emits the byte-identical ``DisambiguationRecord`` contract and a stack swapping
    ``disambiguation.cascade`` → ``disambiguation.embedding`` needs no other change.

Enterprise safety (the whole point — no silent over-merge):
    * A pair merges ONLY when cosine ≥ ``threshold`` AND (default) the entity types are compatible.
    * A candidate that is similar-but-below-threshold, or type-incompatible, is recorded as a
      **rejected** :class:`~latence_core.contracts.MergeDecision` (``applied=False``) with its score
      and reason — never silently folded, never silently dropped.
    * The score written to an audit is clamped to ``[audit_floor, 1.0]`` so a valid
      confidence is produced even for a mildly-negative cosine on an off-model embedder.
    * O(n²) surface pairing is **blocked** (exact-normalised surfaces group for free first) and
      **capped** by ``max_pairs`` so a pathological corpus cannot blow up (the same class of
      the relation fan-out cap).

Determinism (Baseline bar): unique surfaces are processed in sorted order; the context window is a
fixed-width slice at a stable offset; ties break on the lexicographically-smaller surface (the
Union-Find representative rule inherited from the engine). Identical input+config → byte-identical
output whenever the embedder is deterministic (the default ``embedding.hashing`` is; the profile
declares ``deterministic`` from the resolved embedder so it is never overclaimed).

Built through :class:`~latence_core.providers.adapter.AdapterBase` (P3-F2/ADR-0036 §1): the device
seam (``either`` → cpu/cuda for a GPU embedder), :meth:`~...AdapterBase.guard` (a native embedder
fault → typed :class:`~latence_core.errors.ProviderError`, no record content / PII in the message),
and batched embedding. It carries no heavy dep itself — the Embedder is a resolved Provider.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from latence_core.capability import Embedder, ProviderRegistry
from latence_core.disambiguation import (
    MergeAudit,
    ResolvedCluster,
    Strategy,
    _UnionFind,
    normalize_text,
)
from latence_core.errors import ConfigError, ProviderError
from latence_core.offsetmap import OffsetIndex
from latence_core.providers.adapter import AdapterBase
from latence_core.providers.profile import ProviderProfile
from latence_core.stages.disambiguation import CascadeDisambiguator

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import ChunkRecord, EntityMention

# The in-core, dependency-free, DETERMINISTIC Embedder used offline / in CI when no embedder is
# configured. Keeps this Provider CPU-viable and offline by default (ADR-0007/0016) — a learned
# embedder (e.g. 'embedding.sentence_transformers') is an explicit one-line opt-in.
_DEFAULT_EMBEDDER = "embedding.hashing"

# The default cosine threshold. The pod margins put same-entity pairs (with context) at ≈0.96–0.99
# and distinct pairs at ≈0.76–0.81, so 0.90 cleanly separates them and is enterprise-safe out of the
# box (no over-merge) yet useful (catches real cross-surface duplicates) — NOT the fragile
# bare-surface 0.83 that the pod proved over-merges.
_DEFAULT_THRESHOLD = 0.90

# The default context window (characters) taken around a mention inside its chunk. A sentence-ish
# span so the embedding reflects the mention's meaning-in-context. >0 by default (the rework: the
# bare-surface path is only the graceful no-chunk fallback, not the default behaviour).
_DEFAULT_CONTEXT_WINDOW = 160

# The confidence floor a merge audit is clamped to, so an off-model embedder's mildly-negative
# cosine still yields a valid (>0) MergeDecision confidence rather than raising on the contract.
_DEFAULT_AUDIT_FLOOR = 0.0


class EmbeddingDisambiguator(CascadeDisambiguator, AdapterBase):
    """Embedding-based Disambiguator: merge mentions by surface+context cosine similarity.

    A drop-in for :class:`~latence_core.stages.disambiguation.CascadeDisambiguator` replacing the
    lexical 5-strategy blocking with a **learned** step: embed each unique surface's surface+context
    and union-finds surfaces whose cosine ≥ ``threshold`` (type-compatible). It inherits the
    record building, KB linking, relation normalisation and deterministic emission unchanged.

    Config (beyond the cascade's ``aliases`` / ``kb`` / relation-label keys, all still honoured):
        embedder: dict | None  — the Embedder Provider to embed with, a
            ``{"provider": <entry-point name>, "config": {...}}`` block (e.g.
            ``{"provider": "embedding.sentence_transformers", "config": {"device": "cuda"}}``).
            Absent ⇒ the in-core deterministic ``embedding.hashing`` (offline, CI-safe).
        threshold: float  — cosine cutoff for a merge (default ``0.90``, the context-safe value).
        context_window: int  — characters of chunk context taken around a mention (default ``160``);
            ``0`` embeds the bare surface (the old, over-merging behaviour — not recommended).
        require_type_compatibility: bool  — gate a merge on same/compatible entity types (default
            ``True``; a similar but type-incompatible pair is logged rejected, never merged).
        max_pairs: int  — the O(n²) cross-surface comparison cap (default ``100_000``); a corpus
            exceeding it raises a :class:`~latence_core.errors.ConfigError` naming the knob, rather
            than silently doing partial work.
        audit_floor: float  — the confidence floor a merge audit is clamped to (default ``0.0``).
        device / batch_size: — AdapterBase device + batching knobs, applied to the embedder call.
    """

    #: A native ``ValueError``/``KeyError`` from a resolved embedder means an operator handed a bad
    #: config → :class:`~latence_core.errors.ConfigError` via :meth:`guard`; else a runtime
    #: :class:`~latence_core.errors.ProviderError`.
    config_exceptions = (ValueError, KeyError)

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: ProviderRegistry | None = None,
    ) -> None:
        config = dict(config or {})
        # The cascade half owns the resolver/KB/normalizer config (aliases, kb, label maps, …).
        CascadeDisambiguator.__init__(self, config)
        # The AdapterBase half owns device/batching/guard + the config accessor.
        AdapterBase.__init__(self, config)
        self._threshold = float(config.get("threshold", _DEFAULT_THRESHOLD))
        self._context_window = int(config.get("context_window", _DEFAULT_CONTEXT_WINDOW))
        if self._context_window < 0:
            msg = f"{self.provider_name}: context_window must be >= 0, got {self._context_window}."
            raise ConfigError(msg)
        self._require_type = bool(config.get("require_type_compatibility", True))
        self._max_pairs = int(config.get("max_pairs", 100_000))
        if self._max_pairs < 1:
            msg = f"{self.provider_name}: max_pairs must be >= 1, got {self._max_pairs}."
            raise ConfigError(msg)
        self._audit_floor = float(config.get("audit_floor", _DEFAULT_AUDIT_FLOOR))
        self._embedder_spec = config.get("embedder")
        self._registry = registry
        self._embedder: Embedder | None = None

    # -- profile ------------------------------------------------------------

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): learned ER over a resolved Embedder.

        ``compute="either"``: the Provider itself is pure-Python (union-find + cosine); the compute
        it needs follows its resolved Embedder — the default ``embedding.hashing`` is CPU, a learned
        embedder opt-in accelerates on GPU (the device seam routes it). License is the *code*
        license of this pure-Python package (Apache-2.0, verified per ADR-0012 — this package's own
        code); a learned embedder opt-in carries its OWN verified weights license in ITS profile
        (H-B1 — this Provider never claims the embedder's license as its own).

        ``deterministic=False`` at the class level: whether a run is byte-identical depends on the
        *resolved* embedder (the default hashing embedder is deterministic; a learned float embedder
        is not guaranteed byte-identical across BLAS/hardware). Declared conservatively False rather
        than overclaim reproducibility for an embedder that may not have it; a deployment that pins
        deterministic hashing embedder gets a byte-identical run in practice, and the enterprise
        regression test asserts exactly that on the default (offline) path.
        """
        return ProviderProfile(
            compute="either",
            memory_mb=128,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_source="LICENSE",
            license_verified_on="2026-07-08",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
        )

    @property
    def provider_name(self) -> str:
        """The registered entry-point name, for error tags (never carries content)."""
        return "disambiguation.embedding"

    # -- the overridden blocking/merge seam ---------------------------------

    def _resolve_clusters(
        self,
        mention_list: list[EntityMention],
        chunks: Iterable[ChunkRecord],
    ) -> list[ResolvedCluster]:
        """Cluster mentions by surface+context embedding cosine (the learned ER, W1 rework #94).

        Blocking-first for scale: mentions are grouped by their exact normalised surface (an exact
        block merges for free). Then each unique surface is embedded as ``surface + context`` (the
        context is a fixed window of the surface's first-seen mention's chunk ``content`` around the
        mention offset; no chunk → bare surface, the graceful fallback) and every ordered pair of
        DISTINCT surfaces is compared: cosine ≥ ``threshold`` (and type-compatible when required)
        unions them, recorded as an APPLIED :class:`~latence_core.disambiguation.MergeAudit`; a
        similar-but-below or type-incompatible pair is recorded as a REJECTED audit (no silent
        over-merge). Union-Find yields the canonical clusters (lexicographic representative), sorted
        by canonical text — the exact :class:`~latence_core.disambiguation.ResolvedCluster` the
        cascade returns, so the inherited record building is unchanged.
        """
        if not mention_list:
            return []

        chunk_by_id = {c.record_id: c for c in chunks}

        # Group mentions by normalised surface (the exact block). Keep a stable display spelling
        # (first-seen) and the per-surface type/document/confidence evidence, in sorted order.
        by_norm: dict[str, list[EntityMention]] = {}
        display: dict[str, str] = {}
        for mention in mention_list:
            norm = normalize_text(mention.text)
            if not norm:
                continue
            by_norm.setdefault(norm, []).append(mention)
            display.setdefault(norm, mention.text)
        surfaces = sorted(by_norm)

        # Build one surface+context string per unique surface (deterministic: the first-seen
        # mention in input order, whose chunk window is a fixed slice), then embed them all in one
        # batched call (input-order preserved) so the vectors align to ``surfaces`` by index.
        context_texts = [
            self._surface_context_text(display[norm], by_norm[norm][0], chunk_by_id)
            for norm in surfaces
        ]
        vectors = self._embed_all(context_texts)

        labels_by_norm = {
            norm: frozenset(m.label for m in ms) for norm, ms in by_norm.items()
        }

        uf = _UnionFind()
        for norm in surfaces:
            uf.add(norm)

        audits: list[MergeAudit] = []
        comparisons = 0
        for i, target in enumerate(surfaces):
            for j in range(i + 1, len(surfaces)):
                comparisons += 1
                if comparisons > self._max_pairs:
                    msg = (
                        f"{self.provider_name}: cross-surface comparisons exceeded max_pairs="
                        f"{self._max_pairs} ({len(surfaces)} distinct surfaces). Raise max_pairs "
                        f"or add a stronger blocking key."
                    )
                    raise ConfigError(msg)
                source = surfaces[j]
                audit = self._propose(
                    source, target, display, labels_by_norm, vectors[j], vectors[i]
                )
                if audit is None:
                    continue
                audits.append(audit)
                if audit.applied:
                    uf.union(source, target)

        clusters = self._build_clusters(uf, by_norm, display, audits)
        clusters.sort(key=lambda c: c.canonical_text)
        return clusters

    # -- context + embedding ------------------------------------------------

    def _surface_context_text(
        self,
        surface: str,
        mention: EntityMention,
        chunk_by_id: dict[str, ChunkRecord],
    ) -> str:
        """Build the ``surface + context`` string embedded for one surface (graceful fallback).

        Locate the mention's parent chunk by ``chunk_record_id`` and take a fixed window of its
        markup-stripped ``content`` centred on the mention's position INSIDE that content.

        Coordinate correctness (the W1-rework fix, ADR-0031): ``mention.char_start`` is an offset
        in the parent document's ORIGINAL markdown, but ``chunk.content`` is markup-**stripped**, so
        the naive ``mention_start - chunk_start`` over-shoots the true content offset once markup
        was stripped *before* the mention within its chunk (and can push it past the content end,
        which would drop to a wrong or bare window on any real markdown corpus). We invert the
        mention's original offset back to a content-local offset through the chunk's ``offset_map``,
        via the sanctioned :meth:`~latence_core.offsetmap.OffsetIndex.resolve_original` — the exact
        inverse of the forward resolver Entity Extraction used. A chunk with **no** ``offset_map``
        (a pre-v11 chunk, or a document with no map) degrades to the identity
        ``mention_start - chunk_start`` shift, exact on the plain-text path (one identity segment).

        A missing chunk, an offset that cannot be located, or ``context_window == 0`` degrades to
        the bare surface — never a crash. The window is a stable slice so the text (and thus the
        embedding) is deterministic.
        """
        if self._context_window <= 0:
            return surface
        chunk = chunk_by_id.get(mention.chunk_record_id)
        if chunk is None or not chunk.content:
            return surface
        content = chunk.content
        chunk_start = chunk.provenance.char_start
        mention_start = mention.provenance.char_start
        if chunk_start is None or mention_start is None:
            return surface
        if chunk.offset_map is not None:
            # Invert the mention's ORIGINAL-markdown offset to a content-local offset through the
            # chunk's stripped→original map (never raises; a gap/out-of-range offset clamps into the
            # chunk content). This is the coordinate the stripped ``content`` must be indexed with.
            local = OffsetIndex(chunk.offset_map).resolve_original(mention_start)
        else:
            # No map (pre-v11 chunk or a document with no markup): the identity shift is exact.
            local = mention_start - chunk_start
        if local < 0 or local > len(content):
            # The offset does not fall inside this chunk's content (a mismatched chunk, or a legacy
            # chunk whose lower-bound shift overshot); fall back to the bare surface rather than
            # embed a wrong window.
            return surface
        half = self._context_window // 2
        start = max(0, local - half)
        end = min(len(content), local + len(surface) + half)
        context = content[start:end].strip()
        if not context:
            return surface
        # Join surface and context so the embedder sees the mention AND its meaning-in-context.
        return f"{surface} {context}"

    def _embed_all(self, texts: list[str]) -> list[list[float]]:
        """Embed every surface+context string through the resolved Embedder, in input order.

        The Embedder is resolved lazily (first use) from the plugin registry, so constructing the
        Provider (which the Runner does per Stage) never loads a model. Each batch goes through
        :meth:`guard` so a native embedder fault becomes a typed
        :class:`~latence_core.errors.ProviderError` carrying no raw text. Output order matches input
        order (the AdapterBase batching contract), so vectors align to the surface list by index.
        """
        if not texts:
            return []
        embedder = self._resolve_embedder()

        def run_batch(batch: list[str]) -> list[list[float]]:
            with self.guard("embed"):
                out = list(embedder.embed(batch))
            if len(out) != len(batch):
                msg = (
                    f"{self.provider_name}: embedder returned {len(out)} vectors for a batch of "
                    f"{len(batch)}."
                )
                raise ProviderError(msg)
            return out

        return list(self.batched(texts, run_batch))

    def _resolve_embedder(self) -> Embedder:
        """Resolve (once) the Embedder Provider from ``config['embedder']`` (default hashing).

        ``spec`` is ``None`` (⇒ the in-core deterministic ``embedding.hashing``) or a
        ``{"provider", "config"}`` mapping naming any Embedder Provider. The named Provider loads
        from the ``latence.providers`` registry and checked against the
        :class:`~latence_core.capability.Embedder` seam, so a mis-named Provider fails loudly at
        wiring time rather than silently producing no vectors.
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
                msg = (
                    f"{self.provider_name}: config 'embedder.config' must be a mapping if present."
                )
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
                f"not satisfy the Embedder capability (needs a 'dimension' property + an "
                f"'embed(texts)' method)."
            )
            raise ConfigError(msg)
        self._embedder = embedder
        return embedder

    # -- pair proposal (no silent over-merge) -------------------------------

    def _propose(
        self,
        source: str,
        target: str,
        display: dict[str, str],
        labels_by_norm: dict[str, frozenset[str]],
        source_vec: list[float],
        target_vec: list[float],
    ) -> MergeAudit | None:
        """Propose a merge for one ordered surface pair; return an audit (applied or rejected).

        Cosine ≥ ``threshold`` AND (when required) type-compatible ⇒ an APPLIED audit; a
        similar-but-below-threshold pair, or a type-incompatible pair, ⇒ a REJECTED audit
        (``applied=False``) with the reason — never silently folded or dropped. A pair that is not
        even similar (well below threshold) returns ``None`` (no audit noise for unrelated
        surfaces), matching the cascade's "no decision fired" behaviour. The recorded confidence is
        the cosine clamped to ``[audit_floor, 1.0]`` so the contract's 0..1 bound always holds.
        """
        sim = _cosine(source_vec, target_vec)
        src_disp, tgt_disp = display[source], display[target]
        confidence = _clamp(sim, self._audit_floor, 1.0)
        if sim < self._threshold:
            # Unrelated surfaces produce no audit; a NEAR miss (within a small margin) is logged as
            # a rejected below-threshold decision so an operator can see what was almost merged.
            if sim < self._threshold - 0.1:
                return None
            return MergeAudit(
                source_text=src_disp,
                target_text=tgt_disp,
                strategy=Strategy.EMBEDDING,
                confidence=confidence,
                applied=False,
                reason=(
                    f"embedding: cos={round(sim, 6)} < threshold={self._threshold} "
                    f"(below policy, not merged)"
                ),
            )
        if self._require_type and not self._types_compatible(source, target, labels_by_norm):
            return MergeAudit(
                source_text=src_disp,
                target_text=tgt_disp,
                strategy=Strategy.EMBEDDING,
                confidence=confidence,
                applied=False,
                reason=(
                    f"embedding: cos={round(sim, 6)} >= threshold but entity types incompatible "
                    f"(type guard, not merged)"
                ),
            )
        return MergeAudit(
            source_text=src_disp,
            target_text=tgt_disp,
            strategy=Strategy.EMBEDDING,
            confidence=confidence,
            applied=True,
            reason=f"embedding: cos={round(sim, 6)} >= threshold={self._threshold}",
        )

    @staticmethod
    def _types_compatible(
        source: str, target: str, labels_by_norm: dict[str, frozenset[str]]
    ) -> bool:
        """True if the two surfaces share at least one entity type (the same-type guard).

        Mirrors the cascade's per-surface label-set compatibility check (issue #35): two surfaces
        are type-compatible if their observed label sets intersect (or either is unlabeled). This
        stops a high-cosine but different-typed pair (e.g. an ORG surface vs a PERSON surface) from
        merging even when the embedding is close.
        """
        src_labels = labels_by_norm.get(source, frozenset())
        tgt_labels = labels_by_norm.get(target, frozenset())
        if not src_labels or not tgt_labels:
            return True
        return bool(src_labels & tgt_labels)

    # -- cluster assembly (mirrors the engine's ResolvedCluster shape) ------

    @staticmethod
    def _build_clusters(
        uf: _UnionFind,
        by_norm: dict[str, list[EntityMention]],
        display: dict[str, str],
        audits: list[MergeAudit],
    ) -> list[ResolvedCluster]:
        """Fold the Union-Find groups into :class:`~latence_core.disambiguation.ResolvedCluster`s.

        For each group (keyed by its lexicographic representative surface): canonical text is the
        representative's display spelling; the entity type is the majority member label; members are
        every mention id (sorted); documents the distinct document ids (sorted); the confidence the
        mean member confidence; the merge audits are those whose endpoints fell in this group. This
        is the exact :class:`~latence_core.disambiguation.ResolvedCluster` the cascade's engine
        emits, so the inherited record building consumes it unchanged.
        """
        groups: dict[str, list[str]] = {}
        for norm in by_norm:
            groups.setdefault(uf.find(norm), []).append(norm)

        # Attach each audit to its TARGET surface's representative cluster. An APPLIED audit's
        # endpoints share a group (they were unioned), so it lands on that cluster; a REJECTED audit
        # (endpoints in different clusters) is preserved on the target's cluster rather than dropped
        # so no below-policy / type-blocked decision is silently lost (no silent over-merge audit).
        audits_by_rep: dict[str, list[MergeAudit]] = {}
        for audit in audits:
            target_norm = normalize_text(audit.target_text)
            rep = uf.find(target_norm) if target_norm in by_norm else target_norm
            audits_by_rep.setdefault(rep, []).append(audit)

        clusters: list[ResolvedCluster] = []
        for rep, norms in groups.items():
            members = [m for norm in norms for m in by_norm[norm]]
            mention_ids = sorted(m.record_id for m in members)
            document_ids = sorted({m.provenance.document_id for m in members})
            labels = [m.label for m in members]
            entity_type = _majority(labels)
            confidence = (
                sum(m.confidence for m in members) / len(members) if members else 0.0
            )
            cluster_audits = audits_by_rep.get(rep, [])
            clusters.append(
                ResolvedCluster(
                    canonical_text=display[rep],
                    entity_type=entity_type,
                    member_mention_ids=mention_ids,
                    document_ids=document_ids,
                    merge_audits=cluster_audits,
                    confidence=round(confidence, 6),
                )
            )
        return clusters


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors (0.0 if either is degenerate/zero-length)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


def _majority(labels: list[str]) -> str:
    """The majority label (ties broken lexicographically); empty ⇒ a safe default."""
    if not labels:
        return "ENTITY"
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    # Sort by (-count, label) so the most frequent wins and ties break deterministically.
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
