"""Disambiguation engine — the ported entity-resolution / linking / normalisation assets (S8).

Ported faithfully (PORT-AS-IS, ASSET-INVENTORY.md) from three private
``latenceai-dataset-intelligence`` assets, re-homed here as one pure, contract-free,
CPU-viable, deterministic engine behind the single Disambiguation seam:

1. **EntityResolver** (``resolution/entity_resolver.py``, 5/5) — the **5-strategy
   cascade** (exact → alias → acronym → substring → embedding), a **Union-Find** merge
   map, and a **confidence-weighted, audited merge log** so nothing is silently
   over-merged. :class:`EntityResolver` is the port: it folds a stream of mention views
   into canonical clusters, recording every candidate merge (applied or not) as a
   :class:`MergeAudit`.

2. **GLinker linking pipeline** (``linking/backends.py``, 4.5/5) — the L2 candidate-gen →
   L3 biencoder → L4 rerank → L0 threshold cascade with a **KB-text fallback**. The CPU
   reference here keeps the *cascade shape* verbatim (candidate generation by
   normalised-token overlap, a deterministic lexical similarity standing in for the
   biencoder/rerank score, and an L0 acceptance threshold) so the seam and the audit are
   real; the FAISS/GPU biencoder is the pluggable Provider upgrade (S8 AC — the CPU path
   is required, the GPU path optional). :class:`KnowledgeBase` is the port.

3. **RelationNormalizer** (``relations/normalizer.py``, 4.5/5) — GLinker-style relation L2
   resolution: **fuzzy + alias** label normalisation, **type filtering**, and **inverse
   detection + direction swap**. :class:`RelationNormalizer` is the port.

This is a pure engine (no framework contracts, no IO). The reference Provider
(:mod:`latence_core.stages.disambiguation`) wraps it and stamps the contracts. It lives in
core (like :mod:`latence_core.chunking`, :mod:`latence_core.pagemap`, and
:mod:`latence_core.profiling`) because entity resolution is the algorithmic crown of the
"AI-ready" claim and a Provenance/Evidence concern the framework owns.

Determinism: the cascade folds mentions in a stable, sorted order; Union-Find always
points the cluster at the lexicographically smallest canonical text; every emitted list is
sorted. Identical input + config ⇒ byte-identical clusters (the Baseline bar).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

# Normalisation for the EXACT strategy and for token overlap: fold case, collapse
# internal whitespace, drop surrounding punctuation. Deterministic and dependency-free.
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Canonical-form a surface string for exact matching (Unicode + case + whitespace folded).

    Applies **Unicode NFKC** compatibility normalisation before case/whitespace folding so
    that *compatibility-confusable* surfaces — fullwidth Latin (``ＡＣＭＥ``), typographic
    ligatures (``ﬁle``), superscript/subscript letters, Roman-numeral glyphs (``Ⅳ`` → ``iv``)
    — canonicalise to the **same** key as their plain-ASCII spelling and therefore resolve
    into one canonical entity, closing the "an adversary hides a duplicate behind a
    look-alike glyph" split vector (issue #75). ``str.casefold`` (not ``str.lower``) is used
    so case folding is Unicode-correct (e.g. ``ß`` folds to ``ss``).

    NFKC deliberately does **not** fold *cross-script homoglyphs* — a Cyrillic ``А`` (U+0410)
    stays distinct from a Latin ``A`` (U+0041) because they are genuinely different
    characters. That asymmetry is the safe posture for this Stage: legitimate compatibility
    variants merge, but a mixed-script look-alike is never *silently over-merged* into an
    unrelated entity (S8 "no silent over-merge"). A future confusable-skeleton Provider can
    tighten homoglyph detection behind the same seam without changing this default.

    Determinism: NFKC and casefold are pure, table-driven Unicode operations — identical
    input ⇒ identical output, independent of platform (the Baseline bar).
    """
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text).strip().casefold())


def _tokens(text: str) -> list[str]:
    """NFKC-folded, casefolded word tokens (the token-overlap unit for candidate gen).

    Folds through the same NFKC + casefold pipeline as :func:`normalize_text` so token-level
    strategies (acronym, substring, embedding similarity, KB candidate generation) see
    compatibility-confusable surfaces identically to the exact strategy — otherwise a
    fullwidth-spelled name would tokenise differently from its ASCII twin and evade both the
    acronym and the overlap rungs. ``_WORD_RE`` matches Unicode word runs (letters/digits,
    underscore excluded), so non-Latin scripts tokenise too. Deterministic.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WORD_RE.findall(folded)


def acronym_of(text: str) -> str:
    """The initials of a multi-word name (``International Business Machines`` → ``ibm``).

    Fewer than two words has no acronym (returns ``""``), so a single word is never
    treated as its own one-letter acronym.
    """
    words = _tokens(text)
    if len(words) < 2:
        return ""
    return "".join(w[0] for w in words if w)


def token_similarity(a: str, b: str) -> float:
    """Deterministic lexical (Jaccard) similarity over word tokens, in ``[0, 1]``.

    The CPU stand-in for the GLinker biencoder/rerank score: the intersection-over-union
    of the two token sets. Identical strings score 1.0; disjoint strings 0.0. Dependency-
    free and order-independent, so a seeded run is byte-identical. The learned biencoder is
    the pluggable Provider upgrade behind the same seam (S8 AC).
    """
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


# -- the 5-strategy resolver cascade -----------------------------------------


class Strategy(StrEnum):
    """The resolver cascade rung that proposed a merge (mirrors the contract MergeStrategy)."""

    EXACT = "exact"
    ALIAS = "alias"
    ACRONYM = "acronym"
    SUBSTRING = "substring"
    EMBEDDING = "embedding"


@dataclass(frozen=True)
class MentionInput:
    """The minimal, contract-free view of an EntityMention the resolver folds in.

    The Provider projects each :class:`~latence_core.contracts.EntityMention` into this
    before resolution, keeping the engine IO/contract-independent.
    """

    mention_id: str
    document_id: str
    text: str
    label: str
    confidence: float


@dataclass(frozen=True)
class MergeAudit:
    """One audited merge decision the resolver proposed (applied or not).

    Mirrors the contract :class:`~latence_core.contracts.MergeDecision`. ``applied`` is
    False when the strategy's confidence fell below the policy — the merge is recorded but
    NOT folded (no silent over-merge, S8 AC).
    """

    source_text: str
    target_text: str
    strategy: Strategy
    confidence: float
    applied: bool
    reason: str


@dataclass
class ResolvedCluster:
    """A resolved canonical cluster: the engine output the Provider maps to a contract.

    ``canonical_text`` is the chosen surface (the lexicographically smallest normalised
    member, for determinism); ``entity_type`` the majority label; ``member_mention_ids``
    every mention folded in (sorted); ``document_ids`` the distinct documents it spans;
    ``merge_audits`` the decisions that built it; ``confidence`` the mean member confidence.
    """

    canonical_text: str
    entity_type: str
    member_mention_ids: list[str]
    document_ids: list[str]
    merge_audits: list[MergeAudit]
    confidence: float


class _UnionFind:
    """Union-Find over surface-string keys with a lexicographic-min representative.

    Ported from the EntityResolver merge map: near-linear union/find with path
    compression. The representative of a set is always its lexicographically smallest key,
    so the canonical text a cluster reports is deterministic regardless of merge order (the
    ``union`` direction is fixed by key ordering, not arrival order).
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, key: str) -> None:
        self._parent.setdefault(key, key)

    def find(self, key: str) -> str:
        self.add(key)
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression.
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Point the larger key at the smaller so the representative is the lexicographic
        # minimum — deterministic canonical selection, independent of call order.
        low, high = (ra, rb) if ra < rb else (rb, ra)
        self._parent[high] = low

    def groups(self) -> dict[str, list[str]]:
        """Map each representative to its members (sorted), for deterministic iteration."""
        out: dict[str, list[str]] = {}
        for key in self._parent:
            out.setdefault(self.find(key), []).append(key)
        for members in out.values():
            members.sort()
        return out


class EntityResolver:
    """The ported 5-strategy resolver cascade + Union-Find + audited merge log (S8).

    Folds a stream of :class:`MentionInput`s into canonical clusters. For each ordered pair
    of distinct surface strings the cascade tries, in descending precision:

    1. **exact** — identical :func:`normalize_text` forms (confidence 1.0).
    2. **alias** — a configured alias maps one surface to another (``alias_confidence``).
    3. **acronym** — one surface is the :func:`acronym_of` the other (``acronym_confidence``).
    4. **substring** — one normalised name is a whole-token substring of the other, gated by
       ``min_substring_ratio`` so ``a`` ⊄ ``about`` (score = length ratio).
    5. **embedding** — :func:`token_similarity` ≥ ``embedding_threshold`` (the CPU lexical
       stand-in for the pluggable vector Provider).

    The first strategy that *fires* (produces a candidate at all) decides the pair; if its
    confidence ≥ ``min_confidence`` the pair is unioned (``applied=True``), otherwise the
    decision is logged with ``applied=False`` and NOT merged — the "no silent over-merge"
    guarantee. Every fired decision is recorded in the audit log.

    **Precision guards on the two low-precision rungs (issue #35).** The acronym and
    substring strategies were the two rungs the original ``dataset-intelligence`` resolver
    gated and the first port dropped, letting them merge different-typed, unrelated surfaces
    at applied confidence ("IT" ↔ "Information Technology", "cat" ↔ "Central Atlantic
    Trust"). Restored here:

    * ``require_type_compatibility`` (**default on**) — the two surfaces must share at least
      one entity type (an untyped surface is treated as compatible). This is the guard that
      kills the issue's over-merge examples; it is safe to default on because it never blocks
      a legitimately same-typed cross-document merge (e.g. IBM ↔ International Business
      Machines, both ``ORG``).
    * ``require_shared_document`` (**default off**) — the stricter evidence requirement that
      the two surfaces co-occur in a common document. Left opt-in because it also blocks the
      *legitimate* cross-document merges the Stage exists to make (the same-type acronym of a
      name that only appears in a different document), so it is a per-adopter precision knob,
      not a default.

    A fired rung that fails an enabled guard is recorded as a below-guard decision
    (``applied=False``) — audited, not silently merged and not silently dropped.

    Determinism: surfaces are processed in sorted order and Union-Find picks the
    lexicographic-min canonical, so identical input yields byte-identical clusters.
    """

    def __init__(
        self,
        *,
        aliases: dict[str, str] | None = None,
        min_confidence: float = 0.82,
        alias_confidence: float = 0.95,
        acronym_confidence: float = 0.9,
        embedding_threshold: float = 0.75,
        min_substring_ratio: float = 0.5,
        enable_embedding: bool = True,
        require_type_compatibility: bool = True,
        require_shared_document: bool = False,
    ) -> None:
        # Aliases are normalised on both sides so lookup is case/whitespace-insensitive.
        self._aliases = {
            normalize_text(k): normalize_text(v) for k, v in (aliases or {}).items()
        }
        self._min_confidence = min_confidence
        self._alias_confidence = alias_confidence
        self._acronym_confidence = acronym_confidence
        self._embedding_threshold = embedding_threshold
        self._min_substring_ratio = min_substring_ratio
        self._enable_embedding = enable_embedding
        # Precision guards for the low-precision acronym + substring rungs (S8 port
        # fidelity, issue #35). The original dataset-intelligence EntityResolver gated
        # these two rungs on a same-entity-type compatibility check AND a shared-document
        # evidence requirement; the first port dropped both, letting acronym/substring
        # merge different-typed, semantically-unrelated surfaces ("IT" ↔ "Information
        # Technology", "cat" ↔ "Central Atlantic Trust"). Re-introduced here, on by
        # default, tunable off for the permissive original-cascade behaviour.
        self._require_type_compatibility = require_type_compatibility
        self._require_shared_document = require_shared_document
        self._mentions: list[MentionInput] = []

    def add(self, mention: MentionInput) -> None:
        """Fold one mention into the resolver's working set (streaming-friendly)."""
        self._mentions.append(mention)

    def add_all(self, mentions: list[MentionInput]) -> None:
        for mention in mentions:
            self.add(mention)

    def resolve(self) -> tuple[list[ResolvedCluster], list[MergeAudit]]:
        """Run the cascade over the folded mentions; return clusters + the full audit log.

        Returns the canonical clusters (sorted by canonical text) and the complete audit
        log (every fired decision, applied or not) so the Provider can attach the merge
        decisions to the entity and surface below-policy decisions in the Quality Report.
        """
        # Group mentions by their normalised surface, keeping a stable original spelling
        # (the first-seen, so the audit reads naturally). Sorted for determinism.
        by_norm: dict[str, list[MentionInput]] = {}
        display: dict[str, str] = {}
        for mention in self._mentions:
            norm = normalize_text(mention.text)
            if not norm:
                continue
            by_norm.setdefault(norm, []).append(mention)
            display.setdefault(norm, mention.text)

        # Per-surface evidence for the precision guards (issue #35): the distinct entity
        # types the surface was tagged with, and the distinct documents it appears in.
        labels_by_norm = {
            norm: frozenset(m.label for m in mentions) for norm, mentions in by_norm.items()
        }
        docs_by_norm = {
            norm: frozenset(m.document_id for m in mentions)
            for norm, mentions in by_norm.items()
        }

        surfaces = sorted(by_norm)
        uf = _UnionFind()
        for norm in surfaces:
            uf.add(norm)

        audits: list[MergeAudit] = []
        # Compare every ordered pair once (i < j), source=j folded into target=i.
        for i, target in enumerate(surfaces):
            for source in surfaces[i + 1 :]:
                decision = self._propose(
                    source, target, display, labels_by_norm, docs_by_norm
                )
                if decision is None:
                    continue
                audits.append(decision)
                if decision.applied:
                    uf.union(source, target)

        clusters = self._build_clusters(uf, by_norm, display, audits)
        clusters.sort(key=lambda c: c.canonical_text)
        return clusters, audits

    def _propose(
        self,
        source: str,
        target: str,
        display: dict[str, str],
        labels_by_norm: dict[str, frozenset[str]],
        docs_by_norm: dict[str, frozenset[str]],
    ) -> MergeAudit | None:
        """Try the cascade on one ordered pair; return the first fired decision, else None."""
        src_disp, tgt_disp = display[source], display[target]

        # 1. exact — normalised forms already equal would be the same surface key, so a
        #    distinct pair is never exactly equal here; kept for completeness/parity with
        #    the ported cascade (an EXACT tie is resolved by the by_norm grouping itself).

        # 2. alias
        if self._aliases.get(source) == target or self._aliases.get(target) == source:
            return self._decision(
                src_disp, tgt_disp, Strategy.ALIAS, self._alias_confidence,
                f"alias: {src_disp!r} ~ {tgt_disp!r}",
            )

        # 3. acronym — gated by the precision guards (issue #35): a same-type compatibility
        #    check and a shared-document-evidence requirement, both restored from the
        #    original asset. A fired acronym whose surfaces are type-incompatible or share
        #    no document is recorded as a below-guard decision (applied=False), never
        #    silently merged (no silent over-merge) and never silently dropped.
        if acronym_of(target) == source or acronym_of(source) == target:
            return self._guarded_decision(
                source, target, src_disp, tgt_disp, Strategy.ACRONYM,
                self._acronym_confidence, f"acronym: {src_disp!r} ~ {tgt_disp!r}",
                labels_by_norm, docs_by_norm,
            )

        # 4. substring (whole-token containment, length-ratio gated) — same precision guards.
        sub = self._substring_score(source, target)
        if sub is not None:
            return self._guarded_decision(
                source, target, src_disp, tgt_disp, Strategy.SUBSTRING, sub,
                f"substring: {src_disp!r} within {tgt_disp!r}",
                labels_by_norm, docs_by_norm,
            )

        # 5. embedding (lexical stand-in; inert when disabled — the no-embedder CPU path)
        if self._enable_embedding:
            sim = token_similarity(source, target)
            if sim >= self._embedding_threshold:
                return self._decision(
                    src_disp, tgt_disp, Strategy.EMBEDDING, round(sim, 6),
                    f"embedding: sim={round(sim, 6)}",
                )
        return None

    def _guarded_decision(
        self,
        source: str,
        target: str,
        src_disp: str,
        tgt_disp: str,
        strategy: Strategy,
        confidence: float,
        reason: str,
        labels_by_norm: dict[str, frozenset[str]],
        docs_by_norm: dict[str, frozenset[str]],
    ) -> MergeAudit:
        """Build a decision for a low-precision rung, applying the type + shared-doc guards.

        The acronym and substring rungs are the two low-precision strategies the original
        resolver gated (issue #35). A guard failure forces ``applied=False`` with an audit
        reason naming the failed guard — the merge is recorded (no silent over-merge) but
        NOT folded, restoring the original's precision without becoming a silent drop.
        """
        failed = self._guard_failure(source, target, labels_by_norm, docs_by_norm)
        if failed is None:
            return self._decision(src_disp, tgt_disp, strategy, confidence, reason)
        # A guard blocks the merge: record it as an audited, below-guard non-merge.
        return MergeAudit(
            source_text=src_disp,
            target_text=tgt_disp,
            strategy=strategy,
            confidence=confidence,
            applied=False,
            reason=f"{reason} [blocked: {failed}]",
        )

    def _guard_failure(
        self,
        source: str,
        target: str,
        labels_by_norm: dict[str, frozenset[str]],
        docs_by_norm: dict[str, frozenset[str]],
    ) -> str | None:
        """The name of the first precision guard the pair fails, or None if both pass.

        * **type compatibility** — the two surfaces must share at least one entity type
          (a surface with no recorded label is treated as compatible, so an untyped mention
          stream is not blocked wholesale).
        * **shared-document evidence** — the two surfaces must both appear in at least one
          common document, the original asset's evidence requirement for these rungs.
        """
        if self._require_type_compatibility:
            src_labels = labels_by_norm.get(source, frozenset())
            tgt_labels = labels_by_norm.get(target, frozenset())
            if src_labels and tgt_labels and src_labels.isdisjoint(tgt_labels):
                return "type-incompatible"
        if self._require_shared_document:
            src_docs = docs_by_norm.get(source, frozenset())
            tgt_docs = docs_by_norm.get(target, frozenset())
            if src_docs.isdisjoint(tgt_docs):
                return "no-shared-document"
        return None

    def _substring_score(self, source: str, target: str) -> float | None:
        """Whole-token substring containment score, or None if it does not apply.

        One name must be a contiguous whole-word run inside the other; the score is the
        shorter/longer token-count ratio, gated by ``min_substring_ratio`` so a tiny
        fragment (``a`` in ``a b c d``) does not merge. Token-based, so ``Acme`` matches
        ``Acme Corporation`` but not ``Acmestic``.
        """
        st, tt = _tokens(source), _tokens(target)
        if not st or not tt:
            return None
        short, long = (st, tt) if len(st) <= len(tt) else (tt, st)
        if not _contains_subsequence(long, short):
            return None
        ratio = len(short) / len(long)
        if ratio < self._min_substring_ratio:
            return None
        return round(ratio, 6)

    def _decision(
        self, source: str, target: str, strategy: Strategy, confidence: float, reason: str
    ) -> MergeAudit:
        applied = confidence >= self._min_confidence
        note = reason if applied else f"{reason} [below policy {self._min_confidence}]"
        return MergeAudit(
            source_text=source,
            target_text=target,
            strategy=strategy,
            confidence=confidence,
            applied=applied,
            reason=note,
        )

    def _build_clusters(
        self,
        uf: _UnionFind,
        by_norm: dict[str, list[MentionInput]],
        display: dict[str, str],
        audits: list[MergeAudit],
    ) -> list[ResolvedCluster]:
        """Assemble the Union-Find groups into ResolvedClusters with their audit slices."""
        clusters: list[ResolvedCluster] = []
        for root, members in sorted(uf.groups().items()):
            member_set = set(members)
            mentions: list[MentionInput] = []
            for norm in members:
                mentions.extend(by_norm.get(norm, []))
            if not mentions:
                continue
            labels = Counter(m.label for m in mentions)
            # Majority label; ties broken lexicographically for determinism.
            entity_type = min(labels.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            mention_ids = sorted(m.mention_id for m in mentions)
            doc_ids = sorted({m.document_id for m in mentions})
            confidence = round(sum(m.confidence for m in mentions) / len(mentions), 6)
            # A cluster carries two kinds of audit:
            #  * APPLIED merges whose source AND target both landed in this cluster — the
            #    decisions that actually built it;
            #  * BELOW-POLICY decisions whose TARGET surface is in this cluster — the merges
            #    that were proposed against this canonical surface but rejected by the policy.
            #    Attaching them here (deterministically, by the target's cluster) is what
            #    surfaces the "logged, not applied" audit trail (no silent over-merge, S8 AC);
            #    without it a below-policy decision — whose source stays in a different
            #    cluster — would be attached to neither and silently lost.
            cluster_audits = sorted(
                (
                    a
                    for a in audits
                    if (
                        a.applied
                        and normalize_text(a.target_text) in member_set
                        and normalize_text(a.source_text) in member_set
                    )
                    or (not a.applied and normalize_text(a.target_text) in member_set)
                ),
                key=lambda a: (a.applied, a.target_text, a.source_text),
            )
            clusters.append(
                ResolvedCluster(
                    canonical_text=display[root],
                    entity_type=entity_type,
                    member_mention_ids=mention_ids,
                    document_ids=doc_ids,
                    merge_audits=cluster_audits,
                    confidence=confidence,
                )
            )
        return clusters


def _contains_subsequence(haystack: list[str], needle: list[str]) -> bool:
    """True if ``needle`` appears as a contiguous run of whole tokens in ``haystack``."""
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i] == first and haystack[i : i + len(needle)] == needle:
            return True
    return False


# -- GLinker-style KB linking ------------------------------------------------


@dataclass(frozen=True)
class KBEntry:
    """One canonical entity in the external knowledge base."""

    kb_id: str
    name: str
    entity_type: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class LinkResult:
    """The outcome of linking one canonical surface against the KB.

    ``kb_id``/``kb_name`` are populated on a match above the L0 threshold; ``score`` is the
    lexical similarity; ``linked`` is False for the graceful fallback (no candidate cleared
    L0) — the mention keeps its resolved canonical form as an unlinked entity (S8 AC).
    """

    linked: bool
    score: float
    kb_id: str | None = None
    kb_name: str | None = None


class KnowledgeBase:
    """The ported GLinker linking cascade: L2 candidate-gen → L3/L4 score → L0 threshold.

    A CPU-viable reference over an in-memory KB (a list of :class:`KBEntry`). The cascade
    shape is preserved verbatim from ``linking/backends.py``:

    * **L2 candidate generation** — restrict to KB entries sharing at least one normalised
      token with the query (an inverted-index lookup), so scoring is not O(KB) per query.
    * **L3 biencoder / L4 rerank** — score each candidate by :func:`token_similarity`
      against the entry name *and* its aliases, keeping the max (the CPU lexical stand-in
      for the learned biencoder+rerank; the FAISS/GPU scorer is the pluggable upgrade).
    * **L0 threshold** — accept the top candidate only if its score ≥ ``threshold``;
      otherwise return the **KB-text fallback** (``linked=False``) — an unlinked entity, not
      a forced wrong link.

    Type filtering: when the query carries an entity type and the candidate declares one,
    a mismatch is rejected (the ported type-filter guard). An empty KB links nothing (every
    query falls back), which is the valid "configurable KB with graceful fallback" path.
    """

    def __init__(self, entries: list[KBEntry] | None = None, *, threshold: float = 0.5) -> None:
        self._entries = list(entries or [])
        self._threshold = threshold
        # L2 inverted index: normalised token -> candidate entry indices.
        self._index: dict[str, set[int]] = {}
        for idx, entry in enumerate(self._entries):
            for surface in (entry.name, *entry.aliases):
                for tok in _tokens(surface):
                    self._index.setdefault(tok, set()).add(idx)

    def link(self, text: str, entity_type: str | None = None) -> LinkResult:
        """Link one surface to the KB via the L2→L3/L4→L0 cascade, or fall back."""
        if not self._entries:
            return LinkResult(linked=False, score=0.0)
        # L2: candidate generation by shared token.
        candidate_ids: set[int] = set()
        for tok in _tokens(text):
            candidate_ids |= self._index.get(tok, set())
        if not candidate_ids:
            return LinkResult(linked=False, score=0.0)

        # L3/L4: score candidates; deterministic tie-break by (score desc, kb_id).
        best: tuple[float, str, str] | None = None
        for idx in sorted(candidate_ids):
            entry = self._entries[idx]
            if (
                entity_type is not None
                and entry.entity_type is not None
                and entity_type != entry.entity_type
            ):
                continue  # type filter
            score = max(
                token_similarity(text, surface) for surface in (entry.name, *entry.aliases)
            )
            score = round(score, 6)
            candidate = (score, entry.kb_id, entry.name)
            if best is None or (candidate[0], -_rank(candidate[1])) > (best[0], -_rank(best[1])):
                best = candidate
        if best is None:
            return LinkResult(linked=False, score=0.0)
        score, kb_id, kb_name = best
        # L0: threshold. Below it, the KB-text fallback (unlinked, not force-linked).
        if score < self._threshold:
            return LinkResult(linked=False, score=score)
        return LinkResult(linked=True, score=score, kb_id=kb_id, kb_name=kb_name)


def _rank(kb_id: str) -> int:
    """Stable secondary key so equal-score candidates tie-break deterministically."""
    return sum(ord(c) for c in kb_id)


# -- relation normalisation --------------------------------------------------


@dataclass(frozen=True)
class RelationInput:
    """The contract-free view of a RelationMention the normalizer folds in."""

    relation_id: str
    document_id: str
    label: str
    head_mention_id: str
    tail_mention_id: str
    head_text: str
    tail_text: str
    confidence: float


@dataclass(frozen=True)
class NormalizedRelationValues:
    """The engine output for one normalised relation (contract-free)."""

    source_relation_id: str
    label: str
    head_mention_id: str
    tail_mention_id: str
    head_text: str
    tail_text: str
    confidence: float
    inverted: bool


class RelationNormalizer:
    """The ported RelationNormalizer: label alias/fuzzy resolution + inverse direction swap.

    Ported from ``relations/normalizer.py`` (GLinker-style relation L2 resolution):

    * **alias + fuzzy label normalisation** — a configured ``label_aliases`` map folds a
      surface label to its canonical (``employed_by`` → ``works_for``); an unmapped label
      is fuzzy-matched to the closest configured canonical via :func:`token_similarity`
      above ``label_threshold``, else kept as-is (lowercased).
    * **type filtering** — when ``allowed_labels`` is configured, a relation whose
      normalised label is not in the set is dropped (returns ``None``), so the graph only
      carries the schema's relation types.
    * **inverse detection + direction swap** — when the normalised label is a configured
      inverse (``inverses`` maps ``employs`` ↔ ``works_for``), the head/tail are swapped and
      the canonical label substituted, so the graph stores one canonical direction.
    """

    def __init__(
        self,
        *,
        label_aliases: dict[str, str] | None = None,
        inverses: dict[str, str] | None = None,
        allowed_labels: set[str] | None = None,
        label_threshold: float = 0.6,
    ) -> None:
        self._label_aliases = {
            _norm_label(k): _norm_label(v) for k, v in (label_aliases or {}).items()
        }
        # inverses maps a surface label -> (canonical_label). Presence means "this label is
        # the inverse direction; rewrite to canonical_label and swap head/tail".
        self._inverses = {_norm_label(k): _norm_label(v) for k, v in (inverses or {}).items()}
        self._allowed = {_norm_label(x) for x in allowed_labels} if allowed_labels else None
        self._label_threshold = label_threshold
        self._canonical_vocab = sorted(
            set(self._label_aliases.values())
            | set(self._inverses.values())
            | (self._allowed or set())
        )

    def normalize(self, relation: RelationInput) -> NormalizedRelationValues | None:
        """Normalise one relation's label + direction, or drop it (type filter).

        Returns ``None`` when the type filter rejects the normalised label — a signal to
        the Provider to drop the relation and count it, never to emit a mis-typed relation.
        """
        label = _norm_label(relation.label)
        label = self._resolve_label(label)

        inverted = False
        head_id, tail_id = relation.head_mention_id, relation.tail_mention_id
        head_text, tail_text = relation.head_text, relation.tail_text
        if label in self._inverses:
            label = self._inverses[label]
            head_id, tail_id = tail_id, head_id
            head_text, tail_text = tail_text, head_text
            inverted = True

        if self._allowed is not None and label not in self._allowed:
            return None

        return NormalizedRelationValues(
            source_relation_id=relation.relation_id,
            label=label,
            head_mention_id=head_id,
            tail_mention_id=tail_id,
            head_text=head_text,
            tail_text=tail_text,
            confidence=relation.confidence,
            inverted=inverted,
        )

    def _resolve_label(self, label: str) -> str:
        """Alias-map a label, else fuzzy-match to the closest canonical, else keep it."""
        if label in self._label_aliases:
            return self._label_aliases[label]
        if not self._canonical_vocab:
            return label
        best_label, best_score = label, 0.0
        for canon in self._canonical_vocab:
            score = token_similarity(label, canon)
            if score > best_score:
                best_label, best_score = canon, score
        if best_score >= self._label_threshold:
            return best_label
        return label


def _norm_label(label: str) -> str:
    """Canonical-form a relation label (lowercase, whitespace→underscore)."""
    return _WS_RE.sub("_", label.strip().lower())
