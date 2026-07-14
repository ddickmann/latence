"""EntityResolver — cross-document entity fusion (the audited precision-first cascade).

Ported (PORT-AS-IS) from ``dataset_intelligence.resolution.entity_resolver``. After entity linking
assigns ``canonical_entity_id`` to mentions, the resolver fuses canonical entities that are
semantically equivalent but the linker missed, via a cascade in descending precision:

  1. exact case-insensitive name match             (confidence 1.0)
  2. alias overlap                                  (0.9)
  3. acronym expansion (same-label + initials)      (0.85)
  4. substring containment (same-label + shared docs)   (0.7)
  5. fuzzy edit-distance (same-label + high char-similarity ≥ ``min_similarity``)   — DEFAULT ON
  6. embedding-by-label (Granite STS; brute-force, FAISS-GPU for groups > 1000)     — DEFAULT OFF

Transitive merges are collapsed (A->B, B->C => A->C) with cycle detection. Every merge is recorded
as a :class:`MergeDecision` with reason + confidence + provenance so nothing is silently
over-merged (the §19 HARD REQUIREMENT). Type-constrained: acronym/substring/fuzzy never cross
labels; the embedding rung groups by label so cross-type merges are impossible.

PRECISION POSTURE (ER audit). The embedding-by-label rung is **OFF by default**: on a real corpus of
short entity names Granite's retrieval embeddings put distinct short names (person/date/org) too
close together (wrong-merge cosines 0.85-0.89 overlap the right 0.86-0.95), so there is no clean
threshold and it catastrophically over-merges (every person collapsing into one). Recall is carried
instead by the neural linker + the string cascade + the conservative fuzzy rung (5); the embedding
rung is an explicit opt-in and, when enabled, defaults to a safe ≥0.93 cutoff and REQUIRES a
discriminative (non-Granite-retrieval) STS embedder — the retrieval-oriented Granite r2 is unsafe
for it. The fuzzy rung (5) recovers OCR/spelling variants (``Kai Kölsch`` ↔ ``Kai Kolsch``) that the
string rungs cannot (not exact/substring) and the embedding rung must not (it over-merges).

DISTINGUISHING-TOKEN GUARD (fuzzy rung, pod-observed on a German legal/shareholder corpus). A
threshold alone cannot separate an OCR spelling-variant from two entities that differ only in a
*distinguishing token* — the char-ratio of ``K² Ventures UG`` vs ``D² Ventures UG`` (0.93) is HIGHER
than the legit ``Kai Kusch`` vs ``Kai Kölsch`` variant (0.84). So before a fuzzy merge is allowed
(same-label + ratio ≥ ``min_similarity``) the rung additionally requires BOTH sub-guards to pass,
each individually testable: (1) :meth:`_numeric_tokens` — the numeric tokens of the two surfaces
must be EQUAL (``§ 3 Abs. 2 GwG`` → {3,2} vs ``§ 11 Abs. 6 GwG`` → {11,6} ⇒ BLOCK; distinct legal
paragraphs); (2) :meth:`_leading_code_differs` — the leading token must not be a distinct short code
(``K²``/``D²`` → ``k2``/``d2`` differ in the leading char at equal length ⇒ BLOCK; distinct orgs).
``Kai Kölsch`` vs ``Kai Kolsch`` has no numbers ({} == {}) and shares the leading token ``Kai`` ⇒
both sub-guards pass ⇒ the OCR variant still merges. The guard only ever BLOCKS a merge — it never
creates one — so it cannot introduce a false merge, only prevent one. The tuned default
``min_similarity`` is 0.84 (admits the legit ``Kai Kusch`` variant) but the guard, not the
threshold, is the real fix for the numeric/code confusions.

DI adaptation vs the source: the STS embedder is the injected framework
:class:`~latence_disambig_glinker.embedder.LabelEncoder` (Granite r2, ADR-0045), not
``ontology.embeddings.embed_labels`` (EmbeddingGemma/vllm-factory). FAISS is an OPTIONAL dep — the
FAISS-GPU path is used for label groups > 1000 and falls back to brute-force when ``faiss`` is
absent. The merge ALGORITHM is byte-for-byte the enterprise one (plus the new fuzzy rung).
"""

from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from latence_disambig_glinker.embedder import cosine
from latence_disambig_glinker.models import CanonicalEntityRecord, MentionRecord

if TYPE_CHECKING:
    from collections.abc import Callable

    from latence_disambig_glinker.embedder import LabelEncoder

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\d+")


def _default_name_key(text: str) -> str:
    """NFKC + casefold + whitespace-fold — byte-identical to ``latence_core.disambiguation``'s.

    The resolver groups canonical names by this key so its clusters agree with the framework's
    content-addressed id derivation ``entity:{normalize_text(canonical_text)}`` (A-F1). The Provider
    injects the framework's own ``normalize_text`` (agreement guaranteed by construction — same
    callable); this default replicates it (Unicode NFKC compatibility fold + ``str.casefold`` +
    whitespace collapse) so the engine stays framework-independent when used standalone. Confusable
    surfaces — the ligature ``ﬁle`` vs ``file``, fullwidth Latin, super/subscripts — fold to ONE
    key, closing the "same framework record_id, dropped provenance" collision the audit found.
    """
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text).strip().casefold())


@dataclass
class MergeDecision:
    """Record of a merge decision for auditability (§19: no silent over-merge)."""

    source_entity_id: str
    target_entity_id: str
    reason: str
    confidence: float
    source_name: str = ""
    target_name: str = ""


class EntityResolver:
    """Fuses semantically-equivalent entities across documents (audited).

    Args:
        acronym_expand: whether to attempt acronym expansion.
        use_fuzzy_merge: enable the conservative fuzzy edit-distance rung (DEFAULT ON — it is
            precision-safe at ``min_similarity`` on same-label names and is the OCR/spelling-variant
            recall the string rungs and the (off-by-default) embedding rung cannot supply).
        min_similarity: normalized char-similarity cutoff (:func:`difflib` ratio, in ``[0, 1]``) for
            the fuzzy rung (default ``0.84`` — admits the legit ``Kai Kusch`` ↔ ``Kai Kölsch`` OCR
            variant; the distinguishing-token guard, not the threshold, blocks the numeric/code
            confusions). Conservative — same-label only.
        fuzzy_shared_doc_guard: additionally require the two entities to share a document before a
            fuzzy merge (default ``False``: OCR variants often live in different docs).
        embedding_merge_threshold: cosine cutoff for the embedding-by-label rung. Default ``0.93``
            (safe floor). WARNING: this rung is precision-unsafe with a retrieval-oriented embedder
            (Granite r2 puts distinct short names within 0.85–0.89) — enable it ONLY with a
            discriminative STS embedder, never the corpus retrieval embedder.
        use_embedding_merge: enable the Granite STS embedding rung. DEFAULT ``False`` (see the class
            docstring PRECISION POSTURE — it catastrophically over-merges short names on real data).
        label_encoder: the framework STS embedder (Granite); ``None`` disables the embedding rung.
        name_normalizer: the surface→key normaliser for the exact-name/alias grouping rungs. The
            Provider injects the framework's ``normalize_text`` so the resolver's clusters agree
            with the framework record_id derivation (A-F1); ``None`` uses the byte-identical
            in-engine default (:func:`_default_name_key`).
        faiss_group_threshold: label-group size above which the embedding rung switches from
            brute-force O(n^2) to the FAISS-GPU O(n*k) path (test-injectable; default 1000).
    """

    # Group size (entities per label) above which we switch from brute-force O(n^2) to FAISS-GPU.
    _FAISS_GROUP_THRESHOLD = 1000

    def __init__(
        self,
        acronym_expand: bool = True,
        use_fuzzy_merge: bool = True,
        min_similarity: float = 0.84,
        fuzzy_shared_doc_guard: bool = False,
        embedding_merge_threshold: float = 0.93,
        use_embedding_merge: bool = False,
        label_encoder: LabelEncoder | None = None,
        name_normalizer: Callable[[str], str] | None = None,
        faiss_group_threshold: int = _FAISS_GROUP_THRESHOLD,
    ) -> None:
        self.acronym_expand = acronym_expand
        self.use_fuzzy_merge = use_fuzzy_merge
        self.min_similarity = min_similarity
        self.fuzzy_shared_doc_guard = fuzzy_shared_doc_guard
        self.embedding_merge_threshold = embedding_merge_threshold
        self.use_embedding_merge = use_embedding_merge and label_encoder is not None
        self._label_encoder = label_encoder
        self._normalize = name_normalizer or _default_name_key
        self._faiss_group_threshold = faiss_group_threshold
        self.merge_log: list[MergeDecision] = []

    def resolve(
        self,
        entities: list[CanonicalEntityRecord],
        mentions: list[MentionRecord],
        dataset_id: str,
    ) -> tuple[list[CanonicalEntityRecord], list[MentionRecord]]:
        """Run entity resolution across all canonical entities.

        Returns ``(resolved_entities, updated_mentions)`` with merged duplicates.
        """
        self.merge_log.clear()
        if not entities:
            return entities, mentions

        merge_map: dict[str, str] = {}  # old_id -> canonical_id

        self._merge_by_exact_name(entities, merge_map)
        self._merge_by_aliases(entities, merge_map)
        if self.acronym_expand:
            self._merge_by_acronym(entities, merge_map)
        self._merge_by_substring(entities, merge_map)
        if self.use_fuzzy_merge:
            self._merge_by_fuzzy(entities, merge_map)
        if self.use_embedding_merge:
            self._merge_by_embedding(entities, merge_map)

        if not merge_map:
            logger.info("Entity resolution: no merges needed")
            return entities, mentions

        resolved_entities, survivor_map = self._apply_entity_merges(entities, merge_map, dataset_id)
        updated_mentions = self._update_mention_ids(mentions, survivor_map)
        logger.info(
            "Entity resolution: %d entities -> %d after %d merges",
            len(entities),
            len(resolved_entities),
            len(self.merge_log),
        )
        return resolved_entities, updated_mentions

    # -- strategies ---------------------------------------------------------

    def _merge_by_exact_name(
        self, entities: list[CanonicalEntityRecord], merge_map: dict[str, str]
    ) -> None:
        """Merge entities with identical canonical_name (framework-normalised, confusable-safe).

        Uses :attr:`_normalize` (the framework's ``normalize_text`` when Provider-injected) so
        confusable surfaces collapse to one cluster BEFORE the framework derives its record_id from
        the survivor's ``canonical_name`` — otherwise 'ﬁle' and 'file' survive as two clusters that
        the framework then maps to the SAME ``entity:file`` id, dropping one's provenance (A-F1).
        """
        by_name: dict[str, list[CanonicalEntityRecord]] = defaultdict(list)
        for e in entities:
            by_name[self._normalize(e.canonical_name)].append(e)
        for _name, group in by_name.items():
            if len(group) <= 1:
                continue
            group.sort(key=lambda e: e.evidence_count, reverse=True)
            canonical = group[0]
            for dup in group[1:]:
                if dup.entity_id not in merge_map:
                    merge_map[dup.entity_id] = canonical.entity_id
                    self.merge_log.append(
                        MergeDecision(
                            source_entity_id=dup.entity_id,
                            target_entity_id=canonical.entity_id,
                            reason="exact_name_match",
                            confidence=1.0,
                            source_name=dup.canonical_name,
                            target_name=canonical.canonical_name,
                        )
                    )

    def _merge_by_aliases(
        self, entities: list[CanonicalEntityRecord], merge_map: dict[str, str]
    ) -> None:
        """Merge entities where one's name appears in another's alias list."""
        name_to_entity: dict[str, CanonicalEntityRecord] = {}
        for e in entities:
            if e.entity_id not in merge_map:
                name_to_entity[self._normalize(e.canonical_name)] = e

        for e in entities:
            if e.entity_id in merge_map:
                continue
            for alias in e.aliases:
                alias_lower = self._normalize(alias)
                if alias_lower in name_to_entity:
                    target = name_to_entity[alias_lower]
                    if target.entity_id != e.entity_id and target.entity_id not in merge_map:
                        if e.evidence_count >= target.evidence_count:
                            merge_map[target.entity_id] = e.entity_id
                            self.merge_log.append(
                                MergeDecision(
                                    source_entity_id=target.entity_id,
                                    target_entity_id=e.entity_id,
                                    reason="alias_overlap",
                                    confidence=0.9,
                                    source_name=target.canonical_name,
                                    target_name=e.canonical_name,
                                )
                            )
                        else:
                            merge_map[e.entity_id] = target.entity_id
                            self.merge_log.append(
                                MergeDecision(
                                    source_entity_id=e.entity_id,
                                    target_entity_id=target.entity_id,
                                    reason="alias_overlap",
                                    confidence=0.9,
                                    source_name=e.canonical_name,
                                    target_name=target.canonical_name,
                                )
                            )
                            break

    def _merge_by_acronym(
        self, entities: list[CanonicalEntityRecord], merge_map: dict[str, str]
    ) -> None:
        """Merge entities where one is an acronym of the other (same label type)."""
        short_entities = [
            e
            for e in entities
            if e.entity_id not in merge_map
            and len(e.canonical_name) <= 6
            and e.canonical_name.isupper()
        ]
        long_entities = [
            e for e in entities if e.entity_id not in merge_map and len(e.canonical_name) > 6
        ]

        for short in short_entities:
            acronym = short.canonical_name.strip().upper()
            for long_ent in long_entities:
                if long_ent.entity_id in merge_map:
                    continue
                if short.label != long_ent.label:  # type-constrained
                    continue
                words = re.findall(r"\b[A-ZÄÖÜ]\w*", long_ent.canonical_name)
                initials = "".join(w[0].upper() for w in words if w)
                if initials == acronym:
                    merge_map[short.entity_id] = long_ent.entity_id
                    self.merge_log.append(
                        MergeDecision(
                            source_entity_id=short.entity_id,
                            target_entity_id=long_ent.entity_id,
                            reason="acronym_expansion",
                            confidence=0.85,
                            source_name=short.canonical_name,
                            target_name=long_ent.canonical_name,
                        )
                    )
                    break

    def _merge_by_substring(
        self, entities: list[CanonicalEntityRecord], merge_map: dict[str, str]
    ) -> None:
        """Merge short whole-word substrings of longer same-label names sharing a document."""
        active = [e for e in entities if e.entity_id not in merge_map]
        active.sort(key=lambda e: len(e.canonical_name))

        for i, short in enumerate(active):
            if short.entity_id in merge_map:
                continue
            short_lower = short.canonical_name.strip().lower()
            if len(short_lower) < 3:
                continue
            for long_ent in active[i + 1 :]:
                if long_ent.entity_id in merge_map:
                    continue
                if short.label != long_ent.label:  # type-constrained
                    continue
                long_lower = long_ent.canonical_name.strip().lower()
                if long_lower.startswith(short_lower + " ") or long_lower.endswith(
                    " " + short_lower
                ):
                    shared_docs = set(short.document_ids) & set(long_ent.document_ids)
                    if shared_docs:
                        merge_map[short.entity_id] = long_ent.entity_id
                        self.merge_log.append(
                            MergeDecision(
                                source_entity_id=short.entity_id,
                                target_entity_id=long_ent.entity_id,
                                reason="substring_containment",
                                confidence=0.7,
                                source_name=short.canonical_name,
                                target_name=long_ent.canonical_name,
                            )
                        )
                        break

    @staticmethod
    def _numeric_tokens(surface: str) -> frozenset[str]:
        """Numeric tokens of a surface (distinguishing-token guard #1) — set semantics.

        NFKC-fold first so superscripts/fullwidth digits become ASCII (``§ 3`` → ``3``, ``K²`` →
        ``K2``), then collect every digit run. Two fuzzy candidates may merge ONLY if these sets are
        EQUAL: ``§ 3 Abs. 2 GwG`` → {'3','2'} vs ``§ 11 Abs. 6 GwG`` → {'11','6'} differ ⇒ BLOCK
        (distinct legal paragraphs); ``Kai Kölsch`` → {} equals ``Kai Kolsch`` → {} ⇒ allow.
        """
        return frozenset(_NUM_RE.findall(unicodedata.normalize("NFKC", surface)))

    @staticmethod
    def _leading_code_differs(surface_a: str, surface_b: str) -> bool:
        """True if the two leading tokens are distinct SHORT CODES (distinguishing-token guard #2).

        Fires ONLY when the first whitespace-token of each NFKC-folded surface has the SAME length,
        differs ONLY in its leading character, and looks like a short discriminating code — ≤3 chars
        (incl. superscripts, which NFKC folds into the count) OR all-caps. ``K² Ventures UG`` vs
        ``D² Ventures UG`` → leading ``K2``/``D2`` differ in the leading char at equal length,
        ≤3 chars ⇒ True (BLOCK, distinct orgs). ``Kai …`` vs ``Kai …`` share the leading token ⇒
        False (allow the OCR merge). A whole-word length mismatch (``Notare`` vs ``Notar``) or a
        non-leading difference (``Müller`` vs ``Muller``) ⇒ False (allow).
        """
        tokens_a = unicodedata.normalize("NFKC", surface_a).strip().split()
        tokens_b = unicodedata.normalize("NFKC", surface_b).strip().split()
        if not tokens_a or not tokens_b:
            return False
        a, b = tokens_a[0], tokens_b[0]
        if a == b or len(a) != len(b):
            return False
        if a[0] == b[0] or a[1:] != b[1:]:  # must differ ONLY in the leading character
            return False
        looks_like_code = len(a) <= 3 or (a.isupper() and b.isupper())
        return looks_like_code

    def _fuzzy_guard_blocks(self, surface_a: str, surface_b: str) -> bool:
        """Distinguishing-token guard: BLOCK a fuzzy merge whose surfaces differ in a number/code.

        Conservative and BLOCK-ONLY (never creates a merge): a fuzzy merge is vetoed if the numeric
        tokens differ (guard #1) OR the leading token is a distinct short code (guard #2). Both
        sub-guards are individually testable (:meth:`_numeric_tokens` and
        :meth:`_leading_code_differs`).
        """
        if self._numeric_tokens(surface_a) != self._numeric_tokens(surface_b):
            return True
        return self._leading_code_differs(surface_a, surface_b)

    def _merge_by_fuzzy(
        self, entities: list[CanonicalEntityRecord], merge_map: dict[str, str]
    ) -> None:
        """Merge same-label names with high normalized char-similarity (OCR/spelling variants).

        The conservative fuzzy rung (ER audit B-F4: the source documented ``min_similarity`` but
        left it a DEAD knob). It recovers what the string cascade cannot (``Kai Kölsch`` vs
        ``Kai Kolsch`` is neither exact nor a whole-word substring) and the embedding rung must not
        (Granite over-merges distinct short names). Same-label ONLY (type-constrained), gated by the
        :func:`difflib` ratio on the framework-normalised surfaces >= :attr:`min_similarity`
        (default 0.84 — high enough that ``Kai …`` never folds into ``Dennis Dickmann`` /
        ``Max Mustermann``, whose ratios sit far below, low enough to admit the legit ``Kai Kusch``
        variant at 0.84). Optionally also requires a shared doc (:attr:`fuzzy_shared_doc_guard`).

        The DISTINGUISHING-TOKEN GUARD (:meth:`_fuzzy_guard_blocks`) then blocks any merge whose
        surfaces differ in a number (``§ 3 Abs. 2 GwG`` vs ``§ 11 Abs. 6 GwG``) or a leading short
        code (``K² Ventures UG`` vs ``D² Ventures UG``) — cases the ratio alone cannot separate. It
        only BLOCKS, never creates, a merge. Writes into ``merge_map`` so the transitive-collapse in
        :meth:`_apply_entity_merges` bridges a variant chain into ONE survivor, and audits each
        merge (``reason="fuzzy_edit_distance"``, confidence = the similarity).
        """
        active = [e for e in entities if e.entity_id not in merge_map]
        by_label: dict[str, list[CanonicalEntityRecord]] = defaultdict(list)
        for e in active:
            by_label[e.label.lower()].append(e)

        merges = 0
        for _label, group in by_label.items():
            if len(group) < 2:
                continue
            # Highest-evidence first: a variant folds into the best-attested survivor of the group.
            group.sort(key=lambda e: e.evidence_count, reverse=True)
            keys = [self._normalize(e.canonical_name) for e in group]
            for i in range(len(group)):
                if group[i].entity_id in merge_map:
                    continue
                for j in range(i + 1, len(group)):
                    if group[j].entity_id in merge_map:
                        continue
                    sim = difflib.SequenceMatcher(None, keys[i], keys[j]).ratio()
                    if sim < self.min_similarity:
                        continue
                    if self.fuzzy_shared_doc_guard and not (
                        set(group[i].document_ids) & set(group[j].document_ids)
                    ):
                        continue
                    if self._fuzzy_guard_blocks(
                        group[i].canonical_name, group[j].canonical_name
                    ):
                        continue  # numeric/leading-code distinguisher differs — not a variant
                    merge_map[group[j].entity_id] = group[i].entity_id
                    self.merge_log.append(
                        MergeDecision(
                            source_entity_id=group[j].entity_id,
                            target_entity_id=group[i].entity_id,
                            reason="fuzzy_edit_distance",
                            confidence=sim,
                            source_name=group[j].canonical_name,
                            target_name=group[i].canonical_name,
                        )
                    )
                    merges += 1
        if merges:
            logger.info(
                "Fuzzy edit-distance entity merges: %d (min_similarity=%.2f)",
                merges,
                self.min_similarity,
            )

    def _merge_by_embedding(
        self, entities: list[CanonicalEntityRecord], merge_map: dict[str, str]
    ) -> None:
        """Merge entities with high Granite-STS embedding similarity, grouped by label type.

        Groups by label (prevents cross-type merges). For large groups (>1000) uses a FAISS-GPU
        index for O(n*k) search; else brute-force O(n^2). FAISS is optional — absent, it falls back
        to brute-force.
        """
        if self._label_encoder is None:
            return
        active = [e for e in entities if e.entity_id not in merge_map]
        if len(active) < 2:
            return

        by_label: dict[str, list[CanonicalEntityRecord]] = defaultdict(list)
        for e in active:
            by_label[e.label.lower()].append(e)

        total = 0
        for _label, group in by_label.items():
            if len(group) < 2:
                continue
            texts = [" | ".join([e.canonical_name, *e.aliases[:3]]) for e in group]
            embeddings = self._label_encoder.embed_labels(texts)
            if len(group) >= self._faiss_group_threshold:
                total += self._faiss_gpu_merge(group, embeddings, merge_map)
            else:
                total += self._brute_force_merge(group, embeddings, merge_map)
        if total > 0:
            logger.info(
                "Embedding-based entity merges: %d (threshold=%.2f)",
                total,
                self.embedding_merge_threshold,
            )

    def _brute_force_merge(
        self,
        group: list[CanonicalEntityRecord],
        embeddings: list[list[float]],
        merge_map: dict[str, str],
    ) -> int:
        """O(n^2) pairwise similarity — used for small groups (pure-Python, numpy-free)."""
        merges = 0
        n = len(group)
        for i in range(n):
            if group[i].entity_id in merge_map:
                continue
            for j in range(i + 1, n):
                if group[j].entity_id in merge_map:
                    continue
                sim = cosine(embeddings[i], embeddings[j])
                if sim >= self.embedding_merge_threshold:
                    if group[i].evidence_count >= group[j].evidence_count:
                        src, tgt = group[j], group[i]
                    else:
                        src, tgt = group[i], group[j]
                    merge_map[src.entity_id] = tgt.entity_id
                    self.merge_log.append(
                        MergeDecision(
                            source_entity_id=src.entity_id,
                            target_entity_id=tgt.entity_id,
                            reason="embedding_similarity",
                            confidence=sim,
                            source_name=src.canonical_name,
                            target_name=tgt.canonical_name,
                        )
                    )
                    merges += 1
        return merges

    def _faiss_gpu_merge(
        self,
        group: list[CanonicalEntityRecord],
        embeddings: list[list[float]],
        merge_map: dict[str, str],
    ) -> int:
        """FAISS-GPU approximate-NN merge for large groups (O(n*k)); brute-force if faiss absent.

        numpy enters ONLY here (with faiss) — both OPTIONAL/GPU-only — so the CPU-reachable path
        stays numpy-free. faiss (or numpy) absent ⇒ graceful brute-force fallback (never a crash).
        """
        try:
            import faiss
            import numpy as np
        except ImportError:
            logger.warning(
                "faiss/numpy absent — brute-force for %d entities", len(group)
            )
            return self._brute_force_merge(group, embeddings, merge_map)

        n = len(group)
        emb_f32 = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
        d = emb_f32.shape[1]
        k = min(20, n)

        cpu_index = faiss.IndexFlatIP(d)
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            res.setTempMemory(128 * 1024 * 1024)
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            logger.info("No GPU available for FAISS — using CPU index")
            index = cpu_index

        index.add(emb_f32)
        similarities, neighbors = index.search(emb_f32, k)

        merges = 0
        for i in range(n):
            if group[i].entity_id in merge_map:
                continue
            for rank in range(1, k):  # skip self at rank 0
                j = int(neighbors[i, rank])
                if j < 0 or j == i:
                    continue
                sim = float(similarities[i, rank])
                if sim < self.embedding_merge_threshold:
                    break  # sorted descending
                if group[j].entity_id in merge_map:
                    continue
                if group[i].evidence_count >= group[j].evidence_count:
                    src, tgt = group[j], group[i]
                else:
                    src, tgt = group[i], group[j]
                merge_map[src.entity_id] = tgt.entity_id
                self.merge_log.append(
                    MergeDecision(
                        source_entity_id=src.entity_id,
                        target_entity_id=tgt.entity_id,
                        reason="embedding_similarity_faiss",
                        confidence=sim,
                        source_name=src.canonical_name,
                        target_name=tgt.canonical_name,
                    )
                )
                merges += 1
        logger.info("FAISS GPU merge: %d entities, k=%d, %d merges", n, k, merges)
        return merges

    # -- incremental resolution --------------------------------------------

    def resolve_incremental(
        self,
        new_entities: list[CanonicalEntityRecord],
        existing_entities: list[CanonicalEntityRecord],
        mentions: list[MentionRecord],
        dataset_id: str,
        cached_embeddings: dict[str, list[float]] | None = None,
    ) -> tuple[list[CanonicalEntityRecord], list[MentionRecord], list[MergeDecision]]:
        """Incremental resolution: compare new entities against existing only (O(N*M)).

        Existing entities are not re-compared against each other (already resolved).
        Returns ``(merged_entities, updated_mentions, merge_decisions)``.

        ``cached_embeddings`` maps an existing entity's ``entity_id`` to its precomputed
        label-embedding vector (perf parity with the source signature): supplied vectors skip the
        per-call re-embed of the existing side in the embedding rung.
        """
        self.merge_log.clear()
        if not new_entities:
            return list(existing_entities), mentions, []

        merge_map: dict[str, str] = {}
        all_entities = existing_entities + new_entities

        self._incremental_merge_by_exact_name(new_entities, existing_entities, merge_map)
        self._incremental_merge_by_aliases(new_entities, existing_entities, merge_map)
        if self.acronym_expand:
            self._incremental_merge_by_acronym(new_entities, existing_entities, merge_map)
        self._incremental_merge_by_substring(new_entities, existing_entities, merge_map)
        if self.use_fuzzy_merge:
            self._incremental_merge_by_fuzzy(new_entities, existing_entities, merge_map)
        if self.use_embedding_merge:
            self._incremental_merge_by_embedding(
                new_entities, existing_entities, merge_map, cached_embeddings
            )

        if not merge_map:
            merged = list(existing_entities) + list(new_entities)
            logger.info("Incremental: no merges, %d new entities kept", len(new_entities))
            return merged, mentions, []

        resolved_entities, survivor_map = self._apply_entity_merges(
            all_entities, merge_map, dataset_id
        )
        updated_mentions = self._update_mention_ids(mentions, survivor_map)
        logger.info(
            "Incremental resolution: %d new entities, %d merged into existing",
            len(new_entities),
            len(merge_map),
        )
        return resolved_entities, updated_mentions, list(self.merge_log)

    def _incremental_merge_by_exact_name(
        self,
        new_entities: list[CanonicalEntityRecord],
        existing_entities: list[CanonicalEntityRecord],
        merge_map: dict[str, str],
    ) -> None:
        existing_by_name: dict[str, CanonicalEntityRecord] = {}
        for e in existing_entities:
            key = self._normalize(e.canonical_name)
            prev = existing_by_name.get(key)
            if prev is None or e.evidence_count > prev.evidence_count:
                existing_by_name[key] = e
        for ne in new_entities:
            if ne.entity_id in merge_map:
                continue
            key = self._normalize(ne.canonical_name)
            if key in existing_by_name:
                merge_map[ne.entity_id] = existing_by_name[key].entity_id
                self.merge_log.append(
                    MergeDecision(
                        source_entity_id=ne.entity_id,
                        target_entity_id=existing_by_name[key].entity_id,
                        reason="incremental_exact_name",
                        confidence=1.0,
                        source_name=ne.canonical_name,
                        target_name=existing_by_name[key].canonical_name,
                    )
                )

    def _incremental_merge_by_aliases(
        self,
        new_entities: list[CanonicalEntityRecord],
        existing_entities: list[CanonicalEntityRecord],
        merge_map: dict[str, str],
    ) -> None:
        existing_name_map: dict[str, CanonicalEntityRecord] = {}
        for e in existing_entities:
            existing_name_map[self._normalize(e.canonical_name)] = e
            for alias in e.aliases:
                existing_name_map[self._normalize(alias)] = e
        for ne in new_entities:
            if ne.entity_id in merge_map:
                continue
            for name in [ne.canonical_name, *ne.aliases]:
                target = existing_name_map.get(self._normalize(name))
                if target and target.entity_id != ne.entity_id:
                    merge_map[ne.entity_id] = target.entity_id
                    self.merge_log.append(
                        MergeDecision(
                            source_entity_id=ne.entity_id,
                            target_entity_id=target.entity_id,
                            reason="incremental_alias_overlap",
                            confidence=0.9,
                            source_name=ne.canonical_name,
                            target_name=target.canonical_name,
                        )
                    )
                    break

    def _incremental_merge_by_acronym(
        self,
        new_entities: list[CanonicalEntityRecord],
        existing_entities: list[CanonicalEntityRecord],
        merge_map: dict[str, str],
    ) -> None:
        short_new = [
            e
            for e in new_entities
            if e.entity_id not in merge_map
            and len(e.canonical_name) <= 6
            and e.canonical_name.isupper()
        ]
        long_existing = [e for e in existing_entities if len(e.canonical_name) > 6]
        for short in short_new:
            acronym = short.canonical_name.strip().upper()
            for long_ent in long_existing:
                if short.label != long_ent.label:
                    continue
                words = re.findall(r"\b[A-ZÄÖÜ]\w*", long_ent.canonical_name)
                initials = "".join(w[0].upper() for w in words if w)
                if initials == acronym:
                    merge_map[short.entity_id] = long_ent.entity_id
                    self.merge_log.append(
                        MergeDecision(
                            source_entity_id=short.entity_id,
                            target_entity_id=long_ent.entity_id,
                            reason="incremental_acronym",
                            confidence=0.85,
                            source_name=short.canonical_name,
                            target_name=long_ent.canonical_name,
                        )
                    )
                    break

    def _incremental_merge_by_substring(
        self,
        new_entities: list[CanonicalEntityRecord],
        existing_entities: list[CanonicalEntityRecord],
        merge_map: dict[str, str],
    ) -> None:
        for ne in new_entities:
            if ne.entity_id in merge_map:
                continue
            ne_lower = ne.canonical_name.strip().lower()
            if len(ne_lower) < 3:
                continue
            for ee in existing_entities:
                if ne.label != ee.label:
                    continue
                ee_lower = ee.canonical_name.strip().lower()
                whole_word_substring = (
                    ee_lower.startswith(ne_lower + " ")
                    or ee_lower.endswith(" " + ne_lower)
                    or ne_lower.startswith(ee_lower + " ")
                    or ne_lower.endswith(" " + ee_lower)
                )
                if whole_word_substring and (set(ne.document_ids) & set(ee.document_ids)):
                    merge_map[ne.entity_id] = ee.entity_id
                    self.merge_log.append(
                        MergeDecision(
                            source_entity_id=ne.entity_id,
                            target_entity_id=ee.entity_id,
                            reason="incremental_substring",
                            confidence=0.7,
                            source_name=ne.canonical_name,
                            target_name=ee.canonical_name,
                        )
                    )
                    break

    def _incremental_merge_by_fuzzy(
        self,
        new_entities: list[CanonicalEntityRecord],
        existing_entities: list[CanonicalEntityRecord],
        merge_map: dict[str, str],
    ) -> None:
        """Fuzzy edit-distance: new entities vs existing only (same-label, >= min_similarity).

        The distinguishing-token guard (:meth:`_fuzzy_guard_blocks`) vetoes a candidate whose
        numeric tokens or leading short code differ, so a distinct paragraph/org is never the best.
        """
        existing_by_label: dict[str, list[CanonicalEntityRecord]] = defaultdict(list)
        for e in existing_entities:
            existing_by_label[e.label.lower()].append(e)

        for ne in new_entities:
            if ne.entity_id in merge_map:
                continue
            ne_key = self._normalize(ne.canonical_name)
            best: tuple[float, CanonicalEntityRecord] | None = None
            for ee in existing_by_label.get(ne.label.lower(), ()):
                if self.fuzzy_shared_doc_guard and not (
                    set(ne.document_ids) & set(ee.document_ids)
                ):
                    continue
                if self._fuzzy_guard_blocks(ne.canonical_name, ee.canonical_name):
                    continue  # numeric/leading-code distinguisher differs — not a variant
                ee_key = self._normalize(ee.canonical_name)
                sim = difflib.SequenceMatcher(None, ne_key, ee_key).ratio()
                if sim >= self.min_similarity and (best is None or sim > best[0]):
                    best = (sim, ee)
            if best is not None:
                merge_map[ne.entity_id] = best[1].entity_id
                self.merge_log.append(
                    MergeDecision(
                        source_entity_id=ne.entity_id,
                        target_entity_id=best[1].entity_id,
                        reason="fuzzy_edit_distance",
                        confidence=best[0],
                        source_name=ne.canonical_name,
                        target_name=best[1].canonical_name,
                    )
                )

    def _incremental_merge_by_embedding(
        self,
        new_entities: list[CanonicalEntityRecord],
        existing_entities: list[CanonicalEntityRecord],
        merge_map: dict[str, str],
        cached_embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        """Embedding-based merge: new entities vs existing only.

        ``cached_embeddings`` (existing ``entity_id`` -> precomputed vector) lets the existing side
        skip its per-call re-embed: an existing entity present in the cache reuses its vector, and
        only the cache-miss remainder of the group is embedded (source perf-parity, A-F4).
        """
        if self._label_encoder is None:
            return
        active_new = [e for e in new_entities if e.entity_id not in merge_map]
        if not active_new or not existing_entities:
            return

        by_label: dict[str, tuple[list[CanonicalEntityRecord], list[CanonicalEntityRecord]]] = {}
        for e in active_new:
            by_label.setdefault(e.label.lower(), ([], []))[0].append(e)
        for e in existing_entities:
            lbl = e.label.lower()
            if lbl in by_label:
                by_label[lbl][1].append(e)

        total = 0
        for _label, (new_group, existing_group) in by_label.items():
            if not new_group or not existing_group:
                continue
            new_texts = [" | ".join([e.canonical_name, *e.aliases[:3]]) for e in new_group]
            new_emb = self._label_encoder.embed_labels(new_texts)
            existing_emb = self._existing_embeddings(existing_group, cached_embeddings)
            for i in range(len(new_group)):
                if new_group[i].entity_id in merge_map:
                    continue
                sims = [cosine(new_emb[i], existing_emb[j]) for j in range(len(existing_group))]
                best_j = max(range(len(sims)), key=lambda j: sims[j]) if sims else 0
                best_sim = sims[best_j] if sims else 0.0
                if best_sim >= self.embedding_merge_threshold:
                    merge_map[new_group[i].entity_id] = existing_group[best_j].entity_id
                    self.merge_log.append(
                        MergeDecision(
                            source_entity_id=new_group[i].entity_id,
                            target_entity_id=existing_group[best_j].entity_id,
                            reason="incremental_embedding",
                            confidence=best_sim,
                            source_name=new_group[i].canonical_name,
                            target_name=existing_group[best_j].canonical_name,
                        )
                    )
                    total += 1
        if total > 0:
            logger.info("Incremental embedding merges: %d", total)

    def _existing_embeddings(
        self,
        existing_group: list[CanonicalEntityRecord],
        cached_embeddings: dict[str, list[float]] | None,
    ) -> list[list[float]]:
        """Embeddings for an existing-entity group, reusing ``cached_embeddings`` on a hit (A-F4).

        A cache miss (or no cache) falls back to embedding that entity's label text, so the result
        is always aligned 1:1 with ``existing_group`` — only the cache misses are re-embedded.
        """
        assert self._label_encoder is not None  # guarded by the caller
        if not cached_embeddings:
            texts = [" | ".join([e.canonical_name, *e.aliases[:3]]) for e in existing_group]
            return self._label_encoder.embed_labels(texts)

        out: list[list[float]] = [[] for _ in existing_group]
        misses: list[int] = []
        for idx, e in enumerate(existing_group):
            cached = cached_embeddings.get(e.entity_id)
            if cached is not None:
                out[idx] = list(cached)
            else:
                misses.append(idx)
        if misses:
            miss_texts = [
                " | ".join([existing_group[i].canonical_name, *existing_group[i].aliases[:3]])
                for i in misses
            ]
            for i, vec in zip(misses, self._label_encoder.embed_labels(miss_texts), strict=True):
                out[i] = vec
        return out

    # -- apply merges -------------------------------------------------------

    def _apply_entity_merges(
        self,
        entities: list[CanonicalEntityRecord],
        merge_map: dict[str, str],
        dataset_id: str,
    ) -> tuple[list[CanonicalEntityRecord], dict[str, str]]:
        """Apply the merge map: fuse merged entities into canonical ones (transitive-resolved).

        Returns ``(resolved_entities, survivor_map)`` where ``survivor_map`` maps EVERY input
        entity_id to the entity_id that actually SURVIVES its group — the highest-evidence
        ``primary``, which is NOT necessarily the merge-map transitive root. Mention remapping must
        use this survivor id, not the root, or a mention whose entity was absorbed by a
        lower-evidence root (e.g. an acronym folding a higher-evidence full name) would point at a
        non-emitted id (A-F2).
        """
        resolved_map = _resolve_transitive(merge_map)
        groups: dict[str, list[CanonicalEntityRecord]] = defaultdict(list)
        for e in entities:
            groups[resolved_map.get(e.entity_id, e.entity_id)].append(e)

        result: list[CanonicalEntityRecord] = []
        survivor_map: dict[str, str] = {}
        for _target_id, group in groups.items():
            if len(group) == 1:
                survivor_map[group[0].entity_id] = group[0].entity_id
                result.append(group[0])
                continue
            group.sort(key=lambda e: e.evidence_count, reverse=True)
            primary = group[0]
            for member in group:
                survivor_map[member.entity_id] = primary.entity_id
            all_aliases = set(primary.aliases)
            all_mention_ids = list(primary.mention_ids)
            all_document_ids = set(primary.document_ids)
            total_evidence = primary.evidence_count
            for other in group[1:]:
                all_aliases.add(other.canonical_name)
                all_aliases.update(other.aliases)
                all_mention_ids.extend(other.mention_ids)
                all_document_ids.update(other.document_ids)
                total_evidence += other.evidence_count
            all_aliases.discard(primary.canonical_name)
            result.append(
                CanonicalEntityRecord(
                    entity_id=primary.entity_id,
                    dataset_id=dataset_id,
                    canonical_name=primary.canonical_name,
                    label=primary.label,
                    aliases=sorted(all_aliases),
                    mention_ids=all_mention_ids,
                    document_ids=sorted(all_document_ids),
                    evidence_count=total_evidence,
                    confidence=max(e.confidence for e in group),
                    source="resolved",
                    version=primary.version + 1,
                )
            )
        result.sort(key=lambda e: e.evidence_count, reverse=True)
        return result, survivor_map

    def _update_mention_ids(
        self, mentions: list[MentionRecord], survivor_map: dict[str, str]
    ) -> list[MentionRecord]:
        """Remap each mention's canonical_entity_id to the SURVIVING entity id (A-F2).

        ``survivor_map`` (from :meth:`_apply_entity_merges`) points every merged entity id at the
        highest-evidence survivor that is actually emitted — so a mention never dangles at a
        transitive-root id that lost the evidence tie-break and was folded away.
        """
        updated: list[MentionRecord] = []
        for m in mentions:
            survivor = m.canonical_entity_id and survivor_map.get(m.canonical_entity_id)
            if survivor and survivor != m.canonical_entity_id:
                updated.append(m.model_copy(update={"canonical_entity_id": survivor}))
            else:
                updated.append(m)
        return updated


def _resolve_transitive(merge_map: dict[str, str]) -> dict[str, str]:
    """Resolve transitive merge chains (A->B, B->C => A->C) with cycle detection/breaking."""
    resolved: dict[str, str] = {}
    cycles_logged: set[frozenset[str]] = set()
    for source in merge_map:
        target = merge_map[source]
        chain = [source]
        seen = {source}
        while target in merge_map and target not in seen:
            seen.add(target)
            chain.append(target)
            target = merge_map[target]
        if target in seen:
            cycle_key = frozenset(chain)
            if cycle_key not in cycles_logged:
                cycles_logged.add(cycle_key)
                logger.warning(
                    "Merge cycle detected: %s — breaking at %r",
                    " -> ".join([*chain, target]),
                    target,
                )
        resolved[source] = target
    return resolved
