"""EntityResolver: the audited 5-strategy fusion cascade (ported from dataset-intelligence).

Proves each rung fires, merges are type-constrained + audited (no silent over-merge, §19),
transitive chains + cycles are handled, and the Granite-STS embedding rung merges semantically
(via a deterministic fake encoder) — plus incremental resolution.
"""

from __future__ import annotations

import math

from conftest import FakeLabelEncoder
from latence_core.disambiguation import normalize_text
from latence_disambig_glinker.entity_resolver import EntityResolver, _resolve_transitive
from latence_disambig_glinker.models import CanonicalEntityRecord, MentionRecord


def _ent(
    eid: str,
    name: str,
    label: str = "organization",
    aliases: list[str] | None = None,
    docs: list[str] | None = None,
    evidence: int = 1,
    confidence: float = 0.9,
) -> CanonicalEntityRecord:
    return CanonicalEntityRecord(
        entity_id=eid,
        dataset_id="ds",
        canonical_name=name,
        label=label,
        aliases=aliases or [],
        mention_ids=[f"{eid}-m{i}" for i in range(evidence)],
        document_ids=docs or ["d1"],
        evidence_count=evidence,
        confidence=confidence,
    )


def test_exact_name_match_merges_case_insensitive() -> None:
    ents = [_ent("e1", "Acme Corp", evidence=3), _ent("e2", "acme corp", evidence=1)]
    resolver = EntityResolver(use_embedding_merge=False)
    resolved, _m = resolver.resolve(ents, [], "ds")
    assert len(resolved) == 1
    assert resolver.merge_log[0].reason == "exact_name_match"
    assert resolver.merge_log[0].confidence == 1.0


def test_acronym_expansion_merges_same_label() -> None:
    """IBM ↔ International Business Machines merges (initials match, same label); audit records it."""
    ents = [
        _ent("e1", "International Business Machines", evidence=5),
        _ent("e2", "IBM", evidence=2),
    ]
    resolver = EntityResolver(use_embedding_merge=False)
    resolved, _m = resolver.resolve(ents, [], "ds")
    assert len(resolved) == 1
    assert any(d.reason == "acronym_expansion" for d in resolver.merge_log)


def test_acronym_is_type_constrained_across_labels() -> None:
    """The acronym rung NEVER crosses labels: FBI (org) does not fold into a PERSON full name."""
    ents = [
        _ent("e1", "Federal Bureau Investigation", label="person", evidence=5),
        _ent("e2", "FBI", label="organization", evidence=2),
    ]
    resolver = EntityResolver(use_embedding_merge=False)
    resolved, _m = resolver.resolve(ents, [], "ds")
    assert len(resolved) == 2  # type guard blocks the cross-label acronym merge
    assert not any(d.reason == "acronym_expansion" for d in resolver.merge_log)


def test_substring_requires_same_label_and_shared_document() -> None:
    shared = [
        _ent("e1", "Acme Industries", docs=["d1"], evidence=3),
        _ent("e2", "Acme", docs=["d1"], evidence=1),
    ]
    resolver = EntityResolver(use_embedding_merge=False)
    resolved, _m = resolver.resolve(shared, [], "ds")
    assert len(resolved) == 1
    assert any(d.reason == "substring_containment" for d in resolver.merge_log)

    # No shared document => no substring merge (the enterprise evidence requirement).
    disjoint = [
        _ent("e1", "Acme Industries", docs=["d1"], evidence=3),
        _ent("e2", "Acme", docs=["d2"], evidence=1),
    ]
    resolver2 = EntityResolver(use_embedding_merge=False)
    resolved2, _m2 = resolver2.resolve(disjoint, [], "ds")
    assert len(resolved2) == 2


def test_embedding_merge_uses_injected_encoder_and_is_type_grouped() -> None:
    """Meta ↔ Facebook merges by STS cosine (no string overlap); IBM stays distinct."""
    encoder = FakeLabelEncoder({"meta": 0.10, "facebook": 0.12, "ibm": 2.0})
    ents = [
        _ent("e1", "Meta", evidence=3),
        _ent("e2", "Facebook", evidence=1),
        _ent("e3", "IBM", evidence=1),
    ]
    resolver = EntityResolver(
        use_embedding_merge=True, embedding_merge_threshold=0.85, label_encoder=encoder
    )
    resolved, _m = resolver.resolve(ents, [], "ds")
    reasons = {d.reason for d in resolver.merge_log}
    assert "embedding_similarity" in reasons
    names = {e.canonical_name for e in resolved}
    assert "IBM" in names  # distinct, not swept in
    assert len(resolved) == 2  # {Meta+Facebook}, {IBM}


def test_embedding_merge_disabled_without_encoder() -> None:
    """No encoder => embedding rung is inert (CPU no-embedder path), no crash."""
    ents = [_ent("e1", "Meta"), _ent("e2", "Facebook")]
    resolver = EntityResolver(use_embedding_merge=True, label_encoder=None)
    resolved, _m = resolver.resolve(ents, [], "ds")
    assert len(resolved) == 2


# --------------------------------------------------------------------------- #
# FIX-1: the embedding-by-label rung is OFF by default (precision).
# --------------------------------------------------------------------------- #


class _CollapsingEncoder:
    """Pathological STS encoder: every name → the SAME unit vector (cosine 1.0 for all pairs).

    Models the real Granite failure mode on short entity names — every distinct name embeds too close
    — so a resolver that ran the embedding rung would collapse ALL same-label entities into one.
    """

    def embed_labels(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_embedding_merge_is_off_by_default_precision_safe() -> None:
    """FIX-1: the embedding rung is OFF by default, so distinct short same-label names never collapse.

    Even with a collapsing encoder present (every name cosine 1.0), the DEFAULT resolver does not
    embedding-merge — the rung is off. This is the regression guarding the 49-false-merge collapse:
    flip the default back to True and distinct names would fold into one, failing here.
    """
    resolver = EntityResolver(label_encoder=_CollapsingEncoder())
    assert resolver.use_embedding_merge is False  # default off, even with an encoder wired
    assert resolver.embedding_merge_threshold == 0.93  # safe floor if opt-in re-enabled
    ents = [
        _ent("e1", "Dennis Dickmann", label="person"),
        _ent("e2", "Max Mustermann", label="person"),
        _ent("e3", "Kai Kölsch", label="person"),
    ]
    resolved, _m = resolver.resolve(ents, [], "ds")
    assert len(resolved) == 3  # nothing merged (embedding off; fuzzy ratios far below 0.90)
    assert not any(d.reason.startswith("embedding") for d in resolver.merge_log)


def test_embedding_rung_still_collapses_if_opt_in_enabled_proving_the_default_matters() -> None:
    """Contrast: with the rung explicitly re-enabled, the collapsing encoder DOES over-merge.

    Proves the default (FIX-1) is what protects precision — the danger is real, not hypothetical.
    """
    resolver = EntityResolver(
        use_embedding_merge=True, embedding_merge_threshold=0.93, label_encoder=_CollapsingEncoder()
    )
    ents = [
        _ent("e1", "Dennis Dickmann", label="person"),
        _ent("e2", "Max Mustermann", label="person"),
    ]
    resolved, _m = resolver.resolve(ents, [], "ds")
    assert len(resolved) == 1  # the over-merge the default (off) prevents


# --------------------------------------------------------------------------- #
# FIX-3: the conservative fuzzy edit-distance rung (OCR/spelling-variant recall).
# --------------------------------------------------------------------------- #


def test_fuzzy_rung_default_enabled_and_folds_ocr_adjacent_variant() -> None:
    """FIX-3: the fuzzy rung is ON by default (min_similarity 0.84) and folds an OCR-adjacent pair."""
    resolver = EntityResolver()
    assert resolver.use_fuzzy_merge is True
    assert resolver.min_similarity == 0.84
    ents = [
        _ent("e1", "Kai Kölsch", label="person", evidence=3),
        _ent("e2", "Kai Kolsch", label="person", evidence=1),  # ratio 0.90 → merges at default
    ]
    resolved, _m = resolver.resolve(ents, [], "ds")
    assert len(resolved) == 1
    assert any(d.reason == "fuzzy_edit_distance" for d in resolver.merge_log)


def test_fuzzy_admits_the_two_edit_ocr_variant_at_tuned_default() -> None:
    """The tuned default (0.84) admits the legit ``Kai Kusch`` ↔ ``Kai Kölsch`` variant (ratio 0.84).

    This is why the default was lowered from 0.90: the two-edit OCR variant sits exactly at the band.
    The guard (no numbers, shared leading token ``Kai``) does not block it.
    """
    resolver = EntityResolver()
    ents = [
        _ent("e1", "Kai Kölsch", label="person", evidence=3),
        _ent("e2", "Kai Kusch", label="person", evidence=1),
    ]
    resolved, _m = resolver.resolve(ents, [], "ds")
    assert len(resolved) == 1
    assert any(d.reason == "fuzzy_edit_distance" for d in resolver.merge_log)


# --------------------------------------------------------------------------- #
# Distinguishing-token guard: BLOCK fuzzy merges that differ only in a number / leading code.
# --------------------------------------------------------------------------- #


def test_numeric_tokens_helper_extracts_over_normalized_surface() -> None:
    """Unit: ``_numeric_tokens`` folds superscripts/§-numbers via NFKC and returns the digit set."""
    assert EntityResolver._numeric_tokens("§ 3 Abs. 2 GwG") == frozenset({"3", "2"})
    assert EntityResolver._numeric_tokens("§ 11 Abs. 6 GwG") == frozenset({"11", "6"})
    assert EntityResolver._numeric_tokens("Kai Kölsch") == frozenset()
    # NFKC folds the superscript ² into an ASCII 2 → both K²/D² carry the same numeric token.
    assert EntityResolver._numeric_tokens("K² Ventures UG") == frozenset({"2"})


def test_leading_code_differs_helper() -> None:
    """Unit: ``_leading_code_differs`` fires only for a distinct short leading code."""
    assert EntityResolver._leading_code_differs("K² Ventures UG", "D² Ventures UG") is True
    assert EntityResolver._leading_code_differs("Kai Kölsch", "Kai Kolsch") is False  # same lead
    assert EntityResolver._leading_code_differs("Notare", "Notar") is False  # length mismatch
    assert EntityResolver._leading_code_differs("Müller", "Muller") is False  # non-leading diff
    assert EntityResolver._leading_code_differs("A1 GmbH", "B1 GmbH") is True  # short code


def test_fuzzy_numeric_guard_blocks_distinct_legal_paragraphs() -> None:
    """``§ 3 Abs. 2 GwG`` vs ``§ 11 Abs. 6 GwG`` (ratio ≥ threshold) is BLOCKED by the numeric guard.

    min_similarity=0.80 is BELOW the 0.83 pair ratio, so the pair passes the threshold and reaches
    the guard — proving it is the numeric-token guard, not the threshold, that blocks the merge.
    """
    resolver = EntityResolver(min_similarity=0.80)
    ents = [
        _ent("e1", "§ 3 Abs. 2 GwG", label="legal document", evidence=2),
        _ent("e2", "§ 11 Abs. 6 GwG", label="legal document", evidence=1),
    ]
    resolved, _m = resolver.resolve(ents, [], "ds")
    assert len(resolved) == 2  # distinct paragraphs stay apart
    assert not any(d.reason == "fuzzy_edit_distance" for d in resolver.merge_log)


def test_fuzzy_leading_code_guard_blocks_distinct_orgs() -> None:
    """``K² Ventures UG`` vs ``D² Ventures UG`` (ratio 0.93 ≥ default) is BLOCKED by the code guard."""
    resolver = EntityResolver()  # default 0.84; the 0.93 ratio clears the threshold
    ents = [
        _ent("e1", "K² Ventures UG", label="organization", evidence=2),
        _ent("e2", "D² Ventures UG", label="organization", evidence=1),
    ]
    resolved, _m = resolver.resolve(ents, [], "ds")
    assert len(resolved) == 2  # distinct orgs stay apart
    assert not any(d.reason == "fuzzy_edit_distance" for d in resolver.merge_log)


def test_fuzzy_guard_still_merges_german_morphological_variants() -> None:
    """The guard never blocks number-free, same-leading-token variants: Notare/Notar, Fragebogen/s."""
    resolver = EntityResolver(min_similarity=0.80)
    ents = [
        _ent("e1", "Notare", label="organization", evidence=2),
        _ent("e2", "Notar", label="organization", evidence=1),
        _ent("e3", "Fragebogens", label="legal document", evidence=2),
        _ent("e4", "Fragebogen", label="legal document", evidence=1),
    ]
    resolved, _m = resolver.resolve(ents, [], "ds")
    reasons = [d.reason for d in resolver.merge_log]
    assert reasons.count("fuzzy_edit_distance") == 2  # both morphological pairs fold
    assert len(resolved) == 2


def test_incremental_fuzzy_numeric_guard_blocks_distinct_paragraph() -> None:
    """Incremental: a new paragraph is NOT folded into an existing one that differs in its numbers."""
    existing = [_ent("ex", "§ 3 Abs. 2 GwG", label="legal document", evidence=5)]
    new = [_ent("n1", "§ 11 Abs. 6 GwG", label="legal document", evidence=1)]
    resolver = EntityResolver(min_similarity=0.80)
    merged, _m, decisions = resolver.resolve_incremental(new, existing, [], "ds")
    assert len(merged) == 2
    assert not any(d.reason == "fuzzy_edit_distance" for d in decisions)


def test_fuzzy_merges_full_ocr_variant_set_to_one_in_its_band() -> None:
    """The four OCR variants of one person consolidate into ONE entity (transitive single-linkage).

    'Kai Kusch' sits ~0.84 from the others (two edits), so the FULL four-variant collapse needs
    min_similarity lowered into its precision-safe band — the negatives (Dennis/Max) sit ≤0.35, so
    0.80 is still a wide margin. Asserts the chain (Kölsch–Kolsch–Koisch–Kusch) folds to one survivor.
    """
    resolver = EntityResolver(min_similarity=0.80)
    ents = [
        _ent("e1", "Kai Kölsch", label="person", evidence=4),
        _ent("e2", "Kai Kolsch", label="person", evidence=3),
        _ent("e3", "Kai Koisch", label="person", evidence=2),
        _ent("e4", "Kai Kusch", label="person", evidence=1),
    ]
    resolved, _m = resolver.resolve(ents, [], "ds")
    assert len(resolved) == 1
    assert resolved[0].canonical_name == "Kai Kölsch"  # highest-evidence survivor
    assert all(d.reason == "fuzzy_edit_distance" for d in resolver.merge_log)


def test_fuzzy_does_not_merge_distinct_same_label_names_at_default() -> None:
    """Distinct same-label names (low char-similarity) never fuzzy-merge at the default 0.90."""
    resolver = EntityResolver()  # fuzzy on, embedding off (both defaults)
    ents = [
        _ent("e1", "Dennis Dickmann", label="person"),
        _ent("e2", "Max Mustermann", label="person"),
        _ent("e3", "Kai Kölsch", label="person"),
    ]
    resolved, _m = resolver.resolve(ents, [], "ds")
    assert len(resolved) == 3
    assert not any(d.reason == "fuzzy_edit_distance" for d in resolver.merge_log)


def test_fuzzy_is_type_constrained_across_labels() -> None:
    """The fuzzy rung NEVER crosses labels: identical-surname variants of different types stay apart."""
    resolver = EntityResolver(min_similarity=0.80)
    ents = [
        _ent("e1", "Kai Kölsch", label="person"),
        _ent("e2", "Kai Kolsch", label="organization"),
    ]
    resolved, _m = resolver.resolve(ents, [], "ds")
    assert len(resolved) == 2  # cross-label guard blocks the fuzzy merge
    assert not any(d.reason == "fuzzy_edit_distance" for d in resolver.merge_log)


def test_fuzzy_shared_doc_guard_blocks_cross_document_merge_when_enabled() -> None:
    """With the (opt-in) shared-doc guard, a fuzzy merge requires the two entities share a document."""
    disjoint = [
        _ent("e1", "Kai Kölsch", label="person", docs=["d1"]),
        _ent("e2", "Kai Kolsch", label="person", docs=["d2"]),
    ]
    resolver = EntityResolver(min_similarity=0.90, fuzzy_shared_doc_guard=True)
    resolved, _m = resolver.resolve(disjoint, [], "ds")
    assert len(resolved) == 2  # no shared doc → guarded out

    shared = [
        _ent("e3", "Kai Kölsch", label="person", docs=["d1"]),
        _ent("e4", "Kai Kolsch", label="person", docs=["d1"]),
    ]
    resolver2 = EntityResolver(min_similarity=0.90, fuzzy_shared_doc_guard=True)
    resolved2, _m2 = resolver2.resolve(shared, [], "ds")
    assert len(resolved2) == 1  # shared doc → merges


def test_incremental_fuzzy_folds_new_variant_into_existing() -> None:
    existing = [_ent("ex", "Kai Kölsch", label="person", evidence=5)]
    new = [_ent("n1", "Kai Kolsch", label="person", evidence=1)]
    resolver = EntityResolver()
    merged, _m, decisions = resolver.resolve_incremental(new, existing, [], "ds")
    assert len(merged) == 1
    assert decisions[0].reason == "fuzzy_edit_distance"


def test_mention_ids_are_remapped_on_merge() -> None:
    ents = [_ent("e1", "Acme Corp", evidence=2), _ent("e2", "acme corp", evidence=1)]
    mentions = [
        MentionRecord(
            mention_id="m1", dataset_id="ds", document_id="d1", text="acme corp",
            label="organization", canonical_entity_id="e2",
        )
    ]
    resolver = EntityResolver(use_embedding_merge=False)
    _resolved, updated = resolver.resolve(ents, mentions, "ds")
    assert updated[0].canonical_entity_id == "e1"  # remapped to the surviving entity


def test_transitive_merge_and_cycle_break() -> None:
    # A->B, B->C collapses to A->C.
    assert _resolve_transitive({"A": "B", "B": "C"}) == {"A": "C", "B": "C"}
    # A cycle A->B->A is detected + broken (no infinite loop), producing a finite map.
    out = _resolve_transitive({"A": "B", "B": "A"})
    assert set(out) == {"A", "B"}


def test_resolve_incremental_only_compares_new_vs_existing() -> None:
    existing = [_ent("ex1", "International Business Machines", evidence=9)]
    new = [_ent("n1", "IBM", evidence=1)]
    resolver = EntityResolver(use_embedding_merge=False)
    merged, _mentions, decisions = resolver.resolve_incremental(new, existing, [], "ds")
    assert len(merged) == 1  # IBM folded into the existing full name
    assert decisions and decisions[0].reason == "incremental_acronym"


def test_no_merges_returns_inputs_unchanged() -> None:
    ents = [_ent("e1", "Alpha"), _ent("e2", "Bravo")]
    resolver = EntityResolver(use_embedding_merge=False)
    resolved, _m = resolver.resolve(ents, [], "ds")
    assert len(resolved) == 2
    assert resolver.merge_log == []


# --------------------------------------------------------------------------- #
# A-F1: confusable surfaces group by the framework's normalize_text.
# --------------------------------------------------------------------------- #


def test_confusable_ligature_names_merge_under_framework_normalization() -> None:
    """'ﬁle' (U+FB01 ligature) and 'file' fold to one cluster (NFKC), so no dropped provenance (A-F1).

    Under the old ``str.lower()`` grouping they survive as TWO clusters that the framework then maps
    to the SAME ``entity:file`` record_id — silently dropping one's mentions. The default in-engine
    normaliser is byte-identical to the framework's; assert that too so they cannot drift.
    """
    assert normalize_text("ﬁle") == normalize_text("file") == "file"
    ents = [_ent("e1", "ﬁle", evidence=1), _ent("e2", "file", evidence=3)]
    mentions = [
        MentionRecord(
            mention_id="m1", dataset_id="ds", document_id="d1", text="ﬁle",
            label="organization", canonical_entity_id="e1",
        ),
        MentionRecord(
            mention_id="m2", dataset_id="ds", document_id="d1", text="file",
            label="organization", canonical_entity_id="e2",
        ),
    ]
    resolver = EntityResolver(use_embedding_merge=False)
    resolved, updated = resolver.resolve(ents, mentions, "ds")
    assert len(resolved) == 1  # one cluster, not two colliding ones
    survivor_id = resolved[0].entity_id
    # No dropped provenance: both mentions carried into the surviving cluster + point at it.
    assert set(resolved[0].mention_ids) == {"e1-m0", "e2-m0", "e2-m1", "e2-m2"}
    assert {m.canonical_entity_id for m in updated} == {survivor_id}


def test_resolver_uses_injected_normalizer() -> None:
    """A caller can inject any normaliser (the Provider injects the framework's ``normalize_text``)."""
    ents = [_ent("e1", "ACME", evidence=1), _ent("e2", "acme", evidence=2)]
    resolver = EntityResolver(use_embedding_merge=False, name_normalizer=normalize_text)
    resolved, _m = resolver.resolve(ents, [], "ds")
    assert len(resolved) == 1


# --------------------------------------------------------------------------- #
# A-F2: mentions remap to the emitted SURVIVOR id, not the merge-map root.
# --------------------------------------------------------------------------- #


def test_acronym_absorbing_higher_evidence_remaps_mentions_to_survivor() -> None:
    """IT(evidence=3) absorbs Information Technology(evidence=1): mentions point at the survivor (A-F2).

    The acronym rung writes merge_map[IT]=InformationTechnology, but the emitted survivor is IT (more
    evidence). The old code remapped mentions to the merge-map ROOT (the full-name id), leaving them
    dangling at a non-emitted entity. Now every mention points at the actual emitted survivor id.
    """
    ents = [
        _ent("it_id", "IT", evidence=3),
        _ent("info_id", "Information Technology", evidence=1),
    ]
    mentions = [
        MentionRecord(
            mention_id="m_it", dataset_id="ds", document_id="d1", text="IT",
            label="organization", canonical_entity_id="it_id",
        ),
        MentionRecord(
            mention_id="m_full", dataset_id="ds", document_id="d1",
            text="Information Technology", label="organization", canonical_entity_id="info_id",
        ),
    ]
    resolver = EntityResolver(use_embedding_merge=False)
    resolved, updated = resolver.resolve(ents, mentions, "ds")
    assert len(resolved) == 1
    survivor_id = resolved[0].entity_id
    assert survivor_id == "it_id"  # highest-evidence member survives
    emitted_ids = {e.entity_id for e in resolved}
    # No mention dangles: every canonical_entity_id references an EMITTED entity (the survivor).
    for m in updated:
        assert m.canonical_entity_id in emitted_ids
    assert {m.canonical_entity_id for m in updated} == {"it_id"}


# --------------------------------------------------------------------------- #
# Embedding rung: the FAISS threshold path + the faiss-absent brute-force fallback.
# --------------------------------------------------------------------------- #


def test_embedding_merge_faiss_path_falls_back_to_brute_force_when_faiss_absent() -> None:
    """A group over the (injected) FAISS threshold hits the faiss path; faiss absent → brute-force.

    Injecting a tiny ``faiss_group_threshold`` forces the large-group branch without needing 1000+
    entities; ``faiss`` is not installed in the CPU env, so :meth:`_faiss_gpu_merge` degrades to the
    numpy-free brute-force merge — the merge still fires (never a crash), which is the enterprise
    fallback the audit asked be exercised.
    """
    encoder = FakeLabelEncoder({"meta": 0.10, "facebook": 0.12})
    ents = [_ent("e1", "Meta", evidence=3), _ent("e2", "Facebook", evidence=1)]
    resolver = EntityResolver(
        use_embedding_merge=True,
        embedding_merge_threshold=0.85,
        label_encoder=encoder,
        faiss_group_threshold=2,  # group of 2 now takes the faiss (→ fallback) branch
    )
    resolved, _m = resolver.resolve(ents, [], "ds")
    assert len(resolved) == 1  # merged via the brute-force fallback
    assert any(d.reason == "embedding_similarity" for d in resolver.merge_log)


# --------------------------------------------------------------------------- #
# Incremental rungs beyond acronym: exact / alias / substring / embedding.
# --------------------------------------------------------------------------- #


def test_incremental_exact_name_folds_new_into_existing() -> None:
    existing = [_ent("ex", "Acme Corp", evidence=5)]
    new = [_ent("n1", "acme corp", evidence=1)]
    resolver = EntityResolver(use_embedding_merge=False)
    merged, _m, decisions = resolver.resolve_incremental(new, existing, [], "ds")
    assert len(merged) == 1
    assert decisions[0].reason == "incremental_exact_name"


def test_incremental_alias_overlap_folds_new_into_existing() -> None:
    existing = [_ent("ex", "Big Corporation", aliases=["BigCo"], evidence=5)]
    new = [_ent("n1", "BigCo", evidence=1)]
    resolver = EntityResolver(use_embedding_merge=False)
    merged, _m, decisions = resolver.resolve_incremental(new, existing, [], "ds")
    assert len(merged) == 1
    assert decisions[0].reason == "incremental_alias_overlap"


def test_incremental_substring_requires_same_label_and_shared_document() -> None:
    existing = [_ent("ex", "Acme Industries", docs=["d1"], evidence=5)]
    new = [_ent("n1", "Acme", docs=["d1"], evidence=1)]
    resolver = EntityResolver(use_embedding_merge=False)
    merged, _m, decisions = resolver.resolve_incremental(new, existing, [], "ds")
    assert len(merged) == 1
    assert decisions[0].reason == "incremental_substring"

    # No shared document -> no substring merge.
    disjoint_new = [_ent("n2", "Acme", docs=["d9"], evidence=1)]
    resolver2 = EntityResolver(use_embedding_merge=False)
    merged2, _m2, _d2 = resolver2.resolve_incremental(disjoint_new, existing, [], "ds")
    assert len(merged2) == 2


def test_incremental_embedding_merges_new_vs_existing() -> None:
    encoder = FakeLabelEncoder({"meta": 0.10, "facebook": 0.12})
    existing = [_ent("ex", "Facebook", evidence=5)]
    new = [_ent("n1", "Meta", evidence=1)]
    resolver = EntityResolver(
        use_embedding_merge=True, embedding_merge_threshold=0.85, label_encoder=encoder
    )
    merged, _m, decisions = resolver.resolve_incremental(new, existing, [], "ds")
    assert len(merged) == 1
    assert decisions[0].reason == "incremental_embedding"


def test_incremental_embedding_reuses_cached_embeddings_for_existing_side() -> None:
    """``cached_embeddings`` skips re-embedding the existing side (perf parity with source, A-F4)."""

    class _SpyEncoder:
        def __init__(self, angles: dict[str, float]) -> None:
            self._inner = FakeLabelEncoder(angles)
            self.embedded: list[str] = []

        def embed_labels(self, texts: list[str]) -> list[list[float]]:
            self.embedded.extend(texts)
            return self._inner.embed_labels(texts)

    encoder = _SpyEncoder({"meta": 0.10, "facebook": 0.12})
    existing = [_ent("ex", "Facebook", evidence=5)]
    new = [_ent("n1", "Meta", evidence=1)]
    # The existing entity's precomputed vector (its own "Facebook" embedding).
    cached = {"ex": [math.cos(0.12), math.sin(0.12)]}
    resolver = EntityResolver(
        use_embedding_merge=True, embedding_merge_threshold=0.85, label_encoder=encoder
    )
    merged, _m, decisions = resolver.resolve_incremental(
        new, existing, [], "ds", cached_embeddings=cached
    )
    assert len(merged) == 1  # still merges, using the cached existing vector
    assert decisions[0].reason == "incremental_embedding"
    # Only the NEW side was embedded; the existing "Facebook" text was served from the cache.
    assert encoder.embedded == ["Meta"]
