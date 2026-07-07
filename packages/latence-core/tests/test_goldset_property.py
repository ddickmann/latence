"""Property + adversarial hardening for the gold-set scoring engine (#76, Hardening bar).

The unit tests in ``test_goldset.py`` pin specific scored scenarios; this suite pins the
*invariants* the maximum-cardinality one-to-one matcher must uphold under **randomized, seeded**
inputs — the property-test half of the #76 hardening bar for the gold-set harness:

* **matching-count invariants** — for any gold/prediction set of one label, ``true_positives +
  false_positives == len(predictions)`` and ``true_positives + false_negatives == len(gold)``:
  every prediction is exactly one TP or FP, every gold is exactly one TP or FN, nothing is
  double-counted or lost (the one-to-one property, ADR-0008);
* **the matching is a valid assignment** — ``true_positives <= min(len(gold), len(pred))``, so
  the matcher never claims more matches than either side can supply (no double-count of a gold
  span across two predictions);
* **the matcher is optimal, not a single-pass greedy** — its TP count equals a brute-force
  maximum-bipartite-matching oracle, so it never understates recall/precision by "stealing" a
  gold span a later prediction was its only match for (the #76 correctness fix);
* **overlapping / adjacent spans** — the named adversarial shapes: many predictions piled on one
  gold span yield exactly one TP; adjacent (touching-but-disjoint, ``[0,4)`` vs ``[4,8)``) spans
  never match (empty intersection);
* **degenerate corners** — empty gold (all predictions are FP, recall 0.0 by convention), empty
  predictions / all-FN (all gold are FN, precision 0.0), all-FP;
* **determinism** — the same seeded input scored twice is byte-identical (the Baseline/G1
  seeded-run property).

Deterministic and dependency-free: randomness comes from a seeded :class:`random.Random`, so the
suite is reproducible (no ``hypothesis`` dep — ADR-0016 keeps core lean). Each property is checked
across many seeds; a brute-force maximum-matching oracle cross-checks the matcher's TP count.
"""

from __future__ import annotations

import random

from latence_core.contracts import Classification, EntityMention, Provenance, RelationMention
from latence_core.goldset import (
    ConfusionCounts,
    GoldDocument,
    GoldEntity,
    GoldRelation,
    GoldSet,
    _match_relations,
    _max_match,
    _span_overlap_jaccard,
    score_entities,
    score_relations,
)

_SEEDS = range(200)


# -- helpers -----------------------------------------------------------------


def _mention(
    record_id: str,
    label: str,
    char_start: int,
    char_end: int,
    *,
    document_record_id: str = "doc1",
    file_name: str = "a.txt",
) -> EntityMention:
    prov = Provenance(
        source_uri=f"file:///c/{file_name}",
        file_name=file_name,
        file_type="txt",
        file_size=1000,
        document_id=f"sha256:{document_record_id}",
        char_start=char_start,
        char_end=char_end,
        page_start=1,
        page_end=1,
    )
    return EntityMention(
        record_id=record_id,
        document_record_id=document_record_id,
        chunk_record_id=f"{document_record_id}#chunk-0",
        index=0,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        label=label,
        text="x",
        confidence=0.9,
    )


def _relation(
    record_id: str,
    label: str,
    head_id: str,
    tail_id: str,
    *,
    document_record_id: str = "doc1",
    file_name: str = "a.txt",
) -> RelationMention:
    prov = Provenance(
        source_uri=f"file:///c/{file_name}",
        file_name=file_name,
        file_type="txt",
        file_size=1000,
        document_id=f"sha256:{document_record_id}",
        char_start=0,
        char_end=1,
        page_start=1,
        page_end=1,
    )
    return RelationMention(
        record_id=record_id,
        document_record_id=document_record_id,
        index=0,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        label=label,
        head_mention_id=head_id,
        tail_mention_id=tail_id,
        head_text="h",
        tail_text="t",
        confidence=0.8,
    )


def _random_spans(rng: random.Random, n: int, *, labels: list[str]) -> list[tuple[int, str, int, int]]:
    """A list of ``(id, label, start, end)`` tuples with random (possibly degenerate) spans."""
    spans: list[tuple[int, str, int, int]] = []
    for i in range(n):
        start = rng.randint(0, 30)
        end = start + rng.randint(0, 8)  # >= start (a valid half-open span; may be zero-length)
        spans.append((i, rng.choice(labels), start, end))
    return spans


def _brute_force_max_tp(
    gold: list[tuple[int, str, int, int]],
    pred: list[tuple[int, str, int, int]],
    min_overlap: float,
) -> int:
    """A maximum-bipartite-matching oracle for the TP count over a SINGLE label.

    Builds the qualifies-edges (same label + overlap threshold) and computes the maximum matching
    via augmenting paths (Kuhn's algorithm) — an INDEPENDENT implementation from the one under
    test. The scorer's ``_max_match`` must agree with it: both are maximum-cardinality matchings,
    so the cross-check catches any regression that loses a matchable pair (under-counting recall).
    """
    adj: dict[int, list[int]] = {}
    for pi, (_pid, p_label, p_start, p_end) in enumerate(pred):
        adj[pi] = []
        for gi, (_gid, g_label, g_start, g_end) in enumerate(gold):
            if g_label != p_label:
                continue
            overlap = _span_overlap_jaccard(p_start, p_end, g_start, g_end)
            qualifies = overlap > 0.0 if min_overlap == 0.0 else overlap >= min_overlap
            if qualifies:
                adj[pi].append(gi)

    match_gold: dict[int, int] = {}

    def _augment(pi: int, seen: set[int]) -> bool:
        for gi in adj[pi]:
            if gi in seen:
                continue
            seen.add(gi)
            if gi not in match_gold or _augment(match_gold[gi], seen):
                match_gold[gi] = pi
                return True
        return False

    matched = 0
    for pi in range(len(pred)):
        if _augment(pi, set()):
            matched += 1
    return matched


# -- ConfusionCounts derivation properties -----------------------------------


def test_confusion_counts_derivations_bounded_for_random_counts() -> None:
    rng = random.Random(1234)
    for _ in range(2000):
        c = ConfusionCounts(
            true_positives=rng.randint(0, 50),
            false_positives=rng.randint(0, 50),
            false_negatives=rng.randint(0, 50),
        )
        # precision/recall/f1 are always well-defined probabilities (never a ZeroDivisionError).
        assert 0.0 <= c.precision <= 1.0
        assert 0.0 <= c.recall <= 1.0
        assert 0.0 <= c.f1 <= 1.0
        # F1 lies between precision and recall (the harmonic-mean bound), modulo rounding.
        lo, hi = sorted((c.precision, c.recall))
        assert lo - 1e-6 <= c.f1 <= hi + 1e-6
        assert c.predicted == c.true_positives + c.false_positives
        assert c.actual == c.true_positives + c.false_negatives


# -- matcher invariants ------------------------------------------------------


def test_max_match_partitions_every_prediction_and_gold() -> None:
    # The core one-to-one invariant across randomized single-label span sets: every prediction is
    # exactly one TP or FP and every gold is exactly one TP or FN — no double-count, nothing lost.
    for seed in _SEEDS:
        rng = random.Random(seed)
        gold = _random_spans(rng, rng.randint(0, 8), labels=["ORG"])
        pred = _random_spans(rng, rng.randint(0, 8), labels=["ORG"])
        min_overlap = rng.choice([0.0, 0.25, 0.5, 0.8, 1.0])
        counts = _max_match(gold, pred, min_overlap)
        assert counts.true_positives + counts.false_positives == len(pred), seed
        assert counts.true_positives + counts.false_negatives == len(gold), seed
        # A valid assignment can match at most min(|gold|, |pred|).
        assert counts.true_positives <= min(len(gold), len(pred)), seed


def test_max_match_tp_equals_maximum_matching_oracle() -> None:
    # The scorer's ``_max_match`` TP count must equal an INDEPENDENT brute-force maximum-matching
    # oracle for every seeded single-label input — it may never leave a matchable pair on the table
    # (which would understate recall/precision). This is the cross-check that guards the #76 fix.
    for seed in _SEEDS:
        rng = random.Random(seed + 5000)
        gold = _random_spans(rng, rng.randint(0, 7), labels=["ORG"])
        pred = _random_spans(rng, rng.randint(0, 7), labels=["ORG"])
        min_overlap = rng.choice([0.0, 0.5, 1.0])
        scorer_tp = _max_match(gold, pred, min_overlap).true_positives
        oracle = _brute_force_max_tp(gold, pred, min_overlap)
        assert scorer_tp == oracle, (seed, scorer_tp, oracle)


def test_matcher_finds_maximum_matching_not_single_pass_greedy() -> None:
    # Regression guard for the #76 correctness fix. A naive "each prediction grabs its
    # best-overlap unmatched gold" single pass UNDER-counts true positives — it understates both
    # recall and precision, the very numbers the gold-set harness exists to report faithfully.
    #
    # The steal shape (verified to give TP=1 under the old greedy, TP=2 under maximum matching):
    #   gold  g0 = [10,20)  (the "shared" span both predictions overlap)
    #         g1 = [0, 5)   (reachable ONLY by p0)
    #   pred  p0 = [0,20)   overlaps g0 (0.5, its best) AND g1 (0.25)
    #         p1 = [10,20)  overlaps g0 (1.0) ONLY — g1 is out of reach
    # A single-pass greedy processes p0 first, hands it g0 (its highest overlap), then p1 can only
    # reach g0 (taken) -> false positive. Result: TP=1. But the maximum matching is 2:
    # p0 -> g1, p1 -> g0. The augmenting-path matcher finds it.
    gold = [(0, "ORG", 10, 20), (1, "ORG", 0, 5)]
    pred = [(0, "ORG", 0, 20), (1, "ORG", 10, 20)]
    counts = _max_match(gold, pred, 0.0)
    assert counts.true_positives == 2, "maximum matching must credit both predictions, not 1"
    assert counts.false_positives == 0 and counts.false_negatives == 0
    # Precision/recall would have been understated to 0.5 by the old single-pass greedy.
    assert counts.precision == 1.0 and counts.recall == 1.0


def test_max_match_multi_label_is_sum_of_per_label_matches() -> None:
    # Matching is label-partitioned: the overall TP/FP/FN must equal the sum over each label's
    # independent match (a cross-label span pair can never match). Property over random label mixes.
    labels = ["ORG", "PERSON", "LOC"]
    for seed in _SEEDS:
        rng = random.Random(seed + 9000)
        gold = _random_spans(rng, rng.randint(0, 10), labels=labels)
        pred = _random_spans(rng, rng.randint(0, 10), labels=labels)
        min_overlap = rng.choice([0.0, 0.5])
        overall = _max_match(gold, pred, min_overlap)
        per_label = ConfusionCounts()
        for label in labels:
            g_l = [t for t in gold if t[1] == label]
            p_l = [t for t in pred if t[1] == label]
            per_label.add(_max_match(g_l, p_l, min_overlap))
        assert (overall.true_positives, overall.false_positives, overall.false_negatives) == (
            per_label.true_positives,
            per_label.false_positives,
            per_label.false_negatives,
        ), seed


# -- adversarial span shapes (overlapping / adjacent) ------------------------


def test_overlapping_predictions_on_one_gold_yield_exactly_one_tp() -> None:
    # Named adversarial shape: N predictions all piled onto ONE gold span. Exactly one is credited
    # (one-to-one); the rest are false positives. Would regress to N TPs if a gold span could be
    # matched by more than one prediction (the double-count the one-to-one matcher forbids).
    for seed in _SEEDS:
        rng = random.Random(seed + 12000)
        n = rng.randint(1, 12)
        gold = [(0, "ORG", 0, 10)]
        # Every prediction overlaps the single gold span (all inside [0,10)).
        pred = [(i, "ORG", rng.randint(0, 4), rng.randint(6, 10)) for i in range(n)]
        counts = _max_match(gold, pred, 0.0)
        assert counts.true_positives == 1, seed
        assert counts.false_positives == n - 1, seed
        assert counts.false_negatives == 0, seed


def test_adjacent_spans_never_match() -> None:
    # Adjacent = touching but disjoint half-open spans ([a,b) and [b,c)): the intersection is
    # empty, so Jaccard is 0.0 and they are NOT a match at any threshold (including any-overlap).
    for seed in _SEEDS:
        rng = random.Random(seed + 15000)
        boundary = rng.randint(1, 40)
        left_start = rng.randint(0, boundary - 1)
        right_end = boundary + rng.randint(1, 8)
        gold = [(0, "ORG", left_start, boundary)]
        pred = [(0, "ORG", boundary, right_end)]  # starts exactly where gold ends
        counts = _max_match(gold, pred, 0.0)
        assert counts.true_positives == 0, seed
        assert counts.false_positives == 1 and counts.false_negatives == 1, seed
        assert _span_overlap_jaccard(left_start, boundary, boundary, right_end) == 0.0, seed


# -- degenerate corners: empty gold / all-FP / all-FN ------------------------


def test_empty_gold_all_predictions_are_false_positives() -> None:
    for seed in _SEEDS:
        rng = random.Random(seed + 18000)
        pred = _random_spans(rng, rng.randint(0, 10), labels=["ORG", "PERSON"])
        counts = _max_match([], pred, rng.choice([0.0, 0.5, 1.0]))
        assert counts.true_positives == 0 and counts.false_negatives == 0
        assert counts.false_positives == len(pred)
        assert counts.recall == 0.0  # no gold ⇒ recall 0.0 by convention (never div-by-zero)


def test_empty_predictions_all_gold_are_false_negatives() -> None:
    for seed in _SEEDS:
        rng = random.Random(seed + 21000)
        gold = _random_spans(rng, rng.randint(0, 10), labels=["ORG", "PERSON"])
        counts = _max_match(gold, [], rng.choice([0.0, 0.5, 1.0]))
        assert counts.true_positives == 0 and counts.false_positives == 0
        assert counts.false_negatives == len(gold)
        assert counts.precision == 0.0  # no predictions ⇒ precision 0.0


def test_all_false_positives_when_no_span_qualifies() -> None:
    # Every prediction sits far from every gold span (disjoint), so nothing qualifies: pure FP+FN.
    for seed in _SEEDS:
        rng = random.Random(seed + 24000)
        gold = [(i, "ORG", i * 2, i * 2 + 1) for i in range(rng.randint(1, 6))]
        # Predictions in a disjoint far region.
        pred = [(i, "ORG", 1000 + i * 5, 1002 + i * 5) for i in range(rng.randint(1, 6))]
        counts = _max_match(gold, pred, 0.0)
        assert counts.true_positives == 0
        assert counts.false_positives == len(pred)
        assert counts.false_negatives == len(gold)


# -- end-to-end scorer properties (through score_entities/score_relations) ----


def test_score_entities_matches_partition_invariant_end_to_end() -> None:
    # The public scorer preserves the partition invariant per document: for a single-document,
    # single-label gold set, overall TP+FP == #mentions and TP+FN == #gold.
    for seed in _SEEDS:
        rng = random.Random(seed + 27000)
        gold_spans = _random_spans(rng, rng.randint(0, 6), labels=["ORG"])
        pred_spans = _random_spans(rng, rng.randint(0, 6), labels=["ORG"])
        gold = GoldSet(
            documents=[
                GoldDocument(
                    file_name="a.txt",
                    entities=[
                        GoldEntity(label=lbl, char_start=s, char_end=e)
                        for (_i, lbl, s, e) in gold_spans
                    ],
                )
            ],
            match_min_overlap=rng.choice([0.0, 0.5]),
        )
        mentions = [
            _mention(f"m{i}", lbl, s, e) for (i, lbl, s, e) in pred_spans
        ]
        overall, _ = score_entities(mentions, gold, {"doc1": "a.txt"})
        assert overall.true_positives + overall.false_positives == len(pred_spans), seed
        assert overall.true_positives + overall.false_negatives == len(gold_spans), seed


def test_score_entities_is_deterministic_over_random_inputs() -> None:
    # The seeded-run byte-identity property: scoring the same input twice yields identical counts.
    for seed in _SEEDS:
        rng = random.Random(seed + 30000)
        pred_spans = _random_spans(rng, rng.randint(0, 8), labels=["ORG", "PERSON"])
        gold_spans = _random_spans(rng, rng.randint(0, 8), labels=["ORG", "PERSON"])
        gold = GoldSet(
            documents=[
                GoldDocument(
                    file_name="a.txt",
                    entities=[
                        GoldEntity(label=lbl, char_start=s, char_end=e)
                        for (_i, lbl, s, e) in gold_spans
                    ],
                )
            ]
        )
        mentions = [_mention(f"m{i}", lbl, s, e) for (i, lbl, s, e) in pred_spans]
        a_overall, a_per = score_entities(mentions, gold, {"doc1": "a.txt"})
        b_overall, b_per = score_entities(mentions, gold, {"doc1": "a.txt"})
        assert a_overall.model_dump() == b_overall.model_dump(), seed
        assert {k: v.model_dump() for k, v in a_per.items()} == {
            k: v.model_dump() for k, v in b_per.items()
        }, seed


def test_score_relations_partition_invariant_over_random_inputs() -> None:
    # The relation scorer upholds the same one-to-one partition: TP+FP == #predicted relations,
    # TP+FN == #gold relations, whatever the endpoint alignment. Endpoints are wired so a random
    # fraction align (TP-able) and the rest do not (forced FP).
    for seed in _SEEDS:
        rng = random.Random(seed + 33000)
        n_gold = rng.randint(0, 5)
        n_pred = rng.randint(0, 5)
        gold_rels = [
            GoldRelation(
                label="works_for",
                head=GoldEntity(label="PERSON", char_start=0, char_end=5),
                tail=GoldEntity(label="ORG", char_start=10, char_end=14),
            )
            for _ in range(n_gold)
        ]
        gold = GoldSet(documents=[GoldDocument(file_name="a.txt", relations=gold_rels)])
        mentions = [
            _mention("h", "PERSON", 0, 5),
            _mention("t", "ORG", 10, 14),
            _mention("wrong", "PERSON", 10, 14),  # a wrong-type tail for forced-FP relations
        ]
        relations: list[RelationMention] = []
        for i in range(n_pred):
            # Half the time build an aligning relation, half a mis-aligned one (wrong tail type).
            tail = "t" if rng.random() < 0.5 else "wrong"
            relations.append(_relation(f"r{i}", "works_for", "h", tail))
        overall, _ = score_relations(relations, mentions, gold, {"doc1": "a.txt"})
        assert overall.true_positives + overall.false_positives == n_pred, seed
        assert overall.true_positives + overall.false_negatives == n_gold, seed
        assert overall.true_positives <= min(n_gold, n_pred), seed


def _brute_force_max_relation_tp(
    gold: list[tuple[int, GoldRelation]],
    pred: list[tuple[int, RelationMention]],
    mention_by_id: dict[str, EntityMention],
    min_overlap: float,
) -> int:
    """Independent maximum-matching oracle for the relation TP count.

    Builds the qualifies-edges (label match + BOTH endpoints resolving by span overlap) and returns
    the maximum matching via augmenting paths — a separate implementation from ``_match_relations``,
    so agreement catches any regression that loses a matchable relation pair (under-counting).
    """
    from latence_core.goldset import _endpoint_matches

    adj: dict[int, list[int]] = {}
    for pi, (_pi_id, rel) in enumerate(pred):
        head = mention_by_id.get(rel.head_mention_id)
        tail = mention_by_id.get(rel.tail_mention_id)
        adj[pi] = []
        for gi, (_gi_id, gold_rel) in enumerate(gold):
            if gold_rel.label != rel.label:
                continue
            if _endpoint_matches(gold_rel.head, head, min_overlap) and _endpoint_matches(
                gold_rel.tail, tail, min_overlap
            ):
                adj[pi].append(gi)

    match_gold: dict[int, int] = {}

    def _augment(pi: int, seen: set[int]) -> bool:
        for gi in adj[pi]:
            if gi in seen:
                continue
            seen.add(gi)
            if gi not in match_gold or _augment(match_gold[gi], seen):
                match_gold[gi] = pi
                return True
        return False

    matched = 0
    for pi in range(len(pred)):
        if _augment(pi, set()):
            matched += 1
    return matched


def test_relation_matcher_finds_maximum_matching_not_single_pass_greedy() -> None:
    # Regression guard mirroring the ENTITY steal-shape test, for the relation matcher (#76).
    # A single-pass greedy over relations under-counts identically: one prediction steals a gold
    # relation a later prediction was its only qualifying match for.
    #
    # Two gold relations sharing the same head entity but distinct tail entities:
    #   g0: head P[0,5)  -> tail O0[10,15)
    #   g1: head P[0,5)  -> tail O1[20,25)
    # Two predicted relations:
    #   p0: head hP[0,5) -> tail hBoth[10,25)  overlaps BOTH O0 and O1 -> qualifies for g0 AND g1
    #   p1: head hP[0,5) -> tail hO1[20,25)    overlaps ONLY O1        -> qualifies for g1 ONLY
    # A single-pass greedy hands p0 -> g0 (first qualifying gold in order), then p1 can only reach
    # g1... which is still free here, so to force the steal we order golds so p0 grabs g1 first:
    #   greedy p0 -> g0, p1 -> g1  gives TP=2 already; the real steal needs p0's ONLY overlap-tail
    #   to be the shared one. Construct so p0 qualifies for g0 & g1 but p1 ONLY for g0:
    gold = [
        (0, GoldRelation(
            label="rel",
            head=GoldEntity(label="P", char_start=0, char_end=5),
            tail=GoldEntity(label="O", char_start=10, char_end=15),
        )),
        (1, GoldRelation(
            label="rel",
            head=GoldEntity(label="P", char_start=0, char_end=5),
            tail=GoldEntity(label="O", char_start=20, char_end=25),
        )),
    ]
    # Mentions: shared head hP; a tail spanning both O spans (hBoth) qualifies for g0 & g1;
    # a tail only over O0 (hO0) qualifies for g0 only.
    mentions = {
        "hP": _mention("hP", "P", 0, 5),
        "hBoth": _mention("hBoth", "O", 10, 25),  # overlaps both [10,15) and [20,25)
        "hO0": _mention("hO0", "O", 10, 15),  # overlaps only [10,15)
    }
    pred = [
        (0, _relation("p0", "rel", "hP", "hBoth")),  # qualifies for g0 AND g1
        (1, _relation("p1", "rel", "hP", "hO0")),  # qualifies for g0 ONLY
    ]
    # Single-pass greedy (p0 first): p0 grabs g0 (first qualifying), p1 then also finds g0 taken and
    # g1 does NOT qualify for p1 -> FP. Greedy TP=1. Maximum matching: p0->g1, p1->g0 => TP=2.
    counts = _match_relations(gold, pred, mentions, 0.0)
    assert counts.true_positives == 2, "maximum relation matching must credit both, not 1"
    assert counts.false_positives == 0 and counts.false_negatives == 0


def test_relation_matcher_tp_equals_maximum_matching_oracle() -> None:
    # The relation scorer's TP count must equal an INDEPENDENT maximum-matching oracle across seeded
    # inputs with VARYING endpoint spans (so steal shapes can actually arise) — the cross-check the
    # partition-invariant test alone cannot provide.
    for seed in _SEEDS:
        rng = random.Random(seed + 36000)
        # A pool of tail entity spans; gold relations pick distinct tails, predicted relations wire
        # to mentions whose spans may cover one, several, or none of them.
        n_gold = rng.randint(0, 5)
        n_pred = rng.randint(0, 5)
        head_gold = GoldEntity(label="P", char_start=0, char_end=5)
        tail_offsets = [10, 20, 30, 40, 50]
        gold: list[tuple[int, GoldRelation]] = []
        for gi in range(n_gold):
            off = tail_offsets[gi % len(tail_offsets)]
            gold.append((
                gi,
                GoldRelation(
                    label="rel",
                    head=head_gold,
                    tail=GoldEntity(label="O", char_start=off, char_end=off + 5),
                ),
            ))
        # Mentions with random spans; the head is always the aligning P, tails vary.
        mentions: dict[str, EntityMention] = {"hP": _mention("hP", "P", 0, 5)}
        pred: list[tuple[int, RelationMention]] = []
        for pi in range(n_pred):
            t_start = rng.choice(tail_offsets) + rng.randint(-3, 3)
            t_end = t_start + rng.randint(0, 20)  # may span several gold tail spans
            tid = f"t{pi}"
            mentions[tid] = _mention(tid, "O", max(t_start, 0), max(t_end, 0))
            head_id = "hP" if rng.random() < 0.9 else "missing"  # occasionally a bad head
            pred.append((pi, _relation(f"p{pi}", "rel", head_id, tid)))
        min_overlap = rng.choice([0.0, 0.5])
        scorer_tp = _match_relations(gold, pred, mentions, min_overlap).true_positives
        oracle = _brute_force_max_relation_tp(gold, pred, mentions, min_overlap)
        assert scorer_tp == oracle, (seed, scorer_tp, oracle)


# -- adversarial scale: deep augmenting path must not overflow the stack -------


def test_max_match_dense_cluster_no_recursion_error() -> None:
    # DoS/adversarial guard (#76): a densely-overlapping cluster of many same-label spans forces a
    # long alternating augmenting path. A recursive matcher blows Python's recursion limit (~1000)
    # and crashes; the iterative matcher must handle it. N well above the limit, all mutually
    # overlapping so the candidate graph is dense.
    n = 3000
    gold = [(i, "ORG", 0, 100) for i in range(n)]
    pred = [(i, "ORG", 0, 100) for i in range(n)]
    counts = _max_match(gold, pred, 0.0)
    # All mutually overlap and share the label: a perfect matching of size n.
    assert counts.true_positives == n
    assert counts.false_positives == 0 and counts.false_negatives == 0


def test_max_match_deep_augment_chain_no_recursion_error() -> None:
    # A staircase that forces the augmenting search to re-route a long chain of already-matched
    # predictions (the worst case for recursion depth), at N above the recursion limit.
    n = 2000
    # gold i occupies [i, i+2); pred i occupies [i, i+2) so it overlaps gold i and gold i-1/i+1.
    gold = [(i, "ORG", i, i + 2) for i in range(n)]
    pred = [(i, "ORG", i, i + 2) for i in range(n)]
    counts = _max_match(gold, pred, 0.0)
    assert counts.true_positives == n, "overlapping staircase admits a perfect matching"
