"""Gold-set scoring engine (S10): deterministic per-Stage precision/recall/F1.

Unit tests for the pure scorer: ConfusionCounts derivations (incl. the empty-denominator
guards), span-overlap matching (any-overlap vs a min-Jaccard threshold), one-to-one
maximum-cardinality assignment (no double-counting, no single-pass-greedy under-count),
per-label breakdown, relation endpoint resolution, and the contract validators on the
annotation models.
"""

from __future__ import annotations

import pytest
from latence_core.contracts import Classification, EntityMention, Provenance, RelationMention
from latence_core.goldset import (
    ConfusionCounts,
    GoldDocument,
    GoldEntity,
    GoldRelation,
    GoldSet,
    score_entities,
    score_relations,
)
from pydantic import ValidationError


def _mention(
    *,
    record_id: str,
    label: str,
    char_start: int,
    char_end: int,
    document_record_id: str = "doc1",
    file_name: str = "a.txt",
    page: int = 1,
) -> EntityMention:
    prov = Provenance(
        source_uri=f"file:///c/{file_name}",
        file_name=file_name,
        file_type="txt",
        file_size=1000,
        document_id=f"sha256:{document_record_id}",
        char_start=char_start,
        char_end=char_end,
        page_start=page,
        page_end=page,
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
    *,
    record_id: str,
    label: str,
    head_id: str,
    tail_id: str,
    char_start: int,
    char_end: int,
    document_record_id: str = "doc1",
    file_name: str = "a.txt",
) -> RelationMention:
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


# -- ConfusionCounts ---------------------------------------------------------


def test_confusion_counts_derivations() -> None:
    c = ConfusionCounts(true_positives=8, false_positives=2, false_negatives=4)
    assert c.predicted == 10
    assert c.actual == 12
    assert c.precision == 0.8
    assert round(c.recall, 4) == 0.6667
    # F1 = 2PR/(P+R)
    assert round(c.f1, 4) == round(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12), 4)


def test_confusion_counts_empty_denominators_never_divide_by_zero() -> None:
    # No predictions -> precision 0.0; no gold -> recall 0.0; both -> f1 0.0.
    assert ConfusionCounts().precision == 0.0
    assert ConfusionCounts().recall == 0.0
    assert ConfusionCounts().f1 == 0.0
    only_fp = ConfusionCounts(false_positives=3)
    assert only_fp.precision == 0.0
    assert only_fp.recall == 0.0


def test_confusion_counts_add() -> None:
    a = ConfusionCounts(true_positives=1, false_positives=1, false_negatives=1)
    a.add(ConfusionCounts(true_positives=2, false_positives=0, false_negatives=3))
    assert (a.true_positives, a.false_positives, a.false_negatives) == (3, 1, 4)


# -- entity scoring ----------------------------------------------------------


def _gold_entities(*entities: GoldEntity, file_name: str = "a.txt", overlap: float = 0.0) -> GoldSet:
    return GoldSet(
        documents=[GoldDocument(file_name=file_name, entities=list(entities))],
        match_min_overlap=overlap,
    )


def test_score_entities_perfect_match() -> None:
    gold = _gold_entities(
        GoldEntity(label="ORG", char_start=0, char_end=4),
        GoldEntity(label="PERSON", char_start=10, char_end=15),
    )
    mentions = [
        _mention(record_id="m1", label="ORG", char_start=0, char_end=4),
        _mention(record_id="m2", label="PERSON", char_start=10, char_end=15),
    ]
    overall, per_label = score_entities(mentions, gold, {"doc1": "a.txt"})
    assert (overall.true_positives, overall.false_positives, overall.false_negatives) == (2, 0, 0)
    assert overall.precision == 1.0 and overall.recall == 1.0 and overall.f1 == 1.0
    assert per_label["ORG"].true_positives == 1
    assert per_label["PERSON"].true_positives == 1


def test_score_entities_false_positive_and_false_negative() -> None:
    gold = _gold_entities(
        GoldEntity(label="ORG", char_start=0, char_end=4),
        GoldEntity(label="ORG", char_start=20, char_end=24),  # missed
    )
    mentions = [
        _mention(record_id="m1", label="ORG", char_start=0, char_end=4),  # TP
        _mention(record_id="m2", label="ORG", char_start=50, char_end=54),  # FP (no overlap)
    ]
    overall, _ = score_entities(mentions, gold, {"doc1": "a.txt"})
    assert (overall.true_positives, overall.false_positives, overall.false_negatives) == (1, 1, 1)
    assert overall.precision == 0.5
    assert overall.recall == 0.5


def test_score_entities_label_mismatch_is_not_a_match() -> None:
    # Same span, wrong label -> not a match (FP + FN), label-sensitive.
    gold = _gold_entities(GoldEntity(label="ORG", char_start=0, char_end=4))
    mentions = [_mention(record_id="m1", label="PERSON", char_start=0, char_end=4)]
    overall, _ = score_entities(mentions, gold, {"doc1": "a.txt"})
    assert overall.true_positives == 0
    assert overall.false_positives == 1
    assert overall.false_negatives == 1


def test_score_entities_min_overlap_threshold_rejects_partial_span() -> None:
    # A prediction overlapping only 3/6 chars (Jaccard 0.5) fails an 0.8 threshold.
    gold = _gold_entities(GoldEntity(label="ORG", char_start=0, char_end=4), overlap=0.8)
    partial = [_mention(record_id="m1", label="ORG", char_start=2, char_end=6)]
    overall, _ = score_entities(partial, gold, {"doc1": "a.txt"})
    assert overall.true_positives == 0 and overall.false_positives == 1
    # The exact span passes.
    exact = [_mention(record_id="m1", label="ORG", char_start=0, char_end=4)]
    ok, _ = score_entities(exact, gold, {"doc1": "a.txt"})
    assert ok.true_positives == 1


def test_score_entities_greedy_one_to_one_no_double_count() -> None:
    # One gold span, two overlapping predictions -> exactly one TP, one FP (no double-count).
    gold = _gold_entities(GoldEntity(label="ORG", char_start=0, char_end=10))
    mentions = [
        _mention(record_id="m1", label="ORG", char_start=0, char_end=10),
        _mention(record_id="m2", label="ORG", char_start=0, char_end=9),
    ]
    overall, _ = score_entities(mentions, gold, {"doc1": "a.txt"})
    assert overall.true_positives == 1
    assert overall.false_positives == 1
    assert overall.false_negatives == 0


def test_score_entities_ignores_unannotated_documents() -> None:
    # A mention in a document with no gold annotations is neither TP/FP/FN.
    gold = _gold_entities(GoldEntity(label="ORG", char_start=0, char_end=4), file_name="a.txt")
    mentions = [
        _mention(record_id="m1", label="ORG", char_start=0, char_end=4, document_record_id="doc1"),
        _mention(
            record_id="m2",
            label="ORG",
            char_start=0,
            char_end=4,
            document_record_id="doc2",
            file_name="other.txt",
        ),
    ]
    overall, _ = score_entities(mentions, gold, {"doc1": "a.txt", "doc2": "other.txt"})
    assert overall.true_positives == 1
    assert overall.false_positives == 0  # doc2's mention is out of scope, not a FP


def test_score_entities_is_deterministic() -> None:
    gold = _gold_entities(
        GoldEntity(label="ORG", char_start=0, char_end=4),
        GoldEntity(label="PERSON", char_start=10, char_end=15),
    )
    mentions = [
        _mention(record_id="m1", label="ORG", char_start=0, char_end=4),
        _mention(record_id="m2", label="PERSON", char_start=10, char_end=15),
    ]
    a = score_entities(mentions, gold, {"doc1": "a.txt"})
    b = score_entities(mentions, gold, {"doc1": "a.txt"})
    assert a[0].model_dump() == b[0].model_dump()
    assert {k: v.model_dump() for k, v in a[1].items()} == {
        k: v.model_dump() for k, v in b[1].items()
    }


# -- relation scoring --------------------------------------------------------


def test_score_relations_matches_when_both_endpoints_align() -> None:
    gold = GoldSet(
        documents=[
            GoldDocument(
                file_name="a.txt",
                relations=[
                    GoldRelation(
                        label="works_for",
                        head=GoldEntity(label="PERSON", char_start=0, char_end=5),
                        tail=GoldEntity(label="ORG", char_start=10, char_end=14),
                    )
                ],
            )
        ]
    )
    mentions = [
        _mention(record_id="h", label="PERSON", char_start=0, char_end=5),
        _mention(record_id="t", label="ORG", char_start=10, char_end=14),
    ]
    relations = [
        _relation(
            record_id="r1",
            label="works_for",
            head_id="h",
            tail_id="t",
            char_start=0,
            char_end=14,
        )
    ]
    overall, per_label = score_relations(relations, mentions, gold, {"doc1": "a.txt"})
    assert overall.true_positives == 1
    assert overall.false_positives == 0
    assert overall.false_negatives == 0
    assert per_label["works_for"].f1 == 1.0


def test_score_relations_endpoint_mismatch_is_a_false_positive() -> None:
    # Correct label, but the tail mention aligns to the wrong entity type -> not credited.
    gold = GoldSet(
        documents=[
            GoldDocument(
                file_name="a.txt",
                relations=[
                    GoldRelation(
                        label="works_for",
                        head=GoldEntity(label="PERSON", char_start=0, char_end=5),
                        tail=GoldEntity(label="ORG", char_start=10, char_end=14),
                    )
                ],
            )
        ]
    )
    mentions = [
        _mention(record_id="h", label="PERSON", char_start=0, char_end=5),
        _mention(record_id="t", label="PERSON", char_start=10, char_end=14),  # wrong type
    ]
    relations = [
        _relation(
            record_id="r1", label="works_for", head_id="h", tail_id="t", char_start=0, char_end=14
        )
    ]
    overall, _ = score_relations(relations, mentions, gold, {"doc1": "a.txt"})
    assert overall.true_positives == 0
    assert overall.false_positives == 1
    assert overall.false_negatives == 1


def test_score_relations_no_predictions_all_false_negatives() -> None:
    gold = GoldSet(
        documents=[
            GoldDocument(
                file_name="a.txt",
                relations=[
                    GoldRelation(
                        label="works_for",
                        head=GoldEntity(label="PERSON", char_start=0, char_end=5),
                        tail=GoldEntity(label="ORG", char_start=10, char_end=14),
                    )
                ],
            )
        ]
    )
    overall, _ = score_relations([], [], gold, {})
    assert overall.true_positives == 0
    assert overall.false_negatives == 1
    assert overall.recall == 0.0


# -- annotation contracts ----------------------------------------------------


def test_gold_entity_rejects_inverted_span() -> None:
    with pytest.raises(ValidationError):
        GoldEntity(label="ORG", char_start=10, char_end=4)


def test_gold_set_from_mapping_and_totals() -> None:
    data = {
        "documents": [
            {
                "file_name": "a.txt",
                "entities": [{"label": "ORG", "char_start": 0, "char_end": 4}],
                "relations": [
                    {
                        "label": "works_for",
                        "head": {"label": "PERSON", "char_start": 0, "char_end": 5},
                        "tail": {"label": "ORG", "char_start": 10, "char_end": 14},
                    }
                ],
            }
        ],
        "match_min_overlap": 0.5,
    }
    gold = GoldSet.from_mapping(data)
    assert gold.total_entities == 1
    assert gold.total_relations == 1
    assert gold.match_min_overlap == 0.5
    assert set(gold.by_file_name()) == {"a.txt"}


def test_gold_set_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        GoldSet.from_mapping({"documents": [], "bogus": 1})
