"""Chained Redaction Stages COMPOSE their masks — no redactor is wired-but-dead (audit R5-12).

The defect this closes had two halves, and only both together make a second redactor real:

1. **Structural** — both committed GPU deploy stacks declared ``redact_pii`` and ``redact_financial``
   as parallel SIBLINGS off ``content_screen`` while ``export_corpus`` depended only on
   ``redact_financial``. The Runner threads only the declared parent's output into an Export, so the
   learned PII redactor's masks (person/email/phone/address/credit-card) were discarded wholesale
   while its Stage still reported success, was certified by the #198 placement probe, and
   contributed its counts to the Quality Report.
2. **Behavioural** — merely chaining them would not have fixed it either:
   :func:`finalize_redacted_chunk` masks from the chunk's clean ``content`` and OVERWRITES
   ``masked_content`` / ``pii_spans``, so a downstream Redactor CLOBBERED every mask its upstream
   sibling had applied.

The fix composes upstream spans in the one shared seam every Provider funnels through (the same
place the ADR-0044 universal financial floor lives), so masks accumulate across a redaction chain.

The end-to-end tests below are the discriminating probe for that pair: two ``redaction.hybrid_rule``
Stages with DISJOINT detector sets, so each mask is attributable to exactly one Stage and a lost
mask is unambiguous.

Disjoint detectors are NOT sufficient on their own, though — they are exactly the configuration in
which a greedy first-wins union never bites (audit R6-invariants-0, R6-wiring-2). The shipped
``full-stack`` / ``gpu-learned`` stacks chain two LEARNED Redactors whose ``pii_labels`` overlap
heavily, so two models scoring the same entity with different boundaries is the expected case. The
span-level tests therefore drive the shared seam directly across every overlap topology, and assert
COVERAGE MONOTONICITY: composing may only ever ADD masked characters, never remove one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    PageSlice,
    PageSpan,
    Provenance,
    RedactionAction,
)
from latence_core.page_index import PageIndexResolver
from latence_core.pipeline import Pipeline
from latence_core.redaction_policy import (
    RedactionPolicy,
    finalize_redacted_chunk,
    plan_chunk_redaction,
)
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

# packages/latence-core/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_STACKS = sorted((_REPO_ROOT / "deploy" / "env").glob("*/stack.yaml"))

_PII_LINE = "Acme Corp and Erika Mustermann, erika@acme.example, IBAN DE44500105175407324931.\n"


def _chained_pipeline(store: str, docs_dir: Path) -> Pipeline:
    """source -> parse -> chunk -> redact_a (email) -> redact_b (iban) -> export."""
    return Pipeline.from_yaml_text(
        f"""
name: redaction-composition
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
      extensions: [txt]
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: redact_a
    capability: redaction
    provider: redaction.hybrid_rule
    depends_on: [chunk]
    config:
      detectors: [email]
      default_action: mask
      policies:
        internal:
          action: mask
  - name: redact_b
    capability: redaction
    provider: redaction.hybrid_rule
    depends_on: [redact_a]
    config:
      detectors: [iban]
      default_action: mask
      policies:
        internal:
          action: mask
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [redact_b]
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text(_PII_LINE, encoding="utf-8")
    return docs


def _exported(store: str, run_id: str) -> list[ChunkRecord]:
    s = Storage.for_uri(store)
    uri = f"{store}/_latence/runs/{run_id}/export/records.jsonl"
    return [
        ChunkRecord.model_validate(json.loads(line))
        for line in s.read_text(uri).splitlines()
        if line.strip()
    ]


def test_chained_redactors_compose_masks_into_the_exported_corpus(tmp_path: Path) -> None:
    """The exported corpus carries BOTH Stages' masks — the downstream one does not clobber.

    Pre-fix this failed on the email: ``redact_b`` re-masked the clean ``content`` from scratch, so
    the exported text read
    ``'Acme Corp and Erika Mustermann, erika@acme.example, IBAN [IBAN].'`` — the IBAN from the
    exported branch masked, the email that ``redact_a`` detected in the clear.
    """
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    LocalRunner().run(_chained_pipeline(store, docs), run_id="compose")
    records = _exported(store, "compose")

    memo = next(r for r in records if r.content and "Acme Corp" in r.content)
    assert memo.content is not None
    # The Export materialises masked_content as the corpus text.
    assert "erika@acme.example" not in memo.content, (
        f"upstream redactor's email mask was lost: {memo.content!r}"
    )
    assert "DE44500105175407324931" not in memo.content, (
        f"downstream redactor's IBAN mask was lost: {memo.content!r}"
    )
    assert "[EMAIL]" in memo.content
    assert "[IBAN]" in memo.content
    # Both Stages' spans survive on the record, each attributable to its own detector.
    assert {sp.pii_type for sp in memo.pii_spans} == {"email", "iban"}


def test_single_redactor_stack_is_byte_identical(tmp_path: Path) -> None:
    """OFF-BY-DEFAULT: with one Redaction Stage there are no upstream spans, so nothing changes."""
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: redaction-single
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs}'
      sensitivity: internal
      extensions: [txt]
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: redact
    capability: redaction
    provider: redaction.hybrid_rule
    depends_on: [chunk]
    config:
      detectors: [email, iban]
      default_action: mask
      policies:
        internal:
          action: mask
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [redact]
"""
    )
    LocalRunner().run(pipeline, run_id="single")
    memo = next(r for r in _exported(store, "single") if r.content and "Acme Corp" in r.content)
    assert memo.content is not None
    assert "[EMAIL]" in memo.content
    assert "[IBAN]" in memo.content
    assert {sp.pii_type for sp in memo.pii_spans} == {"email", "iban"}


# -- audit R6-invariants-0: composition must never REMOVE a mask ---------------------------------
#
# The R5-12 union above resolved overlaps greedily — sort by (start, -length, source) and DROP any
# span that overlaps an accepted one. That silently UN-REDACTED text in three ways, none of which
# the three tests above can see (they use DISJOINT spans). All fixtures are synthetic: 4111…1111 is
# the published Visa test number, never real PAN data (ADR-0044).

_SYNTHETIC_CARD = "4111111111111111"


def _finalize(
    chunk: ChunkRecord,
    action: RedactionAction,
    accepted: list[tuple[int, int, str, str, str, float]],
) -> ChunkRecord:
    """Drive the ONE seam every Redactor funnels through, at the chunk level."""
    policy = RedactionPolicy(policies=None, default_action=action)
    plan = plan_chunk_redaction(
        chunk,
        policy,
        detector_types={"person", "credit_card"},
        sensitive_set=set(),
        min_chars=1,
        resolver=PageIndexResolver(),
    )
    return finalize_redacted_chunk(chunk, plan, accepted=accepted, salt="s")


def _chunk(text: str) -> ChunkRecord:
    return ChunkRecord(
        record_id="d1#chunk:0",
        document_record_id="d1",
        index=0,
        provenance=Provenance(
            source_uri="file:///c/a.txt",
            file_name="a.txt",
            file_type="txt",
            file_size=len(text),
            document_id="sha256:d1",
            char_start=0,
            char_end=len(text),
            page_start=1,
            page_end=1,
        ),
        classification=Classification(language="en", sensitivity="internal"),
        content=text,
        token_count=len(text.split()),
        # v19: a one-page fixture — the chunk's own page spans are that single page.
        page_slice=PageSlice(
            pages=[PageSpan(page_number=1, char_start=0, char_end=len(text))],
            total_chars=len(text),
        ),
    )


def test_upstream_tag_span_cannot_swallow_the_financial_floor() -> None:
    """An upstream TAG Stage must not un-mask the ADR-0044 floor's card span.

    Pre-fix the union sorted purely by offset, so the prior TAG person span at 6 outranked the
    floor's MASK ``credit_card`` span at 20 and DROPPED it; ``apply_masking`` then skips the TAG
    span (TAG leaves text readable by design) and NOTHING masked the region — the exported
    ``masked_content`` carried the full 16-digit card verbatim.
    """
    text = f"Note: Jane Roe Card {_SYNTHETIC_CARD} filed."
    # Upstream Stage: audit-only TAG over 'Jane Roe Card 4111111111111111'.
    upstream = _finalize(
        _chunk(text), RedactionAction.TAG, [(6, 36, "person", text[6:36], "gliner:person", 0.9)]
    )
    assert upstream.masked_content == text  # TAG leaves the text readable — as documented.

    # Downstream `redact_financial`: detects nothing itself; the universal floor finds the card.
    final = _finalize(upstream, RedactionAction.MASK, [])

    assert final.masked_content is not None
    assert _SYNTHETIC_CARD not in final.masked_content, (
        f"upstream TAG span swallowed the financial floor: {final.masked_content!r}"
    )
    assert final.masked_content == "Note: Jane Roe Card [CREDIT_CARD] filed."


def test_partially_overlapping_span_does_not_re_expose_a_masked_card() -> None:
    """Chained learned Redactors: a partial overlap must CLIP, not drop the prior mask.

    Reachable with the shipped ``gpu-learned`` stack — two masking Stages whose learned span
    boundaries need not coincide. Pre-fix the downstream person span [8,26) caused the prior,
    already-applied ``credit_card`` mask [22,38) to be discarded WHOLE, yielding
    ``'Contact [PERSON]111111111111 end.'`` — 12 digits of a fully-masked card re-exposed.
    """
    text = f"Contact Jane Roe Card {_SYNTHETIC_CARD} end."
    upstream = _finalize(
        _chunk(text),
        RedactionAction.MASK,
        [(22, 38, "credit_card", text[22:38], "gliner_pii", 0.9)],
    )
    assert upstream.masked_content == "Contact Jane Roe Card [CREDIT_CARD] end."

    final = _finalize(
        upstream, RedactionAction.MASK, [(8, 26, "person", text[8:26], "gliner2", 0.9)]
    )

    assert final.masked_content is not None
    assert "111111111111" not in final.masked_content, (
        f"prior card mask was truncated, re-exposing digits: {final.masked_content!r}"
    )
    assert final.masked_content == "Contact [PERSON][CREDIT_CARD] end."


def test_overlapping_same_type_spans_from_chained_learned_redactors_compose() -> None:
    """Two learned Redactors scoring the SAME entity with DIFFERENT boundaries (audit R6-wiring-2).

    The shipped ``full-stack`` / ``gpu-learned`` stacks chain ``redact_pii`` -> ``redact_financial``
    with heavily overlapping ``pii_labels`` (``person``, ``email``, ``phone number``, ``address``,
    ``credit card number`` are in BOTH), so both Stages emitting a ``person`` span over the same name
    with different boundaries is the EXPECTED case, not an edge case. ``pii_type`` is a tie-breaker
    in the composition sort, so pin the same-type shape explicitly — the other partial-overlap tests
    all use two different types.

    Pre-fix the downstream span [9,17) sorted first and the prior, already-applied [14,28) mask was
    dropped WHOLE, yielding ``'Kontakt: [PERSON] Mustermann arbeitet hier.'`` — a surname that was
    masked before the second Stage existed, exported in the clear.
    """
    text = "Kontakt: Herr Max Mustermann arbeitet hier."

    # Stage 1 (`redact_pii`): "Max Mustermann".
    upstream = _finalize(
        _chunk(text), RedactionAction.MASK, [(14, 28, "person", text[14:28], "gliner_pii", 0.9)]
    )
    assert upstream.masked_content == "Kontakt: Herr [PERSON] arbeitet hier."

    # Stage 2 (`redact_financial`): re-detects on the clean content, boundaries "Herr Max".
    final = _finalize(
        upstream, RedactionAction.MASK, [(9, 17, "person", text[9:17], "gliner2", 0.9)]
    )

    assert final.masked_content is not None
    assert "Mustermann" not in final.masked_content, (
        f"adding a redaction Stage un-masked an already-masked name: {final.masked_content!r}"
    )
    assert final.masked_content == "Kontakt: [PERSON][PERSON] arbeitet hier."


def test_provider_span_overlapping_the_floor_does_not_truncate_it() -> None:
    """The SAME hole in ``_merge_floor`` — no chaining needed, one Redactor is enough (ADR-0044).

    A single Provider person span [8,26) partially overlapping the floor's ``credit_card`` [22,38)
    dropped the floor span outright pre-fix, leaking the card's tail on the plain single-Redactor
    path that every stack runs.
    """
    text = f"Contact Jane Roe Card {_SYNTHETIC_CARD} end."

    final = _finalize(
        _chunk(text), RedactionAction.MASK, [(8, 26, "person", text[8:26], "gliner2", 0.9)]
    )

    assert final.masked_content is not None
    assert "111111111111" not in final.masked_content, (
        f"floor span truncated by an overlapping Provider span: {final.masked_content!r}"
    )
    assert final.masked_content == "Contact [PERSON][CREDIT_CARD] end."


def _masked_offsets(chunk: ChunkRecord) -> set[int]:
    """Every ``content`` offset a mutating (non-TAG) span covers on this chunk."""
    return {
        i
        for span in chunk.pii_spans
        if span.action is not RedactionAction.TAG
        for i in range(span.char_start, span.char_end)
    }


@pytest.mark.parametrize(
    ("prior_span", "this_span"),
    [
        ((5, 15), (10, 20)),  # this Stage overlaps the prior span's tail
        ((10, 20), (5, 15)),  # this Stage overlaps the prior span's head
        ((5, 25), (10, 15)),  # this Stage strictly inside the prior span
        ((10, 15), (5, 25)),  # prior span strictly inside this Stage's
        ((5, 15), (15, 25)),  # exactly abutting — no overlap
        ((5, 15), (5, 15)),  # identical spans
        ((5, 10), (20, 25)),  # fully disjoint
    ],
)
def test_composition_never_removes_a_masked_character(
    prior_span: tuple[int, int], this_span: tuple[int, int]
) -> None:
    """COVERAGE MONOTONICITY: composing may only ADD masked characters, never remove one.

    The invariant the greedy union broke, stated directly and checked over every overlap topology:
    a character masked by the upstream Stage stays masked after the downstream Stage composes.
    """
    text = "A" * 40
    p_start, p_end = prior_span
    t_start, t_end = this_span

    upstream = _finalize(
        _chunk(text),
        RedactionAction.MASK,
        [(p_start, p_end, "person", text[p_start:p_end], "up", 0.9)],
    )
    before = _masked_offsets(upstream)
    assert before == set(range(p_start, p_end))

    final = _finalize(
        upstream,
        RedactionAction.MASK,
        [(t_start, t_end, "credit_card", text[t_start:t_end], "down", 0.9)],
    )

    after = _masked_offsets(final)
    assert before <= after, (
        f"composition UNMASKED offsets {sorted(before - after)} "
        f"(prior={prior_span}, this={this_span})"
    )
    assert set(range(t_start, t_end)) <= after, "this Stage's own mask was lost"
    # apply_masking's precondition: ordered and non-overlapping.
    starts_ends = [(s.char_start, s.char_end) for s in final.pii_spans]
    assert starts_ends == sorted(starts_ends)
    assert all(a[1] <= b[0] for a, b in zip(starts_ends, starts_ends[1:], strict=False))


def _stage_map(stack_path: Path) -> dict[str, dict[str, Any]]:
    doc = yaml.safe_load(stack_path.read_text(encoding="utf-8"))
    return {s["name"]: s for s in doc.get("stages", [])}


def _ancestors(stages: dict[str, dict[str, Any]], start: str) -> set[str]:
    seen: set[str] = set()
    frontier = list(stages[start].get("depends_on") or [])
    while frontier:
        name = frontier.pop()
        if name in seen or name not in stages:
            continue
        seen.add(name)
        frontier.extend(stages[name].get("depends_on") or [])
    return seen


@pytest.mark.parametrize("stack_path", _ENV_STACKS, ids=[p.parent.name for p in _ENV_STACKS])
def test_every_redaction_stage_reaches_the_corpus_export(stack_path: Path) -> None:
    """No committed deploy stack may carry a wired-but-dead Redactor (audit R5-12).

    A Redaction Stage that is not an ancestor of the corpus Export (the
    ``export.jsonl_parquet`` Stage that materialises ``masked_content`` as the RAG corpus) runs,
    reports success, and contributes Quality-Report counts while none of its masks reach the
    deliverable — every gate reads green over an under-redacted corpus.
    """
    stages = _stage_map(stack_path)
    redactors = {n for n, s in stages.items() if s.get("capability") == "redaction"}
    if not redactors:
        pytest.skip("stack declares no redaction Stage")
    corpus_exports = [
        n
        for n, s in stages.items()
        if s.get("capability") == "export" and s.get("provider") == "export.jsonl_parquet"
    ]
    assert corpus_exports, f"{stack_path} redacts but exports no corpus"
    for export_name in corpus_exports:
        reachable = _ancestors(stages, export_name)
        dead = sorted(redactors - reachable)
        assert not dead, (
            f"{stack_path.parent.name}: redaction Stage(s) {dead} never reach corpus export "
            f"{export_name!r} — their masks are discarded while the Stage reports success."
        )


# -- audit R6-cross_lane_integration-14: the Quality Report must describe the CORPUS ---------------
#
# Chaining the two Redactors (R5-12) fixed the corpus but broke its audit trail: the Quality Report
# folds the FLAT list of every Redaction Stage's output, so on the shipped chained topology each
# chunk is tallied twice — once as `redact_pii` emitted it, once as `redact_financial` re-emitted it
# with the upstream spans composed in. The R5 tests above never look at RedactionQuality, so the
# whole suite stayed green over counts that scale with chain length, not with what reached the RAG
# corpus.


def _report_probe_pipeline(store: str, docs_dir: Path) -> Pipeline:
    """The shipped shape: two chained Redactors with OVERLAPPING detector sets, export off the 2nd.

    ``redact_financial`` re-detects ``email`` (as the shipped stacks' ``pii_labels`` overlap does),
    so the upstream email span is re-emitted downstream — the double-count. Its ``phone`` detector
    also claims the card number at the SAME extent as the upstream financial floor's ``credit_card``
    span, and the provider wins that tie — the phantom type.
    """
    return Pipeline.from_yaml_text(
        f"""
name: redaction-report-counts
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
      extensions: [txt]
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: redact_pii
    capability: redaction
    provider: redaction.hybrid_rule
    depends_on: [chunk]
    config:
      detectors: [email]
      default_action: mask
      policies:
        internal:
          action: mask
  - name: redact_financial
    capability: redaction
    provider: redaction.hybrid_rule
    depends_on: [redact_pii]
    config:
      detectors: [email, phone]
      default_action: mask
      policies:
        internal:
          action: mask
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [redact_financial]
"""
    )


def test_chained_redactors_do_not_double_count_in_the_quality_report(tmp_path: Path) -> None:
    """RedactionQuality must equal the spans in the EXPORTED corpus, not the sum over the chain.

    Pre-fix, on this exact shipped shape (one document carrying an email, a phone number and the
    synthetic card ``4111111111111111``), the report read
    ``pii_total=5 per_type={'credit_card': 1, 'email': 2, 'phone': 2}`` while the exported record
    carried **3** spans of types ``['email', 'phone', 'phone']``. Two independent errors:

    * ``email: 2`` — ``redact_pii`` emitted the span and ``redact_financial`` re-emitted the same
      one via ``_compose_prior_spans``; both Stages' chunks are in the flat rollup list. On the
      shipped ``gpu-learned`` / ``full-stack`` stacks EVERY person/email/phone/address span the
      learned ``redaction.gliner_pii`` finds is double-counted this way.
    * ``credit_card: 1`` — a PHANTOM. The ADR-0044 financial floor's card span from ``redact_pii``
      was superseded downstream by an identical-extent ``phone`` span (provider wins the tie), so
      the report certified a ``credit_card`` redaction no exported span records.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text(
        f"Erika Mustermann, erika@acme.example, phone 555-123-4567, card {_SYNTHETIC_CARD}.\n",
        encoding="utf-8",
    )
    store = f"file://{tmp_path}/store"

    report = LocalRunner().run(_report_probe_pipeline(store, docs), run_id="rq")

    exported = _exported(store, "rq")
    corpus_types: dict[str, int] = {}
    corpus_actions: dict[str, int] = {}
    for rec in exported:
        for span in rec.pii_spans:
            corpus_types[span.pii_type] = corpus_types.get(span.pii_type, 0) + 1
            corpus_actions[span.action.value] = corpus_actions.get(span.action.value, 0) + 1
    corpus_total = sum(corpus_types.values())
    # Guard the fixture: the probe is only discriminating while the corpus really carries spans.
    assert corpus_total == 3, f"fixture drifted, corpus spans: {corpus_types}"

    assert report.redaction is not None
    rq = report.redaction
    assert rq.pii_total == corpus_total, (
        f"report tallies {rq.pii_total} PII spans but the exported corpus carries {corpus_total} "
        f"— counts scale with chain length, not with the corpus"
    )
    assert rq.per_type == dict(sorted(corpus_types.items())), (
        f"report per_type {rq.per_type} != exported corpus {corpus_types}"
    )
    assert rq.per_action == dict(sorted(corpus_actions.items()))
    # The phantom, stated directly: no type may be certified that the corpus never carries.
    assert "credit_card" not in rq.per_type, (
        "report names a credit_card redaction that no exported span records"
    )
    assert rq.documents_redacted == 1
    assert rq.documents_with_pii == 1


def test_single_redactor_quality_counts_are_unchanged(tmp_path: Path) -> None:
    """OFF-BY-DEFAULT: one Redaction Stage has no duplicate record_ids, so the rollup is identical."""
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: redaction-single-counts
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs}'
      sensitivity: internal
      extensions: [txt]
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: redact
    capability: redaction
    provider: redaction.hybrid_rule
    depends_on: [chunk]
    config:
      detectors: [email, iban]
      default_action: mask
      policies:
        internal:
          action: mask
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [redact]
"""
    )
    report = LocalRunner().run(pipeline, run_id="single-counts")
    assert report.redaction is not None
    assert report.redaction.pii_total == 2
    assert report.redaction.per_type == {"email": 1, "iban": 1}
    assert report.redaction.per_action == {"mask": 2}
    assert report.redaction.documents_redacted == 1
    assert report.redaction.documents_with_pii == 1


def test_duplicate_documents_keep_their_full_pii_tally(tmp_path: Path) -> None:
    """A corpus with two BYTE-IDENTICAL documents must not have its PII tally halved (audit R7).

    ``record_id``s are content-addressed (``sha256:<digest>`` at Source, ``…#chunk-<i>`` at Chunk),
    so two byte-identical documents legitimately produce two exported records that share one
    ``record_id``. De-duplicating the redaction rollup by ``record_id`` therefore silently drops
    real, separately-exported PII from the operator's audit trail (ADR-0034 counts-only
    observability) on the plain, single-Redactor default path. The report must describe the
    EXPORTED corpus, so it is asserted against the exported records rather than a literal.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    # Byte-identical on purpose: same content hash => same document_id => same chunk record_id.
    (docs / "memo_a.txt").write_text(_PII_LINE, encoding="utf-8")
    (docs / "memo_b.txt").write_text(_PII_LINE, encoding="utf-8")
    store = f"file://{tmp_path}/store"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: redaction-duplicate-docs
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs}'
      sensitivity: internal
      extensions: [txt]
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: redact
    capability: redaction
    provider: redaction.hybrid_rule
    depends_on: [chunk]
    config:
      detectors: [email, iban]
      default_action: mask
      policies:
        internal:
          action: mask
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [redact]
"""
    )
    report = LocalRunner().run(pipeline, run_id="duplicate-docs")

    exported = _exported(store, "duplicate-docs")
    # Guard the fixture: the probe only discriminates while the duplicate really is a duplicate.
    assert len(exported) == 2, f"fixture drifted, exported {len(exported)} records"
    assert len({r.record_id for r in exported}) == 1, (
        "fixture drifted: the two documents no longer share a content-addressed record_id"
    )
    corpus_types: dict[str, int] = {}
    for rec in exported:
        for span in rec.pii_spans:
            corpus_types[span.pii_type] = corpus_types.get(span.pii_type, 0) + 1
    assert corpus_types == {"email": 2, "iban": 2}, f"fixture drifted, corpus spans: {corpus_types}"

    assert report.redaction is not None
    rq = report.redaction
    assert rq.pii_total == sum(corpus_types.values()), (
        f"report tallies {rq.pii_total} PII spans but the exported corpus carries "
        f"{sum(corpus_types.values())} — duplicate documents were deduped out of the audit trail"
    )
    assert rq.per_type == dict(sorted(corpus_types.items())), (
        f"report per_type {rq.per_type} != exported corpus {corpus_types}"
    )
    assert rq.per_action == {"mask": 4}
