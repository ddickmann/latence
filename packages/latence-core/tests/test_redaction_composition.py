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

The tests below are the discriminating probe: two ``redaction.hybrid_rule`` Stages with DISJOINT
detector sets, so each mask is attributable to exactly one Stage and a lost mask is unambiguous.
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
    Provenance,
    RedactionAction,
)
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
        _chunk(text), RedactionAction.MASK, [(22, 38, "credit_card", text[22:38], "gliner_pii", 0.9)]
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
