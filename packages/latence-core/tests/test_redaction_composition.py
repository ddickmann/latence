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

import pytest
import yaml
from latence_core.contracts import ChunkRecord
from latence_core.pipeline import Pipeline
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


def _stage_map(stack_path: Path) -> dict[str, dict]:
    doc = yaml.safe_load(stack_path.read_text(encoding="utf-8"))
    return {s["name"]: s for s in doc.get("stages", [])}


def _ancestors(stages: dict[str, dict], start: str) -> set[str]:
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
