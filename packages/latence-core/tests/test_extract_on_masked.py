"""W18 (ADR-0044) — the optional ``extract_on: masked`` mode (mask-before-extract).

Default (``unmasked``) is byte-identical to pre-W18: extraction reads the raw chunk text, so the KG
keeps the real entities and redaction masks only the exported corpus. ``masked`` feeds extraction the
REDACTED chunk stream (``masked_content``), so PII never reaches entity/relation extraction or the KG
— the stricter privacy posture.

These tests prove: (1) in ``masked`` mode a PII string present in a chunk does NOT appear in any
EntityMention while the non-PII entity still extracts; (2) ``unmasked`` mode is byte-identical to a
pipeline with no ``extract_on`` field at all (the no-regression proof); (3) the Pipeline validator
rejects a ``masked`` stack whose extraction does not run after redaction (no silent privacy hole).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _resume_equality import output_artifacts
from latence_core.contracts import EntityMention
from latence_core.pipeline import ExtractOn, Pipeline
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

# A chunk with a PII email (masked by redaction.hybrid_rule) and a non-PII ORG (kept).
_EMAIL = "alice@secret.com"
_DOC = f"Contact {_EMAIL} at Acme headquarters for the Acme roadmap."


def _pipeline_text(store: str, docs_dir: Path, *, extract_on: str | None) -> str:
    # entities depends_on `redact` so masked_content exists before extraction; the mode switch then
    # decides whether the extractor reads content or masked_content. A gazetteer targets BOTH the PII
    # email and the non-PII ORG, so the two modes are distinguishable.
    header = f"extract_on: {extract_on}\n" if extract_on is not None else ""
    return f"""
name: extract-on-masked
storage_uri: {store}
{header}stages:
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
  - name: redact
    capability: redaction
    provider: redaction.hybrid_rule
    depends_on: [chunk]
    config:
      detectors: [email]
      default_action: mask
  - name: entities
    capability: entity_extraction
    provider: entity.gazetteer
    depends_on: [redact]
    config:
      labels:
        ORG:
          terms: [Acme]
        CONTACT:
          terms: ['{_EMAIL}']
"""


def _run(tmp_path: Path, *, extract_on: str | None, run_id: str) -> tuple[Storage, str]:
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "d.txt").write_text(_DOC + "\n", encoding="utf-8")
    store = f"file://{tmp_path}/store"
    pipe = Pipeline.from_yaml_text(_pipeline_text(store, docs, extract_on=extract_on))
    LocalRunner().run(pipe, run_id=run_id)
    return Storage.for_uri(store), store


def _artifacts(storage: Storage, store: str, run_id: str) -> dict[str, bytes]:
    # Artifacts are keyed relative to the run dir; with the SAME docs + store the only variable is
    # extract_on, so any byte difference IS a behavioural regression.
    return output_artifacts(storage, store, run_id)


def _mentions(storage: Storage, store: str, run_id: str) -> list[EntityMention]:
    ckpt = f"{store}/_latence/runs/{run_id}/checkpoints/entities.jsonl"
    return [
        EntityMention.model_validate_json(line)
        for line in storage.read_text(ckpt).splitlines()
        if line.strip()
    ]


def test_masked_mode_keeps_pii_out_of_extraction(tmp_path: Path) -> None:
    """masked → the PII email is absent from every EntityMention; the ORG still extracts."""
    storage, store = _run(tmp_path, extract_on="masked", run_id="masked")
    mentions = _mentions(storage, store, "masked")
    # The non-PII entity is still found (extraction is not blinded, just fed masked text).
    assert any(m.text == "Acme" and m.label == "ORG" for m in mentions)
    # The PII never reaches extraction: no mention carries it (surface or type).
    assert all(_EMAIL not in m.text for m in mentions)
    assert not any(m.label == "CONTACT" for m in mentions)


def test_unmasked_mode_keeps_entities(tmp_path: Path) -> None:
    """unmasked (default) → extraction sees the real text, so the PII IS extracted (KG keeps it)."""
    storage, store = _run(tmp_path, extract_on="unmasked", run_id="unmasked")
    mentions = _mentions(storage, store, "unmasked")
    assert any(m.text == "Acme" and m.label == "ORG" for m in mentions)
    assert any(m.text == _EMAIL and m.label == "CONTACT" for m in mentions)


def test_unmasked_is_byte_identical_to_no_field(tmp_path: Path) -> None:
    """The no-regression proof: ``extract_on: unmasked`` is byte-identical to omitting the field.

    Both runs use the SAME docs + store (only the run_id and the extract_on setting differ), so any
    byte difference in the persisted artifacts would be a real regression in the default path.
    """
    storage, store = _run(tmp_path, extract_on=None, run_id="no_field")
    _run(tmp_path, extract_on="unmasked", run_id="explicit")
    assert _artifacts(storage, store, "no_field") == _artifacts(storage, store, "explicit")


def test_default_extract_on_is_unmasked() -> None:
    """A stack with no ``extract_on`` defaults to the byte-identical unmasked mode."""
    pipe = Pipeline.from_yaml_text(
        "name: n\nstorage_uri: file:///s\nstages:\n"
        "  - {name: src, capability: source, provider: source.local_folder}\n"
    )
    assert pipe.extract_on is ExtractOn.UNMASKED


def test_masked_requires_extraction_after_redaction() -> None:
    """masked with an extraction Stage NOT downstream of redaction is a loud structural error."""
    with pytest.raises(ValueError, match="depend .transitively. on a redaction Stage"):
        Pipeline.from_yaml_text(
            """
name: bad
storage_uri: file:///s
extract_on: masked
stages:
  - {name: src, capability: source, provider: source.local_folder}
  - {name: parse, capability: parse, provider: parser.plaintext, depends_on: [src]}
  - {name: chunk, capability: chunk, provider: chunk.markdown, depends_on: [parse]}
  - {name: redact, capability: redaction, provider: redaction.hybrid_rule, depends_on: [chunk]}
  - name: entities
    capability: entity_extraction
    provider: entity.gazetteer
    depends_on: [chunk]
"""
        )


def test_masked_requires_a_redaction_stage() -> None:
    """masked with no redaction Stage at all is rejected (there is nothing to mask from)."""
    with pytest.raises(ValueError, match="requires a redaction Stage"):
        Pipeline.from_yaml_text(
            """
name: bad
storage_uri: file:///s
extract_on: masked
stages:
  - {name: src, capability: source, provider: source.local_folder}
  - {name: parse, capability: parse, provider: parser.plaintext, depends_on: [src]}
  - {name: chunk, capability: chunk, provider: chunk.markdown, depends_on: [parse]}
  - name: entities
    capability: entity_extraction
    provider: entity.gazetteer
    depends_on: [chunk]
"""
        )
