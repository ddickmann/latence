"""P3-F2 §2 — the Runner FAILs loud on a malformed ProviderProfile declaration.

The P3-F1 correctness lens found (Sev-3, deferred to P3-F2): a Provider that declares a
**malformed** ``PROFILE``/``profile()`` (``profile_of`` raises ``TypeError``) was caught by
the Runner's device-routing try/except and **fell through to normal execution** as "no
profile", so the bad declaration never resurfaced. The fix (issue §2): a malformed profile
is an author bug and must surface as a typed :class:`~latence_core.errors.ConfigError` (a FAIL
Quality Report + a typed-error span), NOT run silently.

These are the **Runner-path** regression tests that were missing (the unit-level
``profile_of`` malformed test lives in ``test_provider_profile.py``). They assert three
distinct outcomes stay distinct:

* a provider whose ``PROFILE = "not a profile"`` → the stage FAILs with a ``ConfigError``
  (category ``config``), a FAIL report is persisted, and the malformed provider is **never
  executed** (a bad declaration never gets to run);
* an **absent** profile still means "follow the request" (None) — a normal run, not a fail;
* an **import/load error** still defers to the normal execution path that re-raises with full
  failure handling — not double-handled, not mapped to a config error here.

Deterministic + seeded (``LATENCE_CUDA=0``, no model download).
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from latence_core import ProviderRegistry
from latence_core.errors import ConfigError
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord, EntityMention


class _MalformedProfileExtractor:
    """A Provider whose ``PROFILE`` is not a ProviderProfile — a malformed declaration.

    ``profile_of`` raises ``TypeError`` for this; the Runner must map that to a ConfigError
    and FAIL the stage. Its ``extract`` raises if ever called, proving the malformed provider
    is never executed (the declaration error fires first).
    """

    PROFILE = "not a profile"  # malformed: not a ProviderProfile

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def extract(self, chunks: Iterable[ChunkRecord]) -> Iterator[EntityMention]:
        msg = "a malformed-profile provider must never run (the declaration error fires first)"
        raise AssertionError(msg)


class _NoProfileExtractor:
    """A Provider that declares NO profile — the absent case (must follow the request, run OK)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._terms: list[tuple[str, str]] = []
        for label, spec in (cfg.get("labels") or {}).items():
            for term in spec.get("terms", []):
                self._terms.append((str(label), str(term)))

    def extract(self, chunks: Iterable[ChunkRecord]) -> Iterator[EntityMention]:
        from latence_core.contracts import EntityMention as _EM
        from latence_core.offsetmap import OffsetIndex
        from latence_core.page_index import PageIndexResolver

        resolver = PageIndexResolver()
        for chunk in chunks:
            offset_index = OffsetIndex(chunk.offset_map) if chunk.offset_map is not None else None
            base = chunk.provenance.char_start or 0
            idx = 0
            for label, term in self._terms:
                pos = chunk.content.find(term)
                if pos < 0:
                    continue
                if offset_index is not None:
                    ds, de = offset_index.resolve_span(pos, pos + len(term))
                else:
                    ds, de = base + pos, base + pos + len(term)
                # Through the seam, which owns the half-open end clamp (v19).
                ps, pe = resolver.pages_for_span(chunk, ds, de)
                prov = chunk.provenance.model_copy(
                    update={"char_start": ds, "char_end": de, "page_start": ps, "page_end": pe}
                )
                yield _EM(
                    record_id=f"{chunk.record_id}:m{idx}",
                    provenance=prov,
                    classification=chunk.classification,
                    document_record_id=chunk.document_record_id,
                    chunk_record_id=chunk.record_id,
                    index=idx,
                    label=label,
                    text=term,
                    confidence=1.0,
                )
                idx += 1


class _RaisingProfileExtractor:
    """A Provider whose ``profile()`` classmethod BODY raises while building the descriptor.

    The class loads fine and declares a profile via the *preferred* classmethod form, but the
    body constructs an **invalid** ProviderProfile (``memory_mb=-1`` violates ``ge=0``), so
    ``profile_of`` propagates a pydantic ``ValidationError`` (a ``ValueError``, NOT a
    ``TypeError``). The Runner must map this to the same fail-loud ConfigError as a malformed
    declaration — it is the same class of author bug. Its ``extract`` raises if ever called.
    """

    @classmethod
    def profile(cls) -> Any:
        from latence_core import ProviderProfile

        return ProviderProfile(
            compute="cpu",
            memory_mb=-1,  # invalid: ge=0 → ValidationError raised from the profile() body
            license="Apache-2.0",
            license_verified=True,
            deterministic=True,
            batch=False,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def extract(self, chunks: Iterable[ChunkRecord]) -> Iterator[EntityMention]:
        msg = "an invalid-profile provider must never run (the profile() error fires first)"
        raise AssertionError(msg)


def _registry_with(name: str, cls: type) -> ProviderRegistry:
    reg = ProviderRegistry()
    discovered = dict(reg._discover())  # noqa: SLF001 - test-only seam access
    ep = metadata.EntryPoint(name=name, value="test:injected", group="latence.providers")
    object.__setattr__(ep, "load", lambda: cls)
    discovered[name] = ep
    reg._entry_points = discovered  # noqa: SLF001
    return reg


def _pipeline(store: str, docs_dir: Path, *, entity_provider: str) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: malformed-profile-slice
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
  - name: entities
    capability: entity_extraction
    provider: {entity_provider}
    depends_on: [chunk]
    config:
      device: auto
      labels:
        ORG:
          terms: [Acme Corp]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [entities]
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text("Acme Corp shipped the order.\n", encoding="utf-8")
    return docs


def test_malformed_profile_fails_stage_with_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    runner = LocalRunner(registry=_registry_with("entity.malformed", _MalformedProfileExtractor))

    # The malformed declaration FAILs the stage loud as a typed ConfigError — NOT a silent
    # "no profile" run. (The provider's extract() raises if run, so a silent run would have
    # been an AssertionError, not a ConfigError — this pins the fail-loud behavior.)
    with pytest.raises(ConfigError) as excinfo:
        runner.run(_pipeline(store, docs, entity_provider="entity.malformed"), run_id="malformed")
    assert "malformed ProviderProfile" in str(excinfo.value)
    assert "entities" in str(excinfo.value)


def test_malformed_profile_persists_fail_report_with_config_category(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    runner = LocalRunner(registry=_registry_with("entity.malformed", _MalformedProfileExtractor))
    with pytest.raises(ConfigError):
        runner.run(_pipeline(store, docs, entity_provider="entity.malformed"), run_id="malformed")

    # A FAIL Quality Report is persisted, and the entity stage carries error_count=1 +
    # the stable ``config`` category tag (the report half of the fail-loud behavior).
    s = Storage.for_uri(store)
    report_uri = f"{store}/_latence/runs/malformed/quality-report.json"
    assert s.exists(report_uri)
    text = s.read_text(report_uri)
    assert '"config"' in text
    # The malformed provider was never executed → the entities checkpoint was never finalized.
    ent_ckpt = f"{store}/_latence/runs/malformed/checkpoints/entities.jsonl"
    assert not s.exists(ent_ckpt)


def test_absent_profile_still_follows_request_and_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The DISTINCT-from-malformed case: a provider that declares NO profile is fine — it
    # follows the request and runs normally (None means "follow request", not "fatal").
    monkeypatch.setenv("LATENCE_CUDA", "0")
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner(registry=_registry_with("entity.no_profile", _NoProfileExtractor)).run(
        _pipeline(store, docs, entity_provider="entity.no_profile"), run_id="absent"
    )
    entities = next(m for m in report.stages if m.name == "entities")
    assert entities.error_count == 0
    assert entities.compute is None  # no profile declared → None columns, never fabricated
    assert entities.device_selected == "cpu"  # followed the request (either/None → cpu)
    assert entities.records_out >= 1  # it actually ran and extracted


def test_unknown_provider_still_defers_to_execution_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The import/load-error case must NOT be mapped to a config error here — it defers to the
    # normal execution path, which re-loads and raises the real KeyError (unknown provider).
    monkeypatch.setenv("LATENCE_CUDA", "0")
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    # A registry WITHOUT the named entity provider → load() raises KeyError at execution.
    runner = LocalRunner()
    with pytest.raises(KeyError):
        runner.run(_pipeline(store, docs, entity_provider="entity.does_not_exist"), run_id="unknown")


def test_profile_body_that_raises_also_fails_loud_with_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The OTHER author-bug shape (found by the P3-F2 correctness lens): a ``profile()``
    # classmethod whose BODY raises while building an INVALID ProviderProfile — pydantic
    # ValidationError (a ValueError, NOT a TypeError). It must get the SAME fail-loud
    # disposition as a malformed declaration: a typed ConfigError + a FAIL report, and the
    # provider is never executed. (Before the P3-F2 catch was widened past ``TypeError``,
    # this ValidationError escaped ``run()`` unmapped with no FAIL report.)
    monkeypatch.setenv("LATENCE_CUDA", "0")
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    runner = LocalRunner(registry=_registry_with("entity.raising", _RaisingProfileExtractor))
    with pytest.raises(ConfigError) as excinfo:
        runner.run(_pipeline(store, docs, entity_provider="entity.raising"), run_id="raising")
    assert "malformed ProviderProfile" in str(excinfo.value)
    assert "entities" in str(excinfo.value)

    # The report half: a FAIL report is persisted with the stable ``config`` category, and the
    # invalid-profile provider was never executed (its entities checkpoint never finalized).
    s = Storage.for_uri(store)
    report_uri = f"{store}/_latence/runs/raising/quality-report.json"
    assert s.exists(report_uri)
    assert '"config"' in s.read_text(report_uri)
    assert not s.exists(f"{store}/_latence/runs/raising/checkpoints/entities.jsonl")
