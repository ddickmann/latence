"""Runner: checkpoint, resume-skips-completed, and full-pipeline idempotency."""

from __future__ import annotations

import json
from pathlib import Path

from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.storage import Storage


def _pipeline(docs_dir: Path, store: str) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: t
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs_dir}'}}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [parse]
"""
    )


def test_run_writes_checkpoints(docs_dir: Path, tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    p = _pipeline(docs_dir, store)
    LocalRunner().run(p, run_id="r1")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/r1"
    for stage in ("source", "parse", "export"):
        assert s.exists(f"{base}/checkpoints/{stage}.jsonl")
    assert s.exists(f"{base}/manifest.json")
    assert s.exists(f"{base}/wal.log")
    assert s.exists(f"{base}/quality-report.json")


def test_resume_skips_completed_stages(docs_dir: Path, tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    p = _pipeline(docs_dir, store)
    LocalRunner().run(p, run_id="r1")

    # Second run over the same run_id: every stage must resume from checkpoint.
    report2 = LocalRunner().run(p, run_id="r1")
    assert all(m.from_checkpoint for m in report2.stages)
    assert report2.contracts_complete


def test_partial_kill_then_resume(docs_dir: Path, tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    p = _pipeline(docs_dir, store)
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/r1"

    # Simulate a crash after Source committed but before Parse/Export by running
    # once, then deleting the parse+export checkpoints (and their exports).
    LocalRunner().run(p, run_id="r1")
    for stage in ("parse", "export"):
        s._fs.rm(f"{base}/checkpoints/{stage}.jsonl")

    report = LocalRunner().run(p, run_id="r1")
    # Source resumed from checkpoint; parse+export re-ran.
    by_name = {m.name: m for m in report.stages}
    assert by_name["source"].from_checkpoint is True
    assert by_name["parse"].from_checkpoint is False
    assert by_name["export"].from_checkpoint is False
    assert s.exists(f"{base}/export/records.jsonl")


def test_idempotent_output(docs_dir: Path, tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    p = _pipeline(docs_dir, store)
    s = Storage.for_uri(store)
    jsonl = f"{store}/_latence/runs/r1/export/records.jsonl"

    LocalRunner().run(p, run_id="r1")
    first = s.read_bytes(jsonl)
    LocalRunner().run(p, run_id="r1")
    second = s.read_bytes(jsonl)
    # No-op re-run leaves byte-identical output.
    assert first == second


# -- converging DAG ({a, b} -> c) multi-input + resume ----------------------


def _converging_pipeline(dir_a: Path, dir_b: Path, store: str) -> Pipeline:
    """Two Source stages fanning INTO one Parse stage: c depends_on [sa, sb]."""
    return Pipeline.from_yaml_text(
        f"""
name: converge
storage_uri: {store}
stages:
  - name: sa
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{dir_a}'}}
  - name: sb
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{dir_b}'}}
  - name: c
    capability: parse
    provider: parser.plaintext
    depends_on: [sa, sb]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [c]
"""
    )


def test_converging_dag_gathers_both_parents(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    (dir_a / "a1.txt").write_text("alpha document one\n", encoding="utf-8")
    dir_b = tmp_path / "b"
    dir_b.mkdir()
    (dir_b / "b1.txt").write_text("beta document two\n", encoding="utf-8")
    (dir_b / "b2.txt").write_text("beta document three\n", encoding="utf-8")

    store = f"file://{tmp_path}/store"
    p = _converging_pipeline(dir_a, dir_b, store)
    report = LocalRunner().run(p, run_id="cv")

    # c (the converging Parse) must see ALL three docs: 1 from sa + 2 from sb.
    by_name = {m.name: m for m in report.stages}
    assert by_name["c"].records_in == 3
    assert by_name["c"].records_out == 3
    # And the report's document_count reflects the whole converged corpus.
    assert report.document_count == 3


def test_converging_dag_resume_sees_both_parents_current_outputs(tmp_path: Path) -> None:
    """The regression guard: delete ONE parent's checkpoint + the child's, resume.

    The recomputed parent and the checkpointed parent must BOTH feed the child.
    The old single-``current`` threading corrupted this: on resume the child saw
    only the last-loaded parent, silently dropping the other's records.
    """
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    (dir_a / "a1.txt").write_text("alpha only\n", encoding="utf-8")
    dir_b = tmp_path / "b"
    dir_b.mkdir()
    (dir_b / "b1.txt").write_text("beta one\n", encoding="utf-8")
    (dir_b / "b2.txt").write_text("beta two\n", encoding="utf-8")

    store = f"file://{tmp_path}/store"
    p = _converging_pipeline(dir_a, dir_b, store)
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/cv"

    LocalRunner().run(p, run_id="cv")

    # Simulate a crash that lost sa's checkpoint and everything downstream of it,
    # while sb's checkpoint survived.
    for stage in ("sa", "c", "export"):
        s._fs.rm(f"{base}/checkpoints/{stage}.jsonl")

    report = LocalRunner().run(p, run_id="cv")
    by_name = {m.name: m for m in report.stages}

    # sa re-ran (checkpoint absent); sb resumed from its surviving checkpoint.
    assert by_name["sa"].from_checkpoint is False
    assert by_name["sb"].from_checkpoint is True
    # THE BUG GUARD: c gathered BOTH parents' current outputs — 1 (sa) + 2 (sb).
    assert by_name["c"].records_in == 3
    assert by_name["c"].records_out == 3
    assert report.document_count == 3

    # And the exported corpus really contains all three source documents.
    import json

    jsonl = f"{base}/export/records.jsonl"
    rows = [json.loads(line) for line in s.read_text(jsonl).splitlines() if line.strip()]
    file_names = {r["provenance"]["file_name"] for r in rows}
    assert file_names == {"a1.txt", "b1.txt", "b2.txt"}


# -- issue #30: secret scrubbing on manifest write --------------------------


def test_scrub_sensitive_redacts_nested_secret_keys() -> None:
    """The scrubber redacts secret-looking keys anywhere in a nested manifest tree."""
    from latence_core.runner import scrub_sensitive

    payload = {
        "name": "p",
        "stages": [
            {
                "provider": "relation.llm",
                "config": {
                    "model": "gpt-4o",
                    "api_key": "sk-INLINE-SECRET",
                    "nested": {"auth_token": "tok-123", "client_secret": "cs-9"},
                    "params": [{"password": "hunter2"}, {"note": "safe"}],
                },
            }
        ],
    }
    scrubbed = scrub_sensitive(payload)
    import json

    dumped = json.dumps(scrubbed)
    # No secret VALUE survives anywhere in the serialized manifest.
    for leaked in ("sk-INLINE-SECRET", "tok-123", "cs-9", "hunter2"):
        assert leaked not in dumped, f"secret {leaked!r} leaked into manifest"
    # Non-secret data is untouched; secret KEYS remain (documenting a secret was set).
    assert scrubbed["stages"][0]["config"]["model"] == "gpt-4o"  # type: ignore[index]
    assert scrubbed["stages"][0]["config"]["api_key"] == "***REDACTED***"  # type: ignore[index]
    assert scrubbed["stages"][0]["config"]["params"][1]["note"] == "safe"  # type: ignore[index]


def test_manifest_never_persists_inlined_api_key(docs_dir: Path, tmp_path: Path) -> None:
    """End-to-end: an operator who inlines api_key into a Stage config must not have it
    written cleartext to manifest.json (issue #30). Reverting the scrub fails here."""
    store = f"file://{tmp_path}/store"
    secret = "sk-DO-NOT-LEAK-1234567890"
    p = Pipeline.from_yaml_text(
        f"""
name: t
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs_dir}', api_key: '{secret}', auth_token: '{secret}'}}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [parse]
"""
    )
    LocalRunner().run(p, run_id="r1")
    s = Storage.for_uri(store)
    manifest = s.read_text(f"{store}/_latence/runs/r1/manifest.json")
    assert secret not in manifest, "inlined secret leaked into manifest.json"
    assert "***REDACTED***" in manifest


# -- audit-R7: certificate-based credentials were the scrubber's blind spot --


def test_scrub_sensitive_redacts_certificate_and_passphrase_keys() -> None:
    """Certificate auth key names carry NONE of the api_key/token/secret/password substrings.

    Entra ID app-only Graph access is recommended by Microsoft with a *certificate*, so
    ``private_key``/``certificate``/``passphrase`` are the realistic inlined credentials — and
    every one of them slipped through the substring list unredacted before audit-R7. The short
    forms (``client_cert``, ``tls_pem``) match on a whole key part, so an innocent key that
    merely *contains* those letters (``certainty_threshold``) is left alone.
    """
    from latence_core.runner import scrub_sensitive

    payload = {
        "config": {
            # A literal PEM header would trip the repo's detect-secrets gate, so the value is
            # unmistakably fake. What is under test is the KEY NAME, not the value's shape.
            "private_key": "PEM-BODY-LEAKME-not-a-real-key",
            "certificate": "CERTBLOB",
            "passphrase": "hunter2",
            "client_cert": "CERT2",
            "tls_pem": "PEMBLOB",
            "keystore": "KSBLOB",
            # Not a secret: must survive, or the manifest stops documenting the run.
            "certainty_threshold": 0.8,
            "model": "gpt-4o",
        }
    }
    scrubbed = scrub_sensitive(payload)
    dumped = json.dumps(scrubbed)
    for leaked in ("LEAKME", "CERTBLOB", "hunter2", "CERT2", "PEMBLOB", "KSBLOB"):
        assert leaked not in dumped, f"credential {leaked!r} leaked into manifest"
    # ``scrub_sensitive`` returns ``object`` (it walks an arbitrary JSON-able tree), so narrow
    # with isinstance rather than laundering the type through a ``type: ignore``.
    assert isinstance(scrubbed, dict)
    config = scrubbed["config"]
    assert isinstance(config, dict)
    for key in ("private_key", "certificate", "passphrase", "client_cert", "tls_pem", "keystore"):
        assert config[key] == "***REDACTED***", key
    # Over-redaction is a defect too: a substring match on "cert" would eat this one.
    assert config["certainty_threshold"] == 0.8
    assert config["model"] == "gpt-4o"


def test_manifest_never_persists_an_inlined_private_key(docs_dir: Path, tmp_path: Path) -> None:
    """End-to-end (audit-R7): a PEM private key + passphrase inlined into a Stage config.

    Before the fix this run wrote the key VERBATIM to ``manifest.json`` — and for a SharePoint
    stage it did so *before* ``reject_inline_secrets`` could abort the run, so the operator got
    a ConfigError **and** a cleartext key persisted in the run tree.
    """
    store = f"file://{tmp_path}/store"
    # Unmistakably fake stand-in for a PEM body: a literal PEM header in a committed test file
    # trips the repo's detect-secrets gate, and the key NAME is what is under test here.
    pem = "PEM-BODY-LEAKME-not-a-real-key"
    passphrase = "hunter2-DO-NOT-LEAK"
    p = Pipeline.from_yaml_text(
        f"""
name: t
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs_dir}', private_key: '{pem}', passphrase: '{passphrase}',
              certificate: 'CERTBLOB'}}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [parse]
"""
    )
    LocalRunner().run(p, run_id="r1")
    s = Storage.for_uri(store)
    manifest = s.read_text(f"{store}/_latence/runs/r1/manifest.json")
    for leaked in (pem, "LEAKME", passphrase, "CERTBLOB"):
        assert leaked not in manifest, f"{leaked!r} leaked into manifest.json"
    assert "***REDACTED***" in manifest
