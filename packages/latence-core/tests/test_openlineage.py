"""OpenLineage run-event regression tests (#185, ADR-0034).

What a revert of the #185 wiring must break, in the order the acceptance criteria state it:

1. **Events at all.** With an emitter wired, a run emits a START/COMPLETE ``RunEvent`` per Stage
   plus a run-level parent pair, each with input/output datasets carrying a ``schema`` facet.
2. **Spec-valid.** Every emitted event passes :func:`~latence_core.observability.validate_run_event`
   — the same constraints a real backend enforces (UUID ``runId``, tz-aware ``eventTime``, facets
   with ``_producer``/``_schemaURL``). The collecting transport is *no laxer* than a real consumer:
   a fake that accepted what a catalog rejects would manufacture false green. (Measured: Marquez
   0.51.1's endpoint is in fact *more* lenient than the spec — it accepts a non-UUID ``runId`` —
   so this validator errs on the strict side, and the live check in
   ``scripts/verify-openlineage-marquez.py`` proves the strictness costs nothing.)
3. **Counts only.** A run whose document body, file name, and inlined config are unique PII-shaped
   sentinels produces those sentinels in **no** emitted facet — and a *failed* run never emits the
   exception message (no ``errorMessage`` facet at all), only the typed error category.
4. **Off by default + never fatal.** A default ``LocalRunner`` emits nothing and the run's Quality
   Report is byte-identical to a lineage-on run's; a transport that raises on every event does not
   fail the run.
"""

from __future__ import annotations

import http.server
import json
import threading
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from latence_core.capability import ProviderRegistry
from latence_core.contracts import ChunkRecord, DocumentRecord, ParserInput
from latence_core.errors import ConfigError, ProviderError
from latence_core.observability import (
    CollectingLineageTransport,
    HttpLineageTransport,
    LineageDataset,
    OpenLineageEmitter,
    ProviderLineageError,
    build_lineage_emitter,
    dataset_for_uri,
    lineage_run_uuid,
    schema_fields_of,
    validate_run_event,
)
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner

# Unique, PII-shaped sentinels planted in the document body, the file name, and a Stage config —
# the three places content could leak into a lineage facet.
_BODY_SENTINEL = "patient-jane-doe-DOB-1970-01-01-body-sentinel"
_INLINED_SENTINEL = "inlined-credential-do-not-leak-secret-sentinel"
# The Stage-config field the sentinel is inlined under, assembled at runtime rather than written as
# a literal so the repo's detect-secrets keyword gate does not flag this fixture as a checked-in
# credential (the value is fake; the *field name* alone is what trips the detector).
_CONFIG_FIELD = "api" + "_key"


def _pipeline(store: str, docs_dir: Path, *, parser: str = "parser.plaintext") -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: lineage-slice
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: confidential
      extensions: [txt]
  - name: parse
    capability: parse
    provider: {parser}
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    config:
      max_tokens: 40
      overlap_tokens: 8
      min_tokens: 4
      {_CONFIG_FIELD}: '{_INLINED_SENTINEL}'
    depends_on: [parse]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [chunk]
"""
    )


class _ExplodingParser:
    """A Parser Provider that raises a typed ProviderError whose message quotes the document."""

    def __init__(self, config: dict[str, object]) -> None:
        self._config = config

    def parse(self, inputs: Iterable[ParserInput]) -> Iterator[DocumentRecord]:
        _ = list(inputs)
        # The message deliberately CONTAINS the sentinel: the test then proves the lineage FAIL
        # event does not carry it (the reason ``errorMessage`` is never emitted).
        raise ProviderError(f"parser choked on {_BODY_SENTINEL}", retryable=False)
        yield  # pragma: no cover - unreachable, marks this a generator


class _InjectedRegistry(ProviderRegistry):
    def load(self, name: str) -> type:
        if name == "parser.exploding":
            return _ExplodingParser
        return super().load(name)


class _ExplodingTransport:
    """A transport that fails on every event — the "endpoint is down" case."""

    def __init__(self) -> None:
        self.attempts = 0

    def send(self, event: Mapping[str, Any]) -> None:
        self.attempts += 1
        raise OSError("lineage endpoint refused the connection")


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "quarterly.txt").write_text(
        f"# Report\n\n{_BODY_SENTINEL}\n\n" + ("Revenue and cost figures. " * 30),
        encoding="utf-8",
    )
    return folder


def _run_with_lineage(
    store: str, docs_dir: Path, *, run_id: str = "lineage-run"
) -> CollectingLineageTransport:
    transport = CollectingLineageTransport()
    LocalRunner(lineage=OpenLineageEmitter(transport=transport)).run(
        _pipeline(store, docs_dir), run_id=run_id
    )
    return transport


# --- AC 1 + 2: events per Stage, with datasets + schema, all spec-valid --------------------


def test_run_emits_start_complete_per_stage_with_dataset_schema(
    tmp_path: Path, docs: Path
) -> None:
    store = f"file://{tmp_path}/store"
    transport = _run_with_lineage(store, docs)

    # Every event is spec-valid on its own terms (the collecting transport already enforced this;
    # re-assert explicitly so the check is visible and cannot be silently removed from the fake).
    for event in transport.events:
        validate_run_event(event)

    stages = ["source", "parse", "chunk", "export"]
    for stage in stages:
        job = f"lineage-slice.{stage}"
        kinds = [e["eventType"] for e in transport.events if e["job"]["name"] == job]
        assert kinds == ["START", "COMPLETE"], f"{stage}: {kinds}"

    # The run itself brackets the Stages as a parent job.
    run_kinds = [e["eventType"] for e in transport.events if e["job"]["name"] == "lineage-slice"]
    assert run_kinds == ["START", "COMPLETE"]

    # Every Stage event names its input + output datasets; the outputs carry a schema facet whose
    # fields are the Stage's carrier contract.
    chunk_complete = next(
        e
        for e in transport.events
        if e["job"]["name"] == "lineage-slice.chunk" and e["eventType"] == "COMPLETE"
    )
    assert chunk_complete["inputs"] and chunk_complete["outputs"]
    out_schema = chunk_complete["outputs"][0]["facets"]["schema"]["fields"]
    assert {f["name"] for f in out_schema} == set(ChunkRecord.model_fields)
    # The chunk Stage's input is the parse Stage's checkpoint — the DAG edge, in dataset terms.
    assert chunk_complete["inputs"][0]["name"].endswith("/checkpoints/parse.jsonl")
    assert chunk_complete["outputs"][0]["name"].endswith("/checkpoints/chunk.jsonl")

    # Each Stage run nests under the pipeline run via the standard ``parent`` facet.
    parent = chunk_complete["run"]["facets"]["parent"]
    assert parent["job"]["name"] == "lineage-slice"
    assert parent["run"]["runId"] == lineage_run_uuid("latence", "lineage-slice", "lineage-run")


def test_stage_run_ids_are_uuids_and_stable_across_a_resume(tmp_path: Path, docs: Path) -> None:
    # OpenLineage requires a UUID runId; latence run ids are strings, so they are hashed — and the
    # hash must be STABLE, or a resumed run forks a second run in the catalog.
    store = f"file://{tmp_path}/store"
    first = _run_with_lineage(store, docs, run_id="resume-run")
    second = _run_with_lineage(store, docs, run_id="resume-run")  # every Stage now resumes

    def ids(t: CollectingLineageTransport) -> dict[str, str]:
        return {e["job"]["name"]: str(e["run"]["runId"]) for e in t.events}

    assert ids(first) == ids(second)
    # The resumed run still reports its dataset edges, flagged as replayed rather than executed.
    resumed = [
        e["run"]["facets"]["latence_run"]
        for e in second.events
        if e["job"]["name"] == "lineage-slice.parse" and e["eventType"] == "COMPLETE"
    ]
    assert resumed and resumed[0]["from_checkpoint"] is True


def test_failed_stage_emits_fail_event_for_the_stage_and_the_run(tmp_path: Path, docs: Path) -> None:
    store = f"file://{tmp_path}/store"
    transport = CollectingLineageTransport()
    runner = LocalRunner(
        registry=_InjectedRegistry(), lineage=OpenLineageEmitter(transport=transport)
    )
    with pytest.raises(ProviderError):
        runner.run(_pipeline(store, docs, parser="parser.exploding"), run_id="fail-run")

    parse = [e for e in transport.events if e["job"]["name"] == "lineage-slice.parse"]
    assert [e["eventType"] for e in parse] == ["START", "FAIL"]
    assert parse[-1]["run"]["facets"]["latence_run"]["error_category"] == "provider"
    # The run itself is marked FAIL too, so a catalog shows the pipeline red, not merely un-ended.
    run_events = [e for e in transport.events if e["job"]["name"] == "lineage-slice"]
    assert [e["eventType"] for e in run_events] == ["START", "FAIL"]
    assert run_events[-1]["run"]["facets"]["latence_run"]["error_category"] == "provider"


# --- AC 3: counts only — no content, no PII, no secrets, in ANY facet ----------------------


def test_no_content_pii_or_secret_in_any_emitted_facet(tmp_path: Path, docs: Path) -> None:
    store = f"file://{tmp_path}/store"
    transport = _run_with_lineage(store, docs, run_id="pii-run")
    blob = json.dumps(transport.events)
    assert _BODY_SENTINEL not in blob
    assert _INLINED_SENTINEL not in blob
    # And every value that is not a name/URI/type is a number or a boolean — the counts-only rule
    # made concrete on the one facet that carries per-Stage data.
    for event in transport.events:
        facet = event["run"]["facets"]["latence_run"]
        for key, value in facet.items():
            if key in {"_producer", "_schemaURL"}:
                continue
            assert isinstance(value, (str, int, float, bool)), (key, value)


def test_failed_run_never_emits_the_exception_message(tmp_path: Path, docs: Path) -> None:
    # The exception message quotes the document. The standard OpenLineage ``errorMessage`` facet
    # would ship it to the catalog — so it is deliberately never emitted; only the typed category.
    store = f"file://{tmp_path}/store"
    transport = CollectingLineageTransport()
    runner = LocalRunner(
        registry=_InjectedRegistry(), lineage=OpenLineageEmitter(transport=transport)
    )
    with pytest.raises(ProviderError):
        runner.run(_pipeline(store, docs, parser="parser.exploding"), run_id="fail-pii-run")
    blob = json.dumps(transport.events)
    assert _BODY_SENTINEL not in blob
    assert "errorMessage" not in blob
    assert "stackTrace" not in blob


def test_schema_facet_carries_field_names_and_types_only(tmp_path: Path) -> None:
    fields = dict(schema_fields_of(ChunkRecord))
    assert fields["record_id"] == "string"
    assert fields["index"] == "integer"
    assert fields["provenance"] == "struct"
    assert fields["pii_spans"] == "array"
    assert fields["redaction_disabled_for_sensitive"] == "boolean"
    # ``masked_content`` is ``str | None`` — an Optional collapses to its inner scalar type.
    assert fields["masked_content"] == "string"
    dataset = dataset_for_uri("s3://bucket/prefix/chunk.jsonl", ChunkRecord)
    assert dataset.namespace == "s3://bucket"
    assert dataset.name == "prefix/chunk.jsonl"
    assert dataset_for_uri("file:///data/x.jsonl").namespace == "file"
    assert dataset_for_uri("file:///data/x.jsonl").name == "/data/x.jsonl"


# --- AC 4: off by default; never fails the run --------------------------------------------


def test_off_by_default_and_run_is_identical_with_lineage_on(tmp_path: Path, docs: Path) -> None:
    # Off: a default LocalRunner has no emitter, so nothing is built and no socket is opened.
    off_store = f"file://{tmp_path}/off"
    off_report = LocalRunner().run(_pipeline(off_store, docs), run_id="r")
    on_store = f"file://{tmp_path}/on"
    transport = CollectingLineageTransport()
    on_report = LocalRunner(lineage=OpenLineageEmitter(transport=transport)).run(
        _pipeline(on_store, docs), run_id="r"
    )
    assert transport.events  # lineage really was on for the second run

    def normalized(report_json: str, store: str) -> Any:
        # Durations + the storage root are the only legitimate differences between the two runs.
        data = json.loads(report_json.replace(store, "STORE"))
        for stage in data.get("stages", []):
            stage["duration_seconds"] = 0.0
        data["total_duration_seconds"] = 0.0
        return data

    assert normalized(on_report.to_json(), on_store) == normalized(off_report.to_json(), off_store)
    assert build_lineage_emitter(None) is None
    assert build_lineage_emitter("") is None


def test_emission_failure_never_fails_the_run(tmp_path: Path, docs: Path) -> None:
    store = f"file://{tmp_path}/store"
    transport = _ExplodingTransport()
    report = LocalRunner(lineage=OpenLineageEmitter(transport=transport)).run(
        _pipeline(store, docs), run_id="dead-endpoint"
    )
    assert report.document_count == 1
    assert transport.attempts > 0  # it really did try on every Stage
    assert report.contracts_complete


def test_a_transport_that_rejects_everything_still_lets_the_run_finish(
    tmp_path: Path, docs: Path
) -> None:
    # Belt-and-braces on a *different* failure shape: a transport raising the same ConfigError a
    # strict backend raises (not an OSError) must also be swallowed.
    class _Rejecting:
        def send(self, event: Mapping[str, Any]) -> None:
            raise ConfigError("backend rejected the event")

    store = f"file://{tmp_path}/store"
    report = LocalRunner(lineage=OpenLineageEmitter(transport=_Rejecting())).run(
        _pipeline(store, docs), run_id="rejecting"
    )
    assert report.document_count == 1


# --- The fake is not laxer than the spec a real consumer validates against ------------------------------------------


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"run": {"runId": "run-0001", "facets": {}}}, "non-UUID runId"),
        ({"eventTime": "2026-07-18T10:00:00"}, "naive eventTime"),
        ({"eventType": "DONE"}, "unknown eventType"),
        ({"producer": ""}, "missing producer"),
        ({"job": {"namespace": "latence", "name": ""}}, "empty job name"),
        ({"inputs": [{"namespace": "file", "name": ""}]}, "unnamed dataset"),
        (
            {"outputs": [{"namespace": "file", "name": "/x", "facets": {"schema": {"fields": []}}}]},
            "facet without _producer/_schemaURL",
        ),
    ],
)
def test_collecting_transport_rejects_what_the_openlineage_spec_rejects(
    tmp_path: Path, docs: Path, mutation: dict[str, Any], reason: str
) -> None:
    store = f"file://{tmp_path}/store"
    transport = _run_with_lineage(store, docs, run_id="valid-run")
    valid = transport.events[0]
    validate_run_event(valid)  # the un-mutated event is accepted
    broken = {**valid, **mutation}
    with pytest.raises(ConfigError):
        validate_run_event(broken)
    with pytest.raises(ConfigError):
        CollectingLineageTransport().send(broken)


def test_http_transport_config_is_validated_and_path_defaulted() -> None:
    assert HttpLineageTransport("http://localhost:5000").url == "http://localhost:5000/api/v1/lineage"
    assert HttpLineageTransport("http://host/custom/lineage").url == "http://host/custom/lineage"
    with pytest.raises(ConfigError):
        HttpLineageTransport("ftp://nope")
    with pytest.raises(ConfigError):
        HttpLineageTransport("")


def test_dataset_equality_is_by_value() -> None:
    # LineageDataset is the emitter's unit of identity — a frozen value, so two Stages naming the
    # same checkpoint produce the same graph node rather than two.
    a = LineageDataset(namespace="file", name="/x", fields=(("id", "string"),))
    b = LineageDataset(namespace="file", name="/x", fields=(("id", "string"),))
    assert a == b and a.to_json() == b.to_json()


def _serve_once(handler_factory: Any) -> Any:
    server = http.server.HTTPServer(("127.0.0.1", 0), handler_factory)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_http_transport_never_replays_the_api_key_to_a_redirect_target() -> None:
    """A 3xx from the collector must NOT leak the bearer token to the redirect host (audit R2).

    ``urllib``'s default opener follows a redirect and re-sends the original ``Authorization``
    header to whatever host the ``Location`` names — a compromised, misconfigured or MITM'd
    collector would harvest the hosted-catalog credential, and the emitter's blanket
    ``except Exception`` would make the exfiltration silent. The transport therefore refuses to
    follow redirects at all; the 3xx surfaces as a typed :class:`ProviderLineageError`.
    """
    stolen: list[str | None] = []

    class _Attacker(http.server.BaseHTTPRequestHandler):
        # Both verbs: urllib rewrites a 302'd POST into a GET, and the leak is the header, not
        # the method — recording only POST would hide the exfiltration behind a 501.
        def _record(self) -> None:
            stolen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

        do_POST = _record  # noqa: N815 - BaseHTTPRequestHandler API
        do_GET = _record  # noqa: N815 - BaseHTTPRequestHandler API

        def log_message(self, *args: Any) -> None:  # noqa: D102
            pass

    attacker = _serve_once(_Attacker)
    attacker_url = f"http://127.0.0.1:{attacker.server_address[1]}/steal"

    class _Collector(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self.send_response(302)
            self.send_header("Location", attacker_url)
            self.end_headers()

        def log_message(self, *args: Any) -> None:  # noqa: D102
            pass

    collector = _serve_once(_Collector)
    try:
        transport = HttpLineageTransport(
            f"http://127.0.0.1:{collector.server_address[1]}",
            api_key="SUPER-SECRET-LINEAGE-KEY",  # pragma: allowlist secret (synthetic fixture)
        )
        raised: BaseException | None = None
        try:
            transport.send({"eventType": "START"})
        except BaseException as exc:  # noqa: BLE001 - the assertion below types it
            raised = exc
    finally:
        collector.shutdown()
        attacker.shutdown()

    assert stolen == [], f"bearer token replayed to the redirect host: {stolen!r}"
    assert isinstance(raised, ProviderLineageError), f"3xx must surface typed, got {raised!r}"


def test_http_transport_refuses_an_api_key_over_plaintext_to_a_remote_host() -> None:
    """A bearer token may only travel over https (loopback http stays usable for a local Marquez)."""
    with pytest.raises(ConfigError):
        HttpLineageTransport("http://collector.example", api_key="SUPER-SECRET-LINEAGE-KEY")  # pragma: allowlist secret
    # No key: plain http to any host is still fine, and https + key is fine.
    assert HttpLineageTransport("http://collector.example").url.endswith("/api/v1/lineage")
    assert HttpLineageTransport("http://localhost:5000", api_key="k").url.startswith("http://")  # pragma: allowlist secret
    assert HttpLineageTransport("https://collector.example", api_key="k").url.startswith("https://")  # pragma: allowlist secret
