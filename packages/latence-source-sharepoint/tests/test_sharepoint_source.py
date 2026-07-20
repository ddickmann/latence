"""The SharePoint Source Provider's observable behaviour, driven through the faithful Graph fake.

Every assertion here is about what the connector *does* — the bytes it transfers, the Provenance
it stamps, the requests it makes, the errors it raises — never about a flag that records intent.
"""

from __future__ import annotations

import urllib.parse
from typing import TYPE_CHECKING, Any

import pytest
from latence_core.capability import Source
from latence_core.contracts import ParserInput
from latence_core.errors import ConfigError, ProviderError
from latence_core.state.delta_detector import DeltaDetector, ManifestEntry, content_hash
from latence_core.storage import Storage
from latence_source_sharepoint.provider import SharePointSource
from latence_source_sharepoint.testing import (
    DOWNLOAD_HOST,
    GRAPH_HOST,
    FakeFile,
    FakeGraphApi,
    FakeGraphTransport,
)

if TYPE_CHECKING:
    from pathlib import Path

TOKEN_ENV = "LATENCE_SHAREPOINT_TOKEN"


@pytest.fixture()
def storage(tmp_path: Path) -> Storage:
    """A Storage the Source ignores — SharePoint is a source SYSTEM, not a filesystem."""
    return Storage.for_uri(f"file://{tmp_path}")


def _api(*files: FakeFile) -> FakeGraphApi:
    return FakeGraphApi(files=list(files))


def _source(
    transport: FakeGraphTransport,
    *,
    monkeypatch: pytest.MonkeyPatch,
    token: str = "valid-token",
    **overrides: Any,
) -> SharePointSource:
    monkeypatch.setenv(TOKEN_ENV, token)
    config: dict[str, Any] = {
        "drive_id": transport.api.drive_id,
        "graph_base_url": GRAPH_HOST,
        "transport": transport,
    }
    config.update(overrides)
    return SharePointSource(config)


def _produce(source: SharePointSource, storage: Storage) -> list[ParserInput]:
    return list(source.produce(storage))


# ---------------------------------------------------------------------------
# The Source seam: it IS a Source, and it stamps Provenance
# ---------------------------------------------------------------------------


def test_satisfies_the_source_capability() -> None:
    assert isinstance(SharePointSource({"drive_id": "d"}), Source)


def test_lists_and_fetches_documents_with_sharepoint_provenance(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = FakeGraphTransport(
        _api(
            FakeFile(path="/Handbook/onboarding.md", content=b"# Onboarding\n\nWelcome.\n"),
            FakeFile(path="/nda.md", content=b"# NDA\n\nContoso and Fabrikam agree.\n"),
        )
    )
    inputs = _produce(_source(transport, monkeypatch=monkeypatch), storage)

    assert [i.provenance.file_name for i in inputs] == ["nda.md", "onboarding.md"]
    # Raw bytes are handed downstream undecoded — decoding belongs to the Parser (ADR-0019).
    assert inputs[0].content == b"# NDA\n\nContoso and Fabrikam agree.\n"
    for parser_input in inputs:
        assert parser_input.provenance.source_system == "sharepoint"
        assert parser_input.provenance.source_uri.startswith("sharepoint://drive-1/")
        assert "itemId=item-" in parser_input.provenance.source_uri
        assert parser_input.provenance.file_size == len(parser_input.content)
    # The document identity resolves back to the exact Graph item.
    handbook = next(i for i in inputs if i.provenance.file_name == "onboarding.md")
    assert handbook.provenance.source_uri == (
        "sharepoint://drive-1/Handbook/onboarding.md?itemId=item-Handbook-onboarding.md"
    )


def test_source_uri_survives_uri_syntax_in_a_legal_sharepoint_name(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``#``/space/``%`` are legal in SharePoint names and must not destroy the identity.

    Parsed, not substring-matched: an unescaped ``#`` pushes ``?itemId=`` into the URI
    *fragment*, so a real parser recovers no item id at all — and a literal ``%`` decodes to a
    different name. The URI is the only field carrying the Graph identity downstream, so it has
    to round-trip through the standard parser (audit R6-connectors_ingestion-10).
    """
    drive_path = "/Q1 #final/Report%20final & C++ draft.md"
    transport = FakeGraphTransport(_api(FakeFile(path=drive_path, content=b"# Q1\n")))
    (parser_input,) = _produce(_source(transport, monkeypatch=monkeypatch), storage)

    parts = urllib.parse.urlsplit(parser_input.provenance.source_uri)
    # The `#` must not have opened a fragment...
    assert parts.fragment == ""
    # ...so the Graph identity is recoverable by a parser, exactly as Graph reports it.
    assert urllib.parse.parse_qs(parts.query)["itemId"] == [transport.api.item_id(drive_path)]
    # ...and the human location decodes back to the drive path, `%` included and NOT decoded
    # into a space (that would name a different document).
    assert urllib.parse.unquote(parts.path) == drive_path


def test_only_configured_extensions_are_read(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = FakeGraphTransport(
        _api(
            FakeFile(path="/keep.md", content=b"# keep\n"),
            FakeFile(path="/skip.bin", content=b"\x00\x01"),
        )
    )
    inputs = _produce(_source(transport, monkeypatch=monkeypatch), storage)
    assert [i.provenance.file_name for i in inputs] == ["keep.md"]
    # A filtered-out document is never downloaded either.
    assert not any("skip" in url for url in transport.download_urls)


def test_traversal_is_deterministic_across_runs(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = [
        FakeFile(path="/b/second.md", content=b"second"),
        FakeFile(path="/a/first.md", content=b"first"),
        FakeFile(path="/a/nested/third.md", content=b"third"),
    ]
    first = _produce(_source(FakeGraphTransport(_api(*files)), monkeypatch=monkeypatch), storage)
    second = _produce(_source(FakeGraphTransport(_api(*files)), monkeypatch=monkeypatch), storage)
    assert [i.provenance.document_id for i in first] == [i.provenance.document_id for i in second]
    assert len(first) == 3


def test_duplicate_document_names_are_refused_and_name_strategy_path_resolves_them(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = [
        FakeFile(path="/2025/report.md", content=b"old"),
        FakeFile(path="/2026/report.md", content=b"new"),
    ]
    with pytest.raises(ConfigError, match="duplicate document name"):
        _produce(_source(FakeGraphTransport(_api(*files)), monkeypatch=monkeypatch), storage)

    resolved = _produce(
        _source(FakeGraphTransport(_api(*files)), monkeypatch=monkeypatch, name_strategy="path"),
        storage,
    )
    assert sorted(i.provenance.file_name for i in resolved) == [
        "2025_report.md",
        "2026_report.md",
    ]


# ---------------------------------------------------------------------------
# Delta: the content-hash manifest, and the skip that costs nothing
# ---------------------------------------------------------------------------


def test_unchanged_document_keeps_its_id_and_a_changed_one_gets_a_new_one(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _produce(
        _source(
            FakeGraphTransport(_api(FakeFile(path="/a.md", content=b"v1"))),
            monkeypatch=monkeypatch,
        ),
        storage,
    )
    same = _produce(
        _source(
            FakeGraphTransport(_api(FakeFile(path="/a.md", content=b"v1"))),
            monkeypatch=monkeypatch,
        ),
        storage,
    )
    changed = _produce(
        _source(
            FakeGraphTransport(_api(FakeFile(path="/a.md", content=b"v2"))),
            monkeypatch=monkeypatch,
        ),
        storage,
    )
    assert before[0].provenance.document_id == same[0].provenance.document_id
    assert before[0].provenance.document_id != changed[0].provenance.document_id


def test_excluded_document_is_skipped_without_transferring_a_byte(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Delta win: the DeltaRunner injects UNCHANGED ids, and this Source skips the download.

    Asserted by counting the fake transport's byte-transferring calls, because "it was skipped"
    is only true if the bytes never moved — a flag saying so would prove nothing.
    """
    files = [
        FakeFile(path="/unchanged.md", content=b"# unchanged\n"),
        FakeFile(path="/changed.md", content=b"# changed\n"),
    ]
    baseline_transport = FakeGraphTransport(_api(*files))
    baseline = _produce(_source(baseline_transport, monkeypatch=monkeypatch), storage)
    assert len(baseline_transport.download_urls) == 4  # 2 docs x (302 + redirect fetch)

    unchanged_id = next(
        i.provenance.document_id
        for i in baseline
        if i.provenance.file_name == "unchanged.md"
    )
    delta_transport = FakeGraphTransport(_api(*files))
    delta = _produce(
        _source(
            delta_transport, monkeypatch=monkeypatch, exclude_document_ids=[unchanged_id]
        ),
        storage,
    )

    assert [i.provenance.file_name for i in delta] == ["changed.md"]
    assert delta_transport.download_urls  # the changed document WAS fetched
    assert not any("unchanged" in url for url in delta_transport.download_urls)


def test_deletion_and_change_are_surfaced_by_the_content_hash_manifest(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs over a drive that changed → the core DeltaDetector classifies add/delete/unchanged."""

    def manifest(*files: FakeFile) -> dict[str, ManifestEntry]:
        produced = _produce(
            _source(FakeGraphTransport(_api(*files)), monkeypatch=monkeypatch), storage
        )
        return {
            i.provenance.document_id: ManifestEntry(
                document_id=i.provenance.document_id,
                content_hash=content_hash(
                    i.provenance.document_id,
                    i.content.decode("utf-8") if isinstance(i.content, bytes) else i.content,
                ),
                file_name=i.provenance.file_name,
            )
            for i in produced
        }

    previous = manifest(
        FakeFile(path="/keep.md", content=b"keep"),
        FakeFile(path="/edit.md", content=b"before"),
        FakeFile(path="/gone.md", content=b"gone"),
    )
    current = manifest(
        FakeFile(path="/keep.md", content=b"keep"),
        FakeFile(path="/edit.md", content=b"after"),
    )

    diff = DeltaDetector.diff(previous, current)
    names = {doc_id: entry.file_name for doc_id, entry in {**previous, **current}.items()}
    assert [names[d] for d in diff.unchanged] == ["keep.md"]
    assert [names[d] for d in diff.added] == ["edit.md"]
    assert sorted(names[d] for d in diff.deleted) == ["edit.md", "gone.md"]


def test_a_drive_without_hashes_falls_back_to_hashing_the_downloaded_bytes(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    hashless = FakeFile(path="/a.md", content=b"payload", hash_algorithm=None)
    first = _produce(
        _source(FakeGraphTransport(_api(hashless)), monkeypatch=monkeypatch), storage
    )
    second = _produce(
        _source(FakeGraphTransport(_api(hashless)), monkeypatch=monkeypatch), storage
    )
    assert first[0].provenance.document_id == second[0].provenance.document_id

    # Exclusion still works on that fallback id (it just cannot avoid the transfer).
    excluded = _produce(
        _source(
            FakeGraphTransport(_api(hashless)),
            monkeypatch=monkeypatch,
            exclude_document_ids=[first[0].provenance.document_id],
        ),
        storage,
    )
    assert excluded == []


def test_document_id_is_one_namespace_across_every_hash_facet_state(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-connectors_ingestion-10: unchanged bytes keep ONE id whatever Graph reports.

    Graph computes ``file.hashes`` asynchronously after upload and the algorithm differs by
    backend (``quickXorHash`` on SharePoint Online, ``sha256Hash`` on OneDrive Personal), so the
    SAME untouched item is listed hashless on one run and hashed on the next. If the id is a
    function of the facet rather than of the bytes, the DeltaRunner sees DELETED+ADDED for a
    document nobody touched: it is re-downloaded and re-extracted, and a no-op delta costs more
    than a full run. The id must be a function of the CONTENT alone.
    """
    body = b"# unchanged bytes\n"
    produced_ids: dict[str | None, str] = {}
    enumerated_ids: dict[str | None, list[str]] = {}
    for algorithm in (None, "quickXorHash", "sha256Hash"):
        item = FakeFile(path="/a.md", content=body, hash_algorithm=algorithm)
        produced_ids[algorithm] = _produce(
            _source(FakeGraphTransport(_api(item)), monkeypatch=monkeypatch), storage
        )[0].provenance.document_id
        enumerated_ids[algorithm] = list(
            _source(FakeGraphTransport(_api(item)), monkeypatch=monkeypatch).document_ids(
                storage
            )
        )

    assert len(set(produced_ids.values())) == 1, (
        "the same bytes minted different document_ids per hash facet state: " f"{produced_ids}"
    )
    # Whatever the enumeration seam DOES report must be exactly what produce() stamps, or the
    # intersection the runner does is against ids nothing ever stamps. It reports an id only
    # when the listing can content-address the item — a hash-less or quickXorHash listing is
    # left out rather than downloaded, because that discovery fetch doubles the cost of every
    # run (audit R3-connectors_ingestion-8); the document is then simply re-extracted.
    assert enumerated_ids == {
        None: [],
        "quickXorHash": [],
        "sha256Hash": [produced_ids["sha256Hash"]],
    }

    # ...and it is still a CONTENT address: different bytes, different id.
    changed = _produce(
        _source(
            FakeGraphTransport(_api(FakeFile(path="/a.md", content=b"# changed bytes\n"))),
            monkeypatch=monkeypatch,
        ),
        storage,
    )
    assert changed[0].provenance.document_id not in set(produced_ids.values())


def test_a_listing_hash_that_contradicts_the_bytes_is_a_typed_error(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0034: "same id ⇒ byte-identical" is enforced once the bytes are in hand, not assumed."""
    transport = FakeGraphTransport(_api(FakeFile(path="/a.md", content=b"# real\n")))
    transport.api.corrupt_listing_hash_for = "/a.md"
    with pytest.raises(ProviderError, match="content hash"):
        _produce(_source(transport, monkeypatch=monkeypatch), storage)


# ---------------------------------------------------------------------------
# Secrets: environment-only, enforced and unleaked
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        {"drive_id": "d", "token": "s3cret"},
        {"drive_id": "d", "auth": {"mode": "env_token", "client_secret": "s3cret"}},
        {"drive_id": "d", "auth": {"mode": "env_token", "credentials": {"password": "s3cret"}}},
    ],
)
def test_inlined_credentials_are_refused_by_config(config: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="environment-only"):
        SharePointSource(config)


def test_a_missing_token_env_var_is_a_config_error(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    transport = FakeGraphTransport(_api(FakeFile(path="/a.md", content=b"x")))
    source = SharePointSource(
        {"drive_id": "drive-1", "graph_base_url": GRAPH_HOST, "transport": transport}
    )
    with pytest.raises(ConfigError, match=TOKEN_ENV):
        _produce(source, storage)


def test_the_token_value_never_appears_in_an_error_message(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "super-secret-bearer-value"
    transport = FakeGraphTransport(_api(FakeFile(path="/a.md", content=b"x")))
    source = _source(transport, monkeypatch=monkeypatch, token=secret)
    with pytest.raises(ProviderError) as raised:
        _produce(source, storage)
    assert "401" in str(raised.value)
    assert secret not in str(raised.value)
    assert secret not in repr(source)


def test_the_bearer_token_is_never_replayed_to_the_download_host(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The faithful fake's download host rejects a credentialed request, as the real one does."""
    transport = FakeGraphTransport(_api(FakeFile(path="/a.md", content=b"payload")))
    _produce(_source(transport, monkeypatch=monkeypatch), storage)
    redirected = [call for call in transport.calls if call.url.startswith(DOWNLOAD_HOST)]
    assert redirected, "the download must go through Graph's 302 to a pre-authenticated URL"
    assert all(not call.had_authorization for call in redirected)


def test_client_credentials_exchange_reads_the_secret_from_the_environment(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api(FakeFile(path="/a.md", content=b"x"))
    transport = FakeGraphTransport(api)
    monkeypatch.setenv("SP_CLIENT_ID", api.client_id)
    monkeypatch.setenv("SP_CLIENT_SECRET", api.client_secret)
    source = SharePointSource(
        {
            "drive_id": api.drive_id,
            "graph_base_url": GRAPH_HOST,
            "transport": transport,
            "auth": {
                "mode": "client_credentials",
                "tenant_id": api.tenant_id,
                "client_id_env": "SP_CLIENT_ID",
                "client_secret_env": "SP_CLIENT_SECRET",
            },
        }
    )
    assert len(_produce(source, storage)) == 1
    # Exactly one token exchange for the whole run: the token is cached until near expiry.
    assert sum(1 for call in transport.calls if "oauth2" in call.url) == 1


def test_a_wrong_client_secret_is_rejected_the_way_entra_id_rejects_it(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api(FakeFile(path="/a.md", content=b"x"))
    transport = FakeGraphTransport(api)
    monkeypatch.setenv("SP_CLIENT_ID", api.client_id)
    monkeypatch.setenv("SP_CLIENT_SECRET", "wrong")
    source = SharePointSource(
        {
            "drive_id": api.drive_id,
            "graph_base_url": GRAPH_HOST,
            "transport": transport,
            "auth": {
                "mode": "client_credentials",
                "tenant_id": api.tenant_id,
                "client_id_env": "SP_CLIENT_ID",
                "client_secret_env": "SP_CLIENT_SECRET",
            },
        }
    )
    with pytest.raises(ProviderError) as raised:
        _produce(source, storage)
    assert "invalid_client" in str(raised.value)
    assert "wrong" not in str(raised.value)


def test_an_unknown_auth_mode_is_refused_rather_than_silently_downgraded() -> None:
    with pytest.raises(ConfigError, match="unknown auth mode"):
        SharePointSource({"drive_id": "d", "auth": {"mode": "anonymous"}})
