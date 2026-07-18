"""``SharePointSource`` — the SharePoint/OneDrive Provider of the Source Capability (#182).

Behind the existing Source seam (ADR-0014) with **no change to any Stage contract**: it
implements the same one-method ``produce(storage) -> Iterator[ParserInput]`` the reference
``source.local_folder`` implements, hands the *raw* bytes downstream for the Parser to decode
(ADR-0019), and stamps the same :class:`~latence_core.contracts.Provenance` /
:class:`~latence_core.contracts.Classification` every other Source stamps. Point a stack's
``source`` Stage at ``source.sharepoint`` and the rest of the pipeline is unchanged.

**Provenance** (the AC "source system and document identity, preserved to Export"):

* ``source_system`` — ``"sharepoint"``.
* ``source_uri`` — ``sharepoint://{drive_id}/{folder}/{name}?itemId={graph item id}``: the
  human location inside the drive *and* the stable Graph identity that resolves back to the
  document, in the one URI field the frozen contract provides. Every downstream record
  inherits it unchanged, so an exported chunk still names the SharePoint item it came from.
* ``document_id`` — the **content address** (see below).

**Delta** (ADR-0018) — the property that makes an incremental run cheap. Graph reports a
content hash for each ``driveItem`` (``quickXorHash`` on SharePoint Online, ``sha256Hash`` on
OneDrive Personal) in the *listing*, before any download. The connector mints
``document_id = sha256(<algorithm>:<hash>|<size>)`` from that, so:

* a document whose bytes did not change keeps the same id across runs — the DeltaRunner
  intersects ids with the parent Corpus Version, marks it UNCHANGED, and injects it into this
  Provider's ``exclude_document_ids``;
* **and this Provider then skips it without downloading it.** The exclusion is checked against
  the id derived from the *listing*, before :meth:`~latence_source_sharepoint.graph.GraphClient.download`
  is ever called. That is the whole point: an incremental run over a 50k-document library
  transfers only the documents that actually changed. The offline suite asserts it by counting
  the fake transport's download calls, not by reading a flag.
* a document that changed gets a new id (ADDED) and retires the old one (DELETED); a document
  removed from the library simply stops appearing in the listing, so the manifest diff surfaces
  it as DELETED. This is exactly the reference ``source.local_folder`` semantics — content
  addressing means "same id" is "byte-identical", which is what the Runner's cross-run reuse
  relies on. (A stable-identity Source would report UPDATED instead; see
  :mod:`latence_core.state.delta_detector`.)

If a drive reports no hash for an item, the connector falls back to downloading and hashing
the bytes — correct, just not free — rather than inventing an identity from mutable metadata
like ``eTag``, which would churn the corpus on a metadata-only edit.

**Secrets** are environment-only and enforced (:mod:`~latence_source_sharepoint.auth`): a
config that inlines a token is a :class:`~latence_core.errors.ConfigError`, so no credential
can reach the run manifest, the stack file, or git.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from latence_core.contracts import Classification, ParserInput, Provenance
from latence_core.errors import ConfigError
from latence_core.language import UNKNOWN_LANGUAGE, detect_language
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

from latence_source_sharepoint.auth import build_token_provider, reject_inline_secrets
from latence_source_sharepoint.graph import DEFAULT_GRAPH_BASE_URL, GraphClient
from latence_source_sharepoint.transport import HttpTransport, UrllibTransport

if TYPE_CHECKING:
    from collections.abc import Iterator

    from latence_core.storage import Storage

    from latence_source_sharepoint.graph import GraphItem

SOURCE_SYSTEM = "sharepoint"

# Mirrors the reference Source: only a text type can be language-sniffed from raw bytes;
# anything else is handed on with language ``und`` and classified authoritatively at Parse.
_TEXT_EXTENSIONS = frozenset({"txt", "md", "markdown", "text"})

# The default accepted extension set, matching ``source.local_folder`` so a stack can swap
# Sources without also having to widen the filter.
_DEFAULT_EXTENSIONS = frozenset({"txt", "md", "markdown"})

_NAME_STRATEGIES = frozenset({"item", "path"})


def _resolve_transport(spec: Any) -> HttpTransport:
    """Resolve ``config['transport']``: a live object, a ``"module:factory"`` string, or None.

    The string form is what a YAML stack can express, so an adopter behind a corporate proxy
    or mTLS supplies their own HTTP transport without forking the Provider — and it is how the
    offline dry-run path (``latence_source_sharepoint.testing:demo_transport``) is selected.
    """
    if spec is None:
        return UrllibTransport()
    if isinstance(spec, HttpTransport):
        return spec
    if not isinstance(spec, str):
        msg = (
            "SharePointSource 'transport' must be an HttpTransport, or a 'module:factory' "
            f"string naming a zero-argument factory that returns one; got {type(spec).__name__}."
        )
        raise ConfigError(msg)
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        msg = (
            f"SharePointSource 'transport' string {spec!r} is not a 'module:factory' "
            "reference (e.g. 'latence_source_sharepoint.testing:demo_transport')."
        )
        raise ConfigError(msg)
    from importlib import import_module

    try:
        factory = getattr(import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        msg = f"SharePointSource could not import the transport factory {spec!r}: {exc}"
        raise ConfigError(msg) from exc
    transport = factory()
    if not isinstance(transport, HttpTransport):
        msg = (
            f"SharePointSource transport factory {spec!r} returned "
            f"{type(transport).__name__}, which does not implement HttpTransport.request."
        )
        raise ConfigError(msg)
    return transport


def _string_list(config: dict[str, Any], key: str) -> list[str] | None:
    raw = config.get(key)
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(isinstance(entry, str) for entry in raw):
        msg = f"SharePointSource {key!r} must be a list of strings."
        raise ConfigError(msg)
    return [str(entry) for entry in raw]


class SharePointSource:
    """Source Provider that lists and fetches documents from a SharePoint document library.

    Config:
        drive_id: str — the Graph drive (document library) id. Required.
            ``GET /sites/{site-id}/drives`` lists them; ``GET /sites/{hostname}:/sites/{path}``
            resolves a site URL to a site id.
        root_path: str — drive-relative folder to read, ``""`` (the default) for the whole
            drive, ``"/Contracts"`` for one folder.
        extensions: list[str] | None — extensions to read (no dot). Defaults to text/markdown,
            like ``source.local_folder``; widen it to feed a PDF/office-capable Parser.
        name_strategy: ``"item"`` (default) | ``"path"`` — how ``Provenance.file_name`` is
            derived. Downstream keys (the gold-set harness) assume a unique file name per
            corpus, and a real document library routinely holds the same file name in two
            folders. ``"item"`` uses the bare name and raises on a collision (the reference
            Source's behaviour); ``"path"`` uses the drive-relative path with ``/`` replaced
            by ``_``, which is unique by construction.
        sensitivity / category — stamped onto every record's Classification.
        exclude_document_ids: list[str] — content-addressed ids to skip. The DeltaRunner
            injects the UNCHANGED ids here; this Provider skips them **without downloading**.
        auth: dict — ``{"mode": "env_token", "token_env": ...}`` (default) or
            ``{"mode": "client_credentials", "tenant_id": ..., "client_id_env": ...,
            "client_secret_env": ...}``. Names environment variables, never values.
        graph_base_url / page_size / max_retries / max_retry_after_seconds — Graph plumbing.
        transport — an ``HttpTransport``, or a ``"module:factory"`` string. Defaults to the
            standard-library HTTPS transport.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2).

        Pure-code and dependency-free: this Provider ships **no model weights**
        (``model_id=None``) and pulls no third-party library — it speaks Graph's REST/JSON API
        over the Python standard library, so its code license is this repo's Apache-2.0
        (verified per ADR-0012; weights-and-code trivially the same for a pure-code Provider)
        and ``latence-core`` takes no new dependency (ADR-0016).

        ``compute="cpu"`` (no accelerator), ``batch=False`` (Graph is fetched per item), and
        ``deterministic=True``: for a fixed drive state the listing is walked in sorted order
        and every document id is content-addressed, so two runs produce identical output. It is
        network-bound rather than compute-bound, hence a small declared footprint and no
        ``cost_per_1k`` (Graph is billed by tenant licensing, not per call).
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=32,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_weights=None,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-18",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        # Enforced BEFORE anything else: a config that inlines a credential never gets as far
        # as opening a connection, so it can never be written to the run manifest either.
        reject_inline_secrets(config)

        self._root_path = str(config.get("root_path", ""))
        self._sensitivity = str(config.get("sensitivity", "unknown"))
        category = config.get("category")
        self._category = None if category is None else str(category)

        raw_extensions = _string_list(config, "extensions")
        self._extensions = (
            _DEFAULT_EXTENSIONS
            if raw_extensions is None
            else frozenset(entry.lstrip(".").lower() for entry in raw_extensions)
        )
        self._excluded = frozenset(_string_list(config, "exclude_document_ids") or ())

        self._name_strategy = str(config.get("name_strategy", "item")).lower()
        if self._name_strategy not in _NAME_STRATEGIES:
            msg = (
                f"SharePointSource 'name_strategy' must be one of {sorted(_NAME_STRATEGIES)!r}; "
                f"got {self._name_strategy!r}."
            )
            raise ConfigError(msg)

        auth_config = config.get("auth") or {}
        if not isinstance(auth_config, dict):
            msg = "SharePointSource 'auth' must be a mapping."
            raise ConfigError(msg)

        transport = _resolve_transport(config.get("transport"))
        self._drive_id = str(config.get("drive_id", ""))
        self._client = GraphClient(
            transport=transport,
            token_provider=build_token_provider(auth_config, transport=transport),
            drive_id=self._drive_id,
            base_url=str(config.get("graph_base_url", DEFAULT_GRAPH_BASE_URL)),
            page_size=int(config.get("page_size", 200)),
            max_retries=int(config.get("max_retries", 5)),
            max_retry_after_seconds=float(config.get("max_retry_after_seconds", 300.0)),
        )

    def produce(self, storage: Storage) -> Iterator[ParserInput]:
        """Yield one :class:`ParserInput` per SharePoint document, raw bytes and all.

        ``storage`` is unused: SharePoint is a source *system*, not an fsspec filesystem, so
        the connector reaches it over Graph. The parameter stays because it is the Source
        Capability's signature and the Runner passes it — the seam is unchanged (ADR-0014).
        """
        seen_names: dict[str, str] = {}
        for item in self._client.list_files(self._root_path):
            extension = item.name.rsplit(".", 1)[-1].lower() if "." in item.name else ""
            if extension not in self._extensions:
                continue
            file_name = self._file_name(item)
            previous = seen_names.get(file_name)
            if previous is not None:
                msg = (
                    f"SharePointSource: duplicate document name {file_name!r} at "
                    f"{item.drive_path!r} and {previous!r}. Downstream keys (the gold-set "
                    "harness, Provenance) must not merge distinct documents. Set "
                    "'name_strategy: path' to disambiguate by folder, or narrow 'root_path'."
                )
                raise ConfigError(msg)
            seen_names[file_name] = item.drive_path

            # The Delta win: the identity comes from the LISTING when Graph reported a hash,
            # so an excluded (unchanged) document is skipped with zero bytes transferred.
            listed_id = _listed_document_id(item)
            if listed_id is not None:
                if listed_id in self._excluded:
                    continue
                data = self._client.download(item)
                document_id = listed_id
            else:
                data = self._client.download(item)
                document_id = "sha256:" + hashlib.sha256(data).hexdigest()
                if document_id in self._excluded:
                    continue

            language = (
                detect_language(data.decode("utf-8", errors="replace"))
                if extension in _TEXT_EXTENSIONS
                else UNKNOWN_LANGUAGE
            )
            yield ParserInput(
                provenance=Provenance(
                    source_uri=self._source_uri(item),
                    file_name=file_name,
                    file_type=extension,
                    file_size=len(data),
                    document_id=document_id,
                    source_system=SOURCE_SYSTEM,
                ),
                classification=Classification(
                    language=language,
                    category=self._category,
                    sensitivity=self._sensitivity,
                ),
                content=data,
            )

    def _file_name(self, item: GraphItem) -> str:
        if self._name_strategy == "path":
            return item.drive_path.lstrip("/").replace("/", "_")
        return item.name

    def _source_uri(self, item: GraphItem) -> str:
        return f"sharepoint://{self._drive_id}{item.drive_path}?itemId={item.item_id}"


def _listed_document_id(item: GraphItem) -> str | None:
    """The content address derivable from the listing alone, or ``None`` if Graph reported none.

    ``sha256`` over the algorithm-qualified hash and the size: stable across runs, changes iff
    the content changes, and — crucially — computable without downloading the document.
    """
    if item.content_hash is None:
        return None
    digest = hashlib.sha256(f"{item.content_hash}|{item.size}".encode()).hexdigest()
    return "sha256:" + digest
