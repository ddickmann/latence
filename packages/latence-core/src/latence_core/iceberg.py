"""OPTIONAL Iceberg-table registration for the Export (#184, ADR-0017).

The Parquet deliverable stays the source of truth. This module adds ONE optional touchpoint on
top of it: the corpus Parquet the Export already wrote is *also* registered as an Apache Iceberg
table in the customer's own **Iceberg REST catalog** (Unity, Polaris, Glue-via-REST, Lakekeeper,
the Apache ``iceberg-rest-fixture`` …), so the same bytes become governable tables with no glue
code.

Why registration and not a second materialisation
-------------------------------------------------
ADR-0017 says the AI-ready deliverable is *files*, never a database. Iceberg is still files plus
table metadata, so the only new thing is the catalog pointer. Accordingly this module NEVER
re-serialises the corpus through a different writer: it takes the Parquet the Export produced,
publishes an **immutable, run-stamped copy** of it next to the deliverable
(``<out_dir>/iceberg-data/<basename>-<token>.parquet``) and hands that path to Iceberg's
``add_files``. The copy is a streamed byte copy through the :class:`~latence_core.storage.Storage`
seam (O(chunk) memory, never O(corpus)), and it exists for one reason: an Iceberg data file must be
**immutable**, while ``records.parquet`` is deliberately rewritten by every run. Pointing the
catalog at ``records.parquet`` itself would leave every previous snapshot dangling at a path whose
bytes have since changed — a silently-corrupt table. So the deliverable keeps its stable name and
the catalog gets content that can never move under it.

Delta semantics (honest)
------------------------
``mode`` picks what a *second* run means, and neither choice is magic:

* ``append`` (default) — the run's data file is appended as a new snapshot. This is right for a
  delta run whose Export emitted only the newly affected records (ADR-0018/0029). Running a **full**
  Export twice in append mode therefore duplicates every row; that is the documented,
  unsurprising consequence of "append", not a bug.
* ``overwrite`` — the run's data file becomes the table's entire contents (a delete-all + add in one
  transaction, i.e. ONE atomic commit; readers see the old snapshot or the new one, never a gap).
  This is right for a full re-export. Superseded data files are left on disk, unreferenced by the
  current snapshot but still reachable through Iceberg time travel — expiring them is the catalog
  operator's retention policy, not ours.

The heavy dependency
--------------------
``pyiceberg`` is imported **lazily, inside the call**, and is NOT a runtime dependency of
``latence-core`` (ADR-0016). It ships as the ``latence-core[iceberg]`` extra, exactly like the
optional observability exporters (ADR-0047): absent + unconfigured is the default and costs
nothing; configured-but-absent fails loudly at wiring time with the install line, never silently.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from os.path import abspath
from typing import TYPE_CHECKING, Any

from latence_core.errors import ConfigError, ProviderError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

    import pyarrow as pa

    from latence_core.storage import Storage

# The two documented delta dispositions (see the module docstring).
_APPEND = "append"
_OVERWRITE = "overwrite"
_MODES = (_APPEND, _OVERWRITE)

# Table/snapshot property keys. Namespaced so they never collide with the catalog's own or with
# Iceberg's reserved ``write.*`` / ``schema.*`` keys, and identical on the table and on every
# snapshot so a governance reader can answer "which contract version produced this?" from either.
PROP_CONTRACT_VERSION = "latence.contract.schema_version"
PROP_RECORD_VERSIONS = "latence.record.schema_versions"
PROP_SOURCE = "latence.export.source"
PROP_MODE = "latence.export.mode"

_INSTALL_HINT = (
    "The Iceberg Export output needs the optional 'pyiceberg' dependency: "
    "pip install 'latence-core[iceberg]'."
)


@dataclass(frozen=True)
class IcebergTarget:
    """The validated ``iceberg`` Export config block.

    Parsing is separated from doing so a malformed block is a loud
    :class:`~latence_core.errors.ConfigError` at Export **wiring** time — before a single record is
    read — rather than a surprise after a long pipeline has already written its deliverable.
    """

    uri: str
    namespace: tuple[str, ...]
    table: str
    mode: str
    warehouse: str | None
    catalog_name: str
    properties: dict[str, str]
    table_properties: dict[str, str]

    @property
    def identifier(self) -> tuple[str, ...]:
        """The table's full catalog identifier (``(*namespace, table)``)."""
        return (*self.namespace, self.table)

    @classmethod
    def parse(cls, spec: Any, *, default_table: str) -> IcebergTarget:
        """Validate one ``iceberg`` config block into a target, or raise ``ConfigError``."""
        if not isinstance(spec, dict):
            msg = (
                "Export config 'iceberg' must be a mapping "
                "{'uri': <REST catalog endpoint>, ...} or absent; "
                f"got {type(spec).__name__}."
            )
            raise ConfigError(msg)
        uri = spec.get("uri")
        if not isinstance(uri, str) or not uri.strip():
            msg = (
                "Export config 'iceberg.uri' is required and must be the non-empty base URL of an "
                "Iceberg REST catalog (e.g. 'http://localhost:8181')."
            )
            raise ConfigError(msg)
        mode = str(spec.get("mode", _APPEND))
        if mode not in _MODES:
            msg = (
                f"Export config 'iceberg.mode' must be one of {list(_MODES)}; got {mode!r}. "
                "'append' adds this run's records as a new snapshot (the delta-run disposition); "
                "'overwrite' replaces the table contents with this run (the full re-export "
                "disposition)."
            )
            raise ConfigError(msg)
        namespace = cls._parse_namespace(spec.get("namespace", "latence"))
        table = spec.get("table", default_table)
        if not isinstance(table, str) or not table.strip():
            msg = "Export config 'iceberg.table' must be a non-empty string if present."
            raise ConfigError(msg)
        warehouse = spec.get("warehouse")
        if warehouse is not None and not isinstance(warehouse, str):
            msg = "Export config 'iceberg.warehouse' must be a string if present."
            raise ConfigError(msg)
        catalog_name = spec.get("catalog_name", "latence")
        if not isinstance(catalog_name, str) or not catalog_name.strip():
            msg = "Export config 'iceberg.catalog_name' must be a non-empty string if present."
            raise ConfigError(msg)
        return cls(
            uri=uri,
            namespace=namespace,
            table=table,
            mode=mode,
            warehouse=warehouse,
            catalog_name=catalog_name,
            properties=cls._parse_str_map(spec.get("properties"), "iceberg.properties"),
            table_properties=cls._parse_str_map(
                spec.get("table_properties"), "iceberg.table_properties"
            ),
        )

    @staticmethod
    def _parse_namespace(value: Any) -> tuple[str, ...]:
        """A namespace is ``"a.b"`` or ``["a", "b"]``; both normalise to a level tuple."""
        levels: list[str]
        if isinstance(value, str):
            levels = [part for part in value.split(".") if part]
        elif isinstance(value, (list, tuple)):
            levels = [str(part) for part in value if str(part)]
        else:
            msg = (
                "Export config 'iceberg.namespace' must be a dotted string ('lake.latence') or a "
                f"list of levels; got {type(value).__name__}."
            )
            raise ConfigError(msg)
        if not levels:
            msg = "Export config 'iceberg.namespace' must name at least one namespace level."
            raise ConfigError(msg)
        return tuple(levels)

    @staticmethod
    def _parse_str_map(value: Any, key: str) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            msg = f"Export config '{key}' must be a mapping of string keys to values if present."
            raise ConfigError(msg)
        return {str(k): str(v) for k, v in value.items()}


class IcebergCorpusTable:
    """Registers the Export's Parquet deliverable as a table in an Iceberg REST catalog.

    A deep module behind a two-call interface: construct it from the ``iceberg`` config block
    (which validates), then :meth:`register` once per Export run. Everything else — the REST
    session, namespace/table creation, schema conversion, the immutable data-file publication, the
    append-vs-overwrite transaction and the version stamping — is internal.
    """

    def __init__(self, spec: Any, *, default_table: str) -> None:
        self._target = IcebergTarget.parse(spec, default_table=default_table)

    @property
    def target(self) -> IcebergTarget:
        """The validated target (the identifier a caller can report or assert on)."""
        return self._target

    def register(
        self,
        *,
        storage: Storage,
        parquet_uri: str,
        out_dir: str,
        basename: str,
        arrow_schema: pa.Schema,
        contract_version: int,
        record_versions: Iterable[int],
    ) -> str:
        """Publish this run's corpus into the Iceberg table; return the registered data-file URI.

        ``parquet_uri`` is the Export's just-finalized ``records.parquet``. It is copied — streamed,
        never buffered — to an immutable, run-stamped sibling under ``<out_dir>/iceberg-data/``,
        and THAT path is what the catalog references (see the module docstring on why the
        deliverable's own, rewritten-every-run path must not be handed to Iceberg).

        Every failure on this path — catalog construction, namespace/table resolution, the data-file
        copy, the commit — surfaces as a typed :class:`~latence_core.errors.ProviderError` naming
        the table and the catalog (ADR-0034). A raw ``pyiceberg`` exception never escapes.
        """
        data_uri = "<not yet published>"
        try:
            catalog = self._catalog()
            table = self._ensure_table(catalog, arrow_schema)
            data_uri = self._publish_data_file(storage, parquet_uri, out_dir, basename)
            stamps = self._stamps(parquet_uri, contract_version, record_versions)
            with table.transaction() as tx:
                if self._target.mode == _OVERWRITE and table.current_snapshot() is not None:
                    # Delete-all + add in ONE transaction ⇒ one atomic catalog commit: a concurrent
                    # reader sees the previous snapshot or the new one, never an empty table.
                    # Guarded on there being something to delete: a first run into a fresh table
                    # has no rows, and issuing the delete anyway only earns a "did not match any
                    # records" warning for an operation that is definitionally a no-op.
                    from pyiceberg.expressions import AlwaysTrue

                    tx.delete(AlwaysTrue(), snapshot_properties=stamps)
                tx.set_properties(**stamps)
                tx.add_files([_absolute_uri(data_uri)], snapshot_properties=stamps)
        except (ConfigError, ProviderError):
            # Already typed and already specific (the install hint, the unreachable catalog, the
            # namespace/table failures below). Re-wrapping would only bury the precise message.
            raise
        except Exception as exc:  # noqa: BLE001 - normalise any catalog/commit failure
            msg = (
                f"Iceberg registration of {data_uri!r} into "
                f"{'.'.join(self._target.identifier)} via {self._target.uri!r} failed: {exc}"
            )
            raise ProviderError(msg) from exc
        return data_uri

    def _catalog(self) -> Any:
        """The real ``pyiceberg`` REST catalog client for the configured endpoint."""
        try:
            from pyiceberg.catalog.rest import RestCatalog
        except ImportError as exc:  # pragma: no cover - exercised by the import-guard test
            raise ConfigError(_INSTALL_HINT) from exc
        properties = dict(self._target.properties)
        properties["uri"] = self._target.uri
        if self._target.warehouse is not None:
            properties["warehouse"] = self._target.warehouse
        try:
            return RestCatalog(self._target.catalog_name, **properties)
        except Exception as exc:  # noqa: BLE001 - normalise connection/handshake failures
            msg = f"Iceberg REST catalog at {self._target.uri!r} is unreachable or refused: {exc}"
            raise ProviderError(msg) from exc

    def _ensure_table(self, catalog: Any, arrow_schema: pa.Schema) -> Any:
        """Load the target table, creating the namespace + table (from the Parquet schema) if new.

        The Iceberg schema is derived from the **exact Arrow schema of the Parquet deliverable**, so
        the table mirrors the documented consumer contract column-for-column: the same
        ``record_id`` / ``schema_version`` / ``provenance`` / ``classification`` / ``content`` /
        ``media_type`` spine, plus exactly whichever optional signal columns (``embedding``,
        ``sparse_*``, ``fde_embedding``, the W13 context columns) this Export was configured to
        write. A schema drift between an existing table and this run therefore surfaces from
        Iceberg's own ``add_files`` validation rather than from a guess made here.

        A catalog fault anywhere here (an unavailable server, an auth expiry, a rejected create) is
        normalised into a :class:`~latence_core.errors.ProviderError` naming the table and the
        catalog URI, never re-raised as a raw ``pyiceberg`` exception (ADR-0034). The one fault that
        is not an error is a LOST CREATE RACE: a concurrent Export creating the same table between
        this run's load and its create earns ``TableAlreadyExistsError``, and the right response is
        to load the table it just created and carry on.
        """
        from pyiceberg.exceptions import (
            NamespaceAlreadyExistsError,
            NoSuchTableError,
            TableAlreadyExistsError,
        )

        try:
            catalog.create_namespace(self._target.namespace)
        except NamespaceAlreadyExistsError:
            pass
        except Exception as exc:  # noqa: BLE001 - normalise catalog failures
            msg = (
                f"Iceberg namespace {'.'.join(self._target.namespace)!r} could not be created via "
                f"{self._target.uri!r}: {exc}"
            )
            raise ProviderError(msg) from exc
        try:
            return catalog.load_table(self._target.identifier)
        except NoSuchTableError:
            pass  # Expected on the first run into a fresh catalog: create it below.
        except Exception as exc:  # noqa: BLE001 - normalise catalog failures
            raise self._table_error("loaded", exc) from exc
        try:
            return catalog.create_table(
                self._target.identifier,
                schema=arrow_schema,
                properties=dict(self._target.table_properties),
            )
        except TableAlreadyExistsError:
            pass  # Lost create race — a concurrent writer got there first; load THEIR table.
        except Exception as exc:  # noqa: BLE001 - normalise catalog failures
            raise self._table_error("created", exc) from exc
        try:
            return catalog.load_table(self._target.identifier)
        except Exception as exc:  # noqa: BLE001 - normalise catalog failures
            raise self._table_error("loaded after a concurrent create", exc) from exc

    def _table_error(self, verb: str, exc: Exception) -> ProviderError:
        """The one typed error shape for every table-resolution failure (ADR-0034)."""
        msg = (
            f"Iceberg table {'.'.join(self._target.identifier)!r} could not be {verb} via "
            f"{self._target.uri!r}: {exc}"
        )
        return ProviderError(msg)

    def _stamps(
        self, parquet_uri: str, contract_version: int, record_versions: Iterable[int]
    ) -> dict[str, str]:
        """The version stamp written onto BOTH the table properties and the snapshot summary.

        ``latence.contract.schema_version`` is the framework contract version this build emits (the
        same constant that defaults every record's ``schema_version`` field and therefore the
        Parquet ``schema_version`` column) and ``latence.record.schema_versions`` is what the rows
        of THIS run actually carried — normally one value equal to the contract version, and more
        than one only for a corpus that mixes records produced by different framework versions. Both
        are stamped so the table never claims a version its own rows do not have.
        """
        observed = sorted(set(record_versions))
        return {
            PROP_CONTRACT_VERSION: str(contract_version),
            PROP_RECORD_VERSIONS: ",".join(str(v) for v in observed),
            PROP_SOURCE: parquet_uri,
            PROP_MODE: self._target.mode,
        }

    @staticmethod
    def _publish_data_file(
        storage: Storage, parquet_uri: str, out_dir: str, basename: str
    ) -> str:
        """Stream-copy the deliverable to an immutable, run-stamped Iceberg data file.

        Streamed through the Storage seam in fixed-size chunks (``shutil.copyfileobj``), so the
        Export's O(batch) memory bound (ADR-0033) survives the optional output, and published with
        the same temp-then-rename discipline as every other artifact (ADR-0010), so a crash mid-copy
        never leaves a half-written file for the catalog to point at.
        """
        data_dir = f"{out_dir}/iceberg-data"
        storage.makedirs(data_dir)
        data_uri = f"{data_dir}/{basename}-{uuid.uuid4().hex}.parquet"
        tmp_uri = storage.temp_uri(data_uri)
        with storage.open(parquet_uri, "rb") as src, storage.open(tmp_uri, "wb") as dst:
            shutil.copyfileobj(src, dst)
        storage.finalize_atomic(tmp_uri, data_uri)
        return data_uri


def _absolute_uri(uri: str) -> str:
    """A scheme-qualified, absolute URI — what an Iceberg data-file path must be.

    Storage URIs may be bare local paths (``/runs/x/export/...``); Iceberg records the data-file
    location verbatim in its manifest, so a relative or scheme-less path would be unresolvable for
    any other engine reading the table. Object-store URIs already carry their scheme and pass
    through untouched.
    """
    if "://" in uri:
        return uri
    return f"file://{abspath(uri)}"


def build_iceberg_output(spec: Any, *, default_table: str) -> IcebergCorpusTable | None:
    """The Export's wiring hook: ``None`` when the block is absent (the default), else a target.

    Off by default is the whole point (#184): with no ``iceberg`` block the Export never constructs
    this object, never imports ``pyiceberg``, and writes byte-identical output.
    """
    if spec is None:
        return None
    return IcebergCorpusTable(spec, default_table=default_table)


__all__ = [
    "PROP_CONTRACT_VERSION",
    "PROP_MODE",
    "PROP_RECORD_VERSIONS",
    "PROP_SOURCE",
    "IcebergCorpusTable",
    "IcebergTarget",
    "build_iceberg_output",
]
