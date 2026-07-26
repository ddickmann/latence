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
publishes an **immutable, content-addressed copy** of it into the target table's own location
(``<table.location()>/data/<basename>-<sha256>.parquet``) and hands that path to Iceberg's
``add_files``. The copy is a streamed byte copy through the :class:`~latence_core.storage.Storage`
seam (O(chunk) memory, never O(corpus)), and it exists for one reason: an Iceberg data file must be
**immutable**, while ``records.parquet`` is deliberately rewritten by every run. Pointing the
catalog at ``records.parquet`` itself would leave every previous snapshot dangling at a path whose
bytes have since changed — a silently-corrupt table. So the deliverable keeps its stable name and
the catalog gets content that can never move under it.

Why the TABLE's location and not the Export's ``out_dir``
---------------------------------------------------------
Immutable bytes are only half of what a catalog reference needs; the other half is a **durable
path**. The Export's ``out_dir`` is ``<storage_uri>/_latence/runs/<run_id>/export``
(:meth:`~latence_core.runner.LocalRunner._export_dir`) — the single most ephemeral directory in the
system. :meth:`~latence_core.delta_runner.DeltaRunner._purge_run_artifacts` ``remove_tree``s every
entry under ``_latence/runs`` on EVERY purge, of ANY document, because a run tree holds a purged
document's raw text in re-derivable artifacts. A data file published there was therefore destroyed
by an erasure that had nothing to do with it, while the committed snapshot went on recording its
absolute path verbatim in the Avro manifest: ``table.scan()`` raised ``FileNotFoundError`` for
good, and in ``append`` mode the post-purge re-export landed under a NEW run id, so the dangling
path stayed referenced forever. Ordinary run-retention cleanup broke it the same way.

The table's own location is the one place with the right lifetime: it belongs to the catalog, whose
operator already owns retention for it (``expire_snapshots`` / ``remove_orphan_files`` scan exactly
there), and nothing in this framework ever deletes it. It is also where every other Iceberg writer
puts its data files. A consequence worth stating plainly: enabling the ``iceberg`` block now leaves
the Export deliverable byte-for-byte identical — the optional output adds nothing to ``export/``.

The copy is named after the **sha256 of its own bytes**, not a random per-run token, because this
path is part of an export deliverable and the framework promises exports are reproducible: a random
name would put a hidden ``uuid4`` on the export path and make ``latence stack validate``'s
determinism check (which walks ``export/`` recursively and compares two fresh runs key-for-key)
report ``export diverged`` on a perfectly deterministic pipeline. Content addressing also *is* the
immutability guarantee rather than merely approximating it — a path named after its own digest can
only ever hold one byte sequence — and it stops repeated runs into a stable ``out_dir`` from
accumulating a full-corpus copy each.

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

import re
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
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

# The streaming copy/hash unit for the published data file. Fixed and modest so the optional
# output inherits the Export's O(batch) memory bound (ADR-0033) regardless of corpus size.
_COPY_CHUNK_BYTES = 1 << 20

# Table/snapshot property keys. Namespaced so they never collide with the catalog's own or with
# Iceberg's reserved ``write.*`` / ``schema.*`` keys, and identical on the table and on every
# snapshot so a governance reader can answer "which contract version produced this?" from either.
PROP_CONTRACT_VERSION = "latence.contract.schema_version"
PROP_RECORD_VERSIONS = "latence.record.schema_versions"
PROP_SOURCE = "latence.export.source"
PROP_MODE = "latence.export.mode"
#: Stamped on the snapshot a Purge-driven erasure commits, so the reason the table went empty is
#: legible from the table's own history rather than only from an operator's memory.
PROP_ERASURE = "latence.export.erasure"

#: The exact shape :meth:`IcebergCorpusTable._publish_data_file` gives every file it publishes:
#: ``<basename>-<sha256 of the bytes>.parquet``. It is the OWNERSHIP SCOPE of the erasure sweep —
#: the analogue of the Qdrant sink's ``_latence_id_namespace`` marker. A file in the table's data
#: directory that does not match was written by some other engine, and the sweep never touches it
#: (Iceberg's own writers name files ``<partition>-<task>-<uuid4>.parquet``, whose trailing segment
#: is a 36-char dashed UUID, never a bare 64-hex digest — so the two spaces cannot collide).
_PUBLISHED_DATA_FILE = re.compile(r"-[0-9a-f]{64}\.parquet$")

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
        basename: str,
        arrow_schema: pa.Schema,
        contract_version: int,
        record_versions: Iterable[int],
    ) -> str:
        """Publish this run's corpus into the Iceberg table; return the registered data-file URI.

        ``parquet_uri`` is the Export's just-finalized ``records.parquet``. It is copied — streamed,
        never buffered — to an immutable, content-addressed file under the target table's own
        location (``<table.location()>/data/``), and THAT path is what the catalog references (see
        the module docstring on why the deliverable's own, rewritten-every-run path must not be
        handed to Iceberg, and why the copy must not live in the ephemeral run tree either).

        Every failure on this path — catalog construction, namespace/table resolution, the data-file
        copy, the commit — surfaces as a typed :class:`~latence_core.errors.ProviderError` naming
        the table and the catalog (ADR-0034). A raw ``pyiceberg`` exception never escapes.

        A failure AFTER the copy is published also cleans the copy up (see
        :meth:`_discard_unreferenced`): the file is published before the transaction opens because
        ``add_files`` can only reference a file that already exists, so a commit that never lands
        would otherwise leave a full corpus-sized Parquet inside the deliverable directory that no
        manifest references and no catalog-side tool can reach. A failure *during* the publication
        cleans up its own half-written temp copy the same way (:meth:`_publication_failed`), so no
        failure on this path — before, during or after — leaves corpus bytes in the warehouse.
        """
        data_uri = "<not yet published>"
        published_here = False
        data_storage = storage
        table = None
        try:
            catalog = self._catalog()
            table = self._ensure_table(catalog, arrow_schema)
            data_storage, data_dir = self._data_location(table)
            data_uri, published_here = self._publish_data_file(
                storage, data_storage, parquet_uri, data_dir, basename
            )
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
                # ``check_duplicate_files=False``: the data file is content-addressed, so a run
                # that re-exports IDENTICAL bytes into the SAME out_dir now maps onto the one
                # existing path instead of a fresh random copy of the same bytes. Iceberg's
                # duplicate guard would turn that into a hard failure and silently change the
                # documented delta semantics — ``append`` says a repeated FULL export duplicates
                # every row (see the module docstring), and it must keep saying that. Referencing
                # one immutable file from two snapshots is exactly the old behaviour minus the
                # redundant second copy on disk.
                tx.add_files(
                    [_absolute_uri(data_uri)],
                    snapshot_properties=stamps,
                    check_duplicate_files=False,
                )
        except (ConfigError, ProviderError):
            # Already typed and already specific (the install hint, the unreachable catalog, the
            # namespace/table failures below, and a failed data-file publication). Re-wrapping would
            # only bury the precise message. Each of these either happens BEFORE anything is written
            # to the warehouse (catalog construction, namespace creation, table load/create, drift
            # refusal) or has ALREADY swept what it wrote and said so in its own message
            # (:meth:`_publication_failed`), so there is nothing left for this handler to clean.
            raise
        except Exception as exc:  # noqa: BLE001 - normalise any catalog/commit failure
            cleanup = (
                self._discard_unreferenced(data_storage, data_uri, exc, table)
                if published_here
                else ""
            )
            msg = (
                f"Iceberg registration of {data_uri!r} into "
                f"{'.'.join(self._target.identifier)} via {self._target.uri!r} failed: {exc}"
                f"{cleanup}"
            )
            raise ProviderError(msg) from exc
        return data_uri

    def erase(self) -> str:
        """Erase this derived table's contents for a Purge; return a counts-only operator note.

        The GDPR half of the optional output (audit R7-invariants-1). A Purge is a *physical*
        erasure — ``DeltaRunner._delete`` deletes the source file, expunges the derived records from
        every committed Corpus Version and ``remove_tree``s the whole ``_latence/runs`` tree — but
        the data file this module publishes lives in the CATALOG's warehouse, which nothing in the
        purge path reaches. It is a full copy of the corpus, ``content`` included, so a purge that
        skipped it left the erased text readable through an ordinary ``table.scan()`` while the
        tutorial and the threat model both promised removal "everywhere".

        Erasing is *total*, not row-scoped, and that is a deliberate consequence of the deliverable
        contract rather than a shortcut: the exported schema is the flat consumer contract
        (``record_id`` / ``provenance`` / ``content`` / …) with the document id living INSIDE the
        ``provenance`` JSON string, so there is no column an Iceberg predicate could filter a
        purged document out by. Totality is also safe here in a way it would never be for the corpus
        itself, because ADR-0017 makes this table strictly DERIVED: the caller runs this BEFORE the
        purge's survivor re-run, whose own :meth:`register` republishes exactly the surviving corpus
        — in ``append`` mode as much as in ``overwrite``, since it is appending onto an emptied
        table. The table therefore converges on the survivors within the same purge command.

        Two steps, in this order:

        1. **The snapshot.** A delete-all commit, so the table's CURRENT snapshot references none of
           the pre-purge data files. Skipped when the table has no snapshot yet (nothing to delete).
        2. **The bytes.** Every data file THIS framework published into the table's ``data/``
           directory (:data:`_PUBLISHED_DATA_FILE` — content-addressed, therefore unambiguously
           ours) is physically removed. Step 1 alone would leave the erased text sitting in the
           warehouse, reachable by time travel and by anyone with filesystem access, which is not an
           erasure at all.

        The price, stated plainly because erasure always has one: **pre-purge snapshots become
        unreadable.** Their manifests still name files that no longer exist, so time-travelling to a
        snapshot older than the purge fails. That is what "physically removed everywhere and cannot
        be recovered" means; a right-to-be-forgotten erasure that left the old snapshots readable
        would not be one. The catalog operator can drop the stale metadata with ``expire_snapshots``
        whenever they like.

        Fail-closed and typed (ADR-0034): if the catalog is unreachable, or ``pyiceberg`` is not
        installed while an ``iceberg`` block IS configured, this raises rather than letting the
        purge report a success it did not achieve. A table that was never created is a no-op.
        """
        catalog = self._catalog()
        table = self._load_for_erasure(catalog)
        if table is None:
            return ""
        data_storage, data_dir = self._data_location(table)
        try:
            if table.current_snapshot() is not None:
                from pyiceberg.expressions import AlwaysTrue

                with table.transaction() as tx:
                    tx.delete(
                        AlwaysTrue(),
                        snapshot_properties={PROP_ERASURE: "purge", PROP_MODE: self._target.mode},
                    )
        except Exception as exc:  # noqa: BLE001 - normalise any commit failure (ADR-0034)
            msg = (
                f"Iceberg table {'.'.join(self._target.identifier)!r} at {self._target.uri!r} "
                f"could not be erased for a Purge: {exc}. The purged document's derived rows may "
                "still be readable from this table; erase or drop it before treating the erasure "
                "as complete."
            )
            raise ProviderError(msg) from exc
        removed = self._sweep_published_data_files(data_storage, data_dir)
        return f"erased {'.'.join(self._target.identifier)} (data files removed: {removed})"

    def _load_for_erasure(self, catalog: Any) -> Any | None:
        """The target table, or ``None`` when it was never created (nothing was ever published)."""
        from pyiceberg.exceptions import NoSuchNamespaceError, NoSuchTableError

        try:
            return catalog.load_table(self._target.identifier)
        except (NoSuchTableError, NoSuchNamespaceError):
            # Nothing was ever registered here, so there is nothing to erase. Deliberately NOT an
            # error: a purge on a stack whose Iceberg output has not run yet must still succeed.
            return None
        except Exception as exc:  # noqa: BLE001 - normalise catalog failures
            raise self._table_error("loaded for a Purge erasure", exc) from exc

    def _sweep_published_data_files(self, storage: Storage, data_dir: str) -> int:
        """Physically remove the corpus copies THIS framework published; return how many.

        Scoped by :data:`_PUBLISHED_DATA_FILE`, so a table this Export shares with another engine
        keeps that engine's files. Counts only (ADR-0034): no path, no content, ever surfaces in the
        note this returns.
        """
        try:
            entries = storage.ls(data_dir)
        except FileNotFoundError:
            return 0
        except Exception as exc:  # noqa: BLE001 - normalise any filesystem failure
            msg = (
                f"Iceberg table {'.'.join(self._target.identifier)!r} had its rows deleted for a "
                f"Purge, but its data directory could not be listed to remove the published corpus "
                f"copies: {exc}. Those files still hold the purged text; remove them manually."
            )
            raise ProviderError(msg) from exc
        removed = 0
        for entry in entries:
            if not _PUBLISHED_DATA_FILE.search(entry):
                continue
            try:
                storage.remove(entry)
            except Exception as exc:  # noqa: BLE001 - a partial erasure must never look complete
                msg = (
                    f"Iceberg table {'.'.join(self._target.identifier)!r} had its rows deleted for "
                    f"a Purge, but a published corpus copy in its data directory could not be "
                    f"removed: {exc}. That file still holds the purged text; remove it manually."
                )
                raise ProviderError(msg) from exc
            removed += 1
        return removed

    def _discard_unreferenced(
        self, storage: Storage, data_uri: str, exc: BaseException, table: Any
    ) -> str:
        """Delete the data file THIS run published when its commit did not land; return the note.

        The published copy is the whole corpus, ``content`` included. Once the transaction has
        failed no snapshot references it, so ``expire_snapshots`` can never collect it (it only
        deletes files recorded in an expired snapshot's manifests). It now lands in the table's own
        location, where the catalog operator's ``remove_orphan_files`` *could* eventually reach it —
        but "eventually, if they run it" is not a disposal guarantee for a full-corpus copy of
        possibly sensitive text, so a registration that failed still cleans up after itself and the
        warehouse is left exactly as the failed run found it.

        Two deliberate exceptions to that:

        * **Unknown commit state.** ``CommitStateUnknownException`` is pyiceberg's signal that the
          catalog may have APPLIED the commit before the response was lost (a 500/502/504 on the
          commit endpoint). Deleting a file a live snapshot references would corrupt the table, so
          the file is kept and the typed error says so — the same discipline every other Iceberg
          engine follows for this case. This one path can still leave a file behind; that is the
          safe side of a genuinely ambiguous outcome, not an oversight.
        * **A file some snapshot references.** Only a copy :meth:`_publish_data_file` actually
          finalized is eligible; when content addressing mapped this run onto a file an EARLIER run
          already committed, that file belongs to that run's snapshot and is never touched (the
          caller gates on ``published_here``). ``published_here`` is a NECESSARY but not a
          SUFFICIENT condition, which is why the catalog is asked as well (audit
          R7-data_contracts_sinks-4): it is decided by ``exists(data_uri)`` followed by the
          publish, a check-then-act pair. Two concurrent Exports of IDENTICAL
          corpus content content-address onto the SAME path; if both evaluate ``exists()`` before
          either rename lands, both observe ``False``, both finalize that path and BOTH claim
          authorship. A failed commit then deleted a file the other run had already committed a
          snapshot for, and because the manifest records the absolute path forever the governed
          table was permanently unreadable — the R6 dangling-data-file mode re-entered by a
          different door. So the destructive step now answers to the CATALOG, which is the
          serialization point for what is referenced: :meth:`_snapshot_references` asks the table's
          fresh metadata whether ANY snapshot (not just the current one — ``overwrite`` leaves
          earlier snapshots reachable through time travel) points at this path, and a file that is
          referenced, or whose status cannot be established at all, is KEPT and reported.

        A residual window is accepted deliberately: if the concurrent run has published but not yet
        committed when the check runs, its file can still be swept. That case fails LOUDLY on the
        other side — ``add_files`` reads each Parquet footer, so a missing file surfaces there as a
        typed :class:`~latence_core.errors.ProviderError` — rather than silently corrupting a
        committed table, and closing it entirely would need a cross-writer lock the Storage seam
        does not offer.

        The removal is best-effort and never masks the original failure: a failure to delete is
        appended to the same typed error, so the operator learns the path to sweep (ADR-0034 — no
        silent errors, and no second exception hiding the first).
        """
        from pyiceberg.exceptions import CommitStateUnknownException

        if isinstance(exc, CommitStateUnknownException):
            return (
                f" The commit's outcome is UNKNOWN, so the data file {data_uri!r} was KEPT on "
                "purpose: the catalog may have applied the commit, and deleting a file a snapshot "
                "references would corrupt the table. Verify the table's latest snapshot, then let "
                "the catalog's own orphan-file policy remove it if the commit did not land."
            )
        referenced = self._snapshot_references(table, data_uri)
        if referenced is True:
            return (
                f" The data file {data_uri!r} was KEPT on purpose: a committed snapshot of "
                f"{'.'.join(self._target.identifier)} already references it (a concurrent Export "
                "of identical content published the same content-addressed path), and deleting it "
                "would break that table."
            )
        if referenced is None:
            return (
                f" The data file {data_uri!r} was KEPT on purpose: this run could not establish "
                f"from {self._target.uri!r} whether a snapshot of "
                f"{'.'.join(self._target.identifier)} references it, and an unprovable deletion of "
                "a full-corpus file is not a cleanup. Check the table's snapshots, then let the "
                "catalog's own orphan-file policy remove it if nothing points at it."
            )
        try:
            storage.remove(data_uri)
        except Exception as cleanup_exc:  # noqa: BLE001 - never mask the failure being reported
            return (
                f" The data file {data_uri!r} this run published is referenced by no snapshot but "
                f"could not be removed ({cleanup_exc}); delete it manually."
            )
        return f" The data file {data_uri!r} this run published was removed (no snapshot uses it)."

    @staticmethod
    def _snapshot_references(table: Any, data_uri: str) -> bool | None:
        """Does ANY snapshot of ``table`` reference ``data_uri``? ``None`` when unknowable.

        The proof :meth:`_discard_unreferenced` needs before it deletes a full-corpus file that a
        concurrent writer may also have published (audit R7-data_contracts_sinks-4). ``exists()``
        cannot answer this — it says the file is there, never who is entitled to it — so the
        question is put to the catalog, which is the only serialization point for what is
        referenced.

        Deliberately over-broad in two directions, because a false "unreferenced" corrupts a table
        while a false "referenced" only leaves a file for the operator's orphan-file policy:

        * EVERY snapshot is walked, not just the current one. ``overwrite`` supersedes data files
          without unlinking them, and Iceberg time travel keeps those older snapshots readable, so
          "the current snapshot does not use it" is not the same as "nothing uses it".
        * Metadata is refreshed first, so a snapshot the concurrent run committed AFTER this run
          loaded the table is seen. A refresh that fails falls back to the metadata already in hand
          rather than giving up — stale evidence of a reference is still evidence.

        Any failure to answer returns ``None`` (keep the file and say so), never an exception:
        this runs on a path that is already reporting a failure, and a second one must not bury it.
        """
        if table is None:
            return None
        candidates = {data_uri, _absolute_uri(data_uri)}
        candidates |= {uri.split("://", 1)[-1] for uri in tuple(candidates)}
        with suppress(Exception):  # stale metadata still answers; never mask the failure
            table = table.refresh()
        try:
            io = table.io
            for snapshot in table.metadata.snapshots:
                for manifest in snapshot.manifests(io):
                    for entry in manifest.fetch_manifest_entry(io, discard_deleted=False):
                        path = str(entry.data_file.file_path)
                        if path in candidates or path.split("://", 1)[-1] in candidates:
                            return True
        except Exception:  # noqa: BLE001 - unknowable is its own answer (the file is then KEPT)
            return None
        return False

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
        write. A schema drift between an existing table and this run is therefore REFUSED here, in
        BOTH directions, by :meth:`_assert_no_drift` — Iceberg's own ``add_files`` validation only
        catches the widening half (see that method).

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
            existing = catalog.load_table(self._target.identifier)
        except NoSuchTableError:
            pass  # Expected on the first run into a fresh catalog: create it below.
        except Exception as exc:  # noqa: BLE001 - normalise catalog failures
            raise self._table_error("loaded", exc) from exc
        else:
            self._assert_no_drift(existing, arrow_schema)
            return existing
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
            raced = catalog.load_table(self._target.identifier)
        except Exception as exc:  # noqa: BLE001 - normalise catalog failures
            raise self._table_error("loaded after a concurrent create", exc) from exc
        # The race winner may have been a DIFFERENTLY-CONFIGURED Export: its table is authoritative
        # and this run must be held to it, exactly like any other pre-existing table.
        self._assert_no_drift(raced, arrow_schema)
        return raced

    def _assert_no_drift(self, table: Any, arrow_schema: pa.Schema) -> None:
        """Refuse a run whose columns differ from the existing table's, in EITHER direction.

        ``add_files`` validates only one half of this. A run carrying an EXTRA column is rejected by
        Iceberg ("PyArrow table contains more columns … use union_by_name"), but a run MISSING a
        column the table declares is silently ACCEPTED and back-filled with NULLs: unwire the
        embedder (a config edit, a second pipeline writing the same table, a delta run wired without
        the signal) and every appended row gets ``embedding = NULL`` while the schema still
        advertises the column — in ``mode: overwrite`` the whole governed table's vectors become
        NULL, with exit 0 and no warning. That is a silent error (ADR-0034) and a table that no
        longer mirrors the consumer contract (#184), so both directions are refused here, BEFORE the
        transaction, leaving the existing table untouched.

        Compared on column NAMES: that is the contract surface the Export controls (which optional
        signal columns it writes), and the only drift ``add_files`` fails to catch. Type-level drift
        stays with Iceberg's own validation, which does catch it.
        """
        table_columns = [field.name for field in table.schema().fields]
        run_columns = list(arrow_schema.names)
        if table_columns == run_columns:
            return
        missing = [name for name in table_columns if name not in run_columns]
        unexpected = [name for name in run_columns if name not in table_columns]
        detail = []
        if missing:
            detail.append(f"this run does not write {', '.join(missing)}")
        if unexpected:
            detail.append(f"this run writes {', '.join(unexpected)}, which the table does not have")
        if not detail:  # Same names, different order — still not the documented column-for-column.
            detail.append(f"column order differs: table {table_columns}, run {run_columns}")
        msg = (
            f"Iceberg table {'.'.join(self._target.identifier)!r} at {self._target.uri!r} has a "
            f"different schema than this Export run writes ({'; '.join(detail)}). Registering it "
            "would silently NULL-fill or reject columns, so the table is left untouched. Configure "
            "this Export with the same signal columns as the table, or target a different table."
        )
        raise ProviderError(msg)

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

    def _data_location(self, table: Any) -> tuple[Storage, str]:
        """The directory an immutable data file for ``table`` belongs in, plus its Storage.

        ``<table.location()>/data`` — the table's OWN location in the catalog's warehouse, which is
        the only path with the lifetime a snapshot reference needs. The previous home,
        ``<out_dir>/iceberg-data/``, sat inside ``_latence/runs/<run_id>/`` and was destroyed
        wholesale by :meth:`~latence_core.delta_runner.DeltaRunner._purge_run_artifacts` on any
        purge (and by ordinary run-retention cleanup), leaving the committed snapshot pointing at a
        deleted absolute path — an unreadable, unrecoverable table. See the module docstring.

        The warehouse may be a different filesystem than the Export's (a local run writing into an
        ``s3://`` warehouse is the normal enterprise shape), so the data file gets its OWN
        :class:`~latence_core.storage.Storage` bound to that location's scheme. A scheme the
        Storage seam refuses surfaces as its own typed ``ConfigError`` naming the scheme; anything
        else becomes a typed :class:`~latence_core.errors.ProviderError` naming the table
        (ADR-0034) — never a raw ``pyiceberg``/fsspec traceback.
        """
        from latence_core.storage import Storage as _Storage

        try:
            location = str(table.location()).rstrip("/")
        except Exception as exc:  # noqa: BLE001 - normalise any catalog/metadata failure
            raise self._table_error("located for its data files", exc) from exc
        if not location:
            msg = (
                f"Iceberg table {'.'.join(self._target.identifier)!r} at {self._target.uri!r} "
                "reports no location, so there is no durable place to publish its data file. "
                "Configure the catalog with a warehouse location."
            )
            raise ProviderError(msg)
        data_dir = f"{location}/data"
        try:
            return _Storage.for_uri(data_dir), data_dir
        except ConfigError:
            # Already a precise, typed scheme refusal (SchemeNotAllowedError) — do not bury it.
            raise
        except Exception as exc:  # noqa: BLE001 - normalise any filesystem-construction failure
            msg = (
                f"Iceberg table {'.'.join(self._target.identifier)!r} has location {location!r}, "
                f"which this Export cannot write its data file to: {exc}"
            )
            raise ProviderError(msg) from exc

    def _publish_data_file(
        self,
        src_storage: Storage,
        dst_storage: Storage,
        parquet_uri: str,
        data_dir: str,
        basename: str,
    ) -> tuple[str, bool]:
        """Stream-copy the deliverable to an immutable, CONTENT-ADDRESSED Iceberg data file.

        Returns the published URI and whether THIS call created it — the second half is what lets
        :meth:`_discard_unreferenced` clean up after a failed commit without ever deleting a file
        an earlier run's snapshot already points at.

        Streamed through the Storage seam in fixed-size chunks, so the Export's O(batch) memory
        bound (ADR-0033) survives the optional output, and published with the same
        temp-then-rename discipline as every other artifact (ADR-0010), so a crash mid-copy never
        leaves a half-written file for the catalog to point at.

        The published name is ``<basename>-<sha256 of the bytes>.parquet``, and the digest is
        folded into the streaming copy — one pass, never a whole-corpus buffer. Content addressing
        (rather than a random per-run token) is what makes this path satisfy BOTH of the
        constraints it sits between:

        * **Iceberg immutability** — the catalog must never reference a path whose bytes later
          change. A path named after the sha256 of its own contents can only ever hold one byte
          sequence, which is strictly stronger than "a fresh random name each run".
        * **The framework's determinism invariant** — the export tree is a deliverable, and
          ``stacks.validate`` proves reproducibility by walking ``export/`` RECURSIVELY and
          comparing two fresh runs key-for-key. A random filename put a hidden ``uuid4`` on the
          export path and made that check report ``export diverged`` on a perfectly deterministic
          pipeline. Equal corpus bytes now yield an equal key.

        An identical data file already present is left ALONE rather than rewritten: its bytes are
        identical by construction, and a file a previous snapshot may already point at must not be
        replaced underneath the catalog. That also bounds repeated runs of identical content, which
        previously accumulated one full-corpus copy per run.

        ``src_storage`` reads the Export's deliverable; ``dst_storage`` writes the table's warehouse
        location. They are two Storages because the two can be different filesystems (see
        :meth:`_data_location`); the copy therefore streams through memory rather than asking either
        store for a server-side copy it may not be able to perform across the boundary.
        """
        digest = sha256()
        data_uri = ""
        created = False

        def _content_addressed_name() -> str | None:
            """The publish target, resolved once the bytes (and therefore the digest) exist.

            ``Storage.atomic_writer`` calls this after the copy has closed: the content is only
            knowable once it has been read, and reading it twice (hash pass + copy pass) would
            double the I/O on a full corpus for nothing. Returning ``None`` publishes nothing — an
            identical data file is already there (its bytes are identical by construction) and a
            file a previous snapshot may point at must not be replaced underneath the catalog.
            """
            nonlocal data_uri, created
            data_uri = f"{data_dir}/{basename}-{digest.hexdigest()}.parquet"
            created = not dst_storage.exists(data_uri)
            return data_uri if created else None

        try:
            with (
                dst_storage.atomic_writer(
                    f"{data_dir}/{basename}.parquet", destination=_content_addressed_name
                ) as dst,
                src_storage.open(parquet_uri, "rb") as src,
            ):
                while chunk := src.read(_COPY_CHUNK_BYTES):
                    digest.update(chunk)
                    dst.write(chunk)
        except Exception as exc:  # noqa: BLE001 - normalise into the typed Provider failure
            raise ProviderError(self._publication_failed(exc)) from exc
        return data_uri, created

    def _publication_failed(self, exc: BaseException) -> str:
        """The typed message for a failed publication, carrying Storage's disposal of the copy.

        The publication's half of :meth:`_discard_unreferenced`, and for the same reason. While the
        copy is in flight the temp path already holds the ENTIRE corpus, chunk ``content``
        included; a transport drop, an ENOSPC or a permission-denied rename in that window used to
        leave it in the customer's warehouse where no snapshot references it, no
        ``expire_snapshots`` can collect it, and only a manual ``remove_orphan_files`` would ever
        reach it — and because each temp name carries a fresh random token, every retry against a
        flaky store dropped ANOTHER full-corpus copy.

        The sweep itself is no longer this module's job: :meth:`Storage.atomic_writer` owns it, so
        the Runner's checkpoints and the Export's deliverable get the same protection this table
        used to have alone. There is no ambiguity to respect (unlike
        ``CommitStateUnknownException``): the catalog has not been told about the path and no
        snapshot can possibly reference a temp name, so deleting it can never corrupt a table.
        Storage's removal is best-effort and never masks the original failure — it records the
        outcome as PEP-678 notes on the propagating exception, which are folded in here so the
        operator reads the disposal (and, when the store refused the delete, the exact path still
        holding the corpus text) on the one typed error (ADR-0034).
        """
        disposal = " ".join(getattr(exc, "__notes__", ()))
        return (
            f"Iceberg data file for {'.'.join(self._target.identifier)} could not be published "
            f"into {self._target.uri!r}: {exc}. Nothing was registered. {disposal}".rstrip()
        )


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
    "PROP_ERASURE",
    "PROP_MODE",
    "PROP_RECORD_VERSIONS",
    "PROP_SOURCE",
    "IcebergCorpusTable",
    "IcebergTarget",
    "build_iceberg_output",
]
