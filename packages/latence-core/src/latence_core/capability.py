"""Capability protocols and the entry-point Provider registry (ADR-0004).

A Capability is a narrow, model-agnostic interface a Stage depends on. A Provider
is a concrete implementation, registered as a plugin under the ``latence.providers``
entry-point group and discovered dynamically via ``importlib.metadata`` — the
framework never names a model, only Capabilities.

For S1 the Capabilities are the three spine Stages (Source, Parser, Export). Each
is a ``typing.Protocol`` so any object with the right shape satisfies it — no
base class or import of core required for a third-party Provider.
"""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import DocumentRecord, ParserInput, Record
    from latence_core.storage import Storage

ENTRY_POINT_GROUP = "latence.providers"


@runtime_checkable
class Source(Protocol):
    """Lists and fetches documents from where they live, stamping Provenance."""

    def produce(self, storage: Storage) -> Iterator[ParserInput]:
        """Yield one raw document per source document, with initial Provenance.

        The yielded :class:`~latence_core.contracts.ParserInput` carries the
        undecoded source ``content`` (bytes) — decoding and the turn into
        markdown belong to the Parser (ADR-0019), not the Source.
        """
        ...


@runtime_checkable
class Parser(Protocol):
    """Turns raw source documents into parsed markdown records."""

    def parse(self, inputs: Iterable[ParserInput]) -> Iterator[DocumentRecord]:
        """Consume raw :class:`ParserInput`s and yield parsed markdown records."""
        ...


@runtime_checkable
class Export(Protocol):
    """Materializes AI-ready outputs (JSONL + Parquet in S1)."""

    def export(self, records: Iterable[Record], storage: Storage, out_dir: str) -> list[str]:
        """Write records to ``out_dir`` on ``storage``; return the output URIs."""
        ...


class ProviderRegistry:
    """Discovers Providers from ``latence.providers`` entry points.

    Discovery is lazy and cached: entry points are enumerated once, then each is
    loaded (imported) only when its name is first requested. This keeps the
    registry cheap even when many heavy Provider packages are installed.
    """

    def __init__(self, group: str = ENTRY_POINT_GROUP) -> None:
        self._group = group
        self._entry_points: dict[str, metadata.EntryPoint] | None = None
        self._loaded: dict[str, type] = {}

    def _discover(self) -> dict[str, metadata.EntryPoint]:
        if self._entry_points is None:
            eps = metadata.entry_points(group=self._group)
            self._entry_points = {ep.name: ep for ep in eps}
        return self._entry_points

    def names(self) -> list[str]:
        return sorted(self._discover())

    def load(self, name: str) -> type:
        """Load and return the Provider class registered under ``name``."""
        if name in self._loaded:
            return self._loaded[name]
        eps = self._discover()
        if name not in eps:
            available = ", ".join(self.names()) or "<none>"
            msg = (
                f"No Provider registered as {name!r} in group "
                f"{self._group!r}. Available: {available}"
            )
            raise KeyError(msg)
        obj = eps[name].load()
        if not isinstance(obj, type):
            msg = f"Provider {name!r} did not resolve to a class (got {type(obj)!r})."
            raise TypeError(msg)
        self._loaded[name] = obj
        return obj
