"""Providers a measurement retired (ADR-0075; the owner's rule of 14 Nov: an older module is kept
only where a measurement shows it adds value).

Measured on the text-side gold set of ``benchmark/resolution-modules/`` (684 mentions of 179
organisations; ``results.json``) against the same run on exact surfaces, under a rule fixed
before the run: kept when pairwise F1 rises by at least 0.02 and precision does not fall.

A retired Provider is still registered, so a Pipeline that names it still runs; no shipped
stack and not the demo names it, and ``latence stack check`` says so, as a warning, of a stack
that does.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

RETIRED: dict[str, str] = {
    "disambiguation.cascade": (
        "disambiguation.cascade is retired (ADR-0075): over exact surfaces its fuzzy rungs added "
        "0.019 pairwise F1 at a loss of 0.19 precision on the text-side gold set "
        "(benchmark/resolution-modules); use disambiguation.exact_surface, which takes the same "
        "aliases and KB"
    ),
    "disambiguation.embedding": (
        "disambiguation.embedding without an `embedder` runs on the in-core hashing embedder, "
        "which is retired (ADR-0075): it lost 0.043 pairwise F1 and 0.33 precision against exact "
        "surfaces on the text-side gold set (benchmark/resolution-modules); configure a learned "
        "embedder, or use disambiguation.exact_surface"
    ),
}
"""Each retired use, with the measurement it rests on and what to use instead."""


def retired_use(provider: str, config: Mapping[str, Any] | None) -> str | None:
    """Why this use of a Provider is retired, or ``None`` where it is not. The embedding
    Disambiguator is retired only on its hashing default (no ``embedder`` configured)."""
    if provider == "disambiguation.embedding" and (config or {}).get("embedder"):
        return None
    return RETIRED.get(provider)
