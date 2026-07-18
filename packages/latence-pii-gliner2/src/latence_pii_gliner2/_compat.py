"""The ``gliner2`` version/API compatibility gate for ``redaction.gliner2`` (#192, ADR-0016/0034).

The Provider drives ONE narrow slice of the ``gliner2`` framework:

.. code-block:: python

    model = GLiNER2.from_pretrained(model_id, quantize=…, compile=…)   # W9 perf knobs
    model.batch_extract_entities(texts, entity_types, threshold=…,
                                 include_spans=True, include_confidence=True, max_len=…)

That slice does **not** exist across the whole ``gliner2`` 1.x line. Verified against the
released distributions (sdists/wheels from pypi.org/project/gliner2, inspected 2026-07-18):

===========  =========================================================================
version      the slice this Provider needs
===========  =========================================================================
1.0.x        ``batch_extract_entities`` **does not exist** → a native ``AttributeError``
1.1.x        no ``include_spans`` / ``max_len`` kwargs → ``TypeError`` (and, without
             spans, no char offsets at all → nothing to redact)
1.2.0–1.2.5  no ``max_len`` kwarg → ``TypeError``
1.2.6+       ``max_len`` + ``include_spans`` present; ``from_pretrained`` accepts the
             native ``quantize``/``compile`` kwargs
1.3.x        the pod-validated line (1.3.2) — identical API surface to 1.2.6
===========  =========================================================================

The **package metadata pin** (``gliner2[local]>=1.3,<2`` in ``pyproject.toml``, ADR-0016: the
per-Provider package owns its pin) is the primary gate — an unsupported version then fails at
dependency RESOLUTION, before any code runs. This module is the second gate, for the environment
the resolver never saw (a pod with a hand-``pip install``ed ``gliner2``, the exact case that
produced the field failure): the installed version is checked at model load and an unsupported one
becomes a **typed** :class:`~latence_core.errors.ProviderError` naming the supported range and the
install command (ADR-0034) — never a raw native ``AttributeError``/``TypeError`` an operator cannot
act on.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from latence_core.errors import ProviderError

_log = logging.getLogger("latence.provider.gliner2_pii")

#: Lowest ``gliner2`` release whose ``batch_extract_entities`` accepts ``include_spans`` +
#: ``max_len`` and whose ``from_pretrained`` accepts the native ``quantize``/``compile`` kwargs.
#: 1.3 (not 1.2.6) is the floor: 1.3.x is the line actually pod-validated for this Provider.
MIN_GLINER2 = (1, 3)

#: Exclusive upper bound — a 2.x may break the API this Provider drives (ADR-0016 major cap).
MAX_GLINER2_EXCLUSIVE = (2,)

#: The human-readable specifier, kept byte-identical to the ``pyproject.toml`` pin so the error
#: message and the package metadata can never drift apart.
SUPPORTED_SPEC = ">=1.3,<2"

_VERSION_RE = re.compile(r"^\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def _parse(version: str) -> tuple[int, ...] | None:
    """The leading numeric ``(major, minor, micro)`` of a version string, or ``None``.

    Tolerant of the suffixes a real distribution carries (``1.3.2.post1``, ``1.4.0rc1``,
    ``1.3.2+local``): only the leading numeric release segment decides support, which is exactly
    what the API table above is keyed on.
    """
    match = _VERSION_RE.match(version)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def gliner2_version(module: Any) -> str | None:
    """The installed ``gliner2`` version string, or ``None`` if it cannot be determined.

    Reads the package's own ``__version__`` (present on every released ``gliner2`` 1.x) and falls
    back to the installed distribution metadata, so it works for a source checkout too.
    """
    version = getattr(module, "__version__", None)
    if isinstance(version, str) and version:
        return version
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as dist_version
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib on 3.11+
        return None
    try:
        return dist_version("gliner2")
    except PackageNotFoundError:
        return None


def require_supported_gliner2(module: Any, *, provider_name: str) -> None:
    """Raise a typed :class:`ProviderError` if the installed ``gliner2`` is out of range.

    Called BEFORE the model load, so an unsupported install fails fast with an actionable message
    instead of downloading weights and then dying inside the framework on a native fault. A version
    that cannot be determined is **logged and allowed**: the metadata pin already gated resolution,
    and the API-surface check below still catches an actually-missing method as a typed error.
    """
    version = gliner2_version(module)
    if version is None:
        _log.warning(
            "%s: could not determine the installed gliner2 version; this Provider supports "
            "gliner2 %s",
            provider_name,
            SUPPORTED_SPEC,
        )
        return
    parsed = _parse(version)
    if parsed is not None and parsed >= MIN_GLINER2 and parsed[:1] < MAX_GLINER2_EXCLUSIVE:
        return
    detail = (
        "its release version could not be parsed"
        if parsed is None
        else "it is outside the supported range"
    )
    msg = (
        f"{provider_name} requires gliner2 {SUPPORTED_SPEC} — the release line whose "
        "GLiNER2.batch_extract_entities accepts 'include_spans'/'max_len' and whose "
        f"from_pretrained accepts the native 'quantize'/'compile' kwargs. The installed gliner2 is "
        f"{version} ({detail}); gliner2 < 1.3 either has no 'batch_extract_entities' at all "
        "(1.0.x) or rejects those kwargs (1.1.x/1.2.0-1.2.5), which is not a usable redaction "
        f"path. Install a supported version: pip install 'gliner2[local]{SUPPORTED_SPEC}'."
    )
    raise ProviderError(msg)


def require_batched_entity_api(model: Any, *, provider_name: str) -> None:
    """Raise a typed :class:`ProviderError` if the loaded model has no batched-entity API.

    The backstop for anything the version check cannot see (a fork, a vendored checkpoint wrapper,
    an API-client object rather than a local model): the missing method surfaces as a typed,
    actionable framework error rather than the raw ``AttributeError`` an operator got in the field.
    """
    if callable(getattr(model, "batch_extract_entities", None)):
        return
    msg = (
        f"{provider_name} batches via model.batch_extract_entities(texts, entity_types, "
        "threshold=…, include_spans=True, include_confidence=True, max_len=…), but the loaded "
        "GLiNER2 object exposes no callable 'batch_extract_entities'. That method was added in "
        f"gliner2 1.1 and reached its current signature in 1.2.6; this Provider supports gliner2 "
        f"{SUPPORTED_SPEC}. Install a supported version (pip install "
        f"'gliner2[local]{SUPPORTED_SPEC}') or point 'model' at a checkpoint the GLiNER2 framework "
        "can load. See scripts/validate-encoder-batching-on-gpu.md Gate 0."
    )
    raise ProviderError(msg)
