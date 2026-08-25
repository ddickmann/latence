"""The ``gliner2`` 2.x version/API compatibility gate for the GLiNER 2.5 Providers (ADR-0064).

The second gate, mirroring :mod:`latence_pii_gliner2._compat`. The **package metadata pin**
(``gliner2[local]>=2.0,<3`` in ``pyproject.toml``, ADR-0016: the per-Provider package owns its
pin) is the PRIMARY gate — an unsupported version fails at dependency RESOLUTION, before any code
runs. This module is the gate for the environment the resolver never saw: a pod with a
hand-``pip install``ed ``gliner2``, which is exactly the case that produced the #192 field failure
on the 1.x redactor. An unsupported install becomes a **typed**
:class:`~latence_core.errors.ProviderError` (ADR-0034) naming the supported range and the install
command — never the raw native ``AttributeError``/``TypeError`` an operator cannot act on.

What the 2.5 Providers actually drive, and where each fact was read (the unpacked ``gliner2``
2.0.0 wheel, inspected 2026-08-25):

.. code-block:: python

    from gliner2 import AutoExtractor                 # gliner2/__init__.py::_LAZY
    model = AutoExtractor.from_pretrained(model_id, quantize=…, compile=…)
    #   gliner2/auto.py::AutoExtractor.from_pretrained — dispatches on the checkpoint's saved
    #   `architecture` (gliner2/configuration.py::architecture_from_config); `quantize`/`compile`
    #   are accepted, they are members of gliner2/models/loading.py::MODEL_LOAD_OPTIONS.
    model.batch_extract_entities(texts, entity_types, batch_size=8, threshold=…,
                                 format_results=True, include_confidence=True,
                                 include_spans=True, max_len=…, overlap_policy=None)
    model.batch_extract_relations(texts, relation_types, …)   # same signature
    #   gliner2/inference/runtime.py::ExtractorRuntimeMixin — BOTH public model classes
    #   (SpanExtractor and BoundaryExtractor) mix it in, so the surface is architecture-independent.

**Why the major boundary is a hard one, not a cautious cap.** ``AutoExtractor.from_pretrained``
raises ``ArchitectureMismatchError`` rather than converting between heads:

    "Span and boundary heads are not checkpoint-compatible, and automatic architecture conversion
    is not supported."  — ``gliner2/auto.py``

The 2.5 checkpoints are ``boundary``; every 1.x checkpoint is ``span`` (a checkpoint with no
``architecture`` field loads as ``"span"``). So a 1.x-era checkpoint id pointed at these Providers
is a loud, typed failure, and the legacy ``GLiNER2`` class — still exported in 2.x, and still
span-only by its own docstring — is deliberately NOT the loader here.

===========  =========================================================================
version      the slice these Providers need
===========  =========================================================================
< 2.0        no ``gliner2.auto`` / ``AutoExtractor`` at all, and no ``boundary``
             architecture → a native ``ImportError``/``AttributeError``, and no
             checkpoint in the 2.5 family can be loaded
2.0.x        the line this package is written against: ``AutoExtractor`` +
             ``BoundaryExtractor`` + the batched ``batch_extract_entities`` /
             ``batch_extract_relations`` with ``include_spans``/``max_len``
>= 3         a major bump may break the surface above (the 1.x → 2.x bump did)
===========  =========================================================================
"""

from __future__ import annotations

import logging
import re
from typing import Any

from latence_core.errors import ProviderError

_log = logging.getLogger("latence.provider.gliner25")

#: Lowest ``gliner2`` release exposing ``AutoExtractor`` and the ``boundary`` architecture the
#: GLiNER 2.5 checkpoints are. 2.0 is the floor because ``gliner2.auto`` does not exist below it.
MIN_GLINER2 = (2, 0)

#: Exclusive upper bound — a 3.x may break the API these Providers drive (ADR-0016 major cap).
MAX_GLINER2_EXCLUSIVE = (3,)

#: The human-readable specifier, kept byte-identical to the ``pyproject.toml`` pin so the error
#: message and the package metadata can never drift apart. Pinned by a test.
SUPPORTED_SPEC = ">=2.0,<3"

#: The checkpoint architecture the GLiNER 2.5 family is built on
#: (``gliner2/models/boundary/engine.py::BoundaryExtractor.architecture``). Named here because it
#: is the thing that makes a 1.x checkpoint un-loadable by these Providers, and vice versa.
BOUNDARY_ARCHITECTURE = "boundary"

#: The batched entry points both Providers drive, on
#: ``gliner2.inference.runtime.ExtractorRuntimeMixin``.
BATCHED_METHODS: tuple[str, ...] = ("batch_extract_entities", "batch_extract_relations")

_VERSION_RE = re.compile(r"^\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def _parse(version: str) -> tuple[int, ...] | None:
    """The leading numeric ``(major, minor, micro)`` of a version string, or ``None``.

    Tolerant of the suffixes a real distribution carries (``2.0.0.post1``, ``2.1.0rc1``,
    ``2.0.0+local``): only the leading numeric release segment decides support, which is exactly
    what the API table above is keyed on.
    """
    match = _VERSION_RE.match(version)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def gliner2_version(module: Any) -> str | None:
    """The installed ``gliner2`` version string, or ``None`` if it cannot be determined.

    Reads the package's own ``__version__`` (``gliner2/__init__.py`` sets it eagerly, above the
    lazy-attribute table, so reading it imports no torch) and falls back to the installed
    distribution metadata so a source checkout still answers.
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

    Called BEFORE the model load — i.e. before any weight download — so an unsupported install
    fails fast with an actionable message instead of fetching a checkpoint and then dying inside
    the framework on a native fault. A version that cannot be determined is **logged and
    allowed**: the metadata pin already gated resolution, and the API-surface checks below still
    catch an actually-missing class or method as a typed error.
    """
    version = gliner2_version(module)
    if version is None:
        _log.warning(
            "%s: could not determine the installed gliner2 version; these Providers support "
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
        f"{provider_name} requires gliner2 {SUPPORTED_SPEC} — the release line that ships "
        "gliner2.auto.AutoExtractor and the 'boundary' architecture the GLiNER 2.5 checkpoints "
        f"are. The installed gliner2 is {version} ({detail}); gliner2 < 2.0 has no "
        "AutoExtractor and cannot load a boundary checkpoint at all, and the span checkpoints it "
        "does load are a different Provider package (latence-extract-gliner2 / "
        f"latence-pii-gliner2, pinned gliner2>=1.3,<2). Install a supported version: pip install "
        f"'gliner2[local]{SUPPORTED_SPEC}'."
    )
    raise ProviderError(msg)


def require_auto_extractor(module: Any, *, provider_name: str) -> Any:
    """Return ``gliner2.AutoExtractor``, or a typed :class:`ProviderError` if it is absent.

    ``AutoExtractor`` — not the legacy ``GLiNER2`` class — is the loader, because it is the one
    that dispatches on the checkpoint's saved architecture. ``GLiNER2.from_pretrained`` "remains
    span-only" by the library's own docstring, so pointing it at a 2.5 (boundary) checkpoint
    cannot work. Accessing the attribute triggers ``gliner2``'s lazy import of
    ``gliner2.auto``, which is why this runs inside the Provider's backbone-import factory (i.e.
    after the shared load seam has neutralized the TorchScript JIT and capped the intra-op
    threads, #190) rather than at module import.
    """
    auto = getattr(module, "AutoExtractor", None)
    if auto is not None and callable(getattr(auto, "from_pretrained", None)):
        return auto
    msg = (
        f"{provider_name} loads its checkpoint through gliner2.AutoExtractor.from_pretrained "
        "(the architecture-dispatching loader — the legacy GLiNER2 class is span-only and cannot "
        "load a GLiNER 2.5 boundary checkpoint), but the installed gliner2 exposes no such "
        f"callable. These Providers support gliner2 {SUPPORTED_SPEC}; install a supported "
        f"version: pip install 'gliner2[local]{SUPPORTED_SPEC}'."
    )
    raise ProviderError(msg)


def require_boundary_model(model: Any, *, provider_name: str, model_id: str) -> None:
    """Raise a typed :class:`ProviderError` unless the loaded model is a ``boundary`` extractor.

    ``AutoExtractor`` dispatches on the checkpoint, so pointing ``model:`` at a 1.x-era ``span``
    checkpoint loads **successfully** and then behaves like the other package's Provider — same
    method names, different head, different quality, and none of this package's tuning. That is a
    silent wrong-model run, which is worse than a crash, so it is refused here by reading the
    class's own ``architecture`` marker
    (``gliner2/inference/engine.py`` sets ``architecture = "span"`` /
    ``gliner2/models/boundary/engine.py`` sets ``architecture = "boundary"``).

    A model exposing NO ``architecture`` attribute at all is **allowed with a warning**: the
    attribute is a class-level marker rather than a documented contract, and refusing an
    unrecognised wrapper would turn a forward-compatible library change into an outage. The
    batched-API check is the hard gate; this one is the wrong-checkpoint gate.
    """
    architecture = getattr(model, "architecture", None)
    if architecture is None:
        _log.warning(
            "%s: the loaded gliner2 model exposes no 'architecture' marker, so the "
            "boundary-checkpoint check could not run; verify that %r is a GLiNER 2.5 checkpoint",
            provider_name,
            model_id,
        )
        return
    if str(architecture) == BOUNDARY_ARCHITECTURE:
        return
    msg = (
        f"{provider_name}: checkpoint {model_id!r} loaded as the "
        f"{str(architecture)!r} architecture, but this Provider is the GLiNER 2.5 "
        f"({BOUNDARY_ARCHITECTURE}) family. A 'span' checkpoint is the gliner2 1.x line: run it "
        "through 'fused_entity_relation.gliner2' / 'redaction.gliner2' (package "
        "latence-extract-gliner2 / latence-pii-gliner2), which pin gliner2>=1.3,<2. Point "
        "'model' at a GLiNER 2.5 checkpoint (e.g. fastino/gliner2.5-multi-v1) instead."
    )
    raise ProviderError(msg)


def require_batched_api(model: Any, *, provider_name: str, methods: tuple[str, ...]) -> None:
    """Raise a typed :class:`ProviderError` if the loaded model lacks the batched entry points.

    The backstop for anything the version check cannot see (a fork, a vendored wrapper, or a
    ``from_api`` CLOUD client rather than a local model): a missing method surfaces as a typed,
    actionable framework error at LOAD rather than the raw ``AttributeError`` an operator got in
    the field on the 1.x redactor (#192). ``methods`` is the caller's slice of
    :data:`BATCHED_METHODS` — the redactor drives entities only, the fused extractor drives both,
    and a Provider must not fail on a method it never calls.
    """
    missing = [name for name in methods if not callable(getattr(model, name, None))]
    if not missing:
        return
    msg = (
        f"{provider_name} batches via {' + '.join(missing)}(texts, types, threshold=…, "
        "include_spans=True, include_confidence=True, max_len=…), but the loaded gliner2 model "
        f"exposes no callable {missing!r}. Those live on "
        "gliner2.inference.runtime.ExtractorRuntimeMixin, which both public model classes mix in; "
        f"these Providers support gliner2 {SUPPORTED_SPEC}. Install a supported version (pip "
        f"install 'gliner2[local]{SUPPORTED_SPEC}') or point 'model' at a checkpoint the "
        "AutoExtractor can load locally."
    )
    raise ProviderError(msg)
