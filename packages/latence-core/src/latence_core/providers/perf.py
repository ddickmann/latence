"""The shared **performance seam** every learned Provider loads its model through (W8-perf-accel).

Enterprise throughput has one cross-cutting shape repeated across every learned Provider: pick a
load **dtype** (bf16 on a GPU, fp32 on CPU — the fp32-on-GPU default is a ~2× miss), pick an
**attention implementation** (``sdpa`` by default, ``flash_attention_2`` only when the
``flash_attn`` package is importable and the operator permits it, with a graceful ``sdpa`` fallback
when a FA2 load raises), and optionally **compile** the forward (``torch.compile`` behind a guard so
a compile/warmup failure logs and runs the uncompiled model, never crashes). W8 factors that into
one dep-light module so every adapter applies the SAME knobs, resolved **once**, honestly — the P8
autotuner turns them per-device from here.

Dep-light like the rest of ``latence-core`` (ADR-0016): this module resolves the *decisions* as
plain strings **without importing torch** (:func:`resolve_dtype_name`, :func:`resolve_attn_impl`),
so a Provider registry that only imports the module never pulls in torch. The single place that
must materialise a real ``torch.dtype`` (:func:`load_perf_kwargs`, whose dict is spread into a
native ``from_pretrained``) imports torch **lazily and defensively**: on a torch-free host it
degrades to the dtype *name string* (which HF ``from_pretrained`` also accepts as ``torch_dtype``),
so the module imports and the offline perf-seam tests run without torch installed.

None of these knobs changes an adapter's *contract* — same inputs, same output records/spans/labels.
bf16 vs fp32 shifts float bits slightly, but for the extract/embed/OCR tasks here the emitted
text/labels/spans are unchanged (asserted identical on the pod; the profile carries the note).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from latence_core.providers.device import cuda_available

if TYPE_CHECKING:
    from collections.abc import Mapping

_log = logging.getLogger("latence.perf")

# The two load dtypes W8 resolves between. bf16 on a GPU (the throughput default — fp32 on a GPU is
# the ~2× miss W8 closes); fp32 on CPU (bf16 CPU kernels are not universally faster, can regress).
# Names, not torch objects, so the *decision* is torch-free (ADR-0016); :func:`load_perf_kwargs`
# materialises the real ``torch.dtype`` (or keeps the name string on a torch-free host).
DTYPE_BF16 = "bfloat16"
DTYPE_FP32 = "float32"

# The attention implementations. ``sdpa`` is the safe, always-available default (PyTorch's fused
# scaled-dot-product attention); ``flash_attention_2`` is opt-in and only when the ``flash_attn``
# package is importable — a FA2 request on a host without it degrades to ``sdpa`` (never a crash).
ATTN_SDPA = "sdpa"
ATTN_FA2 = "flash_attention_2"

# The valid explicit dtype config values (an operator may pin one). ``auto`` (the default) resolves
# by device.
_VALID_DTYPE = frozenset({"auto", DTYPE_BF16, DTYPE_FP32})


def _effective_is_cuda(device: str | None) -> bool:
    """Whether the model will run on CUDA, for the dtype default.

    An explicit resolved device (the AdapterBase device seam hands ``"cpu"``/``"cuda"``) wins; a
    provider that does not resolve a device itself (the GLiNER family lets the library auto-place)
    passes ``None`` and we fall back to :func:`~latence_core.providers.device.cuda_available` — the
    same override-driven (LATENCE_CUDA), torch-free CUDA probe the device seam uses, so the dtype
    default matches where the model will actually land.
    """
    if device is not None:
        return device == "cuda"
    return cuda_available()


def resolve_dtype_name(device: str | None, config: Mapping[str, Any]) -> str:
    """Resolve the load dtype **name** (``"bfloat16"``/``"float32"``) — torch-free (ADR-0016).

    ``config["dtype"]`` may pin ``"bfloat16"``/``"float32"`` explicitly (an operator override), or
    ``"auto"`` (the default) to resolve by device: **bf16 on CUDA, fp32 on CPU**. An unrecognised
    value is a hard error the caller surfaces as a ConfigError — never a silent wrong dtype.

    Returns the *name* so the decision stays torch-free; :func:`load_perf_kwargs` turns it into a
    real ``torch.dtype`` where torch is present (and keeps the string where it is not — HF
    ``from_pretrained`` accepts the name string for ``torch_dtype`` too).
    """
    raw = config.get("dtype", "auto")
    name = str(raw).lower()
    if name not in _VALID_DTYPE:
        valid = ", ".join(sorted(_VALID_DTYPE))
        msg = f"invalid dtype {raw!r}: must be one of {valid}"
        raise ValueError(msg)
    if name in (DTYPE_BF16, DTYPE_FP32):
        return name
    # auto: bf16 on GPU, fp32 on CPU.
    return DTYPE_BF16 if _effective_is_cuda(device) else DTYPE_FP32


def flash_attn_available() -> bool:
    """Whether the optional ``flash_attn`` package is importable (the FA2 gate) — no import of it.

    Uses :func:`importlib.util.find_spec` so the probe never actually imports ``flash_attn`` (which
    would pull CUDA extensions on a host that has the wheel but no GPU) — it only asks whether the
    module *could* be imported. FA2 is selected only when this is ``True`` (and the operator permits
    it); otherwise the load stays on ``sdpa``.
    """
    import importlib.util  # noqa: PLC0415 - stdlib, kept local so import cost is only paid on use

    try:
        return importlib.util.find_spec("flash_attn") is not None
    except (ImportError, ValueError):  # pragma: no cover - a broken/partial install
        # ``find_spec`` can raise for a partially-installed/namespace-broken package; treat any
        # such breakage as "not available" so a FA2 probe never crashes the load.
        return False


def resolve_attn_impl(config: Mapping[str, Any]) -> str:
    """Resolve the attention implementation (``"sdpa"`` default; ``"flash_attention_2"`` opt-in).

    ``config["attn_implementation"]`` controls it:

    * unset / ``"sdpa"`` → ``"sdpa"`` (the always-available default).
    * ``"flash_attention_2"`` (or the ``config["flash_attention"]`` boolean shorthand) → FA2, **only
      when** :func:`flash_attn_available` — otherwise it degrades to ``"sdpa"`` (a FA2 request on a
      host without the package is honoured as best-effort, never a crash). A logged note records the
      downgrade so an operator who asked for FA2 and did not get it can see why.
    * any other value → ``"sdpa"`` (unknown impl → the safe default; a typo never fails the load).

    The actual FA2 **load** can still raise on a model that does not support it; the caller wraps
    ``from_pretrained`` so that surfaces as the graceful ``sdpa`` retry (see the OCR/ST loads).
    """
    raw = config.get("attn_implementation")
    if raw is not None:
        wants_fa2 = str(raw).lower() == ATTN_FA2
    else:
        wants_fa2 = bool(config.get("flash_attention", False))
    if not wants_fa2:
        return ATTN_SDPA
    if flash_attn_available():
        return ATTN_FA2
    _log.info("flash_attention_2 requested but 'flash_attn' is not importable; using sdpa")
    return ATTN_SDPA


def _torch_dtype(name: str) -> Any:
    """Map a dtype name to a real ``torch.dtype``, or the name string on a torch-free host.

    Imports torch **lazily and defensively** (ADR-0016): where torch is present the returned dict
    carries the real ``torch.dtype`` object from_pretrained expects; where it is not (the offline
    perf-seam unit env), the *name string* is returned unchanged — HF from_pretrained accepts the
    string for ``torch_dtype`` too, so the seam still resolves a valid kwarg and the module imports.
    """
    try:
        import torch  # noqa: PLC0415 - optional heavy dep, imported only when materialising a dtype
    except ImportError:  # pragma: no cover - exercised only on a torch-free host
        return name
    return getattr(torch, name)


def load_perf_kwargs(device: str | None, config: Mapping[str, Any]) -> dict[str, Any]:
    """The perf kwargs to spread into a learned model's ``from_pretrained`` (dtype + attn).

    Resolves the load **dtype** (bf16 on GPU / fp32 on CPU by default, or the ``config["dtype"]``
    override) and the **attention implementation** (``sdpa`` default, ``flash_attention_2`` opt-in
    when importable) and returns them as the two kwargs HF ``from_pretrained`` reads:

        {"torch_dtype": <torch.dtype|name>, "attn_implementation": "sdpa"|"flash_attention_2"}

    ``torch_dtype`` carries a real ``torch.dtype`` where torch is importable, else the dtype name
    string (both accepted by from_pretrained) so the seam is usable on a torch-free host. This is
    the single dict every learned Provider spreads at load so NONE loads bare fp32 (the regression
    guard greps for exactly this).
    """
    dtype_name = resolve_dtype_name(device, config)
    return {
        "torch_dtype": _torch_dtype(dtype_name),
        "attn_implementation": resolve_attn_impl(config),
    }


def load_with_fa2_fallback(
    loader: Any,
    model_id: str,
    perf: Mapping[str, Any],
) -> Any:
    """Call ``loader(model_id, **perf)``, retrying once on ``sdpa`` if a FA2 load raises.

    The single graceful-degradation shape every learned model load reuses (W8): a
    ``flash_attention_2`` load can raise on a backbone/arch that has no FA2 kernel; that is caught
    and retried once with ``attn_implementation="sdpa"`` so the model still loads — never a crash.
    A non-FA2 load error propagates unchanged (the adapter's ``guard`` maps it to the typed
    taxonomy). ``perf`` is the :func:`load_perf_kwargs` dict (torch_dtype + attn_impl).
    """
    kwargs = dict(perf)
    try:
        return loader(model_id, **kwargs)
    except Exception:  # noqa: BLE001 - deliberate: retry a FA2 load on sdpa before surfacing
        if kwargs.get("attn_implementation") != ATTN_FA2:
            raise
        _log.info("flash_attention_2 load failed for %r; retrying with sdpa", model_id)
        kwargs["attn_implementation"] = ATTN_SDPA
        return loader(model_id, **kwargs)


def load_then_cast(
    loader: Any,
    model_id: str,
    config: Mapping[str, Any],
    device: str | None,
) -> Any:
    """Load a model **plain**, then ``model.to(dtype)`` — for libs that reject perf kwargs (W8).

    The pod-validated load path for the plain ``gliner`` library (``gliner`` 0.2.27): its
    ``GLiNER.from_pretrained`` **rejects** ``torch_dtype`` AND ``attn_implementation``
    (``TypeError: UniEncoderSpanModel.__init__() got an unexpected keyword argument 'torch_dtype'``)
    and the encoder does NOT support ``attn_implementation`` / FlashDeBERTa. So instead of spreading
    the perf kwargs into ``from_pretrained`` (which crashes on real weights), this loads the model
    ``maybe_compile`` still applies afterwards (guarded), independently of this cast.
    a no-op on CPU / when the resolved dtype is fp32.

    The cast is applied only when the resolved dtype is a real, non-fp32 ``torch.dtype`` **and** the
    model is on cuda — bf16 on CPU is not a universal win, and fp32 is the identity. If torch is not
    importable (the offline env) the resolved dtype is a name string, not a ``torch.dtype``, so the
    cast is skipped (there is nothing to cast to) and the bare model is returned — the seam stays
    torch-free-to-import. ``maybe_compile`` still applies afterwards (guarded), independently.
    """
    model = loader(model_id)
    dtype_name = resolve_dtype_name(device, config)
    if dtype_name == DTYPE_FP32 or not _effective_is_cuda(device):
        return model
    dtype = _torch_dtype(dtype_name)
    # On a torch-free host ``_torch_dtype`` returns the NAME string (not a torch.dtype); there is no
    # real dtype to cast to, so skip the cast rather than call ``.to("bfloat16")``.
    if isinstance(dtype, str):
        return model
    return model.to(dtype)


def maybe_compile(model: Any, config: Mapping[str, Any], *, provider_name: str = "provider") -> Any:
    """``torch.compile`` the model when ``config["compile"]`` is truthy — guarded, never crashes.

    Off by default (``config["compile"]`` falsy → the model is returned unchanged, a pure no-op) so
    the warmup cost is only paid where the autotuner has decided compile wins. When on, wraps the
    forward with ``torch.compile(model, mode="reduce-overhead")`` — but behind a broad guard: ANY
    failure (torch missing, an unsupported model, a compile/warmup error) is **logged** and the
    **uncompiled** model returned, so a bad compile can only ever cost throughput, never correctness
    or a crash. The compiled module is a drop-in for the eager one (same forward signature/outputs),
    so an adapter swaps it in transparently.
    """
    if not config.get("compile", False):
        return model
    try:
        import torch  # noqa: PLC0415 - optional heavy dep, only imported when compile is requested

        return torch.compile(model, mode="reduce-overhead")
    except Exception:  # noqa: BLE001 - deliberate: a compile failure must degrade to eager, never crash
        _log.warning(
            "%s: torch.compile failed; running the uncompiled model (throughput only)",
            provider_name,
            exc_info=True,
        )
        return model
