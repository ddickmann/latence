"""The shared **performance seam** every learned Provider loads its model through (W8-perf-accel).

**Start here: the four entry points.** A learned Provider does not assemble this module's steps —
it calls exactly ONE function, keyed by the native library its adapter owns:

* :func:`load_gliner_model` — the plain ``gliner`` library (``entity.gliner``,
  ``redaction.gliner_pii``, ``relation.gliner_relex``);
* :func:`load_gliner2_model` — the ``gliner2`` framework (``fused_entity_relation.gliner2``,
  ``redaction.gliner2``);
* :func:`load_transformers_model` — plain ``transformers`` models (the OCR VLMs);
* :func:`load_sentence_transformer` — ``sentence-transformers`` checkpoints
  (``embedding.sentence_transformers``).

**Process-global torch state has an owner**: :class:`TorchRuntime`. Three of this module's steps
mutate state that outlives any one Provider — the ``PYTORCH_JIT`` env switch + the in-process JIT
state (:func:`neutralize_torch_jit`), the torch intra-op thread pool
(:func:`cap_intra_op_threads`), and the TorchDynamo compilation state (:func:`maybe_compile`).
Those three functions are thin delegators to the CURRENT runtime
(:func:`current_torch_runtime`), which RECORDS the prior value of everything it changes; a test
boundary or a Stage boundary wraps work in :func:`scoped_torch_runtime` to have the mutations made
inside it restored on exit. Unscoped callers keep the exact process-lifetime semantics they always
had (the root runtime records but is never restored).

Each takes ``(import_model_cls, model_id, config, decision, *, provider_name)`` and returns a model
that is ready to run. The **ordering** of JIT neutralization, backbone import, device check,
dtype/quantize, load, placement + verification, window bound, thread cap and compile lives inside
them — a caller cannot get it wrong, because a caller never sees it. The backbone import in
particular is handed in as a factory precisely so "neutralize before import" (#190) is a straight
line of code here rather than a rule enforced by linting the callers' source. See the C4 section at
the bottom of this module for the full reasoning.

Everything between here and that section is a STEP of those sequences. The steps stay public
because the P8 autotuner and the OOM/compile-fault predicates are used on their own — but a
Provider that reaches for them individually to build a load is re-opening #190/#191 by hand.

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

import contextlib
import logging
from typing import TYPE_CHECKING, Any, TypeVar, cast

from latence_core.errors import ConfigError, LatenceError, ProviderError
from latence_core.providers.device import cuda_available

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
    from typing import TypeAlias

    from latence_core.providers.device import DeviceDecision

    #: A caller-supplied typed-error policy for the native load: ``guard(op)`` returns a context
    #: manager that maps whatever the block raises into the ADR-0034 taxonomy.
    #: :meth:`~latence_core.providers.adapter.AdapterBase.guard` satisfies it exactly. It is a
    #: PARAMETER rather than an import because the dependency runs the other way (``adapter``
    #: imports ``perf``), and because a Provider that is not an ``AdapterBase`` can pass ``None``
    #: and take the default mapping.
    LoadGuard: TypeAlias = Callable[[str], contextlib.AbstractContextManager[Any]]

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


def place_on_device(
    model: Any, device: str | None, *, provider_name: str = "provider"
) -> str | None:
    """Move a plain-``gliner`` model onto the **resolved** device at load, and PROVE it landed.

    The plain ``gliner`` library does NOT auto-place: ``GLiNER.from_pretrained`` loads onto CPU
    (``map_location`` defaults there), so a Provider that never calls ``.to(device)`` runs on CPU no
    matter what the device seam resolved — the exact silent performance cliff #191 measured on the
    pod (``device: cuda`` configured, ~11.5 s/chunk, the GPU idle, and **no warning**). ADR-0036's
    contract is honour-the-resolved-device-or-fail-loudly, so this is deliberately NOT best-effort:

    * ``device is None`` (the seam skipped, or a Provider that resolves no device) → no-op; the
      resident device is read back and returned for logging only.
    * otherwise ``model.to(device)`` — a placement fault propagates as a
      :class:`~latence_core.errors.ProviderError` naming the provider + device (never swallowed).
    * the resident device is then read back from the model's parameters and **verified**: a model
      that reports a different device than the one requested is a
      :class:`~latence_core.errors.ProviderError`, not a logged shrug. Silence is the bug (#191).
      A model whose device cannot be read at all (a test double, an exotic wrapper) is accepted —
      there is nothing to contradict — but the placement call still had to succeed.

    Returns the resident device string (or ``None`` when unreadable). Torch is never imported here:
    the move and the read-back go through the model object itself (ADR-0016).
    """
    if device is not None:
        try:
            model.to(device)
        except Exception as exc:
            msg = f"{provider_name}: could not place the model on device {device!r}: {exc}"
            raise ProviderError(msg) from exc
    resident = _detect_model_device(model)
    if device is None or resident is None:
        return resident
    # ``cuda`` matches ``cuda:0`` (torch names the ordinal); compare on the device TYPE.
    if resident.split(":", 1)[0] != device.split(":", 1)[0]:
        msg = (
            f"{provider_name}: model was placed on {device!r} but reports device {resident!r} — "
            "refusing to run on the wrong device (a silent CPU fallback is a ~10x throughput "
            "cliff, ADR-0036)"
        )
        raise ProviderError(msg)
    _log.info("%s: model resident on %s (resolved device %s)", provider_name, resident, device)
    return resident


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

    W9-jit: the TorchScript JIT is neutralized (:func:`neutralize_torch_jit`) before the load —
    **defence-in-depth only**. By the time a gliner Provider reaches this loader it has already
    evaluated ``from gliner import GLiNER``, which scripts transformers' deberta_v2
    relative-position helpers at module-import time, and nothing here can un-script them. The
    Provider's own call *ahead of that import* is the fix (ADR-0036 W9-jit); this one covers a
    caller that arrives with the backbone not yet imported. Throughput-only, byte-identical output.
    **Capability-gated (T-DEV-1):** ``config`` is threaded through, so on a known-good architecture
    this is a no-op and the JIT stays enabled.
    """
    neutralize_torch_jit(config=config)
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


# The class names + module prefixes that mark a torch.compile / TorchDynamo / Inductor backend
# failure. Detected **torch-free** by name (never importing torch) so the autotuner's correctness
# gate can recognise the W9 lesson — an inference-time ``BackendCompilerFailed`` / dynamo error —
# on a host without torch installed. The real lesson (gliner2-multi-v1 + torch 2.4.1) is that a
# ``compile: true`` setting can load fine and only blow up at the FIRST forward pass; the gate must
# reject those, so this walks the exception's ``__cause__``/``__context__`` chain and matches any
# link by class name or the module the exception type is defined in.
_COMPILE_FAULT_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "BackendCompilerFailed",
        "InductorError",
        "CompilationError",
        "Unsupported",  # torch._dynamo.exc.Unsupported
        "TorchRuntimeError",
        "BackendError",
    }
)
# Module prefixes of the torch compile/dynamo/inductor stack. A fault whose exception type is
# DEFINED in one of these modules is a compile-backend failure even if its class name is generic.
_COMPILE_FAULT_MODULE_PREFIXES: tuple[str, ...] = (
    "torch._dynamo",
    "torch._inductor",
    "torch.fx",
    "torch._functorch",
)


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """The exception + its ``__cause__``/``__context__`` chain (dedup, cycle-safe).

    A compile-backend failure often surfaces WRAPPED (a provider ``guard`` maps a native fault to a
    typed ``ProviderError`` with the real dynamo error as ``__cause__``); the gate must see through
    that, so it inspects every link of the chain, not just the outermost type.
    """
    seen: set[int] = set()
    chain: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    return chain


def is_compile_backend_failure(exc: BaseException) -> bool:
    """Whether ``exc`` (or anything in its cause chain) is a torch.compile / dynamo backend fault.

    **Torch-free** — matches by exception class name (:data:`_COMPILE_FAULT_CLASS_NAMES`) or by the
    module the exception type is defined in (:data:`_COMPILE_FAULT_MODULE_PREFIXES`), never
    importing
    torch. This is the single "is this a compile-backend failure" reasoning the W8 seam and the P8
    autotuner's correctness gate share: the whole point of the gate is that a box which cannot
    compile (an inference-time ``BackendCompilerFailed`` on the first forward — the W9 lesson)
    simply
    never gets ``compile: true``. A fault that is NOT a compile failure (a real bug, an OOM that is
    not compile-related) is still a rejection by the gate — but recognising the compile class lets
    the honest report name it precisely ("compile: rejected (BackendCompilerFailed)").
    """
    for link in _exception_chain(exc):
        cls = type(link)
        if cls.__name__ in _COMPILE_FAULT_CLASS_NAMES:
            return True
        module = getattr(cls, "__module__", "") or ""
        if any(module.startswith(prefix) for prefix in _COMPILE_FAULT_MODULE_PREFIXES):
            return True
        # A dynamo error sometimes only names itself in its message (e.g. a bare RuntimeError whose
        # text is "BackendCompilerFailed: ..."); match that conservatively so the gate is not fooled
        # by a fault the backend raised as a generic type with a telltale message.
        text = str(link)
        if "BackendCompilerFailed" in text or "torch._dynamo" in text or "torch._inductor" in text:
            return True
    return False


# Class names / message fragments of a CUDA (or accelerator) out-of-memory fault. ``torch`` raises
# ``torch.cuda.OutOfMemoryError`` (a ``RuntimeError`` subclass) whose message is ``"CUDA out of
# memory. Tried to allocate ..."``; older torch and some backends raise a bare ``RuntimeError`` with
# the same telltale text; the CUDA driver's own string is ``CUDA_ERROR_OUT_OF_MEMORY``. Matched
# torch-free (by name / message), exactly like the compile-fault gate, so ``latence-core`` never
# imports torch to reason about it.
_OOM_FAULT_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "OutOfMemoryError",  # torch.cuda.OutOfMemoryError / torch.OutOfMemoryError
        "OutOfMemoryError_",
    }
)
_OOM_MESSAGE_FRAGMENTS: tuple[str, ...] = (
    "CUDA out of memory",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "HIP out of memory",  # ROCm
    "out of memory",  # generic torch OOM RuntimeError (lower-cased match below)
)


def is_cuda_oom(exc: BaseException) -> bool:
    """Whether ``exc`` (or anything in its cause chain) is a CUDA/accelerator out-of-memory fault.

    **Torch-free** — matches by exception class name (:data:`_OOM_FAULT_CLASS_NAMES`) or by a
    telltale message fragment (:data:`_OOM_MESSAGE_FRAGMENTS`), never importing torch, and walks the
    ``__cause__``/``__context__`` chain so it sees through a provider ``guard`` that wrapped the
    native ``OutOfMemoryError`` in a typed ``ProviderError`` (the OOM is the ``__cause__``).

    This is the single "is this an OOM" predicate the learned Providers' micro-batch backoff
    shares: an OOM on a large forward is *recoverable* by splitting the batch and retrying smaller
    (the output is batch-size-invariant), so the Provider halves and retries rather than crashing. A
    fault that is NOT an OOM (a real bug, a compile failure) is not caught by the backoff and
    surfaces as before.
    """
    for link in _exception_chain(exc):
        cls = type(link)
        if cls.__name__ in _OOM_FAULT_CLASS_NAMES:
            return True
        text = str(link).lower()
        if any(frag.lower() in text for frag in _OOM_MESSAGE_FRAGMENTS):
            return True
    return False


def empty_cuda_cache(*, provider_name: str = "provider") -> None:
    """Best-effort reclaim of a just-finished model's memory — GC first, then the CUDA cache.

    Two distinct, independently-useful steps, in this order:

    1. ``gc.collect()`` — **run unconditionally, torch or not.** A torch ``nn.Module`` is typically
       held alive by a reference cycle (hooks, parameter back-refs), so dropping the last Python
       reference to a Provider does NOT free its model by refcount — the several-GiB CUDA storages
       stay allocated until a cycle collection runs. Python's own GC is triggered by *Python*
       object-count thresholds, which the tiny Provider wrapper never trips (its weight is off-heap
       in CUDA), so between two model Stages the collector may never fire on its own. An explicit
       ``gc.collect()`` is therefore what actually reclaims the previous Stage's model — the
       cross-stage VRAM-lifecycle fix (dogfood-2b-process). It runs even on a torch-free host both
       because the collection is the load-bearing step and so this seam's effect is observable in an
       offline test.
    2. ``torch.cuda.empty_cache()`` — hands the now-freed cached/fragmented blocks back to the
       driver so the NEXT Stage's model (or a smaller OOM-retry batch) allocates from a contiguous
       pool. Torch-free-safe (a host without torch, or without CUDA, silently skips this step).

    Fully guarded — reclaiming memory is an optimisation, never a correctness step, so any failure
    is logged at DEBUG and swallowed; it never raises.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - cache clearing is best-effort, never fatal
        _log.debug("%s: CUDA cache clear skipped (%s)", provider_name, type(exc).__name__)


def inference_context() -> contextlib.AbstractContextManager[Any]:
    """The context every learned Provider's model forward runs INSIDE — ``torch.inference_mode()``.

    The GLiNER2 encoders here only ever run **forward** (extract/redact) — there is no backward
    pass, ever. But a plain forward under the default (grad-enabled) mode still builds an autograd
    graph and pins every intermediate activation alive for a backward that never comes. Across a
    long ``latence process`` run those never-freed activation tensors ACCUMULATE — pod-measured at
    27.28 GiB resident on a model that is only ~2-3 GB — until the next allocation CUDA-OOMs at the
    redact stage (dogfood-2b-process). ``torch.cuda.empty_cache()`` cannot reclaim any of it: the
    memory is held *live* by the graph, not merely cached. ``torch.inference_mode()`` is the cure
    at the source — it disables autograd tracking for the block, so no graph is built and each
    forward's activations free the moment they fall out of scope. Resident memory then stays at
    model size plus one batch's transient activation instead of climbing without bound.

    This does NOT change any output — ``inference_mode`` only suppresses grad bookkeeping; the
    emitted spans/labels/records are byte-identical (the redaction floor, ADR-0044, is untouched).

    **Torch-free-safe** (ADR-0016), mirroring :func:`load_perf_kwargs`/:func:`empty_cuda_cache`:
    torch is imported lazily and defensively. On a host without torch (the offline CPU test env) it
    returns :func:`contextlib.nullcontext` so the ``with`` block still runs — the wrap is a no-op
    there rather than an import error. The single place a real ``torch.inference_mode()`` is
    materialised imports torch only when this is called, so a Provider registry that merely imports
    this module never pulls in torch.
    """
    try:
        import torch  # noqa: PLC0415 - optional heavy dep, imported only when entering inference
    except Exception:  # noqa: BLE001 - torch-free host (offline test env): degrade to a no-op context
        return contextlib.nullcontext()
    return cast("contextlib.AbstractContextManager[Any]", torch.inference_mode())


# ---------------------------------------------------------------------------
# TorchRuntime — the OWNER of the process-global torch state this seam mutates.
# ---------------------------------------------------------------------------
#
# Three steps in this module reach past the model object into state that belongs to the whole
# process: ``neutralize_torch_jit`` (the ``PYTORCH_JIT`` env switch + ``torch.jit._state``),
# ``cap_intra_op_threads`` (``torch.set_num_threads``), and ``maybe_compile`` (TorchDynamo's
# compilation caches/guards). Before Wave 3-H those mutations had no owner and no reset — the exact
# fragility FINDINGS F7 (an order-dependent ``prims.iota`` flake in a 5,000-test single-interpreter
# run) and F9 (an interpreter-shutdown segfault with torch loaded) are symptoms of. The runtime does
# not change WHAT the three functions do; it records the prior value of every global they touch so a
# boundary that wants isolation (a real-weights test, a Stage) can get it back.


class TorchRuntime:
    """Records the process-global torch state the perf seam mutates, and can restore it.

    The three global-mutating steps (:func:`neutralize_torch_jit`, :func:`cap_intra_op_threads`,
    :func:`maybe_compile`) delegate to the *current* runtime (:func:`current_torch_runtime`). Each
    mutation records the FIRST prior value it displaces within this runtime's lifetime:

    * ``"jit_env"`` — the prior ``PYTORCH_JIT`` value (``None`` = the variable was unset);
    * ``"jit_enabled"`` — the prior ``torch.jit._state._enabled.enabled`` bool;
    * ``"num_threads"`` — the prior ``torch.get_num_threads()`` int;
    * ``"dynamo"`` — ``True`` when ``torch.compile`` was invoked (dynamo state has no readable
      "prior"; the restore is ``torch._dynamo.reset()``, dropping the compilation caches/guards
      back to a fresh state).

    :meth:`restore` puts every recorded global back — guarded (never raises, torch-free-safe) and
    idempotent (the records are consumed, so a second call is a no-op). A runtime is NOT
    thread-safe; the perf seam runs on the single-process Runner (ADR-0033).
    """

    def __init__(self) -> None:
        self._changes: dict[str, Any] = {}

    @property
    def recorded(self) -> dict[str, Any]:
        """A snapshot of what this runtime changed: key → the prior value it would restore."""
        return dict(self._changes)

    def _record(self, key: str, prior: Any) -> None:
        """Record the FIRST prior value per key — later mutations of the same lever keep it."""
        self._changes.setdefault(key, prior)

    # -- the three owned mutations (the module functions are thin delegators) --

    def maybe_compile(
        self, model: Any, config: Mapping[str, Any], *, provider_name: str = "provider"
    ) -> Any:
        """The :func:`maybe_compile` body — records that dynamo state was touched."""
        if not config.get("compile", False):
            return model
        try:
            import torch  # noqa: PLC0415 - optional heavy dep, only imported when compile is on

            # Recorded BEFORE the call: even a failed compile may have touched dynamo state, and
            # the restore (a dynamo reset) is safe either way. Pure flag set — cannot raise.
            self._record("dynamo", True)
            return torch.compile(model, mode="reduce-overhead")
        except Exception:  # noqa: BLE001 - deliberate: a compile failure degrades to eager, never crashes
            _log.warning(
                "%s: torch.compile failed; running the uncompiled model (throughput only)",
                provider_name,
                exc_info=True,
            )
            return model

    def neutralize_torch_jit(
        self, *, provider_name: str = "provider", config: Mapping[str, Any] | None = None
    ) -> bool:
        """The :func:`neutralize_torch_jit` body — records the env switch + in-process JIT state."""
        import os  # noqa: PLC0415 - stdlib, kept local so import cost is only paid on use

        if config is not None and not torch_jit_must_be_neutralized(
            config, provider_name=provider_name
        ):
            return False

        self._record("jit_env", os.environ.get(TORCH_JIT_ENV))
        os.environ[TORCH_JIT_ENV] = "0"
        try:
            import torch  # noqa: PLC0415 - optional heavy dep; the model load imports it next

            state = torch.jit._state  # noqa: SLF001 - the only switch torch exposes for this
            enabled_proxy = getattr(state, "_enabled", None)
            prior = getattr(enabled_proxy, "enabled", None) if enabled_proxy is not None else None
            if isinstance(prior, bool):
                self._record("jit_enabled", prior)
            disable = getattr(state, "disable", None)
            if callable(disable):
                disable()
            if enabled_proxy is not None and getattr(enabled_proxy, "enabled", False):
                # A torch whose ``disable()`` is missing/renamed: set the flag the proxy reads.
                enabled_proxy.enabled = False
            off = enabled_proxy is not None and not getattr(enabled_proxy, "enabled", False)
        except Exception:  # noqa: BLE001 - deliberate: neutralizing the JIT must never crash a load
            _log.info(
                "%s: could not disable the torch JIT in-process (PYTORCH_JIT=0 set for any later "
                "torch import)",
                provider_name,
                exc_info=True,
            )
            return False
        _log.info(
            "%s: torch JIT neutralized (PYTORCH_JIT=0%s) — nvrtc-free mdeberta forward, "
            "throughput only",
            provider_name,
            ", runtime state disabled" if off else "",
        )
        return off

    def cap_intra_op_threads(self, n: int | None, *, provider_name: str = "provider") -> int | None:
        """The :func:`cap_intra_op_threads` body — records the prior torch thread-pool size."""
        if n is None:
            return None
        try:
            import torch  # noqa: PLC0415 - optional heavy dep, imported only when a cap is applied

            prior: Any = None
            with contextlib.suppress(Exception):
                # Best-effort prior read (a stub torch may not expose it); the cap still applies.
                prior = int(torch.get_num_threads())
            torch.set_num_threads(n)
        except Exception:  # noqa: BLE001 - deliberate: a thread-cap failure must never crash the load
            _log.warning(
                "%s: could not cap intra-op threads to %d (left at torch default)",
                provider_name,
                n,
                exc_info=True,
            )
            return None
        if isinstance(prior, int):
            self._record("num_threads", prior)
        _log.info(
            "%s: capped torch intra-op threads to %d (many-core oversubscription guard)",
            provider_name,
            n,
        )
        return n

    # -- restore --------------------------------------------------------------

    def restore(self) -> None:
        """Put every recorded global back to its prior value — guarded, idempotent, never raises.

        The records are consumed (a second call is a no-op). Every step is individually suppressed:
        restoring is insurance, never a correctness step, and it must work on a host where torch was
        monkeypatched away between the mutation and the restore (the offline test envs do exactly
        that). ``"jit_enabled": False`` needs no undo — the seam only ever disables, so a prior-off
        JIT was off before and stays off.
        """
        import os  # noqa: PLC0415 - stdlib, kept local so import cost is only paid on use
        import sys  # noqa: PLC0415 - stdlib, kept local so import cost is only paid on use

        changes, self._changes = self._changes, {}
        if changes.pop("dynamo", False) and "torch" in sys.modules:
            with contextlib.suppress(Exception):
                import torch._dynamo  # noqa: PLC0415 - only reached when compile ran, so torch exists

                torch._dynamo.reset()
        if "num_threads" in changes:
            with contextlib.suppress(Exception):
                import torch  # noqa: PLC0415 - only reached when a cap was applied through torch

                torch.set_num_threads(changes["num_threads"])
        if changes.get("jit_enabled") is True:
            with contextlib.suppress(Exception):
                import torch  # noqa: PLC0415 - only reached when the JIT was disabled through torch

                state = torch.jit._state  # noqa: SLF001 - the only switch torch exposes for this
                enable = getattr(state, "enable", None)
                if callable(enable):
                    enable()
                proxy = getattr(state, "_enabled", None)
                if proxy is not None and not getattr(proxy, "enabled", False):
                    proxy.enabled = True
        if "jit_env" in changes:
            prior_env = changes["jit_env"]
            if prior_env is None:
                os.environ.pop(TORCH_JIT_ENV, None)
            else:
                os.environ[TORCH_JIT_ENV] = str(prior_env)


#: The runtime stack. The bottom entry is the process-lifetime ROOT runtime: it records what the
#: unscoped process changed (diagnostics) but is never restored, so callers that never scope keep
#: the exact semantics the three functions always had. :func:`scoped_torch_runtime` pushes a fresh
#: runtime so mutations made inside the scope land in — and are restored by — that runtime.
_RUNTIME_STACK: list[TorchRuntime] = [TorchRuntime()]


def current_torch_runtime() -> TorchRuntime:
    """The runtime that owns torch-global mutations right now (innermost scope, else the root)."""
    return _RUNTIME_STACK[-1]


@contextlib.contextmanager
def scoped_torch_runtime() -> Iterator[TorchRuntime]:
    """Scope the process-global torch mutations made inside the ``with`` block; restore on exit.

    The boundary FINDINGS F7/F9 asked for: a real-weights test (or a Stage runner) wraps the work
    that loads/compiles a model, and on exit the ``PYTORCH_JIT`` env switch, the in-process JIT
    state, the intra-op thread cap and the dynamo compilation state go back to what they were when
    the scope was entered — so one boundary's torch state cannot leak into the next. Scopes nest:
    each records and restores only its own mutations. The yielded runtime exposes
    :attr:`TorchRuntime.recorded` for assertions/diagnostics.
    """
    runtime = TorchRuntime()
    _RUNTIME_STACK.append(runtime)
    try:
        yield runtime
    finally:
        with contextlib.suppress(ValueError):
            _RUNTIME_STACK.remove(runtime)
        runtime.restore()


def maybe_compile(model: Any, config: Mapping[str, Any], *, provider_name: str = "provider") -> Any:
    """``torch.compile`` the model when ``config["compile"]`` is truthy — guarded, never crashes.

    Off by default (``config["compile"]`` falsy → the model is returned unchanged, a pure no-op) so
    the warmup cost is only paid where the autotuner has decided compile wins. When on, wraps the
    forward with ``torch.compile(model, mode="reduce-overhead")`` — but behind a broad guard: ANY
    failure (torch missing, an unsupported model, a compile/warmup error) is **logged** and the
    **uncompiled** model returned, so a bad compile can only ever cost throughput, never correctness
    or a crash. The compiled module is a drop-in for the eager one (same forward signature/outputs),
    so an adapter swaps it in transparently.

    Thin delegator: the mutation is performed — and the touched dynamo state RECORDED — by the
    current :class:`TorchRuntime`, so a :func:`scoped_torch_runtime` boundary can reset it.
    """
    return current_torch_runtime().maybe_compile(model, config, provider_name=provider_name)


# ---------------------------------------------------------------------------
# Torch **JIT** neutralization (W9-jit) — the second half of the compile guard.
# ---------------------------------------------------------------------------

#: The environment switch torch reads to decide whether its **TorchScript JIT** is enabled at all.
#: ``PYTORCH_JIT=0`` makes ``torch.jit.script`` / ``torch.jit.trace`` a **pass-through** (the
#: decorated function is returned unchanged, run as ordinary eager Python) instead of building a
#: ScriptFunction whose fused kernels are compiled at runtime by nvrtc.
TORCH_JIT_ENV = "PYTORCH_JIT"

#: The env override that PINS the detected compute capability (``"8.6"``, ``"12.0"``). Same posture
#: as ``LATENCE_CUDA`` in the device seam: honoured first so a CI host, a container without
#: ``nvidia-smi``, or an operator who knows better is deterministic; an unparseable value is ignored
#: (falls through to the probe) rather than guessed.
LATENCE_COMPUTE_CAPABILITY_ENV = "LATENCE_COMPUTE_CAPABILITY"

#: The highest compute capability positively known to have a **healthy nvrtc JIT path**, i.e. the
#: ceiling of the "leave the TorchScript JIT alone" set. Hopper (9.0) and everything below it —
#: Volta 7.0, Turing 7.5, Ampere 8.0/8.6, Ada 8.9 — shipped against CUDA 11/12 toolchains whose
#: ``libnvrtc-builtins`` is present, and the FM-NVRTC-JIT fault has never been observed on them.
#:
#: Blackwell starts at **10.0** (B100/B200/GB200) and the RTX 50-series / RTX PRO parts report
#: **12.0** — the box the workaround was written on. 10.0 sits on the KILL side deliberately even
#: though the observed fault was 12.0: it is the same CUDA-13-era generation, we have never run one,
#: and the error is asymmetric (a wrong skip is a first-forward native crash; a wrong kill is
#: throughput). Widen this ceiling only against a measurement, never against a hunch.
JIT_SAFE_MAX_CAPABILITY = (9, 0)

#: The valid ``torch_jit`` operator-override spellings. ``auto`` (the default) is the capability
#: gate; a truthy value pins the JIT **on** (skip the kill); a falsy value pins the kill.
_VALID_TORCH_JIT = frozenset({"auto", "on", "off", "true", "false", "1", "0", "yes", "no"})
_TORCH_JIT_ON = frozenset({"on", "true", "1", "yes"})


def _parse_capability(text: str) -> tuple[int, int] | None:
    """Parse a ``"<major>.<minor>"`` capability string, or ``None`` when it is not one.

    Deliberately strict: anything that is not two non-negative integers separated by a dot — an old
    ``nvidia-smi``'s ``"Field 'compute_cap' invalid"``, an empty line, ``"N/A"`` — is *unknown*, and
    unknown means the JIT gets killed. A lenient parse here would manufacture a false "known-good"
    and turn a slow run into a crashing one.
    """
    major, _, minor = text.strip().partition(".")
    if not major.isdigit() or not minor.isdigit():
        return None
    parsed = (int(major), int(minor))
    if parsed[0] < 1:
        return None  # 0.x is not a real capability; treat it as unreadable
    return parsed


def _capability_from_imported_torch() -> tuple[int, int] | None:
    """Read the capability from a torch that is **already imported** — never importing it.

    ``sys.modules`` only: importing torch here would defeat the whole mechanism, because torch
    parses ``PYTORCH_JIT`` once **at import** and the env lever must be set before that happens.
    When another library has already pulled torch in, this is the most accurate answer available;
    when it has not, the caller falls through to the torch-free ``nvidia-smi`` probe.
    """
    import sys  # noqa: PLC0415 - stdlib, kept local so import cost is only paid on use

    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        cuda = torch.cuda
        raw = cuda.get_device_capability()
        major, minor = int(raw[0]), int(raw[1])
    except Exception:  # noqa: BLE001 - a torch that cannot answer is UNKNOWN, never "safe"
        return None
    return _parse_capability(f"{major}.{minor}")


def _capability_from_nvidia_smi() -> tuple[int, int] | None:
    """Read the capability from ``nvidia-smi`` — torch-free (ADR-0016), the primary probe.

    ``nvidia-smi --query-gpu=compute_cap --format=csv,noheader`` prints one line per GPU. The
    **lowest** wins: the TorchScript switch is process-wide, so a mixed host must not be declared
    known-good on the strength of its best card. Every failure mode — tool absent, non-zero exit,
    timeout, an old driver that does not know the ``compute_cap`` field — returns ``None``, i.e.
    unknown, i.e. the kill stays on.
    """
    import shutil  # noqa: PLC0415 - stdlib, kept local so import cost is only paid on use
    import subprocess  # noqa: PLC0415, S404 - a read-only, fixed-argv driver query

    if shutil.which("nvidia-smi") is None:
        return None
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:  # noqa: BLE001 - a probe that cannot run is unknown, never a crash
        return None
    if proc.returncode != 0:
        return None
    parsed = [_parse_capability(line) for line in (proc.stdout or "").splitlines() if line.strip()]
    readable = [c for c in parsed if c is not None]
    if not readable or len(readable) != len(parsed):
        # An unparseable line means at least one GPU is unaccounted for — unknown, not "the rest".
        return None
    return min(readable)


def detect_compute_capability() -> tuple[int, int] | None:
    """The host's CUDA compute capability as ``(major, minor)``, or ``None`` when unknown.

    Probe order, and every step of it torch-free:

    1. ``LATENCE_COMPUTE_CAPABILITY`` — the deterministic override (CI, a container without the
       driver CLI, an operator who knows the box).
    2. A torch that is **already in** ``sys.modules`` — the most accurate reading, and free.
    3. ``nvidia-smi --query-gpu=compute_cap`` — the same torch-free signal class
       :func:`~latence_core.providers.device.cuda_available` already probes with.
    4. ``None``. Unknown is a first-class answer here, not a fallback to a guess: every caller
       treats it as "assume the worst".

    **This function must never import torch.** ``PYTORCH_JIT`` is read by torch at *torch import*,
    so a probe that imported torch to answer would spend the lever it is deciding whether to pull.
    """
    import os  # noqa: PLC0415 - stdlib, kept local so import cost is only paid on use

    override = os.environ.get(LATENCE_COMPUTE_CAPABILITY_ENV)
    if override:
        pinned = _parse_capability(override)
        if pinned is not None:
            return pinned
    from_torch = _capability_from_imported_torch()
    if from_torch is not None:
        return from_torch
    return _capability_from_nvidia_smi()


def resolve_torch_jit_policy(config: Mapping[str, Any]) -> str:
    """Resolve the ``torch_jit`` operator override to ``"auto"`` / ``"on"`` / ``"off"``.

    ``auto`` (unset, the default) hands the decision to :func:`detect_compute_capability`. ``on``
    keeps the TorchScript JIT — the escape hatch for an operator who has a working nvrtc on a box
    the gate would otherwise be conservative about. ``off`` pins the neutralization regardless of
    what the capability says. A bool is accepted (``True`` → ``on``), as are the usual string
    spellings; anything else is a hard :class:`ValueError` the caller surfaces as a ConfigError —
    a typo must never resolve to a silently different policy on a box where the wrong answer is a
    native crash.
    """
    raw = config.get("torch_jit", "auto")
    if isinstance(raw, bool):
        return "on" if raw else "off"
    text = str(raw).strip().lower()
    if text not in _VALID_TORCH_JIT:
        valid = ", ".join(sorted(_VALID_TORCH_JIT))
        msg = f"invalid torch_jit {raw!r}: must be a bool or one of {valid}"
        raise ValueError(msg)
    if text == "auto":
        return "auto"
    return "on" if text in _TORCH_JIT_ON else "off"


def torch_jit_must_be_neutralized(
    config: Mapping[str, Any], *, provider_name: str = "provider"
) -> bool:
    """Whether this host needs the TorchScript JIT killed — **fail safe** (T-DEV-1).

    The whole decision, in one readable place, because the cost of the two errors is not
    symmetric. ``neutralize_torch_jit`` exists for FM-NVRTC-JIT alone: a missing
    ``libnvrtc-builtins.so.13.0`` on Blackwell / CUDA-13. That fault surfaces on the **first
    forward**, so no load-time ``try/except`` can catch it — a wrong "skip" is a native **crash**,
    while a wrong "kill" is merely **slow** (throughput-only, byte-identical output). So:

    * ``torch_jit: off``/``on`` — the operator's explicit call wins outright.
    * capability **unknown** (no torch, no ``nvidia-smi``, an unreadable answer) → **kill**.
    * capability **> (9, 0)** (:data:`JIT_SAFE_MAX_CAPABILITY`) → **kill**.
    * a positively-read capability at or below the ceiling → **skip**, and log the fact so the
      pod's own log is the record that the JIT ran enabled.

    Measured stake (DECISIONS.md §D2): the kill fires on every GLiNER/GLiNER2/GLinker load, i.e. on
    ``extract``, ``redact`` and ``disambiguate`` — the three stages that were **1,604 s of the prior
    run's 1,743 s**. On sm_86 that cost bought nothing.
    """
    policy = resolve_torch_jit_policy(config)
    if policy != "auto":
        _log.info(
            "%s: torch JIT policy pinned to %r by config (capability gate bypassed)",
            provider_name,
            policy,
        )
        return policy == "off"
    capability = detect_compute_capability()
    if capability is None:
        _log.info(
            "%s: compute capability unknown (no torch, no nvidia-smi, or an unreadable answer) — "
            "neutralizing the torch JIT to fail SAFE (a wrong skip is a first-forward nvrtc crash; "
            "a wrong kill is throughput only). Pin %s to take it back.",
            provider_name,
            LATENCE_COMPUTE_CAPABILITY_ENV,
        )
        return True
    if capability > JIT_SAFE_MAX_CAPABILITY:
        _log.info(
            "%s: compute capability %d.%d is above the nvrtc-safe ceiling %d.%d — neutralizing the "
            "torch JIT (FM-NVRTC-JIT: libnvrtc-builtins.so.13.0 on Blackwell/CUDA-13)",
            provider_name,
            capability[0],
            capability[1],
            JIT_SAFE_MAX_CAPABILITY[0],
            JIT_SAFE_MAX_CAPABILITY[1],
        )
        return True
    _log.info(
        "%s: compute capability %d.%d is a known-good nvrtc architecture — leaving the TorchScript "
        "JIT ENABLED (the FM-NVRTC-JIT workaround is a Blackwell/CUDA-13 tax, not an Ampere one)",
        provider_name,
        capability[0],
        capability[1],
    )
    return False


def neutralize_torch_jit(
    *, provider_name: str = "provider", config: Mapping[str, Any] | None = None
) -> bool:
    """Turn the TorchScript JIT OFF for this process — **capability-gated**, never crashes (W9-jit).

    **T-DEV-1 — read this first.** Handed a Provider ``config`` this is *conditional*: it consults
    :func:`torch_jit_must_be_neutralized` and does nothing at all on a positively-identified
    known-good architecture (Hopper and below — Ampere sm_86 is the target pod). The kill exists
    only for FM-NVRTC-JIT on Blackwell/CUDA-13, and it fires on every GLiNER/GLiNER2/GLinker load,
    i.e. on ``extract``/``redact``/``disambiguate`` — the three stages that were 1,604 s of the
    prior run's 1,743 s. Running it unconditionally was an Ampere tax (DECISIONS.md §D2).

    Called **bare** (``config=None``) it is the unconditional applier it has always been — the shape
    the gate itself delegates to, and the shape a caller with no config in hand takes. The gate is
    therefore opt-in at the CALL SITE, which is also what lets this keep one name: the AST ordering
    guard in ``test_perf_fp32_regression_guard.py`` matches ``neutralize_torch_jit`` lexically to
    prove the backbone import cannot precede it.

    The **gap** the W9 compile guard did not cover. :func:`maybe_compile` /
    :func:`load_gliner2_optimized` neutralize ``torch.compile`` (dynamo/inductor); they do NOT touch
    the *other* runtime compiler in torch — the **TorchScript JIT**. The mdeberta-v3 encoder every
    GLiNER-family Provider is built on scripts its relative-position helper with
    ``@torch.jit.script`` (``build_relative_position`` in transformers' DeBERTa-v2 modeling module),
    and on a **Blackwell / CUDA-13** box executing that ScriptFunction dies in native code:

        RuntimeError: ... failed to open libnvrtc-builtins.so.13.0

    — the JIT fuser asking nvrtc to compile kernels against a builtins library that is not there for
    that CUDA/torch pairing. Exactly the W9 lesson repeated: the fault surfaces on the **first
    forward**, not at load, so no load-time try/except can see it. The proven field workaround is
    the env switch ``PYTORCH_JIT=0``; this function is that workaround **encoded in the seam**, so
    it is not operator folklore.

    Two levers, because either one alone has a hole:

    * ``os.environ["PYTORCH_JIT"] = "0"`` — read by torch **at import**, so a torch not yet
      imported (the normal case: the Provider's model load is what pulls torch in) comes up with
      the JIT already off, and every later ``@torch.jit.script`` at module-import time is a no-op.
    * ``torch.jit._state.disable()`` — the same switch at **runtime**, for a torch that is ALREADY
      imported (another library got there first), so scripting is neutralized from here on. Applied
      before the model load, i.e. before transformers lazily imports its DeBERTa modeling module and
      evaluates the ``@torch.jit.script`` decorator.

    **Throughput-only degradation, byte-identical output** (the ADR-0036 compile-guard discipline):
    an un-scripted function is the *same* Python function the JIT would have compiled — same math,
    same values, only without the fused-kernel speedup. Nothing about the emitted entities /
    relations / offsets changes.

    Torch-free-safe and never raises: on a host without torch (the offline test env) the env switch
    is still set — so a torch imported later is still JIT-free — and the runtime lever is a logged
    no-op. Returns whether the **in-process** JIT state is now known-off (``False`` when torch is
    absent or exposes no JIT state to read back; the env switch is set either way). A **gated call
    that decides to skip** also returns ``False`` — the in-process JIT is then known-ON, and neither
    lever is pulled (an operator's externally-exported ``PYTORCH_JIT`` is left exactly as it was).

    Thin delegator: the mutation is performed — and the prior env value + in-process JIT state
    RECORDED — by the current :class:`TorchRuntime`, so a :func:`scoped_torch_runtime` boundary can
    put both levers back.
    """
    return current_torch_runtime().neutralize_torch_jit(
        provider_name=provider_name, config=config
    )


# The valid explicit gliner2 quantize/compile config values (an operator may pin one). ``auto``
# (the default) resolves by device: True on cuda, False on cpu.
_VALID_GLINER2_KNOB = frozenset({"auto", "true", "false", "1", "0", "yes", "no"})
_TRUTHY_GLINER2_KNOB = frozenset({"true", "1", "yes"})


def resolve_gliner2_knob(name: str, device: str | None, config: Mapping[str, Any]) -> bool:
    """Resolve a gliner2 ``quantize``/``compile`` config knob to a bool — torch-free (W9).

    ``config[name]`` may pin the knob (a bool, or a string ``"true"``/``"false"``/…), or be unset /
    ``"auto"`` (the default) to resolve by device: **True on cuda, False on cpu**. Unlike the plain
    ``gliner`` library, the ``gliner2`` framework's ``from_pretrained`` accepts native ``quantize``
    (fp16) and ``compile`` (``torch.compile``, fused GPU kernels) kwargs — pod-measured 2.83× (PII)
    at byte-identical output — so W9 threads them through this dep-light resolver (no torch import;
    reuses the same override-driven CUDA probe the dtype default uses). Both knobs are a **GPU** win
    and off on cpu by default, so ``auto`` on a CPU-only host resolves to ``False`` (never a wasted
    fp16/compile attempt off-GPU). An unrecognised value is a hard error the caller surfaces as a
    ConfigError — never a silent wrong knob.
    """
    raw = config.get(name, "auto")
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text not in _VALID_GLINER2_KNOB:
        valid = ", ".join(sorted(_VALID_GLINER2_KNOB))
        msg = f"invalid {name} {raw!r}: must be a bool or one of {valid}"
        raise ValueError(msg)
    if text == "auto":
        return _effective_is_cuda(device)
    return text in _TRUTHY_GLINER2_KNOB


def load_gliner2_optimized(
    loader: Any,
    model_id: str,
    device: str | None,
    config: Mapping[str, Any],
    *,
    provider_name: str = "provider",
) -> Any:
    """Load a ``gliner2`` model with native ``quantize``/``compile`` — guarded, never crashes (W9).

    The pod-validated optimized load path for the ``gliner2`` framework (``gliner2`` 1.3.2): its
    ``GLiNER2.from_pretrained(model_id, quantize=…, compile=…)`` accepts native ``quantize`` (fp16)
    and ``compile`` (``torch.compile`` + fused GPU kernels, "no extra dependencies required")
    kwargs — pod-measured **2.83× (PII) / 1.74× (combined NER+RE) at byte-identical output**. Both
    resolve via :func:`resolve_gliner2_knob` (default True on cuda / False on cpu, or the config
    override).

    **Guarded (never crashes):** an optimized load can fail (a torch/CUDA/compile fault, an older
    ``gliner2`` that does not accept the kwargs); ANY such failure is **logged** and the loader is
    retried **plain** (``loader(model_id)``) so the model still loads — a bad quantize/compile can
    only ever cost throughput, never correctness or a crash. When both knobs resolve ``False`` (the
    cpu default) the loader is called plain directly (no kwargs to pass, no fallback needed). The
    ``quantize``/``compile`` config values are resolved *before* the load so an invalid value is a
    ConfigError the adapter surfaces, not a silent fallback.

    **W9-jit — the other compiler.** ``compile``/``quantize`` only govern ``torch.compile``; the
    ``gliner2`` backbone is mdeberta-v3, which scripts its relative-position helper with
    ``@torch.jit.script`` and, on a Blackwell/CUDA-13 box, dies in nvrtc on the FIRST forward
    (``libnvrtc-builtins.so.13.0``) — a crash no load-time guard can catch. The TorchScript JIT is
    neutralized (:func:`neutralize_torch_jit`) here before the load, unconditionally, for EVERY
    gliner2 load path (optimized or plain) — but that is **defence-in-depth, not the fix**: the
    helpers are scripted when ``from gliner2 import GLiNER2`` is evaluated, one statement before
    this loader is entered, and no lever can un-script them afterwards. The Provider's own call
    *ahead of its backbone import* is what actually closes #190 (ADR-0036 W9-jit).
    Throughput-only, byte-identical output.
    """
    quantize = resolve_gliner2_knob("quantize", device, config)
    compile_ = resolve_gliner2_knob("compile", device, config)
    neutralize_torch_jit(provider_name=provider_name, config=config)
    if not quantize and not compile_:
        return loader(model_id)
    try:
        return loader(model_id, quantize=quantize, compile=compile_)
    except Exception:  # noqa: BLE001 - deliberate: an optimized load must degrade to plain, never crash
        _log.warning(
            "%s: gliner2 optimized load (quantize=%s, compile=%s) failed; loading plain "
            "(throughput only)",
            provider_name,
            quantize,
            compile_,
            exc_info=True,
        )
        return loader(model_id)


# ---------------------------------------------------------------------------
# Model token-window bound (W12-gliner-window) — the hard model-boundary guard.
# ---------------------------------------------------------------------------

#: The default gliner-family model token window. The gliner / gliner2 encoders are mdeberta-v3,
#: whose tokenizer processes at most this many tokens; a longer input is SILENTLY truncated (tail
#: entities/relations lost, offsets past the window unreliable). Pod-measured: with a chunk cap of
#: 768 *chunker* tokens, the mdeberta token length hit median 666 / MAX 921 — 2 of 7 chunks exceed
#: 768 — because the chunker counts in its OWN tokenizer, ~1.2× fewer tokens than mdeberta. So every
#: gliner-family extractor bounds its window EXPLICITLY + CONFIGURABLY at this default; content past
#: it is a *documented* boundary (not extracted), never a silent surprise.
DEFAULT_GLINER_MAX_LEN = 768


def resolve_max_len(config: dict[str, Any], default: int = DEFAULT_GLINER_MAX_LEN) -> int:
    """The configured model token-window bound (``max_len``), clamped ``>= 1`` (default 768).

    A per-Provider knob so an operator can raise/lower the window for a different checkpoint; the
    clamp means a configured 0/negative degrades to ``1`` (a one-token window) rather than an empty
    or negative bound. Pure config read — no torch, no model.
    """
    return max(1, int(config.get("max_len", default)))


# ---------------------------------------------------------------------------
# Intra-op thread cap + fp16 device placement (W9-scaling) — the two fixes for
# the fp16-eager / thread-oversubscription hang the gliner2 providers hit at
# scale on a many-core Blackwell host (RTX PRO 4500, torch 2.8/cu128).
# ---------------------------------------------------------------------------

#: The default cap on torch **intra-op** threads a gliner2 provider applies at load. On a many-core
#: host (the diagnosed Blackwell box has 128 cores) torch's default intra-op pool is
#: ``os.cpu_count()``, and gliner2's internal batching then oversubscribes to ~129 threads — a
#: measured ~520% CPU spin with the GPU at 0% and NO forward progress (even with ``OMP_NUM_THREADS``
#: exported, because once torch is imported it owns its own intra-op pool, not the OMP env). A small
#: cap keeps CPU-side tokenisation/pre-processing from thrashing the scheduler; on a GPU the heavy
#: math is off-CPU, so a low cap costs ~nothing. Configurable per provider (``intra_op_threads``).
#:
#: T-DEV-2: this is now the **reference** value, not the answer —
#: :func:`derive_intra_op_threads` reads the machine and reproduces it exactly at
#: :data:`REFERENCE_CPU_CORES`.
DEFAULT_INTRA_OP_THREADS = 8

#: The core count :data:`DEFAULT_INTRA_OP_THREADS` was derived against (the diagnosed Blackwell
#: host). Named so the derivation can be checked against its own provenance in a test.
REFERENCE_CPU_CORES = 128


def effective_cpu_count() -> int | None:
    """The cores this **process** may actually use — not the ones the host advertises (T-DEV-2).

    ``os.cpu_count()`` reports the machine's cores. On a rented pod that is the *host's* number:
    ``nproc`` says 128 while the container's cgroup / CPU affinity allows 16, or 4. Sizing a thread
    pool off the host number on such a slice is the same many-core oversubscription W9-scaling
    introduced the cap to prevent, arrived at from the other end.

    Order: ``os.process_cpu_count()`` (3.13+, the affinity-aware answer) →
    ``len(sched_getaffinity)`` (Linux) → ``os.cpu_count()`` → ``None``. Never raises; ``None``
    means unreadable, which callers
    treat as "keep the reference default" rather than as a number.
    """
    import os  # noqa: PLC0415 - stdlib, kept local so import cost is only paid on use

    # ``getattr`` rather than direct attribute access for both affinity probes: neither exists on
    # every platform (``process_cpu_count`` is 3.13+, ``sched_getaffinity`` is Linux-only), and a
    # direct reference would not type-check on a macOS dev box.
    probes: list[Callable[[], Any]] = []
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if callable(process_cpu_count):
        probes.append(process_cpu_count)
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if callable(sched_getaffinity):
        probes.append(lambda: len(sched_getaffinity(0)))
    probes.append(os.cpu_count)

    for probe in probes:
        try:
            value = probe()
        except Exception:  # noqa: BLE001 - an unreadable probe falls through to the next one
            continue
        if isinstance(value, int) and value > 0:
            return value
    return None


def derive_intra_op_threads(cores: int | None) -> int:
    """The intra-op cap for a host with ``cores`` usable cores — reproduces 8 at 128 (T-DEV-2).

    ``min(8, cores)``, floored at 1. The 8 is the pod-measured point
    (:data:`DEFAULT_INTRA_OP_THREADS` against :data:`REFERENCE_CPU_CORES`); the ``min`` is what
    stops a 4-core container slice from
    being handed a cap larger than the cores it owns. ``None`` (unreadable) keeps the reference
    value rather than guessing downward — an unknown core count is not evidence of a small one.
    """
    if cores is None:
        return DEFAULT_INTRA_OP_THREADS
    return max(1, min(DEFAULT_INTRA_OP_THREADS, int(cores)))


def resolve_intra_op_threads(
    config: Mapping[str, Any], default: int | None = None
) -> int | None:
    """Resolve the intra-op thread cap (``config["intra_op_threads"]``) — torch-free (W9-scaling).

    Unset → the **derived** default (:func:`derive_intra_op_threads` over
    :func:`effective_cpu_count`; 8 on the 128-core reference host, less on a smaller *container
    slice*). An explicit int pins it. A value ``<= 0`` **disables** the cap (returns
    ``None`` — leave torch's default pool untouched, the explicit opt-out for a box that wants all
    cores). A positive value is clamped to ``os.cpu_count()`` (never request more threads than
    cores). A non-int is a hard error the caller surfaces as a ConfigError. Pure config read — no
    torch import (ADR-0016), so a registry that only imports the module never pulls torch.

    RESIDUAL, recorded rather than papered over: the clamp on an **explicit** request still reads
    ``os.cpu_count()`` (the host), not :func:`effective_cpu_count` (the slice), so on a
    cgroup-limited pod an operator asking for 100_000 threads is still clamped to the host's
    count. Tightening it
    would change ``test_perf_seam.py::test_resolve_intra_op_threads_default_and_clamp``, which
    computes its expectation from ``os.cpu_count()`` and is outside this lane's ownership. The
    DEFAULT — the value that actually ships — is affinity-aware, which is where the defect was.
    """
    if default is None:
        default = derive_intra_op_threads(effective_cpu_count())
    raw = config.get("intra_op_threads", default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        msg = f"invalid intra_op_threads {raw!r}: must be an int (>= 1 to cap, <= 0 to disable)"
        raise ValueError(msg)
    if raw <= 0:
        return None
    import os  # noqa: PLC0415 - stdlib, only to read the core count when a cap is requested

    cores = os.cpu_count() or raw
    return min(int(raw), int(cores))


def cap_intra_op_threads(n: int | None, *, provider_name: str = "provider") -> int | None:
    """Apply an intra-op thread cap via ``torch.set_num_threads(n)`` — guarded, never crashes (W9).

    ``n`` is the resolved cap (:func:`resolve_intra_op_threads`); ``None`` is the explicit opt-out
    (no cap applied, torch's default pool kept). Imports torch **lazily and defensively**: on a
    torch-free host (the offline test env) or if torch rejects the call it is a **logged no-op**
    (returns ``None``) — capping threads is best-effort throughput insurance against the many-core
    oversubscription spin, never a hard dependency or a crash. Returns the applied cap, else
    ``None``.

    Thin delegator: the mutation is performed — and the prior pool size RECORDED — by the current
    :class:`TorchRuntime`, so a :func:`scoped_torch_runtime` boundary can restore it.
    """
    return current_torch_runtime().cap_intra_op_threads(n, provider_name=provider_name)


def _detect_model_device(model: Any) -> str | None:
    """Best-effort **read** of a model's resident torch device as a string — never raises.

    Tries the common wrapper shapes (``model.device``, ``model.model.device``, the first parameter's
    device) and returns a lowercase device string (``"cuda"``/``"cuda:0"``/``"cpu"``) or ``None`` if
    none can be read. Purely diagnostic — used to LOG where inference will actually run so the
    fp16-on-CPU failure (a fp16-converted model that silently stayed on CPU) is visible, not silent.
    """
    for holder in (model, getattr(model, "model", None)):
        if holder is None:
            continue
        dev = getattr(holder, "device", None)
        if dev is not None:
            return str(dev).lower()
        params = getattr(holder, "parameters", None)
        if callable(params):
            try:
                first = next(iter(params()))
            except (StopIteration, TypeError):
                continue
            dev = getattr(first, "device", None)
            if dev is not None:
                return str(dev).lower()
    return None


def ensure_gliner2_on_cuda(
    model: Any, *, cuda: bool, provider_name: str = "provider"
) -> str | None:
    """Ensure a gliner2 model is on cuda when a GPU is present, then LOG its device — guarded (W9).

    The fp16-eager GPU-idle hang the diagnosis flagged (``quantize=True`` → "Converted model to
    fp16" then a CPU thread-spin with the GPU at 0%) is consistent with a fp16-converted model that
    **stayed on CPU** — fp16 eager on CPU is pathologically slow and CPU-bound. This is the
    defence: when a GPU is present the model is moved to cuda best-effort (``model.to("cuda")``,
    tried across the ``model`` / ``model.model`` wrapper shapes), and its resident device is read
    back and **logged** so an operator can SEE where inference lands (the fp16-on-CPU condition is
    no longer silent). Torch-free-safe: on a torch-free host (offline tests) ``cuda`` is False under
    the ``LATENCE_CUDA=0`` probe and this is a no-op returning ``None``. NEVER raises — placement is
    insurance (``from_pretrained``/``quantize`` may already place correctly, in which case
    ``.to("cuda")`` is a harmless no-op), and the exact GLiNER2 placement API is pod-validated by
    the maintainer.
    """
    if not cuda:
        return _detect_model_device(model)
    moved = False
    for target in (model, getattr(model, "model", None)):
        if target is None:
            continue
        to = getattr(target, "to", None)
        if callable(to):
            try:
                to("cuda")
            except Exception:  # noqa: BLE001 - best-effort move; a rejecting wrapper is not fatal
                continue
            moved = True
            break
    device = _detect_model_device(model)
    if device is not None and "cuda" in device:
        _log.info("%s: gliner2 model on %s (cuda placement confirmed)", provider_name, device)
    elif device is not None:
        _log.warning(
            "%s: gliner2 model reports device=%s AFTER cuda placement (move %s) — fp16-on-CPU "
            "risk; set quantize=false if the run is CPU-bound with the GPU idle",
            provider_name,
            device,
            "applied" if moved else "not available",
        )
    elif not moved:
        _log.warning(
            "%s: could not place the gliner2 model on cuda (no .to on the wrapper) or read its "
            "device; verify GPU placement on the pod",
            provider_name,
        )
    return device


def apply_gliner_max_len(model: Any, max_len: int) -> None:
    """Bound a GLiNER-library model's token window to ``max_len`` (``config.max_len``), guarded.

    The ``gliner`` library truncates every input to ``model.config.max_len`` (mdeberta-v3's window);
    the truncation is SILENT, so a chunk that tokenizes past it loses its tail entities/relations
    and any offset past the window is unreliable. Setting it here makes the window an EXPLICIT,
    configurable, documented boundary (the hard model-boundary guard) instead of a silent surprise.

    **Guarded (never crashes):** a model / config object that exposes no settable ``max_len`` (a
    test double, a future architecture, a frozen config) is left unchanged — bounding the window is
    best-effort and must never fault a run. Applied to the raw model *before* any ``torch.compile``
    wrap so it takes effect on the underlying module.
    """
    cfg = getattr(model, "config", None)
    if cfg is None or not hasattr(cfg, "max_len"):
        return
    try:
        cfg.max_len = max_len
    except Exception:  # noqa: BLE001 - best-effort window bound; a rejecting config never crashes
        _log.warning(
            "could not set model.config.max_len=%d (window bound skipped); model default applies",
            max_len,
        )


# ---------------------------------------------------------------------------
# Token-budget micro-batching (dogfood-2b-process) — decouple the model-FORWARD
# tensor batch from the progress-cadence ``batch_size`` so a learned encoder's
# single ``batch_extract_entities`` call can never demand an OOM-sized activation.
# ---------------------------------------------------------------------------
#
# The gliner2 (mdeberta-v3) span-scorer pads a group of N texts into ONE tensor and scores every
# span position, so its peak activation scales with (N × seq_len) — NOT with N alone. A whole
# progress-``batch_size`` group (default 32) of long chunks therefore demands a single multi-GiB
# allocation. Measured on a 32 GB Blackwell pod (dogfood-2b-process): batch_size=32 of ~768-token
# chunks spiked ~+17 GiB and OOM'd at redact, while batch_size=4 peaked at 21.6 GiB and passed
# cleanly. The cure is to split each forward's texts into MICRO-BATCHES bounded by a conservative
# token budget (and a hard text-count cap), issue one native call per micro-batch, and concatenate
# the per-text results IN ORDER — the output is batch-invariant, so this is byte-identical while the
# peak allocation stays bounded for ANY configured ``batch_size``/``max_len``/label count.

#: The hard cap on texts packed into ONE model forward. This is the PRIMARY safety bound: gliner2
#: pads every text in a micro-batch up to the batch's longest sequence, so the padded worst case is
#: ``max_texts × max_len`` regardless of how short the other texts are — a token budget summed over
#: *actual* lengths cannot see that padding. Pinned to **4** because a batch of 4 × 768-token chunks
#: peaked at 21.6 GiB and passed on the 32 GB pod (batch_size=32 OOM'd). With the default 768 window
#: this bounds the padded forward to 4 × 768 = 3072 token-positions — the empirically-safe point.
DEFAULT_MICROBATCH_MAX_TEXTS = 4

#: The token budget for ONE model forward: the sum of the micro-batch's per-text token lengths
#: (each capped at ``max_len``) may not exceed this. = 4 × :data:`DEFAULT_GLINER_MAX_LEN` (3072),
#: the pod-safe "4 texts of 768 tokens" point. This SECONDARY bound bites when texts are long but
#: ``max_len`` is raised above the default (where ``max_texts`` alone would allow a larger padded
#: tensor); with the default window ``max_texts`` is the tighter of the two. Both default SAFE on a
#: modest GPU; both are overridable per-provider (:func:`resolve_microbatch_bounds`) for a box with
#: more or less VRAM, but the defaults never assume the 32 GB card.
DEFAULT_MICROBATCH_TOKEN_BUDGET = 4 * DEFAULT_GLINER_MAX_LEN

# ---------------------------------------------------------------------------
# T-DEV-2 — those two constants encoded ONE pod. Derive them from the machine.
# ---------------------------------------------------------------------------
#
# "batch-4 peaked at 21.6 GiB" was measured on a **32 GB** card. On the 24 GB RTX 3090 that is
# 2.4 GiB of headroom — before the OCR server and before fragmentation (DECISIONS.md §D3). The
# constants above stay as the *reference point*; the functions below turn them into a derivation
# that reproduces them EXACTLY on the reference hardware and lowers them on a smaller card.
#
# Two deliberate asymmetries, both stated so a reader can disagree with the right thing:
#
# * Detection may only LOWER the bound, never raise it. 4 × 768 is the one micro-batch size we have
#   ever measured a peak for. Extrapolating upward from a single point on an 80 GB card would be
#   inventing data; more VRAM stays reachable, as an explicit ``microbatch_max_texts``.
# * UNKNOWN VRAM keeps the reference default — the opposite of the T-DEV-1 JIT gate. There the fault
#   is an uncatchable first-forward native crash, so unknown must fail safe. Here the fault is a
#   CUDA OOM, which the providers' reactive micro-batch backoff (:func:`is_cuda_oom`) recovers from,
#   so collapsing every un-probeable host to a batch of one would trade permanent throughput for
#   insurance against a recoverable event.

#: The env override that PINS total VRAM in MiB, bypassing the probe. Same posture as
#: ``LATENCE_CUDA`` / ``LATENCE_COMPUTE_CAPABILITY``: deterministic for CI, and an escape hatch for
#: a container where the driver CLI is not reachable.
LATENCE_TOTAL_VRAM_MIB_ENV = "LATENCE_TOTAL_VRAM_MIB"

#: The card :data:`DEFAULT_MICROBATCH_MAX_TEXTS` / :data:`DEFAULT_MICROBATCH_TOKEN_BUDGET` were
#: measured on — 32 GiB, in MiB. Named so the derivation can be checked against its own provenance.
REFERENCE_VRAM_MIB = 32 * 1024


def _vram_from_imported_torch() -> int | None:
    """Total VRAM in MiB from a torch that is **already imported** — never importing it.

    ``sys.modules`` only (ADR-0016: ``latence-core`` must not pull torch to size a batch). The
    **smallest** device wins: one micro-batch bound is applied process-wide, so it has to be the one
    the smallest card survives.
    """
    import sys  # noqa: PLC0415 - stdlib, kept local so import cost is only paid on use

    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        count = int(torch.cuda.device_count())
        if count < 1:
            return None
        totals = [int(torch.cuda.get_device_properties(i).total_memory) for i in range(count)]
    except Exception:  # noqa: BLE001 - a torch that cannot answer is UNKNOWN, never a fabricated size
        return None
    smallest = min(totals)
    return smallest // (1024 * 1024) if smallest > 0 else None


def _vram_from_nvidia_smi() -> int | None:
    """Total VRAM in MiB from ``nvidia-smi`` — torch-free, the primary probe.

    ``--query-gpu=memory.total --format=csv,noheader,nounits`` prints one integer (MiB) per GPU; the
    smallest wins, for the same process-wide reason. Every failure mode — tool absent, non-zero
    exit, timeout, a driver answering ``[N/A]`` — returns ``None`` (unknown), never a parsed 0.
    """
    import shutil  # noqa: PLC0415 - stdlib, kept local so import cost is only paid on use
    import subprocess  # noqa: PLC0415, S404 - a read-only, fixed-argv driver query

    if shutil.which("nvidia-smi") is None:
        return None
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            [  # noqa: S607
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:  # noqa: BLE001 - a probe that cannot run is unknown, never a crash
        return None
    if proc.returncode != 0:
        return None
    lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    if not lines or not all(line.isdigit() for line in lines):
        # ``[N/A]`` from a driver that cannot report it must not parse into "0 MiB of VRAM".
        return None
    values = [int(line) for line in lines]
    return min(values) if min(values) > 0 else None


def detect_total_vram_mib() -> int | None:
    """Total VRAM of the smallest visible GPU, in MiB, or ``None`` when unknown (T-DEV-2).

    ``LATENCE_TOTAL_VRAM_MIB`` → a torch already in ``sys.modules`` → ``nvidia-smi`` → ``None``.
    Torch-free by construction (ADR-0016). ``None`` is a first-class answer, not a fallback to a
    guess: :func:`derive_microbatch_max_texts` treats it as "keep the reference default".
    """
    import os  # noqa: PLC0415 - stdlib, kept local so import cost is only paid on use

    override = os.environ.get(LATENCE_TOTAL_VRAM_MIB_ENV, "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)
    from_torch = _vram_from_imported_torch()
    if from_torch is not None:
        return from_torch
    return _vram_from_nvidia_smi()


def derive_microbatch_max_texts(vram_mib: int | None) -> int:
    """The micro-batch text cap for a card of ``vram_mib`` — reproduces 4 at 32 GiB (T-DEV-2).

    ``floor(4 × vram / 32 GiB)``, clamped into ``[1, 4]``. Linear because the only structure the one
    measurement supports is a *ratio*: 4 texts fitted a 21.6 GiB peak into 32 GiB, so the same
    fraction of a smaller card is the honest scaling — a two-parameter ``base + N × per_text`` model
    fitted to a single point would look more principled while being less justified.

    * 32 GiB → **4** (byte-identical to the pre-T-DEV-2 default; the old pod does not change)
    * 24 GiB → **3** (the RTX 3090 target)
    * ≥ 40 GiB → **4** — the cap never rises above the measured point (see the module note)
    * ``None`` → **4** — unknown keeps the reference; OOM here is recoverable, unlike the JIT fault
    """
    if vram_mib is None:
        return DEFAULT_MICROBATCH_MAX_TEXTS
    scaled = (DEFAULT_MICROBATCH_MAX_TEXTS * int(vram_mib)) // REFERENCE_VRAM_MIB
    return max(1, min(DEFAULT_MICROBATCH_MAX_TEXTS, scaled))


def estimate_token_length(text: str, max_len: int) -> int:
    """A conservative, torch-free estimate of a text's model-token length, capped at ``max_len``.

    The real mdeberta-v3 token count needs the tokenizer (torch/transformers), which
    ``latence-core`` must not import (ADR-0016). This proxy assumes ~4 characters per token (the
    usual English average) and rounds UP, then caps at ``max_len`` because the provider threads
    ``max_len`` into the native call and the model truncates past it — so a text can contribute at
    most ``max_len`` positions to the padded batch. Over-estimating only splits micro-batches
    *smaller* (safer); the ``max_texts`` cap is the hard guarantee regardless of this estimate.
    """
    if not text:
        return 0
    return min(-(-len(text) // 4), max_len)


def iter_token_budget_microbatches(
    texts: Sequence[str],
    *,
    max_len: int,
    max_texts: int = DEFAULT_MICROBATCH_MAX_TEXTS,
    token_budget: int = DEFAULT_MICROBATCH_TOKEN_BUDGET,
    token_len: Callable[[str, int], int] = estimate_token_length,
) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` half-open index ranges partitioning ``texts`` into safe micro-batches.

    Greedy and order-preserving: it walks ``texts`` once and starts a new micro-batch when adding
    the next text WOULD exceed either the ``max_texts`` count cap or the ``token_budget`` (summed
    per-text ``token_len``, each capped at ``max_len``). Every micro-batch holds **at least one**
    text, so a lone text whose own length exceeds the budget still goes through in a batch of one
    (the reactive OOM backoff is the last-resort net for that pathological case). The ranges tile
    ``texts`` exactly and in order, so the caller can slice ``texts[start:end]`` per micro-batch and
    concatenate the per-text results back in input order — the output is batch-invariant.

    Pure index arithmetic over string lengths — no torch, no model — so ``latence-core`` stays
    torch-free at import (ADR-0016). ``max_texts``/``token_budget`` are clamped ``>= 1`` so a
    degenerate bound degrades to a batch of one rather than looping forever.
    """
    max_texts = max(1, max_texts)
    token_budget = max(1, token_budget)
    n = len(texts)
    start = 0
    while start < n:
        end = start
        tokens = 0
        while end < n:
            length = token_len(texts[end], max_len)
            # Always admit the first text of a micro-batch (even if it alone exceeds the budget);
            # otherwise break before a text that would overflow the count OR the token budget.
            if end > start and (end - start >= max_texts or tokens + length > token_budget):
                break
            tokens += length
            end += 1
        yield (start, end)
        start = end


def resolve_microbatch_bounds(config: Mapping[str, Any]) -> tuple[int, int]:
    """Resolve the ``(max_texts, token_budget)`` micro-batch bounds from a Provider ``config``.

    Reads the optional ``microbatch_max_texts`` / ``microbatch_token_budget`` knobs. **T-DEV-2:**
    when neither is set the defaults are DERIVED from the detected card
    (:func:`detect_total_vram_mib` → :func:`derive_microbatch_max_texts`) rather than read off the
    32 GB pod's constants — 4 / 3072 on that pod exactly as before, 3 / 2304 on the 24 GB RTX 3090,
    and the reference values on a host whose VRAM cannot be read. The token budget stays the
    identity it always was, ``max_texts × DEFAULT_GLINER_MAX_LEN``, so both bounds move together.

    An explicit config value wins over detection in both directions. A non-int or ``< 1`` value is a
    ValueError the caller surfaces as a ConfigError (the same fail-fast discipline as
    ``batch_size``/``max_len``). Torch-free (ADR-0016) — the probe is env → an already-imported
    torch → ``nvidia-smi``, never an import.
    """
    detected_texts = derive_microbatch_max_texts(detect_total_vram_mib())
    max_texts = _positive_int_knob(config, "microbatch_max_texts", detected_texts)
    token_budget = _positive_int_knob(
        config, "microbatch_token_budget", detected_texts * DEFAULT_GLINER_MAX_LEN
    )
    return max_texts, token_budget


# ---------------------------------------------------------------------------
# Label-set BUCKETING (T7-efficient-batching) — how a per-chunk label set and a
# native batched forward stop being mutually exclusive.
# ---------------------------------------------------------------------------
#
# The extraction redesign gives every chunk its OWN small, induced label set (T1/T2). gliner2's
# batched entry points (``batch_extract_entities`` / ``batch_extract_relations``) take ONE label
# schema for the whole call. The two naive readings are both bad:
#
#   * one native call per chunk — correct labels, no batching, terrible throughput;
#   * union the batch's labels — maximal batch width, but back to the ~40-label schema whose
#     precision degradation and (batch × positions × LABELS) span-scoring activation are exactly
#     what the redesign exists to remove (the measured ~17 GiB → CUDA-OOM).
#
# BUCKETING is the third reading, and it dominates both: partition the batch by label set, then
# run ONE native batched forward per bucket. Chunks that share a label set — on a real corpus,
# most chunks within a document or topic — batch together at THEIR OWN small label count; chunks
# with distinct sets form their own buckets. Maximal batch width, never the corpus union.
#
# Two second-order wins fall out, and they are the reason ``compile`` is now safe to default on
# (see :func:`resolve_gliner2_knob` callers): within a bucket the label dimension is CONSTANT, and
# the token-budget micro-batch caps the batch dimension — so gliner2's
# ``torch.compile(..., dynamic=True)`` sees one shape family per bucket instead of a fresh one per
# chunk, which is what made it recompile-thrash before.


_BucketKey = TypeVar("_BucketKey", bound="Hashable")


def iter_label_buckets(keys: Sequence[_BucketKey]) -> Iterator[tuple[_BucketKey, list[int]]]:
    """Partition ``range(len(keys))`` into buckets of indices that share the same ``keys[i]``.

    ``keys[i]`` is item ``i``'s label set (a hashable — in practice a ``tuple[str, ...]`` of the
    chunk's effective labels, order-significant because it is also the order handed to the model).
    Yields ``(key, indices)`` pairs; the caller slices its own texts by ``indices``, issues ONE
    native batched forward per bucket with ``key`` as the schema, and scatters the per-text results
    back by index — so emission stays in input order regardless of bucketing.

    **Deterministic** (ADR-0033 / the byte-identical bar): buckets are yielded in FIRST-SEEN key
    order and each bucket's indices ascend, so the same input produces the same bucket sequence and
    the same per-bucket text order on every run and every host — no set iteration, no sort by a
    hash. Pure index arithmetic over hashables: no torch, no model, so ``latence-core`` stays
    torch-free at import (ADR-0016).

    An empty ``keys`` yields nothing (no bucket, no forward). A single distinct key yields exactly
    one bucket — which is the no-induction / predefined-only case, i.e. every chunk shares the
    config label set and the whole batch is ONE forward, byte-identical to the un-bucketed path.
    """
    buckets: dict[_BucketKey, list[int]] = {}
    for index, key in enumerate(keys):
        buckets.setdefault(key, []).append(index)
    yield from buckets.items()


def _positive_int_knob(config: Mapping[str, Any], key: str, default: int) -> int:
    raw = config.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        msg = f"invalid {key} {raw!r}: must be an int >= 1"
        raise ValueError(msg)
    if raw < 1:
        msg = f"invalid {key} {raw}: must be >= 1"
        raise ValueError(msg)
    return int(raw)


# ---------------------------------------------------------------------------
# The LOAD CHOREOGRAPHY (C4) — three deep entry points, one per native-library
# family. Everything above this line is a STEP; everything below is the whole
# SEQUENCE, so a Provider author never has to know the ordering at all.
# ---------------------------------------------------------------------------
#
# Why this exists. The seven steps a learned Provider used to perform by hand —
#
#   neutralize the JIT → import the backbone → check the DeviceDecision → load+cast →
#   place on the resolved device → bound the token window → compile
#
# — carry FIVE "before" constraints, and not one of them was expressible in a signature. The only
# thing holding them was an AST linter that parsed each Provider's SOURCE and asserted the calls
# were present and correctly sequenced. When the only way to hold an invariant is to lint the
# callers, the invariant belongs INSIDE the module. Three field bugs came out of that gap:
#
#   * #190 — the JIT kill sat inside the loader, i.e. AFTER ``from gliner import GLiNER`` had
#     already let transformers script the deberta_v2 relative-position helpers. The "fix" was a
#     SECOND, EARLIER call copy-pasted into every Provider.
#   * #191 — ``place_on_device`` was wired to one Provider of three; the other two cast to bf16 off
#     a bare availability probe while the weights stayed resident on CPU.
#   * the OCR parsers used a bare ``model.to(device)``, so "prove it landed or raise" covered 3 of
#     5 model-placing Providers.
#
# The fix that removes the CLASS of bug: the backbone import is not something the Provider does and
# then tells us about — it is a ``Callable[[], Any]`` factory the Provider HANDS IN, which these
# entry points call at the one instant that is correct. "Neutralize before import" stops being a
# rule and becomes a straight line of code that a caller cannot reorder because the caller never
# sees it.
#
# What deliberately did NOT move: the config knobs (still read from the Provider's own ``config``),
# the native dependency (each package still owns and imports its own), and the placement CALL
# (``place_on_device`` still runs against the model object the Provider's library produced) —
# ADR-0036 §2's "device placement stays the adapter's job" is about WHO owns the native dep and the
# model object, not about who holds the ordering. Only the SEQUENCING moved inward.


def _require_device(decision: DeviceDecision, provider_name: str) -> str | None:
    """The resolved device to place on, or a loud :class:`ConfigError` — never a silent fallback.

    The first step of every load choreography, and the #191 contract in one place:
    ``select_device`` returns a **skip-with-flag** decision when a GPU-only Provider is asked for on
    a CPU-only host (or CPU is pinned for a GPU-only Provider). Before this, each Provider
    re-derived
    that check by hand — and the two that forgot fell through to a CPU load with the GPU idle, at
    ~11.5 s/chunk, with NO warning. Honour the resolved device or fail loudly (ADR-0036).

    A non-skipped decision whose ``device`` is ``None`` (a Provider that resolves no device and lets
    its library auto-place) is legal and returns ``None`` — placement then reads back the resident
    device for verification only.
    """
    if decision.skipped:
        msg = f"{provider_name}: {decision.reason}"
        raise ConfigError(msg)
    return decision.device


def _import_backbone(
    import_model_cls: Callable[[], Any], config: Mapping[str, Any], provider_name: str
) -> Any:
    """Kill the TorchScript JIT, cap the intra-op threads, THEN evaluate the backbone import.

    The ordering that #190 cost two rounds to learn, expressed as three consecutive statements a
    caller cannot get between:

    1. :func:`neutralize_torch_jit` — sets ``PYTORCH_JIT=0`` (read by torch **at torch import**) and
       disables the runtime state. Must precede step 3, because the backbone import transitively
       evaluates transformers' ``deberta_v2`` module whose relative-position helpers carry
       ``@torch.jit.script``; once they are ``ScriptFunction``s no lever can un-script them, and
       on a
       Blackwell/CUDA-13 box they die in nvrtc on the FIRST forward. **Capability-gated since
       T-DEV-1** (``config`` is threaded in): on a positively-identified known-good architecture
       the JIT is left enabled and step 1 is a logged no-op, because the fault it defends against
       does not exist there.
    2. :func:`cap_intra_op_threads` — the many-core oversubscription guard (~520% CPU spin with the
       GPU at 0%). It imports torch, which is *why* it must come after step 1 and not before: the
       env switch has to be in place for the import torch reads it at. Applied for EVERY family,
       not just
       the one Provider that happened to wire it: the blessed stack runs two mdeberta backbones in
       one process on one many-core host, and protecting only one of them protects neither.
    3. ``import_model_cls()`` — the Provider's own factory. It stays with the Provider because the
       native dependency and its install hint do (ADR-0036 §2); only the INSTANT it is called moved
       here. An ``ImportError`` from it propagates unwrapped — a missing optional dep is an
       actionable operator message ("pip install latence-…"), not a runtime provider fault.
    """
    neutralize_torch_jit(provider_name=provider_name, config=config)
    cap_intra_op_threads(resolve_intra_op_threads(config), provider_name=provider_name)
    return import_model_cls()


def _typed_load(
    load: Callable[[], Any],
    model_id: str,
    provider_name: str,
    guard: LoadGuard | None,
) -> Any:
    """Run a native model load, mapping any native fault to a typed error (ADR-0034).

    ``guard`` is the caller's own typed-error policy — in practice ``AdapterBase.guard``, whose
    ``config_exceptions`` decide which native types mean "the operator's config is wrong"
    (a bad ``model`` id → :class:`~latence_core.errors.ConfigError`) versus "the provider failed"
    (→ :class:`~latence_core.errors.ProviderError`). That policy is per-adapter, so it stays with
    the adapter; what this seam owns is applying it at the RIGHT INSTANT — around the native load
    and **nothing else**, so a missing-dependency ``ImportError`` from the backbone-import factory
    (raised before this call) still reaches the operator as its actionable "pip install …" hint
    instead of being relabelled a provider failure.

    A caller with no such policy (a Provider that is not an :class:`AdapterBase`) passes ``None``
    and gets the default: any native fault becomes a :class:`~latence_core.errors.ProviderError`
    naming the provider, the model id and the native exception's TYPE NAME — never its message,
    which can echo an input (the same no-PII discipline ``AdapterBase.guard`` follows). A framework
    :class:`~latence_core.errors.LatenceError` passes through unwrapped either way.
    """
    if guard is not None:
        with guard("load_model"):
            return load()
    try:
        return load()
    except LatenceError:
        raise
    except Exception as exc:
        msg = (
            f"{provider_name}: could not load model {model_id!r} "
            f"(native {type(exc).__name__})"
        )
        raise ProviderError(msg) from exc


def load_gliner_model(
    import_model_cls: Callable[[], Any],
    model_id: str,
    config: Mapping[str, Any],
    decision: DeviceDecision,
    *,
    provider_name: str,
    guard: LoadGuard | None = None,
) -> Any:
    """Load a plain-``gliner``-library model, ready to run — the WHOLE sequence, one call.

    The single entry point the three plain-``gliner`` Providers (``entity.gliner``,
    ``redaction.gliner_pii``, ``relation.gliner_relex``) load through. Their ``_load_model`` bodies
    were copy-paste identical modulo one provider-name string; this is that body, once.

    **Interface** — everything a caller must know:

    * ``import_model_cls`` — a zero-arg factory returning the ``GLiNER`` class. It is called at the
      one correct instant (after the JIT is dead); a missing dep's ``ImportError`` propagates
      unwrapped so the "pip install …" hint survives.
    * ``model_id`` / ``config`` — the checkpoint and the Provider's own config mapping. Every perf
      knob (``dtype``, ``compile``, ``max_len``, ``intra_op_threads``) is read from ``config`` here;
      an invalid value raises :class:`ValueError` (the caller surfaces it as a ConfigError, as it
      already does for ``batch_size``).
    * ``decision`` — the :class:`~latence_core.providers.device.DeviceDecision` from
      ``select_device``. A **skipped** decision is a :class:`~latence_core.errors.ConfigError` here,
      not a silent CPU fallback (#191).
    * returns the ready model. Raises :class:`~latence_core.errors.ConfigError` (skip / bad knob),
      :class:`~latence_core.errors.ProviderError` (native load fault, or a model that reports a
      different device than the one it was placed on), or ``ImportError`` (missing dep).
    * **no ordering to know.** That is the point.

    Behind it, in order: skip check → JIT kill → thread cap → backbone import → plain load then
    ``.to(bf16)`` (the plain library REJECTS ``torch_dtype``/``attn_implementation`` — pod-proven) →
    placement on the resolved device **with read-back verification** → the mdeberta token-window
    bound (before any compile wrap, so it lands on the underlying module) → guarded
    ``torch.compile``. Output is byte-identical throughout: every step here is throughput or
    fail-loud, never a change to the emitted spans/labels (ADR-0044's redaction floor untouched).
    """
    device = _require_device(decision, provider_name)
    model_cls = _import_backbone(import_model_cls, config, provider_name)
    model = _typed_load(
        lambda: load_then_cast(model_cls.from_pretrained, model_id, config, device),
        model_id,
        provider_name,
        guard,
    )
    place_on_device(model, device, provider_name=provider_name)
    # BEFORE the compile wrap: the bound must land on the underlying module, not on a compiled
    # wrapper that may not forward attribute writes to it.
    apply_gliner_max_len(model, resolve_max_len(dict(config)))
    return maybe_compile(model, config, provider_name=provider_name)


# ---------------------------------------------------------------------------
# Tokenizer-config shape shim (dogfood-2b-process; PROMOTED to core 2026-08-25, ADR-0064)
# ---------------------------------------------------------------------------
#
# It lives here rather than in one Provider package because it is now needed by THREE Providers
# across TWO packages, and the pin that would have kept it private turned out to be wrong.
#
# The field failure: ``latence process`` died entering the mandatory redact stage with
#
#     ProviderError: redaction.gliner2: provider failure in 'load_model' (native AttributeError)
#     root: AttributeError: 'list' object has no attribute 'keys'
#
# raised inside ``transformers`` (>= 4.57) while the loader builds a DeBERTa-v2 tokenizer whose
# ``tokenizer_config.json`` stores ``extra_special_tokens`` as a LIST of the 10 GLiNER2 structural
# tokens — the shape an older ``transformers`` accepted. 4.57 refactored that field to a dict
# ``{name: token}`` and its loader calls ``.keys()`` / ``.items()`` on the value.
#
# The claim that has now been CORRECTED: the shim's original docstring said only
# ``fastino/GLiNER2-Guardrails-PII-Multi`` was affected, "which is exactly why only this PII
# checkpoint is affected", because the sibling ``fastino/gliner2-multi-v1`` stores ``{}``. That was
# true when written and is false now — the GLiNER 2.5 line REVERTS to the list shape, verified on
# 2026-08-25 in the downloaded
# ``models--fastino--gliner2.5-multi-v1/snapshots/*/tokenizer_config.json`` (a 10-element list), and
# it therefore hits BOTH GLiNER 2.5 Providers. A per-checkpoint claim about a shape a third party
# controls is not a thing to pin a module's location on.

#: The tokenizer-config file whose ``extra_special_tokens`` shape transformers >= 4.57 rejects as a
#: list.
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


def coerce_list_extra_special_tokens(model_id: str, *, provider_name: str) -> bool:
    """Coerce a list-valued ``extra_special_tokens`` to a dict in the checkpoint's cached config.

    Returns ``True`` iff the cached ``tokenizer_config.json`` for ``model_id`` was rewritten (its
    ``extra_special_tokens`` was a list and is now the ``{token: token}`` dict shape that
    ``transformers`` >= 4.57 requires). Returns ``False`` — never raises — when there is nothing to
    do (already a dict / absent) or the shim cannot run (no ``huggingface_hub``, offline,
    unreadable, or a read-only cache), so the caller's load proceeds unchanged.

    **Byte-identical output.** All the tokens are ALREADY registered as ``special`` added tokens in
    the checkpoint's ``tokenizer.json`` (the fast tokenizer's source of truth), so tokenization does
    not change; the shim only fixes the *shape* of a redundant config field so the loader stops
    crashing and the tokens remain model-specific special tokens. It mirrors the established
    precedent ``latence_disambig_glinker...GLinkerBackend._patch_tokenizer_config``.

    **Guarded, best-effort, never fatal** (ADR-0034 posture — no *new* silent error, only the
    removal of a crash): every failure path logs at ``debug`` and returns ``False``, and the load
    then fails, if it fails, with exactly the typed ``ProviderError`` the operator already saw.
    **Idempotent**: it rewrites only when the value is actually a list, so an already-dict config —
    or a re-run after the first patch — is a no-op.

    Dep-light (ADR-0016): ``huggingface_hub`` is imported INSIDE, on use, so importing this module
    for Provider discovery never requires it.
    """
    try:
        import json  # noqa: PLC0415 - stdlib, imported on use
        import os  # noqa: PLC0415
        import shutil  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        from huggingface_hub import hf_hub_download  # noqa: PLC0415 - heavy dep, resolved on use
    except ImportError:
        # ``huggingface_hub`` is not installed (it ships transitively with ``gliner2[local]``); with
        # no gliner2 stack there is no load to protect. The load path surfaces the missing dep.
        _log.debug(
            "%s: huggingface_hub unavailable; skipping the extra_special_tokens shape shim",
            provider_name,
        )
        return False

    try:
        config_path = hf_hub_download(model_id, TOKENIZER_CONFIG_FILE)
    except Exception as exc:  # noqa: BLE001 - offline / private / unresolved: best-effort, never fatal
        _log.debug(
            "%s: could not resolve %s for %s; skipping the shape shim (%s)",
            provider_name,
            TOKENIZER_CONFIG_FILE,
            model_id,
            exc,
        )
        return False

    try:
        with open(config_path, encoding="utf-8") as handle:  # noqa: PTH123 - plain path from the hub
            config: dict[str, Any] = json.load(handle)
    except (OSError, ValueError) as exc:
        _log.debug(
            "%s: could not read %s; skipping the shape shim (%s)", provider_name, config_path, exc
        )
        return False

    extra = config.get("extra_special_tokens")
    if not isinstance(extra, list):
        # Already a dict, or absent: nothing to fix. Idempotent no-op.
        return False

    # Preserve every string token as a model-specific special token keyed by itself — the exact dict
    # shape transformers 4.57 iterates. Non-str entries (there are none in the real checkpoints) are
    # dropped rather than smuggled through the isinstance(value, (str, AddedToken)) check the loader
    # would reject.
    coerced = {token: token for token in extra if isinstance(token, str)}
    config["extra_special_tokens"] = coerced

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=os.path.dirname(config_path))  # noqa: PTH120
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        # Atomic same-filesystem rename over the cached file (replacing the snapshot symlink with a
        # real, patched file); readers see either the old or the new config, never a torn one.
        shutil.move(tmp_path, config_path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)  # noqa: PTH108
        _log.debug(
            "%s: could not write the patched %s (read-only cache?); skipping the shape shim (%s)",
            provider_name,
            TOKENIZER_CONFIG_FILE,
            exc,
        )
        return False

    _log.info(
        "%s: coerced list-valued extra_special_tokens (%d tokens) to the dict shape transformers "
        ">= 4.57 requires in %s's cached %s — byte-identical tokenization, load no longer crashes",
        provider_name,
        len(coerced),
        model_id,
        TOKENIZER_CONFIG_FILE,
    )
    return True


def load_gliner2_model(
    import_model_cls: Callable[[], Any],
    model_id: str,
    config: Mapping[str, Any],
    decision: DeviceDecision,
    *,
    provider_name: str,
    guard: LoadGuard | None = None,
) -> Any:
    """Load a ``gliner2``-framework model, ready to run — the WHOLE sequence, one call.

    Same interface as :func:`load_gliner_model` (see it for the parameter contract); the difference
    is the native family, and it is entirely internal:

    * the ``gliner2`` framework's ``from_pretrained`` **accepts** native ``quantize`` (fp16) and
      ``compile`` (``torch.compile`` + fused kernels) kwargs, so the load goes through
      :func:`load_gliner2_optimized` (guarded: a quantize/compile fault degrades to a plain load)
      rather than the load-then-cast path — and there is no separate :func:`maybe_compile`, which
      would compile a model gliner2 already compiled.
    * the token window is a per-call native ``max_len`` argument on ``extract_entities`` /
      ``batch_extract_entities``, not a ``model.config`` attribute, so it is threaded by the caller
      at forward time instead of stamped on the model here.

    **The two knobs now key off the RESOLVED device, not a bare CUDA probe** — the (a) defect this
    entry point closes. ``fused_entity_relation.gliner2``, the Provider carrying the heaviest Stage,
    bypassed the device seam completely: it probed ``cuda_available()`` directly, so
    ``device: cuda``
    was a no-op and, worse, ``device: cpu`` on a GPU box was silently overridden — it still resolved
    ``quantize=True`` and still moved the model to cuda. That is #191's failure class reappearing.
    Here ``quantize``/``compile`` resolve against ``decision.device`` and cuda placement happens
    **iff** the resolved device is cuda, so an operator's ``device: cpu`` is honoured.
    """
    device = _require_device(decision, provider_name)
    model_cls = _import_backbone(import_model_cls, config, provider_name)
    model = _typed_load(
        lambda: load_gliner2_optimized(
            model_cls.from_pretrained, model_id, device, config, provider_name=provider_name
        ),
        model_id,
        provider_name,
        guard,
    )
    # Best-effort (never raises), unlike the plain-gliner placement: ``from_pretrained``/
    # ``quantize`` may already have placed the model, the GLiNER2 wrapper's placement API is
    # pod-validated rather than contractual, and a wrapper that exposes no ``.to`` is not a fault.
    # What it is NOT any more
    # is keyed off a bare availability probe.
    ensure_gliner2_on_cuda(model, cuda=device == "cuda", provider_name=provider_name)
    return model


def load_transformers_model(
    import_model_cls: Callable[[], Any],
    model_id: str,
    config: Mapping[str, Any],
    decision: DeviceDecision,
    *,
    provider_name: str,
    guard: LoadGuard | None = None,
) -> Any:
    """Load a plain ``transformers`` model (VLM / MaskedLM), ready to run — one call.

    Same interface as :func:`load_gliner_model`; the native family here ACCEPTS the perf kwargs, so
    the load spreads :func:`load_perf_kwargs` (bf16-on-GPU + ``sdpa``/FA2) into ``from_pretrained``
    through :func:`load_with_fa2_fallback` — a FA2 load that raises is retried once on ``sdpa``.

    The verified defect this closes: the OCR parsers (``parser.lighton``, ``parser.glm``) used a
    bare ``model = model.to(device)``, so #191's "prove it landed or raise" did not cover every
    model-placing Provider. They now get the same verified placement as everyone else.

    The JIT kill runs for this family too. A VLM backbone may or may not be
    DeBERTa-derived — but the neutralization is throughput-only and byte-identical, so applying it
    unconditionally costs nothing and removes "is this backbone mdeberta?" from the set of questions
    a Provider author has to answer correctly.
    """
    device = _require_device(decision, provider_name)
    model_cls = _import_backbone(import_model_cls, config, provider_name)
    perf = load_perf_kwargs(device, config)
    model = _typed_load(
        lambda: load_with_fa2_fallback(model_cls.from_pretrained, model_id, perf),
        model_id,
        provider_name,
        guard,
    )
    place_on_device(model, device, provider_name=provider_name)
    return maybe_compile(model, config, provider_name=provider_name)


def load_sentence_transformer(
    import_model_cls: Callable[[], Any],
    model_id: str,
    config: Mapping[str, Any],
    decision: DeviceDecision,
    *,
    provider_name: str,
    guard: LoadGuard | None = None,
    inspect_model: Callable[[Any], None] | None = None,
) -> Any:
    """Load a ``sentence-transformers`` checkpoint, ready to run — the WHOLE sequence, one call.

    The entry point ``embedding.sentence_transformers`` loads through — previously the ONE learned
    Provider assembling this module's steps piecemeal (``load_with_fa2_fallback`` + a hand-placed
    ``maybe_compile``), i.e. re-opening by hand the ordering the other three composites exist to
    own. Same interface as :func:`load_gliner_model`, with two family-specific differences:

    * ``import_model_cls`` returns the ``SentenceTransformer`` CLASS; the load is a **constructor**
      call, not ``from_pretrained``: ``st_cls(model_id, device=…, model_kwargs=…)``. ``device=`` is
      ST's own documented placement (``None`` lets ST auto-select — the shape a Provider that
      resolves no device hands in), and the W8 perf kwargs (``torch_dtype`` +
      ``attn_implementation``) thread through ST's documented ``model_kwargs`` pass-through to the
      backbone's ``from_pretrained``. A ``flash_attention_2`` load that raises is retried once on
      ``sdpa`` (:func:`load_with_fa2_fallback`) — graceful degradation, never a crash.
    * ``inspect_model`` — an optional hook run against the **eager** model after the load and
      BEFORE the guarded compile wrap (the caller's read-back/validation step, e.g. the embedder's
      ``get_sentence_embedding_dimension`` width check, which must read the container, not a
      compiled wrapper). Holding the hook here is what keeps "inspect before compile" an ordering
      a caller cannot get wrong. An exception it raises propagates unchanged.

    Behind it, in order: skip check (#191, a skipped decision is a loud ConfigError) → intra-op
    thread cap (W9-scaling; process-wide, so the embedding Stage caps like every other learned
    family) → the backbone import (the factory's ``ImportError`` propagates unwrapped so the
    "pip install …" hint survives) → perf-kwargs resolution → guarded native load with FA2→sdpa
    fallback → ``inspect_model`` on the eager model → guarded ``torch.compile`` (off by default).

    **Deliberately NO :func:`neutralize_torch_jit` here**, unlike the other three composites: the
    kill exists for FM-NVRTC-JIT on ``@torch.jit.script``-ed DeBERTa helpers, and the ST family's
    default backbones (Granite r2 = ModernBERT) carry none — pulling the process-wide
    ``PYTORCH_JIT=0`` lever from this family would add exactly the unowned cross-provider global
    mutation Wave 3-H exists to reduce. A DeBERTa-backed ST checkpoint on Blackwell/CUDA-13 can pin
    ``torch_jit: off`` via its own Provider; the lever stays one config line away.
    """
    device = _require_device(decision, provider_name)
    cap_intra_op_threads(resolve_intra_op_threads(config), provider_name=provider_name)
    st_cls = import_model_cls()
    perf = load_perf_kwargs(device, config)
    model = _typed_load(
        lambda: load_with_fa2_fallback(
            lambda mid, **mk: st_cls(mid, device=device, model_kwargs=mk),
            model_id,
            perf,
        ),
        model_id,
        provider_name,
        guard,
    )
    if inspect_model is not None:
        inspect_model(model)
    return maybe_compile(model, config, provider_name=provider_name)
