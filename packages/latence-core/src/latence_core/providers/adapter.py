"""``AdapterBase`` — the thin, **optional** base every provider adapter reuses (ADR-0036 §1).

A **Provider is an adapter** over a native library (CONTEXT ``Provider``); the Capability
Protocol (ADR-0004) stays the stable I/O contract. Every adapter repeats the same
cross-cutting glue — pick a device, batch its inputs, turn a native library exception into
a **typed** :class:`~latence_core.errors.LatenceError`, and carry
:class:`~latence_core.contracts.Provenance` / :class:`~latence_core.contracts.Classification`
+ character offsets through the native call. P3-F2 factors that glue into one **optional**
:class:`AdapterBase` so a Wave-1 adapter is small and consistent, and those concerns are done
**one right way** instead of five subtly-different ways.

Non-negotiable framing (issue P3-F2):

* **Optional, not a new seam.** :class:`AdapterBase` is a convenience base. A Provider may
  still satisfy a Capability Protocol structurally without it — the shipping in-core
  reference Providers do, and are NOT force-migrated. No change to the Capability protocols;
  ADR-0036 rejected a second protocol layer.
* **Dep-light (ADR-0016).** This lives in ``latence-core`` and pulls in **no** heavy deps
  (no torch, no model libs). It only orchestrates + maps; the adapter subclass owns its
  native dep. Device *placement* stays the adapter's job (it moves its own model, ADR-0003);
  :class:`AdapterBase` only resolves the :class:`~latence_core.providers.device.DeviceDecision`
  and hands it over.

The four reusable pieces (compose them; the base does not force a god-class shape):

1. :meth:`AdapterBase.device` — resolve ``config.device`` (``auto|cpu|cuda``) against the
   subclass's declared :class:`~latence_core.providers.profile.ProviderProfile` via P3-F1's
   :func:`~latence_core.providers.device.select_device`, exposing the chosen device (or the
   **skip** decision) to the subclass. A ``compute="gpu"`` adapter on a CPU-only host sees the
   skip decision, never a fabricated device.
2. :meth:`AdapterBase.batched` — group an input *iterable* into ``config.batch_size`` chunks
   and call the subclass's ``_run_batch`` when the profile is batch-capable, degrading to
   one-item calls otherwise. **Deterministic input-order output** regardless of batching, and
   **no whole-corpus buffering** (the streaming Runner's friend, ADR-0033).
3. :meth:`AdapterBase.guard` — a context manager that catches an exception raised **inside the
   native call** and re-raises it as the right :class:`~latence_core.errors.LatenceError`
   subtype (a genuine provider/runtime failure → :class:`~latence_core.errors.ProviderError`;
   an operator bad-config → :class:`~latence_core.errors.ConfigError`), tagged with the
   provider name + op and **carrying no record content / PII**. A framework
   :class:`~latence_core.errors.LatenceError` passes through unwrapped.
4. :meth:`AdapterBase.carry_provenance` + :meth:`AdapterBase.rebase_offsets` +
   :meth:`AdapterBase.resolve_pages` — map a native library's output back onto the input's
   Provenance/Classification and character offsets, reusing the existing
   :class:`~latence_core.offsetmap.OffsetIndex` seam and the document-scoped
   :class:`~latence_core.page_index.PageIndexResolver` seam (ADR-0031) — never a weaker
   hand-rolled offset mapper, and never a throwaway per-chunk page index — so the
   "offsets/page-provenance intact" invariant is the default.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar, cast, runtime_checkable

from latence_core.errors import ConfigError, LatenceError, ProviderError
from latence_core.offsetmap import OffsetIndex
from latence_core.providers.device import DeviceDecision, select_device
from latence_core.providers.perf import load_perf_kwargs as _resolve_perf_kwargs
from latence_core.providers.perf import maybe_compile as _maybe_compile
from latence_core.providers.profile import ProviderProfile, profile_of

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from latence_core.contracts import (
        ChunkRecord,
        Classification,
        OffsetMap,
        Provenance,
    )
    from latence_core.page_index import PageIndexResolver

# The default batch size when a batch-capable adapter's config supplies none. Small,
# sensible, and (like every batching knob) purely memory-shaping — the outputs are
# input-order identical for any positive batch size (ADR-0033).
DEFAULT_BATCH_SIZE = 32

_In = TypeVar("_In")
_Out = TypeVar("_Out")


def validate_batch_size(raw: object, *, provider_name: str) -> int:
    """Validate a ``config.batch_size`` value to a positive int, or raise ConfigError (W7).

    The shared, dep-light validator the learned ENCODER providers reuse so a ``batch_size`` knob
    is checked **one right way** whether or not the provider subclasses :class:`AdapterBase`
    (:meth:`AdapterBase.batch_size` delegates here). A ``batch_size`` must be an ``int >= 1`` — a
    batch of zero would never make progress, and a ``bool``/float/str is a config typo, not a
    valid size — so a non-positive or non-int value is an operator
    :class:`~latence_core.errors.ConfigError` naming the provider, never a silent fallback.
    """
    if isinstance(raw, bool) or not isinstance(raw, int):
        msg = f"{provider_name}: batch_size must be an int >= 1, got {raw!r}"
        raise ConfigError(msg)
    if raw < 1:
        msg = f"{provider_name}: batch_size must be >= 1, got {raw}"
        raise ConfigError(msg)
    return raw


@runtime_checkable
class ProvenanceCarrier(Protocol):
    """The structural source :meth:`AdapterBase.carry_provenance` reads: anything with a frozen
    :class:`~latence_core.contracts.Provenance` + :class:`~latence_core.contracts.Classification`.

    P3-F2's helper originally typed its source as a :class:`~latence_core.contracts.Record`, but
    the FIRST real adapter over it (H-A1, the pdfplumber Parser) exposed a genuine misfit: a
    Parser's source is a :class:`~latence_core.contracts.ParserInput` (a *raw document*, ADR-0019),
    which carries Provenance + Classification but is **not** a ``Record`` subclass — so a real
    Parser could not reuse the helper for a whole-document carry without a hand-rolled cast (the
    exact "weaker mapper" the audit forbids). Widening the source to this structural Protocol is
    the minimal, backward-compatible fix: every ``Record`` satisfies it (it has both attributes),
    and so does a ``ParserInput``, so both the sub-document carriers (mentions/chunks, ``Record``s)
    and the whole-document Parser carry through the *same* base helper. The helper only ever reads
    ``.provenance`` (frozen, ``model_copy``-ed) and ``.classification`` (returned unchanged), so
    the Protocol names exactly those and nothing more.
    """

    @property
    def provenance(self) -> Provenance: ...

    @property
    def classification(self) -> Classification: ...

# A native-library exception type the adapter classifies as an operator **config** error
# rather than a runtime provider fault. The subclass declares which native exceptions mean
# "the operator handed me a bad config" (a bad model_id, a bad device string, an
# out-of-range parameter) so :meth:`AdapterBase.guard` maps THOSE to ConfigError and
# everything else to ProviderError. ``ValueError``/``KeyError`` are the usual bad-config
# shapes a native lib raises for a malformed argument; the default set is conservative
# (empty) so a subclass opts in explicitly and nothing is mis-classified silently.
_ConfigExc = type[Exception]


class AdapterBase:
    """Optional base carrying the cross-cutting glue every provider adapter repeats (ADR-0036 §1).

    A subclass declares its :class:`~latence_core.providers.profile.ProviderProfile` (a
    ``profile`` classmethod or a ``PROFILE`` ClassVar, the P3-F1 forms) and receives the
    Stage ``config`` dict at construction. It then composes the four helpers:

    * :meth:`device` for the resolved :class:`~latence_core.providers.device.DeviceDecision`,
    * :meth:`batched` for deterministic, streaming-friendly batching,
    * :meth:`guard` for native-exception → typed-taxonomy mapping,
    * :meth:`carry_provenance` / :meth:`rebase_offsets` for Provenance/offset carry.

    It is **not** a Capability — it names no ``parse``/``extract``/… method and imposes no
    protocol. A subclass still implements whichever Capability method it fulfils, so
    ``isinstance(provider, Parser)`` (etc.) is satisfied structurally exactly as for a
    Provider that does not use this base. The base is convenience, never a required layer.
    """

    #: Native exception types the adapter treats as an operator **config** error (mapped to
    #: :class:`~latence_core.errors.ConfigError` by :meth:`guard`). A subclass overrides this
    #: to name the native lib's bad-argument exceptions (e.g. ``(ValueError, KeyError)``);
    #: everything else caught by :meth:`guard` is a runtime :class:`ProviderError`. Empty by
    #: default so nothing is silently mis-classified — a subclass opts in.
    config_exceptions: ClassVar[tuple[_ConfigExc, ...]] = ()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config: dict[str, Any] = dict(config or {})
        # Resolve the device once at construction against the declared profile: the decision
        # is a cheap, pure routing over the profile (no model import, no torch), and a subclass
        # that must move its model reads ``self.device`` right after ``super().__init__``.
        self._device_decision = self._resolve_device()

    # -- config accessors --------------------------------------------------

    @property
    def config(self) -> dict[str, Any]:
        """The Stage ``config`` dict the Provider was constructed with (a defensive copy)."""
        return dict(self._config)

    @classmethod
    def declared_profile(cls) -> ProviderProfile | None:
        """This adapter's declared :class:`ProviderProfile`, read via :func:`profile_of`.

        Reads whichever declaration form the subclass uses (a ``profile`` classmethod or a
        ``PROFILE`` ClassVar), reusing the P3-F1 reader — never fabricating one. A subclass
        that declares none returns ``None`` (the device seam then follows the request, treating
        it as ``compute="either"``). A **malformed** declaration surfaces as ``TypeError`` from
        :func:`profile_of` — the same fail-loud contract the Runner now honors (issue P3-F2 §2).
        """
        return profile_of(cls)

    @property
    def provider_name(self) -> str:
        """A stable, human-readable provider name for error tags (never carries content).

        Defaults to the subclass's class name — used only to label a mapped
        :class:`~latence_core.errors.LatenceError` (provider + op), never to carry any record
        content or PII. A subclass may override to surface its registered entry-point name.
        """
        return type(self).__name__

    # -- 1. device resolution ----------------------------------------------

    def _resolve_device(self) -> DeviceDecision:
        """Route ``config.device`` against the declared profile (skip-with-flag, never raises).

        Reads the ``device`` config value (``auto|cpu|cuda``, default ``auto``), coerces it to
        ``str`` so a non-string typo surfaces as an "unknown device" ConfigError rather than a
        ``TypeError``, and calls the dep-light :func:`~latence_core.providers.device.select_device`
        against the declared profile. A GPU-only adapter on a CPU-only host gets the **skip**
        :class:`~latence_core.providers.device.DeviceDecision` (the G1 posture), never a fake
        device. A malformed ``device`` string is a :class:`~latence_core.errors.ConfigError`.
        """
        requested = str(self._config.get("device", "auto"))
        try:
            return select_device(requested, self.declared_profile())
        except ValueError as exc:
            # ``select_device`` raises a bare ``ValueError`` for an unrecognised device string;
            # re-raise it as the typed ConfigError (a ValueError subclass, so existing
            # ``except ValueError`` callers keep catching it) — an operator config knob, not a bug.
            msg = f"{self.provider_name}: invalid device config: {exc}"
            raise ConfigError(msg) from exc

    @property
    def device(self) -> DeviceDecision:
        """The resolved :class:`~latence_core.providers.device.DeviceDecision` for this adapter.

        The subclass reads ``self.device.device`` (``"cpu"``/``"cuda"``/``None``) to place its
        model, and ``self.device.skipped`` to know it was skipped-with-flag on a host that
        cannot run it. Device **placement** stays the adapter's job (ADR-0003) — this only
        resolves the decision.
        """
        return self._device_decision

    # -- 1b. performance seam (W8) -----------------------------------------

    def load_perf_kwargs(self) -> dict[str, Any]:
        """The load-time perf kwargs (dtype + attn) to spread into ``from_pretrained`` (W8).

        Resolves the load **dtype** (bf16 on GPU / fp32 on CPU by default, or ``config["dtype"]``
        override) and the **attention implementation** (``sdpa`` default, ``flash_attention_2`` only
        when the ``flash_attn`` package is importable and the config permits it) against this
        adapter's resolved device (:attr:`device`), returning the two kwargs HF ``from_pretrained``
        reads::

            model = Model.from_pretrained(model_id, **self.load_perf_kwargs())

        Applied once per load so a learned Provider never loads bare fp32 (the ~2× miss). An invalid
       ``dtype`` config is an operator :class:`~latence_core.errors.ConfigError` naming the provider
        never a silent wrong dtype. The FA2 *load* can still raise on a model that does not support
        it — the adapter wraps the load so that surfaces as a graceful sdpa retry, never a crash.
        """
        try:
            return _resolve_perf_kwargs(self.device.device, self._config)
        except ValueError as exc:
            msg = f"{self.provider_name}: {exc}"
            raise ConfigError(msg) from exc

    def maybe_compile(self, model: _Out) -> _Out:
        """``torch.compile`` the model when ``config["compile"]`` is truthy — guarded (W8).

        Off by default (returns the model unchanged); on, wraps the forward with
        ``torch.compile(mode="reduce-overhead")`` behind a broad guard so any compile/warmup failure
        logs and returns the **uncompiled** model — a bad compile costs only throughput, never
        correctness or a crash. The compiled module is a drop-in for the eager one (same outputs).
        """
        compiled = _maybe_compile(model, self._config, provider_name=self.provider_name)
        return cast("_Out", compiled)

    # -- 2. batching -------------------------------------------------------

    def batch_size(self) -> int:
        """The validated batch size (``config.batch_size``, default :data:`DEFAULT_BATCH_SIZE`).

        Must be ``>= 1`` — a non-positive or non-int ``batch_size`` is an operator
        :class:`~latence_core.errors.ConfigError` (a batch of zero would never make progress).
        """
        raw = self._config.get("batch_size", DEFAULT_BATCH_SIZE)
        return validate_batch_size(raw, provider_name=self.provider_name)

    def batched(
        self,
        items: Iterable[_In],
        run_batch: Callable[[list[_In]], Iterable[_Out]],
    ) -> Iterator[_Out]:
        """Group ``items`` and call ``run_batch`` per group, yielding outputs in **input order**.

        The batching helper the streaming Runner (ADR-0033) is a friend of:

        * When the declared profile is **batch-capable** (``profile.batch is True``), ``items``
          are grouped into :meth:`batch_size` chunks and ``run_batch`` is called once per chunk.
        * When it is **not** batch-capable (or no profile is declared), it degrades to
          one-item batches — ``run_batch`` is called with a single-element list at a time — so a
          non-batch adapter reuses the exact same code path.

        Ordering is **deterministic**: outputs are yielded in the order ``run_batch`` returns
        them, which for a correct adapter mirrors input order (a batch preserves its members'
        order; the batches themselves are consumed in input order). The input iterable is
        consumed **lazily one batch at a time** — the whole corpus is never buffered, so a
        streamed input stays streamed (memory bound O(batch_size), not O(corpus)).

        ``run_batch`` maps a list of inputs to an iterable of outputs; a subclass wraps its
        native call there (typically inside :meth:`guard`). This helper adds no error mapping of
        its own — the subclass owns that so the op label is precise.
        """
        size = self.batch_size() if self._is_batch_capable() else 1
        batch: list[_In] = []
        for item in items:
            batch.append(item)
            if len(batch) >= size:
                yield from run_batch(batch)
                batch = []
        if batch:
            yield from run_batch(batch)

    def _is_batch_capable(self) -> bool:
        """Whether the declared profile opts into batching (``profile.batch``).

        A profile-less adapter is treated as non-batch (one-item calls) — the conservative
        default, since batching is a declared capability, never assumed.
        """
        profile = self.declared_profile()
        return bool(profile is not None and profile.batch)

    # -- 3. native-exception → typed taxonomy (ADR-0034) -------------------

    @contextmanager
    def guard(self, op: str) -> Iterator[None]:
        """Map an exception raised **inside** the ``with`` block to the typed taxonomy (ADR-0034).

        Wrap the native library call so a native exception becomes the right
        :class:`~latence_core.errors.LatenceError`, tagged with the provider name + ``op`` and
        **carrying no record content / PII** (only the provider, the op, and the native
        exception's *type name* — never its message, which could echo an input):

        * a native exception whose type is in :attr:`config_exceptions` → a
          :class:`~latence_core.errors.ConfigError` (an operator-supplied bad config: a bad
          model_id, a bad parameter). Fix the config, not the code.
        * any other native exception → a :class:`~latence_core.errors.ProviderError` (a genuine
          provider/runtime failure). ``retryable=False`` — the adapter does not presume a native
          fault is transient; an endpoint adapter that knows better raises its own retryable
          ProviderError directly.
        * a framework :class:`~latence_core.errors.LatenceError` raised *by* the framework (e.g.
          a contract check the adapter itself made) **passes through unwrapped** — it is already
          typed and already carries the right category; re-wrapping would relabel it.

        The ``op`` is a short, static verb the adapter names (``"parse"``, ``"extract"``,
        ``"embed"``) — not derived from input — so the message is a stable, greppable label.
        Crucially the caught exception's *message* is deliberately **not** interpolated into the
        raised message (only ``type(exc).__name__``), matching the Quality Report's no-PII
        discipline: a native lib that echoes the offending input text in its error must not leak
        it into the framework error a report or log records.
        """
        try:
            yield
        except LatenceError:
            # Already a framework-typed error — it carries the right category and no content.
            # Pass it through unwrapped rather than relabelling it as a provider fault.
            raise
        except Exception as exc:  # noqa: BLE001 - deliberate boundary: map native → typed
            native_type = type(exc).__name__
            if isinstance(exc, self.config_exceptions):
                msg = (
                    f"{self.provider_name}: config error in {op!r} "
                    f"(native {native_type})"
                )
                raise ConfigError(msg) from exc
            msg = (
                f"{self.provider_name}: provider failure in {op!r} "
                f"(native {native_type})"
            )
            raise ProviderError(msg) from exc

    # -- 4. Provenance / offset carry helpers ------------------------------

    @staticmethod
    def carry_provenance(
        source: ProvenanceCarrier,
        *,
        char_start: int | None = None,
        char_end: int | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
    ) -> tuple[Provenance, Classification]:
        """Carry a source's Provenance + Classification onto a produced record.

        ``source`` is any :class:`ProvenanceCarrier` — a :class:`~latence_core.contracts.Record`
        (a sub-document mention/chunk) OR a :class:`~latence_core.contracts.ParserInput` (a Parser's
        raw-document source, ADR-0019); both carry the two fields this helper reads, so a real
        Parser reuses it for a whole-document carry rather than hand-rolling the chain (H-A1).

        The default so an adapter never hand-rolls the Provenance chain: it takes the source's
        immutable :class:`~latence_core.contracts.Provenance` and narrows only the
        offset/page fields the produced record occupies (a sub-record's own span), leaving the
        source lineage (``source_uri``/``document_id``/``file_*``) intact, and returns the
        source :class:`~latence_core.contracts.Classification` **unchanged** (Classification is
        inherited by every downstream record, CONTEXT ``Classification``).

        Only the offset/page fields that are passed are overridden; the rest of the Provenance
        is copied verbatim via ``model_copy(update=...)`` (Provenance is frozen). Passing none
        returns the source Provenance/Classification exactly (a pure carry).
        """
        update: dict[str, object] = {}
        if char_start is not None:
            update["char_start"] = char_start
        if char_end is not None:
            update["char_end"] = char_end
        if page_start is not None:
            update["page_start"] = page_start
        if page_end is not None:
            update["page_end"] = page_end
        provenance = source.provenance.model_copy(update=update) if update else source.provenance
        return provenance, source.classification

    @staticmethod
    def rebase_offsets(
        local_start: int,
        local_end: int,
        *,
        chunk_char_start: int,
        offset_map: OffsetMap | None,
    ) -> tuple[int, int]:
        """Rebase a chunk-local half-open span to true original-markdown offsets (ADR-0031).

        Reuses the existing :class:`~latence_core.offsetmap.OffsetIndex` seam — **not** a weaker
        hand-rolled mapper (issue P3-F2). Chunk ``content`` is markup-stripped, so the naive
        ``chunk_char_start + local`` shift is only a *lower bound* once markup was stripped before
        the span within its chunk; the carried :class:`~latence_core.contracts.OffsetMap` corrects
        it. When a chunk carries no offset map (a pre-v11 chunk, or a markup-free chunk), this
        degrades to the exact ``chunk_char_start + local`` shift — the identity path, unchanged.

        Returns the half-open ``[doc_char_start, doc_char_end)`` span in the parent document's
        assembled markdown, ready for :meth:`resolve_pages` and :meth:`carry_provenance`.
        """
        if offset_map is not None:
            index = OffsetIndex(offset_map)
            return index.resolve_span(local_start, local_end)
        return chunk_char_start + local_start, chunk_char_start + local_end

    @staticmethod
    def resolve_pages(
        chunk: ChunkRecord,
        doc_char_start: int,
        doc_char_end: int,
        *,
        resolver: PageIndexResolver,
    ) -> tuple[int | None, int | None]:
        """Resolve a document-offset span to its source page(s) via the page seam (ADR-0031).

        Goes through the document-scoped :class:`~latence_core.page_index.PageIndexResolver` —
        the ONE seam that owns offset→page resolution — so a produced sub-record cites **its
        own** page, not merely the parent chunk's inherited range. The resolver owns the
        half-open end clamp, the per-document memoisation of the underlying never-crashes
        :class:`~latence_core.pagemap.PageOffsetIndex`, and the corpus-wide drift/degradation
        counters; this helper adds exactly the one thing a record-producing adapter needs on
        top: the degrade to the source chunk's inherited page range when no map is available.

        **Pass ONE resolver per Capability call, and ``resolver.observe(chunk)`` every chunk
        you are handed, before any filter** — since schema v18 a document's ``page_map`` rides
        its FIRST chunk only, and that chunk is very often the one an adapter's ``min_chars``
        (or term-match) filter drops. A per-chunk resolver, or one that only ever sees the
        interesting chunks, silently degrades every later chunk of the document to its
        inherited range: a plausible-looking WRONG page (audit R2)::

            resolver = PageIndexResolver()          # one per extract() call
            for chunk in chunks:
                resolver.observe(chunk)             # BEFORE deciding it is uninteresting
                ...
                pages = self.resolve_pages(chunk, doc_start, doc_end, resolver=resolver)
        """
        pages = resolver.pages_for_span(chunk, doc_char_start, doc_char_end)
        if pages is None:
            # No page map available for this document at all — inherit the chunk's range. The
            # resolver has COUNTED the degradation (``resolver.unresolved``, ADR-0034); this is
            # not a silent fallback.
            return chunk.provenance.page_start, chunk.provenance.page_end
        return pages
