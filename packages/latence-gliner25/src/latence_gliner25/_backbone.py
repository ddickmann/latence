"""The one loader seam and the shared constants both GLiNER 2.5 Providers sit on (ADR-0064).

Why this module exists at all: the two Providers in this package differ in what they *ask* the
model (entity types vs. PII types, plus relations) and in what they *do* with the answer (KG
records vs. masked chunks). Everything between — the deferred import, the compat gate, the load
choreography, the token-window bound and the micro-batch envelope — is one thing, and one thing is
one module. It is deliberately NOT a re-implementation of
:func:`latence_core.providers.perf.load_gliner2_model`: that seam already owns the ordered
sequence (skip-decision → TorchScript-JIT kill → intra-op thread cap → backbone import → guarded
native quantize/compile load → cuda placement) and it takes the loader class as a **factory**, so
GLiNER 2.5 reuses it verbatim by handing it ``AutoExtractor`` instead of ``GLiNER2``. The #190 JIT
neutralization and the many-core thread cap therefore apply to 2.5 for free; nothing here forks
them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from latence_core.errors import ConfigError, ProviderError
from latence_core.providers.perf import (
    DEFAULT_GLINER_MAX_LEN,
    coerce_list_extra_special_tokens,
    load_gliner2_model,
    resolve_max_len,
    resolve_microbatch_bounds,
)
from latence_core.setup.recipe import MDEBERTA_TOKEN_EXPANSION, model_window_for

from latence_gliner25._compat import (
    require_auto_extractor,
    require_batched_api,
    require_boundary_model,
    require_supported_gliner2,
)

if TYPE_CHECKING:
    from latence_core.providers.device import DeviceDecision
    from latence_core.providers.perf import LoadGuard

_log = logging.getLogger("latence.provider.gliner25")

#: The default checkpoint for BOTH Providers in this package.
#:
#: Licensing (ADR-0012 / the research-diligence directive — weights AND code, verified
#: SEPARATELY, recorded 2026-08-25):
#:   * code — the ``gliner2`` library is **Apache-2.0**: the 2.0.0 wheel's own METADATA carries
#:     ``License-Expression: Apache-2.0`` and bundles ``LICENSE`` (read from the unpacked wheel in
#:     this repo's working scratch, and matching github.com/fastino-ai/GLiNER2).
#:   * weights — ``fastino/gliner2.5-multi-v1`` is **Apache-2.0** per its Hugging Face model card,
#:     ungated. This one was NOT re-read from a local artifact at the time of writing; it is the
#:     fact carried into this package from the task's verified ground truth. It is cited in the
#:     ProviderProfile with its date, and re-verifying it is a `hf` model-card read away.
#: 287M parameters, mDeBERTa-v3-base backbone, multilingual, ``max_len=4096`` per the card.
DEFAULT_MODEL = "fastino/gliner2.5-multi-v1"

#: The token window the model card ADVERTISES. It is the library/checkpoint ceiling — the largest
#: ``max_len`` this package will accept at all — and it is emphatically NOT a window of reliable
#: extraction. See :data:`SAFE_MAX_LEN`.
MAX_SUPPORTED_MAX_LEN = 4096

#: The chunk budget everything else here is sized around — and the measurement that CONFIRMS it.
#:
#: **The measurement** (pod GPU, RTX 2000 Ada 16 GB, driver 580.159.04, torch
#: 2.4.1+cu124, transformers 4.57.6, gliner2 2.0.0, ``fastino/gliner2.5-multi-v1``, 2026-08-25).
#: One call at ``max_len=4096`` over a document seeded with a repeated PII sentence, three labels,
#: threshold 0.4 — ``coverage`` is the fraction of the document the returned spans reach:
#:
#: =======  =======  =========  ==============  =========
#: tokens   chars    pii_spans  max_char_seen   coverage
#: =======  =======  =========  ==============  =========
#:    674     2312          24           2310       100%
#:   1262     4335          30           4293        99%
#:   1850     6358          27           5738        90%
#:   2438     8381          13           2869        34%
#:   3026    10404          25           7493        72%
#:   3950    13583          32           9227        68%
#:   4874    16762          32           9227        55%
#:   6050    20808          32           9227        44%
#:   7562    26010          32           9227        35%
#: =======  =======  =========  ==============  =========
#:
#: At 7,562 tokens the document holds 90 SSNs and the model returns 32 spans covering the first
#: 35% of the text. It raises nothing and returns a well-formed result. Two further modes on the
#: same hardware: 12,612 tokens with a single label returned ZERO spans, silently; and longer
#: inputs with three labels hit ``torch.OutOfMemoryError`` inside DeBERTa's
#: ``disentangled_attention_bias`` (2.14 GiB requested, 1.63 GiB free) — disentangled attention is
#: quadratic, so one long chunk OOMs a 16 GB card outright.
#:
#: **The filler is synthetic repetitive prose, so the percentages are directional, not a
#: benchmark.** What is not in doubt is the failure MODE: overflow is silent, partial and
#: NON-MONOTONIC (2,438 tokens scored worse than 3,026), which is strictly worse than a hard
#: truncation because no threshold test catches it reliably — and that useful recall ends far
#: below the advertised 4096.
#:
#: **The number is an OPERATOR DECISION, and the table above confirms it rather than choosing it.**
#: The reliable chunk budget is **1024 tokens** — the windowed-fallback default of ``chunk.page``
#: and the value the wizard emits for the gliner2.5 profile. It is not read off the curve as
#: "still acceptable at 1850"; the curve's job is to show that 1024 is comfortably inside the
#: region where coverage was 99–100% and that the region ends well below the advertised 4096.
SAFE_CHUNK_BUDGET = 1024

#: The window that COVERS :data:`SAFE_CHUNK_BUDGET`, produced by the framework's own derivation
#: rather than typed as a literal: the chunker counts in its OWN tokenizer, ~1.2x fewer tokens than
#: the mdeberta encoder, so the model-side bound must be derived from the chunk budget or it
#: silently truncates. ``model_window_for`` is that derivation and it already exists — reusing it
#: (rather than mirroring 1280) is what keeps the Provider default and the wizard's emitted
#: ``max_len`` incapable of drifting apart. Importing it is dep-light and one-directional: the
#: setup module is pure-Python and deliberately depends on no Provider, so nothing circles back.
SAFE_MAX_LEN = model_window_for(SAFE_CHUNK_BUDGET)

#: The default ``max_len`` these Providers thread into every native call: the window covering the
#: decided 1024-token budget, never the advertised 4096. It is 1.67x the 768 the gliner2 1.x
#: Providers default to — which is the real benefit of the 2.5 line — and going beyond it is an
#: operator's explicit, warned-about choice on extraction and is REFUSED outright on redaction
#: (see :func:`resolve_gliner25_max_len`).
DEFAULT_MAX_LEN = SAFE_MAX_LEN

#: Pod-measured chunker-token → mdeberta-token expansion, taken from the framework's own constant
#: rather than mirrored: a 768-chunker-token cap measured mdeberta median 666 / MAX 921, so 1.2 is
#: the WORST observed ratio — the right one for a truncation boundary, which has to cover the worst
#: chunk rather than the typical one. :func:`require_chunk_fits_window` inverts exactly the
#: derivation :data:`SAFE_MAX_LEN` applies, which is why a wizard-emitted stack cannot trip it.
CHUNKER_TOKEN_EXPANSION = MDEBERTA_TOKEN_EXPANSION


def resolve_gliner25_max_len(
    config: dict[str, Any], *, fail_open: bool, provider_name: str, model_id: str
) -> int:
    """The configured token window — defaulted to the MEASURED-safe one, bounded two ways.

    Reuses the shared :func:`~latence_core.providers.perf.resolve_max_len` knob (so ``max_len``
    means the same thing here as on every other gliner-family Provider) and adds the two facts that
    are specific to this checkpoint family:

    * **the library ceiling.** Above :data:`MAX_SUPPORTED_MAX_LEN` the native call truncates
      anyway, so accepting the value would be a config that lies — a
      :class:`~latence_core.errors.ConfigError` says so instead (ADR-0034).
    * **the measured-safe window.** Between :data:`SAFE_MAX_LEN` and the ceiling, extraction was
      measured to degrade silently, partially and non-monotonically (see :data:`SAFE_MAX_LEN`).
      What happens there depends on what the Provider's failure would COST, which is what
      ``fail_open`` names:

      - ``fail_open=True`` (Redaction) — **refused.** An over-long chunk makes the Stage emit
        "redacted" output that still contains PII, with no error anywhere in the run. That is
        fail-open on a security control, and it is not an operator's to opt into by accident; the
        ADR-0044 floor still catches the deterministic financial set, but nothing catches a name or
        an address. Refused only for the DEFAULT checkpoint, because the measurement is about that
        checkpoint: an operator-supplied ``model`` gets a loud warning instead, same shape as the
        ``guardrails`` switch.
      - ``fail_open=False`` (Extraction) — **allowed, with a WARNING naming the measurement.**
        Losing entities is a recall loss the operator can see in the output and choose to accept;
        it is not a silent safety failure.
    """
    max_len = resolve_max_len(config, DEFAULT_MAX_LEN)
    if max_len > MAX_SUPPORTED_MAX_LEN:
        msg = (
            f"{provider_name}: max_len {max_len} exceeds the GLiNER 2.5 checkpoint window "
            f"({MAX_SUPPORTED_MAX_LEN}); the model truncates past it, so a larger window is a "
            "boundary that is not there. Lower max_len, or point 'model' at a checkpoint with a "
            "wider window and say so."
        )
        raise ConfigError(msg)
    if max_len <= SAFE_MAX_LEN:
        return max_len
    detail = (
        f"max_len {max_len} is above the MEASURED-safe window ({SAFE_MAX_LEN}). The card "
        f"advertises {MAX_SUPPORTED_MAX_LEN}, but a real run (RTX 2000 Ada, gliner2 2.0.0, "
        f"{DEFAULT_MODEL}, 2026-08-25) showed extraction degrading SILENTLY and NON-MONOTONICALLY "
        "past ~1850 tokens — 90 SSNs in a 7.5k-token document returned 32 spans covering the "
        "first 35% of the text, with no error and a well-formed result"
    )
    if not fail_open:
        _log.warning(
            "%s: %s. Entities past the model's effective reach will be missed and nothing will "
            "say so; lower max_len (and the chunk budget with it) if recall matters more than "
            "chunk size",
            provider_name,
            detail,
        )
        return max_len
    if model_id != DEFAULT_MODEL:
        _log.warning(
            "%s: %s. You have pointed 'model' at %r, which that measurement does not describe — "
            "the window is accepted on your evidence, not this package's",
            provider_name,
            detail,
            model_id,
        )
        return max_len
    msg = (
        f"{provider_name}: {detail}. For a REDACTOR that is fail-open: the Stage would emit "
        "'redacted' chunks that still contain PII, with nothing in the run reporting it, and the "
        "ADR-0044 deterministic floor only covers the financial set — not a name or an address. "
        f"Set max_len <= {SAFE_MAX_LEN} (and size the chunk budget to match, e.g. "
        "chunk.page/chunk.markdown max_tokens 1024). If you have measured a wider window on your "
        "own checkpoint, set 'model' to it and this becomes a warning."
    )
    raise ConfigError(msg)


def require_chunk_fits_window(
    *, token_count: int, record_id: str, max_len: int, provider_name: str
) -> None:
    """Refuse a chunk the model cannot see all of — loudly, before it is silently truncated.

    The other half of the same measurement. Bounding ``max_len`` bounds what the model is ASKED to
    read; this bounds what it is HANDED. A chunk longer than the window is truncated by gliner2
    without an error, which for extraction loses the tail's entities and for redaction leaves the
    tail's PII in the exported corpus — the fail-open case, and the reason this refuses rather
    than trimming: a Capability that cannot be served is a loud error, never a quiet partial
    result, and segmentation belongs to the Chunk Stage (ADR-0021), not to a Provider quietly
    re-doing it.

    The estimate is the chunk's OWN ``token_count`` — the chunker's budgeted count — scaled by the
    pod-measured :data:`CHUNKER_TOKEN_EXPANSION`, which is exactly the arithmetic
    ``latence_core.setup.recipe.model_window_for`` inverts when the wizard derives ``max_len`` from
    the chunk budget. A wizard-emitted stack therefore never trips this by construction; a
    hand-written stack whose chunk budget outgrew its model window trips it on the first chunk
    rather than in an audit six months later. A chunk reporting ``token_count == 0`` (a
    hand-built record) is let through: the guard refuses on evidence, never on its absence.
    """
    if token_count <= 0:
        return
    if token_count * CHUNKER_TOKEN_EXPANSION <= max_len:
        return
    needed = -(-int(token_count * CHUNKER_TOKEN_EXPANSION) // 64) * 64
    msg = (
        f"{provider_name}: chunk {record_id} carries {token_count} chunker tokens, which reach "
        f"~{int(token_count * CHUNKER_TOKEN_EXPANSION)} mdeberta tokens at the pod-measured "
        f"{CHUNKER_TOKEN_EXPANSION}x expansion — past this Provider's max_len of {max_len}. "
        "gliner2 truncates a longer input SILENTLY, so the tail would be scanned by nothing and "
        "reported by nothing. Lower the chunk budget (chunk.page / chunk.markdown 'max_tokens') "
        f"so it fits, or raise 'max_len' to at least {needed} — but read the measured-safe window "
        f"({SAFE_MAX_LEN}) first: above it, extraction degrades silently and non-monotonically."
    )
    raise ProviderError(msg)


def resolve_window_safe_microbatch_bounds(
    config: dict[str, Any], *, max_len: int
) -> tuple[int, int]:
    """Micro-batch bounds that do NOT widen the padded forward when ``max_len`` is raised.

    The trap this closes, stated once: gliner2 pads every text in a micro-batch up to the batch's
    longest sequence, so ONE forward's padded activation is ``max_texts x max_len`` positions. The
    shared :func:`~latence_core.providers.perf.resolve_microbatch_bounds` derives its default
    ``max_texts`` from the detected card against the 768-window reference point (4 texts x 768 =
    3072 positions peaked at 21.6 GiB on the 32 GB pod). Taking that default unchanged under a
    1280-token window would silently enlarge the padded forward by 1.67x — "raising max_len without
    lowering max_texts" is an OOM shipped as a default.

    So: the shared resolver still answers (explicit ``microbatch_max_texts`` /
    ``microbatch_token_budget`` knobs win, and the VRAM derivation still lowers the envelope on a
    smaller card), and its answer is then re-expressed in POSITIONS and divided by the actual
    window. The envelope is unchanged from the 1.x default; only its shape moves. Nobody has
    measured the boundary architecture's peak, so nothing here assumes it is cheaper — the
    conservative reading is the honest one, and the reactive CUDA-OOM halving in the Providers is
    still the last-resort net.

    An explicitly configured knob is returned untouched in both directions: an operator with a
    measurement outranks a derivation.
    """
    max_texts, token_budget = resolve_microbatch_bounds(config)
    # The padded-position envelope the shared defaults imply at the REFERENCE window.
    envelope = max_texts * DEFAULT_GLINER_MAX_LEN
    if "microbatch_max_texts" not in config:
        max_texts = max(1, envelope // max(1, max_len))
    if "microbatch_token_budget" not in config:
        token_budget = max_texts * max_len
    return max_texts, token_budget


#: How the compat gate names itself in an operator-facing error. ONE label for both Providers,
#: because :func:`import_auto_extractor` is a zero-argument module-level function rather than a
#: per-Provider closure — see its docstring for why that shape is load-bearing — and because the
#: two Providers share one package, one pin and one library, so "which of the two asked" is not
#: information an operator needs to act.
PACKAGE_LABEL = "latence-gliner25 (fused_entity_relation.gliner25 / redaction.gliner25)"


def import_auto_extractor() -> Any:
    """The deferred backbone-import factory the shared load seam calls (ADR-0016 / #190).

    This function is HANDED to :func:`~latence_core.providers.perf.load_gliner2_model`, which calls
    it only after neutralizing the TorchScript JIT and capping the intra-op threads. That ordering
    is the whole point: importing ``gliner2``'s model classes transitively evaluates transformers'
    ``deberta_v2`` module, whose relative-position helpers are compiled by ``@torch.jit.script`` AT
    IMPORT and can never be un-scripted afterwards — on a Blackwell/CUDA-13 box they then die in
    nvrtc on the FIRST forward (#190). Handing the seam a factory turns "neutralize before import"
    from a rule each Provider restates into a straight line of code inside the seam that no caller
    can reorder.

    **Why it is a module-level zero-argument function rather than a closure over the provider
    name.** ``test_perf_fp32_regression_guard.py`` reads this package's AST and exempts a function
    only when it is passed BY NAME as the first argument of a perf-seam load entry point. A
    ``load_gliner2_model(make_factory(provider_name=...), ...)`` call is a factory-of-a-factory the
    guard cannot follow, so the import would read as unguarded. Rather than weaken the guard for
    one package, the factory takes the shape the guard can verify, and the provider name it would
    have carried becomes :data:`PACKAGE_LABEL` — one label for a family that shares one pin.

    The version gate runs HERE — after ``import gliner2`` (torch-free: the package keeps its model
    classes behind a lazy ``__getattr__``) and BEFORE ``AutoExtractor`` is touched — so an
    unsupported install is a typed ProviderError before a single weight is fetched.
    """
    try:
        import gliner2
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        msg = (
            "The 'gliner2' package is required to run the GLiNER 2.5 Providers. "
            "Install it with: pip install latence-gliner25"
        )
        raise ImportError(msg) from exc
    require_supported_gliner2(gliner2, provider_name=PACKAGE_LABEL)
    return require_auto_extractor(gliner2, provider_name=PACKAGE_LABEL)


def load_gliner25_model(
    model_id: str,
    config: dict[str, Any],
    decision: DeviceDecision,
    *,
    provider_name: str,
    methods: tuple[str, ...],
    guard: LoadGuard | None = None,
) -> Any:
    """Load a GLiNER 2.5 checkpoint, ready to run — the shared seam plus this family's checks.

    The load itself is :func:`~latence_core.providers.perf.load_gliner2_model`, unchanged and
    unforked: it takes the loader class as a factory, and ``AutoExtractor.from_pretrained`` accepts
    the same native ``quantize``/``compile`` kwargs ``GLiNER2.from_pretrained`` did (both are
    members of ``gliner2.models.loading.MODEL_LOAD_OPTIONS`` in 2.0.0), so the guarded optimized
    load, the cuda placement keyed off the RESOLVED device, the JIT kill and the thread cap all
    apply to 2.5 without a line of new choreography.

    What is added afterwards is only what is specific to this family, and both are checked at LOAD
    rather than at the first forward so they are typed and actionable up front (ADR-0034):

    * the loaded object really is a ``boundary`` extractor — i.e. the operator did not point
      ``model`` at a 1.x ``span`` checkpoint, which would load fine and quietly be a different
      model (:func:`~latence_gliner25._compat.require_boundary_model`);
    * it exposes the batched entry points this Provider drives
      (:func:`~latence_gliner25._compat.require_batched_api`) — the #192 field failure, closed the
      same way for this family.
    """
    # MEASURED on a real load, 2026-08-25: the GLiNER 2.5 checkpoints store
    # ``extra_special_tokens`` as a LIST in their ``tokenizer_config.json``, which transformers
    # >= 4.57 rejects with ``AttributeError: 'list' object has no attribute 'keys'`` deep inside
    # ``_set_model_specific_special_tokens`` — the load dies before a single forward. This is the
    # exact field failure ADR-0034 already records for the 1.x PII checkpoint, and the shim that
    # fixes it now lives in the shared perf seam rather than in one Provider package, because the
    # belief that only that one checkpoint was affected turned out to be a claim about a third
    # party's file. Guarded, best-effort, idempotent, byte-identical tokenization: an
    # already-dict-shaped config is a no-op, and every failure path leaves the load exactly as it
    # was. Both Providers in this package need it, so it is applied HERE, once.
    coerce_list_extra_special_tokens(model_id, provider_name=provider_name)
    model = load_gliner2_model(
        import_auto_extractor,
        model_id,
        config,
        decision,
        provider_name=provider_name,
        guard=guard,
    )
    require_boundary_model(model, provider_name=provider_name, model_id=model_id)
    require_batched_api(model, provider_name=provider_name, methods=methods)
    return model


def warn_if_relations_disabled(model: Any, *, provider_name: str) -> None:
    """Log LOUDLY when a loaded boundary model cannot decode relations at all.

    ``BoundaryExtractor._decode_relations`` opens with ``if not getattr(self, "enable_relations",
    False) ... return {}`` — and ``format_results`` then fills every requested relation type with
    an empty list. A checkpoint trained with ``enable_relations=False`` therefore answers a
    relation query with a well-formed, entirely empty result: no exception, no warning, zero edges.
    That is the silent-wrong-output class this repository refuses, so it is surfaced.

    A WARNING rather than a refusal, deliberately: the attribute is an internal marker, a model
    object that does not expose it is not a fault, and a fused Provider whose relation half is dead
    is still a correct entity extractor. The loud log plus the fused Provider's own zero-relation
    reporting is what makes it visible; making it fatal would fail a run over an unverified
    introspection.
    """
    if getattr(model, "enable_relations", True):
        return
    _log.warning(
        "%s: the loaded GLiNER 2.5 checkpoint reports enable_relations=False — its relation head "
        "is not active, so EVERY relation query will return an empty result with no error. "
        "Relation extraction from this Provider will emit zero relations; use a "
        "relation-enabled checkpoint, or run a separate Relation Extraction Provider",
        provider_name,
    )
