"""The Stage-contribution seam behind ``latence setup`` (C7, on top of W15/ADR-0041).

Before this module the guided-setup generator knew every Stage and every Stage knew nothing: two
parallel procedural scripts (``_build_lite_stages`` / ``_build_sota_stages``, ~590 lines) emitted
the SAME 13 Stage blocks with small deltas, twin private helpers shadowed each other, and a THIRD
hand-maintained transcription of the same knowledge lived in the comment-header renderer 900 lines
away. The failure mode is on the record: ADR-0041 §7a documents the SOTA builder emitting
``redaction.gliner_pii`` while the module docstring, the ``FINANCIAL_PII_LABELS`` constant and the
lite builder all named ``redaction.gliner2`` — "the generator had drifted from its own design".

Here a Stage declares its OWN contribution:

* :class:`StageContribution` — ONE Stage's declaration. ``contribute(answers, ctx)`` returns the
  Stage block it wants in the stack, or ``None`` to stay out of it entirely (the toggled and
  detection-gated Stages). ``header_lines(answers, block)`` returns that Stage's own comment-header
  prose — and is called with ``block=None`` when the Stage did NOT contribute, so a Stage also owns
  the prose describing its own ABSENCE. Prose that can only appear beside the block it describes
  cannot drift from it.
* :class:`RecipeContext` — the wiring state ONE walk threads through its contributions: named
  *anchors* (the chunk stream, the label source) and named *fan-ins* (everything Disambiguation
  consumes, everything the KG export waits for). A contribution reads the anchor it needs instead of
  a local variable the neighbouring 200 lines happened to rebind.
* :class:`StackRecipe` — a PROFILE: an ordered literal of contributions plus its per-Stage
  overrides, and its own profile-level header prologue. Ordered literal ⇒ the ADR-0041 §3
  determinism contract (identical answers ⇒ byte-identical YAML) is preserved by construction.

**In-core and profile-scoped, deliberately.** ADR-0041 constraint 1 is "opinionated, not a blank
canvas": this is NOT an entry-point registry a third party plugs into. Every contribution ships in
this module, and the only recipes are :data:`LITE_RECIPE` and :data:`SOTA_RECIPE`. The
operator-facing interface is unchanged — answers in, one validated YAML out.

The deletion test: deleting the two old builders does NOT make complexity vanish. the chunk
budget's coupling to the mdeberta window, the 0.3-vs-0.5 extraction threshold, the ADR-0044
financial-PII floor, the ``redact -> export_corpus`` dataflow are all real, load-bearing
knowledge. It RELOCATES to N small,
individually-testable declarations behind one assembler — which is the depth gain: a caller still
learns one function, and a maintainer changing one Stage now reads one declaration instead of two
procedural scripts and a header renderer.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence, Sized
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:  # pragma: no cover - typing only (SetupAnswers stays with the answer-resolution)
    from latence_core.setup.wizard import SetupAnswers

# --------------------------------------------------------------------------------------------
# The blessed stack's constants (the wizard's ONE opinionated config — ADR-0041)
# --------------------------------------------------------------------------------------------

# The stack name the generated config carries (stable → byte-identical across runs).
STACK_NAME = "latence-setup-stack"

# The ONE blessed PII redactor the wizard emits — in EVERY profile (#197, ADR-0041/0036/0044).
#
# The wizard's one opinionated config must BE the profile's documented design, and it had drifted:
# `lite` (and the wizard's own documented spine) named `redaction.gliner2` while `enterprise-sota`
# emitted `redaction.gliner_pii` — a different package, library, and checkpoint for the same Stage.
# #197 decides ONE, and the decision is `redaction.gliner2` (reasoning recorded in ADR-0041 §7a):
#
#   * ADR-0036 W9 makes gliner2 the UNIFIED extraction+PII engine, and both profiles already run
#     `fused_entity_relation.gliner2` for `extract`. Blessing the gliner2 redactor means the stack
#     resolves ONE library, ONE mdeberta-v3 encoder family, and ONE version pin for both Stages
#     rather than additionally pulling the separate `gliner` library only for `redact`.
#   * The choice is only now a real one: #192 restored `redaction.gliner2` loadability and pinned it
#     (`gliner2[local]>=1.3,<2` + a typed ProviderError for a hand-installed env), so drift fails at
#     dependency resolution. Its twin `redaction.gliner_pii` remains fully supported (#191 placed it
#     on the resolved device) — it is simply not what the guided setup blesses.
#   * Weights AND code are permissive and verified separately (ADR-0012): the `gliner2` framework is
#     Apache-2.0 (code) and the default `fastino/GLiNER2-Guardrails-PII-Multi` checkpoint is
#     Apache-2.0 and ungated (weights) — see packages/latence-pii-gliner2 and ADR-0036 W3.
#   * The ADR-0044 financial-PII floor is unaffected by the choice: it lives at the single shared
#     `finalize_redacted_chunk` seam every Redaction Provider funnels through, so it is
#     non-bypassable by backend. `test_setup_wizard_redactor.py` proves it through THIS Provider
#     with THIS emitted config rather than assuming it.
#
# Since C7 the header comment reads the redactor NAME off the emitted `redact` block rather than off
# this constant, so the class of drift ADR-0041 §7a recorded cannot recur even for a Stage whose
# contribution stops using the constant.
#
# ADR-0064: this is now the `gliner2` ENGINE's redactor rather than a free-standing choice. The
# extraction engine and the redactor are ONE answer (see `ExtractionEngine` below), because the two
# Providers pull mutually exclusive `gliner2` majors into one environment — so a stack naming
# `fused_entity_relation.gliner25` beside `redaction.gliner2` is not merely discouraged, it is
# unrepresentable in the answer struct.
BLESSED_REDACT_PROVIDER = "redaction.gliner2"

# The financial-PII label FLOOR the blessed redactor (`BLESSED_REDACT_PROVIDER`) runs when PII
# is ON.
# Wider than the generic person/email/phone default: the enterprise corpora this stack targets carry
# account/tax/card identifiers. The induced pii_types UNION on top of this floor at run (ADR-0038),
# so the effective set is always ⊇ this list. Kept as one constant so the generator and the tests
# read the exact same set.
FINANCIAL_PII_LABELS: tuple[str, ...] = (
    "person",
    "email",
    "phone number",
    "address",
    "iban",
    "credit card number",
    "tax id",
    "bank account number",
    "social security number",
)

# The parsers the wizard chooses between, keyed by intent (matches the roster).
PARSER_UNIVERSAL = "parser.render"  # universal: mixed types (PDF/office/images), GPU-OCR sub
PARSER_PDF_TEXT = "parser.pdfplumber"  # born-digital PDF text (CPU, no OCR)
PARSER_OCR = "parser.lighton"  # OCR VLM for scans / images
PARSER_OCR_VLLM = "parser.lighton_vllm"  # served-vLLM OCR endpoint (the enterprise-SOTA parse path)
PARSER_PLAINTEXT = "parser.plaintext"  # born-digital text (.txt/.md/.jsonl) — CPU, no OCR, zero dep

# The two parsers the enterprise-SOTA profile chooses between. A GPU pod almost always parses
# scanned/PDF/image corpora through the served-vLLM OCR endpoint, but an enterprise corpus is just
# as often born-digital PLAIN TEXT (.txt/.md/.jsonl exports, log/ticket dumps). For those the OCR
# VLM is the WRONG tool — it only rasterizes pages and POSTs images to the OCR server, it has no
# text passthrough — so forcing it makes a text corpus un-parseable when the OCR box is (correctly)
# not served. The wizard therefore routes a detected plaintext corpus to the CPU plaintext parser
# and keeps every other SOTA choice (Granite r2 included) intact.

# The two setup profiles (tiers). `lite` is the CPU-first, laptop-friendly default (unchanged); the
# `enterprise-sota` profile emits the pod-VALIDATED composed stack (served-vLLM OCR + LLM schema
# induction + fused GLiNER2 ER + the neural GLinker disambiguator + Granite r2 embeddings + ULTRA
# graph completion — see stacks/gpu-sota-glinker.yaml). Both flow through the SAME pure assembler.
PROFILE_LITE = "lite"
PROFILE_SOTA = "enterprise-sota"

# Enterprise-SOTA served-OCR endpoint defaults (the vLLM OpenAI-compatible LightOn-OCR server).
# `base_url` + `model` are prompted (they are deployment-specific); `api_key` is a placeholder a
# local vLLM accepts, and `max_concurrency` is the per-document page fan-out (the provider default).
DEFAULT_OCR_BASE_URL = "http://localhost:8000/v1"
DEFAULT_OCR_MODEL = "lightonocr"
SOTA_OCR_API_KEY = "EMPTY"  # pragma: allowlist secret  (placeholder a local vLLM accepts; no key)
SOTA_OCR_MAX_CONCURRENCY = 8

# The enterprise-SOTA stack is a GPU-pod stack: every learned Stage runs on CUDA (its neural
# biencoders/rerankers/OCR need a GPU). On a CPU-only host these Stages are skip-with-flag at
# `latence stack validate` (device honesty, P3-F1) — the generated config still validates.
SOTA_DEVICE = "cuda"

# The default source extensions when a folder is empty / undetectable — the broad GOOD-stack set.
_DEFAULT_EXTENSIONS: tuple[str, ...] = (
    "txt",
    "md",
    "markdown",
    "pdf",
    "docx",
    "pptx",
    "xlsx",
    "png",
    "jpg",
    "jpeg",
    "zip",
)

# Default LLM endpoint for schema induction (OpenRouter + a small, cheap, capable model).
DEFAULT_LLM_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL = "openai/gpt-4.1-mini"

# --------------------------------------------------------------------------------------------
# The chunk token budget, and the model window it IMPLIES (T-WIZ-1)
# --------------------------------------------------------------------------------------------
#
# The chunk budget used to be a literal — `{"max_tokens": 640, "overlap_tokens": 80,
# "min_tokens": 8}` — with no answer, no flag, and no way to retune a run without hand-editing the
# emitted YAML. It is now an ANSWER; these are its defaults.
#
# 768 is the GLiNER2 window, and picking it is only safe because the model-side bound moves WITH
# it. The trap, pod-measured (docs/TUTORIAL-fresh-pod-walkthrough.md §4): `chunk.markdown` counts
# in its OWN tokenizer (`chunking.estimate_tokens`), which runs ~1.2x FEWER tokens than the
# mdeberta-v3 encoder the gliner family runs — a 768-chunker-token cap measured mdeberta **median
# 666 / MAX 921**, i.e. 2 chunks in 7 past the 768 window, where the encoder SILENTLY truncates
# (tail entities/relations lost, offsets past the window unreliable). So the old 640 was not a
# taste; it was 768/1.2, the largest budget that fit a fixed 768 window.
#
# Making the budget an answer therefore means deriving the window from it rather than pinning
# either: `extract`/`redact` emit `max_len = model_window_for(budget)` on BOTH profiles (the SOTA
# profile used to emit no `max_len` at all and ride the Provider's 768 default — which silently
# truncates the moment an operator raises the budget). At the historical 640 the derivation
# reproduces exactly the historical 768, so the constant GENERALISES the calibration rather than
# replacing it.
CHUNK_MAX_TOKENS = 768
CHUNK_OVERLAP_TOKENS = 80
CHUNK_MIN_TOKENS = 8

#: Pod-measured chunker-token -> mdeberta-token expansion (768 chunker tokens -> max 921 mdeberta,
#: 921/768 = 1.199). Used to size the model window from the chunk budget. Deliberately the MAX
#: observed ratio, not the median (666/768 = 0.87): the window is a truncation boundary, so it must
#: cover the worst chunk, not the typical one.
MDEBERTA_TOKEN_EXPANSION = 1.2

#: Model windows are rounded UP to a multiple of this. A tidy, legible number in the emitted YAML,
#: and it keeps the derived window stable under small budget changes.
MODEL_WINDOW_GRANULARITY = 64

#: The gliner-family model window the PROVIDER defaults to — mirrored from
#: ``providers/perf.py::DEFAULT_GLINER_MAX_LEN`` rather than imported, so the setup module stays a
#: leaf (ADR-0016) and so nothing here depends on the Provider-side perf seam. The mirror is held
#: in sync by ``test_setup_wizard_knobs.py``. It is the reference point for "does this derived knob
#: differ from what the Provider would have done anyway?" — if not, nothing is emitted.
DEFAULT_MODEL_WINDOW = 768

#: The padded micro-batch size (token POSITIONS) the pod measured as safe: GLiNER2 pads every text
#: in a micro-batch up to the batch's longest sequence, so the padded worst case is
#: ``len(batch) x longest`` — and ``DEFAULT_MICROBATCH_MAX_TEXTS = 4`` x ``DEFAULT_GLINER_MAX_LEN =
#: 768`` peaked at **21.6 GiB** and passed on the 32 GB pod (batch_size=32 OOM'd). DECISIONS.md §D3
#: records that on the 24 GB target card that is only 2.4 GiB of headroom, so widening the model
#: window must NOT widen this product: a wider window buys its positions back out of
#: ``microbatch_max_texts`` (:func:`microbatch_max_texts_for`).
POD_SAFE_PADDED_POSITIONS = 4 * 768


def model_window_for(chunk_max_tokens: int) -> int:
    """The gliner-family ``max_len`` that COVERS a chunk budget of ``chunk_max_tokens``.

    ``ceil(budget x MDEBERTA_TOKEN_EXPANSION)`` rounded up to :data:`MODEL_WINDOW_GRANULARITY`.
    640 -> 768 (the historical pairing, reproduced exactly); 768 -> 960 (which covers the pod's
    measured 921). Pure arithmetic, so the byte-identical determinism contract holds.
    """
    needed = math.ceil(max(1, chunk_max_tokens) * MDEBERTA_TOKEN_EXPANSION)
    rounded = -(-needed // MODEL_WINDOW_GRANULARITY) * MODEL_WINDOW_GRANULARITY
    return max(MODEL_WINDOW_GRANULARITY, rounded)


def microbatch_max_texts_for(window: int) -> int:
    """The micro-batch text cap that holds the PADDED forward at :data:`POD_SAFE_PADDED_POSITIONS`.

    A wider model window costs VRAM quadratically in the attention and linearly in the padded
    batch; the pod-safe point is 4 texts x 768 positions. So the cap is
    ``POD_SAFE_PADDED_POSITIONS // window``, clamped ``>= 1`` — 768 -> 4 (the Provider's own
    default, so nothing is emitted for the historical window), 960 -> 3. The operator's explicit
    ``microbatch_max_texts`` answer always wins over this.
    """
    return max(1, POD_SAFE_PADDED_POSITIONS // max(1, window))


# --------------------------------------------------------------------------------------------
# Rasterisation policy (T-OCR-3)
# --------------------------------------------------------------------------------------------
#
# LightOn's published preprocessing recipe — render PDFs at 200 DPI to a target longest dimension
# of 1540 px, aspect preserved — is the DEFAULT and stays the default. What changes is that it was
# unreachable from `latence setup` on EVERY parse path: `parser.lighton` and `parser.lighton_vllm`
# emitted no raster config at all, and `parser.render`'s DPI was a literal. Both are now answers,
# emitted only when they deviate from these values, so an unset flag leaves every parse block
# byte-identical to before (reachability, not retuning).
#
# MIRRORED, not imported: `latence-core` must not depend on a Provider package (ADR-0016), and the
# source of truth is `latence_parser_lighton.rasterize.DEFAULT_DPI` / `DEFAULT_MAX_LONGEST_DIM`.
# `test_setup_wizard_knobs.py::test_raster_defaults_mirror_the_rasteriser` holds the two in sync,
# because an unchecked mirror is the drift class ADR-0041 §7a recorded.
DEFAULT_RASTER_DPI = 200
DEFAULT_RASTER_MAX_LONGEST_DIM = 1540


class SetupConfigError(ValueError):
    """A generated config failed its dry validation (structural / DAG / capability wiring).

    Raised by ``validate_config_text`` with a human-readable message so the CLI can fail loudly with
    a clear reason rather than saving a broken config.
    """


class RecipeWiringError(SetupConfigError):
    """A recipe walked itself into an impossible state (a typed generator bug, never silent).

    Raised when a contribution reads an anchor/fan-in nothing has set, emits a block whose ``name``
    disagrees with its own declared name, or emits a Stage name a previous contribution already
    took. Every one of those is a mistake in a recipe literal, and every one of them would otherwise
    surface far downstream as a confusing DAG error (or, worse, as a silently mis-wired stack).
    ADR-0034: failures are typed and loud.
    """


# --------------------------------------------------------------------------------------------
# The extraction ENGINE — one answer, because the alternative is an unresolvable environment
# --------------------------------------------------------------------------------------------
#
# ADR-0064. `fused_entity_relation.gliner2` + `redaction.gliner2` pin `gliner2>=1.3,<2`;
# `fused_entity_relation.gliner25` + `redaction.gliner25` pin `gliner2>=2.0,<3`. Those windows are
# DISJOINT — gliner2 2.0 is a major release whose 2.5 checkpoints are a different architecture head
# — so a stack naming one package's extractor beside the other package's redactor cannot be
# installed at all. It would validate (both Providers are real, both fulfil their Capability) and
# then fail at `pip install`, which is the worst place to learn it.
#
# The fix is structural, not advisory: the wizard has ONE `engine` answer, and both Stages read
# their Provider off the SAME row of this table. There is no combination of answers that emits a
# mixed pair, so no comment has to ask anyone not to.

#: The gliner2 1.x engine — the blessed default, unchanged (ADR-0036 W9, #197).
ENGINE_GLINER2 = "gliner2"

#: The GLiNER 2.5 engine (ADR-0064): the gliner2 2.x boundary architecture over a 4096-token
#: checkpoint window, with page-preferred chunking as its coherent default.
ENGINE_GLINER25 = "gliner25"


@dataclass(frozen=True)
class ExtractionEngine:
    """One coherent (extractor, redactor, chunker, budget) row — the unit the wizard selects.

    Everything here moves together for a reason:

    * ``extract_provider`` / ``redact_provider`` come from ONE distribution family and therefore
      ONE ``gliner2`` major. Reading both off one row is what makes a mixed pair inexpressible.
    * ``chunker`` / ``chunk_max_tokens`` are the segmentation the engine's model window is sized
      for. A 4096-token checkpoint asked to read 512-token chunks wastes its window; a 768-token
      checkpoint handed 1024-token chunks truncates silently.
    * ``max_window`` is the checkpoint family's DOCUMENTED token window, and where there is one it
      is a REFUSAL: a chunk budget whose derived ``max_len`` exceeds it is rejected at
      answer-construction rather than emitted as a config that silently truncates. ``None`` means
      the family publishes no such ceiling — which is the honest state of the gliner2 1.x line
      (mdeberta-v3 attends by relative position and ``max_len`` is a per-call bound, so no number
      was ever verified for it) and NOT a licence to invent one. Only the GLiNER 2.5 card states a
      window, so only that engine carries the refusal here; the Provider itself enforces the same
      ceiling independently.
    * ``guardrails`` records whether the engine's redactor has a guardrail-trained checkpoint. Only
      the gliner2 1.x family does; asking the general GLiNER 2.5 checkpoint for guardrail categories
      zero-shot would be a security control that reports nothing while looking enabled, and
      ``redaction.gliner25`` refuses the switch outright.
    """

    name: str
    extract_provider: str
    redact_provider: str
    chunker: str
    chunk_max_tokens: int
    max_window: int | None
    guardrails: bool
    summary: str


#: The ``chunk.page`` budget, MIRRORED from ``latence_core.stages.chunk.PAGE_CHUNK_MAX_TOKENS``
#: rather than imported, exactly as :data:`DEFAULT_MODEL_WINDOW` mirrors the Provider-side default
#: — the setup module stays a leaf, and ``test_setup_wizard_knobs.py`` holds the two in sync,
#: because an unchecked mirror is the drift class ADR-0041 §7a recorded.
CHUNK_MAX_TOKENS_PAGE = 1024

#: The GLiNER 2.5 checkpoint window, per the ``fastino/gliner2.5-multi-v1`` model card. Mirrored
#: rather than imported (``latence-core`` must not depend on a Provider package, ADR-0016); the
#: mirror is held in sync with ``latence_gliner25._backbone.MAX_SUPPORTED_MAX_LEN`` by a test.
GLINER25_MAX_WINDOW = 4096

ENGINES: Mapping[str, ExtractionEngine] = MappingProxyType(
    {
        ENGINE_GLINER2: ExtractionEngine(
            name=ENGINE_GLINER2,
            extract_provider="fused_entity_relation.gliner2",
            redact_provider=BLESSED_REDACT_PROVIDER,
            chunker="chunk.markdown",
            chunk_max_tokens=CHUNK_MAX_TOKENS,
            # No ceiling is claimed: nothing in this repository ever verified one for the 1.x
            # checkpoints, and asserting a number here would refuse budgets that work today.
            max_window=None,
            guardrails=True,
            summary=(
                "fastino GLiNER2 1.x (mdeberta-v3 span head): fused NER+RE, a guardrails+PII "
                "redaction checkpoint, a 768-token window. The validated default."
            ),
        ),
        ENGINE_GLINER25: ExtractionEngine(
            name=ENGINE_GLINER25,
            extract_provider="fused_entity_relation.gliner25",
            redact_provider="redaction.gliner25",
            chunker="chunk.page",
            chunk_max_tokens=CHUNK_MAX_TOKENS_PAGE,
            max_window=GLINER25_MAX_WINDOW,
            guardrails=False,
            summary=(
                "fastino GLiNER 2.5 (gliner2 2.x boundary head): fused NER+RE over a 4096-token "
                "window, page-preferred chunking. PII is ZERO-SHOT — there is no GLiNER 2.5 PII "
                "checkpoint — and its recall is UNMEASURED."
            ),
        ),
    }
)

_KNOWN_ENGINES = frozenset(ENGINES)


def engine_for(name: str) -> ExtractionEngine:
    """The :class:`ExtractionEngine` row for ``name``, or a :class:`SetupConfigError`.

    Unlike ``recipe_for``, an unknown engine does NOT fall back: a profile falls back to ``lite``
    because every profile emits a valid stack, whereas guessing an engine would silently pair one
    family's extractor with the operator's expectations of the other. The CLI validates the flag,
    and :meth:`SetupAnswers.__post_init__` validates the struct, so this is the third gate.
    """
    try:
        return ENGINES[name]
    except KeyError as exc:
        msg = f"Unknown extraction engine {name!r}; expected one of {sorted(_KNOWN_ENGINES)}."
        raise SetupConfigError(msg) from exc


#: The ``chunk_max_tokens`` "not answered" sentinel. A budget must be ``>= 1``, so ``0`` is a value
#: no operator can mean, which is what makes it usable as one: it lets
#: :class:`~latence_core.setup.wizard.SetupAnswers` resolve the budget from the chosen ENGINE
#: without turning the field ``int | None`` and pushing a narrowing cast into every read site.
#:
#: Why the resolution must live on the STRUCT rather than only in the CLI: ``build_stack_config``
#: is a public entry point, and a caller doing ``SetupAnswers(engine=ENGINE_GLINER25)`` must get
#: the engine's own coherent budget rather than the other engine's. A first draft resolved it in
#: the CLI alone and produced exactly the mismatch ADR-0041 §7a is about — the generator disagreeing
#: with its own design, visible only in the emitted YAML.
CHUNK_BUDGET_FROM_ENGINE = 0

#: ``chunk_strategy``: take the engine's own chunker (the coherent default), or override it.
CHUNK_STRATEGY_AUTO = "auto"
CHUNK_STRATEGY_WINDOW = "window"
CHUNK_STRATEGY_PAGE = "page"
_CHUNK_STRATEGY_PROVIDERS: Mapping[str, str] = MappingProxyType(
    {CHUNK_STRATEGY_WINDOW: "chunk.markdown", CHUNK_STRATEGY_PAGE: "chunk.page"}
)
_KNOWN_CHUNK_STRATEGIES = frozenset({CHUNK_STRATEGY_AUTO, *_CHUNK_STRATEGY_PROVIDERS})


def chunker_for(engine: ExtractionEngine, strategy: str) -> str:
    """The ``chunk`` Provider name for an engine + strategy pair.

    ``auto`` (the default) takes the engine's own chunker, which is the whole point of the engine
    being one row: the segmentation and the model window are chosen together. The two explicit
    values are the escape hatch for an operator who wants page-preferred chunking with the 1.x
    engine, or the window strategy with 2.5. ``chunk.sentence_window`` is deliberately NOT offered
    here — the wizard has never emitted it, and adding a third strategy to the guided path is its
    own decision, not a side effect of this one.
    """
    if strategy == CHUNK_STRATEGY_AUTO:
        return engine.chunker
    try:
        return _CHUNK_STRATEGY_PROVIDERS[strategy]
    except KeyError as exc:
        msg = (
            f"Unknown chunk strategy {strategy!r}; expected one of "
            f"{sorted(_KNOWN_CHUNK_STRATEGIES)}."
        )
        raise SetupConfigError(msg) from exc



# -- the intake oversize guard, sized from the SCANNED corpus ------------------------------------
#
# `intake_screen` (screening.intake_signature) quarantines any file over `max_bytes` — the
# oversize/DoS guard (issue #62). It MUST stay a real guard, but a FIXED cap silently quarantines
# the very corpus the wizard just scanned: a book-sized 33 MB PDF is 16x over the old flat 2 MiB
# cap, so `latence setup && latence process` produced an EMPTY corpus with no visible failure. So
# the cap is now SIZED from the largest file actually observed under the source folder:
#
#     cap = clamp(round_up_to_mib(largest_observed * HEADROOM), FLOOR, CEILING)
#
# The bounds, justified:
#   * FLOOR (2 MiB) — the previous flat cap, kept as the never-go-below default. It is what an
#     un-scannable / empty / remote source still gets (with a loud warning), so nothing regresses
#     for a corpus that never needed more, and a config generated without a scan is byte-identical
#     to the pre-fix output.
#   * HEADROOM (2x) — files that arrive AFTER setup are the norm (a corpus grows); doubling the
#     largest observed file admits the next, somewhat-larger document without a re-run, while
#     staying within one binary order of magnitude of real, observed evidence. It is deliberately
#     NOT "infinity": a file 2x larger than anything in the corpus is genuinely anomalous.
#   * MiB rounding — a tidy, human-readable cap in the emitted YAML (67797168 -> 68157440) that
#     does not leak the exact byte size of one operator document into the config.
#   * CEILING (2 GiB) — the guard's backstop. Past this the cap stops tracking the corpus, because
#     a multi-GiB single document is a stack-design problem (and a memory-exhaustion vector — the
#     screen reads the bytes), not something to silently wave through. When the ceiling clamps
#     below the largest observed file the CLI warns LOUDLY that those files will be quarantined.
#
# Pure arithmetic over the scan result, so the byte-identical determinism contract holds: the same
# corpus always yields the same cap.
INTAKE_MAX_BYTES_FLOOR = 2 * 1024 * 1024  # 2 MiB — the pre-fix flat cap, now the floor/default.
INTAKE_MAX_BYTES_CEILING = 2 * 1024 * 1024 * 1024  # 2 GiB — the backstop the guard never exceeds.
INTAKE_HEADROOM = 2  # Admit files up to 2x the largest one observed at setup.
_MIB = 1024 * 1024


def intake_max_bytes(largest_source_bytes: int) -> int:
    """The ``intake_screen`` ``max_bytes`` cap for a corpus whose largest file is that many bytes.

    ``0`` (nothing observed — an empty, unreadable, or non-local source) yields
    :data:`INTAKE_MAX_BYTES_FLOOR`, the pre-fix flat cap. Otherwise the largest observed file is
    given :data:`INTAKE_HEADROOM` headroom, rounded up to a whole MiB, and clamped into
    ``[FLOOR, CEILING]``. Pure + deterministic (same corpus -> same cap).
    """
    if largest_source_bytes <= 0:
        return INTAKE_MAX_BYTES_FLOOR
    sized = -(-(largest_source_bytes * INTAKE_HEADROOM) // _MIB) * _MIB  # ceil-div to whole MiB
    return min(max(sized, INTAKE_MAX_BYTES_FLOOR), INTAKE_MAX_BYTES_CEILING)


# The single-line, length-capped form the domain hint takes in the emitted config + header comment.
# The hint is free-text operator input that lands in (a) a YAML comment — where a newline would
# break out of the comment and corrupt the config — and (b) the `induce` Stage's `domain` key, from
# which the inducer Provider injects it into an LLM prompt. Both want one bounded line. 512 chars is
# ample for "German customs law: tariff classification, export-control rulings" and far short of any
# prompt-bloat concern. The Provider sanitizes independently (defence in depth: a hand-edited config
# is not this function's output), so this is about a tidy, well-formed config, not the trust
# boundary.
DOMAIN_HINT_MAX_CHARS = 512
_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_domain_hint(raw: str) -> str:
    """Collapse a free-text domain hint to ONE bounded line (``""`` when it carries no text)."""
    collapsed = _WHITESPACE_RUN.sub(" ", raw).strip()
    return collapsed[:DOMAIN_HINT_MAX_CHARS].strip()


def llm_key_env_var(base_url: str) -> str:
    """The env var the schema-induction LLM reads its key from (mirrors ``label_inducer.llm``).

    An OpenRouter ``base_url`` prefers ``$OPENROUTER_API_KEY``; anything else (direct OpenAI, or a
    keyless on-prem vLLM) reads ``$OPENAI_API_KEY``. The generator NEVER stores the key — this only
    names the var so the CLI can warn when it is unset.
    """
    return "OPENROUTER_API_KEY" if "openrouter" in (base_url or "").lower() else "OPENAI_API_KEY"


# --------------------------------------------------------------------------------------------
# The seam
# --------------------------------------------------------------------------------------------

#: One Stage's mapping as it lands in the emitted YAML (``name``/``capability``/``provider``/
#: ``depends_on``/``config``). Key insertion order IS the emitted field order — ``yaml.safe_dump``
#: runs with ``sort_keys=False`` so the determinism contract is a property of these literals.
StageBlock = dict[str, object]

#: The chunk-level stream every label-consuming Stage reads: ``chunk``, or ``content_screen`` when
#: the guardrail half of the PII toggle is on.
ANCHOR_CHUNK_STREAM = "chunk_stream"

#: The Stage that supplies induced labels — ``induce``, or (predefined-only mode, where NO induce
#: Stage exists) whatever would have fed it.
ANCHOR_LABELS = "labels"

#: Fan-in: every Stage producing entity mentions / relations that Disambiguation must consume.
FANIN_MENTIONS = "mentions"

#: Fan-in: every Stage producing knowledge-graph edges the KG Export must wait for.
FANIN_KG = "kg"


@dataclass
class RecipeContext:
    """The wiring state ONE recipe walk threads through its contributions.

    A contribution never names a neighbour it cannot see. It reads *anchors* (a single upstream
    Stage whose identity depends on which optional Stages emitted) and *fan-ins* (an accumulating
    list of upstream Stages), and writes the ones it redefines. That is what lets
    ``content_screen``, ``induce``, ``header_refs``, ``inline_refs``, ``redact``,
    ``consolidate_types`` and ``complete`` be genuinely optional without any other contribution
    knowing they exist.

    Reading an anchor or fan-in nothing has set is a :class:`RecipeWiringError`, not a ``KeyError``
    or a silent empty list — a recipe literal in the wrong order must fail loudly at generation
    time (ADR-0034).
    """

    _anchors: dict[str, str] = field(default_factory=dict)
    _fanin: dict[str, list[str]] = field(default_factory=dict)
    _emitted: dict[str, StageBlock] = field(default_factory=dict)

    # -- anchors ---------------------------------------------------------------------------
    def anchor(self, name: str) -> str:
        """The Stage currently standing at ``name`` (raises when nothing has set it)."""
        try:
            return self._anchors[name]
        except KeyError:
            msg = (
                f"recipe wiring error: anchor {name!r} was read before any contribution set it "
                f"(anchors set so far: {sorted(self._anchors)})."
            )
            raise RecipeWiringError(msg) from None

    def set_anchor(self, name: str, stage: str) -> None:
        """Point ``name`` at ``stage`` for every later contribution."""
        self._anchors[name] = stage

    # -- fan-ins ---------------------------------------------------------------------------
    def fanin(self, name: str) -> list[str]:
        """A fresh list of the Stages accumulated at ``name`` (raises when nothing has set it)."""
        try:
            return list(self._fanin[name])
        except KeyError:
            msg = (
                f"recipe wiring error: fan-in {name!r} was read before any contribution added to "
                f"it (fan-ins set so far: {sorted(self._fanin)})."
            )
            raise RecipeWiringError(msg) from None

    def add_fanin(self, name: str, stage: str) -> None:
        """Append ``stage`` to the fan-in ``name`` (creating it on first use)."""
        self._fanin.setdefault(name, []).append(stage)

    def replace_fanin(self, name: str, stages: Sequence[str]) -> None:
        """Collapse the fan-in ``name`` onto ``stages`` — a Stage that CONSUMED the whole fan-in and
        now stands in front of it (``consolidate_types`` before Disambiguation)."""
        self._fanin[name] = list(stages)

    # -- what has been emitted -------------------------------------------------------------
    def emitted(self, name: str) -> bool:
        """Whether a Stage called ``name`` has contributed to this walk already."""
        return name in self._emitted

    def block(self, name: str) -> StageBlock:
        """The block a previous contribution emitted under ``name`` (raises when absent)."""
        try:
            return self._emitted[name]
        except KeyError:
            msg = (
                f"recipe wiring error: Stage {name!r} was read before it contributed "
                f"(emitted so far: {sorted(self._emitted)})."
            )
            raise RecipeWiringError(msg) from None

    def _record(self, name: str, block: StageBlock) -> None:
        if name in self._emitted:
            msg = f"recipe wiring error: two contributions both emitted a Stage named {name!r}."
            raise RecipeWiringError(msg)
        self._emitted[name] = block


class StageContribution(ABC):
    """ONE Stage's declaration of what it puts into the blessed stack, and why.

    Subclasses are small frozen dataclasses: their fields ARE the per-Stage overrides a profile
    applies (an extraction threshold, a device pin, whether the corpus Export embeds on the GPU
    tier). Everything a Stage knows about itself — its Provider, its config, what it
    depends on, when it stays out of the stack entirely, and the comment-header prose describing it
    — lives in ONE class, so a change to that Stage is a change to one declaration.

    In-core by design (ADR-0041 constraint 1, "opinionated, not a blank canvas"): this is not an
    entry-point registry. Contributions ship here; profiles compose them in :data:`LITE_RECIPE` /
    :data:`SOTA_RECIPE`.
    """

    #: The emitted Stage's ``name`` — the key every downstream ``depends_on`` wires on. The
    #: assembler rejects a block whose ``name`` disagrees with this.
    name: str
    #: One line saying why this Stage is in the blessed stack. Surfaced by
    #: :attr:`AssembledStack.rationales` so the recipe can describe itself.
    why: str

    @abstractmethod
    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock | None:
        """This Stage's block, or ``None`` to stay out of the stack entirely.

        ``None`` is an OMISSION, not a disabled Stage: no Provider is named, nothing is
        instantiated, and the emitted YAML is byte-identical to a stack that never had the feature.
        A contribution that omits itself is still responsible for any anchor a later contribution
        will read (predefined-only mode points :data:`ANCHOR_LABELS` at the chunk stream).
        """

    def header_lines(self, answers: SetupAnswers, block: StageBlock | None) -> tuple[str, ...]:
        """This Stage's comment-header prose (``()`` by default — most Stages need none).

        Called for EVERY contribution in the recipe, with ``block=None`` when the Stage did not
        contribute — so a Stage owns the prose describing its own absence as well as its presence,
        and prose about a block can only be rendered next to a block that exists. Derive every claim
        from ``block``, never from a module constant: that is what makes the ADR-0041 §7a drift
        (header naming one Provider, generator emitting another) structurally impossible.
        """
        return ()


@dataclass(frozen=True)
class AssembledStack:
    """What ONE recipe walk produced: the Stage blocks, plus everything the renderer needs.

    ``by_name`` and ``header_lines`` are read-only views — the assembler is the only writer, so a
    renderer cannot quietly reshape a stack it is only supposed to describe.
    """

    #: The Stage blocks in emission order — exactly the emitted YAML's ``stages:`` list.
    stages: tuple[StageBlock, ...]
    #: Emitted blocks keyed by Stage name (absent ⇒ that Stage stayed out of the stack).
    by_name: Mapping[str, StageBlock]
    #: Per-contribution comment-header prose, keyed by Stage name. Present for EVERY contribution in
    #: the recipe, including the ones that stayed out (their prose describes the absence).
    header_lines: Mapping[str, tuple[str, ...]]
    #: ``(stage name, why)`` for every Stage that actually contributed, in emission order.
    rationales: tuple[tuple[str, str], ...]


def _no_prologue(answers: SetupAnswers, stack: AssembledStack) -> tuple[str, ...]:
    """A profile that adds no profile-level header prose (the ``lite`` default)."""
    return ()


@dataclass(frozen=True)
class StackRecipe:
    """A PROFILE: an ordered literal of :class:`StageContribution` plus its header prologue.

    Ordered LITERAL, not a sorted registry: Stage order in the emitted YAML is a property of this
    tuple, so the ADR-0041 §3 determinism contract (identical answers ⇒ byte-identical file) holds
    by construction rather than by a sort key someone has to remember to keep stable.
    """

    profile: str
    contributions: tuple[StageContribution, ...]
    #: Profile-level comment-header prose, rendered before the per-Stage prose. Takes the assembled
    #: stack so its claims are derived from emitted blocks rather than re-decided from the answers.
    prologue: Callable[[SetupAnswers, AssembledStack], tuple[str, ...]] = _no_prologue

    def assemble(self, answers: SetupAnswers) -> AssembledStack:
        """Walk the recipe once and collect what each Stage contributed.

        Pure + deterministic: no env, clock, or randomness anywhere in the walk. The ONE
        cross-Stage precondition is checked up front, because ``extract_on: masked`` is a statement
        about a Stage (``redact``) the answers may have switched off entirely.
        """
        if answers.extract_on == "masked" and not answers.pii:
            msg = (
                "extract_on: masked requires PII redaction ON "
                "(there is no redact Stage otherwise)."
            )
            raise ValueError(msg)

        ctx = RecipeContext()
        stages: list[StageBlock] = []
        header: dict[str, tuple[str, ...]] = {}
        rationales: list[tuple[str, str]] = []
        emitted: dict[str, StageBlock] = {}

        for contribution in self.contributions:
            block = contribution.contribute(answers, ctx)
            if block is not None:
                declared = block.get("name")
                if declared != contribution.name:
                    msg = (
                        f"recipe wiring error: {type(contribution).__name__} declares name "
                        f"{contribution.name!r} but emitted a block named {declared!r}."
                    )
                    raise RecipeWiringError(msg)
                ctx._record(contribution.name, block)  # noqa: SLF001 - the assembler owns the ctx
                emitted[contribution.name] = block
                stages.append(block)
                rationales.append((contribution.name, contribution.why))
            header[contribution.name] = contribution.header_lines(answers, block)

        return AssembledStack(
            stages=tuple(stages),
            by_name=MappingProxyType(emitted),
            header_lines=MappingProxyType(header),
            rationales=tuple(rationales),
        )


# --------------------------------------------------------------------------------------------
# Shared blocks + label resolution (the knowledge more than one contribution needs)
# --------------------------------------------------------------------------------------------


def _entity_labels(answers: SetupAnswers, default: Sequence[str]) -> list[str]:
    """The extraction Stage's entity labels: the operator's fixed set, else the profile default.

    In predefined-only mode the config labels are no longer a floor that induced types union onto —
    they are the WHOLE schema, so they must be the operator's own. Order is preserved as given
    (deterministic YAML), never sorted behind the operator's back.
    """
    fixed = (answers.predefined_labels or {}).get("entity_types")
    return list(fixed) if fixed else list(default)


def _relation_labels(answers: SetupAnswers, default: Sequence[str]) -> list[str]:
    """The extraction Stage's relation labels: the operator's fixed set, else the profile
    default."""
    fixed = (answers.predefined_labels or {}).get("relation_types")
    return list(fixed) if fixed else list(default)


def _pii_labels(answers: SetupAnswers) -> list[str]:
    """The redactor's PII labels: the ADR-0044 financial FLOOR, plus any operator-fixed PII types.

    The floor is UNIVERSAL and is never subsetted by a mode switch (README invariant 3): predefined
    -only lets an operator fix the *relevance* schema, not opt out of the safety schema. A leaked
    card number in a chunk the operator considers irrelevant must still be masked. Operator types
    are appended after the floor, deduped, order-stable.
    """
    labels = list(FINANCIAL_PII_LABELS)
    seen = set(labels)
    for label in (answers.predefined_labels or {}).get("pii_types", []):
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _label_candidates(answers: SetupAnswers) -> dict[str, object]:
    """The lexicon inducer's candidate type space, copied out of the answers struct.

    A COPY, not the operator's own object: the emitted block is handed to the YAML dumper and to
    every test that asserts on it, and a Stage config that aliases the answers would let one
    mutation reach both. Order — the operator's, slot by slot — is preserved, so the emitted YAML
    stays a byte-stable function of the supplied pool (T-WIZ-2 / ADR-0041 §3).
    """
    pool = answers.label_candidates or {}
    return {
        slot: (list(value) if isinstance(value, list) else {k: list(v) for k, v in value.items()})
        for slot, value in pool.items()
    }


def tiered_embedder_block(answers: SetupAnswers) -> StageBlock:
    """The device-TIERED ST embedder block (disambiguation context + RAG-corpus vectors).

    IBM Granite Embedding r2 (Apache-2.0, ADR-0045) tiered by the chosen device: the GPU tier
    ``granite-embedding-311m-multilingual-r2`` (768-dim, bf16 + flash_attention_2 with a graceful
    sdpa fallback) on ``cuda``, else the lighter CPU tier
    ``granite-embedding-97m-multilingual-r2`` (384-dim, fp32) for a laptop-CPU box.
    """
    if answers.device == "cuda":
        config: StageBlock = {
            "model": "ibm-granite/granite-embedding-311m-multilingual-r2",
            "dimension": 768,
            "device": answers.device,
            "attn_implementation": "flash_attention_2",
        }
    else:
        config = {
            "model": "ibm-granite/granite-embedding-97m-multilingual-r2",
            "dimension": 384,
            "device": answers.device,
        }
    return {"provider": "embedding.sentence_transformers", "config": config}


def gpu_embedder_block() -> StageBlock:
    """The enterprise-SOTA Granite r2 embedder block (disambiguation resolver + RAG-corpus vectors).

    Always the GPU tier ``granite-embedding-311m-multilingual-r2`` (768-dim, bf16 +
    ``flash_attention_2`` with a graceful sdpa fallback) on ``cuda`` — the SOTA stack is a GPU-pod
    stack (ADR-0045). NOT EmbeddingGemma/vllm-factory. It powers the GLinker resolver's STS
    embedding rung (kept as opt-in wiring even though embedding-merge rides OFF by default) AND the
    ``export_corpus`` dense vectors.
    """
    return {
        "provider": "embedding.sentence_transformers",
        "config": {
            "model": "ibm-granite/granite-embedding-311m-multilingual-r2",
            "dimension": 768,
            "device": SOTA_DEVICE,
            "attn_implementation": "flash_attention_2",
        },
    }


def _embedder_for(answers: SetupAnswers, *, gpu: bool) -> StageBlock:
    return gpu_embedder_block() if gpu else tiered_embedder_block(answers)


def _add_raster_knobs(
    config: StageBlock, answers: SetupAnswers, *, dpi: bool, longest: bool
) -> None:
    """Thread the T-OCR-3 rasterisation knobs onto ``config`` — omitting the untouched defaults.

    ``dpi``/``longest`` say which of the two this particular block is the one that applies (the
    universal renderer's OCR sub-provider only ever sees images, so DPI would be a dead key there).
    A knob equal to LightOn's published default is not emitted at all: the ticket asks for
    reachability, and the acceptance bar is that an unset flag leaves the block byte-identical.
    """
    if dpi and answers.ocr_dpi != DEFAULT_RASTER_DPI:
        config["dpi"] = answers.ocr_dpi
    if longest and answers.ocr_max_longest_dim != DEFAULT_RASTER_MAX_LONGEST_DIM:
        config["max_longest_dim"] = answers.ocr_max_longest_dim


def _add_gliner2_perf_knobs(config: StageBlock, answers: SetupAnswers, *, window: int) -> None:
    """Thread the T-WIZ-3 GLiNER2 performance knobs onto a gliner2 Stage's ``config``.

    ``microbatch_max_texts`` / ``microbatch_token_budget`` / ``intra_op_threads`` are read by BOTH
    gliner2 Providers (``providers/perf.py`` ``resolve_microbatch_bounds`` /
    ``resolve_intra_op_threads``) and were emitted by nothing. ``compile``/``quantize`` stay
    deliberately absent — the Provider resolves them per resolved device (True on cuda, False on
    cpu), so a stack emitted for a GPU box degrades correctly if it is later run on cpu.

    ``microbatch_max_texts`` additionally carries a DERIVED default when the model window is wider
    than the Provider's own 768: GLiNER2 pads a micro-batch to its longest sequence, so the padded
    forward is ``max_texts x window`` and the pod's safe point is 4 x 768 = 21.6 GiB peak. A wider
    window therefore has to buy its positions back out of the text cap
    (:func:`microbatch_max_texts_for`) or the 24 GB card loses its 2.4 GiB of headroom (D3). An
    explicit operator answer always wins; a derived value equal to the Provider default is omitted.
    """
    derived = microbatch_max_texts_for(window)
    max_texts = answers.microbatch_max_texts
    if max_texts is None and derived != POD_SAFE_PADDED_POSITIONS // DEFAULT_MODEL_WINDOW:
        max_texts = derived
    if max_texts is not None:
        config["microbatch_max_texts"] = max_texts
    if answers.microbatch_token_budget is not None:
        config["microbatch_token_budget"] = answers.microbatch_token_budget
    if answers.intra_op_threads is not None:
        config["intra_op_threads"] = answers.intra_op_threads


# --------------------------------------------------------------------------------------------
# The contributions — one Stage, one declaration
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceContribution(StageContribution):
    """``source`` — the operator's folder plus the file types the setup scan actually observed."""

    name: str = "source"
    why: str = "the operator's folder, restricted to the file types the setup scan observed"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock:
        return {
            "name": self.name,
            "capability": "source",
            "provider": "source.local_folder",
            "config": {
                "path": answers.source,
                "sensitivity": "internal",
                "extensions": list(answers.extensions),
            },
        }


@dataclass(frozen=True)
class IntakeScreenContribution(StageContribution):
    """``intake_screen`` — the always-on oversize / zip-bomb / spoofed-type guard (issue #62).

    The oversize cap is SIZED from the scanned corpus (:func:`intake_max_bytes`) so the operator's
    real documents are admitted rather than silently quarantined; it remains a real guard.
    """

    name: str = "intake_screen"
    why: str = "reject oversized / malformed / zip-bomb files before anything reads them"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock:
        return {
            "name": self.name,
            "capability": "intake_screening",
            "provider": "screening.intake_signature",
            "depends_on": ["source"],
            "config": {
                "max_bytes": intake_max_bytes(answers.largest_source_bytes),
                "max_zip_ratio": 50.0,
            },
        }


@dataclass(frozen=True)
class ParseContribution(StageContribution):
    """``parse`` — its config shaped by the chosen parser.

    ``served_ocr`` is the enterprise-SOTA override: that profile's production parse path is the
    served-vLLM LightOn-OCR endpoint (``parser.lighton_vllm``), which needs a base_url/model/
    api_key/concurrency block. Every other parser (the universal renderer, born-digital PDF text,
    the in-process OCR VLM, plain text) emits the same block in either profile. Routing the text
    case is what makes the SOTA profile yield a RUNNABLE stack for a .txt/.md/.jsonl corpus instead
    of forcing OCR at an endpoint that is not served for it.

    T-OCR-3: the rasterisation policy (``dpi`` + ``max_longest_dim`` — LightOn's published
    200/1540) is threaded onto whichever path actually rasterises. ``parser.render`` rasterises
    PDFs ITSELF and hands its OCR sub-provider page IMAGES, never a PDF, so the DPI rides the outer
    block and the downscale rides the sub-block; the two OCR parsers own both. A knob left at its
    default is OMITTED, so an unset flag leaves every parse block byte-identical to before — this
    ticket is reachability, not retuning.
    """

    served_ocr: bool = False
    name: str = "parse"
    why: str = "turn each source document into markdown + a page map"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock:
        if self.served_ocr and answers.parser == PARSER_OCR_VLLM:
            served: StageBlock = {
                "base_url": answers.ocr_base_url,
                "model": answers.ocr_model,
                "api_key": SOTA_OCR_API_KEY,
                "max_concurrency": SOTA_OCR_MAX_CONCURRENCY,
            }
            _add_raster_knobs(served, answers, dpi=True, longest=True)
            return {
                "name": self.name,
                "capability": "parse",
                "provider": PARSER_OCR_VLLM,
                "depends_on": ["intake_screen"],
                "config": served,
            }
        stage: StageBlock = {
            "name": self.name,
            "capability": "parse",
            "provider": answers.parser,
            "depends_on": ["intake_screen"],
        }
        if answers.parser == PARSER_UNIVERSAL:
            sub: StageBlock = {"device": answers.device, "max_new_tokens": 4096}
            _add_raster_knobs(sub, answers, dpi=False, longest=True)
            stage["config"] = {
                "device": answers.device,  # OCR sub-provider is GPU; skip-with-flag on a CPU host
                "dpi": answers.ocr_dpi,
                "soffice_timeout_s": 120,
                "ocr": {"provider": PARSER_OCR, "config": sub},
            }
        elif answers.parser == PARSER_OCR:
            direct: StageBlock = {"device": answers.device, "max_new_tokens": 4096}
            _add_raster_knobs(direct, answers, dpi=True, longest=True)
            stage["config"] = direct
        # parser.pdfplumber / parser.plaintext: born-digital text, no config block (CPU).
        return stage


@dataclass(frozen=True)
class ChunkContribution(StageContribution):
    """``chunk`` — the operator's token budget, defaulting to the GLiNER2 window (T-WIZ-1).

    The budget was a literal (640/80/8) with no answer and no flag: retuning a run meant
    hand-editing the emitted YAML, which is exactly what ``latence setup`` exists to avoid. It is
    now :attr:`SetupAnswers.chunk_max_tokens` / ``chunk_overlap_tokens`` / ``chunk_min_tokens``,
    defaulting to :data:`CHUNK_MAX_TOKENS` (768) — and the gliner-family Stages derive their
    ``max_len`` from it (:func:`model_window_for`), because the chunker counts ~1.2x FEWER tokens
    than the mdeberta encoder and a budget raised without the window would silently truncate.

    Opens :data:`ANCHOR_CHUNK_STREAM`: from here on, every chunk-level Stage reads whatever the
    anchor points at rather than the literal name ``chunk``.
    """

    name: str = "chunk"
    why: str = "split parsed documents at the operator's token budget (default: the GLiNER2 window)"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock:
        ctx.set_anchor(ANCHOR_CHUNK_STREAM, self.name)
        return {
            "name": self.name,
            "capability": "chunk",
            # ADR-0064: the chunker comes from the ENGINE row (or the operator's explicit
            # override), because the segmentation and the model window are one decision — a
            # 4096-token checkpoint reading 512-token chunks wastes its window, and a 768-token
            # checkpoint handed 1024-token chunks truncates silently.
            "provider": chunker_for(engine_for(answers.engine), answers.chunk_strategy),
            "depends_on": ["parse"],
            "config": {
                "max_tokens": answers.chunk_max_tokens,
                "overlap_tokens": answers.chunk_overlap_tokens,
                "min_tokens": answers.chunk_min_tokens,
            },
        }

    def header_lines(self, answers: SetupAnswers, block: StageBlock | None) -> tuple[str, ...]:
        if block is None:
            return ()
        config = block.get("config")
        assert isinstance(config, dict)  # noqa: S101 - the block this contribution just emitted
        budget = int(config["max_tokens"])
        # Read off the emitted block, never off the answers: the prose may only describe the
        # segmentation this walk actually wired (the ADR-0041 §7a discipline).
        provider = str(block.get("provider", ""))
        lines = [
            "#",
            f"# Chunk budget: {budget} chunker tokens / {config['overlap_tokens']} overlap"
            " (`--chunk-max-tokens`).",
            f"#   `extract`/`redact` bound the model window at {model_window_for(budget)} to match."
            " The chunker",
            "#   counts in its OWN tokenizer, ~1.2x FEWER tokens than the mdeberta-v3 encoder",
            "#   the GLiNER family runs (pod-measured: a 768-chunker-token cap hit mdeberta",
            "#   median 666 / MAX 921) and the encoder SILENTLY truncates past its window — so",
            "#   raising the budget raises the window with it.",
        ]
        if provider == "chunk.page":
            lines += [
                "#",
                "#   Chunker: `chunk.page` (ADR-0064) — ONE chunk per source page when the",
                "#   document carries a MEASURED page map, so a retrieval hit is citable as a",
                "#   page without any further resolution. A document with NO page structure (an",
                "#   `assumed_single_page` map, ADR-0060) falls back to the windowed budget above,",
                "#   and a page OVER that budget still splits — a page-sized chunk past the model",
                "#   window would be a silent truncation, and a split is not.",
            ]
        return tuple(lines)


@dataclass(frozen=True)
class ContentScreenContribution(StageContribution):
    """``content_screen`` — the guardrail half of the ONE optional toggle (``answers.pii``).

    Present ⇒ it takes over :data:`ANCHOR_CHUNK_STREAM`, so induction (and, in predefined-only
    mode, extraction) reads the screened stream. Absent ⇒ nothing downstream changes shape.
    """

    name: str = "content_screen"
    why: str = "deterministic keyword pre-filter for prompt injection / harmful content"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock | None:
        if not answers.pii:
            return None
        block: StageBlock = {
            "name": self.name,
            "capability": "content_screening",
            "provider": "screening.content_keyword",
            "depends_on": [ctx.anchor(ANCHOR_CHUNK_STREAM)],
        }
        ctx.set_anchor(ANCHOR_CHUNK_STREAM, self.name)
        return block


@dataclass(frozen=True)
class InduceContribution(StageContribution):
    """``induce`` — schema induction at CHUNK granularity, by LLM or by LEXICON, or NOTHING at all.

    **Per-chunk.** Each chunk is induced against its own text and carries only its own small,
    relevant label set. That is what the extractor downstream needs: the alternative — inducing per
    document, merging every document's labels into one corpus schema, ranking by document frequency
    and capping at 40 — drops the rare-but-locally-critical type from the chunks that needed it AND
    scores every chunk against ~35 types that are not in it. Type consistency across the KG is
    earned DOWNSTREAM by type canonicalization in the resolution phase, not bought here with
    recall. ``granularity: corpus`` / ``doc`` remain selectable by hand-editing the emitted config.

    ``dedup: true`` is emitted explicitly (it is also the Provider default) because it is the lever
    that makes per-chunk induction affordable — one call per DISTINCT chunk text, so a corpus's
    boilerplate costs one call, not one per occurrence — and an operator reading the stack should
    see it rather than have to know it.

    The operator's ``--domain`` hint is threaded onto the Stage's ``domain`` config key, which the
    inducer Provider injects into its OWN prompt (the Provider owns its prompt engineering — the
    wizard hands it the hint, never a hand-crafted prompt string). No hint ⇒ the key is OMITTED
    entirely, so the emitted config and the Provider's prompts are byte-identical to a run without
    one.

    **Predefined-only (T1) is an OMISSION, not a flag**: no Stage, no Provider, no client, no call.
    That is the only implementation of "induction is skipped entirely" that cannot regress into a
    quiet LLM call, and it makes the emitted stack self-evidently free of one. The contribution
    still points :data:`ANCHOR_LABELS` at the chunk stream so every label consumer stays wired.

    **The lexicon path (T-WIZ-2) is a PROVIDER swap and nothing else.** With
    ``answers.label_candidates`` set, the same Stage, on the same upstream, with the same
    ``granularity: chunk`` and ``dedup: true``, is served by ``label_inducer.lexicon`` — a
    deterministic, CPU-only, ZERO-NETWORK tagger over the operator's candidate type space (typically
    the pool ``latence-label-profile --candidates-out`` mines from the corpus). The pool is INLINED
    rather than referenced by path, so the emitted stack is self-contained: one file reaches the
    pod, ``stack validate`` sees the whole configuration, and "this run makes no induction call" is
    readable off the config — there is no endpoint in it to call. The LLM branch's ``model`` /
    ``base_url`` / ``max_tokens`` / ``temperature`` / ``domain`` are all OMITTED, because the
    lexicon Provider reads none of them and an emitted key nothing reads is a config that looks
    tuned and is not.
    """

    name: str = "induce"
    why: str = "induce each chunk's own small, relevant label schema (per-chunk, deduped)"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock | None:
        chunk_stream = ctx.anchor(ANCHOR_CHUNK_STREAM)
        if answers.predefined_labels is not None:
            ctx.set_anchor(ANCHOR_LABELS, chunk_stream)
            return None
        if answers.label_candidates is not None:
            ctx.set_anchor(ANCHOR_LABELS, self.name)
            return {
                "name": self.name,
                "capability": "schema_induction",
                "provider": "label_inducer.lexicon",
                "depends_on": [chunk_stream],
                # Exactly the keys `LexiconLabelInducer.__init__` reads. `min_hits` and
                # `whole_word` are emitted even at their defaults: they are the two knobs that
                # decide what a chunk induces, and `whole_word: false` is the answer a
                # compounding language (German `Rechnungsnummernkreis` ⊃ `Rechnungsnummer`) needs
                # — an operator tuning the run should read them off the stack, not infer them.
                "config": {
                    "granularity": "chunk",
                    "min_hits": answers.label_min_hits,
                    "whole_word": answers.label_whole_word,
                    "dedup": True,
                    "candidates": _label_candidates(answers),
                },
            }
        config: StageBlock = {
            "model": answers.llm_model,
            "base_url": answers.llm_base_url,
            "granularity": "chunk",
            "dedup": True,
            "max_tokens": 2048,
            "max_labels": 40,
            "temperature": 0.0,
        }
        domain = normalize_domain_hint(answers.domain)
        if domain:
            config["domain"] = domain
        ctx.set_anchor(ANCHOR_LABELS, self.name)
        return {
            "name": self.name,
            "capability": "schema_induction",
            "provider": "label_inducer.llm",
            "depends_on": [chunk_stream],
            "config": config,
        }

    def header_lines(self, answers: SetupAnswers, block: StageBlock | None) -> tuple[str, ...]:
        if block is None:
            return (
                "#",
                "# Schema induction: OFF (predefined-only) — you supplied a FIXED label schema, so",
                "#   there is NO `induce` stage in this stack at all and the run makes ZERO"
                " induction",
                "#   LLM calls (no key needed). `extract.config.labels` / `relation_labels` ARE"
                " the",
                "#   schema. The `redact.config.pii_labels` still carry the ADR-0044 financial"
                " floor:",
                "#   a fixed RELEVANCE schema is your choice, the SAFETY floor is not.",
                "#   Delete `predefined_labels` from your setup flags to get per-chunk induction"
                " back.",
            )
        if block["provider"] == "label_inducer.lexicon":
            # Derived from the BLOCK, never re-decided from `answers` (the ADR-0041 §7a drift
            # class): a Stage that stops emitting the lexicon Provider silences this prose.
            config = cast("StageBlock", block["config"])
            pool = cast("dict[str, Sized]", config["candidates"])
            declared = sum(len(types) for types in pool.values())
            return (
                "#",
                "# Schema induction: PER CHUNK, NO LLM (`label_inducer.lexicon`, T-WIZ-2) — every",
                "#   chunk is tagged against the candidate type space inlined at",
                f"#   `induce.config.candidates` ({declared} types over {len(pool)} slot(s)), and"
                " carries",
                "#   only the types its OWN text evidences. The run makes ZERO induction LLM calls"
                " and",
                "#   ZERO network calls of any kind (no key needed): the tagger is pure-CPU and",
                "#   deterministic, so the induced schema is a byte-reproducible function of the"
                " text",
                "#   and this config. Tune it with `min_hits` (how many DISTINCT cues a type"
                " needs)",
                "#   and `whole_word` (set false for compounding languages like German).",
                "#   Drop `--label-candidates` from your setup flags to induce with an LLM"
                " instead.",
            )
        key_var = llm_key_env_var(answers.llm_base_url)
        return (
            "#",
            "# Schema induction: PER CHUNK (`induce.config.granularity: chunk`, T1) — every chunk",
            "#   is induced against its OWN text and carries only its own small, relevant label",
            "#   set, which the extractor unions onto the config floor. That is what the extractor",
            "#   needs: a corpus-merged, salience-capped schema drops the rare-but-critical",
            "#   type from the chunks that needed it AND scores every chunk against types that are",
            "#   not in it. Type consistency across the KG is earned DOWNSTREAM (resolution), not",
            "#   bought here with recall. Cost is bounded by `induce.config.dedup` (one call per",
            "#   DISTINCT chunk text) and, optionally, `max_induction_calls` (a per-run call",
            "#   budget); the Quality Report reports calls, cache hits, and tokens.",
            "#",
            f"# Schema-induction LLM key: read from ${key_var} at run (never stored here).",
        )


@dataclass(frozen=True)
class ExtractContribution(StageContribution):
    """``extract`` — fused NER+RE in ONE pass (GLiNER2).

    T2: the config labels are a PER-CHUNK RELEVANCE floor — a chunk that induced its own types is
    scored against THOSE ALONE (zero-shot precision degrades with label count); a chunk that induced
    nothing falls back to these. ``label_floor: universal`` on the extract config restores the
    config ∪ induced union. T1: in predefined-only mode there is nothing to union — the config
    labels ARE the schema.

    W18 (ADR-0044): in ``extract_on: masked`` mode the extractor reads the REDACTED chunk stream
    (``depends_on: [redact]``, which sets each chunk's masked_content) so PII never reaches the KG;
    the default ``unmasked`` keeps the pre-W18 wiring, so the KG keeps the real entities.

    T-WIZ-1: ``max_len`` is DERIVED from the chunk budget (:func:`model_window_for`) and emitted on
    BOTH profiles. It used to be a per-profile literal — 768 on lite, *omitted entirely* on SOTA —
    which was safe only while the budget was frozen at 640. With the budget an operator answer, a
    fixed (or silently defaulted) window is the silent-truncation bug the pod measured: 768 chunker
    tokens reach up to 921 mdeberta tokens. T-WIZ-3: the micro-batch + intra-op knobs both
    Providers read ride here too (:func:`_add_gliner2_perf_knobs`).

    Overrides carry the profile's tuning: ``threshold`` (0.5 on lite, the 0.3 SOTA recall floor),
    the default label sets, and ``device`` (``None`` ⇒ the
    operator's resolved device; pinned to ``cuda`` on the GPU-pod profile). T7: the extractor
    buckets each batch by label set and runs ONE native batched forward per bucket, and gliner2's
    native ``compile`` rides its own cuda default — deliberately NOT pinned here, so the Provider
    decides per device and a stack emitted for a GPU box degrades correctly if it is later run on
    cpu.
    """

    threshold: float
    entity_defaults: tuple[str, ...]
    relation_defaults: tuple[str, ...]
    device: str | None = None
    name: str = "extract"
    why: str = "fused entity + relation extraction in one zero-shot pass"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock:
        upstream = "redact" if answers.extract_on == "masked" else ctx.anchor(ANCHOR_LABELS)
        window = model_window_for(answers.chunk_max_tokens)
        config: StageBlock = {
            "device": self.device or answers.device,
            "labels": _entity_labels(answers, self.entity_defaults),
            "relation_labels": _relation_labels(answers, self.relation_defaults),
            "threshold": self.threshold,
            "max_len": window,
        }
        _add_gliner2_perf_knobs(config, answers, window=window)
        ctx.add_fanin(FANIN_MENTIONS, self.name)
        return {
            "name": self.name,
            "capability": "fused_entity_relation",
            # ADR-0064: read off the ENGINE row, which is the same row `redact` reads — so the two
            # Stages cannot name Providers from packages that pin disjoint `gliner2` majors.
            "provider": engine_for(answers.engine).extract_provider,
            "depends_on": [upstream],
            "config": config,
        }

    def header_lines(self, answers: SetupAnswers, block: StageBlock | None) -> tuple[str, ...]:
        # Derived from the emitted wiring, not from the answers: the prose may only claim
        # mask-before-extract when `extract` genuinely reads the `redact` stream.
        if block is None or block.get("depends_on") != ["redact"]:
            return ()
        return (
            "#",
            "# extract_on: masked (W18, ADR-0044) — extraction reads the REDACTED chunk stream, so",
            "#   PII never reaches entity/relation extraction or the KG (the stricter privacy",
            "#   posture). `extract` depends_on `redact`. Default `unmasked` (KG keeps entities).",
        )


@dataclass(frozen=True)
class DetectedRefsContribution(StageContribution):
    """A deterministic, zero-dep cross-reference extractor — ONLY for a corpus that carries one.

    A document's MOST explicit relations often live in self-describing metadata headers and in
    inline body citations whose targets are *bare identifiers*, not entity-shaped surface strings.
    An RFC's ``Obsoletes: 7230`` header and its scattered ``RFC 7230`` body citations are the
    canonical case: entity-anchored zero-shot RE recovers NONE of them, so the induced knowledge
    graph misses the corpus's most explicit, load-bearing cross-reference edges.

    The wizard sniffs the corpus for a KNOWN family (``detect_header_family`` /
    ``detect_inline_family``) and stores the ready Stage config on the answers. This contribution
    wires it when — and only when — that detection fired: no detected family ⇒ ``None`` ⇒ the
    emitted YAML and the resulting KG are byte-identical to a run without the feature. Both
    extractors read the SAME chunk stream ``extract`` reads and compose beside it through disjoint
    id namespaces (``#href-`` / ``#iref-``); all of them feed Disambiguation, whose input is the
    concatenation of its parents' output.

    ``answers_field`` names which detection result this instance wires — the ONE thing that differs
    between the header-edge and inline-citation extractors.
    """

    answers_field: str
    provider: str
    name: str = "header_refs"
    why: str = "recover the corpus's explicit cross-references (deterministic, zero-dep)"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock | None:
        detected = getattr(answers, self.answers_field, None)
        if detected is None:
            return None
        ctx.add_fanin(FANIN_MENTIONS, self.name)
        return {
            "name": self.name,
            "capability": "fused_entity_relation",
            "provider": self.provider,
            "depends_on": [ctx.anchor(ANCHOR_LABELS)],
            "config": dict(detected),
        }


@dataclass(frozen=True)
class RedactContribution(StageContribution):
    """``redact`` — learned PII redaction, the PII half of the ONE optional toggle.

    W16 (ADR-0042): Redaction is a chunk→chunk transform — it consumes the induced CHUNK stream and
    sets each chunk's ``masked_content`` (each chunk fits the model window, so no PII is truncated).
    Depending on the label source keeps the induced pii_types UNIONing onto the FINANCIAL_PII floor
    (ADR-0038). ``export_corpus`` then reads THIS redacted chunk stream, so the exported corpus is
    masked (the pre-W16 doc-level ``depends_on: [parse, induce]`` shape is gone).

    T2: the blessed checkpoint is a GUARDRAILS+PII model, so the guard half is switched on too.
    Every chunk carries the universal guard floor (prompt injection / harmful content — a security
    invariant, never subsetted per chunk) UNIONed with whatever guard categories that chunk induced.
    A hit is FLAGGED as a RiskMarker in the corpus, never masked; the deterministic keyword
    ``content_screen`` Stage is a separate, earlier pre-filter.
    """

    device: str | None = None
    name: str = "redact"
    why: str = "detect PII and mask it into each chunk (ADR-0044 financial floor, guardrails on)"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock | None:
        if not answers.pii:
            return None
        # T-WIZ-1: the redactor runs the SAME mdeberta encoder the extractor does over the SAME
        # chunks, so its window is derived from the same chunk budget. A fixed 768 here would have
        # truncated the tail of a longer chunk — i.e. left PII in the exported corpus UNMASKED,
        # which is a correctness failure, not a throughput one.
        window = model_window_for(answers.chunk_max_tokens)
        engine = engine_for(answers.engine)
        config: StageBlock = {
            "device": self.device or answers.device,
            # T1: the ADR-0044 financial floor, plus any operator-fixed pii_types
            # (predefined-only fixes the RELEVANCE schema, never the SAFETY schema).
            "pii_labels": _pii_labels(answers),
            "max_len": window,
            "threshold": 0.5,
        }
        # ADR-0064: the guardrail screen is emitted only when the engine's redactor HAS a
        # guardrail-trained checkpoint. The GLiNER 2.5 family has none — asking a general
        # checkpoint for `prompt injection` zero-shot and recording the answer as a security
        # finding is a control that reports nothing while looking enabled — and
        # `redaction.gliner25` refuses the key outright, so emitting it would produce a config
        # that does not load. The deterministic `content_screen` Stage is unaffected and still
        # rides `answers.pii` on both engines.
        if engine.guardrails:
            config["guardrails"] = True
        _add_gliner2_perf_knobs(config, answers, window=window)
        return {
            "name": self.name,
            "capability": "redaction",
            "provider": engine.redact_provider,
            "depends_on": [ctx.anchor(ANCHOR_LABELS)],
            "config": config,
        }

    def header_lines(self, answers: SetupAnswers, block: StageBlock | None) -> tuple[str, ...]:
        if block is None:
            return ()
        # The redactor NAME is read off the block this very walk emitted — the ADR-0041 §7a drift
        # (header naming gliner2 while the generator emitted gliner_pii) is unrepresentable here.
        provider = block.get("provider")
        caveat: tuple[str, ...] = ()
        if provider == "redaction.gliner25":
            # ADR-0064. The prose must not describe this as the guardrails+PII redactor, because
            # it is not one: there is NO GLiNER 2.5 PII checkpoint, so it drives the GENERAL 2.5
            # checkpoint zero-shot. Saying so in the emitted file is the difference between an
            # operator who knows what their privacy posture rests on and one who does not.
            caveat = (
                "#   NOTE (ADR-0064): there is NO GLiNER 2.5 PII checkpoint, so this Redactor",
                "#   asks the GENERAL 2.5 checkpoint for PII labels ZERO-SHOT. No parity with",
                "#   `redaction.gliner2`'s task-trained checkpoint is claimed and its recall is",
                "#   UNMEASURED. Learned guardrail screening is NOT available on this engine",
                "#   (the deterministic `content_screen` Stage still runs); the ADR-0044",
                "#   financial floor below is model-independent and applies unchanged.",
            )
        return (
            *caveat,
            f"#   Redactor: `{provider}` (#197, ADR-0036 W9, ADR-0064) — the SAME GLiNER",
            "#   engine `extract` runs, so PII detection adds no second model library. Its",
            "#   `pii_labels` are the ADR-0044 financial floor — UNIVERSAL, applied to EVERY",
            "#   chunk and never subsetted (T2); the per-chunk induced pii_types UNION on top",
            "#   at run (ADR-0038), and the universal financial floor masks card/iban/ssn even",
            "#   when",
            "#   the model misses them. Any other Redaction Provider drops in behind the same",
            "#   Capability seam — swap `redact.provider` and nothing else moves.",
            "#   W16 (ADR-0042): `redact` is a CHUNK-LEVEL transform — it consumes the induced",
            "#   chunk stream and sets each chunk's `masked_content` (each chunk fits the model",
            "#   window, so no PII is truncated). export_corpus reads THAT redacted chunk stream",
            "#   and materializes `masked_content` — so the exported corpus is masked (PII-safe;",
            "#   `redact -> export_corpus`). No pre-W16 doc-truncation leak.",
        )


@dataclass(frozen=True)
class ProfilingContribution(StageContribution):
    """``profiling`` — the corpus-level statistical + quality features."""

    name: str = "profiling"
    why: str = "corpus-level density / readability / frequency features for the Quality Report"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock:
        return {
            "name": self.name,
            "capability": "profiling",
            "provider": "profiling.statistical",
            "depends_on": ["parse", "extract"],
        }


@dataclass(frozen=True)
class ConsolidateTypesContribution(StageContribution):
    """``consolidate_types`` — T3 (ADR-0054) corpus-level TYPE-vocabulary canonicalization.

    The resolution phase's type-side twin of the disambiguator. This stack induces its label schema
    PER CHUNK, which is what gives each chunk a small, relevant, uncapped label set (the recall +
    precision win, and the fix for the label-dimension OOM) — and what makes chunk A say ``org``
    where chunk B says ``organization`` and chunk C ``company``. This Stage folds that drift into
    ONE canonical vocabulary and remaps every mention + relation onto it BEFORE ``disambiguate``,
    because the entity resolver votes a cluster's type from its members' labels and gates its
    low-precision rungs on type compatibility: drifted types in ⇒ split clusters and mistyped
    entities out.

    The semantic rung is ON here (it is the only rung that can see ``org`` ≡ ``company``), riding
    the SAME Granite r2 embedder block the resolver already loads over the handful of type STRINGS —
    a negligible marginal cost, no second model. The raw induced label survives on each record and
    the alias map lands on the Quality Report, so the operator sees and can correct every fold.

    It CONSUMES the whole mention fan-in and stands in front of it, so the drifted labels cannot
    reach the KG by a side path.
    """

    name: str = "consolidate_types"
    why: str = "fold per-chunk type drift (org/organization/company) into ONE canonical vocabulary"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock:
        block: StageBlock = {
            "name": self.name,
            "capability": "type_consolidation",
            "provider": "type_consolidation.cascade",
            "depends_on": ctx.fanin(FANIN_MENTIONS),
            "config": {
                "use_embedding_merge": True,
                "embedder": gpu_embedder_block(),
            },
        }
        ctx.replace_fanin(FANIN_MENTIONS, [self.name])
        return block


@dataclass(frozen=True)
class EmbeddingDisambiguateContribution(StageContribution):
    """``disambiguate`` (lite) — learned surface+context linking, no hard-coded KB.

    Corpus-agnostic by construction: nothing about the operator's domain is baked in, so the same
    Stage works on a legal corpus and a maintenance-log corpus.
    """

    name: str = "disambiguate"
    why: str = "link mentions to canonical entities from surface form + context, no KB"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock:
        return {
            "name": self.name,
            "capability": "disambiguation",
            "provider": "disambiguation.embedding",
            "depends_on": ctx.fanin(FANIN_MENTIONS),
            "config": {
                "threshold": 0.90,
                "context_window": 160,
                "embedder": tiered_embedder_block(answers),
            },
        }


@dataclass(frozen=True)
class GlinkerDisambiguateContribution(StageContribution):
    """``disambiguate`` (enterprise-SOTA) — the neural GLinker linker (ADR-0046).

    L2→L3→L4→L0 linking plus the audited 5-strategy resolver, with the Granite r2 embedder attached
    as the opt-in STS rung. ``use_embedding_merge`` / ``min_similarity`` are deliberately NOT set —
    the Provider defaults (embedding-merge OFF, fuzzy + the distinguishing-token guard ON at 0.84)
    are precise by default and left to ride, which keeps the emitted config minimal and lets the
    Provider own its own tuning.
    """

    name: str = "disambiguate"
    why: str = "neural GLinker linking + the audited 5-strategy resolver"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock:
        return {
            "name": self.name,
            "capability": "disambiguation",
            "provider": "disambiguation.glinker",
            "depends_on": ctx.fanin(FANIN_MENTIONS),
            "config": {
                "device": SOTA_DEVICE,
                "context_window": 200,
                "threshold": 0.3,
                "use_reranker": True,
                "use_flash_deberta": True,
                "embedder": gpu_embedder_block(),
            },
        }


@dataclass(frozen=True)
class GraphContribution(StageContribution):
    """``graph`` — the canonical, evidence-linked knowledge graph."""

    name: str = "graph"
    why: str = "assemble the canonical, evidence-linked knowledge graph"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock:
        ctx.add_fanin(FANIN_KG, self.name)
        return {
            "name": self.name,
            "capability": "graph_assembly",
            "provider": "graph.canonical",
            "depends_on": ["disambiguate"],
            "config": {"corpus_id": STACK_NAME},
        }


@dataclass(frozen=True)
class GraphCompletionContribution(StageContribution):
    """``complete`` — ULTRA zero-shot inductive link prediction (enterprise-SOTA only).

    ``skip_if_unavailable``: the checkpoint is not vendored, so on a host without it the Stage
    skips-with-flag (no inferred edges) rather than failing the run — device honesty, not a silent
    pass. It joins the KG fan-in, so ``export_kg`` waits for the completed edges too.
    """

    name: str = "complete"
    why: str = "zero-shot inductive link prediction over the assembled graph (skip-with-flag)"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock:
        ctx.add_fanin(FANIN_KG, self.name)
        return {
            "name": self.name,
            "capability": "graph_completion",
            "provider": "graph_completion.ultra",
            "depends_on": ["graph"],
            "config": {
                "corpus_id": STACK_NAME,
                "scorer": "auto",
                "device": SOTA_DEVICE,
                "target_precision": 0.9,
                "skip_if_unavailable": True,
            },
        }


@dataclass(frozen=True)
class ExportKGContribution(StageContribution):
    """``export_kg`` — materialize the knowledge graph (asserted + any completed edges)."""

    name: str = "export_kg"
    why: str = "materialize the knowledge graph (nodes, edges, per-edge Evidence)"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock:
        return {
            "name": self.name,
            "capability": "export",
            "provider": "export.knowledge_graph",
            "depends_on": ctx.fanin(FANIN_KG),
            "config": {"basename": "graph"},
        }


@dataclass(frozen=True)
class ExportCorpusContribution(StageContribution):
    """``export_corpus`` — the corpus export: the record of the text that was processed.

    W16 (ADR-0042): with PII ON it reads the REDACTED chunk stream (``redact`` set each chunk's
    masked_content), so the exported corpus row is masked; with PII OFF there is no redact Stage,
    so it reads the induced chunk stream directly. Always a dense embedding (ADR-0017), so the
    exported row carries ``induced_labels`` + a vector.
    """

    gpu_embedder: bool = False
    name: str = "export_corpus"
    why: str = "materialize the processed corpus from the (redacted) chunk stream"

    def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock:
        chunk_source = "redact" if ctx.emitted("redact") else ctx.anchor(ANCHOR_LABELS)
        return {
            "name": self.name,
            "capability": "export",
            "provider": "export.jsonl_parquet",
            "depends_on": [chunk_source],
            "config": {
                "basename": "records",
                "embedder": _embedder_for(answers, gpu=self.gpu_embedder),
            },
        }


# --------------------------------------------------------------------------------------------
# The profiles
# --------------------------------------------------------------------------------------------


def _sota_prologue(answers: SetupAnswers, stack: AssembledStack) -> tuple[str, ...]:
    """The enterprise-SOTA profile paragraph — every claim read off an EMITTED block.

    The parse-path sentence follows the ``parse`` Stage's actual Provider, so the header cannot
    advertise a parse path the recipe did not emit — the drift ADR-0041 §7a recorded, made
    unrepresentable.
    """
    parse_block = stack.by_name.get("parse", {})
    if parse_block.get("provider") == PARSER_PLAINTEXT:
        parse_lines = [
            "#   born-digital text parser (parser.plaintext, CPU — the detected corpus is all",
            "#   .txt/.md/.jsonl, so no OCR endpoint is served or needed) +",
        ]
        parse_point = [
            "#   validate` (device honesty). The corpus is plain text, so `parse` decodes it",
            "#   on CPU (no OCR box); point `parse.provider` at parser.lighton_vllm for scans.",
        ]
    else:
        parse_lines = [
            "#   served-vLLM LightOn OCR (parser.lighton_vllm) +",
        ]
        parse_point = [
            "#   validate` (device honesty). Point `parse.config.base_url` at your vLLM OCR",
            "#   box.",
        ]
    return (
        "#",
        "# Profile: enterprise-sota — the pod-VALIDATED composed stack (ADR-0046,",
        "#   stacks/gpu-sota-glinker.yaml):",
        *parse_lines,
        "#   LLM schema induction + fused GLiNER2 entity+relation + the neural GLinker",
        "#   disambiguator + IBM Granite r2 embeddings + ULTRA graph completion. Every learned",
        "#   Stage runs on CUDA; on a CPU-only host they skip-with-flag at `latence stack",
        *parse_point,
    )


#: ``lite`` — the CPU-first, laptop-friendly default. The ONE toggle (``answers.pii``) adds/omits
#: exactly ``content_screen`` + ``redact``; every other Stage is the mandatory spine.
LITE_RECIPE = StackRecipe(
    profile=PROFILE_LITE,
    contributions=(
        SourceContribution(),
        IntakeScreenContribution(),
        ParseContribution(),
        ChunkContribution(),
        ContentScreenContribution(),
        InduceContribution(),
        ExtractContribution(
            threshold=0.5,
            entity_defaults=("organization", "person"),
            relation_defaults=("works for", "partner of"),
        ),
        RedactContribution(),
        ProfilingContribution(),
        EmbeddingDisambiguateContribution(),
        GraphContribution(),
        ExportKGContribution(),
        ExportCorpusContribution(),
    ),
)

#: ``enterprise-sota`` — the pod-VALIDATED composed stack (ADR-0046,
#: ``stacks/gpu-sota-glinker.yaml``): the chosen parse path → chunk[640] → per-chunk LLM schema
#: induction → fused GLiNER2 ER (+ any detected deterministic cross-reference extractors) → learned
#: PII redaction → profiling → type consolidation → the neural GLinker disambiguator → canonical KG
#: → ULTRA completion → KG export + corpus export.
#:
#: Deliberate delta from ``stacks/gpu-sota-glinker.yaml`` (#197): that hand-written stack predates
#: the decision and still wires ``redaction.gliner_pii``; the wizard emits the ONE blessed
#: :data:`BLESSED_REDACT_PROVIDER` in every profile. Both Providers remain supported behind the same
#: ``PIIDetector`` seam — the profile just no longer picks a different one per tier.
SOTA_RECIPE = StackRecipe(
    profile=PROFILE_SOTA,
    contributions=(
        SourceContribution(),
        IntakeScreenContribution(),
        ParseContribution(served_ocr=True),
        ChunkContribution(),
        ContentScreenContribution(),
        InduceContribution(),
        ExtractContribution(
            threshold=0.3,
            entity_defaults=("person", "organization"),
            relation_defaults=("works for", "partner of"),
            device=SOTA_DEVICE,
        ),
        DetectedRefsContribution(
            answers_field="header_refs",
            provider="fused_entity_relation.header_refs",
            name="header_refs",
            why="recover explicit metadata-header cross-references (an RFC's Obsoletes:/Updates:)",
        ),
        DetectedRefsContribution(
            answers_field="inline_refs",
            provider="fused_entity_relation.inline_refs",
            name="inline_refs",
            why="recover inline body citations (an RFC's scattered \"RFC 7230\" references)",
        ),
        RedactContribution(device=SOTA_DEVICE),
        ProfilingContribution(),
        ConsolidateTypesContribution(),
        GlinkerDisambiguateContribution(),
        GraphContribution(),
        GraphCompletionContribution(),
        ExportKGContribution(),
        ExportCorpusContribution(gpu_embedder=True),
    ),
    prologue=_sota_prologue,
)

_RECIPES: Mapping[str, StackRecipe] = MappingProxyType(
    {PROFILE_LITE: LITE_RECIPE, PROFILE_SOTA: SOTA_RECIPE}
)


def recipe_for(profile: str) -> StackRecipe:
    """The :class:`StackRecipe` for ``profile`` — :data:`LITE_RECIPE` for anything unrecognised.

    Unrecognised profiles fall back to ``lite`` rather than raising, exactly as the pre-C7 generator
    did (the CLI validates ``--profile`` against :data:`_KNOWN_PROFILES` before it ever gets here,
    so a bad value cannot reach a saved config).
    """
    return _RECIPES.get(profile, LITE_RECIPE)
