"""The deterministic, quality-baked stack generator behind ``latence setup`` (W15, ADR-0041).

Given a resolved :class:`SetupAnswers` (from the interactive prompts OR the ``--non-interactive``
flags — the SAME struct either way), :func:`build_stack_config` emits a full-spine stack YAML that
reproduces the pod-validated GOOD stack (see ``docs/TUTORIAL-fresh-pod-walkthrough.md`` §3):

    source → intake_screen → parse → chunk[768] → [content_screen] → induce[chunk]
      → extract[gliner2, thr 0.5] → [redact[gliner2, financial-PII]] → profiling
      → disambiguate → graph → export_kg + export_corpus

The *mandatory spine* is always present; the ONE optional toggle is PII redaction + guardrail
screening (``pii``), which adds/omits exactly the ``content_screen`` + ``redact`` stages and
nothing else. The generator is PURE — it never reads the environment, a clock, or a random source —
so identical answers yield a **byte-identical** file (the ``--non-interactive`` determinism
contract, for CI/repro).

The exported corpus is built from the chunk stream, so it carries ``induced_labels`` + a dense
embedding. With PII ON the ``redact`` Stage is a W16 CHUNK-LEVEL transform (ADR-0042): it consumes
the induced chunk stream and sets each chunk's ``masked_content`` (each chunk ≤ the model window,
so no PII is truncated), and ``export_corpus`` reads THAT redacted chunk stream — so the exported
corpus row is genuinely PII-safe (``redact → export_corpus``).

**What lives here (C7).** This module owns the two halves the operator's answers pass through:

1. *reading the corpus* — one bounded, never-raising walk that reports the file types, the largest
   file, and any recognised structured-header cross-reference family (:func:`scan_source`,
   :func:`recommend_parser`, :func:`detect_header_family`, :func:`detect_inline_family`);
2. *assembling and rendering* — walk the profile's recipe, wrap the Stage blocks in the pipeline
   mapping, render the comment header FROM the assembled stack, and dry-validate the result.

Every per-Stage decision (which Provider, which threshold, which ``depends_on``, and the comment
prose describing it) lives with that Stage in :mod:`latence_core.setup.recipe`, so a change to one
Stage is a change to one declaration instead of an edit to two parallel builders and a header
renderer 900 lines away — the drift ADR-0041 §7a recorded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from latence_core.setup.recipe import (
    _DEFAULT_EXTENSIONS,
    _KNOWN_CHUNK_STRATEGIES,
    _KNOWN_ENGINES,
    BLESSED_REDACT_PROVIDER,
    CHUNK_BUDGET_FROM_ENGINE,
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_STRATEGY_AUTO,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_OCR_BASE_URL,
    DEFAULT_OCR_MODEL,
    DEFAULT_RASTER_DPI,
    DEFAULT_RASTER_MAX_LONGEST_DIM,
    DOMAIN_HINT_MAX_CHARS,
    ENGINE_GLINER2,
    FINANCIAL_PII_LABELS,
    INTAKE_HEADROOM,
    INTAKE_MAX_BYTES_CEILING,
    INTAKE_MAX_BYTES_FLOOR,
    PARSER_OCR,
    PARSER_OCR_VLLM,
    PARSER_PDF_TEXT,
    PARSER_PLAINTEXT,
    PARSER_UNIVERSAL,
    PROFILE_LITE,
    PROFILE_SOTA,
    SOTA_DEVICE,
    SOTA_OCR_API_KEY,
    SOTA_OCR_MAX_CONCURRENCY,
    STACK_NAME,
    AssembledStack,
    RecipeWiringError,
    SetupConfigError,
    StackRecipe,
    StageBlock,
    StageContribution,
    engine_for,
    intake_max_bytes,
    llm_key_env_var,
    model_window_for,
    normalize_domain_hint,
    recipe_for,
)

# Re-exported so `latence_core.setup.wizard` stays the ONE import site the CLI, the stacks docs and
# the test-suite already know. The declarations themselves moved to `.recipe` (C7); their names,
# values and meanings did not.
__all__ = [
    "BLESSED_REDACT_PROVIDER",
    "CHUNK_MAX_TOKENS",
    "CHUNK_MIN_TOKENS",
    "CHUNK_OVERLAP_TOKENS",
    "DEFAULT_LLM_BASE_URL",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_OCR_BASE_URL",
    "DEFAULT_OCR_MODEL",
    "DEFAULT_RASTER_DPI",
    "DEFAULT_RASTER_MAX_LONGEST_DIM",
    "DOMAIN_HINT_MAX_CHARS",
    "FINANCIAL_PII_LABELS",
    "INTAKE_HEADROOM",
    "INTAKE_MAX_BYTES_CEILING",
    "INTAKE_MAX_BYTES_FLOOR",
    "PARSER_OCR",
    "PARSER_OCR_VLLM",
    "PARSER_PDF_TEXT",
    "PARSER_PLAINTEXT",
    "PARSER_UNIVERSAL",
    "PREDEFINED_LABEL_SLOTS",
    "PROFILE_LITE",
    "PROFILE_SOTA",
    "SOTA_DEVICE",
    "SOTA_OCR_API_KEY",
    "SOTA_OCR_MAX_CONCURRENCY",
    "STACK_NAME",
    "AssembledStack",
    "CorpusScan",
    "RecipeWiringError",
    "SetupAnswers",
    "SetupConfigError",
    "StackRecipe",
    "StageBlock",
    "StageContribution",
    "build_pipeline_dict",
    "build_stack_config",
    "detect_extensions",
    "detect_header_family",
    "detect_inline_family",
    "intake_max_bytes",
    "llm_key_env_var",
    "normalize_domain_hint",
    "parse_label_candidates",
    "parse_predefined_labels",
    "recipe_for",
    "recommend_parser",
    "recommend_sota_parser",
    "scan_source",
    "validate_config_text",
]

# The parser / profile flag vocabularies the CLI validates `--parser` and `--profile` against.
# They live here, beside the recommenders that pick from them, rather than with the recipes: they
# are about what an OPERATOR may ask for, not about what a Stage contributes.
_KNOWN_PARSERS = frozenset({PARSER_UNIVERSAL, PARSER_PDF_TEXT, PARSER_OCR})
_KNOWN_SOTA_PARSERS = frozenset({PARSER_PLAINTEXT, PARSER_OCR_VLLM})
_KNOWN_PROFILES = frozenset({PROFILE_LITE, PROFILE_SOTA})

# Extensions that read as scanned/image inputs (OCR territory) vs born-digital PDF.
_IMAGE_EXTS = frozenset({"png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "webp"})
_OFFICE_EXTS = frozenset({"docx", "doc", "pptx", "ppt", "xlsx", "xls", "odt", "rtf", "html", "htm"})

# Extensions the plaintext Parser decodes losslessly (bytes -> text/markdown, no OCR, no office
# rendering). A corpus whose file types are ALL in this set is born-digital text — the OCR VLM
# would only mangle it — so the enterprise-SOTA wizard routes it to `parser.plaintext`.
_PLAINTEXT_EXTS = frozenset({"txt", "text", "md", "markdown", "mdown", "jsonl", "ndjson"})


@dataclass(frozen=True)
class CorpusScan:
    """What ONE best-effort walk of the source folder observed (file types + the largest file).

    ``scanned`` is the honesty bit: ``False`` means the walk could not observe the corpus at all —
    a non-local Storage URI, a missing/unreadable folder, or a folder with no files — so
    ``extensions`` is the broad default set and ``largest_bytes`` is 0 (the caller warns and the
    generator falls back to ``INTAKE_MAX_BYTES_FLOOR``). Never raises.
    """

    extensions: tuple[str, ...]
    largest_bytes: int
    scanned: bool


def scan_source(path: str | Path) -> CorpusScan:
    """Walk ``path`` once and report its file types + largest file size (best-effort, never raises).

    ONE walk answers both questions the wizard asks of a corpus — which parser it needs (the
    extensions) and how big its documents actually are (the intake cap) — so a wizard run never
    reads the folder twice or disagrees with itself. A non-local URI (``s3://…``), a missing folder,
    an unreadable tree, or an empty folder all degrade to ``scanned=False`` + the broad default
    extensions rather than failing: setup still emits a config, and the caller surfaces the warning.
    Deterministic (extensions sorted; the largest size is a property of the corpus, not the walk).
    """
    p = Path(str(path).removeprefix("file://"))
    try:
        if not p.exists() or not p.is_dir():
            return CorpusScan(_DEFAULT_EXTENSIONS, 0, scanned=False)
        found: set[str] = set()
        largest = 0
        seen_file = False
        for child in p.rglob("*"):
            try:
                if not child.is_file():
                    continue
                size = child.stat().st_size
            except OSError:
                # An unreadable / vanished / broken-symlink entry is skipped, never a crash: the
                # scan is advisory, and a partial observation still beats no observation.
                continue
            seen_file = True
            if child.suffix:
                found.add(child.suffix.lstrip(".").lower())
            largest = max(largest, size)
    except OSError:
        return CorpusScan(_DEFAULT_EXTENSIONS, 0, scanned=False)
    if not seen_file:
        return CorpusScan(_DEFAULT_EXTENSIONS, 0, scanned=False)
    extensions = tuple(sorted(found)) if found else _DEFAULT_EXTENSIONS
    return CorpusScan(extensions, largest, scanned=True)


#: The three label slots a predefined (fixed) schema may declare — the same slots the inducer
#: induces, so predefined-only is a drop-in replacement for induction rather than a parallel
#: vocabulary. Any other key is REJECTED (a typo like ``entities`` would otherwise silently
#: contribute nothing and the operator would ship a stack with the profile defaults).
PREDEFINED_LABEL_SLOTS: tuple[str, ...] = ("entity_types", "relation_types", "pii_types")


def parse_predefined_labels(raw: object) -> dict[str, list[str]]:
    """Validate an operator-supplied FIXED label schema into the generator's normalized mapping.

    Predefined-only mode (T1) is the escape hatch for a corpus with a KNOWN schema — a compliance
    taxonomy, a fixed entity set — where discovery is not merely unnecessary but undesirable. The
    schema arrives as a JSON/YAML mapping of :data:`PREDEFINED_LABEL_SLOTS` to label lists; this
    turns it into ``{slot: [label, ...]}`` with empty slots omitted.

    Every failure is TYPED and loud (:class:`SetupConfigError`), never coerced: a non-mapping, an
    unknown slot, a non-list slot, a non-string label, or a schema that declares no labels at all
    all raise. Silently accepting a typo'd slot would emit a stack that looks operator-configured
    but runs the profile's DEFAULT labels — the worst possible failure for a fixed-taxonomy corpus.

    Labels keep the operator's own ORDER (deduped, first-seen) — the generator never sorts them
    behind their back, so the emitted YAML is a byte-stable function of what was supplied.
    """
    if not isinstance(raw, dict):
        slots = ", ".join(PREDEFINED_LABEL_SLOTS)
        msg = (
            f"predefined labels must be a mapping of {{{slots}}} to lists of strings, "
            f"got {type(raw).__name__}."
        )
        raise SetupConfigError(msg)
    unknown = sorted(k for k in raw if k not in PREDEFINED_LABEL_SLOTS)
    if unknown:
        msg = (
            f"predefined labels carry unknown slot(s) {unknown}; "
            f"expected only {list(PREDEFINED_LABEL_SLOTS)}."
        )
        raise SetupConfigError(msg)
    parsed: dict[str, list[str]] = {}
    for slot in PREDEFINED_LABEL_SLOTS:
        if slot not in raw:
            continue
        values = raw[slot]
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            msg = f"predefined labels slot {slot!r} must be a list of strings."
            raise SetupConfigError(msg)
        kept: list[str] = []
        for value in values:
            label = value.strip()
            if label and label not in kept:
                kept.append(label)
        if kept:
            parsed[slot] = kept
    if not parsed:
        msg = (
            "predefined labels declare no usable labels — predefined-only mode would emit a stack "
            "with NO schema at all. Supply at least one label, or drop the flag to induce."
        )
        raise SetupConfigError(msg)
    return parsed


def parse_label_candidates(raw: object) -> dict[str, list[str] | dict[str, list[str]]]:
    """Validate an operator-supplied CANDIDATE TYPE SPACE for the ``label_inducer.lexicon`` inducer.

    T-WIZ-2: the non-LLM induction path. The pool is the artifact
    ``latence-label-profile --candidates-out`` writes — ``{slot: {type name: [surface cue, ...]}}``
    over :data:`PREDEFINED_LABEL_SLOTS`, with a slot that selected nothing OMITTED rather than
    emitted empty — and it is INLINED into the emitted stack as the Provider's ``candidates``
    config, so the generated stack is self-contained (no second file to ship to the pod).

    Both of the Provider's per-slot forms are accepted and preserved as written: a plain LIST of
    type names (the name is its own cue) or a ``{type: [cue, ...]}`` MAPPING. This is exactly why
    :func:`parse_predefined_labels` is not the vehicle for it — that one accepts flat name lists
    only, so it structurally cannot carry cues, which are most of a mined pool's value.

    Every failure is TYPED and loud (:class:`SetupConfigError`), never coerced — the same posture
    :func:`parse_predefined_labels` takes and for the same reason: a typo'd slot that silently
    contributed nothing would emit a stack that LOOKS pool-configured while inducing from a pool
    that is missing a third of itself. Names and cues keep the operator's own ORDER (blank entries
    dropped), so the emitted YAML is a byte-stable function of the file.
    """
    if not isinstance(raw, dict) or not raw:
        slots = ", ".join(PREDEFINED_LABEL_SLOTS)
        msg = (
            f"label candidates must be a non-empty mapping of {{{slots}}} to either a list of type "
            f"names or a {{type: [cue, ...]}} mapping, got {type(raw).__name__}."
        )
        raise SetupConfigError(msg)
    unknown = sorted(str(k) for k in raw if k not in PREDEFINED_LABEL_SLOTS)
    if unknown:
        msg = (
            f"label candidates carry unknown slot(s) {unknown}; "
            f"expected only {list(PREDEFINED_LABEL_SLOTS)}."
        )
        raise SetupConfigError(msg)
    parsed: dict[str, list[str] | dict[str, list[str]]] = {}
    for slot in PREDEFINED_LABEL_SLOTS:
        if slot not in raw:
            continue
        slot_value = _parse_candidate_slot(slot, raw[slot])
        # An empty slot is a CONFIGURATION ERROR at the Provider, so an operator who supplied one
        # (or whose profiler selected nothing for it) gets the same OMITTED slot the profiler's own
        # artifact writes, not a stack that raises hours later at construction.
        if slot_value:
            parsed[slot] = slot_value
    if not parsed:
        msg = (
            "label candidates declare no usable types — the lexicon inducer could only ever induce "
            "nothing. Supply at least one type, or drop the flag to induce with an LLM."
        )
        raise SetupConfigError(msg)
    return parsed


def _parse_candidate_slot(slot: str, value: object) -> list[str] | dict[str, list[str]]:
    """One slot of the candidate space, in whichever of the two forms the operator wrote it."""
    if isinstance(value, list):
        names: list[str] = []
        for name in value:
            if not isinstance(name, str):
                msg = f"label candidates slot {slot!r} (list form) must hold type-name strings."
                raise SetupConfigError(msg)
            cleaned = name.strip()
            if cleaned and cleaned not in names:
                names.append(cleaned)
        return names
    if isinstance(value, dict):
        # A type declared with NO cues is kept, not dropped: the Provider reads an empty cue list
        # as "the type name is its own cue", which is what the list form means — so the mapping
        # form can express both without the operator having to switch forms mid-file.
        types: dict[str, list[str]] = {}
        for name, cues in value.items():
            if not isinstance(name, str):
                msg = f"label candidates slot {slot!r} must be keyed by type-name strings."
                raise SetupConfigError(msg)
            if not isinstance(cues, list) or not all(isinstance(c, str) for c in cues):
                msg = (
                    f"label candidates slot {slot!r}: type {name!r} must map to a list of surface "
                    f"cue strings (an empty list means 'the type name is its own cue')."
                )
                raise SetupConfigError(msg)
            cleaned_name = name.strip()
            if not cleaned_name or cleaned_name in types:
                continue
            kept: list[str] = []
            for cue in cues:
                text = cue.strip()
                if text and text not in kept:
                    kept.append(text)
            types[cleaned_name] = kept
        return types
    msg = (
        f"label candidates slot {slot!r} must be a list of type names or a "
        f"{{type: [cue, ...]}} mapping, got {type(value).__name__}."
    )
    raise SetupConfigError(msg)


@dataclass(frozen=True)
class SetupAnswers:
    """The fully-resolved answers the generator turns into a stack config.

    Interactive prompts and ``--non-interactive`` flags both produce this SAME struct, and the
    generator is a pure function of it — so two identical structs yield byte-identical YAML.
    """

    source: str  # Storage URI (or bare path) of the input folder.
    storage: str  # Storage URI for state + exports.
    parser: str  # One of the _KNOWN_PARSERS entry-point names.
    device: str = "auto"  # auto | cpu | cuda — threaded onto every learned/GPU Stage.
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model: str = DEFAULT_LLM_MODEL
    # The profile/tier: `lite` (default, CPU-first, unchanged) or `enterprise-sota` (the
    # pod-validated composed stack). The generator picks the profile's RECIPE from this; other
    # fields are shared.
    profile: str = PROFILE_LITE
    # ADR-0064 — the extraction ENGINE. ONE answer drives BOTH the fused extractor and the
    # redactor, because their Provider packages pin mutually exclusive `gliner2` majors: a stack
    # naming `fused_entity_relation.gliner25` beside `redaction.gliner2` would validate and then
    # fail at `pip install`. Reading both Stages off one `ExtractionEngine` row makes that pair
    # unrepresentable rather than merely discouraged. `gliner2` (the default) is unchanged
    # behaviour; `gliner25` selects the GLiNER 2.5 family and, with `chunk_strategy: auto`, its
    # coherent page-preferred chunking.
    engine: str = ENGINE_GLINER2
    # `auto` (default) takes the engine's own chunker; `window` / `page` override it. The
    # override exists because page-preferred chunking is useful with the 1.x engine too, and
    # because an operator whose corpus is unpaginated may prefer the window strategy explicitly.
    chunk_strategy: str = CHUNK_STRATEGY_AUTO
    # Enterprise-SOTA served-OCR endpoint (only consumed by the SOTA profile's
    # `parser.lighton_vllm` parse Stage; ignored by the lite profile).
    ocr_base_url: str = DEFAULT_OCR_BASE_URL
    ocr_model: str = DEFAULT_OCR_MODEL
    pii: bool = True  # The ONE optional toggle: PII redaction + guardrail screening.
    # W18 (ADR-0044) advanced option: which chunk-text variant extraction reads.
    #   "unmasked" (default) — extraction reads the raw chunk text; the KG keeps real entities
    #                          (byte-identical to pre-W18).
    #   "masked"   — extraction reads the REDACTED chunk stream so PII never reaches the KG at all
    #                (the stricter posture; requires PII redaction ON).
    extract_on: str = "unmasked"
    # Optional free-text domain hint. Recorded as a header comment AND threaded into schema
    # induction (the `induce` Stage's `domain` key), so it genuinely steers the induced schema
    # rather than only annotating the file. Empty ⇒ neither is emitted (byte-identical to a run
    # without a hint).
    domain: str = ""
    # The size (bytes) of the LARGEST file observed under ``source`` at setup — from
    # :func:`scan_source`. It SIZES the `intake_screen` oversize cap (see ``intake_max_bytes``)
    # so the scanned corpus is actually admitted instead of being wholly quarantined at intake.
    # 0 (the default, and what an unreadable / empty / non-local source yields) keeps the pre-fix
    # flat 2 MiB floor.
    largest_source_bytes: int = 0
    extensions: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_EXTENSIONS)
    # dogfood-3-inspect: the detected structured-header cross-reference family's `header_refs` Stage
    # config (from :func:`detect_header_family`), or ``None`` when the corpus carries no recognised
    # metadata-header convention. When set (enterprise-SOTA only — only that recipe carries the
    # contribution), a deterministic `fused_entity_relation.header_refs` extraction Stage recovers
    # the corpus's explicit header cross-references (an RFC's `Obsoletes:`/`Updates:` edges) beside
    # the learned extractor. ``None`` ⇒ NO such Stage is emitted, so config + KG are byte-identical
    # to before.
    header_refs: dict[str, object] | None = None
    # The detected structured-header family's `inline_refs` Stage config (from
    # :func:`detect_inline_family`), or ``None`` when the corpus carries no recognised family. When
    # set (enterprise-SOTA only) a deterministic `fused_entity_relation.inline_refs` extraction
    # Stage recovers the corpus's INLINE body citations (an RFC's scattered "RFC 7230" references —
    # the dogfood measured gap the header-only extractor misses) beside the learned extractor AND
    # the header_refs Stage. ``None`` ⇒ NO such Stage is emitted, so config + KG are byte-identical
    # to before.
    inline_refs: dict[str, object] | None = None
    # T1 — PREDEFINED-ONLY mode: the operator supplies a FIXED label schema and opts out of
    # induction entirely. ``None`` (the default) ⇒ induction is on and per-chunk. When set (a
    # ``{"entity_types": [...], "relation_types": [...], "pii_types": [...]}`` mapping, any slot
    # optional) the generator emits **NO `induce` Stage at all** — not a disabled one, not a
    # short-circuited Provider: the stage is absent, so the run makes ZERO induction LLM calls and
    # every extraction Stage runs each chunk against the operator's fixed labels (the config-label
    # floor, which is what an inducer-free pipeline already uses). This is the escape hatch for a
    # corpus with a KNOWN schema — a compliance taxonomy, a fixed entity set — where discovery is
    # not merely unnecessary but undesirable. The ADR-0044 financial-PII floor is still UNIONed onto
    # the redactor's labels: a fixed relevance schema is an operator's choice, the safety floor is
    # not (README invariant 3).
    predefined_labels: dict[str, list[str]] | None = None
    # T-WIZ-2 — the NON-LLM induction path: an operator-declared CANDIDATE TYPE SPACE (typically
    # the artifact `latence-label-profile --candidates-out` mines from the corpus). ``None`` (the
    # default) ⇒ induction is the frontier `label_inducer.llm` path, byte-identical to before. When
    # set, the `induce` Stage is `label_inducer.lexicon` instead — a deterministic, CPU-only,
    # ZERO-NETWORK tagger — with this pool inlined as its `candidates` config, so the emitted stack
    # is self-contained and makes no induction call by construction (there is no endpoint in it to
    # call). Mutually exclusive with ``predefined_labels``: that one turns induction OFF entirely,
    # this one changes HOW it induces.
    label_candidates: dict[str, list[str] | dict[str, list[str]]] | None = None
    # The lexicon inducer's two precision knobs (only read on that path). ``min_hits`` is how many
    # DISTINCT cues of a type must occur before it is induced — raise it on a noisy corpus where
    # single-word cues over-fire. ``whole_word`` matches cues on word boundaries so `iban` does not
    # fire inside `Caribbean`; German compounding is the reason it must be switchable off
    # (`Rechnungsnummernkreis` contains `Rechnungsnummer`).
    label_min_hits: int = 1
    label_whole_word: bool = True
    # T-WIZ-1 — the chunk token budget. It was a literal (640/80/8) in the recipe, so retuning a
    # run meant hand-editing the emitted YAML. The default is the GLiNER2 window (768); the
    # gliner-family Stages derive their `max_len` from THIS value (`recipe.model_window_for`),
    # because the chunker counts ~1.2x FEWER tokens than the mdeberta encoder and a budget raised
    # without the window silently truncates (pod-measured: 768 chunker -> max 921 mdeberta).
    # ADR-0064: ``0`` (:data:`CHUNK_BUDGET_FROM_ENGINE`, the default) means "not answered — take
    # the ENGINE's own budget", resolved in ``__post_init__`` so every caller of the public
    # ``build_stack_config`` gets the pairing the engine was designed around: 768 for `gliner2`
    # (unchanged), 1024 for `gliner25`'s page-preferred chunking. An explicit value always wins.
    chunk_max_tokens: int = CHUNK_BUDGET_FROM_ENGINE
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS
    chunk_min_tokens: int = CHUNK_MIN_TOKENS
    # T-OCR-3 — the rasterisation policy, LightOn's published 200 DPI / 1540 px longest side. These
    # ARE the defaults and stay them; the ticket is that they were unreachable from setup on every
    # parser path. A value equal to the default is not emitted at all, so an unset flag leaves the
    # `parse` block byte-identical.
    ocr_dpi: int = DEFAULT_RASTER_DPI
    ocr_max_longest_dim: int = DEFAULT_RASTER_MAX_LONGEST_DIM
    # T-WIZ-3 — the GLiNER2 performance knobs both gliner2 Providers read
    # (`providers/perf.py::resolve_microbatch_bounds` / `resolve_intra_op_threads`) and the wizard
    # emitted none of. ``None`` ⇒ not emitted ⇒ the Provider's own default rides, EXCEPT that
    # `microbatch_max_texts` carries a window-derived default so a wider model window cannot widen
    # the padded forward past the pod-safe 4x768 token positions (DECISIONS.md §D3: 21.6 GiB peak,
    # 2.4 GiB of headroom on the 24 GB card). An explicit answer always wins.
    microbatch_max_texts: int | None = None
    microbatch_token_budget: int | None = None
    intra_op_threads: int | None = None

    def __post_init__(self) -> None:
        """Reject the ONE contradictory pair of answers, here, where every caller passes through.

        ``predefined_labels`` means "emit no ``induce`` Stage at all"; ``label_candidates`` means
        "induce per chunk, from this pool, without an LLM". They are different answers to the same
        question, so honouring either silently would emit a stack that LOOKS configured by both
        flags and obeys one. Raised from the struct rather than only from the CLI because the
        generator is a public entry point (``build_stack_config(answers)``) — a contradiction that
        only the CLI caught would reach it unchecked.
        """
        if self.predefined_labels is not None and self.label_candidates is not None:
            msg = (
                "predefined labels and label candidates are mutually exclusive: predefined labels "
                "emit NO `induce` stage at all, while label candidates induce per chunk from your "
                "pool with the zero-network lexicon inducer. Choose one."
            )
            raise SetupConfigError(msg)
        if self.label_min_hits < 1:
            msg = f"label candidates 'min_hits' must be >= 1, got {self.label_min_hits}."
            raise SetupConfigError(msg)
        if self.engine not in _KNOWN_ENGINES:
            msg = (
                f"Unknown extraction engine {self.engine!r}; expected one of "
                f"{sorted(_KNOWN_ENGINES)}."
            )
            raise SetupConfigError(msg)
        if self.chunk_strategy not in _KNOWN_CHUNK_STRATEGIES:
            msg = (
                f"Unknown chunk strategy {self.chunk_strategy!r}; expected one of "
                f"{sorted(_KNOWN_CHUNK_STRATEGIES)}."
            )
            raise SetupConfigError(msg)
        # ADR-0064 — the budget must FIT the engine's checkpoint window, checked here rather than
        # discovered at run time. The gliner-family encoders truncate SILENTLY past their window:
        # for `extract` that loses tail entities and relations, and for `redact` it leaves PII in
        # the exported corpus UNMASKED, which is a correctness failure rather than a throughput
        # one. `build_stack_config(answers)` is a public entry point, so the refusal lives on the
        # struct where every caller passes through, not only on the CLI flag.
        engine = engine_for(self.engine)
        if self.chunk_max_tokens == CHUNK_BUDGET_FROM_ENGINE:
            # ``frozen=True`` is about an operator not mutating an answer set after the fact, not
            # about the struct being unable to finish constructing itself; this is the ordinary
            # dataclass idiom for a derived default and it runs before any reader sees the field.
            object.__setattr__(self, "chunk_max_tokens", engine.chunk_max_tokens)
        window = model_window_for(self.chunk_max_tokens)
        if engine.max_window is not None and window > engine.max_window:
            msg = (
                f"a {self.chunk_max_tokens}-token chunk budget needs a {window}-token model "
                f"window, but the {self.engine!r} engine's checkpoint window is "
                f"{engine.max_window}. The chunker counts ~1.2x FEWER tokens than the mdeberta "
                "encoder, so the window is derived from the budget and the encoder truncates "
                "silently past it. Lower the chunk budget, or choose an engine with a wider "
                "window."
            )
            raise SetupConfigError(msg)


def detect_extensions(path: str | Path) -> tuple[str, ...]:
    """Return the sorted, de-duplicated file extensions present under ``path`` (no dot, lowercased).

    A best-effort local scan used to seed the ``source`` Stage's ``extensions`` and to recommend a
    parser. A non-existent / empty folder returns the broad default set so a config still generates
    (the operator can prune it). Deterministic (sorted) so it never perturbs the byte-identical
    contract. The extensions half of :func:`scan_source` — use that when you also need the corpus's
    file sizes (it answers both from ONE walk).
    """
    return scan_source(path).extensions


def recommend_parser(extensions: tuple[str, ...]) -> str:
    """Recommend a parser from the detected extensions (the wizard's default, always overridable).

    * only born-digital PDF (no images/office) → ``parser.pdfplumber`` (CPU, no OCR needed);
    * scans / images present → ``parser.lighton`` (OCR VLM);
    * anything mixed / office / unknown → ``parser.render`` (the universal renderer).
    """
    exts = {e.lower() for e in extensions}
    if exts and exts <= {"pdf"}:
        return PARSER_PDF_TEXT
    if exts & _IMAGE_EXTS and not (exts & _OFFICE_EXTS):
        return PARSER_OCR
    return PARSER_UNIVERSAL


def recommend_sota_parser(extensions: tuple[str, ...]) -> str:
    """Recommend the enterprise-SOTA parser from the detected extensions (the wizard's default).

    The SOTA profile is a GPU-pod stack whose production parse path is the served-vLLM OCR endpoint
    (``parser.lighton_vllm``) — but that path ONLY rasterizes pages to images and POSTs them to the
    OCR server; it has no plain-text passthrough. So when the scanned corpus is ENTIRELY
    born-digital text (every detected extension in :data:`_PLAINTEXT_EXTS`), the OCR VLM is wrong
    and, on a pod where the OCR box is intentionally not served, would degrade every document to a
    PARSE_ERROR. Such a corpus is routed to the CPU :data:`PARSER_PLAINTEXT` parser instead; any
    other corpus (PDF/office/image/mixed/unknown) keeps the served-OCR path. Always overridable.
    """
    exts = {e.lower() for e in extensions}
    if exts and exts <= _PLAINTEXT_EXTS:
        return PARSER_PLAINTEXT
    return PARSER_OCR_VLLM


# --- structured-header cross-reference recovery (dogfood-3-inspect) -----------------------------
#
# A document's MOST explicit relations often live in self-describing metadata headers whose targets
# are *bare identifiers*, not entity-shaped surface strings, and whose subject is the document
# itself (declared once, far away, in the front matter). An RFC's ``Obsoletes: 7230, 7231`` /
# ``Updates: 3864`` lines are the canonical case: ``7230`` is never tagged as an entity and the
# subject (``RFC 9110``) is only in the ``Request for Comments: 9110`` header — so entity-anchored
# zero-shot RE (the learned ``fused_entity_relation.gliner2``) recovers NONE of them and the induced
# knowledge graph misses the corpus's most explicit, load-bearing cross-reference edges. The
# deterministic, zero-dep ``fused_entity_relation.header_refs`` Provider exists precisely to recover
# them (self id + header targets → synthesized mentions + directed edges) — but only if the wizard
# WIRES it. This is that detection: the wizard sniffs the corpus for a KNOWN structured-header
# family and, when it positively detects one, hands the ready config to the enterprise-SOTA recipe's
# `header_refs` contribution, which composes beside the learned extractor (disjoint ``#href-`` id
# namespace; both feed the same Disambiguation input). A corpus with NO recognised header family
# gets NO such Stage — its emitted YAML and resulting KG stay byte-identical to a run without the
# feature.
#
# Each family pairs a corpus SIGNATURE (the ``self`` header + at least one relation header must both
# appear in sampled text before we wire it) with the ready ``header_refs`` config. The relation
# ``header`` patterns mirror the RFC-editor header grammar the benchmark's own gold-graph builder
# parses (``^Obsoletes:\s*([\d,\s]+)`` etc.), so recovered edges line up 1:1 with the independent
# ``_refs.jsonl`` cross-reference graph. ``id_template`` renders both endpoints as ``RFC <n>`` so a
# synthesized node disambiguates against an in-text ``RFC 7230`` mention and reads as a real RFC
# number to any downstream consumer.
_RFC_HEADER_REFS_CONFIG: dict[str, object] = {
    "self_pattern": r"(?im)^Request for Comments:\s*(\d+)",
    "relations": {
        "obsoletes": {"header": r"(?im)^Obsoletes:\s*([\d,\s]+)"},
        "updates": {"header": r"(?im)^Updates:\s*([\d,\s]+)"},
    },
    "target_pattern": r"(\d+)",
    "id_template": "RFC {id}",
    "entity_label": "rfc",
}

# The companion INLINE-CITATION config (the dogfood measured gap). `header_refs` above recovers a
# document's structured-header edges (`Obsoletes:`/`Updates:`), but the overwhelming majority of an
# RFC corpus's gold cross-references are INLINE body citations ("RFC 7230", "[RFC9110]") scattered
# through the document body — which the header-only extractor structurally cannot see (a body chunk
# carries no self id). `fused_entity_relation.inline_refs` closes that: it resolves each document's
# self id ONCE (the same `Request for Comments:` front-matter self declaration, OR the provenance
# file name `rfcNNNN.txt`) and mines every chunk's body for the `citation_pattern`, emitting
# `references(source_doc, cited)` edges. `exclude_relations` mirrors the header grammar so an
# `Obsoletes:` id is not ALSO emitted as a weaker `references` edge, and `id_template` renders both
# endpoints as `RFC <n>` so they disambiguate against the header_refs / in-text mentions. Same
# detected corpus signature as header_refs; wired beside it.
_RFC_INLINE_REFS_CONFIG: dict[str, object] = {
    "self_pattern": r"(?im)^Request for Comments:\s*(\d+)",
    "source_id_pattern": r"(?i)rfc[\s_-]?(\d+)",
    "citation_pattern": r"(?i)RFCs?[\s-]?(\d{3,5})",
    "relation_label": "references",
    "id_template": "RFC {id}",
    "entity_label": "rfc",
    "exclude_relations": {
        "obsoletes": {"header": r"(?im)^Obsoletes:\s*([\d,\s]+)"},
        "updates": {"header": r"(?im)^Updates:\s*([\d,\s]+)"},
    },
    "target_pattern": r"(\d+)",
}

# The known structured-header families, in detection priority order. Each entry is a tuple
# ``(name, self_regex, [relation_header_regex, ...], header_refs_config, inline_refs_config)``.
# A corpus is detected as that family when a sampled file carries BOTH the self header AND at least
# one relation header. Each family carries BOTH the header-edge config (mines
# `Obsoletes:`/`Updates:`) AND the inline-citation config (mines body "RFC <n>" references) — two
# cross-reference extractors for that family. Extend this list to teach the wizard a new family (an
# e-mail ``In-Reply-To:``/``References:`` or a memo ``Supersedes:`` convention) — no Provider or
# recipe change needed.
_HEADER_FAMILIES: tuple[
    tuple[str, str, tuple[str, ...], dict[str, object], dict[str, object]], ...
] = (
    (
        "rfc",
        r"(?im)^Request for Comments:\s*\d+",
        (r"(?im)^Obsoletes:\s*[\d,\s]+", r"(?im)^Updates:\s*[\d,\s]+"),
        _RFC_HEADER_REFS_CONFIG,
        _RFC_INLINE_REFS_CONFIG,
    ),
)

# Sniffing bounds: read at most this many text files, at most this many leading bytes each. A
# metadata header sits in a document's front matter, so the first few KiB are ample; the caps keep
# setup fast + bounded on a large corpus and never load a whole file into memory.
_HEADER_SNIFF_MAX_FILES = 40
_HEADER_SNIFF_MAX_BYTES = 8192


def detect_header_family(source: str | Path) -> dict[str, object] | None:
    """Sniff ``source`` for a known structured-header cross-reference family (dogfood-3-inspect).

    Best-effort: walk the corpus, sample the leading bytes of up to :data:`_HEADER_SNIFF_MAX_FILES`
    plaintext files, and return the ``header_refs`` Stage config of the FIRST family whose
    signature (its ``self`` header AND at least one relation header) appears in a sampled file — the
    RFC ``Request for Comments:`` + ``Obsoletes:``/``Updates:`` family today. Returns ``None`` when
    no family is recognised (or the source is non-local / empty / unreadable), so the wizard wires
    the deterministic ``header_refs`` extractor ONLY for a corpus that actually carries such headers
    and every other corpus stays byte-identical. Never raises: a decode/OS error on one file skips
    it, and a wholly-unreadable tree yields ``None``. Deterministic — the first match under a sorted
    walk — so a given corpus always yields the same wiring (the byte-identical setup contract).
    """
    return _detect_family(source, which="header")


def detect_inline_family(source: str | Path) -> dict[str, object] | None:
    """Sniff ``source`` for a known INLINE-CITATION cross-reference family (the dogfood gap).

    The companion of :func:`detect_header_family`, keyed off the SAME corpus signature (an RFC's
    ``Request for Comments:`` self header + at least one ``Obsoletes:``/``Updates:`` relation
    header) but returning that family's ``inline_refs`` Stage config — the deterministic
    ``fused_entity_relation.inline_refs`` extractor that recovers the corpus's INLINE body citations
    ("RFC 7230", "[RFC9110]") the header-only extractor structurally cannot see. Returns ``None``
    when no family is recognised (or the source is non-local / empty / unreadable), so the wizard
    wires the inline extractor ONLY for a corpus that carries such references and every other corpus
    stays byte-identical. Best-effort + deterministic + never-raises, exactly like its header twin.
    """
    return _detect_family(source, which="inline")


def _detect_family(source: str | Path, *, which: str) -> dict[str, object] | None:
    """Shared corpus sniff for both family detectors — returns the header OR inline config.

    ``which`` selects which of the matched family's two configs is returned (``"header"`` ->
    ``header_refs``, ``"inline"`` -> ``inline_refs``). Both detectors share ONE bounded, sorted,
    never-raising walk so a given corpus yields the same wiring for each (the byte-identical setup
    contract).
    """
    p = Path(str(source).removeprefix("file://"))
    try:
        if not p.exists() or not p.is_dir():
            return None
    except OSError:
        return None
    compiled = [
        (
            header_config if which == "header" else inline_config,
            re.compile(self_pat),
            [re.compile(rel) for rel in rel_pats],
        )
        for _name, self_pat, rel_pats, header_config, inline_config in _HEADER_FAMILIES
    ]
    sampled = 0
    try:
        children = sorted(p.rglob("*"))
    except OSError:
        return None
    for child in children:
        if sampled >= _HEADER_SNIFF_MAX_FILES:
            break
        try:
            if not child.is_file() or child.suffix.lstrip(".").lower() not in _PLAINTEXT_EXTS:
                continue
            with child.open("rb") as fh:
                head = fh.read(_HEADER_SNIFF_MAX_BYTES)
        except OSError:
            continue
        sampled += 1
        text = head.decode("utf-8", errors="ignore")
        for config, self_re, rel_res in compiled:
            if self_re.search(text) and any(rel_re.search(text) for rel_re in rel_res):
                return config
    return None


# --- assembly + rendering ------------------------------------------------------------------


def _assemble(answers: SetupAnswers) -> tuple[StackRecipe, AssembledStack]:
    """Pick the profile's recipe and walk it once (the ONE place a stack comes into existence)."""
    recipe = recipe_for(answers.profile)
    return recipe, recipe.assemble(answers)


def _pipeline_dict(answers: SetupAnswers, stack: AssembledStack) -> dict[str, object]:
    """Wrap the assembled Stage blocks in the pipeline-level mapping (ordered → deterministic)."""
    pipeline: dict[str, object] = {
        "name": STACK_NAME,
        "auto_generate_labels": True,
    }
    # W18: only emit the switch when it deviates from the default, so an `unmasked` config stays
    # byte-identical to pre-W18 (no new key). Read off the ASSEMBLED wiring rather than the answers:
    # the key may only appear when `extract` genuinely reads the redacted chunk stream.
    extract = stack.by_name.get("extract", {})
    if extract.get("depends_on") == ["redact"]:
        pipeline["extract_on"] = "masked"
    pipeline["storage_uri"] = answers.storage
    pipeline["stages"] = list(stack.stages)
    return pipeline


def build_pipeline_dict(answers: SetupAnswers) -> dict[str, object]:
    """Build the Pipeline mapping (the quality-baked GOOD stack) as an ordered dict.

    Pure + deterministic: no env / clock / randomness. The profile selects a
    :class:`~latence_core.setup.recipe.StackRecipe` — an ordered literal of Stage contributions —
    and the walk assembles it. Both profiles share the pipeline-level wrapper (name,
    ``auto_generate_labels``, the W18 ``extract_on`` switch, ``storage_uri``) and both honour the
    ONE PII toggle + the W18 mask-before-extract option; they differ only in the contributions their
    recipe lists and the per-Stage overrides it applies.
    """
    _, stack = _assemble(answers)
    return _pipeline_dict(answers, stack)


def _render_header(answers: SetupAnswers, recipe: StackRecipe, stack: AssembledStack) -> str:
    """The comment header — every claim DERIVED from the stack that was actually assembled.

    Nothing here re-decides what the generator already decided. The profile paragraph comes from the
    recipe's own prologue, each Stage's paragraph comes from that Stage's ``header_lines`` (rendered
    only when the Stage contributed — and, for schema induction, a paragraph describing its
    ABSENCE), and the PII summary follows the presence of the ``redact`` block. That is the fix for
    the ADR-0041 §7a failure: the header used to be a third hand-maintained transcription of the
    builders' knowledge and had drifted from them.

    Composition ORDER stays here, in the renderer, because it is a property of the document rather
    than of any Stage — and order, unlike content, cannot silently disagree with the stack.
    """
    pii_line = (
        "ON — PII is DETECTED, reported, and CHUNK-LEVEL masked into the exported corpus (W16)."
        if "redact" in stack.by_name
        else "OFF — no content_screen / redact stages (no PII detection or masking)."
    )
    lines = [
        "# latence.stack.yaml — generated by `latence setup` (W15, ADR-0041).",
        "#",
        "# A quality-baked, validated full-spine stack. Regenerate with `latence setup`; run with",
        "# `latence process`. Edit by hand freely — it is a plain, diffable stack config.",
    ]
    lines += list(recipe.prologue(answers, stack))
    lines += [
        "#",
        f"# PII redaction + guardrail screening: {pii_line}",
    ]
    lines += list(stack.header_lines.get("redact", ()))
    lines += list(stack.header_lines.get("extract", ()))
    lines += list(stack.header_lines.get("induce", ()))
    lines += list(stack.header_lines.get("chunk", ()))
    domain = normalize_domain_hint(answers.domain)
    if domain:
        # Provenance AND a pointer at where it acts: the same hint rides `induce.config.domain`,
        # where the inducer Provider folds it into its prompt to steer the induced schema. With no
        # `induce` Stage in the assembled stack the hint is provenance only — say so rather than
        # pointing at a key that is not in the file.
        lines += [
            "#",
            f"# Domain hint: {domain}",
            "#   (also threaded into `induce.config.domain` — it steers the induced schema.)"
            if "induce" in stack.by_name
            else "#   (provenance only — predefined-only mode emits no `induce` stage to steer.)",
        ]
    lines += ["#", ""]
    return "\n".join(lines)


def build_stack_config(answers: SetupAnswers) -> str:
    """Generate the full stack YAML text for ``answers`` (deterministic, comment header + body).

    ONE recipe walk feeds BOTH halves of the file: the body is ``yaml.safe_dump`` of the assembled
    pipeline mapping with ``sort_keys=False`` (so the Stage/field order is exactly the ordered
    dict's), and the comment header is rendered from that same assembled stack. Identical answers ⇒
    byte-identical output, and the header cannot describe a stack other than the one below it.
    """
    recipe, stack = _assemble(answers)
    body: str = yaml.safe_dump(
        _pipeline_dict(answers, stack),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    return _render_header(answers, recipe, stack) + body


def validate_config_text(text: str) -> None:
    """Dry-validate a generated stack config; raise :class:`SetupConfigError` with a clear message.

    Reuses the SAME machinery ``latence stack validate`` builds on, WITHOUT running the pipeline (no
    corpus, no models — setup is offline/CPU):

    1. **structural + DAG** — parse into a :class:`~latence_core.pipeline.Pipeline` (Pydantic +
       the unknown-dep / cycle / duplicate-name DAG validator);
    2. **capability wiring** — ``check_stack_capabilities`` (in ``stacks.capability_guard``)
       structurally checks each INSTALLED Provider's class against the Protocol its Stage's
       Capability declares (torch-free, no instantiation). An un-installed heavy Provider is
       skipped (device-honesty), never a spurious failure — so this passes on a CPU-only host.
    3. **phase boundary** (T1) — ``check_phase_boundary`` (in ``stacks.phase_boundary``) rejects a
       stack that schedules a corpus-aggregating Stage before the chunk-level front half has
       finished. Checked HERE, at generation time, and not only in ``latence stack validate``,
       because the wizard's emitted config is the one stack most adopters will ever run: the
       invariant should be impossible to ship, not merely detectable later.
    """
    from latence_core.pipeline import Pipeline
    from latence_core.stacks.capability_guard import check_stack_capabilities
    from latence_core.stacks.phase_boundary import check_phase_boundary

    try:
        pipeline = Pipeline.from_yaml_text(text)
    except Exception as exc:  # noqa: BLE001 - re-raise as a typed, clear config error
        msg = f"generated config is structurally invalid: {exc}"
        raise SetupConfigError(msg) from exc

    findings = check_stack_capabilities(pipeline)
    if findings:
        detail = "; ".join(f.describe() for f in findings)
        msg = f"generated config has capability-wiring errors: {detail}"
        raise SetupConfigError(msg)

    violations = check_phase_boundary(pipeline)
    if violations:
        detail = "; ".join(v.describe() for v in violations)
        msg = f"generated config violates the chunk-level phase boundary: {detail}"
        raise SetupConfigError(msg)
