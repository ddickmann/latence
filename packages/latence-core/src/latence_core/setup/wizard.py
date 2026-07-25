"""The deterministic, quality-baked stack generator behind ``latence setup`` (W15, ADR-0041).

Given a resolved :class:`SetupAnswers` (from the interactive prompts OR the ``--non-interactive``
flags — the SAME struct either way), :func:`build_stack_config` emits a full-spine stack YAML that
reproduces the pod-validated GOOD stack (see ``docs/TUTORIAL-fresh-pod-walkthrough.md`` §3):

    source → intake_screen → parse → chunk[640] → [content_screen] → induce[chunk]
      → extract[gliner2, thr 0.5] → [redact[gliner2, financial-PII]] → profiling
      → disambiguate → graph → enrich(context) → export_kg + export_corpus(context_columns)

The *mandatory spine* is always present; the ONE optional toggle is PII redaction + guardrail
screening (``pii``), which adds/omits exactly the ``content_screen`` + ``redact`` stages and
nothing else. The generator is PURE — it never reads the environment, a clock, or a random source —
so identical answers yield a **byte-identical** file (the ``--non-interactive`` determinism
contract, for CI/repro).

The exported RAG corpus is built from the ENRICHED chunk stream (``export_corpus → enrich``), so it
carries ``induced_labels`` + ``context_header`` + a dense embedding. With PII ON the ``redact``
Stage is a W16 CHUNK-LEVEL transform (ADR-0042): it consumes the induced chunk stream and sets each
chunk's ``masked_content`` (each chunk ≤ the model window, so no PII is truncated), and ``enrich``
reads THAT redacted chunk stream — so the exported corpus row is genuinely PII-safe AND enriched
(``redact → enrich → export_corpus``, the blessed ``stacks/context-enrichment.yaml`` dataflow).
This closes the pre-W16 gap where the RAG chunks were enriched but un-masked. The generator states
this accurately in the file header.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# The stack name the generated config carries (stable → byte-identical across runs).
STACK_NAME = "latence-setup-stack"

# The ONE blessed PII redactor the wizard emits — in EVERY profile (#197, ADR-0041/0036/0044).
#
# The wizard's one opinionated config must BE the profile's documented design, and it had drifted:
# `lite` (and this module's own documented spine) named `redaction.gliner2` while `enterprise-sota`
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
_KNOWN_PARSERS = frozenset({PARSER_UNIVERSAL, PARSER_PDF_TEXT, PARSER_OCR})

# The two parsers the enterprise-SOTA profile chooses between. A GPU pod almost always parses
# scanned/PDF/image corpora through the served-vLLM OCR endpoint, but an enterprise corpus is just
# as often born-digital PLAIN TEXT (.txt/.md/.jsonl exports, log/ticket dumps). For those the OCR
# VLM is the WRONG tool — it only rasterizes pages and POSTs images to the OCR server, it has no
# text passthrough — so forcing it makes a text corpus un-parseable when the OCR box is (correctly)
# not served. The wizard therefore routes a detected plaintext corpus to the CPU plaintext parser
# and keeps every other SOTA signal (graph_features.pagerank + BM25 + Granite r2) intact.
_KNOWN_SOTA_PARSERS = frozenset({PARSER_PLAINTEXT, PARSER_OCR_VLLM})

# The two setup profiles (tiers). `lite` is the CPU-first, laptop-friendly default (unchanged); the
# `enterprise-sota` profile emits the pod-VALIDATED composed stack (served-vLLM OCR + LLM schema
# induction + fused GLiNER2 ER + the neural GLinker disambiguator + Granite r2 embeddings + ULTRA
# graph completion — see stacks/gpu-sota-glinker.yaml). Both flow through the SAME pure generator.
PROFILE_LITE = "lite"
PROFILE_SOTA = "enterprise-sota"
_KNOWN_PROFILES = frozenset({PROFILE_LITE, PROFILE_SOTA})

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

# The index-time SIGNALS the blessed enterprise-SOTA stack bakes in (ADR-0041: the wizard emits ONE
# opinionated, quality-baked config). Both are pure-Python, CPU-only, deterministic, zero extra deps
# — they add no GPU cost and no new dependency to the pod stack, and both are strictly ADDITIVE:
# omitting the block leaves the export byte-identical (the ADR-0049/0051 guarantee).
#
#   * graph features (ADR-0051, #161) — PageRank centrality + label-propagation community stamped
#     onto every KG node, promoted by export_kg to first-class graph-nodes.parquet columns and
#     projected by `enrich` onto each chunk's context_header. Feeds multi-hop expansion and the
#     Knapsack packer's optional signals. `pagerank` (not `degree`): global influence over the
#     whole citation-like graph an enterprise corpus builds, not just local degree.
#   * BM25 (ADR-0049, #164) — the term-statistics artifact (`bm25-stats.json` +
#     `bm25-postings.parquet`) the query-side sparse rescorer runs on. `tokenizer.regex` is the
#     documented reference tokenizer (lowercase UNICODE alphanumeric runs), matching the retrieval
#     tooling's query-side term unit exactly — the agreement the sparse leg depends on.
#     Scope, stated honestly because this profile is the MULTILINGUAL stack (it emits a
#     multilingual embedder and a multilingual PII model): the reference tokenizer keeps accented
#     and non-Latin words WHOLE, but it is not a word SEGMENTER — a script written without word
#     spacing (Chinese/Japanese/Thai) yields one term per whitespace-delimited run. A corpus in
#     those scripts should plug its own segmenter in behind the `Tokenizer` seam; the dense leg
#     (Granite r2, multilingual) is unaffected either way.
SOTA_GRAPH_FEATURES_PROVIDER = "graph_features.pagerank"
SOTA_BM25_TOKENIZER = "tokenizer.regex"

# Extensions that read as scanned/image inputs (OCR territory) vs born-digital PDF.
_IMAGE_EXTS = frozenset({"png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "webp"})
_OFFICE_EXTS = frozenset({"docx", "doc", "pptx", "ppt", "xlsx", "xls", "odt", "rtf", "html", "htm"})

# Extensions the plaintext Parser decodes losslessly (bytes -> text/markdown, no OCR, no office
# rendering). A corpus whose file types are ALL in this set is born-digital text — the OCR VLM
# would only mangle it — so the enterprise-SOTA wizard routes it to `parser.plaintext`.
_PLAINTEXT_EXTS = frozenset({"txt", "text", "md", "markdown", "mdown", "jsonl", "ndjson"})

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


@dataclass(frozen=True)
class CorpusScan:
    """What ONE best-effort walk of the source folder observed (file types + the largest file).

    ``scanned`` is the honesty bit: ``False`` means the walk could not observe the corpus at all —
    a non-local Storage URI, a missing/unreadable folder, or a folder with no files — so
    ``extensions`` is the broad default set and ``largest_bytes`` is 0 (the caller warns and the
    generator falls back to :data:`INTAKE_MAX_BYTES_FLOOR`). Never raises.
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


class SetupConfigError(ValueError):
    """A generated config failed its dry validation (structural / DAG / capability wiring).

    Raised by :func:`validate_config_text` with a human-readable message so the CLI can fail
    loudly with a clear reason rather than saving a broken config.
    """


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
    # pod-validated composed stack). The generator branches on this ONCE; other fields are shared.
    profile: str = PROFILE_LITE
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
    # :func:`scan_source`. It SIZES the `intake_screen` oversize cap (see :func:`intake_max_bytes`)
    # so the scanned corpus is actually admitted instead of being wholly quarantined at intake.
    # 0 (the default, and what an unreadable / empty / non-local source yields) keeps the pre-fix
    # flat 2 MiB floor.
    largest_source_bytes: int = 0
    extensions: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_EXTENSIONS)
    # dogfood-3-inspect: the detected structured-header cross-reference family's `header_refs` Stage
    # config (from :func:`detect_header_family`), or ``None`` when the corpus carries no recognised
    # metadata-header convention. When set (enterprise-SOTA only), the SOTA builder wires a
    # deterministic `fused_entity_relation.header_refs` extraction Stage that recovers the corpus's
    # explicit header cross-references (an RFC's `Obsoletes:`/`Updates:` edges) beside the learned
    # extractor. ``None`` ⇒ NO such Stage is emitted, so config + KG are byte-identical to before.
    header_refs: dict[str, object] | None = None
    # The detected structured-header family's `inline_refs` Stage config (from
    # :func:`detect_inline_family`), or ``None`` when the corpus carries no recognised family. When
    # set (enterprise-SOTA only), the SOTA builder wires a deterministic
    # `fused_entity_relation.inline_refs` extraction Stage that recovers the corpus's INLINE body
    # citations (an RFC's scattered "RFC 7230" references — the dogfood measured gap the header-only
    # extractor misses) beside the learned extractor AND the header_refs Stage. ``None`` ⇒ NO such
    # Stage is emitted, so config + KG are byte-identical to before.
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
# WIRES it. This is that wiring: the wizard sniffs the corpus for a KNOWN structured-header family
# and, when it positively detects one, emits a ``header_refs`` extraction Stage configured for it
# that composes beside the learned extractor (disjoint ``#href-`` id namespace; both feed the same
# Disambiguation input). A corpus with NO recognised header family gets NO such Stage — its emitted
# YAML and resulting KG stay byte-identical to a run without the feature.
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
# builder change needed.
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
    plaintext files, and return the :data:`header_refs` Stage config of the FIRST family whose
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
    header) but returning that family's :data:`inline_refs` Stage config — the deterministic
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


def llm_key_env_var(base_url: str) -> str:
    """The env var the schema-induction LLM reads its key from (mirrors ``label_inducer.llm``).

    An OpenRouter ``base_url`` prefers ``$OPENROUTER_API_KEY``; anything else (direct OpenAI, or a
    keyless on-prem vLLM) reads ``$OPENAI_API_KEY``. The generator NEVER stores the key — this only
    names the var so the CLI can warn when it is unset.
    """
    return "OPENROUTER_API_KEY" if "openrouter" in (base_url or "").lower() else "OPENAI_API_KEY"


def _induce_stage(answers: SetupAnswers, *, depends_on: str) -> dict[str, object]:
    """The ``induce`` Stage (LLM schema induction at CHUNK granularity) — shared by both profiles.

    **Per-chunk (T1).** Each chunk is induced against its own text and carries only its own small,
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
    wizard hands it the hint, never a hand-crafted prompt string). So a hint genuinely steers the
    induced schema instead of only annotating the file. No hint ⇒ the key is OMITTED entirely, so
    the emitted config and the Provider's prompts are byte-identical to a run without one.
    """
    config: dict[str, object] = {
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
    return {
        "name": "induce",
        "capability": "schema_induction",
        "provider": "label_inducer.llm",
        "depends_on": [depends_on],
        "config": config,
    }


def _wire_induction(
    answers: SetupAnswers, stages: list[dict[str, object]], *, depends_on: str
) -> str:
    """Append the ``induce`` Stage (or NOT, in predefined-only mode) and name the label source.

    Returns the Stage name every label-consuming Stage (extract, redact, enrich, the deterministic
    reference extractors) should depend on: ``"induce"`` normally, and — in predefined-only mode —
    the stage that WOULD have fed induction, because there is no induce Stage to depend on.

    Predefined-only is an OMISSION, not a flag: no Stage, no Provider, no client, no call. That is
    the only implementation of "induction is skipped entirely" that cannot regress into a quiet LLM
    call, and it makes the emitted stack self-evidently free of one.
    """
    if answers.predefined_labels is not None:
        return depends_on
    stages.append(_induce_stage(answers, depends_on=depends_on))
    return "induce"


def _entity_labels(answers: SetupAnswers, default: list[str]) -> list[str]:
    """The extraction Stage's entity labels: the operator's fixed set, else the profile default.

    In predefined-only mode the config labels are no longer a floor that induced types union onto —
    they are the WHOLE schema, so they must be the operator's own. Order is preserved as given
    (deterministic YAML), never sorted behind the operator's back.
    """
    fixed = (answers.predefined_labels or {}).get("entity_types")
    return list(fixed) if fixed else list(default)


def _relation_labels(answers: SetupAnswers, default: list[str]) -> list[str]:
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


def _parse_stage(answers: SetupAnswers) -> dict[str, object]:
    """The ``parse`` Stage, its config shaped by the chosen parser (universal / PDF-text / OCR)."""
    stage: dict[str, object] = {
        "name": "parse",
        "capability": "parse",
        "provider": answers.parser,
        "depends_on": ["intake_screen"],
    }
    if answers.parser == PARSER_UNIVERSAL:
        stage["config"] = {
            "device": answers.device,  # OCR sub-provider is GPU; skip-with-flag on a CPU host
            "dpi": 200,
            "soffice_timeout_s": 120,
            "ocr": {
                "provider": PARSER_OCR,
                "config": {"device": answers.device, "max_new_tokens": 4096},
            },
        }
    elif answers.parser == PARSER_OCR:
        stage["config"] = {"device": answers.device, "max_new_tokens": 4096}
    # parser.pdfplumber / parser.plaintext: born-digital text, no config block (CPU).
    return stage


def _sota_parse_stage(answers: SetupAnswers) -> dict[str, object]:
    """The enterprise-SOTA ``parse`` Stage, shaped by the chosen parser.

    Two parse paths (see :data:`_KNOWN_SOTA_PARSERS`): the production served-vLLM OCR endpoint
    (``parser.lighton_vllm`` — the base_url/model/api_key/concurrency block a scanned corpus needs),
    OR the CPU plaintext parser for a born-digital text corpus (``parser.plaintext`` — no OCR
    endpoint, no config). Routing the text case here is what makes the SOTA profile yield a RUNNABLE
    stack for a .txt/.md/.jsonl corpus instead of forcing OCR at an unserved endpoint. Every other
    SOTA Stage (GLiNER2, GLinker, Granite r2, graph_features, BM25, ULTRA) is unchanged.
    """
    if answers.parser == PARSER_OCR_VLLM:
        return {
            "name": "parse",
            "capability": "parse",
            "provider": PARSER_OCR_VLLM,
            "depends_on": ["intake_screen"],
            "config": {
                "base_url": answers.ocr_base_url,
                "model": answers.ocr_model,
                "api_key": SOTA_OCR_API_KEY,
                "max_concurrency": SOTA_OCR_MAX_CONCURRENCY,
            },
        }
    # A born-digital text corpus: the CPU plaintext parser (no OCR server needed). `_parse_stage`
    # emits the bare parse Stage for any non-OCR parser, so the SOTA text path reuses it verbatim.
    return _parse_stage(answers)


def _embedder_block(answers: SetupAnswers) -> dict[str, object]:
    """The shared ST embedder block (disambiguation context + RAG-corpus vectors).

    Defaults to IBM Granite Embedding r2 (Apache-2.0, ADR-0045) tiered by the chosen device: the
    GPU tier ``granite-embedding-311m-multilingual-r2`` (768-dim, bf16 + flash_attention_2 with a
    graceful sdpa fallback) on ``cuda``, else the lighter CPU tier
    ``granite-embedding-97m-multilingual-r2`` (384-dim, fp32) for a laptop-CPU box.
    """
    if answers.device == "cuda":
        config: dict[str, object] = {
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


def _sota_embedder_block() -> dict[str, object]:
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


def _sota_graph_features_block() -> dict[str, object]:
    """The enterprise-SOTA graph-feature block (ADR-0051, #161) for the ``graph`` Stage.

    PageRank centrality + label-propagation community, stamped onto every KG node and promoted by
    ``export_kg`` to graph-nodes.parquet columns (and projected by ``enrich`` onto each chunk's
    context_header). Pure Python, CPU-only, deterministic, zero extra deps — and additive: with the
    block absent the emitted graph is byte-identical, so the enrichment costs the stack nothing it
    did not already own.
    """
    return {"provider": SOTA_GRAPH_FEATURES_PROVIDER, "config": {}}


def _sota_bm25_block() -> dict[str, object]:
    """The enterprise-SOTA BM25 signal block (ADR-0049, #164) for the corpus Export.

    Materializes the term-statistics artifact (``bm25-stats.json`` + ``bm25-postings.parquet``)
    alongside the corpus Export — the index-time sparse signal the query-side BM25 rescorer runs on.
    ``tokenizer.regex`` is the reference tokenizer (lowercase *Unicode* alphanumeric runs) the
    retrieval tooling documents, and it is the same term unit the query-side rescorer applies — so
    accented and non-Latin terms are indexed and matched, not dropped. It is not a word segmenter:
    an unspaced script (Chinese/Japanese/Thai) yields one term per whitespace-delimited run, and
    such a corpus should plug its own segmenter in behind the Tokenizer seam.
    Deterministic, CPU-only, and additive (absent ⇒ no files, byte-identical).
    """
    return {"provider": SOTA_BM25_TOKENIZER, "config": {}}


def build_pipeline_dict(answers: SetupAnswers) -> dict[str, object]:
    """Build the Pipeline mapping (the quality-baked GOOD stack) as an ordered dict.

    Pure + deterministic: no env / clock / randomness. Branches ONCE on ``answers.profile``:
    ``lite`` (the CPU-first default) or ``enterprise-sota`` (the pod-validated composed stack). Both
    share the pipeline-level wrapper (name, ``auto_generate_labels``, the W18 ``extract_on`` switch,
    ``storage_uri``) and both honour the ONE PII toggle + the W18 mask-before-extract option; they
    differ only in the per-Stage Providers/config the profile-specific stage builder emits.
    """
    masked_extract = answers.extract_on == "masked"
    if masked_extract and not answers.pii:
        msg = "extract_on: masked requires PII redaction ON (there is no redact Stage otherwise)."
        raise ValueError(msg)
    if answers.profile == PROFILE_SOTA:
        stages = _build_sota_stages(answers, masked_extract=masked_extract)
    else:
        stages = _build_lite_stages(answers, masked_extract=masked_extract)

    pipeline: dict[str, object] = {
        "name": STACK_NAME,
        "auto_generate_labels": True,
    }
    # W18: only emit the switch when it deviates from the default, so an `unmasked` config stays
    # byte-identical to pre-W18 (no new key).
    if masked_extract:
        pipeline["extract_on"] = "masked"
    pipeline["storage_uri"] = answers.storage
    pipeline["stages"] = stages
    return pipeline


def _build_lite_stages(
    answers: SetupAnswers, *, masked_extract: bool
) -> list[dict[str, object]]:
    """The ``lite`` profile stages (CPU-first, unchanged from W15/W16/W18).

    The ONE toggle (``answers.pii``) adds/omits exactly the ``content_screen`` + ``redact`` stages;
    every other Stage is the mandatory spine. ``masked_extract`` wires ``extract`` AFTER ``redact``
    so extraction reads the redacted chunk stream (W18) and PII never reaches the KG.
    """
    stages: list[dict[str, object]] = []

    # 1) source — the folder + detected extensions.
    stages.append(
        {
            "name": "source",
            "capability": "source",
            "provider": "source.local_folder",
            "config": {
                "path": answers.source,
                "sensitivity": "internal",
                "extensions": list(answers.extensions),
            },
        }
    )

    # 2) intake screening (always) — reject oversized / malformed / zip-bomb files. The oversize cap
    # is SIZED from the scanned corpus (intake_max_bytes) so the operator's real documents are
    # admitted rather than silently quarantined; it remains a real guard (see the constants).
    stages.append(
        {
            "name": "intake_screen",
            "capability": "intake_screening",
            "provider": "screening.intake_signature",
            "depends_on": ["source"],
            "config": {
                "max_bytes": intake_max_bytes(answers.largest_source_bytes),
                "max_zip_ratio": 50.0,
            },
        }
    )

    # 3) parse — parser-shaped config.
    stages.append(_parse_stage(answers))

    # 4) chunk at 640 (the mdeberta-window-calibrated near-max).
    stages.append(
        {
            "name": "chunk",
            "capability": "chunk",
            "provider": "chunk.markdown",
            "depends_on": ["parse"],
            "config": {"max_tokens": 640, "overlap_tokens": 80, "min_tokens": 8},
        }
    )

    # 5) content screening — the guardrail half of the ONE toggle.
    induce_input = "chunk"
    if answers.pii:
        stages.append(
            {
                "name": "content_screen",
                "capability": "content_screening",
                "provider": "screening.content_keyword",
                "depends_on": ["chunk"],
            }
        )
        induce_input = "content_screen"

    # 6) schema induction (PER CHUNK, T1) — or NOTHING at all in predefined-only mode.
    labels_from = _wire_induction(answers, stages, depends_on=induce_input)

    # 7) fused NER+RE in ONE pass (GLiNER2). T2: the config labels are a PER-CHUNK RELEVANCE
    # floor — a chunk that induced its own types is scored against THOSE ALONE (zero-shot
    # precision degrades with label count); a chunk that induced nothing falls back to these.
    # `label_floor: universal` on the extract config restores the config ∪ induced union.
    # T1: in predefined-only mode there is nothing to union — the config labels ARE the schema.
    # W18: in `extract_on: masked` mode the extractor reads the REDACTED chunk stream (depends_on
    # `redact`, which sets each chunk's masked_content) so PII never reaches the KG; default
    # `unmasked` keeps the pre-W18 wiring, so the KG keeps the real entities.
    extract_input = "redact" if masked_extract else labels_from
    stages.append(
        {
            "name": "extract",
            "capability": "fused_entity_relation",
            "provider": "fused_entity_relation.gliner2",
            "depends_on": [extract_input],
            "config": {
                "device": answers.device,
                "labels": _entity_labels(answers, ["organization", "person"]),
                "relation_labels": _relation_labels(answers, ["works for", "partner of"]),
                "threshold": 0.5,
                "max_len": 768,
            },
        }
    )

    # 8) learned PII redaction (GLiNER2, financial-PII) — the PII half of the ONE toggle.
    # W16 (ADR-0042): Redaction is a chunk->chunk transform — it consumes the induced CHUNK stream
    # and sets each chunk's `masked_content` (each chunk fits the model window, so no PII is
    # truncated). Depending on `induce` keeps the induced pii_types UNIONing onto the FINANCIAL_PII
    # floor (ADR-0038). `enrich` then reads THIS redacted chunk stream so export_corpus is masked
    # AND enriched (the pre-W16 doc-level `depends_on: [parse, induce]` shape is gone).
    if answers.pii:
        stages.append(
            {
                "name": "redact",
                "capability": "redaction",
                "provider": BLESSED_REDACT_PROVIDER,
                "depends_on": [labels_from],
                "config": {
                    "device": answers.device,
                    # T1: the ADR-0044 financial floor, plus any operator-fixed pii_types
                    # (predefined-only fixes the RELEVANCE schema, never the SAFETY schema).
                    "pii_labels": _pii_labels(answers),
                    # T2: the blessed checkpoint is a GUARDRAILS+PII model, so the guard half is
                    # switched on too. Every chunk carries the universal guard floor (prompt
                    # injection / harmful content — a security invariant, never subsetted per
                    # chunk) UNIONed with whatever guard categories that chunk induced. A hit is
                    # FLAGGED as a RiskMarker in the corpus, never masked; the deterministic
                    # keyword content_screen Stage above is a separate, earlier pre-filter.
                    "guardrails": True,
                    "max_len": 768,
                    "threshold": 0.5,
                },
            }
        )

    # 9) corpus profiling.
    stages.append(
        {
            "name": "profiling",
            "capability": "profiling",
            "provider": "profiling.statistical",
            "depends_on": ["parse", "extract"],
        }
    )

    # 10) disambiguation (learned surface+context; no hard-coded KB — corpus-agnostic).
    stages.append(
        {
            "name": "disambiguate",
            "capability": "disambiguation",
            "provider": "disambiguation.embedding",
            "depends_on": ["extract"],
            "config": {
                "threshold": 0.90,
                "context_window": 160,
                "embedder": _embedder_block(answers),
            },
        }
    )

    # 11) canonical, evidence-linked KG.
    stages.append(
        {
            "name": "graph",
            "capability": "graph_assembly",
            "provider": "graph.canonical",
            "depends_on": ["disambiguate"],
            "config": {"corpus_id": STACK_NAME},
        }
    )

    # 12) W13 context enrichment — project the KG back onto each (induced) chunk.
    # W16 (ADR-0042): with PII ON, enrich reads the REDACTED chunk stream (`redact` set each chunk's
    # masked_content), so the exported corpus row is masked AND enriched; with PII OFF there is no
    # redact stage, so enrich reads the induced chunk stream directly. This mirrors the blessed
    # stacks/context-enrichment.yaml `redact → enrich → export_corpus` dataflow.
    enrich_chunk_source = "redact" if answers.pii else labels_from
    stages.append(
        {
            "name": "enrich",
            "capability": "context_enrichment",
            "provider": "context.kg_header",
            "depends_on": [enrich_chunk_source, "graph"],
            "config": {
                "max_neighbors_per_entity": 4,
                "max_triples": 6,
                "min_entity_confidence": 0.0,
                "min_edge_confidence": 0.0,
            },
        }
    )

    # 13a) KG export.
    stages.append(
        {
            "name": "export_kg",
            "capability": "export",
            "provider": "export.knowledge_graph",
            "depends_on": ["graph"],
            "config": {"basename": "graph"},
        }
    )

    # 13b) RAG-corpus export from the ENRICHED chunk stream (context_columns + dense embedding).
    stages.append(
        {
            "name": "export_corpus",
            "capability": "export",
            "provider": "export.jsonl_parquet",
            "depends_on": ["enrich"],
            "config": {
                "basename": "records",
                "context_columns": True,
                "embedder": _embedder_block(answers),
            },
        }
    )

    return stages


def _build_sota_stages(
    answers: SetupAnswers, *, masked_extract: bool
) -> list[dict[str, object]]:
    """The ``enterprise-sota`` profile stages — the pod-VALIDATED composed stack.

    Reproduces ``stacks/gpu-sota-glinker.yaml`` (ADR-0046) with the wizard's quality spine: the
    chosen parse path (served vLLM OCR ``parser.lighton_vllm`` for a scanned/PDF/image corpus, or
    the CPU ``parser.plaintext`` for a born-digital .txt/.md/.jsonl corpus — see
    :func:`recommend_sota_parser` / :func:`_sota_parse_stage`) → chunk[640] → LLM schema
    induction[chunk] (T1; absent entirely in predefined-only mode) → fused GLiNER2
    entity+relation → learned PII redaction (``redaction.gliner2``, #197) → profiling → the neural
    **GLinker** disambiguator (``disambiguation.glinker`` + Granite r2 STS embedder) → canonical KG
    → **ULTRA** graph completion (skip-with-flag when absent) → KG + context-enriched RAG-corpus
    export (Granite r2 vectors). Every learned Stage runs on CUDA (``SOTA_DEVICE``).

    Deliberate delta from ``stacks/gpu-sota-glinker.yaml`` (#197): that hand-written stack predates
    the decision and still wires ``redaction.gliner_pii``; the wizard emits the ONE blessed
    :data:`BLESSED_REDACT_PROVIDER` in every profile. Both Providers remain supported behind the
    same ``PIIDetector`` seam — the profile just no longer picks a different one per tier.

    The ONE PII toggle and the W18 mask-before-extract option ride exactly as in ``lite``: PII adds
    ``content_screen`` + ``redact`` and wires ``redact → enrich → export_corpus`` (W16 chunk-level
    masking); ``masked_extract`` moves ``extract`` after ``redact``. The GLinker/GLiNER2 provider
    defaults (embedding-merge OFF, fuzzy+distinguishing-token-guard ON, ``compile=False``) are left
    to ride — so the ER is precise by default and the config stays minimal.
    """
    stages: list[dict[str, object]] = []

    # 1) source — the folder + detected extensions.
    stages.append(
        {
            "name": "source",
            "capability": "source",
            "provider": "source.local_folder",
            "config": {
                "path": answers.source,
                "sensitivity": "internal",
                "extensions": list(answers.extensions),
            },
        }
    )

    # 2) intake screening (always) — reject oversized / malformed / zip-bomb files. The oversize cap
    # is SIZED from the scanned corpus (intake_max_bytes) so the operator's real documents are
    # admitted rather than silently quarantined; it remains a real guard (see the constants).
    stages.append(
        {
            "name": "intake_screen",
            "capability": "intake_screening",
            "provider": "screening.intake_signature",
            "depends_on": ["source"],
            "config": {
                "max_bytes": intake_max_bytes(answers.largest_source_bytes),
                "max_zip_ratio": 50.0,
            },
        }
    )

    # 3) parse — the served-vLLM LightOn-OCR endpoint for a scanned/PDF/image corpus (the production
    # OCR path, ~2.7x in-process), OR the CPU plaintext parser for a born-digital text corpus. The
    # wizard routes on the detected file types (recommend_sota_parser) so a .txt/.md/.jsonl corpus
    # gets a runnable text parser instead of OCR at an endpoint that is not served for it.
    stages.append(_sota_parse_stage(answers))

    # 4) chunk at 640 (the mdeberta-window-calibrated near-max).
    stages.append(
        {
            "name": "chunk",
            "capability": "chunk",
            "provider": "chunk.markdown",
            "depends_on": ["parse"],
            "config": {"max_tokens": 640, "overlap_tokens": 80, "min_tokens": 8},
        }
    )

    # 5) content screening — the guardrail half of the ONE PII toggle.
    induce_input = "chunk"
    if answers.pii:
        stages.append(
            {
                "name": "content_screen",
                "capability": "content_screening",
                "provider": "screening.content_keyword",
                "depends_on": ["chunk"],
            }
        )
        induce_input = "content_screen"

    # 6) LLM schema induction PER CHUNK (T1) — each chunk keeps its own small, relevant label set —
    # or NOTHING at all in predefined-only mode (the operator's fixed schema, zero LLM calls).
    labels_from = _wire_induction(answers, stages, depends_on=induce_input)

    # 7) fused NER+RE in ONE pass (GLiNER2) — T2: the labels are a PER-CHUNK relevance floor
    # (each chunk is scored against its own induced types; these are the fallback when a chunk
    # induced nothing). T1: in predefined-only mode there is nothing to union — the config labels
    # ARE the schema. threshold 0.3 (the SOTA recall floor); compile defaults to False in the
    # provider (left to ride).
    extract_input = "redact" if masked_extract else labels_from
    stages.append(
        {
            "name": "extract",
            "capability": "fused_entity_relation",
            "provider": "fused_entity_relation.gliner2",
            "depends_on": [extract_input],
            "config": {
                "device": SOTA_DEVICE,
                "labels": _entity_labels(answers, ["person", "organization"]),
                "relation_labels": _relation_labels(answers, ["works for", "partner of"]),
                "threshold": 0.3,
            },
        }
    )

    # 7b) deterministic structured-header cross-reference recovery (dogfood-3-inspect) — ONLY when
    # the wizard positively detected a known metadata-header family (`answers.header_refs`). The
    # in-core, zero-dep `fused_entity_relation.header_refs` Provider mines each document's self id +
    # header targets (an RFC's `Request for Comments:` + `Obsoletes:`/`Updates:` lines) into
    # synthesized identifier mentions + directed edges — exactly the explicit cross-references the
    # entity-anchored learned extractor structurally cannot see. It reads the SAME (unmasked)
    # chunk stream `extract` reads and, via its disjoint `#href-` id namespace, composes beside
    # gliner2: both feed Disambiguation, whose input is the concatenation of its parents' output. No
    # detected family ⇒ this Stage is absent and the pipeline is byte-identical to before.
    disambiguate_inputs = ["extract"]
    if answers.header_refs is not None:
        stages.append(
            {
                "name": "header_refs",
                "capability": "fused_entity_relation",
                "provider": "fused_entity_relation.header_refs",
                "depends_on": [labels_from],
                "config": dict(answers.header_refs),
            }
        )
        disambiguate_inputs.append("header_refs")

    # 7c) deterministic INLINE-CITATION cross-reference recovery (the dogfood measured gap) — ONLY
    # when the wizard positively detected a known metadata-header family (`answers.inline_refs`).
    # The in-core, zero-dep `fused_entity_relation.inline_refs` Provider resolves each document's
    # self id ONCE (its `Request for Comments:` self declaration OR the provenance file name
    # `rfcNNNN.txt`) and mines every chunk's BODY for inline "RFC <n>" citations — the overwhelming
    # majority of an RFC corpus's gold cross-references, which BOTH the entity-anchored learned
    # extractor AND the header-only `header_refs` sibling structurally cannot see (a body chunk
    # carries no self id). It reads the SAME (unmasked) chunk stream `extract` reads and, via its
    # disjoint `#iref-` id namespace, composes beside gliner2 AND header_refs: all feed
    # Disambiguation, whose input is the concatenation of its parents' output. No detected family ⇒
    # this Stage is absent and the pipeline is byte-identical to before.
    if answers.inline_refs is not None:
        stages.append(
            {
                "name": "inline_refs",
                "capability": "fused_entity_relation",
                "provider": "fused_entity_relation.inline_refs",
                "depends_on": [labels_from],
                "config": dict(answers.inline_refs),
            }
        )
        disambiguate_inputs.append("inline_refs")

    # 8) learned PII redaction — the PII half of the ONE toggle, on the SAME blessed Provider `lite`
    # emits (#197: `BLESSED_REDACT_PROVIDER`, the ADR-0036 W9 unified gliner2 engine this profile
    # already runs for `extract`, so `redact` adds no second model library). W16 (ADR-0042): a
    # chunk→chunk transform over the induced chunk stream that sets each chunk's masked_content, so
    # `enrich` reads a masked stream and export_corpus is masked AND enriched.
    if answers.pii:
        stages.append(
            {
                "name": "redact",
                "capability": "redaction",
                "provider": BLESSED_REDACT_PROVIDER,
                "depends_on": [labels_from],
                "config": {
                    "device": SOTA_DEVICE,
                    # T1: the ADR-0044 financial floor, plus any operator-fixed pii_types
                    # (predefined-only fixes the RELEVANCE schema, never the SAFETY schema).
                    "pii_labels": _pii_labels(answers),
                    # T2: the blessed checkpoint is a GUARDRAILS+PII model, so the guard half is
                    # switched on too. Every chunk carries the universal guard floor (prompt
                    # injection / harmful content — a security invariant, never subsetted per
                    # chunk) UNIONed with whatever guard categories that chunk induced. A hit is
                    # FLAGGED as a RiskMarker in the corpus, never masked; the deterministic
                    # keyword content_screen Stage above is a separate, earlier pre-filter.
                    "guardrails": True,
                    "max_len": 768,
                    "threshold": 0.5,
                },
            }
        )

    # 9) corpus profiling.
    stages.append(
        {
            "name": "profiling",
            "capability": "profiling",
            "provider": "profiling.statistical",
            "depends_on": ["parse", "extract"],
        }
    )

    # 10) the enterprise GLinker disambiguator (ADR-0046) — neural L2→L3→L4→L0 linking + the audited
    # 5-strategy resolver. Granite r2 embedder attached (opt-in STS rung). use_embedding_merge /
    # min_similarity are NOT set — the provider defaults (embedding-merge OFF, fuzzy + the
    # distinguishing-token guard ON at 0.84) are precise by default and left to ride.
    stages.append(
        {
            "name": "disambiguate",
            "capability": "disambiguation",
            "provider": "disambiguation.glinker",
            "depends_on": disambiguate_inputs,
            "config": {
                "device": SOTA_DEVICE,
                "context_window": 200,
                "threshold": 0.3,
                "use_reranker": True,
                "use_flash_deberta": True,
                "embedder": _sota_embedder_block(),
            },
        }
    )

    # 11) canonical, evidence-linked KG — WITH the ADR-0051 graph-feature enrichment (#193): the
    # assembler resolves the PageRank computer and stamps centrality + community onto every node,
    # which export_kg promotes to graph-nodes.parquet columns and `enrich` projects onto each
    # chunk's context_header. Those signals feed multi-hop expansion + the Knapsack packer.
    stages.append(
        {
            "name": "graph",
            "capability": "graph_assembly",
            "provider": "graph.canonical",
            "depends_on": ["disambiguate"],
            "config": {
                "corpus_id": STACK_NAME,
                "graph_features": _sota_graph_features_block(),
            },
        }
    )

    # 12) ULTRA graph completion (zero-shot inductive link prediction). skip_if_unavailable: the
    # checkpoint is not vendored, so on a host without it the Stage skips-with-flag (no inferred
    # edges) rather than failing the run.
    stages.append(
        {
            "name": "complete",
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
    )

    # 13) W13 context enrichment — project the KG back onto each (redacted, induced) chunk. With PII
    # ON, enrich reads the REDACTED chunk stream (redact → enrich → export_corpus, W16); with PII
    # OFF there is no redact stage, so enrich reads the induced chunk stream directly.
    enrich_chunk_source = "redact" if answers.pii else labels_from
    stages.append(
        {
            "name": "enrich",
            "capability": "context_enrichment",
            "provider": "context.kg_header",
            "depends_on": [enrich_chunk_source, "graph"],
            "config": {
                "max_neighbors_per_entity": 4,
                "max_triples": 6,
                "min_entity_confidence": 0.0,
                "min_edge_confidence": 0.0,
            },
        }
    )

    # 14a) KG export (asserted + ULTRA-completed edges).
    stages.append(
        {
            "name": "export_kg",
            "capability": "export",
            "provider": "export.knowledge_graph",
            "depends_on": ["graph", "complete"],
            "config": {"basename": "graph"},
        }
    )

    # 14b) RAG-corpus export from the ENRICHED chunk stream (context_columns + Granite r2 vectors)
    # PLUS the ADR-0049 BM25 index-time signal (#194): the term-statistics artifact
    # (bm25-stats.json + bm25-postings.parquet) the query-side sparse rescorer runs on. Without it
    # the blessed stack ships dense-only and the hybrid fuser has no sparse leg.
    stages.append(
        {
            "name": "export_corpus",
            "capability": "export",
            "provider": "export.jsonl_parquet",
            "depends_on": ["enrich"],
            "config": {
                "basename": "records",
                "context_columns": True,
                "embedder": _sota_embedder_block(),
                "bm25": _sota_bm25_block(),
            },
        }
    )

    return stages


def _render_header(answers: SetupAnswers) -> str:
    """The comment header: provenance, the domain hint, the honest PII note, the LLM-key note."""
    key_var = llm_key_env_var(answers.llm_base_url)
    pii_line = (
        "ON — PII is DETECTED, reported, and CHUNK-LEVEL masked into the exported corpus (W16)."
        if answers.pii
        else "OFF — no content_screen / redact stages (no PII detection or masking)."
    )
    lines = [
        "# latence.stack.yaml — generated by `latence setup` (W15, ADR-0041).",
        "#",
        "# A quality-baked, validated full-spine stack. Regenerate with `latence setup`; run with",
        "# `latence process`. Edit by hand freely — it is a plain, diffable stack config.",
    ]
    if answers.profile == PROFILE_SOTA:
        if answers.parser == PARSER_PLAINTEXT:
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
        lines += [
            "#",
            "# Profile: enterprise-sota — the pod-VALIDATED composed stack (ADR-0046,",
            "#   stacks/gpu-sota-glinker.yaml):",
            *parse_lines,
            "#   LLM schema induction + fused GLiNER2 entity+relation + the neural GLinker",
            "#   disambiguator + IBM Granite r2 embeddings + ULTRA graph completion. Every learned",
            "#   Stage runs on CUDA; on a CPU-only host they skip-with-flag at `latence stack",
            *parse_point,
            "#",
            "#   Index-time signals (CPU, deterministic, zero extra deps): graph features on",
            "#   `graph` (ADR-0051 — PageRank centrality + community, exported as graph-nodes",
            "#   columns and projected onto each chunk's context_header) and BM25 on",
            "#   `export_corpus` (ADR-0049 — bm25-stats.json + bm25-postings.parquet for the",
            "#   query-side sparse rescorer). Drop either block to get the plain export back.",
        ]
    lines += [
        "#",
        f"# PII redaction + guardrail screening: {pii_line}",
    ]
    if answers.pii:
        lines += [
            f"#   Redactor: `{BLESSED_REDACT_PROVIDER}` (#197, ADR-0036 W9) — the SAME GLiNER2",
            "#   engine `extract` runs, so PII detection adds no second model library. Its",
            "#   `pii_labels` are the ADR-0044 financial floor — UNIVERSAL, applied to EVERY",
            "#   chunk and never subsetted (T2); the per-chunk induced pii_types UNION on top",
            "#   at run (ADR-0038), and the universal financial floor masks card/iban/ssn even",
            "#   when",
            "#   the model misses them. Any other Redaction Provider drops in behind the same",
            "#   Capability seam — swap `redact.provider` and nothing else moves.",
            "#   W16 (ADR-0042): `redact` is a CHUNK-LEVEL transform — it consumes the induced",
            "#   chunk stream and sets each chunk's `masked_content` (each chunk fits the model",
            "#   window, so no PII is truncated). `enrich` reads THAT redacted chunk stream, and",
            "#   export_corpus materializes `masked_content` — so the RAG corpus is BOTH masked",
            "#   (PII-safe) AND context-enriched (`redact -> enrich -> export_corpus`). No pre-W16",
            "#   doc-truncation leak, and no trade-off between masking and the per-chunk headers.",
        ]
    if answers.pii and answers.extract_on == "masked":
        lines += [
            "#",
            "# extract_on: masked (W18, ADR-0044) — extraction reads the REDACTED chunk stream, so",
            "#   PII never reaches entity/relation extraction or the KG (the stricter privacy",
            "#   posture). `extract` depends_on `redact`. Default `unmasked` (KG keeps entities).",
        ]
    if answers.predefined_labels is None:
        lines += [
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
        ]
    else:
        lines += [
            "#",
            "# Schema induction: OFF (predefined-only) — you supplied a FIXED label schema, so",
            "#   there is NO `induce` stage in this stack at all and the run makes ZERO induction",
            "#   LLM calls (no key needed). `extract.config.labels` / `relation_labels` ARE the",
            "#   schema. The `redact.config.pii_labels` still carry the ADR-0044 financial floor:",
            "#   a fixed RELEVANCE schema is your choice, the SAFETY floor is not.",
            "#   Delete `predefined_labels` from your setup flags to get per-chunk induction back.",
        ]
    domain = normalize_domain_hint(answers.domain)
    if domain:
        # Provenance AND a pointer at where it acts: the same hint rides `induce.config.domain`,
        # where the inducer Provider folds it into its prompt to steer the induced schema. In
        # predefined-only mode there IS no induce stage, so the hint is provenance only — say so
        # rather than pointing at a key that is not in the file.
        lines += [
            "#",
            f"# Domain hint: {domain}",
            "#   (also threaded into `induce.config.domain` — it steers the induced schema.)"
            if answers.predefined_labels is None
            else "#   (provenance only — predefined-only mode emits no `induce` stage to steer.)",
        ]
    lines += ["#", ""]
    return "\n".join(lines)


def build_stack_config(answers: SetupAnswers) -> str:
    """Generate the full stack YAML text for ``answers`` (deterministic, comment header + body).

    The body is ``yaml.safe_dump`` of :func:`build_pipeline_dict` with ``sort_keys=False`` so the
    Stage/field order is exactly the ordered dict's — identical answers ⇒ byte-identical output.
    """
    body: str = yaml.safe_dump(
        build_pipeline_dict(answers),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    return _render_header(answers) + body


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
