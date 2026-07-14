"""The deterministic, quality-baked stack generator behind ``latence setup`` (W15, ADR-0041).

Given a resolved :class:`SetupAnswers` (from the interactive prompts OR the ``--non-interactive``
flags — the SAME struct either way), :func:`build_stack_config` emits a full-spine stack YAML that
reproduces the pod-validated GOOD stack (see ``docs/TUTORIAL-fresh-pod-walkthrough.md`` §3):

    source → intake_screen → parse → chunk[640] → [content_screen] → induce[corpus]
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

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# The stack name the generated config carries (stable → byte-identical across runs).
STACK_NAME = "latence-setup-stack"

# The financial-PII label FLOOR the learned redactor (`redaction.gliner2`) runs when PII is ON.
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

# The three parsers the wizard chooses between, keyed by intent (matches the roster).
PARSER_UNIVERSAL = "parser.render"  # universal: mixed types (PDF/office/images), GPU-OCR sub
PARSER_PDF_TEXT = "parser.pdfplumber"  # born-digital PDF text (CPU, no OCR)
PARSER_OCR = "parser.lighton"  # OCR VLM for scans / images
_KNOWN_PARSERS = frozenset({PARSER_UNIVERSAL, PARSER_PDF_TEXT, PARSER_OCR})

# Extensions that read as scanned/image inputs (OCR territory) vs born-digital PDF.
_IMAGE_EXTS = frozenset({"png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "webp"})
_OFFICE_EXTS = frozenset({"docx", "doc", "pptx", "ppt", "xlsx", "xls", "odt", "rtf", "html", "htm"})

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


class SetupConfigError(ValueError):
    """A generated config failed its dry validation (structural / DAG / capability wiring).

    Raised by :func:`validate_config_text` with a human-readable message so the CLI can fail
    loudly with a clear reason rather than saving a broken config.
    """


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
    pii: bool = True  # The ONE optional toggle: PII redaction + guardrail screening.
    # W18 (ADR-0044) advanced option: which chunk-text variant extraction reads.
    #   "unmasked" (default) — extraction reads the raw chunk text; the KG keeps real entities
    #                          (byte-identical to pre-W18).
    #   "masked"   — extraction reads the REDACTED chunk stream so PII never reaches the KG at all
    #                (the stricter posture; requires PII redaction ON).
    extract_on: str = "unmasked"
    domain: str = ""  # Optional free-text domain hint (recorded as a header comment).
    extensions: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_EXTENSIONS)


def detect_extensions(path: str | Path) -> tuple[str, ...]:
    """Return the sorted, de-duplicated file extensions present under ``path`` (no dot, lowercased).

    A best-effort local scan used to seed the ``source`` Stage's ``extensions`` and to recommend a
    parser. A non-existent / empty folder returns the broad default set so a config still generates
    (the operator can prune it). Deterministic (sorted) so it never perturbs the byte-identical
    contract.
    """
    p = Path(str(path).removeprefix("file://"))
    if not p.exists() or not p.is_dir():
        return _DEFAULT_EXTENSIONS
    found: set[str] = set()
    for child in p.rglob("*"):
        if child.is_file() and child.suffix:
            found.add(child.suffix.lstrip(".").lower())
    if not found:
        return _DEFAULT_EXTENSIONS
    return tuple(sorted(found))


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


def llm_key_env_var(base_url: str) -> str:
    """The env var the schema-induction LLM reads its key from (mirrors ``label_inducer.llm``).

    An OpenRouter ``base_url`` prefers ``$OPENROUTER_API_KEY``; anything else (direct OpenAI, or a
    keyless on-prem vLLM) reads ``$OPENAI_API_KEY``. The generator NEVER stores the key — this only
    names the var so the CLI can warn when it is unset.
    """
    return "OPENROUTER_API_KEY" if "openrouter" in (base_url or "").lower() else "OPENAI_API_KEY"


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
    # parser.pdfplumber: born-digital PDF text, no config block (CPU).
    return stage


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


def build_pipeline_dict(answers: SetupAnswers) -> dict[str, object]:
    """Build the Pipeline mapping (the quality-baked GOOD stack) as an ordered dict.

    Pure + deterministic: no env / clock / randomness. The ONE toggle (``answers.pii``) adds/omits
    exactly the ``content_screen`` + ``redact`` stages; every other Stage is the mandatory spine.
    The W18 advanced option ``answers.extract_on`` (default ``unmasked``, byte-identical) can set
    ``masked``, which wires the extraction Stage AFTER ``redact`` and feeds it the redacted chunk
    stream so PII never reaches the KG.
    """
    masked_extract = answers.extract_on == "masked"
    if masked_extract and not answers.pii:
        msg = "extract_on: masked requires PII redaction ON (there is no redact Stage otherwise)."
        raise ValueError(msg)
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

    # 2) intake screening (always) — reject oversized / malformed / zip-bomb files.
    stages.append(
        {
            "name": "intake_screen",
            "capability": "intake_screening",
            "provider": "screening.intake_signature",
            "depends_on": ["source"],
            "config": {"max_bytes": 2097152, "max_zip_ratio": 50.0},
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

    # 6) schema induction (mandatory quality lever) — one shared canonical schema for the corpus.
    stages.append(
        {
            "name": "induce",
            "capability": "schema_induction",
            "provider": "label_inducer.llm",
            "depends_on": [induce_input],
            "config": {
                "model": answers.llm_model,
                "base_url": answers.llm_base_url,
                "granularity": "corpus",
                "max_tokens": 2048,
                "max_labels": 40,
                "temperature": 0.0,
            },
        }
    )

    # 7) fused NER+RE in ONE pass (GLiNER2); config labels are a FLOOR, induced types union on.
    # W18: in `extract_on: masked` mode the extractor reads the REDACTED chunk stream (depends_on
    # `redact`, which sets each chunk's masked_content) so PII never reaches the KG; default
    # `unmasked` keeps the pre-W18 wiring (depends_on `induce`), so the KG keeps the real entities.
    extract_input = "redact" if masked_extract else "induce"
    stages.append(
        {
            "name": "extract",
            "capability": "fused_entity_relation",
            "provider": "fused_entity_relation.gliner2",
            "depends_on": [extract_input],
            "config": {
                "device": answers.device,
                "labels": ["organization", "person"],
                "relation_labels": ["works for", "partner of"],
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
                "provider": "redaction.gliner2",
                "depends_on": ["induce"],
                "config": {
                    "device": answers.device,
                    "pii_labels": list(FINANCIAL_PII_LABELS),
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
    enrich_chunk_source = "redact" if answers.pii else "induce"
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
        "#",
        f"# PII redaction + guardrail screening: {pii_line}",
    ]
    if answers.pii:
        lines += [
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
    lines += [
        "#",
        f"# Schema-induction LLM key: read from ${key_var} at run (never stored here).",
    ]
    if answers.domain.strip():
        lines += ["#", f"# Domain hint: {answers.domain.strip()}"]
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
    """
    from latence_core.pipeline import Pipeline
    from latence_core.stacks.capability_guard import check_stack_capabilities

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
