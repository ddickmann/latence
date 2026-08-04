# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Pure S3 building blocks: corpus preps, the budget model, the pipeline.yaml generator.

Everything in this module is a pure function of its inputs — no filesystem, no clock, no
network — so the offline suite proves the exact bytes the stage will write:

* **Corpus preps** turn the two text-native raw corpora into deterministic markdown file
  sets: MultiHop-RAG's 609-article ``corpus.json`` and 2WikiMultihopQA's dev-split question
  contexts (one file per DISTINCT paragraph, M6-question-pool style). Files are named
  ``<slug>-<sha12>.md`` — content-addressed, so re-running a prep is idempotent and a
  changed upstream row cannot silently overwrite a different paragraph — and deduped by
  content hash (the same paragraph appears under many questions).
* **The budget model** converts the operator's USD cap into the schema-inducer's
  ``max_induction_calls`` (the provider enforces a CALL budget, not a token budget — that is
  what keeps a budgeted run byte-reproducible), and converts the Quality Report's token
  actuals back into USD for the gate metrics. Prices are the OpenRouter gpt-4.1-mini card
  (verified 2026-08-04); they are an ESTIMATE model and are recorded as such in metrics.
* **The generator** renders one dataset's full-feature pipeline.yaml from the campaign's
  :class:`~latence_benchmark.campaign.wizard.S3Answers` — the gpu-sota spine + per-chunk
  schema induction + hyperedges + graph features + context enrichment + the retrieval-signal
  exports, wired exactly as the blessed stacks wire them (see the inline stage comments).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml

from latence_benchmark.campaign.stages import StageBlockedError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from latence_benchmark.campaign.wizard import S3Answers

#: Pod-side root of everything S3 touches (inputs, pipelines, per-dataset stores, logs).
REMOTE_ROOT = "/workspace/latence-sota/s3"

#: The PII floor stamped onto every chunk (the gpu-sota list). The framework's financial
#: floor and, with guardrails on, the universal guard floor are enforced BY the provider on
#: top of this — they are not configurable away (redaction_policy T2).
PII_LABELS: tuple[str, ...] = (
    "person",
    "email",
    "phone number",
    "address",
    "credit card number",
    "social security number",
)

#: Extraction FLOOR for the induction-driven datasets (the schema-induction stack's lean
#: floor: induced ∪ config, so a small floor makes extraction lean on the induced schema).
INDUCED_FLOOR_ENTITY: tuple[str, ...] = ("person", "organization")
INDUCED_FLOOR_RELATION: tuple[str, ...] = ("works_for",)

#: The corpus-export embedder (the gpu-sota-pod wiring; runs on the pod GPU).
EMBEDDER_MODEL = "ibm-granite/granite-embedding-311m-multilingual-r2"

# -- the induction budget model --------------------------------------------------------------

#: OpenRouter gpt-4.1-mini pricing (USD per token; card read 2026-08-04). Estimates for the
#: call-budget conversion — the HARD bound is the derived max_induction_calls, and the gate
#: metrics report token ACTUALS from the Quality Report priced with these same numbers.
USD_PER_PROMPT_TOKEN = 0.40 / 1_000_000
USD_PER_COMPLETION_TOKEN = 1.60 / 1_000_000
#: Per-call estimates: instruction scaffold + one 640-chunker-token chunk (≈770 model
#: tokens) prompt-side; a structured label JSON completion-side.
EST_PROMPT_TOKENS_PER_CALL = 1600
EST_COMPLETION_TOKENS_PER_CALL = 220
EST_USD_PER_CALL = (
    EST_PROMPT_TOKENS_PER_CALL * USD_PER_PROMPT_TOKEN
    + EST_COMPLETION_TOKENS_PER_CALL * USD_PER_COMPLETION_TOKEN
)


def induction_call_budget(budget_usd_cap: float, induced_datasets: int) -> int:
    """The per-dataset ``max_induction_calls`` under an even split of the USD cap.

    Each dataset is its own ``latence process`` run with its own pipeline.yaml, so the cap
    is divided evenly across the induction-driven datasets of THIS campaign selection —
    generated before any chunk count is known, which is why the split is even rather than
    proportional. Floor division: the sum of shares never exceeds the cap.
    """
    if induced_datasets <= 0:
        return 0
    return int(budget_usd_cap / induced_datasets / EST_USD_PER_CALL)


def induction_usd_actual(prompt_tokens: int, completion_tokens: int) -> float:
    """Token actuals (from the Quality Report's induction fields) priced in USD."""
    return (
        prompt_tokens * USD_PER_PROMPT_TOKEN + completion_tokens * USD_PER_COMPLETION_TOKEN
    )


# -- corpus preps ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PrepFile:
    """One markdown file a corpus prep will write: deterministic name, final content."""

    filename: str
    content: str


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, max_len: int = 60) -> str:
    """Lower-case, hyphen-joined, ASCII-ish slug; ``"doc"`` when nothing survives."""
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")[:max_len].rstrip("-")
    return slug or "doc"


def _prep_file(title: str, text: str) -> tuple[str, PrepFile]:
    """Content-addressed markdown doc: hash keyed on the CONTENT so dedup is exact."""
    content = f"# {title.strip() or 'Untitled'}\n\n{text.strip()}\n"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return digest, PrepFile(filename=f"{slugify(title)}-{digest[:12]}.md", content=content)


def multihop_rag_articles(corpus_json_text: str) -> tuple[PrepFile, ...]:
    """MultiHop-RAG's ``corpus.json`` (609 news articles) → one ``.md`` per distinct article.

    Dedup is by content hash (title + body); output is sorted by filename, so the returned
    tuple — and therefore the written directory — is byte-deterministic regardless of any
    upstream re-ordering.
    """
    try:
        raw = json.loads(corpus_json_text)
    except ValueError as exc:
        msg = f"multihop_rag: corpus.json is not valid JSON: {exc}"
        raise StageBlockedError(msg) from exc
    if not isinstance(raw, list):
        msg = f"multihop_rag: corpus.json must hold a list of articles, got {type(raw).__name__}."
        raise StageBlockedError(msg)
    seen: set[str] = set()
    files: list[PrepFile] = []
    for entry in raw:
        if not isinstance(entry, dict):
            msg = "multihop_rag: corpus.json contains a non-object article entry."
            raise StageBlockedError(msg)
        body = str(entry.get("body", "") or "")
        if not body.strip():
            continue  # An article with no text has nothing to extract from.
        digest, prep = _prep_file(str(entry.get("title", "") or ""), body)
        if digest in seen:
            continue
        seen.add(digest)
        files.append(prep)
    if not files:
        msg = "multihop_rag: corpus.json yielded zero non-empty articles."
        raise StageBlockedError(msg)
    return tuple(sorted(files, key=lambda f: f.filename))


def wiki2multihop_paragraphs(context_column: Iterable[str]) -> tuple[PrepFile, ...]:
    """2Wiki dev contexts → the question-pool corpus: one ``.md`` per DISTINCT paragraph.

    Each row's ``context`` is a JSON string of ``[[title, [sentence, ...]], ...]`` (10
    paragraphs per question, massively overlapping across questions — the M6 question-pool
    construction). Paragraphs are deduped by content hash and emitted title-as-heading +
    joined sentences, sorted by filename.
    """
    seen: set[str] = set()
    files: list[PrepFile] = []
    for raw in context_column:
        try:
            context = json.loads(raw)
        except ValueError as exc:
            msg = f"wiki2multihop: a context cell is not valid JSON: {exc}"
            raise StageBlockedError(msg) from exc
        if not isinstance(context, list):
            msg = "wiki2multihop: a context cell does not hold a paragraph list."
            raise StageBlockedError(msg)
        for paragraph in context:
            if (
                not isinstance(paragraph, list)
                or len(paragraph) != 2
                or not isinstance(paragraph[1], list)
            ):
                msg = "wiki2multihop: a paragraph is not the [title, [sentences]] shape."
                raise StageBlockedError(msg)
            title, sentences = str(paragraph[0]), [str(s) for s in paragraph[1]]
            text = " ".join(s.strip() for s in sentences if s.strip())
            if not text:
                continue
            digest, prep = _prep_file(title, text)
            if digest in seen:
                continue
            seen.add(digest)
            files.append(prep)
    if not files:
        msg = "wiki2multihop: the dev contexts yielded zero non-empty paragraphs."
        raise StageBlockedError(msg)
    return tuple(sorted(files, key=lambda f: f.filename))


# -- the pipeline.yaml generator -------------------------------------------------------------


def _stage(
    name: str,
    capability: str,
    provider: str,
    *,
    depends_on: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {"name": name, "capability": capability, "provider": provider}
    if depends_on:
        spec["depends_on"] = depends_on
    if config:
        spec["config"] = config
    return spec


def generate_pipeline_yaml(
    dataset: str,
    s3: S3Answers,
    *,
    induce: bool,
    max_induction_calls: int = 0,
) -> str:
    """One dataset's full-feature pipeline.yaml — pure, byte-deterministic, pod-rooted.

    :param induce: Whether THIS dataset runs the LLM label-induction stage (the OCR'd group
        under mode per_chunk/doc). ``False`` = the hand-authored label sets are the whole
        schema (the text-native group, and every group under mode ``none``).
    :param max_induction_calls: The dataset's call-budget share (ignored unless ``induce``).

    Wiring decisions carried from the blessed stacks + the S0 inventory, in DAG order:
    intake → parse(plaintext: the corpus is markdown) → chunk {the campaign's knobs} →
    content screening BEFORE induction (never spend LLM calls on quarantined text) →
    optional induce → ONE fused extract stage (ADR-0013; labels = floor, induced UNIONs on)
    → redact off the same chunk stream (extract_on: unmasked — extraction reads raw text,
    masking applies to the exported corpus; guardrails per config) → profiling →
    type_consolidation.exact_surface → disambiguation.exact_surface (the linear full-corpus
    path — the cascade walls past ~10⁴ surfaces, the M6 lesson) → graph.canonical with
    hyperedges (graph.canonical ONLY carries the block) + degree graph_features → context
    enrichment off [redact, graph] (masked AND enriched corpus rows) → KG export with the
    hyperedge KV sidecar + corpus export with the cuda sentence-transformers embedder,
    context columns and the BM25 artifact.
    """
    if induce:
        entity_floor = list(INDUCED_FLOOR_ENTITY)
        relation_floor = list(INDUCED_FLOOR_RELATION)
    else:
        entity_floor = list(s3.entity_labels)
        relation_floor = list(s3.relation_labels)
    extract_parent = "induce" if induce else "content_screen"

    stages: list[dict[str, Any]] = [
        _stage(
            "source",
            "source",
            "source.local_folder",
            config={
                "path": f"file://{REMOTE_ROOT}/input/{dataset}/markdown",
                "sensitivity": "internal",
                "extensions": ["md", "markdown"],
            },
        ),
        _stage(
            "intake_screen",
            "intake_screening",
            "screening.intake_signature",
            depends_on=["source"],
            # 25 MiB: OCR'd markdown of book-length PDFs comfortably exceeds the 2 MiB demo
            # default; the zip-ratio guard is irrelevant for .md but harmless.
            config={"max_bytes": 26214400, "max_zip_ratio": 100.0},
        ),
        _stage("parse", "parse", "parser.plaintext", depends_on=["intake_screen"]),
        _stage(
            "chunk",
            "chunk",
            "chunk.markdown",
            depends_on=["parse"],
            config={
                "max_tokens": s3.chunk_max_tokens,
                "overlap_tokens": s3.chunk_overlap_tokens,
                "min_tokens": s3.chunk_min_tokens,
            },
        ),
        _stage(
            "content_screen",
            "content_screening",
            "screening.content_keyword",
            depends_on=["chunk"],
        ),
    ]
    if induce:
        stages.append(
            _stage(
                "induce",
                "schema_induction",
                "label_inducer.llm",
                depends_on=["content_screen"],
                config={
                    "model": s3.induction_model,
                    "base_url": s3.induction_base_url,
                    "granularity": "chunk" if s3.induction_mode == "per_chunk" else "doc",
                    "max_tokens": 2048,
                    "max_labels": 40,
                    "temperature": 0.0,
                    "max_induction_calls": max_induction_calls,
                    "device": "cuda",
                },
            )
        )
    stages += [
        _stage(
            "extract",
            "fused_entity_relation",
            s3.extraction_provider,
            depends_on=[extract_parent],
            config={
                "model": s3.extraction_model,
                "labels": entity_floor,
                "relation_labels": relation_floor,
                "threshold": s3.extraction_threshold,
                "device": "cuda",
            },
        ),
        _stage(
            "redact",
            "redaction",
            s3.redaction_provider,
            depends_on=[extract_parent],
            config={
                "pii_labels": list(PII_LABELS),
                "guardrails": s3.redaction_guardrails,
                "device": "cuda",
            },
        ),
        _stage(
            "profiling",
            "profiling",
            "profiling.statistical",
            depends_on=["parse", "extract"],
        ),
        _stage(
            "consolidate_types",
            "type_consolidation",
            "type_consolidation.exact_surface",
            depends_on=["extract"],
        ),
        _stage(
            "disambiguate",
            "disambiguation",
            "disambiguation.exact_surface",
            depends_on=["consolidate_types"],
        ),
        _stage(
            "graph",
            "graph_assembly",
            "graph.canonical",
            depends_on=["disambiguate"],
            config={
                "corpus_id": f"s3-{dataset}",
                "hyperedges": {
                    "weak_hyperedges": True,
                    "max_arity": 12,
                    "min_tokens": 25,
                    "max_tokens": 180,
                    "max_window_sentences": 3,
                },
                "graph_features": {"provider": "graph_features.degree"},
            },
        ),
        _stage(
            "enrich",
            "context_enrichment",
            "context.kg_header",
            depends_on=["redact", "graph"],
            config={"max_neighbors_per_entity": 4, "max_triples": 6},
        ),
        _stage(
            "export_kg",
            "export",
            "export.knowledge_graph",
            depends_on=["graph"],
            config={"basename": "graph", "hyperedge_kv": True},
        ),
        _stage(
            "export_corpus",
            "export",
            "export.jsonl_parquet",
            depends_on=["enrich"],
            config={
                "basename": "corpus",
                "context_columns": True,
                "embedder": {
                    "provider": "embedding.sentence_transformers",
                    "config": {
                        "model": EMBEDDER_MODEL,
                        "dimension": 768,
                        "device": "cuda",
                    },
                },
                "bm25": {"provider": "tokenizer.regex"},
            },
        ),
    ]

    document = {
        "schema_version": 2,
        "name": f"s3-{dataset}",
        "storage_uri": f"file://{REMOTE_ROOT}/{dataset}",
        "extract_on": "unmasked",
        "stages": stages,
    }
    body: str = yaml.safe_dump(document, sort_keys=False, default_flow_style=False)
    induction_line = (
        f"induction {s3.induction_mode} ({s3.induction_model}, "
        f"max {max_induction_calls} calls)"
        if induce
        else "induction none (hand-authored labels are the whole schema)"
    )
    header = [
        f"# s3-{dataset} — generated by the s3-pipeline campaign stage. DO NOT hand-tune on",
        "# the pod: regenerate via `latence-campaign run s3-pipeline` so campaign.yaml stays",
        "# the single source of truth (identical s3 config => byte-identical file).",
        f"# chunk {s3.chunk_max_tokens}/{s3.chunk_overlap_tokens}/{s3.chunk_min_tokens} | "
        f"{s3.extraction_provider} ({s3.extraction_model}) @ {s3.extraction_threshold} | "
        f"{induction_line}",
        f"# redaction {s3.redaction_provider} guardrails="
        f"{'on' if s3.redaction_guardrails else 'off'} | extract_on unmasked "
        "(extraction reads raw chunks; masking applies to the exported corpus)",
    ]
    return "\n".join(header) + "\n" + body


# -- Quality-Report readers ------------------------------------------------------------------


def _section(report: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = report.get(name)
    return value if isinstance(value, dict) else {}


def dataset_metrics_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """The gate numbers of one dataset run, read from its ``quality-report.json``.

    Chunks / mentions / relations / entities / hyperedges / PII come from the report's typed
    sections; guardrail RiskMarkers are not rolled up by the Quality Report (they live on
    the exported chunk rows), so the honest byte-backed neighbour — flagged chunks after
    screening — is recorded alongside an explicit pointer instead of a fabricated count.
    """
    graph = _section(report, "graph")
    hyper = graph.get("hyperedges")
    induction: dict[str, Any] | None = None
    for stage in report.get("stages", []):
        if isinstance(stage, dict) and stage.get("capability") == "schema_induction":
            prompt = int(stage.get("induction_prompt_tokens", 0))
            completion = int(stage.get("induction_completion_tokens", 0))
            induction = {
                "calls": int(stage.get("induction_calls", 0)),
                "cache_hits": int(stage.get("induction_cache_hits", 0)),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "budget_exhausted": bool(stage.get("induction_budget_exhausted", False)),
                "usd_actual": round(induction_usd_actual(prompt, completion), 4),
            }
    return {
        "documents": int(report.get("document_count", 0)),
        "chunks": int(_section(report, "chunk").get("chunks_total", 0)),
        "chunks_flagged": int(_section(report, "chunk").get("chunks_flagged", 0)),
        "mentions": int(_section(report, "entities").get("mentions_total", 0)),
        "relations": int(_section(report, "relations").get("relations_total", 0)),
        "entities": int(graph.get("nodes_total", 0)),
        "edges": int(graph.get("edges_total", 0)),
        "hyperedges": (
            int(hyper.get("hyperedges_total", 0)) if isinstance(hyper, dict) else 0
        ),
        "pii_total": int(_section(report, "redaction").get("pii_total", 0)),
        "pii_per_type": dict(_section(report, "redaction").get("per_type", {})),
        "guardrail_note": (
            "guardrail RiskMarkers ride the exported chunk rows (risk_markers), not the "
            "quality report; chunks_flagged counts screening-flagged chunks"
        ),
        "induction": induction,
    }
