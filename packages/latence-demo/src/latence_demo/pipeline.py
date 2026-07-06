"""The demo Pipeline (S13): the full spine over the bundled messy corpus.

Composes S1–S9 into one Pipeline the local Runner executes CPU-only, offline:

    Source
      → Intake Screening        (quarantine dangerous files before Parse)
      → Parse                   (parser.document: real PDFs + text → markdown + page map)
      → Chunk
      → Content Screening       (flag prompt-injection chunks)
      → Entity Extraction       (gazetteer; the swappable Provider)
      → Relation Extraction
      → Redaction               (PII-handled corpus variant)
      → Disambiguation          (canonical entities + normalized relations)
      → Graph Assembly          (the knowledge graph: nodes / edges / per-edge Evidence)
      → Export (knowledge_graph)   the KG as Parquet + TTL + GraphML
      → Export (demo_site)         the self-contained web viewer (KG)
      → Export (jsonl_parquet)     the RAG-ready corpus

Every Provider here ships in ``latence-core`` or ``latence-demo`` — pure-Python, CPU-only,
permissively licensed (ADR-0007/0012), no cloud credentials. The one documented Provider swap
changes exactly one line (``entity.gazetteer`` → a local model or a cloud endpoint) and reruns
the identical Pipeline (ADR-0011); :func:`build_pipeline` takes an ``entity_provider`` /
``entity_config`` so the swap is a one-argument change and the swap YAML is a one-line diff.
"""

from __future__ import annotations

from typing import Any

from latence_core.pipeline import CapabilityKind, Pipeline, Stage

# The zero-shot entity label set + gazetteer terms shared across the demo corpus so
# Disambiguation collapses IBM / International Business Machines and Graph Assembly links
# people to organisations. Passed to the default gazetteer Provider; a swapped-in model
# Provider takes a flat ``labels`` list instead (see build_pipeline).
_GAZETTEER_LABELS: dict[str, dict[str, list[str]]] = {
    "ORG": {
        "terms": [
            "Acme",
            "Acme Corporation",
            "International Business Machines",
            "IBM",
            "Globex",
        ]
    },
    "PERSON": {"terms": ["Alice", "Bob", "Carol"]},
}

# The demo's small Intake-Screening caps: a 2 MiB max so the planted 3 MiB oversized fixture
# is quarantined, and a strict zip-ratio so the planted zip bomb is caught by metadata alone.
_INTAKE_CONFIG: dict[str, Any] = {"max_bytes": 2 * 1024 * 1024, "max_zip_ratio": 50.0}


def build_pipeline(
    storage_uri: str,
    corpus_folder: str,
    *,
    name: str = "latence-demo",
    entity_provider: str = "entity.gazetteer",
    entity_config: dict[str, Any] | None = None,
) -> Pipeline:
    """Construct the full-spine demo :class:`Pipeline` reading ``corpus_folder`` on Storage.

    ``entity_provider``/``entity_config`` are the one-line Provider swap made concrete: the
    default is the in-core CPU gazetteer; passing ``entity.gliner`` (local model) or
    ``entity.endpoint`` (cloud/vLLM) reruns the identical Pipeline against a different Provider
    — the model- and cloud-agnostic proof (ADR-0011). Everything else is fixed.
    """
    if entity_provider == "entity.gazetteer":
        ent_cfg: dict[str, Any] = entity_config or {"labels": _GAZETTEER_LABELS}
    else:
        # A model/endpoint NER Provider takes a flat zero-shot label list (nothing hardcoded).
        ent_cfg = entity_config or {"labels": ["person", "organization"]}

    stages = [
        Stage(
            name="source",
            capability=CapabilityKind.SOURCE,
            provider="source.local_folder",
            config={
                "path": corpus_folder,
                "sensitivity": "internal",
                # Widen the accepted set so the Parser sees PDFs and the planted dangerous
                # fixtures reach Intake Screening (a .zip is read so it can be quarantined).
                "extensions": ["txt", "md", "markdown", "pdf", "zip"],
            },
        ),
        Stage(
            name="intake_screen",
            capability=CapabilityKind.INTAKE_SCREENING,
            provider="screening.intake_signature",
            config=_INTAKE_CONFIG,
            depends_on=["source"],
        ),
        Stage(
            name="parse",
            capability=CapabilityKind.PARSE,
            provider="parser.document",
            depends_on=["intake_screen"],
        ),
        Stage(
            name="chunk",
            capability=CapabilityKind.CHUNK,
            provider="chunk.markdown",
            config={"max_tokens": 256, "overlap_tokens": 32, "min_tokens": 8},
            depends_on=["parse"],
        ),
        Stage(
            name="content_screen",
            capability=CapabilityKind.CONTENT_SCREENING,
            provider="screening.content_keyword",
            depends_on=["chunk"],
        ),
        Stage(
            name="entities",
            capability=CapabilityKind.ENTITY_EXTRACTION,
            provider=entity_provider,
            config=ent_cfg,
            depends_on=["content_screen"],
        ),
        Stage(
            name="relations",
            capability=CapabilityKind.RELATION_EXTRACTION,
            provider="relation.pattern",
            config={
                "window_chars": 200,
                "relations": {
                    "works_for": {
                        "head": "PERSON",
                        "tail": "ORG",
                        "triggers": ["works at", "employed by", "joined"],
                    },
                    "partner_of": {
                        "head": "ORG",
                        "tail": "ORG",
                        "triggers": ["partner of", "partnered with"],
                    },
                },
            },
            depends_on=["content_screen", "entities"],
        ),
        Stage(
            name="redact",
            capability=CapabilityKind.REDACTION,
            provider="redaction.hybrid_rule",
            depends_on=["parse"],
        ),
        Stage(
            name="disambiguate",
            capability=CapabilityKind.DISAMBIGUATION,
            provider="disambiguation.cascade",
            config={
                "kb": [
                    {"kb_id": "Q1", "name": "Acme", "entity_type": "ORG"},
                    {
                        "kb_id": "Q2",
                        "name": "International Business Machines",
                        "entity_type": "ORG",
                        "aliases": ["IBM"],
                    },
                ],
                "kb_threshold": 0.5,
            },
            depends_on=["entities", "relations"],
        ),
        Stage(
            name="graph",
            capability=CapabilityKind.GRAPH_ASSEMBLY,
            provider="graph.canonical",
            config={"corpus_id": name},
            depends_on=["disambiguate"],
        ),
        Stage(
            name="export_kg",
            capability=CapabilityKind.EXPORT,
            provider="export.knowledge_graph",
            config={"basename": "graph"},
            depends_on=["graph"],
        ),
        Stage(
            name="export_site",
            capability=CapabilityKind.EXPORT,
            provider="export.demo_site",
            config={"basename": "index", "pipeline_name": name},
            depends_on=["graph"],
        ),
        Stage(
            name="export_corpus",
            capability=CapabilityKind.EXPORT,
            provider="export.jsonl_parquet",
            config={"basename": "corpus"},
            depends_on=["redact"],
        ),
    ]
    return Pipeline(name=name, storage_uri=storage_uri, stages=stages)
