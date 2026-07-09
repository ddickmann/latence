"""Declarative Pipeline contract (ADR-0015).

A versioned YAML/JSON document is the canonical Pipeline declaration: an ordered
list of Stages, each binding a Capability kind to a Provider (plugin name + config),
plus Storage URIs. A typed Pydantic builder constructs and validates the same
object, and can round-trip back to YAML — config-as-data that platform teams diff
and commit, and what makes the one-line Provider swap literally true.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from latence_core.storage import Storage

# v2 (S10): the Pipeline gains an optional ``gold_set_uri`` — the annotation file an adopter
# drops in for the gold-set harness (per-Stage precision/recall/F1 in the Quality Report,
# ADR-0008). Additive and backward-compatible (defaults to None), so a v1 declaration still
# validates.
PIPELINE_SCHEMA_VERSION = 2


class CapabilityKind(StrEnum):
    """The Capability a Stage fulfils.

    S1 spine: source/parse/export. S3 adds Chunk and Screening's two checkpoints
    (intake before Parse, content after Chunk). S4 adds Entity Extraction (typed
    entity mentions found in chunked text). S5 adds Relation Extraction (typed
    relations between mentions) and a *fused* Entity+Relation Extraction — one
    Provider fulfilling both adjacent Stages in one pass (ADR-0013), so the Pipeline
    is not forced to one Provider per Stage. S6 adds Redaction (PII detection +
    masked/replaced text variant per document), typically wired off the Chunk Stage.
    S7 adds Profiling (corpus-level statistical + quality features), wired off Parse
    (per-document features) and, optionally, Entity Extraction (cross-document entity
    features). S8 adds Disambiguation (corpus-level entity linking + duplicate merge with
    audited, confidence-weighted decisions and Evidence), wired off Entity Extraction
    (required — the mentions it resolves) and, optionally, Relation Extraction (the
    relations it normalises onto the canonical entities). S9 adds Graph Assembly (corpus-level
    knowledge-graph construction — nodes, edges, and per-edge Evidence — from the disambiguated
    canonical entities + normalized relations), wired off Disambiguation. W2-linkpred adds the
    OPTIONAL Graph Completion Capability (ADR-0037): a post-Assembly link-prediction Stage that
    consumes the assembled GraphRecords and appends PREDICTED EDGE-scope GraphRecords, each
    UNMISTAKABLY marked ``properties["inferred"] = True`` (never merged into the asserted edges,
    counted separately in the Quality Report). It depends_on Graph Assembly and is genuinely
    optional — a stack without it is byte-identical to today. S9 also adds the
    **opt-in** Embedding Capability (ADR-0017): an Embedder maps RAG-corpus text to a fixed-width
    vector so the exported corpus carries embeddings a vector DB can ingest. Embedding is *not* a
    standalone DAG node — it augments the corpus Export: a corpus Export Stage names its Embedder
    Provider in ``config['embedder']`` (a ``{provider, config}`` block) and the Runner resolves it
    and attaches a vector per exported row. It is optional — a corpus Export with no ``embedder``
    exports its text unchanged, so any vector DB can still ingest it (the ADR-0017 AC). The
    ``EMBEDDING`` kind names the seam so a Provider can register + be discovered under it; the
    in-core reference is the deterministic ``embedding.hashing`` (the learned e5-small is a
    pluggable package). W4-schema-induction adds the OPTIONAL ``SCHEMA_INDUCTION`` Capability
    (ADR-0038): a chunk→chunk Stage inserted after Content Screening and before the extraction
    Stages that uses a small OpenAI-compatible LLM to induce a document's own entity/relation/PII
    label schema and annotate its chunks (:class:`~latence_core.contracts.InducedLabels`), which the
    extraction Stages UNION onto their config labels. It is genuinely optional — a stack without a
    ``schema_induction`` Stage is byte-identical to today (the induced-labels field defaults None).
    """

    SOURCE = "source"
    INTAKE_SCREENING = "intake_screening"
    PARSE = "parse"
    CHUNK = "chunk"
    CONTENT_SCREENING = "content_screening"
    SCHEMA_INDUCTION = "schema_induction"
    ENTITY_EXTRACTION = "entity_extraction"
    RELATION_EXTRACTION = "relation_extraction"
    FUSED_ENTITY_RELATION = "fused_entity_relation"
    REDACTION = "redaction"
    PROFILING = "profiling"
    DISAMBIGUATION = "disambiguation"
    GRAPH_ASSEMBLY = "graph_assembly"
    GRAPH_COMPLETION = "graph_completion"
    EMBEDDING = "embedding"
    EXPORT = "export"


class Stage(BaseModel):
    """One node in the Pipeline DAG: a named Stage bound to a Provider."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Unique Stage name within the Pipeline.")
    capability: CapabilityKind
    provider: str = Field(..., min_length=1, description="Entry-point name of the Provider plugin.")
    config: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class Pipeline(BaseModel):
    """A DAG of Stages over a set of input documents, declared once."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=PIPELINE_SCHEMA_VERSION)
    name: str = Field(..., min_length=1)
    storage_uri: str = Field(..., min_length=1, description="Base URI for state and exports.")
    stages: list[Stage] = Field(..., min_length=1)
    # S10: an optional gold set (JSON/YAML on Storage) an adopter drops in to get per-Stage
    # precision/recall/F1 on their own corpus in the Quality Report. Opt-in — a run with no
    # gold set simply reports no gold-set section (ADR-0008).
    gold_set_uri: str | None = Field(
        default=None, description="Optional gold-set annotation file (JSON/YAML) on Storage."
    )
    # W4 (ADR-0038): a documentation/intent flag a stack sets to declare that unsupervised
    # schema-induction is ON (the ``schema_induction`` Stage is present and every extractor reads
    # its induced labels). It is an OPT-IN marker for readers + tooling, NOT a behaviour switch —
    # the actual wiring is the presence of the ``schema_induction`` Stage and the extractors'
    # additive induced∪config seam. Defaults False, so every existing stack is byte-identical.
    auto_generate_labels: bool = Field(
        default=False,
        description="Opt-in marker that this stack runs unsupervised schema induction (W4).",
    )

    @model_validator(mode="after")
    def _validate_dag(self) -> Pipeline:
        names = [s.name for s in self.stages]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            msg = f"Duplicate Stage names: {dupes}"
            raise ValueError(msg)
        known = set(names)
        for stage in self.stages:
            for dep in stage.depends_on:
                if dep not in known:
                    msg = f"Stage {stage.name!r} depends on unknown Stage {dep!r}"
                    raise ValueError(msg)
                if dep == stage.name:
                    msg = f"Stage {stage.name!r} cannot depend on itself"
                    raise ValueError(msg)
        return self

    @classmethod
    def from_yaml_text(cls, text: str) -> Pipeline:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            msg = "Pipeline YAML must be a mapping at the top level."
            raise ValueError(msg)
        return cls.model_validate(data)

    @classmethod
    def load(cls, uri: str, storage: Storage) -> Pipeline:
        return cls.from_yaml_text(storage.read_text(uri))

    def to_yaml_text(self) -> str:
        text: str = yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
        return text
