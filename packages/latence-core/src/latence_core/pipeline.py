"""Declarative Pipeline contract (ADR-0015).

A versioned YAML/JSON document is the canonical Pipeline declaration: an ordered
list of Stages, each binding a Capability kind to a Provider (plugin name + config),
plus Storage URIs. A typed Pydantic builder constructs and validates the same
object, and can round-trip back to YAML — config-as-data that platform teams diff
and commit, and what makes the one-line Provider swap literally true.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from latence_core.errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from latence_core.storage import Storage

# v2 (S10): the Pipeline gains an optional ``gold_set_uri`` — the annotation file an adopter
# drops in for the gold-set harness (per-Stage precision/recall/F1 in the Quality Report,
# ADR-0008). Additive and backward-compatible (defaults to None), so a v1 declaration still
# validates.
PIPELINE_SCHEMA_VERSION = 2

# A template placeholder: braces around an UPPER_SNAKE name, as in ``{STORAGE_URI}``,
# ``{CORPUS_DIR}`` or ``{OCR_BASE_URL}``. The shipped stacks carry these where a machine-specific
# value belongs, and whoever executes a stack fills them first (``latence stack validate`` and the
# other harnesses by substitution, an operator with ``sed`` / ``envsubst``). Three look-alikes are
# deliberately NOT placeholders, because they are legitimate config text: a lower-case format
# field (``{id}``), a doubled brace that a format template uses for a literal one
# (``{{LABELS}}``), and a named character in a regular expression (``\N{BULLET}``).
_PLACEHOLDER = re.compile(r"(?<!\{)(?<!\\N)\{[A-Z][A-Z0-9_]*\}(?!\})")


def _placeholders_in(value: object, where: str) -> Iterator[tuple[str, str]]:
    """Yield ``(location, placeholder)`` for every placeholder in a config value, in file order.

    Only string VALUES are read (a mapping's keys are names, not values), recursing through
    mappings and lists. A placeholder that occurs twice in one string is reported once.
    """
    if isinstance(value, str):
        for token in dict.fromkeys(_PLACEHOLDER.findall(value)):
            yield where, token
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _placeholders_in(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _placeholders_in(item, f"{where}[{index}]")


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
    in-core reference is the deterministic ``embedding.hashing`` (the learned Granite r2 default
    is a pluggable package). W4-schema-induction adds the OPTIONAL ``SCHEMA_INDUCTION`` Capability
    (ADR-0038): a chunk→chunk Stage inserted after Content Screening and before the extraction
    Stages that uses a small OpenAI-compatible LLM to induce a document's own entity/relation/PII
    label schema and annotate its chunks (:class:`~latence_core.contracts.InducedLabels`), which the
    extraction Stages UNION onto their config labels. It is genuinely optional — a stack without a
    ``schema_induction`` Stage is byte-identical to today (the induced-labels field defaults None).
    T3 (ADR-0054) adds the OPTIONAL ``TYPE_CONSOLIDATION`` Capability: a corpus-level
    mention→mention Stage placed between extraction and Disambiguation that canonicalizes the
    run's TYPE vocabulary. Per-chunk label induction gives each chunk a small, relevant, uncapped
    label set — and produces type drift (``org``/``organization``/``company``), because nothing
    forces one schema upstream any more. This Stage folds the drifted labels into canonical
    clusters, rewrites every mention's and relation's ``label`` (keeping the raw induced label on
    ``raw_label`` for audit) and emits a :class:`~latence_core.contracts.TypeVocabulary`. It is
    the type-side twin of Disambiguation: consistency is EARNED downstream in the resolution
    phase, never IMPOSED upstream by capping a corpus schema. Genuinely optional — a stack
    without it is byte-identical to today (``raw_label`` defaults None and no label is rewritten).
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
    TYPE_CONSOLIDATION = "type_consolidation"
    DISAMBIGUATION = "disambiguation"
    GRAPH_ASSEMBLY = "graph_assembly"
    GRAPH_COMPLETION = "graph_completion"
    EMBEDDING = "embedding"
    EXPORT = "export"


# The Capability kinds that consume chunk text to extract mentions/relations — the Stages whose
# input the ``extract_on: masked`` mode (W18, ADR-0044) redirects from the raw chunk stream to the
# redacted one. They, and only they, read ``ChunkRecord.content`` to find entities/relations, so
# masking THEIR input is what keeps PII out of the KG under the stricter posture.
_EXTRACTION_KINDS: frozenset[CapabilityKind] = frozenset(
    {
        CapabilityKind.ENTITY_EXTRACTION,
        CapabilityKind.RELATION_EXTRACTION,
        CapabilityKind.FUSED_ENTITY_RELATION,
    }
)


class ExtractOn(StrEnum):
    """Which chunk-text variant the extraction Stages read (W18, ADR-0044).

    * ``unmasked`` (default) — extraction reads the raw chunk ``content``, so the KG keeps the real
      entities. Byte-identical to pre-W18: redaction masks only the exported RAG corpus text.
    * ``masked`` — extraction reads the REDACTED chunk stream (``masked_content``), so PII never
      reaches extraction / the KG at all (the stricter privacy posture some enterprises require).
      The extraction Stages must depend (transitively) on a redaction Stage so ``masked_content``
      is populated before they run — the Pipeline validator enforces this.
    """

    UNMASKED = "unmasked"
    MASKED = "masked"


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
    # W18 (ADR-0044): which chunk-text variant the extraction Stages read. Default ``unmasked`` is
    # byte-identical to pre-W18 (the KG keeps real entities; redaction masks only the exported
    # corpus). ``masked`` feeds extraction the REDACTED chunk stream so PII never reaches the KG —
    # the stricter posture. It is a genuine behaviour switch (unlike ``auto_generate_labels``); the
    # validator below enforces that the extraction Stages depend on a redaction Stage when it is on.
    extract_on: ExtractOn = Field(
        default=ExtractOn.UNMASKED,
        description="Extraction reads unmasked chunk text (default) or the redacted stream.",
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
        self._validate_extract_on()
        return self

    def _validate_extract_on(self) -> None:
        """``extract_on: masked`` needs each extraction Stage to run AFTER a redaction Stage (W18).

        The masked posture works by feeding extraction the redacted chunk stream, so the
        ``masked_content`` must be populated before an extractor reads it. If an extraction Stage
        does not transitively depend on a redaction Stage the mode would silently fall back to
        unmasked text (a privacy hole), so this is a loud, structural error rather than a quiet
        no-op. Default ``unmasked`` skips the check entirely — every existing stack is unaffected.
        """
        if self.extract_on is not ExtractOn.MASKED:
            return
        by_name = {s.name: s for s in self.stages}
        redaction_stages = {
            s.name for s in self.stages if s.capability is CapabilityKind.REDACTION
        }
        if not redaction_stages:
            msg = (
                "extract_on: masked requires a redaction Stage (the extraction Stages read its "
                "masked_content), but the Pipeline declares none."
            )
            raise ValueError(msg)

        def _reaches_redaction(start: str) -> bool:
            seen: set[str] = set()
            frontier = list(by_name[start].depends_on)
            while frontier:
                dep = frontier.pop()
                if dep in redaction_stages:
                    return True
                if dep in seen:
                    continue
                seen.add(dep)
                frontier.extend(by_name[dep].depends_on)
            return False

        for stage in self.stages:
            if stage.capability in _EXTRACTION_KINDS and not _reaches_redaction(stage.name):
                msg = (
                    f"extract_on: masked requires extraction Stage {stage.name!r} to depend "
                    "(transitively) on a redaction Stage so it reads masked_content; wire "
                    "its depends_on through the redact Stage (or re-run `latence setup`)."
                )
                raise ValueError(msg)

    def unresolved_placeholders(self) -> list[tuple[str, str]]:
        """Every template placeholder this Pipeline still carries, as ``(location, placeholder)``.

        Reads ``storage_uri``, ``gold_set_uri`` and every Stage's config, in declaration order. An
        empty list means the Pipeline is concrete. Parsing never rejects a placeholder — the
        static tools (the stack check, the capability guard, the tuner's roster) read the shipped
        templates as they are — so this is asked by whoever is about to EXECUTE the Pipeline.
        """
        found = list(_placeholders_in(self.storage_uri, "storage_uri"))
        found += _placeholders_in(self.gold_set_uri, "gold_set_uri")
        for stage in self.stages:
            found += _placeholders_in(stage.config, f"stages.{stage.name}.config")
        return found

    def require_resolved(self) -> None:
        """Raise :class:`~latence_core.errors.ConfigError` if a template placeholder is unfilled.

        Called at the top of every path that executes a Pipeline, before anything is written. A
        Pipeline that still says ``storage_uri: "{STORAGE_URI}"`` is a template, not a run: executed
        as it is, it writes into a directory literally named ``{STORAGE_URI}`` under the working
        directory, reads no document from the relative path ``{CORPUS_DIR}`` and reports an empty
        corpus as a completed run. The message names each location and its placeholder and no
        other config value, so it cannot quote a credential.
        """
        found = self.unresolved_placeholders()
        if not found:
            return
        listing = "; ".join(f"{where} carries {token}" for where, token in found)
        msg = (
            f"Pipeline {self.name!r} is a template with unfilled placeholder(s): {listing}. "
            "A Pipeline is executed as written; nothing substitutes these at run time. Replace "
            "each one with a real value in a copy of the file (docs/running-a-stack.md, "
            "'Running a stack directly'), or use `latence stack validate`, which fills "
            "{STORAGE_URI} and {CORPUS_DIR} itself."
        )
        raise ConfigError(msg)

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
