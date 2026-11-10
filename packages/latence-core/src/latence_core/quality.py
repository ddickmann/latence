"""Quality Report — a first-class run artifact (ADR-0008).

The S1 skeleton reports what the walking skeleton can substantiate: document
counts, per-Stage timing, and a contract-completeness check (every record has
Provenance and Classification). Later slices extend this same artifact with
per-Stage quality metrics, KG statistics, and gold-set precision/recall.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from latence_core.conformance.report import ConformanceReport
from latence_core.contracts import DeltaChurn, ScreeningFinding

# v3 (S3): the report gains ``chunk`` (Chunk-Stage counts + offset-integrity) and
# ``screening`` (both checkpoints' quarantine/flag findings).
# v4 (S4): the report gains ``entities`` (Entity-Extraction per-type mention counts
# and the confidence distribution).
# v5 (S5): the report gains ``relations`` (Relation-Extraction per-type relation
# counts and the confidence distribution).
# v6 (S6): the report gains ``redaction`` (Redaction per-PII-type detection counts,
# per-action counts, and documents-with-PII) — the S6 acceptance criterion, and NEVER
# any raw PII value (counts only).
# v7 (S7): the report gains ``profiling`` (Profiling per-document feature averages +
# the corpus-level aggregates — document/word/vocabulary counts, corpus Zipf α,
# source coverage, top entity co-occurrences) — the S7 acceptance criterion ("feature
# summary lands in the Quality Report").
# v8 (S8): the report gains ``disambiguation`` (Disambiguation merge/link statistics —
# mentions in, canonical entities out, entities created/merged/singleton, merges applied
# vs below-policy per strategy, KB-linked vs unlinked, and relations normalized/dropped) —
# the S8 acceptance criterion ("merge/link statistics land in the Quality Report").
# v9 (S9): the report gains ``graph`` (Graph-Assembly KG statistics — node/edge counts, nodes
# by type, edges by label, KB-linked node count, and evidence coverage: the fraction of edges
# that carry Evidence) — the S9 acceptance criterion ("KG statistics land in the Quality
# Report"). Counts only (no raw text), safe to persist.
# v10 (S10): the Quality Report is promoted to the full first-class artifact (ADR-0008). The
# ``completeness`` check gains offset-alignment (records whose Provenance char/page spans
# resolve) so contract-completeness covers "offsets aligned", not just Provenance/Classification
# present. The report gains ``drift`` (the corpus-level roll-up of the PageOffsetIndex drift
# diagnostics across every parsed document — the S10 drift-diagnostics AC) and ``goldset`` (the
# gold-set harness result: per-Stage precision/recall/F1 against the adopter's own annotations,
# present only when a gold set was supplied). A ``render()`` produces the human-readable summary
# (the machine-readable JSON + human-readable rendering AC). Counts only, safe to persist.
# v11 (S11): the report gains ``delta`` — the :class:`~latence_core.contracts.DeltaChurn` of an
# incremental delta run (documents ±, entities created/merged/split, edges ±, records
# retracted/purged, drift, and whether a full Reconciliation was triggered) — the S11 acceptance
# criterion ("delta churn lands in the Quality Report"). Present only on a delta run (a fresh
# full run reports ``delta=None``). Counts only (no raw text), safe to persist.
# v12 (S12): each :class:`StageMetrics` gains ``runner`` (which Runner executed the Stage —
# ``"local"`` or a second-Runner adapter name, proving pluggable execution, ADR-0003) and
# ``endpoint_retries`` (how many transient endpoint failures a Stage retried through — endpoint
# fault-tolerance surfaced in the report, S12 AC). Additive with defaults, so a v11 report still
# validates.
# v13 (#63, ADR-0033): :class:`RelationQuality` gains ``fanout_capped_documents`` and
# ``fanout_dropped_relations`` — the audited quadratic-fan-out-guard marker (how many documents
# hit the per-document relation cap and how many relations were dropped as a result). Both default
# to 0, so a v12 report still validates, but a v12 and a v13 report differ in the RelationQuality
# shape, hence the bump.
# v14 (#73, Hardening observability): each :class:`StageMetrics` gains ``error_count`` and
# ``error_category`` — the per-Stage typed-error observability. A Stage that raised a
# :class:`~latence_core.errors.LatenceError` (or any exception) records ``error_count=1`` and the
# stable ``error_category`` tag (``"contract"``/``"config"``/``"provider"``/``"storage"``/
# ``"screening"``/``"unknown"``); a Stage that succeeded records ``error_count=0`` and
# ``error_category=None``. Additive with defaults, so a v13 report still validates.
# v15 (#68): :class:`RelationQuality` gains ``mention_capped_documents`` and
# ``mention_dropped_mentions`` — the audited input-side mention-cap marker (how many documents
# hit the per-document mention INPUT cap that bounds the O(mentions²) candidate transient, and
# how many mentions were dropped). Both default to 0, so a v14 report still validates, but a v14
# and a v15 report differ in the RelationQuality shape, hence the bump.
# v16 (P3-F1, ADR-0036 §2): each :class:`StageMetrics` gains the provider-profile + device-routing
# columns surfaced from the declared :class:`~latence_core.providers.profile.ProviderProfile` and
# the :func:`~latence_core.providers.device.select_device` decision the Runner made for the Stage:
# ``compute`` (cpu|gpu|either|None), ``model_id``, ``license`` (SPDX or UNVERIFIED), the requested
# ``device`` config value, the ``device_selected`` the seam resolved (cpu/cuda/None), ``skipped``
# (True when a GPU-only provider was skipped-with-flag on a CPU-only host — the G1 posture), and
# ``skip_reason``. All default to None/False, so a v15 report still validates; a run whose
# providers declare no profile leaves them None (never fabricated).
# v17 (P3-F3, ADR-0036 §5): the report gains an optional ``conformance`` — the
# :class:`~latence_core.conformance.report.ConformanceReport` roll-up of the enterprise-readiness
# gate (every registered Provider's C1–C6 verdict: valid typed records, graceful failure, license
# recorded + permissive-or-opt-in, determinism-or-documented, device honored, no PII leak). Counts
# only (no record content), safe to persist. Present only when a conformance run was attached
# (the gate is a test-time/tooling sweep, not a per-Stage run) — a normal pipeline run reports
# ``conformance=None``, so a v16 report still validates. The report models live in the pure,
# stage-free :mod:`latence_core.conformance.report`, so the lean quality contract imports no
# heavy harness/fixtures.
# v18 (H-C1 §4): :class:`RedactionQuality` gains ``redaction_disabled_for_sensitive`` — the
# LOUD count of sensitive-classified documents whose redaction control ran no detectors
# (skip / empty types / empty detector set). While it is > 0, ``documents_with_pii: 0`` is
# NOT a clean bill of health — a control was disabled on a sensitive doc. Additive and
# defaulted 0, so a v17 report still validates.
# v19 (W13 context enrichment, ADR-0039): the report gains an optional ``context_enrichment`` —
# a :class:`ContextEnrichmentQuality` roll-up (``chunks_enriched`` / ``avg_neighbors_per_chunk`` /
# ``hub_entities_capped``), counts-only (no content), present only when the run had a
# ``context_enrichment`` Stage. Additive and defaulted None, so a v18 report still validates.
# v20 (the extraction redesign) carries TWO additive report changes that ship in the SAME revision —
# they land together, so no report ever exists with one and not the other:
# v20a (T5, per-chunk induction cost): :class:`StageMetrics` gains the induction COST columns —
# ``induction_calls`` / ``induction_cache_hits`` / ``induction_prompt_tokens`` /
# ``induction_completion_tokens`` / ``induction_budget_exhausted``. Per-chunk induction is one LLM
# call per distinct chunk text, so a run's induction cost has to be readable in the report rather
# than inferred; the dedup cache-hit count is what shows the saving. Additive and defaulted
# 0/False, so a v19 report still validates.
# v20b (T3 type canonicalization, ADR-0054): the report gains an optional ``type_consolidation`` —
# a :class:`TypeConsolidationQuality` roll-up ("N raw induced types -> M canonical types" plus the
# alias->canonical map actually applied, for entities and relations), present only when the run had
# a ``type_consolidation`` Stage. It is the operator's window onto the downstream canonicalization
# that makes per-chunk induction safe: every fold is visible and correctable through the Stage's
# ``aliases`` config. Type labels + counts only (never document text), safe to persist. Additive
# and defaulted None, so a v19 report still validates.
# v20c (T4, incremental delta reconciliation): :class:`~latence_core.contracts.DeltaChurn` gains
# ``canonical_types_added`` / ``type_aliases_added`` / ``canonical_types_total`` — how the delta
# EXTENDED the corpus's canonical type vocabulary (a genuinely new type vs a drifted surface that
# aliased into an existing one). It replaces the frozen-schema story for per-chunk induction, so
# the operator can see the KG's type space growing monotonically. Counts only. Additive and
# defaulted 0, and the rendered churn line is emitted only when the corpus HAS a vocabulary, so a
# report from a pipeline without a Type-Consolidation Stage is byte-identical. No number bump: the
# extraction-redesign report changes ship in one revision (same rule as ADR-0043's own additive
# DeltaChurn fields, which also did not bump).
# v21 (ADR-0068 — the query-time retrieval tooling is removed): the two sections that reported on
# retrieval-only Stages leave the report — ``context_enrichment`` (``ContextEnrichmentQuality``,
# v19) and ``graph.hyperedges`` (``HyperedgeQuality``, with its token-cost distribution). A
# removal, not an additive change: the report models are ``extra="forbid"``, so a v20 report that
# carries either key no longer validates, hence the bump.
QUALITY_SCHEMA_VERSION = 21


class StageMetrics(BaseModel):
    """Per-Stage observability rolled into the Quality Report."""

    model_config = ConfigDict(extra="forbid")

    name: str
    capability: str
    provider: str
    records_in: int = 0
    records_out: int = 0
    duration_seconds: float = 0.0
    from_checkpoint: bool = Field(
        default=False, description="True if this Stage was skipped and resumed from a checkpoint."
    )
    # S12: which Runner executed the Stage (``"local"`` for the built-in LocalRunner, an adapter
    # name like ``"airflow"`` for a second Runner). Proves execution is pluggable (ADR-0003): the
    # same Stage semantics under a different substrate, visible in the Quality Report.
    runner: str = Field(default="local")
    # S12: endpoint-retry observability for an endpoint-backed performance Provider — how many
    # transient failures this Stage retried through before succeeding (0 for a CPU/in-process
    # Provider or a first-try endpoint call). Surfaced so an operator sees a flaky endpoint in the
    # Quality Report rather than only in logs (S12 AC: failures retried/surfaced).
    endpoint_retries: int = Field(default=0)
    # #73 (Hardening observability): the typed-error taxonomy surfaced per Stage. ``error_count``
    # is 0 for a Stage that succeeded and 1 for the Stage that raised. On a Stage failure the Runner
    # persists a FAIL Quality Report — the completed Stages plus the failed one (error_count=1) —
    # and THEN re-raises, so the failed Stage's category lands in the report + logs, not only a
    # transient span; a subsequent successful resume overwrites the report atomically (the failed
    # Stage's checkpoint was never finalized, so no torn write). ``error_category`` is the stable
    # :mod:`latence_core.errors` tag
    # (``"contract"``/``"config"``/``"provider"``/``"storage"``/``"screening"``/``"unknown"``) of
    # that failure, or ``None`` on success. Counts + a category tag only — never the record content
    # that triggered the error (the no-PII discipline the whole report keeps).
    error_count: int = Field(default=0)
    error_category: str | None = Field(default=None)
    # W4 (schema induction, ADR-0038): how many documents the OPTIONAL ``schema_induction`` Stage
    # **failed open** on — an induction that hit a safety net (malformed/truncated/schema-invalid/
    # empty output, or an unreachable endpoint) and fell back to the extractors' config labels so
    # the run continued. 0 for a Stage that induced every document cleanly, and 0 for every
    # non-induction Stage (it never populates this). A count only — never the failing document's
    # content (the no-PII discipline the whole report keeps) — the Quality-Report note the W4
    # fail-open-never-crash posture requires, so an operator sees the LLM degraded rather than only
    # a log line. ``induction_documents`` is the total documents the Stage attempted, so the note
    # reads "N of M documents fell back to config labels".
    induction_fallbacks: int = Field(default=0)
    induction_documents: int = Field(default=0)
    # Robustness fix (induce empty-after-sanitization): the LOUD subset of ``induction_fallbacks``
    # where the induction PARSED + VALIDATED but sanitized to an empty / below-min label set and
    # STILL did so after the empty-recovery re-prompt — i.e. the corpus induced nothing usable and
    # the run degraded to the bare config-label floor (the quality cliff the old silent fallback
    # hid). Surfaced on its own field so an operator sees "N documents induced no domain schema"
    # directly rather than inferring it from a generic fallback count. 0 when every induction
    # produced labels (or when a fallback was a different safety-net miss). A count only — no PII.
    induction_empty_after_sanitization: int = Field(default=0)
    # T5 (per-chunk induction cost): what the induce Stage actually COST, so "induce every chunk"
    # is a number an operator can read rather than a number they have to fear. Per-chunk induction
    # is one LLM call per DISTINCT chunk text, so the honest report needs the calls actually made,
    # how many items the dedup cache answered for free, and the endpoint's own token accounting:
    #   ``induction_calls``            — chat-completion requests issued (a repair / truncation /
    #                                    empty-recovery re-prompt is its own request; transport
    #                                    retries are counted on ``endpoint_retries`` instead, so the
    #                                    two never double-count).
    #   ``induction_cache_hits``       — items served from the chunk-text dedup cache with NO call.
    #                                    The cache-hit RATE is ``cache_hits / induction_documents``
    #                                    (items answered for free / items attempted).
    #   ``induction_prompt_tokens`` /
    #   ``induction_completion_tokens``— summed from each response's ``usage`` block. 0 means the
    #                                    endpoint reported no usage (some on-prem vLLM configs) —
    #                                    an honest unknown, NEVER an estimated token count.
    #   ``induction_budget_exhausted`` — True iff the configured per-run call budget bound the run
    #                                    (jobs beyond it fell back to the config-label floor). The
    #                                    LOUD bit: a bounded run must never look like a full one.
    # All counts only, no content — the no-PII discipline the whole report keeps. 0/False for every
    # non-induction Stage.
    induction_calls: int = Field(default=0)
    induction_cache_hits: int = Field(default=0)
    induction_prompt_tokens: int = Field(default=0)
    induction_completion_tokens: int = Field(default=0)
    induction_budget_exhausted: bool = Field(default=False)
    # P3-F1 (ADR-0036 §2): the provider-profile + device-routing columns for this Stage,
    # read from the Provider's declared ProviderProfile and the device seam's decision.
    # A Provider that declares no profile leaves ``compute``/``model_id``/``license`` None
    # (never fabricated). ``device`` is the requested config value (``auto`` default);
    # ``device_selected`` is what ``select_device`` resolved (``"cpu"``/``"cuda"``, or None
    # when skipped); ``skipped`` + ``skip_reason`` record the G1 skip-with-flag posture for a
    # GPU-only provider on a CPU-only host. All counts/flags only — no fabricated GPU numbers.
    compute: str | None = Field(default=None)
    model_id: str | None = Field(default=None)
    license: str | None = Field(default=None)
    device: str | None = Field(default=None)
    device_selected: str | None = Field(default=None)
    skipped: bool = Field(default=False)
    skip_reason: str | None = Field(default=None)


class CompletenessCheck(BaseModel):
    """Contract-completeness: Provenance + Classification present AND offsets aligned (S10).

    Every exported record must carry Provenance and Classification (enforced by the
    contract, audited here) — but the S10 promotion also audits that a record's offsets
    are *aligned*: a record that carries a char span must carry a **well-ordered** one
    (``char_end >= char_start``), and if it also carries page coordinates those must be
    well-ordered too (``page_end >= page_start``). ``records_with_aligned_offsets`` counts
    records that pass this offset audit; ``offset_bearing_records`` is how many records
    carried any char span to audit (a record with no char span at all is vacuously aligned
    and not counted). So ``offsets_aligned`` is True only when every offset-bearing
    record's span is internally consistent — the "offsets aligned" half of the
    completeness AC. (Page *presence* is separately guaranteed at construction for the
    sub-document carriers whose contracts require it — EntityMention/RelationMention — so
    this audit does not re-require it for whole-document carriers that legitimately have a
    ``[0, len)`` char span and no page span.)
    """

    model_config = ConfigDict(extra="forbid")

    records_checked: int = 0
    records_with_provenance: int = 0
    records_with_classification: int = 0
    offset_bearing_records: int = 0
    records_with_aligned_offsets: int = 0

    @property
    def offsets_aligned(self) -> bool:
        """True if every offset-bearing record carried a well-ordered, page-resolved span."""
        return self.records_with_aligned_offsets == self.offset_bearing_records

    @property
    def passed(self) -> bool:
        return (
            self.records_with_provenance == self.records_checked
            and self.records_with_classification == self.records_checked
            and self.offsets_aligned
        )


class ParseQuality(BaseModel):
    """Parse-Stage quality: page-drift diagnostics + parse-error counts (S2).

    Emitted into the Quality Report when a run parsed documents. ``page_drift_*``
    aggregate the fuzzy offset→page recoveries the ported ``PageOffsetIndex`` made
    while the Runner audited each parsed record's page map — a clean parse has zero
    drift. ``parse_errors`` counts documents the Parser could not decode (emitted
    as ``PARSE_ERROR`` records rather than crashing the run). ``bytes_after_parse``
    sums the assembled-markdown byte size of every parsed record — the observable
    Provenance 'size-after-parse' rolled up across the Stage.
    """

    model_config = ConfigDict(extra="forbid")

    documents_parsed: int = 0
    parse_errors: int = 0
    pages_total: int = 0
    page_map_present: int = 0
    bytes_after_parse: int = 0
    page_drift_resolutions: int = 0
    page_drift_recovered: int = 0
    page_drift_max_chars: int = 0

    @property
    def page_maps_clean(self) -> bool:
        """True if every offset→page audit resolved exactly (no fuzzy recovery)."""
        return self.page_drift_recovered == 0


class ChunkQuality(BaseModel):
    """Chunk-Stage quality: chunk/token counts + offset-preservation integrity (S3).

    Emitted when a run chunked documents. ``chunks_offset_preserving`` counts
    chunks whose Provenance carried a valid source char span AND page span — the
    S3 round-trip invariant (chunk → offset → original page). ``offset_preserving``
    is True only when every chunk preserved its offsets. ``chunks_flagged`` counts
    chunks Content Screening marked with a risk marker.
    """

    model_config = ConfigDict(extra="forbid")

    chunks_total: int = 0
    chunks_offset_preserving: int = 0
    chunks_flagged: int = 0
    tokens_total: int = 0

    @property
    def offset_preserving(self) -> bool:
        """True if every emitted chunk carried a resolvable source offset+page span."""
        return self.chunks_offset_preserving == self.chunks_total


class ScreeningQuality(BaseModel):
    """Screening quality: quarantine + flag dispositions across both checkpoints (S3).

    Emitted when a run had a Screening Stage. Counts are rolled up from the
    per-record :class:`~latence_core.contracts.ScreeningFinding`s, which are also
    carried verbatim in ``findings`` so every Quarantine disposition and reason is
    inspectable in the Quality Report (S3 AC: "Quarantine dispositions and reasons
    appear in the Quality Report").
    """

    model_config = ConfigDict(extra="forbid")

    intake_screened: int = 0
    content_screened: int = 0
    quarantined: int = 0
    flagged: int = 0
    passed: int = 0
    findings: list[ScreeningFinding] = Field(default_factory=list)

    @property
    def total_screened(self) -> int:
        return self.intake_screened + self.content_screened


class ConfidenceHistogram(BaseModel):
    """A fixed-bin histogram of mention confidences (the confidence distribution, S4).

    Ten equal-width bins over ``[0.0, 1.0]``: ``bins[i]`` counts mentions whose
    confidence falls in ``[i/10, (i+1)/10)`` — except a confidence of exactly 1.0,
    which lands in the top bin. Deterministic and dependency-free, so a seeded run
    yields the identical distribution.
    """

    model_config = ConfigDict(extra="forbid")

    bins: list[int] = Field(default_factory=lambda: [0] * 10)

    def add(self, confidence: float) -> None:
        """Increment the bin ``confidence`` falls in (1.0 lands in the top bin)."""
        clamped = min(max(confidence, 0.0), 1.0)
        idx = min(int(clamped * 10), 9)
        self.bins[idx] += 1

    @property
    def count(self) -> int:
        return sum(self.bins)

    @property
    def mean(self) -> float:
        """Mean confidence, estimated at each bin's midpoint (0.0 if empty)."""
        total = self.count
        if total == 0:
            return 0.0
        weighted = sum(count * ((i + 0.5) / 10) for i, count in enumerate(self.bins))
        return round(weighted / total, 6)


class EntityQuality(BaseModel):
    """Entity-Extraction quality: per-type mention counts + confidence distribution (S4).

    Emitted when a run had an Entity-Extraction Stage. ``mentions_total`` is the
    number of typed :class:`~latence_core.contracts.EntityMention`s produced;
    ``per_type`` breaks that down by entity label (the zero-shot label set the
    Pipeline configured); ``confidence`` is the fixed-bin distribution of mention
    confidences. ``mentions_offset_resolving`` counts mentions whose Provenance
    carried a valid source char span AND page span — the S4 round-trip invariant
    (mention → offset → original page).
    """

    model_config = ConfigDict(extra="forbid")

    mentions_total: int = 0
    mentions_offset_resolving: int = 0
    chunks_with_mentions: int = 0
    per_type: dict[str, int] = Field(default_factory=dict)
    confidence: ConfidenceHistogram = Field(default_factory=ConfidenceHistogram)

    @property
    def offset_resolving(self) -> bool:
        """True if every emitted mention carried a resolvable source offset+page span."""
        return self.mentions_offset_resolving == self.mentions_total


class RelationQuality(BaseModel):
    """Relation-Extraction quality: per-type relation counts + confidence dist (S5).

    Emitted when a run had a Relation-Extraction (or fused) Stage. ``relations_total``
    is the number of typed :class:`~latence_core.contracts.RelationMention`s produced;
    ``per_type`` breaks that down by relation label (the per-Pipeline relation label
    set); ``confidence`` is the fixed-bin distribution of relation confidences.
    ``relations_offset_resolving`` counts relations whose Provenance carried a valid
    source char span AND page span — the S5 round-trip (relation → offset → page).
    ``documents_with_relations`` counts distinct parent documents that yielded at
    least one relation. The counts + distribution are the S5 acceptance criterion
    ("relation counts + confidence land in the Quality Report").

    ``fanout_capped_documents`` / ``fanout_dropped_relations`` are the audited
    quadratic-fan-out-guard marker (ADR-0033 / #63): how many documents hit the
    per-document relation OUTPUT cap and how many relations were dropped as a result.
    ``mention_capped_documents`` / ``mention_dropped_mentions`` are the audited
    input-side marker (#68): how many documents hit the per-document mention INPUT cap
    (bounding the ``O(mentions²)`` candidate transient before the output cap trims) and
    how many mentions were dropped as a result. All zero on a normal run; non-zero
    surfaces that a low-cardinality/pathological corpus was bounded rather than silently
    allowed to blow up records + downstream graph edges (or the transient candidate list).
    """

    model_config = ConfigDict(extra="forbid")

    relations_total: int = 0
    relations_offset_resolving: int = 0
    documents_with_relations: int = 0
    fanout_capped_documents: int = 0
    fanout_dropped_relations: int = 0
    mention_capped_documents: int = 0
    mention_dropped_mentions: int = 0
    per_type: dict[str, int] = Field(default_factory=dict)
    confidence: ConfidenceHistogram = Field(default_factory=ConfidenceHistogram)

    @property
    def offset_resolving(self) -> bool:
        """True if every emitted relation carried a resolvable source offset+page span."""
        return self.relations_offset_resolving == self.relations_total


class RedactionQuality(BaseModel):
    """Redaction-Stage quality: per-PII-type + per-action detection counts (S6).

    Emitted when a run had a Redaction Stage. ``pii_total`` is the number of detected
    :class:`~latence_core.contracts.PIISpan`s across the corpus; ``per_type`` breaks
    that down by PII type (e.g. ``email``, ``ssn``) and ``per_action`` by the
    :class:`~latence_core.contracts.RedactionAction` taken (mask/replace/hash/tag).
    ``documents_redacted`` is the number of RedactionRecords (one per document);
    ``documents_with_pii`` counts those that had at least one detected span. The
    per-type + per-action counts are the S6 acceptance criterion ("PII detection
    counts by type land in the Quality Report"). **No raw PII value ever appears
    here** — the report carries counts and types only, never a detected surface
    string (S6 AC: no raw PII written to logs/records).
    """

    model_config = ConfigDict(extra="forbid")

    documents_redacted: int = 0
    documents_with_pii: int = 0
    pii_total: int = 0
    per_type: dict[str, int] = Field(default_factory=dict)
    per_action: dict[str, int] = Field(default_factory=dict)
    redaction_disabled_for_sensitive: int = Field(
        default=0,
        description=(
            "Count of sensitive-classified documents (confidential/restricted/…) whose "
            "redaction control ran NO detectors (skip / empty types / empty detector "
            "set). A LOUD, first-class signal (H-C1 §4): while it is > 0, "
            "``documents_with_pii: 0`` does NOT mean the corpus is clean — a control was "
            "disabled on a sensitive doc. Distinct from 'genuinely no PII found'."
        ),
    )


class ProfilingQuality(BaseModel):
    """Profiling-Stage quality: per-document feature averages + corpus aggregates (S7).

    Emitted when a run had a Profiling Stage. The document-scope half rolls the
    per-document :class:`~latence_core.contracts.DocumentFeatures` into simple averages
    (a compact summary of the corpus's readability / density / diversity), and the
    corpus-scope half surfaces the cross-document aggregates the single
    :class:`~latence_core.contracts.CorpusFeatures` record already computed. This is the
    S7 acceptance criterion — the feature summary lands in the Quality Report — kept to
    counts and averages (no raw document text).

    ``documents_profiled`` is the number of DOCUMENT-scope FeatureRecords; the
    ``avg_*`` fields are their means (0.0 for an empty run). The corpus fields mirror
    the CORPUS-scope record; ``top_co_occurrences`` is the capped, ranked co-occurrence
    list rendered as ``pair-fingerprint`` → document-count. The key is a deterministic,
    **salted** sha256 fingerprint (``"pair:<hex>"``) of the entity pair — the same
    salted-digest construction S6 uses for PII surrogates — *not* the raw ``"left|right"``
    surfaces: a pair from a ``confidential`` source must not have its (possibly PII)
    surface printed into this ``internal``-stamped report. The salt (from the Redaction
    Stage config) defeats rainbow-table recovery, but the digest is a fingerprint, not
    encryption — low-entropy surfaces stay dictionary-recoverable by a reader who knows
    the salt, so this report is not safe to publish to an untrusted audience. The raw
    surfaces live only in the exported, sensitivity-stamped ``CorpusFeatures`` data
    artifact. Equal pairs share a key, so the recurrence signal is preserved.
    """

    model_config = ConfigDict(extra="forbid")

    documents_profiled: int = 0
    avg_readability: float = 0.0
    avg_density: float = 0.0
    avg_type_token_ratio: float = 0.0
    avg_zipf_alpha: float = 0.0
    avg_compression_ratio: float = 0.0
    corpus_total_words: int = 0
    corpus_unique_words: int = 0
    corpus_zipf_alpha: float = 0.0
    language_coverage: dict[str, int] = Field(default_factory=dict)
    category_coverage: dict[str, int] = Field(default_factory=dict)
    sensitivity_coverage: dict[str, int] = Field(default_factory=dict)
    distinct_entities: int = 0
    entity_type_frequency: dict[str, int] = Field(default_factory=dict)
    top_co_occurrences: dict[str, int] = Field(default_factory=dict)


class TypeConsolidationQuality(BaseModel):
    """Type-Consolidation quality: how much TYPE drift the run collapsed (T3, ADR-0054).

    Emitted when a run had a Type-Consolidation Stage. This is the operator-facing half of the
    T3 contract: per-chunk induction is only safe if a human can SEE what the downstream
    canonicalization did and correct it, so the report carries both the headline (``N`` raw
    induced types → ``M`` canonical types) and the full ``alias → canonical`` map that produced
    it. An operator who disagrees with a fold sets the Stage's ``aliases`` config and the next run
    is different — deterministically, and without touching a model.

    Every number here is DERIVED FROM THE EMITTED RECORDS (each mention's ``raw_label`` next to
    its canonical ``label``), not from the Stage's self-report, so the report cannot claim a
    consolidation the records do not show.

    ``raw_entity_types`` / ``canonical_entity_types`` are the distinct entity-type counts before
    and after (equal ⇒ nothing was folded); the relation pair is the same for relation types.
    ``entity_alias_map`` / ``relation_alias_map`` are the rewrites actually applied, sorted.
    ``mentions_remapped`` / ``relations_remapped`` count the records whose label changed.
    Counts + type labels only (never document text), so it is safe to persist.
    """

    model_config = ConfigDict(extra="forbid")

    raw_entity_types: int = 0
    canonical_entity_types: int = 0
    raw_relation_types: int = 0
    canonical_relation_types: int = 0
    mentions_remapped: int = 0
    relations_remapped: int = 0
    entity_alias_map: dict[str, str] = Field(default_factory=dict)
    relation_alias_map: dict[str, str] = Field(default_factory=dict)


class DisambiguationQuality(BaseModel):
    """Disambiguation-Stage quality: merge + link + normalisation statistics (S8).

    Emitted when a run had a Disambiguation Stage. The S8 acceptance criterion — "merge/link
    statistics (created/merged/singleton, confidence) land in the Quality Report" — kept to
    counts (no raw document text), so it is safe to persist.

    ``mentions_in`` is the number of :class:`~latence_core.contracts.EntityMention`s the
    Stage resolved; ``entities_out`` the number of :class:`~latence_core.contracts.CanonicalEntity`s
    produced. ``entities_merged`` counts canonical entities that folded in more than one
    mention (a real merge), ``entities_singleton`` those of exactly one mention (no
    duplicate found) — together the created/merged/singleton breakdown (the algorithm only
    merges, never splits). ``merges_applied`` / ``merges_below_policy`` count audited merge
    decisions that were folded vs recorded but NOT applied (the no-silent-over-merge audit);
    ``merges_by_strategy`` breaks the applied merges down by the resolver cascade rung.
    ``entities_linked`` / ``entities_unlinked`` count canonical entities matched to the
    external KB vs the graceful fallback. ``relations_normalized`` counts relations remapped
    onto canonical entities, ``relations_inverted`` those whose direction was canonicalised
    by inverse detection. ``avg_entity_confidence`` is the mean cluster confidence, an
    at-a-glance quality signal.
    """

    model_config = ConfigDict(extra="forbid")

    mentions_in: int = 0
    entities_out: int = 0
    entities_merged: int = 0
    entities_singleton: int = 0
    merges_applied: int = 0
    merges_below_policy: int = 0
    merges_by_strategy: dict[str, int] = Field(default_factory=dict)
    entities_linked: int = 0
    entities_unlinked: int = 0
    relations_normalized: int = 0
    relations_inverted: int = 0
    avg_entity_confidence: float = 0.0


class GraphQuality(BaseModel):
    """Graph-Assembly quality: canonical knowledge-graph statistics (S9).

    Emitted when a run had a Graph Assembly Stage. The S9 acceptance criterion — "KG
    statistics (node/edge counts, evidence coverage) land in the Quality Report" — kept to
    counts (no raw document text), so it is safe to persist.

    ``nodes_total`` / ``edges_total`` are the graph's size; ``nodes_by_type`` breaks nodes down
    by entity label and ``edges_by_label`` breaks edges down by relation label.
    ``nodes_linked`` counts nodes that carry an external-KB id. ``edges_with_evidence`` counts
    edges whose Evidence names at least one source mention, and ``evidence_coverage`` is that
    fraction over all edges (1.0 = every edge is justified — the S9 "every edge carries
    Evidence" invariant, surfaced as a coverage number). ``avg_node_confidence`` /
    ``avg_edge_confidence`` are at-a-glance quality signals.
    """

    model_config = ConfigDict(extra="forbid")

    nodes_total: int = 0
    edges_total: int = 0
    nodes_linked: int = 0
    nodes_by_type: dict[str, int] = Field(default_factory=dict)
    edges_by_label: dict[str, int] = Field(default_factory=dict)
    edges_with_evidence: int = 0
    evidence_coverage: float = 0.0
    avg_node_confidence: float = 0.0
    avg_edge_confidence: float = 0.0
    # W2-linkpred (ADR-0037): predicted/inferred edges from an OPTIONAL Graph Completion Stage,
    # counted SEPARATELY — NEVER merged into the asserted ``edges_total`` / ``edges_by_label`` /
    # ``evidence_coverage`` (the enterprise-honesty crux — an inferred edge must never inflate the
    # asserted-edge statistics). ``inferred_edges_by_scorer`` breaks them down by the scorer that
    # proposed them. A run with no Graph Completion Stage has ``inferred_edge_count == 0``.
    inferred_edge_count: int = 0
    inferred_edges_by_scorer: dict[str, int] = Field(default_factory=dict)


class DriftDiagnostics(BaseModel):
    """Corpus-level Provenance-integrity drift roll-up (S10 drift-diagnostics AC).

    A finer-grained view of the offset→page drift the ported
    :class:`~latence_core.pagemap.PageOffsetIndex` recovers than the Parse-only
    :class:`ParseQuality.page_drift_*` fields: it audits every :class:`ChunkRecord` that
    carries its parent document's page map, resolving the chunk's ``[char_start, char_end)``
    endpoints through that page map and counting the fuzzy recoveries. (Only ChunkRecords
    are audited today — mentions, relations, and redaction spans inherit the chunk offsets
    the Chunk Stage already validated against the page map, so they carry no independent
    offset→page drift; :meth:`Runner._drift_diagnostics` is the single producer.)
    ``resolutions`` is how many offset→page lookups the audit performed, ``recovered`` how
    many needed the fuzzy fallback (an offset outside any exact page span — a drift), and
    ``max_drift_chars`` the worst such gap. ``records_audited`` is the number of ChunkRecords
    inspected. ``clean`` (recovered == 0) is the headline the report renders: a Provenance
    chain with zero drift means every audited chunk offset resolves exactly to its source
    page.
    """

    model_config = ConfigDict(extra="forbid")

    resolutions: int = 0
    recovered: int = 0
    max_drift_chars: int = 0
    records_audited: int = 0

    @property
    def clean(self) -> bool:
        """True if every audited offset resolved exactly (no fuzzy recovery)."""
        return self.recovered == 0


class LabelScore(BaseModel):
    """Precision/recall/F1 for a single label within a Stage's gold-set score."""

    model_config = ConfigDict(extra="forbid")

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


class StageScore(BaseModel):
    """Precision/recall/F1 for one Stage against the gold set (overall + per label).

    The per-Stage gold-set result: the corpus-total true/false-positive/negative counts
    and their standard derivations, plus the same broken down per label. ``gold_total`` /
    ``predicted_total`` are the annotation and prediction counts, so an adopter reading the
    report sees both the score and the support it rests on.
    """

    model_config = ConfigDict(extra="forbid")

    stage: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    per_label: dict[str, LabelScore] = Field(default_factory=dict)

    @property
    def gold_total(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def predicted_total(self) -> int:
        return self.true_positives + self.false_positives


class GoldSetEvaluation(BaseModel):
    """The gold-set harness result: per-Stage precision/recall/F1 on the adopter's corpus (S10).

    Present in the Quality Report only when a gold set was supplied (an adopter dropped in
    20–50 annotated documents). ``documents_annotated`` is the number of gold documents;
    ``documents_matched`` how many of those the run produced at least one extraction for
    (a mention or relation whose Provenance ``file_name`` matched a gold file). It is a
    *coverage* signal, not an ingest count: an annotated document the Source never read AND
    one the run read but from which nothing was extracted BOTH fall out of the count, so the
    gap ``documents_annotated - documents_matched`` flags un-scored coverage without
    distinguishing "never ingested" from "ingested, zero extractions". Precision/recall/F1
    are unaffected — a gold document with no matched extractions still contributes its
    annotations as false negatives. ``stages`` holds one :class:`StageScore` per evaluated
    Stage (``entities``, ``relations``). This is the S10 headline: the "high-quality,
    AI-ready data" claim made auditable on the adopter's OWN corpus, not a public benchmark.
    """

    model_config = ConfigDict(extra="forbid")

    documents_annotated: int = 0
    documents_matched: int = 0
    match_min_overlap: float = 0.0
    stages: list[StageScore] = Field(default_factory=list)


def _render_induction_cost(stages: list[StageMetrics]) -> list[str]:
    """The human-readable induction-cost section (T5); empty when the run induced nothing.

    Per-chunk induction is one call per DISTINCT chunk text, so "what did induction cost?" is the
    first question an enterprise evaluator asks about the redesign — and the honest answer has to be
    in the report an operator already reads, not derivable from JSON they have to go find. Rendered
    only for Stages that actually attempted an induction, so a stack without a ``schema_induction``
    Stage renders byte-identically to before.

    The cache-hit RATE is ``cache_hits / documents`` — the fraction of items answered for free by
    the chunk-text dedup, i.e. the size of the saving. A ``0 tokens`` reading next to a non-zero
    call count is an endpoint that reports no usage (an honest unknown), NOT a free run; a ``0
    calls`` reading next to a non-zero document count is a Provider that makes none (the
    deterministic tagger). The budget line is printed only when a budget actually BOUND the run,
    because that is the one state an operator must not miss.
    """
    active = [m for m in stages if m.induction_documents or m.induction_calls]
    if not active:
        return []
    lines = ["", "## Induction cost"]
    for m in active:
        rate = (m.induction_cache_hits / m.induction_documents) if m.induction_documents else 0.0
        lines.append(
            f"- {m.name}: {m.induction_calls} call(s) for {m.induction_documents} item(s); "
            f"cache hits {m.induction_cache_hits} ({rate:.1%}); "
            f"tokens {m.induction_prompt_tokens} prompt + {m.induction_completion_tokens} "
            f"completion (as reported by the endpoint); "
            f"fell back {m.induction_fallbacks}"
        )
        if m.induction_budget_exhausted:
            lines.append(
                f"  - BUDGET BOUND this run: item(s) beyond `max_induction_calls` made no call "
                f"and fell back to the config-label floor (see {m.name}.induction_fallbacks)"
            )
    return lines


#: Marker for an optional PER-STAGE quality section on :class:`QualityReport`.
#:
#: ``render()`` used to list the present sections from a hand-maintained ``section_map`` tuple that
#: had to be edited in lockstep with the model — one of the five edit sites adding a Stage's metric
#: required (C5). The marker moves that knowledge onto the field itself: a section field declares
#: ``Field(default=None, json_schema_extra=SECTION)`` and ``render()`` discovers it, in declaration
#: order, with its display name derived from the field name. Adding a counts-only section is then a
#: model field plus its aggregator, and nothing else.
#:
#: It rides ``json_schema_extra`` because that is pydantic's supported per-field metadata channel;
#: it is inert for ``model_dump``/``model_dump_json``, so no report byte moves.
# Typed ``dict[str, Any]`` because that is pydantic's ``JsonDict``; the value is a bool.
SECTION: dict[str, Any] = {"quality_section": True}


def _section_fields() -> list[tuple[str, str]]:
    """The marked section fields as ``(display name, field name)``, in declaration order.

    The display name is the field name in CamelCase (``type_consolidation`` ->
    ``TypeConsolidation``), which reproduces the hand-written names the ``section_map`` tuple
    carried — verified byte-for-byte by the rendered-report regression test.
    """
    return [
        ("".join(part.capitalize() for part in name.split("_")), name)
        for name, info in QualityReport.model_fields.items()
        if isinstance(info.json_schema_extra, dict)
        and info.json_schema_extra.get("quality_section") is True
    ]


class QualityReport(BaseModel):
    """The per-run report emitted alongside the data."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=QUALITY_SCHEMA_VERSION)
    pipeline_name: str
    run_id: str
    document_count: int = 0
    total_duration_seconds: float = 0.0
    stages: list[StageMetrics] = Field(default_factory=list)
    completeness: CompletenessCheck = Field(default_factory=CompletenessCheck)
    contracts_complete: bool = True
    # The optional per-Stage sections, each marked with ``SECTION`` so ``render()`` discovers it
    # instead of a hand-maintained list. Declaration order IS the rendered order.
    # Present when the run had a Parse Stage; ``None`` for a Parse-free pipeline.
    parse: ParseQuality | None = Field(default=None, json_schema_extra=SECTION)
    # Present when the run had a Chunk Stage / a Screening Stage respectively.
    chunk: ChunkQuality | None = Field(default=None, json_schema_extra=SECTION)
    screening: ScreeningQuality | None = Field(default=None, json_schema_extra=SECTION)
    # Present when the run had an Entity-Extraction Stage.
    entities: EntityQuality | None = Field(default=None, json_schema_extra=SECTION)
    # Present when the run had a Relation-Extraction (or fused) Stage.
    relations: RelationQuality | None = Field(default=None, json_schema_extra=SECTION)
    # Present when the run had a Redaction Stage.
    redaction: RedactionQuality | None = Field(default=None, json_schema_extra=SECTION)
    # Present when the run had a Profiling Stage.
    profiling: ProfilingQuality | None = Field(default=None, json_schema_extra=SECTION)
    # T3 (ADR-0054): present when the run had a Type-Consolidation Stage — how much TYPE drift
    # per-chunk induction produced and how much of it the resolution phase collapsed, with the
    # alias->canonical map an operator reviews and corrects. None otherwise.
    type_consolidation: TypeConsolidationQuality | None = Field(
        default=None, json_schema_extra=SECTION
    )
    # Present when the run had a Disambiguation Stage.
    disambiguation: DisambiguationQuality | None = Field(default=None, json_schema_extra=SECTION)
    # Present when the run had a Graph Assembly Stage.
    graph: GraphQuality | None = Field(default=None, json_schema_extra=SECTION)
    # S10: corpus-level Provenance-integrity drift roll-up (present when the run parsed
    # documents with page maps), and the gold-set harness result (present only when a
    # gold set was supplied — an adopter dropped in annotated documents).
    drift: DriftDiagnostics | None = None
    goldset: GoldSetEvaluation | None = None
    # S11: the delta-churn roll-up, present only on an incremental delta run (a fresh full run
    # reports ``delta=None``) — documents ±, entities created/merged/split, edges ±, records
    # retracted/purged, drift, and whether a full Reconciliation was triggered.
    delta: DeltaChurn | None = None
    # P3-F3 (ADR-0036 §5): the Provider-conformance roll-up — every registered Provider's C1–C6
    # enterprise-readiness verdict. Present only when a conformance sweep was attached (the gate
    # is a tooling/test-time sweep, not a per-Stage run); a normal pipeline run reports None.
    conformance: ConformanceReport | None = None

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def render(self) -> str:
        """A compact human-readable Markdown summary of the report (S10 AC).

        The machine-readable half is :meth:`to_json`; this is the human-readable half —
        the same numbers, laid out so an operator or an adopter reads the run's health at a
        glance without parsing JSON. Deterministic (every section is rendered in a fixed
        order and every map is already sorted), so a seeded run renders byte-identically.
        """
        lines: list[str] = []
        lines.append(f"# Quality Report — {self.pipeline_name} ({self.run_id})")
        lines.append("")
        lines.append(f"- documents: {self.document_count}")
        lines.append(f"- duration: {self.total_duration_seconds:.3f}s")
        # Render off the completeness detail so the headline can never disagree with the
        # breakdown printed under it (the stored ``contracts_complete`` flag mirrors it).
        status = "PASS" if self.completeness.passed else "FAIL"
        lines.append(f"- contracts complete: {status}")
        c = self.completeness
        lines.append(
            f"  - provenance {c.records_with_provenance}/{c.records_checked}, "
            f"classification {c.records_with_classification}/{c.records_checked}, "
            f"offsets aligned {c.records_with_aligned_offsets}/{c.offset_bearing_records}"
        )
        if self.drift is not None:
            d = self.drift
            drift_status = "clean" if d.clean else f"{d.recovered} recovered"
            lines.append(
                f"- provenance drift: {drift_status} "
                f"({d.resolutions} lookups, max {d.max_drift_chars} chars)"
            )
        lines.append("")
        lines.append("## Stages")
        for m in self.stages:
            resumed = " (from checkpoint)" if m.from_checkpoint else ""
            # #73: surface a Stage's typed-error category inline when one was recorded, so an
            # operator reads the failure kind (contract/config/provider/…) at a glance.
            errored = f" ERROR[{m.error_category}]" if m.error_count else ""
            lines.append(
                f"- {m.name} [{m.capability}/{m.provider}]: "
                f"{m.records_in} in -> {m.records_out} out, "
                f"{m.duration_seconds:.3f}s{resumed}{errored}"
            )
        # Derived from the model's own SECTION-marked fields (declaration order), so a new section
        # appears here without editing render() — see :data:`SECTION`.
        present = [
            display for display, name in _section_fields() if getattr(self, name) is not None
        ]
        if present:
            lines.append("")
            lines.append(f"## Stage metrics present: {', '.join(present)}")
        lines.extend(_render_induction_cost(self.stages))
        if self.type_consolidation is not None:
            # The T3 operator-facing section: the headline collapse, then the full alias map so a
            # human SEES every rewrite and can correct it in config. Rendered before the alias
            # lines so a large map never buries the number that matters.
            tc = self.type_consolidation
            lines.append("")
            lines.append("## Type consolidation")
            lines.append(
                f"- entity types: {tc.raw_entity_types} raw -> "
                f"{tc.canonical_entity_types} canonical "
                f"({tc.mentions_remapped} mentions remapped)"
            )
            lines.append(
                f"- relation types: {tc.raw_relation_types} raw -> "
                f"{tc.canonical_relation_types} canonical "
                f"({tc.relations_remapped} relations remapped)"
            )
            for title, amap in (
                ("entity aliases", tc.entity_alias_map),
                ("relation aliases", tc.relation_alias_map),
            ):
                if not amap:
                    continue
                lines.append(f"- {title}:")
                lines.extend(f"  - {alias} -> {canonical}" for alias, canonical in amap.items())
        if self.goldset is not None:
            g = self.goldset
            lines.append("")
            lines.append("## Gold-set evaluation")
            lines.append(
                f"- documents: {g.documents_matched}/{g.documents_annotated} matched "
                f"(min overlap {g.match_min_overlap})"
            )
            for score in g.stages:
                lines.append(
                    f"- {score.stage}: P={score.precision:.3f} R={score.recall:.3f} "
                    f"F1={score.f1:.3f} "
                    f"(tp={score.true_positives}, fp={score.false_positives}, "
                    f"fn={score.false_negatives})"
                )
                for label, ls in score.per_label.items():
                    lines.append(
                        f"  - {label}: P={ls.precision:.3f} R={ls.recall:.3f} F1={ls.f1:.3f}"
                    )
        if self.delta is not None:
            dc = self.delta
            recon = "reconciled" if dc.reconciliation_triggered else "incremental"
            frm = "none" if dc.from_version is None else str(dc.from_version)
            lines.append("")
            lines.append(f"## Delta churn (v{frm} -> v{dc.to_version}, {recon})")
            lines.append(
                f"- documents: +{dc.documents_added} ~{dc.documents_updated} "
                f"-{dc.documents_deleted} (={dc.documents_unchanged} unchanged)"
            )
            lines.append(
                f"- entities: +{dc.entities_created} merged {dc.entities_merged} "
                f"split {dc.entities_split}"
            )
            lines.append(f"- edges: +{dc.edges_added} -{dc.edges_retracted}")
            lines.append(
                f"- records: {dc.records_retracted} retracted, {dc.records_purged} purged "
                f"({dc.deletion_mode.value})"
            )
            lines.append(
                f"- affected set: {dc.affected_set_size}/{dc.corpus_record_count} "
                "corpus-level records recomputed"
            )
            lines.append(
                f"- extraction: {dc.documents_extracted} extracted, "
                f"{dc.documents_reused} reused (doc-level Stages skipped)"
            )
            # T4: the canonical type vocabulary the delta EXTENDED. Rendered only when the corpus
            # has one, so a pipeline without a Type-Consolidation Stage reads exactly as before.
            if dc.canonical_types_total:
                lines.append(
                    f"- canonical types: +{dc.canonical_types_added} new, "
                    f"+{dc.type_aliases_added} aliased into existing "
                    f"({dc.canonical_types_total} total)"
                )
            lines.append(f"- drift: {dc.drift:.3f}")
        return "\n".join(lines) + "\n"
