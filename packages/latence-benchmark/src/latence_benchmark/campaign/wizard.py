# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The pure ``campaign.yaml`` generator behind ``latence-campaign init``.

Mirrors the ``latence setup`` contract (W15/ADR-0041 house style): interactive prompts and
``--non-interactive`` flags both resolve into ONE frozen :class:`CampaignAnswers` struct,
contradictions are rejected in ``__post_init__``, and :func:`build_campaign_config` is a PURE
function of that struct — it never reads the environment, a clock, or a random source, so
identical answers yield a **byte-identical** file. The comment header is derived from the
assembled config, never hand-maintained prose about it.

Extensibility is a design constraint, not an afterthought: the emitted YAML is a mapping of
per-concern blocks (``campaign``, ``corpus``; later stages append ``ocr``, ``pipeline``,
``eval`` …), and :func:`parse_campaign_config` validates the blocks it knows while accepting
unknown blocks untouched — so an S2-era config file still round-trips through S1-era code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

import yaml

_T = TypeVar("_T")

#: The 6+2 corpus datasets S0 recommended (dataset-landscape.md), in campaign order:
#: the incumbents first, then the multi-hop triad's third leg, then the new acquisition.
KNOWN_DATASETS: tuple[str, ...] = (
    "musique",
    "hotpotqa",
    "wiki2multihop",
    "multihop_rag",
    "ohr_bench",
    "vidoseek",
    "uda",
)


class CampaignConfigError(ValueError):
    """A campaign config (answers or YAML text) that cannot be used."""


#: The hand-authored wiki/news entity schema for the text-native datasets (S3 spec §2) — and
#: the config FLOOR everywhere induction is off. Kept exactly as ruled in the S3 ticket.
S3_ENTITY_LABELS: tuple[str, ...] = (
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "WORK",
    "EVENT",
    "DATE",
    "PRODUCT",
)

#: The matching hand-authored relation schema (S3 spec §2).
S3_RELATION_LABELS: tuple[str, ...] = (
    "located_in",
    "part_of",
    "member_of",
    "created_by",
    "founded_by",
    "spouse_of",
    "directed_by",
    "works_for",
    "released_in",
)

#: The three legal induction modes for the OCR'd dataset group. ``per_chunk`` is the blessed
#: default (label_inducer.llm, granularity chunk); ``doc`` induces once per document;
#: ``none`` turns induction off entirely — every dataset then extracts with the hand-authored
#: label sets above.
S3_INDUCTION_MODES: tuple[str, ...] = ("per_chunk", "doc", "none")


@dataclass(frozen=True)
class S3Answers:
    """The ``s3:`` block of campaign.yaml — everything the S3 pipeline generator reads.

    All defaults are the S3 spec's rulings (chunk {640,80,8} = the blessed gpu-sota operating
    point under the GLiNER2 768 mdeberta window; extraction fused_entity_relation.gliner2 @
    0.5; induction per_chunk on deepseek-v4-flash via OpenRouter — reasoning pinned OFF —
    under a hard USD cap; learned redaction with guardrails ON). Contradictions are rejected
    here, not discovered on the pod.
    """

    chunk_max_tokens: int = 640
    chunk_overlap_tokens: int = 80
    chunk_min_tokens: int = 8
    extraction_provider: str = "fused_entity_relation.gliner2"
    extraction_model: str = "fastino/gliner2-multi-v1"
    extraction_threshold: float = 0.5
    induction_mode: str = "per_chunk"
    # Dennis's ruling (2026-08 live A/B, 10 real ohr_bench chunks each arm): deepseek-v4-flash
    # with reasoning left ON burns its whole completion budget on reasoning tokens (7/10
    # invalid schemas, no cost advantage); with reasoning DISABLED it returns 9/10 valid
    # schemas at ~$0.000118/call — ~4x cheaper than gpt-4.1-mini's measured $0.00042/call.
    # Hence the default pair: deepseek + induction_reasoning False. None omits the knob
    # entirely (the provider then sends no extra_body — required for a strict vLLM endpoint).
    induction_model: str = "deepseek/deepseek-v4-flash-0731"
    induction_reasoning: bool | None = False
    # In-flight LLM-call fan-out for the inducer (its asyncio.Semaphore). The provider default
    # is a conservative 8; at the S3 corpus scale (UDA alone is 242k calls) 8 means ~21h of
    # induction wall-time, so the campaign default is 32 (~4x). OpenRouter absorbs it; a 429
    # burst is classed + retried by the provider's endpoint-error path.
    induction_max_concurrency: int = 32
    induction_base_url: str = "https://openrouter.ai/api/v1"
    induction_budget_usd_cap: float = 150.0
    entity_labels: tuple[str, ...] = field(default=S3_ENTITY_LABELS)
    relation_labels: tuple[str, ...] = field(default=S3_RELATION_LABELS)
    redaction_provider: str = "redaction.gliner2"
    redaction_guardrails: bool = True

    def __post_init__(self) -> None:
        if self.chunk_max_tokens <= 0:
            msg = f"s3.chunk.max_tokens must be positive (got {self.chunk_max_tokens})."
            raise CampaignConfigError(msg)
        if not 0 <= self.chunk_overlap_tokens < self.chunk_max_tokens:
            msg = (
                f"s3.chunk.overlap_tokens ({self.chunk_overlap_tokens}) must be >= 0 and "
                f"smaller than max_tokens ({self.chunk_max_tokens}) — an overlap that swallows "
                "the whole chunk never advances."
            )
            raise CampaignConfigError(msg)
        if not 0 < self.chunk_min_tokens <= self.chunk_max_tokens:
            msg = (
                f"s3.chunk.min_tokens ({self.chunk_min_tokens}) must be positive and "
                f"<= max_tokens ({self.chunk_max_tokens})."
            )
            raise CampaignConfigError(msg)
        if not 0.0 < self.extraction_threshold <= 1.0:
            msg = (
                f"s3.extraction.threshold must be in (0, 1] (got {self.extraction_threshold})."
            )
            raise CampaignConfigError(msg)
        if self.induction_mode not in S3_INDUCTION_MODES:
            msg = (
                f"s3.induction.mode must be one of {', '.join(S3_INDUCTION_MODES)} "
                f"(got {self.induction_mode!r})."
            )
            raise CampaignConfigError(msg)
        if self.induction_budget_usd_cap < 0:
            msg = (
                "s3.induction.budget_usd_cap must be >= 0 "
                f"(got {self.induction_budget_usd_cap})."
            )
            raise CampaignConfigError(msg)
        if self.induction_max_concurrency < 1:
            msg = (
                "s3.induction.max_concurrency must be >= 1 "
                f"(got {self.induction_max_concurrency})."
            )
            raise CampaignConfigError(msg)
        if self.induction_mode != "none" and self.induction_budget_usd_cap == 0:
            msg = (
                f"s3: induction mode {self.induction_mode!r} with budget_usd_cap 0 is a "
                "contradiction — zero budget means zero calls. Set a positive cap or "
                "mode 'none'."
            )
            raise CampaignConfigError(msg)
        if not self.entity_labels or not all(s.strip() for s in self.entity_labels):
            msg = "s3.labels.entity must be a non-empty list of non-blank labels."
            raise CampaignConfigError(msg)
        if not self.relation_labels or not all(s.strip() for s in self.relation_labels):
            msg = "s3.labels.relation must be a non-empty list of non-blank labels."
            raise CampaignConfigError(msg)
        for name, value in (
            ("extraction.provider", self.extraction_provider),
            ("extraction.model", self.extraction_model),
            ("induction.model", self.induction_model),
            ("induction.base_url", self.induction_base_url),
            ("redaction.provider", self.redaction_provider),
        ):
            if not value.strip():
                msg = f"s3.{name} must be a non-empty string."
                raise CampaignConfigError(msg)


@dataclass(frozen=True)
class CampaignAnswers:
    """The fully-resolved wizard answers — the SAME struct from prompts and from flags.

    :param campaign_root: Directory that will hold ``campaign.yaml``, ``STATE.json`` and the
        per-stage artifact dirs.
    :param datasets: The corpus selection — a subset of :data:`KNOWN_DATASETS`, kept in the
        operator's order (deduplication would be a silent edit; duplicates are rejected).
    :param pod_ssh: SSH target of the GPU pod later stages run on (``user@host[:port]`` or an
        ssh-config alias), or ``""`` for the local-only stages. S1 only records it; the first
        consumer is the S2 OCR stage.
    :param s3: The S3 pipeline-generation block (chunking, extraction, induction, labels,
        redaction). Older configs without an ``s3:`` block parse to these defaults, so an
        S1/S2-era campaign.yaml keeps loading unchanged.
    """

    campaign_root: str
    datasets: tuple[str, ...] = field(default=KNOWN_DATASETS)
    pod_ssh: str = ""
    s3: S3Answers = field(default_factory=S3Answers)

    def __post_init__(self) -> None:
        if not self.campaign_root.strip():
            msg = "campaign_root must be a non-empty path."
            raise CampaignConfigError(msg)
        if not self.datasets:
            msg = (
                "The corpus selection is empty — a campaign with nothing to acquire has no "
                f"first stage. Choose from: {', '.join(KNOWN_DATASETS)}."
            )
            raise CampaignConfigError(msg)
        unknown = [d for d in self.datasets if d not in KNOWN_DATASETS]
        if unknown:
            msg = (
                f"Unknown dataset(s): {', '.join(unknown)}. "
                f"Known: {', '.join(KNOWN_DATASETS)}."
            )
            raise CampaignConfigError(msg)
        seen: set[str] = set()
        dupes: list[str] = []
        for d in self.datasets:
            if d in seen:
                dupes.append(d)
            seen.add(d)
        if dupes:
            msg = f"Dataset(s) selected more than once: {', '.join(sorted(set(dupes)))}."
            raise CampaignConfigError(msg)


def build_campaign_config(answers: CampaignAnswers) -> str:
    """Render ``campaign.yaml`` — pure, byte-deterministic, header derived from the config.

    The header is assembled FROM the same struct the body is, so the two cannot drift; it
    carries the why (what the file is, how gates work, the one next command), the body stays
    machine-clean YAML.
    """
    s3 = answers.s3
    body: str = yaml.safe_dump(
        {
            "campaign": {
                "root": answers.campaign_root,
                "pod_ssh": answers.pod_ssh,
            },
            "corpus": {
                "datasets": list(answers.datasets),
            },
            "s3": {
                "chunk": {
                    "max_tokens": s3.chunk_max_tokens,
                    "overlap_tokens": s3.chunk_overlap_tokens,
                    "min_tokens": s3.chunk_min_tokens,
                },
                "extraction": {
                    "provider": s3.extraction_provider,
                    "model": s3.extraction_model,
                    "threshold": s3.extraction_threshold,
                },
                "induction": {
                    "mode": s3.induction_mode,
                    "model": s3.induction_model,
                    "reasoning": s3.induction_reasoning,
                    "base_url": s3.induction_base_url,
                    "budget_usd_cap": s3.induction_budget_usd_cap,
                    "max_concurrency": s3.induction_max_concurrency,
                },
                "labels": {
                    "entity": list(s3.entity_labels),
                    "relation": list(s3.relation_labels),
                },
                "redaction": {
                    "provider": s3.redaction_provider,
                    "guardrails": s3.redaction_guardrails,
                },
            },
        },
        sort_keys=True,
        default_flow_style=False,
    )
    pod = answers.pod_ssh if answers.pod_ssh else "none (local-only stages until one is set)"
    reasoning = (
        "" if s3.induction_reasoning is None
        else f", reasoning {'on' if s3.induction_reasoning else 'off'}"
    )
    induction = (
        "none (hand-authored labels everywhere)"
        if s3.induction_mode == "none"
        else f"{s3.induction_mode} ({s3.induction_model}{reasoning}, "
        f"cap ${s3.induction_budget_usd_cap:.2f})"
    )
    header_lines = [
        "# latence-campaign configuration — generated by `latence-campaign init`.",
        "# Deterministic: identical answers produce a byte-identical file, so re-running init",
        "# with the same flags reproduces it exactly. Hand-edits are fine; every subcommand",
        "# re-validates the file before using it.",
        "#",
        f"# campaign root : {answers.campaign_root}",
        "#                 (STATE.json + stages/<name>/ metrics + artifacts live here)",
        f"# corpus        : {len(answers.datasets)} dataset(s) — {', '.join(answers.datasets)}",
        "#                 (license lanes are classified and recorded by the s1-corpus stage",
        "#                 gate before any byte is fetched; see stages/s1-corpus/metrics.json)",
        f"# pod ssh       : {pod}",
        f"# s3 pipeline   : chunk {s3.chunk_max_tokens}/{s3.chunk_overlap_tokens}/"
        f"{s3.chunk_min_tokens}; {s3.extraction_provider} @ {s3.extraction_threshold}",
        f"#                 induction {induction}; redaction "
        f"{'guardrails on' if s3.redaction_guardrails else 'guardrails OFF'}",
        "#",
        "# Stages are gated: `run <stage>` refuses until the PREVIOUS stage was `approve`d.",
        "# Later campaign stages append their own config blocks below; unknown blocks are",
        "# preserved by older code, never rejected.",
    ]
    return "\n".join(header_lines) + "\n" + body


def parse_campaign_config(text: str) -> CampaignAnswers:
    """Parse + validate ``campaign.yaml`` text back into :class:`CampaignAnswers`.

    This is also the DRY-VALIDATION step ``init`` runs on its own output before writing —
    generate, re-parse, only then save — so a generator bug can never persist a config the
    rest of the CLI refuses to load. Unknown top-level blocks are accepted (extensibility);
    the known blocks are validated strictly.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        msg = f"campaign.yaml is not valid YAML: {exc}"
        raise CampaignConfigError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"campaign.yaml must hold a mapping, got {type(raw).__name__}."
        raise CampaignConfigError(msg)

    campaign = raw.get("campaign")
    if not isinstance(campaign, dict):
        msg = "campaign.yaml is missing its `campaign:` block (root, pod_ssh)."
        raise CampaignConfigError(msg)
    root = campaign.get("root")
    if not isinstance(root, str):
        msg = "`campaign.root` must be a string path."
        raise CampaignConfigError(msg)
    pod_ssh = campaign.get("pod_ssh", "")
    if not isinstance(pod_ssh, str):
        msg = "`campaign.pod_ssh` must be a string (or empty)."
        raise CampaignConfigError(msg)

    corpus = raw.get("corpus")
    if not isinstance(corpus, dict):
        msg = "campaign.yaml is missing its `corpus:` block (datasets)."
        raise CampaignConfigError(msg)
    datasets = corpus.get("datasets")
    if not isinstance(datasets, list) or not all(isinstance(d, str) for d in datasets):
        msg = "`corpus.datasets` must be a list of dataset names."
        raise CampaignConfigError(msg)

    return CampaignAnswers(
        campaign_root=root,
        datasets=tuple(datasets),
        pod_ssh=pod_ssh,
        s3=_parse_s3_block(raw.get("s3")),
    )


def _sub_block(s3: dict[str, object], name: str, known: tuple[str, ...]) -> dict[str, object]:
    """One ``s3.<name>`` sub-mapping, absent = ``{}``, unknown keys LOUD (a typo'd knob that
    silently falls back to its default would run a 2-pod-day campaign on the wrong setting)."""
    raw = s3.get(name, {})
    if not isinstance(raw, dict):
        msg = f"`s3.{name}` must be a mapping."
        raise CampaignConfigError(msg)
    unknown = sorted(set(raw) - set(known))
    if unknown:
        msg = f"`s3.{name}` has unknown key(s): {', '.join(unknown)}. Known: {', '.join(known)}."
        raise CampaignConfigError(msg)
    return dict(raw)


def _typed(block: dict[str, object], key: str, kind: type[_T], default: _T) -> _T:
    value = block.get(key, default)
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)  # YAML `150` for a cap is not a typo; accept the int spelling.
    if not isinstance(value, kind) or (kind is not bool and isinstance(value, bool)):
        msg = f"`s3` key {key!r} must be a {kind.__name__} (got {value!r})."
        raise CampaignConfigError(msg)
    return value


def _reasoning_knob(block: dict[str, object], default: bool | None) -> bool | None:
    """``s3.induction.reasoning``: a tri-state bool — True/False, or an explicit ``null``.

    ``null`` (YAML) ⇒ ``None`` ⇒ the pipeline omits the knob entirely, so the inducer sends
    no ``extra_body`` at all (required for a strict vLLM endpoint that rejects unknown body
    fields). An absent key falls back to the default (False — Dennis's deepseek ruling).
    """
    value = block.get("reasoning", default)
    if value is not None and not isinstance(value, bool):
        msg = f"`s3.induction.reasoning` must be true, false or null (got {value!r})."
        raise CampaignConfigError(msg)
    return value


def _label_list(block: dict[str, object], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = block.get(key)
    if raw is None:
        return default
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        msg = f"`s3.labels.{key}` must be a list of strings."
        raise CampaignConfigError(msg)
    return tuple(raw)


def _parse_s3_block(raw: object) -> S3Answers:
    """The optional ``s3:`` block → :class:`S3Answers`; absent = all defaults (older configs)."""
    if raw is None:
        return S3Answers()
    if not isinstance(raw, dict):
        msg = "`s3` must be a mapping (chunk, extraction, induction, labels, redaction)."
        raise CampaignConfigError(msg)
    unknown = sorted(set(raw) - {"chunk", "extraction", "induction", "labels", "redaction"})
    if unknown:
        msg = f"`s3` has unknown key(s): {', '.join(unknown)}."
        raise CampaignConfigError(msg)
    chunk = _sub_block(raw, "chunk", ("max_tokens", "overlap_tokens", "min_tokens"))
    extraction = _sub_block(raw, "extraction", ("provider", "model", "threshold"))
    induction = _sub_block(
        raw,
        "induction",
        ("mode", "model", "reasoning", "base_url", "budget_usd_cap", "max_concurrency"),
    )
    labels = _sub_block(raw, "labels", ("entity", "relation"))
    redaction = _sub_block(raw, "redaction", ("provider", "guardrails"))
    defaults = S3Answers()
    return S3Answers(
        chunk_max_tokens=int(_typed(chunk, "max_tokens", int, defaults.chunk_max_tokens)),
        chunk_overlap_tokens=int(
            _typed(chunk, "overlap_tokens", int, defaults.chunk_overlap_tokens)
        ),
        chunk_min_tokens=int(_typed(chunk, "min_tokens", int, defaults.chunk_min_tokens)),
        extraction_provider=str(
            _typed(extraction, "provider", str, defaults.extraction_provider)
        ),
        extraction_model=str(_typed(extraction, "model", str, defaults.extraction_model)),
        extraction_threshold=float(
            _typed(extraction, "threshold", float, defaults.extraction_threshold)
        ),
        induction_mode=str(_typed(induction, "mode", str, defaults.induction_mode)),
        induction_model=str(_typed(induction, "model", str, defaults.induction_model)),
        induction_reasoning=_reasoning_knob(induction, defaults.induction_reasoning),
        induction_base_url=str(
            _typed(induction, "base_url", str, defaults.induction_base_url)
        ),
        induction_budget_usd_cap=float(
            _typed(induction, "budget_usd_cap", float, defaults.induction_budget_usd_cap)
        ),
        induction_max_concurrency=int(
            _typed(induction, "max_concurrency", int, defaults.induction_max_concurrency)
        ),
        entity_labels=_label_list(labels, "entity", defaults.entity_labels),
        relation_labels=_label_list(labels, "relation", defaults.relation_labels),
        redaction_provider=str(
            _typed(redaction, "provider", str, defaults.redaction_provider)
        ),
        redaction_guardrails=bool(
            _typed(redaction, "guardrails", bool, defaults.redaction_guardrails)
        ),
    )
