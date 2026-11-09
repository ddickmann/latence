"""The generated Stage-contract discovery reference — one authoritative, agent-facing document.

A user replaces a component (the LightOn OCR parser with their own parser, GLiNER2 with their
own extractor) by handing an LLM coding agent a document that fully acknowledges the input and
output API of every Stage, such that a replacement cannot violate the end-to-end flow. Writing
that document by hand would recreate exactly the drift :mod:`latence_core.capability_descriptor`
was built to kill — a ninth fact added to a descriptor, a new Capability, a changed carrier
field, and the hand-written contract doc silently lies. So the document is **generated**: this
module walks the descriptor table (the single source of truth), the Capability Protocols (the
behavioural API, via :mod:`inspect`), and the record carriers (the data API, via the Pydantic
field metadata), and renders one reference in two forms — machine-readable JSON
(``docs/reference/stage-contracts.json``) and human-readable Markdown
(``docs/reference/stage-contracts.md``). A drift gate
(``packages/latence-core/tests/test_contract_reference.py``) asserts the committed artifacts are
byte-identical to a fresh render, so the reference inherits the descriptor module's
cannot-go-stale guarantee.

Determinism is part of the contract: iteration follows the descriptor declaration order (the
pipeline's reading order) with Provider listings sorted by name, and the render carries no
timestamp — the byte-identity gate depends on it.

Scope (ADR-0016): this module imports only what ``capability_descriptor`` already imports plus
the stdlib. It never instantiates a Provider and never touches torch; reading the reference runs
no Provider work. The conformance-case cross-check (each cited case must exist in
:mod:`latence_core.conformance.cases`) lives in the drift-gate test, not here, because importing
the case registry pulls the reference-Provider fixtures in.
"""

from __future__ import annotations

import inspect
import json
import re
import tomllib
import types
import typing
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from latence_core.capability import (
    ENTRY_POINT_GROUP,
    DeltaOutcome,
    DeltaProcessor,
    DocumentEnumerator,
    FusedExtraction,
    ScreeningOutcome,
    TypeConsolidation,
    protocol_members,
)
from latence_core.capability_descriptor import (
    CapabilityDescriptor,
    CapabilityLevel,
    StageInputShape,
    descriptors,
    export_carrier_of,
    kind_for_entry_point,
)
from latence_core.contracts import Classification, Provenance, Record
from latence_core.errors import ContractError
from latence_core.pipeline import CapabilityKind

#: Bump on any breaking change to the reference's JSON shape (a consumer keys off this).
SCHEMA_VERSION = 1

#: The one command that proves a replacement Provider (C1–C6, auto-covered by name prefix).
CONFORMANCE_COMMAND = (
    "LATENCE_CUDA=0 PYTHONHASHSEED=0 uv run pytest "
    "packages/latence-core/tests/test_conformance.py "
    "packages/latence-core/tests/test_e2e_conformance.py"
)

#: The command that regenerates the committed artifacts (quoted in the doc + the gate's message).
REGENERATE_COMMAND = "PYTHONHASHSEED=0 .venv/bin/latence contracts --write --repo-root ."


@dataclass(frozen=True, slots=True)
class SignalSeam:
    """One non-Stage Capability seam under the same ``latence.providers`` entry-point group.

    An engine seam is a Capability a
    Provider fulfils and conformance covers, but it is not a DAG Stage, so it has no
    :class:`~latence_core.pipeline.CapabilityKind` and no descriptor. Its conformance case
    equals their prefix (the ``_NON_STAGE_PREFIX_TO_CAPABILITY`` convention in
    ``conformance/cases.py`` — the drift gate asserts the two agree). ``refinements`` are the
    optional companion Protocols a Provider of this seam MAY additionally satisfy.
    """

    prefix: str
    protocol: type
    refinements: tuple[type, ...] = ()


#: The non-Stage seams, in the fixed documentation order. Every entry-point name in the workspace
#: must classify to either a descriptor prefix or one of these (the drift gate enforces it).
SIGNAL_SEAMS: tuple[SignalSeam, ...] = (SignalSeam(prefix="delta", protocol=DeltaProcessor),)

#: Optional companion Protocols per Stage Capability (a Provider MAY also satisfy these).
_STAGE_REFINEMENTS: dict[CapabilityKind, tuple[type, ...]] = {
    CapabilityKind.SOURCE: (DocumentEnumerator,),
}

#: The frozen result dataclasses several Protocols return — part of the output API, documented
#: beside the Protocols that produce them (their docstrings name their producers).
_RESULT_TYPES: tuple[type, ...] = (
    ScreeningOutcome,
    FusedExtraction,
    TypeConsolidation,
    DeltaOutcome,
)

_RST_ROLE = re.compile(r":[A-Za-z]+(?::[A-Za-z]+)?:`([^`]+)`")


def _clean_doc(doc: str | None) -> str:
    """Turn a Sphinx/RST docstring into plain Markdown-safe prose, deterministically.

    Cross-reference roles render as inline code (``:class:`~a.b.C``` → ```C```, the Sphinx
    tilde display rule; an untilded target keeps its dotted path), and RST double-backtick
    literals become Markdown single-backtick code. Nothing else is rewritten — the docstring
    text IS the contract prose and must arrive verbatim.
    """
    if not doc:
        return ""

    def _role(match: re.Match[str]) -> str:
        target = match.group(1)
        if target.startswith("~"):
            target = target[1:].rsplit(".", 1)[-1]
        return f"`{target}`"

    text = _RST_ROLE.sub(_role, doc)
    return text.replace("``", "`")


def _first_paragraph(doc: str) -> str:
    """The first paragraph of a cleaned docstring — the one-line-or-so summary."""
    return doc.split("\n\n", 1)[0].strip()


def _type_repr(annotation: Any) -> str:
    """A readable, deterministic rendering of a (Pydantic-resolved) type annotation."""
    if annotation is None or annotation is type(None):
        return "None"
    origin = typing.get_origin(annotation)
    if origin is None:
        name = getattr(annotation, "__name__", None)
        return name if isinstance(name, str) else str(annotation)
    args = typing.get_args(annotation)
    if origin is types.UnionType or origin is typing.Union:
        return " | ".join(_type_repr(arg) for arg in args)
    if origin is typing.Literal:
        return f"Literal[{', '.join(repr(arg) for arg in args)}]"
    origin_name = getattr(origin, "__name__", None)
    rendered_origin = origin_name if isinstance(origin_name, str) else str(origin)
    return f"{rendered_origin}[{', '.join(_type_repr(arg) for arg in args)}]"


def _member_entries(protocol: type) -> list[dict[str, Any]]:
    """The Protocol's required members — name, kind, signature and full contract docstring."""
    entries: list[dict[str, Any]] = []
    for name in sorted(protocol_members(protocol)):
        attr = inspect.getattr_static(protocol, name)
        if isinstance(attr, property):
            getter = attr.fget
            annotation = ""
            if getter is not None:
                raw = inspect.signature(getter).return_annotation
                if raw is not inspect.Signature.empty:
                    annotation = str(raw).replace("'", "")
            entries.append(
                {
                    "name": name,
                    "kind": "property",
                    "signature": f"{name}: {annotation}" if annotation else name,
                    "doc": _clean_doc(inspect.getdoc(attr)),
                }
            )
            continue
        signature = str(inspect.signature(getattr(protocol, name))).replace("'", "")
        entries.append(
            {
                "name": name,
                "kind": "method",
                "signature": f"{name}{signature}",
                "doc": _clean_doc(inspect.getdoc(getattr(protocol, name))),
            }
        )
    return entries


def _protocol_entry(protocol: type) -> dict[str, Any]:
    """One Protocol as data: its name, full contract docstring, and required members."""
    return {
        "name": protocol.__name__,
        "doc": _clean_doc(inspect.getdoc(protocol)),
        "members": _member_entries(protocol),
    }


def _carrier_entry(model: type[BaseModel]) -> dict[str, Any]:
    """One record carrier as data: its docstring plus every field with type + description.

    ``model_rebuild()`` first, always. A carrier may reference a type defined later in its own
    module (``DocumentRecord.induced_labels`` names ``InducedLabels``, defined 120 lines below
    it), and Pydantic leaves such an annotation as a ``ForwardRef`` until something happens to
    trigger resolution. Without the explicit rebuild the render depends on which modules the
    process imported first — the committed artifact once carried
    ``ForwardRef('InducedLabels | None')`` and went stale the moment an unrelated test warmed
    the process. The byte-identity gate exists to catch exactly that; this line is what makes
    the render a function of the source instead of of import order.
    """
    model.model_rebuild()
    field_entries: list[dict[str, Any]] = []
    for field_name, info in model.model_fields.items():
        field_entries.append(
            {
                "name": field_name,
                "type": _type_repr(info.annotation),
                "required": info.is_required(),
                "description": info.description,
            }
        )
    return {
        "name": model.__name__,
        "doc": _clean_doc(inspect.getdoc(model)),
        "fields": field_entries,
    }


def _result_type_entry(result_type: type) -> dict[str, Any]:
    """One frozen result dataclass as data — the non-Record half of the output API."""
    return {
        "name": result_type.__name__,
        "doc": _clean_doc(inspect.getdoc(result_type)),
        "fields": [
            {"name": f.name, "type": str(f.type)} for f in dataclass_fields(result_type)
        ],
    }


def _input_plan_entry(descriptor: CapabilityDescriptor) -> dict[str, Any]:
    """The Stage's input plan as data: shape, wiring rules with their exact typed messages."""
    plan = descriptor.input_plan
    exclusive: dict[str, Any] | None = None
    if plan.exclusive is not None:
        exclusive = {
            "carrier": plan.exclusive.carrier.__name__,
            "error": plan.exclusive.message,
        }
    return {
        "shape": plan.shape.value,
        "requires": [
            {
                "carrier": requirement.carrier.__name__,
                "subclass_ok": requirement.subclass_ok,
                "error": requirement.message,
            }
            for requirement in plan.requires
        ],
        "exclusive": exclusive,
    }


def classify_entry_point(provider_name: str) -> CapabilityKind | str:
    """The Capability an entry-point name belongs to: a Stage kind, or a signal-seam prefix.

    Raises :class:`~latence_core.errors.ContractError` for a name that classifies to neither —
    an unclassifiable Provider must fail the reference loudly (the reference cannot silently
    omit a seam), exactly as ``kind_for_entry_point`` and the conformance suite refuse it.
    """
    prefix = provider_name.partition(".")[0]
    for seam in SIGNAL_SEAMS:
        if prefix == seam.prefix:
            return seam.prefix
    try:
        return kind_for_entry_point(provider_name)
    except KeyError as exc:
        msg = (
            f"entry point {provider_name!r} matches no Stage Capability prefix and no signal "
            f"seam — the stage-contract reference refuses to render an undocumented seam: {exc}"
        )
        raise ContractError(msg) from exc


def workspace_provider_packages(repo_root: Path) -> dict[str, str]:
    """Every ``latence.providers`` entry point declared in ``packages/*/pyproject.toml``.

    Maps entry-point name → declaring distribution name, sorted by name. This is the collector
    the committed reference uses (the workspace declaration is the truth CI sees, installed or
    not); :func:`installed_provider_packages` is the out-of-repo fallback.
    """
    out: dict[str, str] = {}
    for toml_path in sorted((repo_root / "packages").glob("*/pyproject.toml")):
        with toml_path.open("rb") as handle:
            data = tomllib.load(handle)
        project = data.get("project", {})
        entry_points = project.get("entry-points", {}).get(ENTRY_POINT_GROUP, {})
        for name in entry_points:
            out[name] = str(project.get("name", toml_path.parent.name))
    return dict(sorted(out.items()))


def installed_provider_packages() -> dict[str, str]:
    """Every installed ``latence.providers`` entry point → its distribution name, sorted."""
    from importlib import metadata

    out: dict[str, str] = {}
    for entry_point in metadata.entry_points(group=ENTRY_POINT_GROUP):
        dist = entry_point.dist
        out[entry_point.name] = dist.name if dist is not None else "<unknown distribution>"
    return dict(sorted(out.items()))


def build_reference(provider_packages: dict[str, str]) -> dict[str, Any]:
    """Assemble the whole reference as one deterministic dict (the JSON document).

    ``provider_packages`` maps entry-point name → distribution name (from
    :func:`workspace_provider_packages` or :func:`installed_provider_packages`). Every name is
    classified to its Capability; an unclassifiable name raises
    :class:`~latence_core.errors.ContractError` rather than being silently dropped.
    """
    by_capability: dict[CapabilityKind | str, list[dict[str, str]]] = {}
    for name in sorted(provider_packages):
        key = classify_entry_point(name)
        by_capability.setdefault(key, []).append(
            {"name": name, "package": provider_packages[name]}
        )

    capability_entries: list[dict[str, Any]] = []
    for descriptor in descriptors():
        export_carrier = export_carrier_of(descriptor.kind)
        capability_entries.append(
            {
                "kind": descriptor.kind.value,
                "protocol": _protocol_entry(descriptor.protocol),
                "optional_refinements": [
                    _protocol_entry(refinement)
                    for refinement in _STAGE_REFINEMENTS.get(descriptor.kind, ())
                ],
                "entry_point": {
                    "group": ENTRY_POINT_GROUP,
                    "prefixes": list(descriptor.entry_point_prefixes),
                },
                "level": descriptor.level.value,
                "corpus_when_config": (
                    list(descriptor.corpus_when_config)
                    if descriptor.corpus_when_config is not None
                    else None
                ),
                "dag_node": descriptor.dag_node,
                "accumulates": descriptor.accumulates,
                "input_plan": _input_plan_entry(descriptor),
                "checkpoint_carrier": (
                    descriptor.carrier.__name__ if descriptor.carrier is not None else None
                ),
                "mixed_carrier": descriptor.mixed_carrier,
                "export_carrier": (
                    export_carrier.__name__ if export_carrier is not None else None
                ),
                "quality_section": descriptor.quality_section,
                "conformance_case": descriptor.conformance_case,
                "providers": by_capability.get(descriptor.kind, []),
            }
        )

    seam_entries: list[dict[str, Any]] = []
    for seam in SIGNAL_SEAMS:
        seam_entries.append(
            {
                "prefix": seam.prefix,
                "protocol": _protocol_entry(seam.protocol),
                "optional_refinements": [
                    _protocol_entry(refinement) for refinement in seam.refinements
                ],
                "entry_point": {"group": ENTRY_POINT_GROUP, "prefixes": [seam.prefix]},
                "conformance_case": seam.prefix,
                "providers": by_capability.get(seam.prefix, []),
            }
        )

    carrier_models: list[type[BaseModel]] = [Record, Provenance, Classification]
    for descriptor in descriptors():
        carrier = descriptor.carrier
        if (
            carrier is not None
            and issubclass(carrier, BaseModel)
            and carrier not in carrier_models
        ):
            carrier_models.append(carrier)

    return {
        "schema": SCHEMA_VERSION,
        "generated_by": "latence_core.contract_reference",
        "regenerate": REGENERATE_COMMAND,
        "entry_point_group": ENTRY_POINT_GROUP,
        "conformance_command": CONFORMANCE_COMMAND,
        "legend": {
            "capability_descriptor": _clean_doc(inspect.getdoc(CapabilityDescriptor)),
            "capability_level": _clean_doc(inspect.getdoc(CapabilityLevel)),
            "stage_input_shape": _clean_doc(inspect.getdoc(StageInputShape)),
        },
        "capabilities": capability_entries,
        "signal_seams": seam_entries,
        "carriers": [_carrier_entry(model) for model in carrier_models],
        "result_types": [_result_type_entry(result) for result in _RESULT_TYPES],
    }


def render_json(reference: dict[str, Any]) -> str:
    """The machine-readable rendering — stable key order, no timestamp, trailing newline."""
    return json.dumps(reference, indent=2, ensure_ascii=False) + "\n"


def _table_cell(text: str | None) -> str:
    """Escape a value for a one-line Markdown table cell."""
    if not text:
        return ""
    return text.replace("|", "\\|").replace("\n", " ")


def _render_protocol(entry: dict[str, Any], heading_level: int) -> list[str]:
    """The Markdown block for one Protocol: contract docstring, then each member's contract."""
    marker = "#" * heading_level
    lines: list[str] = [f"{marker} Protocol `{entry['name']}`", ""]
    if entry["doc"]:
        lines += [entry["doc"], ""]
    for member in entry["members"]:
        lines += ["```python", member["signature"], "```", ""]
        if member["doc"]:
            lines += [member["doc"], ""]
    return lines


def _render_providers(providers: list[dict[str, str]]) -> list[str]:
    """The Markdown table of existing Providers behind one seam (or a none-yet note)."""
    if not providers:
        return ["No Provider is currently registered for this seam.", ""]
    lines = ["| Provider | Package |", "| --- | --- |"]
    lines += [f"| `{p['name']}` | `{p['package']}` |" for p in providers]
    lines.append("")
    return lines


def _render_capability(entry: dict[str, Any]) -> list[str]:
    """The Markdown section for one Stage Capability."""
    lines: list[str] = [f"### `{entry['kind']}` — {entry['protocol']['name']}", ""]
    prefixes = ", ".join(f"`{prefix}.<variant>`" for prefix in entry["entry_point"]["prefixes"])
    checkpoint = (
        f"`{entry['checkpoint_carrier']}`" if entry["checkpoint_carrier"] is not None else "none"
    )
    if entry["mixed_carrier"]:
        checkpoint += " (mixed: entity + relation mentions in one checkpoint, read polymorphically)"
    export = f"`{entry['export_carrier']}`" if entry["export_carrier"] is not None else "none"
    quality = (
        f"`{entry['quality_section']}`"
        if entry["quality_section"]
        else "per-Stage metrics only"
    )
    facts = [
        f"- **Register as**: {prefixes} under `{entry['entry_point']['group']}`",
        f"- **Level**: `{entry['level']}` — **DAG node**: `{str(entry['dag_node']).lower()}` — "
        f"**accumulates**: `{str(entry['accumulates']).lower()}`",
        f"- **Receives** (input shape): `{entry['input_plan']['shape']}`",
        f"- **Emits** (checkpoint carrier): {checkpoint}",
        f"- **An Export above it re-emits**: {export}",
        f"- **Quality Report section**: {quality}",
        f"- **Conformance case**: `{entry['conformance_case']}`",
    ]
    if entry["corpus_when_config"] is not None:
        key, value = entry["corpus_when_config"]
        facts.append(
            f"- **Config-dependent level**: `{key}: {value}` promotes a Stage of this kind to "
            f"corpus level"
        )
    lines += facts + [""]

    plan = entry["input_plan"]
    if plan["requires"] or plan["exclusive"] is not None:
        lines += ["**Wiring rules** (violations raise exactly these typed errors):", ""]
        for requirement in plan["requires"]:
            match = "subclass accepted" if requirement["subclass_ok"] else "exact carrier"
            lines += [
                f"- requires `{requirement['carrier']}` ({match}):",
                f"  > {_table_cell(requirement['error'])}",
            ]
        if plan["exclusive"] is not None:
            lines += [
                f"- input must be NOTHING BUT `{plan['exclusive']['carrier']}`:",
                f"  > {_table_cell(plan['exclusive']['error'])}",
            ]
        lines.append("")

    lines += _render_protocol(entry["protocol"], heading_level=4)
    for refinement in entry["optional_refinements"]:
        lines += [f"#### Optional refinement — `{refinement['name']}`", ""]
        if refinement["doc"]:
            lines += [refinement["doc"], ""]
        for member in refinement["members"]:
            lines += ["```python", member["signature"], "```", ""]
            if member["doc"]:
                lines += [member["doc"], ""]

    lines += ["**Existing Providers**", ""]
    lines += _render_providers(entry["providers"])
    return lines


def _render_seam(entry: dict[str, Any]) -> list[str]:
    """The Markdown section for one non-Stage signal seam."""
    lines: list[str] = [f"### `{entry['prefix']}` — {entry['protocol']['name']}", ""]
    lines += [
        f"- **Register as**: `{entry['prefix']}.<variant>` under `{entry['entry_point']['group']}`",
        f"- **Conformance case**: `{entry['conformance_case']}`",
        "",
    ]
    lines += _render_protocol(entry["protocol"], heading_level=4)
    for refinement in entry["optional_refinements"]:
        lines += [f"#### Optional refinement — `{refinement['name']}`", ""]
        if refinement["doc"]:
            lines += [refinement["doc"], ""]
        for member in refinement["members"]:
            lines += ["```python", member["signature"], "```", ""]
            if member["doc"]:
                lines += [member["doc"], ""]
    lines += ["**Existing Providers**", ""]
    lines += _render_providers(entry["providers"])
    return lines


def _render_carrier(entry: dict[str, Any]) -> list[str]:
    """The Markdown section for one record carrier: docstring + field table."""
    lines: list[str] = [f"### `{entry['name']}`", ""]
    if entry["doc"]:
        lines += [entry["doc"], ""]
    lines += ["| Field | Type | Required | Description |", "| --- | --- | --- | --- |"]
    for field in entry["fields"]:
        required = "yes" if field["required"] else "no"
        lines.append(
            f"| `{field['name']}` | `{_table_cell(field['type'])}` | {required} | "
            f"{_table_cell(field['description'])} |"
        )
    lines.append("")
    return lines


def render_markdown(reference: dict[str, Any]) -> str:
    """The human-readable rendering of the SAME reference dict the JSON serializes.

    Rendering from the already-built dict (never re-reading the sources) is what guarantees the
    two committed artifacts cannot disagree with each other.
    """
    lines: list[str] = [
        "# Stage contracts — the replacement contract for every Capability",
        "",
        "<!-- GENERATED by latence_core.contract_reference — do NOT edit by hand. -->",
        f"<!-- Regenerate: {reference['regenerate']} -->",
        "",
        "This page is **generated** from the framework's single source of truth — the",
        "`CapabilityDescriptor` table, the Capability Protocols and the record contracts",
        "(see [Capability protocols](capability.md) and [Data contracts](contracts.md) for the",
        "full API pages) — and a drift gate keeps it byte-identical to those sources. It exists",
        "so that a coding agent can replace **any** component at **any** Stage — a parser, an",
        "extractor, a disambiguator — with full knowledge of what that Stage receives and what",
        "it must emit, such that a replacement cannot violate the end-to-end flow.",
        "",
        f"Machine-readable twin: [stage-contracts.json](stage-contracts.json). "
        f"Regenerate both with `{reference['regenerate']}`.",
        "",
        "## Replacing a Provider — the five steps",
        "",
        "Worked example: replace `parser.lighton` (the LightOn OCR parser) with your own parser.",
        "",
        "1. **Read the contract entry.** Find the Capability's section below — for a parser,",
        "   the `parse` entry: its Protocol (`Parser`, one method",
        "   `parse(inputs: Iterable[ParserInput]) -> Iterator[DocumentRecord]`), the input shape",
        "   the Runner delivers, the carrier it must emit, and the wiring rules. The method and",
        "   carrier docstrings reproduced here ARE the contract — offsets, page maps, Provenance",
        "   and determinism obligations included.",
        "2. **Implement the Protocol.** Capabilities are structural `typing.Protocol`s: any object",
        "   with the right shape satisfies one — no base class, no import of `latence-core`",
        "   required. The end-to-end path is [Authoring a Provider](../providers/README.md); the",
        "   real template is the [example provider](example-provider.md).",
        "3. **Declare the entry point** in your package's `pyproject.toml` under",
        '   `[project.entry-points."latence.providers"]`, named `<prefix>.<variant>` with the',
        "   Capability's prefix from its entry below — e.g. `parse.myparser =",
        '   "my_pkg.provider:MyParser"`. The prefix IS the routing: an unknown prefix is refused',
        "   loudly at discovery, never silently skipped.",
        "4. **Run the conformance suite.** Your Provider is auto-covered by its name prefix; the",
        "   C1–C6 gate (valid typed records, graceful typed failure, license evidence,",
        "   determinism-or-documented, device honoured, no PII leak) is the machine check that a",
        "   replacement holds the contract — see",
        "   [Passing Provider Conformance](../CONFORMANCE.md):",
        "",
        "   ```bash",
        f"   {reference['conformance_command']}",
        "   ```",
        "",
        "5. **Pin it in the pipeline.** Name the new Provider in the Stage's `provider:` field of",
        "   your stack/pipeline YAML (or let `latence setup` write it — e.g. its `--parser`",
        "   option), and run.",
        "",
        "**The quality stance:** the contract gate guarantees the *flow*, not the *quality* — a",
        "conformance-green replacement cannot break the pipeline, but the quality of its output",
        "is the responsibility of the user who introduces it; measure it with the benchmark",
        "harness before trusting it.",
        "",
        "## How to read an entry",
        "",
        "The per-Capability facts below are the fields of one frozen `CapabilityDescriptor`; the",
        "field meanings, in the words of the source module:",
        "",
        reference["legend"]["capability_descriptor"],
        "",
        "**Levels** (`document` vs `corpus`):",
        "",
        reference["legend"]["capability_level"],
        "",
        "**Input shapes** (how the Runner delivers a Stage's input):",
        "",
        reference["legend"]["stage_input_shape"],
        "",
        "## Stage Capabilities",
        "",
    ]
    for entry in reference["capabilities"]:
        lines += _render_capability(entry)
    lines += [
        "## Signal seams (non-Stage Capabilities)",
        "",
        "These Capabilities register under the same `latence.providers` group and pass the same",
        "conformance gate, but they are not DAG Stages: no `CapabilityKind`, no checkpoint of",
        "their own. They are resolved by the engine (the delta processor).",
        "",
    ]
    for seam_entry in reference["signal_seams"]:
        lines += _render_seam(seam_entry)
    lines += [
        "## Protocol result types",
        "",
        "The frozen dataclasses several Protocols return — the non-Record half of the output",
        "API. Each docstring names its producer.",
        "",
    ]
    for result in reference["result_types"]:
        lines += [f"### `{result['name']}`", ""]
        if result["doc"]:
            lines += [result["doc"], ""]
        lines += [
            "| Field | Type |",
            "| --- | --- |",
        ]
        lines += [
            f"| `{field['name']}` | `{_table_cell(field['type'])}` |"
            for field in result["fields"]
        ]
        lines.append("")
    lines += [
        "## Record carriers",
        "",
        "The Pydantic contracts that flow between Stages. Every carrier extends `Record`",
        "(Provenance + Classification are required — a record cannot lose its lineage at a Stage",
        "boundary). Nested value types (page maps, offset maps, Evidence, spans) are documented",
        "on the full [Data contracts](contracts.md) page.",
        "",
    ]
    for carrier in reference["carriers"]:
        lines += _render_carrier(carrier)
    return "\n".join(lines).rstrip() + "\n"


def render_reference_files(repo_root: Path) -> dict[Path, str]:
    """The committed artifacts as ``{path: content}`` — one builder for the CLI and the gate."""
    reference = build_reference(workspace_provider_packages(repo_root))
    out_dir = repo_root / "docs" / "reference"
    return {
        out_dir / "stage-contracts.json": render_json(reference),
        out_dir / "stage-contracts.md": render_markdown(reference),
    }
