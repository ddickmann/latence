"""``matrix/<stage>.yaml`` — the candidate-provider list for one Stage (ADR-0036 §4, P3-F5).

A ``matrix/<stage>.yaml`` names ONE Stage (by :class:`~latence_core.pipeline.CapabilityKind`) and
the candidate Providers to compare for it. The bake-off holds every OTHER Stage at the base stack
(``stacks/default.yaml``) and swaps in each candidate for the target Stage — a real controlled
comparison. Example::

    stage: parse
    stage_name: parse            # the node in the base stack to swap (optional; inferred by kind)
    candidates:
      - provider: parser.plaintext
      - provider: parser.pdfplumber
      - provider: parser.document
      - provider: entity.gliner   # GPU/learned candidate — device-gated, skipped-with-flag on CPU
        config: {device: cuda}

Each candidate may carry a per-candidate ``config`` override merged onto the base stage's config
(so a GPU candidate can pin ``device: cuda`` and a chunker candidate its own ``max_tokens``). The
spec is a strict Pydantic contract — an unknown key or a malformed Stage kind is a hard error, not a
silently-ignored typo — parsed with the same one-schema/dual-serialization discipline as the
Pipeline (ADR-0006).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from latence_core.pipeline import CapabilityKind

if TYPE_CHECKING:
    from collections.abc import Mapping

# The bake-off matrix schema version (additive changes bump it, like the Pipeline's).
MATRIX_SCHEMA_VERSION = 1


class MatrixCandidate(BaseModel):
    """One candidate Provider to bake off for the target Stage, with an optional config override.

    ``provider`` is the registered Provider name (the entry-point name under
    ``latence.providers``). ``config`` is an optional per-candidate override merged ONTO the base
    stack's target-Stage config (shallow key-wise), so a GPU candidate pins ``device: cuda`` and a
    chunker candidate its own ``max_tokens`` without duplicating the whole stage config. A candidate
    with no override runs with the base stage's config verbatim — the controlled comparison.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(..., min_length=1, description="Registered candidate Provider name.")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Per-candidate config override merged onto the base."
    )


class MatrixSpec(BaseModel):
    """A ``matrix/<stage>.yaml`` — one target Stage + the candidate Providers to compare for it.

    ``stage`` is the target :class:`~latence_core.pipeline.CapabilityKind` (``parse``/``chunk``/
    ``entity_extraction``/…). ``stage_name`` names the node in the base stack to swap; when omitted
    it is inferred as the unique Stage of that Capability in the base stack (an ambiguous base stack
    with two Stages of the kind requires an explicit ``stage_name``). ``candidates`` is the ordered
    list to bake off — declared order is the report's row order (deterministic). ``base_stack``
    overrides which stack holds the other stages constant (defaults to the blessed
    ``stacks/default.yaml`` the harness resolves).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=MATRIX_SCHEMA_VERSION)
    stage: CapabilityKind = Field(..., description="Target Stage Capability the candidates fulfil.")
    stage_name: str | None = Field(
        default=None, description="Base-stack node to swap (inferred by kind when omitted)."
    )
    base_stack: str | None = Field(
        default=None, description="Stack whose OTHER stages are held constant (default: default)."
    )
    candidates: list[MatrixCandidate] = Field(
        ..., min_length=1, description="Ordered candidate Providers to compare."
    )

    @model_validator(mode="after")
    def _distinct_candidates(self) -> MatrixSpec:
        """A candidate Provider may appear at most once — a duplicate row is a spec error.

        Two identical ``provider`` rows would produce two indistinguishable table rows and double
        the run cost for no signal; surface it as a config error rather than silently baking the
        same Provider twice.
        """
        names = [c.provider for c in self.candidates]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            msg = f"Duplicate candidate provider(s) in matrix: {dupes}"
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml_text(cls, text: str) -> MatrixSpec:
        """Parse + validate a ``matrix/<stage>.yaml`` document (a mapping at the top level)."""
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            msg = "Matrix YAML must be a mapping at the top level."
            raise ValueError(msg)
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> MatrixSpec:
        """Validate a parsed mapping into a :class:`MatrixSpec`."""
        return cls.model_validate(data)
