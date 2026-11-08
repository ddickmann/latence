"""Provider-neutral adapter for the hosted generative model (ticket 20 spike, decision 6).

Two tiers, chosen by the owner on 2026-11-08:
  - "deep": relation/class definitions and type-induction proposals.
  - "fast": a short explanation per review-band item, shown to the expert.

The engine only ever sees ``TextLLM``; a deployment can route calls through its own cloud tenant
by supplying a different client. Spike rule: only synthetic or public data is sent; client data
needs an explicit, per-deployment egress decision.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Tier = Literal["deep", "fast"]

DEFAULT_MODELS: dict[str, str] = {"deep": "claude-opus-5-5", "fast": "claude-haiku-5-5"}
DEFAULT_EFFORT: dict[str, str] = {"deep": "high", "fast": "low"}


class TextLLM(Protocol):
    def complete(self, prompt: str, *, tier: Tier, system: str | None = None, max_tokens: int = 2000) -> str: ...


@dataclass
class StubLLM:
    """Deterministic stand-in for tests: records every call, never touches the network."""

    reply: str = "stub"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(self, prompt: str, *, tier: Tier, system: str | None = None, max_tokens: int = 2000) -> str:
        self.calls.append({"prompt": prompt, "tier": tier, "system": system, "max_tokens": max_tokens})
        return self.reply


class ClaudeLLM:
    """``TextLLM`` over the Anthropic Messages API. Credentials come from the environment."""

    def __init__(self, client: Any | None = None, models: Mapping[str, str] | None = None) -> None:
        if client is None:
            import anthropic  # imported lazily so tests and offline runs need no SDK

            client = anthropic.Anthropic()
        self.client = client
        self.models = {**DEFAULT_MODELS, **(models or {})}

    def complete(self, prompt: str, *, tier: Tier, system: str | None = None, max_tokens: int = 2000) -> str:
        kwargs: dict[str, Any] = {
            "model": self.models[tier],
            "max_tokens": max_tokens,
            "output_config": {"effort": DEFAULT_EFFORT[tier]},
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if tier == "deep":
            # On a policy decline the API re-runs the request on a fallback model in the same call.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
            resp = self.client.beta.messages.create(**kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model declined ({getattr(resp.stop_details, 'category', None)})")
        return "".join(b.text for b in resp.content if b.type == "text").strip()


EXPLAIN_SYSTEM = (
    "You help a master-data expert decide whether two records describe the same real-world "
    "entity. Be brief and concrete: name the fields that agree, the fields that conflict, and "
    "what a reviewer should check. Do not decide for the expert."
)


def explain_review_item(llm: TextLLM, left: Mapping[str, Any], right: Mapping[str, Any], p_match: float) -> str:
    """Two-to-four sentence explanation of a review-band pair, for the expert workbench."""
    prompt = (
        f"Calibrated match probability: {p_match:.2f} (uncertain band).\n"
        f"Record A: {json.dumps(dict(left), default=str, sort_keys=True)}\n"
        f"Record B: {json.dumps(dict(right), default=str, sort_keys=True)}\n"
        "Explain in 2-4 sentences."
    )
    return llm.complete(prompt, tier="fast", system=EXPLAIN_SYSTEM, max_tokens=2000)


def define_relation(llm: TextLLM, label: str, head_type: str, tail_type: str, examples: list[str]) -> str:
    """One-sentence definition of a relation label in context (EDC-style canonicalisation)."""
    ex = "\n".join(f"- {e}" for e in examples[:8])
    prompt = (
        f"Relation label: {label!r}, from a {head_type} to a {tail_type}.\n"
        f"Example sentences:\n{ex}\n"
        "Write one precise sentence defining what this relation asserts, so it can be told "
        "apart from look-alike relations. Output only the sentence."
    )
    return llm.complete(prompt, tier="deep", max_tokens=16000)
