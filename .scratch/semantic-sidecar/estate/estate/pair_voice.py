"""A hosted model on the pairs of records the linker left for a person (ticket 48).

The owner's rule (14 Nov): a hosted model may decide pairs in the uncertain band only under the
deployment's stated egress rule, off by default, each call recorded. Here:

- **Off unless switched on.** ``link.hosted_model.enabled: true`` in the estate file, with an
  egress rule beside it: ``fields`` (which fields of a record may leave: of ``name``, ``city``
  and ``country``, never a key or an id), ``model``, ``prices`` and ``max_usd``. Switched on
  without a rule, it refuses rather than guess one.
- **Only those fields leave**, for each of the two records, without their ids.
- **A cap.** Before each call its worst case (every character sent counted as a token, the whole
  output allowance used) is added to what the calls made so far could have cost; a call that
  would take that past ``max_usd`` is not made, nor any after it.
- **Each call recorded** in the log: the pair, what was sent, the model that served it, the
  answer, its worst case and its cost at the listed prices.
- **It decides nothing.** The verdict goes on the review item as a vote with the model as its
  actor (ADR-0069: a model's verdict accepts nothing by itself). The pair stays for a person.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

FIELDS = ("name", "city", "country")  # what a record of the linker carries that may be sent
VERDICTS = ("same", "different", "unsure")
INSTRUCTIONS = ("Two records from two business systems are shown with a few of their fields. Say whether they name the same "
                "organisation: answer with one word, same, different or unsure.")


class EgressRefused(ValueError):
    """The estate switched the hosted model on without a rule it can be held to."""


class PairClient(Protocol):
    def judge(self, request: dict[str, Any]) -> dict[str, Any]:
        """``{"verdict": ..., "served_by": ..., "usage": {"input_tokens", "output_tokens"}}``."""
        ...


def rule_of(link: dict[str, Any]) -> dict[str, Any] | None:
    """The egress rule of the estate's link block, or ``None`` when the model is off."""
    spec = link.get("hosted_model") or {}
    if not spec.get("enabled"):
        return None
    fields = spec.get("fields")
    if not fields:
        raise EgressRefused("link.hosted_model is switched on with no egress rule: name the fields that may leave (link.hosted_model.fields)")
    unknown = sorted(set(fields) - set(FIELDS))
    if unknown:
        raise EgressRefused(f"link.hosted_model.fields names {', '.join(unknown)}: only {', '.join(FIELDS)} of a record may leave")
    prices = spec.get("prices") or {}
    if not {"input_per_1m", "output_per_1m"} <= set(prices) or spec.get("max_usd") is None:
        raise EgressRefused("link.hosted_model states no prices or no max_usd: the calls cannot be held under a limit")
    return {"model": str(spec.get("model") or "gpt-6-luna"), "fields": tuple(f for f in FIELDS if f in fields), "prices": dict(prices),
            "max_usd": float(spec["max_usd"]), "max_output_tokens": int(spec.get("max_output_tokens") or 200)}


def _request(rule: dict[str, Any], sent: dict[str, Any]) -> dict[str, Any]:
    return {"model": rule["model"], "instructions": INSTRUCTIONS, "input": json.dumps(sent, ensure_ascii=False, sort_keys=True),
            "max_output_tokens": rule["max_output_tokens"], "store": False}


def _worst_case(request: dict[str, Any], prices: dict[str, Any]) -> float:
    sent = len(json.dumps(request, ensure_ascii=False))
    return round((sent * float(prices["input_per_1m"]) + int(request["max_output_tokens"]) * float(prices["output_per_1m"])) / 1e6, 6)


def ask_about_pairs(review: list[dict[str, Any]], records: dict[str, dict[str, Any]], rule: dict[str, Any] | None,
                    client: PairClient | None, log: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The review items with the model's votes on the pairs of records, and what was asked."""
    if rule is None:
        return review, {"state": "off"}
    if client is None:
        raise EgressRefused("link.hosted_model is switched on and no client was given to the build")
    out, calls, committed, spent, stopped = [], 0, 0.0, 0.0, 0
    asked = [r for r in review if r.get("a") in records and r.get("b") in records]
    for item in review:
        if item not in asked:
            out.append(item)
            continue
        sent = {end: {f: records[item[end]][f] for f in rule["fields"] if records[item[end]].get(f) is not None} for end in ("a", "b")}
        request = _request(rule, sent)
        bound = _worst_case(request, rule["prices"])
        if stopped or committed + bound > rule["max_usd"]:
            stopped += 1
            out.append(item)
            continue
        answer = client.judge(request)
        verdict = str(answer.get("verdict"))
        if verdict not in VERDICTS:
            raise RuntimeError(f"the model answered {verdict!r} on {item['id']}; one of {', '.join(VERDICTS)} was asked for")
        usage = answer.get("usage") or {}
        usd = round((int(usage.get("input_tokens", 0)) * float(rule["prices"]["input_per_1m"])
                     + int(usage.get("output_tokens", 0)) * float(rule["prices"]["output_per_1m"])) / 1e6, 6)
        served = str(answer.get("served_by") or rule["model"])
        committed, spent, calls = committed + bound, spent + usd, calls + 1
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"pair": item["id"], "sent": sent, "model": served, "answer": verdict, "worst_case_usd": bound, "usd": usd},
                                ensure_ascii=False, sort_keys=True) + "\n")
        out.append({**item, "model_vote": {"verdict": verdict, "actor": f"model:{served}"}})
    return out, {"state": "asked", "pairs": len(asked), "calls": calls, "usd": round(spent, 6), "stopped_at_cap": stopped}


class OpenAIPairClient:
    """The hosted model over OpenAI's Responses API, on the official SDK, one call per pair. The
    key is read by the SDK from the environment. No silent second attempt: a call that fails is
    reported, not repeated and billed again."""

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            import openai  # imported here so that tests and builds without this voice need no SDK

            client = openai.OpenAI(max_retries=0)
        self.client = client

    def judge(self, request: dict[str, Any]) -> dict[str, Any]:
        response = self.client.responses.create(**request)
        if response.status != "completed":
            raise RuntimeError(f"the model's answer is {response.status}; no vote")
        word = str(response.output_text).strip().lower().strip(".")
        u = response.usage
        return {"verdict": word, "served_by": response.model, "usage": {"input_tokens": int(u.input_tokens), "output_tokens": int(u.output_tokens)}}
