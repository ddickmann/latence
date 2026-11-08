"""The hosted-LLM adapter routes tiers correctly and never needs the network in tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm import ClaudeLLM, StubLLM, define_relation, explain_review_item


class _FakeMessages:
    def __init__(self, stop_reason: str = "end_turn") -> None:
        self.kwargs: dict = {}
        self.stop_reason = stop_reason

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            stop_details=SimpleNamespace(category="x"),
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=" ok ")],
        )


def _fake_client(stop_reason: str = "end_turn"):
    plain, beta = _FakeMessages(stop_reason), _FakeMessages(stop_reason)
    return SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta)), plain, beta


def test_fast_tier_uses_the_small_model_without_fallback_beta() -> None:
    client, plain, beta = _fake_client()
    out = ClaudeLLM(client).complete("q", tier="fast")
    assert out == "ok"
    assert plain.kwargs["model"] == "claude-haiku-5-5" and "fallbacks" not in plain.kwargs
    assert beta.kwargs == {}


def test_deep_tier_uses_the_large_model_with_server_side_fallback() -> None:
    client, plain, beta = _fake_client()
    ClaudeLLM(client).complete("q", tier="deep")
    assert beta.kwargs["model"] == "claude-opus-5-5"
    assert beta.kwargs["fallbacks"] == "default"
    assert plain.kwargs == {}


def test_refusal_is_raised_not_returned_as_text() -> None:
    client, _, _ = _fake_client("refusal")
    with pytest.raises(RuntimeError):
        ClaudeLLM(client).complete("q", tier="fast")


def test_helpers_pick_their_tier() -> None:
    stub = StubLLM()
    explain_review_item(stub, {"name": "Acme Ltd"}, {"name": "ACME LIMITED"}, 0.6)
    define_relation(stub, "supplies", "Organization", "Organization", ["A supplies B with steel."])
    assert [c["tier"] for c in stub.calls] == ["fast", "deep"]
    assert "0.60" in stub.calls[0]["prompt"]
