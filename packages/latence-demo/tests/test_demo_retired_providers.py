"""The demo names no Provider a measurement retired (ADR-0075)."""

from __future__ import annotations

from latence_core.stacks.retired import retired_use


def test_the_demo_names_no_retired_provider() -> None:
    from latence_demo.pipeline import build_pipeline

    pipeline = build_pipeline("file://out", "corpus")
    assert [s.provider for s in pipeline.stages
            if retired_use(s.provider, s.config) is not None] == []
