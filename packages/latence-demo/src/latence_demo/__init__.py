"""latence-demo — the showable one-command demo (S13, ADR-0011).

Ingests a bundled messy corpus and produces parsed markdown, a browsable knowledge graph, and
the Quality Report — CPU-only, offline, self-contained web viewer, plus a documented one-line
Provider swap. Composes the S1–S9 spine; adds only the corpus, the viewer Export Provider, and
the DemoBuilder that ties them together.
"""

from __future__ import annotations

from latence_demo.builder import DemoBuilder, DemoResult
from latence_demo.contracts import (
    DangerousFixtureOutcome,
    DemoRunSummary,
    DemoSiteManifest,
    GraphSummary,
)
from latence_demo.corpus import CorpusManifest, build_corpus_bytes, stage_corpus
from latence_demo.pipeline import build_pipeline
from latence_demo.viewer import DemoSiteExport, render_site

__all__ = [
    "CorpusManifest",
    "DangerousFixtureOutcome",
    "DemoBuilder",
    "DemoResult",
    "DemoRunSummary",
    "DemoSiteExport",
    "DemoSiteManifest",
    "GraphSummary",
    "build_corpus_bytes",
    "build_pipeline",
    "render_site",
    "stage_corpus",
]
