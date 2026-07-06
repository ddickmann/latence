"""Typed contracts for the showable demo (S13).

The demo composes S1–S9 and adds two small first-class artifacts of its own — the
site-manifest the viewer Export writes next to the HTML, and the run summary the
CLI returns — both versioned Pydantic models so the demo's own boundary is typed
and diffable exactly like every inter-Stage contract (Baseline bar). Neither
carries raw document text or PII: counts, ids, and dispositions only, safe to
persist and to render in a page an operator hands to a client.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# v1 (S13): the demo site-manifest + run summary land as versioned contracts.
DEMO_SCHEMA_VERSION = 1


class GraphSummary(BaseModel):
    """The knowledge-graph shape the viewer renders (nodes / edges / Evidence counts).

    Derived by the :class:`~latence_demo.viewer.DemoSiteExport` Provider from the
    :class:`~latence_core.contracts.GraphRecord`s that flow through the Export seam — so
    it reflects exactly the graph that was assembled, not a model claim. Counts only.
    """

    model_config = ConfigDict(extra="forbid")

    nodes_total: int = 0
    edges_total: int = 0
    nodes_linked: int = Field(default=0, description="Nodes carrying an external-KB id.")
    edges_with_evidence: int = Field(
        default=0, description="Edges whose Evidence names >=1 source mention."
    )
    nodes_by_type: dict[str, int] = Field(default_factory=dict)
    edges_by_label: dict[str, int] = Field(default_factory=dict)


class DemoSiteManifest(BaseModel):
    """What the viewer Export wrote — the self-contained page + the graph it rendered.

    Persisted as JSON next to ``index.html`` so a resumed/automated run can assert the
    viewer materialised without re-parsing the HTML. ``self_contained`` records the
    invariant the S13 AC demands: the page pulls no external CDN/network resource.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=DEMO_SCHEMA_VERSION)
    site_uri: str = Field(..., min_length=1, description="URI of the written index.html.")
    graph: GraphSummary = Field(default_factory=GraphSummary)
    self_contained: bool = Field(
        default=True, description="True when the page references no external network resource."
    )


class DangerousFixtureOutcome(BaseModel):
    """One planted dangerous fixture and the Screening disposition it actually received.

    The demo plants deliberately dangerous files (type-spoof, zip-bomb, oversized,
    prompt-injection) and then proves Screening caught each — ``expected`` is the planted
    intent, ``observed`` what the run's Quality Report actually recorded, ``matched`` their
    agreement. A mismatch is the demo failing loudly rather than shipping a false "it's
    safe" story (S13 AC: a dangerous fixture is visibly Quarantined; an injection is
    visibly flagged).
    """

    model_config = ConfigDict(extra="forbid")

    file_name: str
    expected: str
    observed: str
    matched: bool


class DemoRunSummary(BaseModel):
    """The result the ``latence-demo`` CLI returns / prints — the run's headline (S13).

    A typed, versioned roll-up: where the artifacts landed, the KG shape, whether every
    planted dangerous fixture was caught as intended, and whether contract-completeness
    held. Rendered to stdout and persisted, so the one-command demo is self-describing and
    machine-checkable (the E2E test asserts against this model, not scraped text).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=DEMO_SCHEMA_VERSION)
    pipeline_name: str
    run_id: str
    storage_uri: str
    run_dir: str
    site_uri: str
    documents_ingested: int = 0
    quarantined: int = 0
    flagged: int = 0
    graph: GraphSummary = Field(default_factory=GraphSummary)
    dangerous: list[DangerousFixtureOutcome] = Field(default_factory=list)
    contracts_complete: bool = True

    @property
    def dangerous_all_caught(self) -> bool:
        """True when every planted dangerous fixture received its expected disposition."""
        return all(d.matched for d in self.dangerous)

    def render(self) -> str:
        """A compact human-readable summary printed by the CLI (deterministic)."""
        lines = [
            f"Latence demo — {self.pipeline_name} ({self.run_id})",
            f"  storage:   {self.storage_uri}",
            f"  run dir:   {self.run_dir}",
            f"  viewer:    {self.site_uri}",
            f"  documents: {self.documents_ingested} ingested, "
            f"{self.quarantined} quarantined, {self.flagged} flagged",
            f"  graph:     {self.graph.nodes_total} nodes, {self.graph.edges_total} edges, "
            f"{self.graph.nodes_linked} KB-linked, "
            f"{self.graph.edges_with_evidence} edges with Evidence",
            f"  contracts: {'complete' if self.contracts_complete else 'INCOMPLETE'}",
            "  screening checks:",
        ]
        for d in self.dangerous:
            mark = "ok" if d.matched else "MISMATCH"
            lines.append(f"    - {d.file_name}: expected {d.expected}, got {d.observed} [{mark}]")
        return "\n".join(lines) + "\n"
