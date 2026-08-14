# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The wizard→component mapping for the ``retrieval:`` block: one home, one pinned claim.

``s3_lib.generate_pipeline_yaml`` is the model this follows: the wizard block is answers, and a
pure function turns those answers into the artifact its consumer actually runs on. Here the
consumer is ``latence_retrieval.rescue.GraphRescueRetrieval`` and the artifact is its config
mapping, so :func:`rescue_config` is the ONE place the block's names become the component's
names. Before it existed the mapping lived only in the guide's prose, which meant a rename on
either side broke nothing and was discovered at query time.

Import-independence is preserved deliberately (the wizard's rule): nothing here imports
latence-retrieval — the campaign package *configures* the production component, it never runs
it. The lockstep is proven from the TEST side instead, where importing both is free:
``tests/test_campaign_retrieval_wizard.py::TestRescueConfig`` asserts every emitted key is in
``GraphRescueRetrieval._CONFIG_KEYS`` and that the defaults actually construct one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from latence_benchmark.campaign.wizard import RetrievalAnswers


def rescue_config(answers: RetrievalAnswers) -> dict[str, Any]:
    """One ``retrieval:`` block → the ``GraphRescueRetrieval`` config mapping.

    Every key is a wizard knob under its component name; the one non-identity mapping is the
    budget, because the block carries BOTH per-selector budgets (so switching selectors keeps
    each one's measured operating point) while the component takes the single ``budget`` of the
    selector in force. Selector ``none`` emits no budget at all — the component ignores it.

    What this deliberately does NOT emit: the component's deployment-side knobs (``pool_k``,
    ``select_pool_k``, ``rank_score_k``, the gate thresholds, ``hyperedge_kv_path``). They are
    not campaign answers, and their defaults ARE the measured operating point; the caller merges
    what its deployment owns on top. Note the consequence for selector ``coe``: it selects over
    the hypergraph lane, so the caller must add ``hyperedge_kv_path`` (plus the injected
    ``hyperedge_documents`` resolver) or construction is refused — loudly, at construction.
    """
    config: dict[str, Any] = {
        "graph_mode": answers.graph_mode,
        "selector": answers.selector,
        "seed_top_k": answers.seed_top_k,
        "hop_weight": answers.hop_weight,
        "community_weight": answers.community_weight,
        "centrality_weight": answers.centrality_weight,
        "seed_docs_k": answers.seed_docs_k,
        "seed_node_cap": answers.seed_node_cap,
        "rescue_cap": answers.rescue_cap,
        "pack_centrality_weight": answers.pack_centrality_weight,
    }
    budget = {"packed": answers.packed_budget, "coe": answers.coe_budget}.get(answers.selector)
    if budget is not None:
        config["budget"] = budget
    return config


def render_rescue_config(answers: RetrievalAnswers) -> str:
    """:func:`rescue_config` as reviewable YAML — pure, byte-deterministic, header derived.

    The header is assembled from the same mapping the body is (the ``build_campaign_config``
    house rule), so the two cannot drift.
    """
    config = rescue_config(answers)
    body: str = yaml.safe_dump(config, sort_keys=True, default_flow_style=False)
    header = [
        "# GraphRescueRetrieval config, derived from campaign.yaml's `retrieval:` block by",
        "# latence_benchmark.campaign.retrieval_lib.rescue_config — the one wizard->component",
        "# mapping (pinned to the component's _CONFIG_KEYS by the retrieval-wizard tests).",
        f"# selector {config['selector']}"
        + (f" @ {config['budget']} tok" if "budget" in config else " (ranked union, no budget)")
        + f" | graph {config['graph_mode']}",
        "# Deployment-side knobs (pool_k, select_pool_k, rank_score_k, the gate thresholds,",
        "# hyperedge_kv_path) are NOT campaign answers: their component defaults are the",
        "# measured operating point, and the deployment merges its own on top.",
    ]
    return "\n".join(header) + "\n" + body
