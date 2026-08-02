# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The ``latence-campaign`` staged-run engine (e2e-sota-run S1).

This package productizes the shape ``benchmark/e2e/supervise.sh`` proved on the pod — a
stage table, filesystem done-probes, atomic ``STATE.json``, ``blocked`` as a terminal state
carried with a reason — as an importable, offline-testable Python module, wrapped in a wizard
that mirrors the ``latence setup`` house style (frozen answers struct, pure byte-deterministic
generator, flag > prompt > default, dry-validate before write).

Layout of a campaign directory::

    <campaign_root>/
      campaign.yaml          # the wizard's artifact (committable)
      STATE.json             # atomic, rewritten in full on every transition
      stages/<name>/         # per-stage metrics.json + artifacts

One deliberate difference from ``supervise.sh``: stages here are separated by an OPERATOR
GATE. A stage that finishes is ``done``, not passed — ``latence-campaign approve <stage>``
records the sign-off, and the next stage's ``run`` refuses until then. A 10-hour GPU campaign
should not cascade into stage N+1 on the strength of stage N's own opinion of itself.
"""

from latence_benchmark.campaign.stages import (
    CampaignContext,
    GateRefusedError,
    StageBlockedError,
    StageSpec,
    registry,
    run_stage,
)
from latence_benchmark.campaign.state import CampaignState, CampaignStateError, StageState
from latence_benchmark.campaign.wizard import (
    KNOWN_DATASETS,
    CampaignAnswers,
    CampaignConfigError,
    build_campaign_config,
    parse_campaign_config,
)

__all__ = [
    "KNOWN_DATASETS",
    "CampaignAnswers",
    "CampaignConfigError",
    "CampaignContext",
    "CampaignState",
    "CampaignStateError",
    "GateRefusedError",
    "StageBlockedError",
    "StageSpec",
    "StageState",
    "build_campaign_config",
    "parse_campaign_config",
    "registry",
    "run_stage",
]
