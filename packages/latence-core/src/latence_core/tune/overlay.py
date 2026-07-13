"""The runner-side overlay — apply a tuned config's per-Provider knobs, fingerprint-guarded (W11).

The runner overlays a tuned config's winning per-Provider knobs onto the stack's stage configs ONLY
when the tuned config's ``device_fingerprint`` MATCHES the current device — else it warns and uses
the stack defaults (never silently applies a mismatched tune, the reviewer's guard). Config-overlay
ONLY: no contract change, no offset change — the same shallow per-stage config merge the bake-off
uses (``{**stage.config, **winning_knobs}``), so a tuned run is byte-for-byte the stack it would be
with those knobs hand-written into the YAML.

Pure + light (no runner/stage-heavy imports beyond the Pipeline contract): a ``Pipeline`` in, a
``Pipeline`` out. The CLI ``latence run --tuned tuned/<device>.yaml`` (and a stack's ``tune:`` key)
drives it before handing the Pipeline to the Runner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from latence_core.pipeline import Pipeline, Stage
from latence_core.tune.fingerprint import DeviceFingerprint, probe_device_fingerprint
from latence_core.tune.report import TuneReport

_log = logging.getLogger("latence.tune")


@dataclass(frozen=True)
class OverlayResult:
    """The outcome of an overlay attempt — the (possibly unchanged) Pipeline + why (W11).

    * ``pipeline`` — the overlaid Pipeline on a fingerprint MATCH, else the input Pipeline UNCHANGED
      (defaults, never a mismatched tune silently applied).
    * ``applied`` — whether the tune was applied (the fingerprint matched).
    * ``reason`` — a human-readable explanation (applied N Providers, or the mismatch that
      vetoed it),
      for the CLI to echo (and a warning is logged on a mismatch).
    * ``stages_overlaid`` — the stage names whose config got tuned knobs (empty on a mismatch / when
      no stage referenced a tuned Provider).
    """

    pipeline: Pipeline
    applied: bool
    reason: str
    stages_overlaid: tuple[str, ...] = ()


def _overlay_stage(stage: Stage, knobs: dict[str, object]) -> Stage:
    """Return a copy of ``stage`` with the tuned ``knobs`` shallow-merged onto its config.

    The candidate's knobs OVERRIDE only the keys they name (``dtype``/``compile``/``batch_size``/…)
    and the stage inherits the rest — identical to the bake-off's controlled ``{**base,
    **override}``
    merge, so nothing but the tuned knobs changes. An empty knob dict is a no-op (the baseline won).
    A nested Export/Disambiguation ``config['embedder']`` whose provider is the tuned one is merged
    inside its own ``config`` block (the embedder is an opt-in sub-config, not a top-level node).
    """
    if not knobs:
        return stage.model_copy()
    merged = {**stage.config, **knobs}
    return stage.model_copy(update={"config": merged})


def _overlay_embedder(stage: Stage, tuned_provider: str, knobs: dict[str, object]) -> Stage | None:
    """Overlay knobs onto a nested ``config['embedder']`` whose provider is ``tuned_provider``.

    Returns the updated stage if this stage carries a matching nested embedder (so its embed knobs
    are tuned), else ``None`` (this stage has no such embedder). Kept separate from the top-level
    overlay so a stage can be tuned at BOTH levels (its own provider AND its nested embedder)
    without
    one clobbering the other's config merge.
    """
    config = stage.config
    embedder = config.get("embedder")
    if not (isinstance(embedder, dict) and embedder.get("provider") == tuned_provider):
        return None
    if not knobs:
        return None
    inner = embedder.get("config")
    inner_config = dict(inner) if isinstance(inner, dict) else {}
    merged_embedder = {**embedder, "config": {**inner_config, **knobs}}
    return stage.model_copy(update={"config": {**config, "embedder": merged_embedder}})


def overlay_tuned_config(
    pipeline: Pipeline,
    tune: TuneReport,
    *,
    device_fingerprint: DeviceFingerprint | None = None,
) -> OverlayResult:
    """Overlay ``tune``'s per-Provider winning knobs onto ``pipeline`` IF the fingerprint matches.

    The fingerprint guard (the crux): ``device_fingerprint`` (the current device's, probed if not
    injected) must MATCH the tuned config's ``tune.device_fingerprint`` — else the overlay is
    VETOED,
    a warning is logged, and the pipeline is returned UNCHANGED (stack defaults). This is what stops
    a
    tune measured on one GPU from being silently applied on another (where a winning ``compile:
    true``
    could crash). On a match, each stage whose Provider (top-level or nested embedder) has a
    :class:`ProviderTuning` gets its winning knobs shallow-merged onto its config — config-only, no
    contract/offset change.
    """
    current = device_fingerprint or probe_device_fingerprint()
    if not current.matches(tune.device_fingerprint):
        reason = (
            f"tuned config was measured on device {tune.device_fingerprint.slug()} "
            f"(key {tune.device_fingerprint.key}) but this host is {current.slug()} "
            f"(key {current.key}); NOT applying it — using stack defaults."
        )
        _log.warning("%s", reason)
        return OverlayResult(pipeline=pipeline, applied=False, reason=reason)

    tunings = tune.per_provider()
    new_stages: list[Stage] = []
    overlaid: list[str] = []
    for stage in pipeline.stages:
        updated = stage
        touched = False
        # Top-level provider tune.
        tuning = tunings.get(stage.provider)
        if tuning is not None and tuning.overlay_config():
            updated = _overlay_stage(updated, tuning.overlay_config())
            touched = True
        # Nested embedder tune (ADR-0017 opt-in embedder) — may co-exist with the top-level one.
        for provider, provider_tuning in tunings.items():
            embed_updated = _overlay_embedder(updated, provider, provider_tuning.overlay_config())
            if embed_updated is not None:
                updated = embed_updated
                touched = True
        if touched:
            overlaid.append(stage.name)
        new_stages.append(updated if touched else stage.model_copy())

    overlaid_pipeline = pipeline.model_copy(update={"stages": new_stages})
    reason = (
        f"applied tuned config for device {current.slug()} (key {current.key}) to "
        f"{len(overlaid)} stage(s): {', '.join(overlaid) or '(none referenced a tuned provider)'}"
    )
    _log.info("%s", reason)
    return OverlayResult(
        pipeline=overlaid_pipeline,
        applied=True,
        reason=reason,
        stages_overlaid=tuple(overlaid),
    )
