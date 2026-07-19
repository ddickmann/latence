"""Every committed ``deploy/env/*/stack.yaml`` component is actually CONSTRUCTIBLE (audit R5).

The gap this closes: an environment's stack could name a real Provider with a config that Provider
*rejects*, and every green gate would still say OK.

* ``latence env verify`` (gate 1 of ``smoke.sh`` — the only gate ``scripts/verify-local.sh`` runs)
  cross-checks provider **names** against the stack. It never constructs one.
* ``latence stack validate`` on a CPU host skips-with-flag every ``device: cuda`` Stage **before**
  the Provider is constructed, so it returns PASS.

The defect therefore only surfaced on a real GPU pod, at gate 3 — the exact "discovered on a pod,
hours in" failure mode the environment spec exists to eliminate. Concretely: both
``deploy/env/gpu-learned/stack.yaml`` and ``deploy/env/full-stack/stack.yaml`` declared
``redaction.gliner2`` with ``{device: cuda}`` and no ``pii_labels``, which that Provider hard-raises
on, making #198's GPU-placement proof unreachable for one of the five declared
``learned_providers``.

Construction is cheap and offline: every in-tree Provider defers its heavy backend import to first
use (ADR-0016), so ``__init__`` only validates config. A Provider whose optional backend is genuinely
absent from *this* environment raises ``ImportError`` at construction and is SKIPPED — a missing
extra is an environment fact, not a stack defect. Any other error is a real config defect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from latence_core.capability import ProviderRegistry
from latence_core.envspec import stack_components
from latence_core.setup.wizard import BLESSED_REDACT_PROVIDER, FINANCIAL_PII_LABELS

# packages/latence-core/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_ROOT = _REPO_ROOT / "deploy" / "env"
_ENV_STACKS = sorted(_ENV_ROOT.glob("*/stack.yaml"))


def _ids(paths: list[Path]) -> list[str]:
    return [p.parent.name for p in paths]


def _unconstructible(stack_path: Path) -> tuple[list[str], int]:
    """``(defect descriptions, number of components that really constructed)`` for one stack.

    The checker the tests below share, so the bite test drives the SAME code path as the guard.
    """
    defects: list[str] = []
    constructed = 0
    for component in stack_components(stack_path):
        try:
            provider_cls = ProviderRegistry().load(component.provider)
        except Exception as exc:  # noqa: BLE001 - registry/entry-point faults are reported, not raised
            defects.append(f"{component.provider}: not loadable — {type(exc).__name__}: {exc}")
            continue
        try:
            provider_cls(dict(component.config))
        except ImportError:
            # Optional heavy backend absent from THIS environment (ADR-0016). Not a stack defect.
            continue
        except Exception as exc:  # noqa: BLE001 - the defect class this guard exists for
            defects.append(f"{component.provider}: {type(exc).__name__}: {exc}")
            continue
        constructed += 1
    return defects, constructed


def test_at_least_one_environment_stack_is_committed() -> None:
    """A parametrised suite over an empty glob passes vacuously; this makes that impossible."""
    assert _ENV_STACKS, f"no deploy/env/*/stack.yaml found under {_ENV_ROOT}"


@pytest.mark.parametrize("stack_path", _ENV_STACKS, ids=_ids(_ENV_STACKS))
def test_every_component_of_a_committed_environment_stack_constructs(stack_path: Path) -> None:
    """The committed config satisfies every Provider it names — no pod round-trip to find out."""
    defects, constructed = _unconstructible(stack_path)
    assert not defects, (
        f"{stack_path.relative_to(_REPO_ROOT)} declares components that cannot be constructed "
        f"from their committed config:\n  " + "\n  ".join(defects)
    )
    assert constructed, (
        f"{stack_path.relative_to(_REPO_ROOT)}: nothing was actually constructed — the guard would "
        "pass vacuously in this environment"
    )


def test_the_guard_bites_when_a_stack_drops_a_required_config_key(tmp_path: Path) -> None:
    """Strip ``pii_labels`` back off the gpu-learned redactor: the checker MUST report it.

    This is the exact committed shape the audit found. A guard that cannot be made to fail is not
    a guard.
    """
    source = _ENV_ROOT / "gpu-learned" / "stack.yaml"
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    stripped = 0
    for stage in raw["stages"]:
        if stage.get("provider") == BLESSED_REDACT_PROVIDER:
            stage["config"].pop("pii_labels", None)
            stripped += 1
    assert stripped == 1, "expected exactly one redaction.gliner2 Stage in the gpu-learned stack"

    mutated = tmp_path / "stack.yaml"
    mutated.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    defects, _constructed = _unconstructible(mutated)
    assert any(BLESSED_REDACT_PROVIDER in defect for defect in defects), defects


@pytest.mark.parametrize("stack_path", _ENV_STACKS, ids=_ids(_ENV_STACKS))
def test_the_learned_redactor_carries_the_financial_pii_floor(stack_path: Path) -> None:
    """ADR-0044: wherever the blessed redactor runs, the financial-PII floor is really declared.

    ``docs/compatibility-matrix.md`` and ADR-0041 §7a both claim ``redaction.gliner2`` "carries the
    ADR-0044 financial floor". This asserts the committed files carry it, rather than the docs
    merely saying so — the generator (``wizard``) already emits exactly this set.
    """
    for component in stack_components(stack_path):
        if component.provider != BLESSED_REDACT_PROVIDER:
            continue
        labels = component.config.get("pii_labels")
        assert isinstance(labels, list), (
            f"{stack_path.relative_to(_REPO_ROOT)}: {BLESSED_REDACT_PROVIDER} declares no "
            "'pii_labels' list (the Provider rejects that config outright)"
        )
        missing = [label for label in FINANCIAL_PII_LABELS if label not in labels]
        assert not missing, (
            f"{stack_path.relative_to(_REPO_ROOT)}: {BLESSED_REDACT_PROVIDER} is below the "
            f"ADR-0044 financial-PII floor — missing {missing}"
        )
