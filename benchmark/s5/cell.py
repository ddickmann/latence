# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The S5 result cell — ONE module owning every shape a ``<dataset>__<leg>.json`` file can take.

Before this module, :mod:`benchmark.s5.run_matrix` wrote five undeclared payload shapes inline
and :mod:`benchmark.s5.report` read them back through dozens of ``.get`` calls — the schema
existed only as the intersection of what the writer happened to emit and what the reader
happened to probe. This module states it once. The shapes:

* **ok / ladder** — the run header plus ``metrics`` (:func:`ladder_body`). Written for the
  ladder rungs, the ablation legs, and the equal-fusion baseline generally.
* **ok / rescue** — the header plus ``metrics``, ``budget_curve``, ``gate`` and
  ``select_budget`` (:func:`rescue_body`).
* **ok / tuned** — the header plus ``metrics`` (tuned-on-test), ``metrics_equal_weight_on_test``,
  ``tuning``, ``split`` and ``tunable`` (:func:`tuned_body`).
* **missing-inputs** — a refusal, no header: ``status``, ``missing``, ``dataset``, ``leg``
  (:func:`missing_inputs_cell`). Written before any export is loaded, which is why it carries
  no resolution or fusion provenance.
* **not-fusion-tunable** — the rescue family under ``--fusion tuned``
  (:func:`not_fusion_tunable_body`): a stated refusal, because the rescue legs append their
  graph lanes rather than weighting them.

**The merge rule, stated instead of implied.** A cell an actual run writes is
``{**header, **body}`` (:meth:`Header.merge`). The header always says ``status: "ok"``; a
*refusal body* deliberately overwrites it (``not-fusion-tunable`` here,
``insufficient-split`` from :meth:`benchmark.s5.tuning.InsufficientSplit.to_dict`) — the run
provenance is kept, the claim of success is replaced. This was always the writer's behaviour;
it is now its documented contract.

**Byte discipline.** Cells are the campaign's oracle surface: :func:`write_cell` serialises
with ``sort_keys=True`` (so dict build order can never leak into the file) and every
constructor here passes values through untouched — no coercion, no rounding, no defaulting —
so the key SET and the float reprs are exactly the ones the accumulators produced, including
:meth:`~benchmark.s5.metrics.CellAccumulator.summary`'s ``None``s and its deliberately-omitted
``doc_recall@k`` keys.

:class:`Cell` is the read side: the report never probes a raw payload dict again, it asks the
one wrapper whose accessors are named after the schema above.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# -- the status vocabulary ----------------------------------------------------------------------

OK = "ok"
MISSING_INPUTS = "missing-inputs"
NOT_FUSION_TUNABLE = "not-fusion-tunable"
INSUFFICIENT_SPLIT = "insufficient-split"

# -- where cells live ---------------------------------------------------------------------------

#: Each fusion variant's cell directory, relative to a run's ``--out``. Separate directories,
#: never separate *keys* in one directory: the equal-weight matrix is a long-running job, and a
#: variant must not be able to overwrite, reorder or shadow one of its cells even by accident.
#:
#: It lives HERE, beside :func:`cell_path`, because "where a cell file is" and "what a cell file
#: is called" are one fact — and the campaign's ``s5-retrieval`` done-probe needs that fact
#: without importing the whole harness (this module is stdlib-only on purpose).
RESULTS_DIRS = {"equal": "results", "tuned": "results-weighted"}


def results_dir(out: Path, fusion: str) -> Path:
    """The cell directory ``fusion`` owns under a run's output root."""
    return out / RESULTS_DIRS[fusion]

#: Why the rescue family has no tuned-fusion cell — written verbatim into the refusal body.
NOT_FUSION_TUNABLE_REASON = (
    "the graph-rescue family appends its graph lanes to the base list instead of "
    "fusing them into it, so it has no per-signal RRF weight vector to fit"
)


# -- writers ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Header:
    """The run provenance every successfully-run cell carries, whatever its body says.

    ``resolution`` is the gold-resolution record for *this run's sample* (the standalone
    ``resolution/<dataset>.json`` file remains the full-set authority — see
    :func:`benchmark.s5.report._resolution_for`).
    """

    dataset: str
    leg: str
    fusion: str
    vector_source: str
    control: bool
    limit: int | None
    resolution: dict[str, Any]

    def merge(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """``{**header, **body}`` — the cell payload, with the status-overwrite rule.

        The header contributes ``status: "ok"``; a refusal body (``not-fusion-tunable``,
        ``insufficient-split``) overwrites it while keeping the provenance keys. That override
        is the point of the merge order, not an accident of it.
        """
        return {
            "status": OK,
            "dataset": self.dataset,
            "leg": self.leg,
            "fusion": self.fusion,
            "vector_source": self.vector_source,
            "control": self.control,
            "limit": self.limit,
            "resolution": self.resolution,
            **body,
        }


def missing_inputs_cell(dataset: str, leg: str, missing: Sequence[str]) -> dict[str, Any]:
    """The refusal cell: the artifacts ``leg`` needed and ``dataset``'s export does not have."""
    return {"status": MISSING_INPUTS, "missing": list(missing), "dataset": dataset, "leg": leg}


def ladder_body(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """The baseline body — one metric row, exactly as the accumulator summarised it."""
    return {"metrics": dict(metrics)}


def rescue_body(
    metrics: Mapping[str, Any],
    budget_curve: Mapping[str, Any],
    gate: Mapping[str, Any],
    select_budget: int,
) -> dict[str, Any]:
    """The rescue body: the headline row, the recall-over-budget curve, and the gate stats.

    ``metrics`` is the curve's ``select_budget`` point; ``budget_curve`` keys are the stringified
    budgets (JSON object keys are strings — stringified by the caller so the file and the
    in-memory payload agree).
    """
    return {
        "metrics": dict(metrics),
        "budget_curve": dict(budget_curve),
        "gate": dict(gate),
        "select_budget": select_budget,
    }


def not_fusion_tunable_body() -> dict[str, Any]:
    """The rescue family's ``--fusion tuned`` refusal — overwrites the header's ``status``."""
    return {"status": NOT_FUSION_TUNABLE, "reason": NOT_FUSION_TUNABLE_REASON}


def tuned_body(
    metrics: Mapping[str, Any],
    metrics_equal_weight_on_test: Mapping[str, Any],
    tuning: Mapping[str, Any],
    split: Mapping[str, Any],
    *,
    tunable: bool,
) -> dict[str, Any]:
    """The weighted body: both fusion variants on the same test half, plus the fit's audit trail."""
    return {
        "metrics": dict(metrics),
        "metrics_equal_weight_on_test": dict(metrics_equal_weight_on_test),
        "tuning": dict(tuning),
        "split": dict(split),
        "tunable": tunable,
    }


def cell_path(results_dir: Path, dataset: str, leg: str) -> Path:
    """Where one (dataset, leg) cell lives — the resume probe and the writer use the same name."""
    return results_dir / f"{dataset}__{leg}.json"


def write_cell(results_dir: Path, dataset: str, leg: str, payload: Mapping[str, Any]) -> None:
    """Serialise one cell — ``sort_keys=True``, so build order can never leak into the bytes."""
    cell_path(results_dir, dataset, leg).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_cells(results_dir: Path, dataset: str, legs: Sequence[str]) -> dict[str, Any]:
    """Read ``dataset``'s cells for ``legs`` back — absent files are absent keys, never nulls."""
    out: dict[str, Any] = {}
    for leg in legs:
        path = cell_path(results_dir, dataset, leg)
        if path.is_file():
            out[leg] = json.loads(path.read_text(encoding="utf-8"))
    return out


# -- the read side ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One cell payload behind schema-named accessors — the report's only way into a cell.

    Total by design: every accessor answers for every shape (a key another shape owns reads as
    its empty value), because the report renders refusals and successes through the same tables
    and must never crash on the shape it is precisely there to surface.
    """

    payload: Mapping[str, Any]

    @staticmethod
    def of(value: object) -> Cell | None:
        """``value`` as a cell, or ``None`` for anything that is not a payload mapping."""
        return Cell(value) if isinstance(value, Mapping) else None

    @property
    def status(self) -> str:
        """The cell's status claim; a payload without one reads as ``"not run"``."""
        return str(self.payload.get("status", "not run"))

    @property
    def is_ok(self) -> bool:
        """Whether this cell claims a successful run (the only shapes carrying ``metrics``)."""
        return self.payload.get("status") == OK

    @property
    def missing(self) -> list[str]:
        """The refusal's absent-artifact list; empty on every other shape."""
        value = self.payload.get("missing", [])
        return [str(item) for item in value] if isinstance(value, list) else []

    @property
    def metrics(self) -> Mapping[str, Any]:
        """The headline metric row; empty on refusal shapes."""
        value = self.payload.get("metrics")
        return value if isinstance(value, Mapping) else {}

    @property
    def gate(self) -> Mapping[str, Any] | None:
        """The rescue gate stats, or ``None`` on every non-rescue shape."""
        value = self.payload.get("gate")
        return value if isinstance(value, Mapping) else None

    @property
    def budget_curve(self) -> Mapping[str, Any] | None:
        """The rescue recall-over-budget curve (budget string → metric row), or ``None``."""
        value = self.payload.get("budget_curve")
        return value if isinstance(value, Mapping) else None

    @property
    def tuning(self) -> Mapping[str, Any]:
        """The tuned cell's fit audit trail; empty on every other shape."""
        value = self.payload.get("tuning")
        return value if isinstance(value, Mapping) else {}

    @property
    def split(self) -> Mapping[str, Any]:
        """The tuned cell's dev/test split record; empty on every other shape."""
        value = self.payload.get("split")
        return value if isinstance(value, Mapping) else {}

    @property
    def equal_metrics(self) -> Mapping[str, Any]:
        """The tuned cell's equal-weight row on the same test half; empty elsewhere."""
        value = self.payload.get("metrics_equal_weight_on_test")
        return value if isinstance(value, Mapping) else {}

    @property
    def not_tunable(self) -> bool:
        """Whether the cell explicitly recorded ``tunable: false`` (a single-signal leg's fit)."""
        return self.payload.get("tunable") is False

    @property
    def resolution(self) -> Mapping[str, Any] | None:
        """This run's sampled gold-resolution record, or ``None`` where none was embedded."""
        value = self.payload.get("resolution")
        return value if isinstance(value, Mapping) else None

    @property
    def limit(self) -> object:
        """The ``--limit`` the cell was run under (``None`` = the full query set)."""
        return self.payload.get("limit")


__all__ = [
    "INSUFFICIENT_SPLIT",
    "MISSING_INPUTS",
    "NOT_FUSION_TUNABLE",
    "NOT_FUSION_TUNABLE_REASON",
    "OK",
    "RESULTS_DIRS",
    "Cell",
    "Header",
    "cell_path",
    "ladder_body",
    "load_cells",
    "missing_inputs_cell",
    "not_fusion_tunable_body",
    "rescue_body",
    "results_dir",
    "tuned_body",
    "write_cell",
]
