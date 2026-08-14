# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The harness's API: ``run(plan)``, the plan's layout, the probes' cell list, the refusals.

Wave 5-K promoted ``run(plan) -> MatrixOutcome`` to THE entry and demoted argparse to an
adapter over it. The properties that had to survive that move are the ones pinned here:

* the default ``--out`` and the ``{dataset}__{leg}.json`` naming did not move — cells written
  before the change and after it are the same files,
* the ``PYTHONHASHSEED`` refusal and the sample seed sit at the API entry, so the campaign
  stage (which never touches ``parse_args``) cannot produce a non-reproducible cell,
* a :class:`~benchmark.s5.datasets.CorpusLayout` redirects every input together, and the
  DEFAULT layout resolves to exactly the paths the module constants always resolved to.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from benchmark.s5 import run_matrix
from benchmark.s5.cell import RESULTS_DIRS, cell_path
from benchmark.s5.datasets import DATASETS, DEFAULT_LAYOUT, CorpusLayout, dataset_specs
from benchmark.s5.legs import ALL_LEG_NAMES

if TYPE_CHECKING:
    from pathlib import Path


class TestDefaultsDidNotMove:
    def test_default_out_is_the_campaign_stage_directory(self) -> None:
        # The campaign stage writes to <campaign_root>/stages/s5-retrieval; the CLI default
        # must BE that directory for the committed campaign, or the two entries would fill
        # different matrices from the same config.
        assert run_matrix.DEFAULT_OUT == (
            run_matrix.REPO_ROOT / "benchmark" / "sota-campaign" / "stages" / "s5-retrieval"
        )
        assert run_matrix.parse_args([]).out == run_matrix.DEFAULT_OUT

    def test_cell_naming_and_the_two_fusion_namespaces_are_unchanged(self) -> None:
        assert RESULTS_DIRS == {"equal": "results", "tuned": "results-weighted"}
        plan = run_matrix.parse_args(["--datasets", "vidoseek", "--legs", "dense"])
        assert plan.results_dir == plan.out / "results"
        assert run_matrix.planned_cells(plan) == (
            plan.out / "results" / "vidoseek__dense.json",
        )
        tuned = run_matrix.parse_args(
            ["--datasets", "vidoseek", "--legs", "dense", "--fusion", "tuned"]
        )
        assert tuned.results_dir == tuned.out / "results-weighted"

    def test_the_sample_seed_is_the_one_the_matrix_was_run_under(self) -> None:
        assert run_matrix.SAMPLE_SEED == 20260809


class TestPlannedCells:
    def test_every_dataset_x_leg_in_matrix_order(self) -> None:
        plan = run_matrix.parse_args(
            ["--datasets", "musique,vidoseek", "--legs", "bm25,dense"]
        )
        # Legs are canonicalised into ladder order by parse_args, not left as typed.
        assert plan.legs == ("dense", "bm25")
        assert [p.name for p in run_matrix.planned_cells(plan)] == [
            "musique__dense.json",
            "musique__bm25.json",
            "vidoseek__dense.json",
            "vidoseek__bm25.json",
        ]

    def test_a_resolution_only_plan_owns_no_cells(self) -> None:
        plan = run_matrix.parse_args(["--datasets", "vidoseek", "--resolution-only"])
        assert run_matrix.planned_cells(plan) == ()


class TestCorpusLayout:
    def test_the_default_layout_resolves_to_the_module_constants(self) -> None:
        # Byte-identical default behaviour: the parameterisation may not move one path.
        assert CorpusLayout.under(run_matrix.REPO_ROOT) == DEFAULT_LAYOUT
        default = dataset_specs()
        assert {n: s.export_dir for n, s in default.items()} == {
            n: s.export_dir for n, s in DATASETS.items()
        }
        assert {n: s.s2_root for n, s in default.items()} == {
            n: s.s2_root for n, s in DATASETS.items()
        }

    def test_a_redirected_layout_moves_every_input_together(self, tmp_path: Path) -> None:
        specs = dataset_specs(CorpusLayout.under(tmp_path))
        assert specs["vidoseek"].export_dir == (
            tmp_path / "benchmark" / "sota-corpus" / "s3-exports" / "vidoseek"
        )
        assert specs["vidoseek"].s2_root == (
            tmp_path / "benchmark" / "sota-corpus" / "s2-output"
        )
        assert specs["musique"].export_dir == (
            tmp_path / "benchmark" / "official-data" / "musique-full" / "export"
        )

    def test_a_plans_specs_follow_its_own_layout(self, tmp_path: Path) -> None:
        plan = run_matrix.Plan(
            datasets=("vidoseek",),
            legs=("dense",),
            limit=None,
            out=tmp_path / "out",
            force=False,
            layout=CorpusLayout.under(tmp_path),
        )
        assert plan.specs["vidoseek"].export_dir.is_relative_to(tmp_path)
        # The output root is deliberately independent of the layout.
        assert plan.results_dir == tmp_path / "out" / "results"


class TestDeterminismRefusal:
    def test_run_refuses_without_the_pinned_hash_seed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        plan = run_matrix.Plan(
            datasets=(), legs=("dense",), limit=None, out=tmp_path, force=False
        )
        # At the API entry — the campaign stage never calls main(), and must be refused too.
        with pytest.raises(run_matrix.MatrixRefused, match="PYTHONHASHSEED=0"):
            run_matrix.run(plan)

    def test_the_cli_prints_the_same_refusal_and_exits_2(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        assert run_matrix.main(["--datasets", "vidoseek"]) == 2
        assert capsys.readouterr().err == (
            "refusing to run: set PYTHONHASHSEED=0 (the hypergraph entry is hash-order "
            "sensitive; a run without it is not reproducible).\n"
        )


class TestRunReportsWhatItDid:
    def test_absent_inputs_produce_refusal_cells_a_report_and_an_outcome(
        self, tmp_path: Path
    ) -> None:
        """An export-less layout: every leg refuses, and ``run`` says so instead of faking.

        This is also the cheap end-to-end of the new API — cells on disk, REPORT.md and
        metrics.json written, and a :class:`MatrixOutcome` whose counts match the filesystem.
        """
        plan = run_matrix.Plan(
            datasets=("musique",),
            legs=("dense", "bm25"),
            limit=None,
            out=tmp_path / "out",
            force=False,
            layout=CorpusLayout.under(tmp_path / "nothing-here"),
        )
        outcome = run_matrix.run(plan)

        assert outcome.plan is plan
        assert outcome.written == run_matrix.planned_cells(plan)
        assert outcome.cells == run_matrix.planned_cells(plan)
        assert outcome.report == tmp_path / "out" / "REPORT.md"
        assert outcome.metrics == tmp_path / "out" / "metrics.json"
        assert outcome.report.is_file() and outcome.metrics.is_file()

        payload = json.loads(
            cell_path(plan.results_dir, "musique", "dense").read_text(encoding="utf-8")
        )
        assert payload["status"] == "missing-inputs"
        assert payload["missing"] == ["corpus.parquet"]

    def test_a_second_run_rewrites_nothing_but_still_reports_the_cells(
        self, tmp_path: Path
    ) -> None:
        plan = run_matrix.Plan(
            datasets=("musique",),
            legs=("dense",),
            limit=None,
            out=tmp_path / "out",
            force=False,
            layout=CorpusLayout.under(tmp_path / "nothing-here"),
        )
        run_matrix.run(plan)
        again = run_matrix.run(plan)
        assert again.written == ()  # resume never redoes a cell it already has
        assert len(again.cells) == 1


def test_the_leg_vocabulary_is_the_leg_table() -> None:
    """``--legs all`` is the table, in table order — the report's column order."""
    assert run_matrix.parse_args(["--legs", "all"]).legs == ALL_LEG_NAMES
