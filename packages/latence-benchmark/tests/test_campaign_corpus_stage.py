# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The s1-corpus stage: gate-before-bytes, fetcher seam, manifest + metrics artifacts.

Everything here is offline: real fetchers are exercised only against ``file://`` URLs and
local directories; the Hub fetcher stays behind the seam (its skip-if-exists behavior is the
library's documented contract, and the REAL acquisition is the operator's run — no fake
green about it here).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from latence_benchmark.campaign.corpus_stage import (
    CORPUS_PLAN,
    KIND_HF_SNAPSHOT,
    KIND_LOCAL_REFERENCE,
    KIND_URL,
    LANE_STRICTLY_LOCAL,
    STAGE_NAME,
    UDA_HTML_RULING,
    DatasetPlan,
    FetchResult,
    corpus_done_probe,
    fetch_local_reference,
    fetch_urls,
    license_gate,
    run_corpus_stage,
)
from latence_benchmark.campaign.stages import (
    CampaignContext,
    StageBlockedError,
    stage_dir,
)
from latence_benchmark.campaign.wizard import KNOWN_DATASETS, CampaignAnswers
from latence_benchmark.licensing import UNVERIFIED, DatasetLicense

if TYPE_CHECKING:
    from collections.abc import Mapping

    from latence_benchmark.campaign.corpus_stage import Fetcher

REPO_ROOT = Path(__file__).resolve().parents[3]


def permissive_license() -> DatasetLicense:
    return DatasetLicense(
        dataset_license="CC-BY-4.0",
        code_license="MIT",
        commercial_use=True,
        attribution_required=True,
        verified_on="2026-08-02",
        source_url="https://example.test",
    )


def make_repo(tmp_path: Path) -> Path:
    """A minimal fixture checkout: the incumbents' dirs + the REAL manifest tool."""
    repo = tmp_path / "repo"
    for incumbent in ("musique-full", "hotpot-full"):
        d = repo / "benchmark" / "official-data" / incumbent
        d.mkdir(parents=True)
        (d / "data.jsonl").write_text('{"q": 1}\n')
    manifest_dst = repo / "benchmark" / "corpus" / "manifest.py"
    manifest_dst.parent.mkdir(parents=True)
    # The REAL repo tool, reused verbatim — the integration under test is ours, not a fake's.
    shutil.copy(REPO_ROOT / "benchmark" / "corpus" / "manifest.py", manifest_dst)
    return repo


def make_ctx(
    tmp_path: Path,
    datasets: tuple[str, ...],
    *,
    accept_non_commercial: bool = False,
) -> CampaignContext:
    return CampaignContext(
        campaign_root=tmp_path / "campaign",
        repo_root=make_repo(tmp_path),
        answers=CampaignAnswers(campaign_root=str(tmp_path / "campaign"), datasets=datasets),
        accept_non_commercial=accept_non_commercial,
    )


def writing_fetchers(files_per_dataset: int = 2) -> Mapping[str, Fetcher]:
    """Fakes that land real bytes under dest — so the manifest measures something true."""

    def fake(plan: DatasetPlan, dest: Path) -> FetchResult:
        dest.mkdir(parents=True, exist_ok=True)
        for i in range(files_per_dataset):
            (dest / f"{plan.name}-{i}.json").write_text("{}")
        return FetchResult(fetched=files_per_dataset, skipped=0)

    def local(plan: DatasetPlan, dest: Path) -> FetchResult:
        return FetchResult(fetched=0, skipped=0, note="referenced")

    return {KIND_HF_SNAPSHOT: fake, KIND_URL: fake, KIND_LOCAL_REFERENCE: local}


class TestPlanTable:
    def test_every_known_dataset_is_classified(self) -> None:
        # The wizard's vocabulary and the gate's plan table must never drift apart:
        # a selectable dataset without a lane would block at run time.
        assert set(CORPUS_PLAN) == set(KNOWN_DATASETS)

    def test_every_license_is_verified_and_dated(self) -> None:
        for name, plan in CORPUS_PLAN.items():
            assert plan.license.verified, f"{name} carries an UNVERIFIED license"
            assert plan.license.verified_on == "2026-08-02"
            assert plan.license.source_url.startswith("https://")

    def test_hub_plans_are_pinned_to_full_shas(self) -> None:
        for plan in CORPUS_PLAN.values():
            if plan.kind == KIND_HF_SNAPSHOT:
                assert len(plan.revision) == 40, f"{plan.name} is not pinned"
                assert plan.repo_id

    def test_incumbents_are_local_references(self) -> None:
        for name in ("musique", "hotpotqa"):
            plan = CORPUS_PLAN[name]
            assert plan.kind == KIND_LOCAL_REFERENCE
            assert plan.local_source.startswith("benchmark/official-data/")

    def test_uda_excludes_the_html_subsets(self) -> None:
        plan = CORPUS_PLAN["uda"]
        joined = " ".join(plan.allow_patterns)
        assert "fin" in joined and "paper_tab" in joined
        assert "nq" not in joined and "feta" not in joined
        assert UDA_HTML_RULING in plan.notes


class TestLicenseGate:
    def test_unclassified_dataset_is_refused(self) -> None:
        with pytest.raises(StageBlockedError, match="NO license-lane classification"):
            license_gate(["musique"], accept_non_commercial=False, plan={})

    def test_unverified_license_is_refused_with_no_override(self) -> None:
        plan = {
            "musique": DatasetPlan(
                name="musique",
                lane=LANE_STRICTLY_LOCAL,
                license=DatasetLicense(
                    dataset_license=UNVERIFIED,
                    code_license="MIT",
                    commercial_use=True,
                    attribution_required=False,
                    verified_on="",
                    source_url="https://example.test",
                ),
                kind=KIND_LOCAL_REFERENCE,
                local_source="x",
            )
        }
        with pytest.raises(StageBlockedError, match="UNVERIFIED"):
            license_gate(["musique"], accept_non_commercial=True, plan=plan)

    def test_ohr_bench_requires_the_explicit_acknowledgement(self) -> None:
        with pytest.raises(StageBlockedError, match="accept-non-commercial"):
            license_gate(["ohr_bench"], accept_non_commercial=False)
        cleared = license_gate(["ohr_bench"], accept_non_commercial=True)
        assert cleared[0].name == "ohr_bench"

    def test_permissive_selection_passes_without_acknowledgement(self) -> None:
        cleared = license_gate(
            ["musique", "uda", "multihop_rag"], accept_non_commercial=False
        )
        assert [p.name for p in cleared] == ["musique", "uda", "multihop_rag"]

    def test_gate_runs_before_any_fetcher(self, tmp_path: Path) -> None:
        calls: list[str] = []

        def spy(plan: DatasetPlan, dest: Path) -> FetchResult:
            calls.append(plan.name)
            return FetchResult(fetched=0, skipped=0)

        ctx = make_ctx(tmp_path, ("musique", "ohr_bench"))  # no acknowledgement
        fetchers: Mapping[str, Fetcher] = {
            KIND_LOCAL_REFERENCE: spy, KIND_HF_SNAPSHOT: spy, KIND_URL: spy
        }
        with pytest.raises(StageBlockedError, match="accept-non-commercial"):
            run_corpus_stage(ctx, fetchers=fetchers)
        # musique is classified BEFORE ohr_bench in the selection, yet no byte moved:
        # the whole selection is gated before the first fetcher runs.
        assert calls == []


class TestFetchers:
    def test_local_reference_verifies_presence_without_touching_dest(
        self, tmp_path: Path
    ) -> None:
        repo = make_repo(tmp_path)
        plan = CORPUS_PLAN["musique"]
        dest = repo / "benchmark" / "sota-corpus" / "musique" / "raw"
        result = fetch_local_reference(plan, dest, repo_root=repo)
        assert "referenced in place" in result.note
        assert not dest.exists()  # nothing copied — one copy of the incumbents, ever

    def test_local_reference_blocks_when_the_dataset_is_missing(
        self, tmp_path: Path
    ) -> None:
        repo = make_repo(tmp_path)
        shutil.rmtree(repo / "benchmark" / "official-data" / "hotpot-full")
        with pytest.raises(StageBlockedError, match="missing or empty"):
            fetch_local_reference(
                CORPUS_PLAN["hotpotqa"], tmp_path / "dest", repo_root=repo
            )

    def test_url_fetcher_skips_existing_files(self, tmp_path: Path) -> None:
        source = tmp_path / "served.json"
        source.write_text('{"payload": true}')
        plan = DatasetPlan(
            name="direct",
            lane=LANE_STRICTLY_LOCAL,
            license=permissive_license(),
            kind=KIND_URL,
            urls=(("served.json", source.as_uri()),),
        )
        dest = tmp_path / "dest"

        first = fetch_urls(plan, dest)
        assert (first.fetched, first.skipped) == (1, 0)
        assert (dest / "served.json").read_text() == '{"payload": true}'

        second = fetch_urls(plan, dest)  # resumable: nothing re-downloaded
        assert (second.fetched, second.skipped) == (0, 1)

    def test_url_fetcher_failure_blocks_and_leaves_no_partial_file(
        self, tmp_path: Path
    ) -> None:
        plan = DatasetPlan(
            name="direct",
            lane=LANE_STRICTLY_LOCAL,
            license=permissive_license(),
            kind=KIND_URL,
            urls=(("gone.json", (tmp_path / "does-not-exist.json").as_uri()),),
        )
        dest = tmp_path / "dest"
        with pytest.raises(StageBlockedError, match="failed to fetch"):
            fetch_urls(plan, dest)
        assert list(dest.glob("*")) == []


class TestStageRun:
    def test_full_run_writes_manifest_and_metrics(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path, ("musique", "hotpotqa", "uda"))
        note = run_corpus_stage(ctx, fetchers=writing_fetchers())
        assert "3 dataset(s)" in note

        out = stage_dir(ctx.campaign_root, STAGE_NAME)
        manifest = json.loads((out / "manifest.json").read_text())
        assert (out / "MANIFEST.md").is_file()
        by_name = {s["name"]: s for s in manifest["sources"]}
        assert set(by_name) == {"musique", "hotpotqa", "uda"}
        # Measured, not claimed: the incumbents' referenced dirs hold 1 file each, the
        # fake hub fetch landed 2 for uda.
        assert by_name["musique"]["file_count"] == 1
        assert by_name["uda"]["file_count"] == 2
        # Only the attribution lane is publishable.
        assert by_name["musique"]["publishable"] is True
        assert by_name["hotpotqa"]["publishable"] is False

        metrics = json.loads((out / "metrics.json").read_text())
        assert metrics["stage"] == STAGE_NAME
        assert metrics["accept_non_commercial"] is False
        assert metrics["datasets"]["uda"]["lane"] == "share-alike-local"
        assert metrics["datasets"]["uda"]["provenance"].startswith("hf:qinchuanhui/UDA-QA@")
        assert UDA_HTML_RULING in metrics["rulings"]

    def test_raw_bytes_land_under_sota_corpus(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path, ("wiki2multihop",))
        run_corpus_stage(ctx, fetchers=writing_fetchers())
        raw = ctx.repo_root / "benchmark" / "sota-corpus" / "wiki2multihop" / "raw"
        assert sorted(p.name for p in raw.iterdir()) == [
            "wiki2multihop-0.json",
            "wiki2multihop-1.json",
        ]

    def test_fetcher_failure_surfaces_as_blocked_reason(self, tmp_path: Path) -> None:
        def broken(plan: DatasetPlan, dest: Path) -> FetchResult:
            raise StageBlockedError(f"{plan.name}: connection reset by peer")

        ctx = make_ctx(tmp_path, ("multihop_rag",))
        with pytest.raises(StageBlockedError, match="connection reset"):
            run_corpus_stage(ctx, fetchers={KIND_HF_SNAPSHOT: broken})

    def test_done_probe_tracks_the_selection(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path, ("musique",))
        assert corpus_done_probe(ctx) is False
        run_corpus_stage(ctx, fetchers=writing_fetchers())
        assert corpus_done_probe(ctx) is True

        # Editing the selection re-opens the stage: the old gate artifact does not cover it.
        wider = CampaignContext(
            campaign_root=ctx.campaign_root,
            repo_root=ctx.repo_root,
            answers=CampaignAnswers(
                campaign_root=str(ctx.campaign_root), datasets=("musique", "hotpotqa")
            ),
        )
        assert corpus_done_probe(wider) is False

    def test_metrics_record_the_non_commercial_acknowledgement(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path, ("ohr_bench",), accept_non_commercial=True)
        run_corpus_stage(ctx, fetchers=writing_fetchers())
        metrics = json.loads(
            (stage_dir(ctx.campaign_root, STAGE_NAME) / "metrics.json").read_text()
        )
        assert metrics["accept_non_commercial"] is True
        assert metrics["datasets"]["ohr_bench"]["lane"] == "research-only-local"
