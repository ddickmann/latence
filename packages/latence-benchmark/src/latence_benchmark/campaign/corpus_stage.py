# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""Stage ``s1-corpus`` — acquire the campaign datasets behind the license gate.

The order of operations is the point: the **license gate runs before any byte moves**
(:func:`latence_benchmark.licensing.require_usable`, the ADR-0012 pattern applied to data),
so a restricted corpus can never be silently downloaded and then argued about. Each dataset
carries a hand-verified :class:`~latence_benchmark.licensing.DatasetLicense` plus a campaign
**lane** — what we may DO with the bytes once they land:

* ``share-alike-local`` — CC BY-SA-derived text; kept local by campaign discipline.
* ``strictly-local``    — permissive annotations over third-party copyrighted text
  (MultiHop-RAG news, ViDoSeek source PDFs); the raw corpus never leaves this machine.
* ``research-only-local`` — non-commercial corpus (OHR-Bench); acquisition additionally
  requires the operator's explicit ``--accept-non-commercial``, recorded in the metrics.

Fetching is idempotent and resumable at file granularity, and every fetcher sits behind ONE
seam (:class:`Fetcher`) so the offline suite injects fakes; the real implementations ride
``huggingface_hub`` (the existing ``[hub]`` extra) for Hub sources and stdlib ``urllib`` for
direct URLs.

Gate artifact: ``stages/s1-corpus/metrics.json`` + the corpus manifest rendered by the
existing ``benchmark/corpus/manifest.py`` (measured counts, not brochure numbers).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from latence_benchmark.campaign.stages import (
    CampaignContext,
    StageBlockedError,
    StageSpec,
    stage_dir,
)
from latence_benchmark.licensing import DatasetLicense, LicenseGateError, require_usable

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from types import ModuleType

STAGE_NAME = "s1-corpus"

#: Repo-relative root the raw acquisitions land under (gitignored; see .gitignore).
SOTA_CORPUS_DIR = "benchmark/sota-corpus"

LANE_SHARE_ALIKE_LOCAL = "share-alike-local"
LANE_STRICTLY_LOCAL = "strictly-local"
LANE_RESEARCH_ONLY = "research-only-local"

KIND_HF_SNAPSHOT = "hf-snapshot"
KIND_URL = "url"

#: The S1 ruling on UDA's HTML-backed subsets, recorded verbatim into the stage metrics.
UDA_HTML_RULING = (
    "uda: HTML subsets EXCLUDED (NqText -> wiki_nq, FetaTab -> wiki_feta) — no parser in "
    "the framework supports HTML input (S0 inventory), so acquiring them would gate the "
    "campaign on bytes nothing can consume. FinHybrid + PaperTab only; TAT/paper_text also "
    "skipped (not in the S1 spec)."
)


@dataclass(frozen=True)
class DatasetPlan:
    """One dataset's acquisition plan: where from, under which license, into which lane.

    :param name: Campaign dataset name (a
        :data:`~latence_benchmark.campaign.wizard.KNOWN_DATASETS` entry).
    :param lane: One of the module's ``LANE_*`` classifications.
    :param license: The separately verified corpus+toolkit license record the gate enforces.
    :param kind: ``hf-snapshot`` | ``url`` — selects the fetcher.
    :param repo_id: Hugging Face dataset repo id (``hf-snapshot`` only).
    :param revision: Full 40-hex commit SHA pin (``hf-snapshot`` only) — branches move.
    :param allow_patterns: Repo subset to fetch (``hf-snapshot``); empty = whole repo.
    :param urls: ``(filename, url)`` pairs (``url`` only), fetched with stdlib urllib,
        skip-if-exists per file.
    :param notes: Rulings and caveats carried into metrics + manifest.
    """

    name: str
    lane: str
    license: DatasetLicense
    kind: str
    repo_id: str = ""
    revision: str = ""
    allow_patterns: tuple[str, ...] = ()
    urls: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def provenance(self) -> str:
        """One line saying where the bytes come from, for metrics + manifest."""
        if self.kind == KIND_HF_SNAPSHOT:
            return f"hf:{self.repo_id}@{self.revision}"
        return "; ".join(url for _, url in self.urls)


def _no_upstream_toolkit() -> str:
    # The gate verifies corpus and toolkit licenses SEPARATELY (ADR-0012). These datasets
    # are scored by our own harness — no upstream evaluation code is executed or vendored —
    # so the toolkit half is recorded as exactly that, never as UNVERIFIED (which would be
    # a hard stop about code we do not run).
    return "not-applicable (scored by the latence harness; no upstream toolkit executed)"


#: The acquisition table. Lanes and handling per the S1 brief
#: (issues/wayfinder-e2e-sota-run/tickets/s1-…md); licenses + revisions verified against the
#: Hugging Face API on 2026-08-02 (license tag + pinned commit SHA read per repo).
CORPUS_PLAN: Mapping[str, DatasetPlan] = {
    "wiki2multihop": DatasetPlan(
        name="wiki2multihop",
        lane=LANE_SHARE_ALIKE_LOCAL,
        license=DatasetLicense(
            dataset_license="Apache-2.0 (dataset card); paragraph text CC-BY-SA Wikipedia-derived",
            code_license=_no_upstream_toolkit(),
            commercial_use=True,
            attribution_required=True,
            verified_on="2026-08-02",
            source_url="https://huggingface.co/datasets/xanhho/2WikiMultihopQA",
            notes=(
                "HF mirror of the Alab-NII/2wikimultihop release; card license tag "
                "apache-2.0 read from the Hub API. The paragraph text derives from "
                "Wikipedia (share-alike), so the corpus stays local."
            ),
        ),
        kind=KIND_HF_SNAPSHOT,
        repo_id="xanhho/2WikiMultihopQA",
        revision="612bc5039a457880d9e7d84c3b0a4cf154b70e4f",
        allow_patterns=("*.parquet", "README.md"),
        notes=("Gold evidence triples double as extraction gold (S3 synergy).",),
    ),
    "multihop_rag": DatasetPlan(
        name="multihop_rag",
        lane=LANE_STRICTLY_LOCAL,
        license=DatasetLicense(
            dataset_license="ODC-BY-1.0 (annotations); news article text under publisher copyright",
            code_license=_no_upstream_toolkit(),
            commercial_use=True,
            attribution_required=True,
            verified_on="2026-08-02",
            source_url="https://huggingface.co/datasets/yixuantt/MultiHopRAG",
            notes=(
                "Card license tag odc-by read from the Hub API. The 609-article news "
                "corpus text remains under its publishers' copyright: research fair-use "
                "posture, STRICTLY LOCAL, never redistributed or published."
            ),
        ),
        kind=KIND_HF_SNAPSHOT,
        repo_id="yixuantt/MultiHopRAG",
        revision="71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82",
        allow_patterns=("MultiHopRAG.json", "corpus.json", "README.md"),
    ),
    "ohr_bench": DatasetPlan(
        name="ohr_bench",
        lane=LANE_RESEARCH_ONLY,
        license=DatasetLicense(
            dataset_license="research-only (non-commercial)",
            code_license="Apache-2.0 (github.com/opendatalab/OHR-Bench toolkit; not executed)",
            commercial_use=False,
            attribution_required=True,
            verified_on="2026-08-02",
            source_url="https://github.com/opendatalab/OHR-Bench",
            notes=(
                "LICENSE CONFLICT: the HF card tag says cc-by-4.0, the project README "
                "says research-only. Resolved CONSERVATIVELY as research-only/keep-local "
                "(dataset-landscape.md S0 ruling); redistribution is blocked until "
                "clarified with opendatalab."
            ),
        ),
        kind=KIND_HF_SNAPSHOT,
        repo_id="opendatalab/OHR-Bench",
        revision="7f833e3eda9a571a9ea545a8f6d476fa1685033d",
        allow_patterns=("pdfs.zip", "OHR-Bench.parquet", "OHR-Bench_v2.parquet", "README.md"),
        notes=("The OCR anchor: per-OCR-pipeline retrieval numbers to compare S2 against.",),
    ),
    "vidoseek": DatasetPlan(
        name="vidoseek",
        lane=LANE_STRICTLY_LOCAL,
        license=DatasetLicense(
            dataset_license="Apache-2.0 (annotations); source-PDF copyright retained upstream",
            code_license=_no_upstream_toolkit(),
            commercial_use=True,
            attribution_required=True,
            verified_on="2026-08-02",
            source_url="https://huggingface.co/datasets/Qiuchen-Wang/ViDoSeek",
            notes=(
                "Card license tag apache-2.0 read from the Hub API (the brief's "
                "autumncc/ViDoSeek id 307-redirects here). PDF provenance is unverified "
                "upstream, so the documents stay strictly local."
            ),
        ),
        kind=KIND_HF_SNAPSHOT,
        repo_id="Qiuchen-Wang/ViDoSeek",
        revision="e91a92ba5f38690696c7e66be5c5474b54c6e791",
        allow_patterns=("vidoseek.json", "vidoseek_pdf_document.zip", "README.md"),
        notes=("ViDoSeek split only; the repo's SlideVQA files are not part of the S1 spec.",),
    ),
    "uda": DatasetPlan(
        name="uda",
        lane=LANE_SHARE_ALIKE_LOCAL,
        license=DatasetLicense(
            dataset_license="CC-BY-SA-4.0",
            code_license="Apache-2.0 (github.com/qinchuanhui/UDA-Benchmark toolkit; not executed)",
            commercial_use=True,
            attribution_required=True,
            verified_on="2026-08-02",
            source_url="https://huggingface.co/datasets/qinchuanhui/UDA-QA",
            notes=(
                "Card license tag cc-by-sa-4.0 read from the Hub API. Share-alike: the "
                "corpus stays local. FinHybrid + PaperTab subsets only (see the recorded "
                "HTML ruling)."
            ),
        ),
        kind=KIND_HF_SNAPSHOT,
        repo_id="qinchuanhui/UDA-QA",
        revision="d4367103fe8fe86b3bb76c66be8eafc4fb4117b2",
        allow_patterns=(
            "fin/*",
            "paper_tab/*",
            "src_doc_files/fin_docs.zip",
            "src_doc_files/paper_docs.zip",
            "extended_qa_info/fin_qa.json",
            "extended_qa_info/paper_tab_qa.json",
            "README.md",
        ),
        notes=(UDA_HTML_RULING,),
    ),
}


@dataclass(frozen=True)
class FetchResult:
    """What one dataset fetch actually did (resumability made visible)."""

    fetched: int
    skipped: int
    note: str = ""


class Fetcher(Protocol):
    """THE seam: materialise one dataset plan into ``dest``. Tests inject fakes here.

    Implementations must be idempotent and resumable at file granularity: a file that
    already exists at ``dest`` is skipped, never re-downloaded, never truncated.
    """

    def __call__(self, plan: DatasetPlan, dest: Path) -> FetchResult: ...


# -- real fetcher implementations ------------------------------------------------------------


def fetch_hf_snapshot(plan: DatasetPlan, dest: Path) -> FetchResult:
    """Materialise a pinned Hub snapshot into ``dest`` via ``huggingface_hub``.

    Rides the existing ``[hub]`` extra (lazy import, ADR-0016). ``snapshot_download`` is
    itself resumable at file granularity — files already present at the pinned revision are
    not re-fetched — which is exactly the contract the seam promises.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover — exercised via the message, not the import
        msg = (
            f"{plan.name}: remote acquisition needs the optional [hub] extra "
            "(`pip install latence-benchmark[hub]`)."
        )
        raise StageBlockedError(msg) from exc

    before = _count_files(dest)
    snapshot_download(
        repo_id=plan.repo_id,
        repo_type="dataset",
        revision=plan.revision,
        local_dir=str(dest),
        allow_patterns=list(plan.allow_patterns) or None,
    )
    after = _count_files(dest)
    return FetchResult(fetched=max(after - before, 0), skipped=before)


def fetch_urls(plan: DatasetPlan, dest: Path) -> FetchResult:
    """Fetch direct URLs with stdlib urllib; skip-if-exists per file, temp+rename per file."""
    fetched = 0
    skipped = 0
    dest.mkdir(parents=True, exist_ok=True)
    for filename, url in plan.urls:
        target = dest / filename
        if target.exists() and target.stat().st_size > 0:
            skipped += 1
            continue
        tmp = target.with_name(target.name + ".part")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            tmp.replace(target)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            msg = f"{plan.name}: failed to fetch {url}: {exc}"
            raise StageBlockedError(msg) from exc
        fetched += 1
    return FetchResult(fetched=fetched, skipped=skipped)


def default_fetchers() -> Mapping[str, Fetcher]:
    """The real fetcher per plan ``kind``. Tests build their own mapping instead."""
    return {
        KIND_HF_SNAPSHOT: fetch_hf_snapshot,
        KIND_URL: fetch_urls,
    }


def _count_files(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file())


# -- the license gate ------------------------------------------------------------------------


def license_gate(
    names: Iterable[str],
    *,
    accept_non_commercial: bool,
    plan: Mapping[str, DatasetPlan] = CORPUS_PLAN,
) -> list[DatasetPlan]:
    """Classify + clear every selected dataset BEFORE any byte moves.

    An unclassified dataset — one with no lane in the plan table — is refused outright:
    "we have not classified it" must never degrade into "fetch it and sort the license out
    later". Injectable ``plan`` so the refusal path is offline-testable.
    """
    cleared: list[DatasetPlan] = []
    for name in names:
        entry = plan.get(name)
        if entry is None:
            msg = (
                f"Dataset {name!r} has NO license-lane classification in the s1-corpus plan "
                "table. The gate runs before acquisition, so an unclassified dataset is "
                "never fetched — classify it (lane + verified DatasetLicense) in "
                "latence_benchmark.campaign.corpus_stage.CORPUS_PLAN first."
            )
            raise StageBlockedError(msg)
        try:
            require_usable(
                name, entry.license, accept_non_commercial=accept_non_commercial
            )
        except LicenseGateError as exc:
            raise StageBlockedError(str(exc)) from exc
        cleared.append(entry)
    return cleared


# -- the stage itself ------------------------------------------------------------------------


def _load_manifest_module(repo_root: Path) -> ModuleType:
    """Load the existing ``benchmark/corpus/manifest.py`` (reused, not re-implemented).

    It lives outside any installed package — it is repo tooling — so it is loaded from the
    checkout the campaign operates on. Blocking (not crashing) when absent keeps the
    failure inside the stage's own vocabulary.
    """
    path = repo_root / "benchmark" / "corpus" / "manifest.py"
    if not path.is_file():
        msg = (
            f"benchmark/corpus/manifest.py not found under {repo_root} — the corpus "
            "manifest is rendered by the existing repo tool, so the stage needs a "
            "latence-framework checkout as --repo-root."
        )
        raise StageBlockedError(msg)
    spec = importlib.util.spec_from_file_location("_latence_campaign_corpus_manifest", path)
    if spec is None or spec.loader is None:  # pragma: no cover — importlib internal failure
        msg = f"Could not load {path} as a module."
        raise StageBlockedError(msg)
    module = importlib.util.module_from_spec(spec)
    # Registered under its spec name BEFORE exec: the module defines dataclasses, and the
    # dataclass machinery resolves string annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dataset_root(ctx: CampaignContext, plan: DatasetPlan) -> Path:
    """Where a dataset's bytes live: its raw dir under ``benchmark/sota-corpus``."""
    return ctx.repo_root / SOTA_CORPUS_DIR / plan.name / "raw"


def _metrics_path(ctx: CampaignContext) -> Path:
    return stage_dir(ctx.campaign_root, STAGE_NAME) / "metrics.json"


def corpus_done_probe(ctx: CampaignContext) -> bool:
    """The filesystem's answer: metrics exist AND cover exactly the selected datasets.

    Comparing the selection (not just existence) means editing ``corpus.datasets`` in
    campaign.yaml genuinely re-opens the stage instead of riding an old gate artifact.
    """
    path = _metrics_path(ctx)
    if not path.is_file():
        return False
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    datasets = recorded.get("datasets")
    if not isinstance(datasets, dict):
        return False
    return set(datasets) == set(ctx.answers.datasets)


def corpus_progress_probe(ctx: CampaignContext) -> str:
    """Files landed so far under ``benchmark/sota-corpus`` — the honest growing number."""
    return f"{_count_files(ctx.repo_root / SOTA_CORPUS_DIR)} files under {SOTA_CORPUS_DIR}"


def run_corpus_stage(
    ctx: CampaignContext,
    fetchers: Mapping[str, Fetcher] | None = None,
) -> str:
    """Gate → fetch → measure → record. Returns the one-line operator note.

    :param fetchers: Fetcher per plan ``kind``; ``None`` uses the real implementations.
        The offline suite injects fakes here — the seam the whole stage is built around.
    """
    active = fetchers if fetchers is not None else default_fetchers()

    # 1. License gate, for the WHOLE selection, before any fetcher runs. A selection that
    #    is half-fetched and then refused would leave restricted bytes already on disk.
    cleared = license_gate(
        ctx.answers.datasets, accept_non_commercial=ctx.accept_non_commercial
    )

    # 2. Fetch (idempotent, per dataset).
    results: dict[str, FetchResult] = {}
    for plan in cleared:
        fetcher = active.get(plan.kind)
        if fetcher is None:
            msg = f"{plan.name}: no fetcher for plan kind {plan.kind!r}."
            raise StageBlockedError(msg)
        dest = dataset_root(ctx, plan)
        dest.mkdir(parents=True, exist_ok=True)
        results[plan.name] = fetcher(plan, dest)

    # 3. Measure what actually landed (never what the fetch claims) + render the manifest
    #    with the existing repo tool.
    manifest_mod = _load_manifest_module(ctx.repo_root)
    sources: list[Any] = []
    dataset_metrics: dict[str, Any] = {}
    for plan in cleared:
        root = dataset_root(ctx, plan)
        count, total, histogram = manifest_mod.scan_tree(root)
        sources.append(
            manifest_mod.SourceManifest(
                name=plan.name,
                licence=plan.license.dataset_license,
                # Every lane in the plan keeps the corpus on this machine, so no source
                # is publishable. The field stays: the manifest tool requires it.
                publishable=False,
                root=str(root),
                file_count=count,
                total_bytes=total,
                by_extension=dict(histogram),
                notes=(f"lane: {plan.lane}", f"provenance: {plan.provenance}", *plan.notes),
            )
        )
        result = results[plan.name]
        dataset_metrics[plan.name] = {
            "lane": plan.lane,
            "provenance": plan.provenance,
            "license": {
                "dataset": plan.license.dataset_license,
                "code": plan.license.code_license,
                "commercial_use": plan.license.commercial_use,
                "verified_on": plan.license.verified_on,
                "source_url": plan.license.source_url,
            },
            "fetched": result.fetched,
            "skipped": result.skipped,
            "file_count": count,
            "total_bytes": total,
            "notes": list(plan.notes),
        }

    out_dir = stage_dir(ctx.campaign_root, STAGE_NAME)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_mod.write_manifest(sources, out_dir)

    rulings = [UDA_HTML_RULING] if "uda" in results else []
    metrics = {
        "stage": STAGE_NAME,
        "accept_non_commercial": ctx.accept_non_commercial,
        "datasets": dataset_metrics,
        "rulings": rulings,
    }
    metrics_tmp = _metrics_path(ctx).with_name("metrics.json.tmp")
    metrics_tmp.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics_tmp.replace(_metrics_path(ctx))

    total_files = sum(m["file_count"] for m in dataset_metrics.values())
    total_bytes = sum(m["total_bytes"] for m in dataset_metrics.values())
    return (
        f"{len(cleared)} dataset(s) acquired/verified, {total_files} files, "
        f"{total_bytes} bytes; manifest + metrics under {out_dir}"
    )


CORPUS_STAGE = StageSpec(
    name=STAGE_NAME,
    title="Acquire the corpus behind the license gate; measure + record the manifest",
    done_probe=corpus_done_probe,
    progress_probe=corpus_progress_probe,
    run=run_corpus_stage,
)
