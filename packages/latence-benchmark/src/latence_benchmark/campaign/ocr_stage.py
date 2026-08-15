# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""Stage ``s2-ocr`` — LightOnOCR-2-1B on the pod turns the PDF corpus into markdown.

The stage runs CLIENT-side and treats the pod as a file server plus one long-lived job:

1. **Fin cap** — UDA-financial is capped to the top-150 most-questioned docs (counts from
   ``uda/raw/fin/test_*.parquet``, ties broken by ascending name; selection + QA coverage
   recorded in metrics). Everything else OCRs in full.
2. **Zip extraction** into ``benchmark/sota-corpus/<ds>/pdfs/`` — skip-if-exists per file,
   zip-slip guarded (:func:`latence_benchmark.campaign.ocr_lib.safe_zip_entry`).
3. **Remote orchestration behind ONE seam** (:class:`PodTransport`): push PDFs + the pod
   scripts, launch the SUPERVISED worker (``run_worker.sh``: up to 5 ``python -u``
   relaunches of the skip-if-exists worker, which itself crash-isolates rendering into
   disposable child processes) under ``nohup`` with a pid file, then POLL its atomically
   written ``state.json`` from here — monotone progress counter, 1800 s stall budget, an
   injected clock/sleep so the loop is offline-testable. A pod job found alive on entry is
   **re-entered, not relaunched** (that is what makes ``resume`` safe mid-flight).
4. On ``done``: pull ``markdown/`` per dataset, the worker's metrics and failures log,
   stop the vLLM daemon (the pod is shared — S3 needs the VRAM), pull the spot-check
   pages the worker retained (the streaming worker frees each doc's page images once its
   markdown is assembled — the /workspace quota bound — keeping only the deterministic
   12-doc sample under ``spot-img/``), and write the stage's own metrics atomically.

Any pod failure — refused launch, ``phase: failed``, a stalled counter — surfaces as
:class:`~latence_benchmark.campaign.stages.StageBlockedError` with the reason, and the
stage is resumable: the file manifests (local pdfs, remote pages, pulled markdown) mean a
rerun redoes only what is missing. Text-native datasets (musique, hotpotqa, wiki2multihop,
multihop_rag) are recorded as ``skipped_native`` and never leave the machine.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol, cast

from latence_benchmark.campaign import ocr_lib
from latence_benchmark.campaign.corpus_stage import SOTA_CORPUS_DIR
from latence_benchmark.campaign.stages import (
    CampaignContext,
    StageBlockedError,
    StageSpec,
    stage_dir,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

STAGE_NAME = "s2-ocr"

OCR_MODEL = "lightonai/LightOnOCR-2-1B"
#: Pinned model revision recorded in every provenance header (the HF main-branch commit
#: verified for the S0 diligence read on 2026-08-02).
OCR_MODEL_REVISION = "main"

#: Datasets whose corpus is PDFs and therefore passes through the OCR gate.
OCR_DATASETS: tuple[str, ...] = ("ohr_bench", "vidoseek", "uda")
#: Text-native datasets — already markdown/text; recorded as skipped, never converted.
NATIVE_DATASETS: tuple[str, ...] = ("musique", "hotpotqa", "wiki2multihop", "multihop_rag")

#: UDA-financial cap (Dennis, 2026-08-02): top-150 most-questioned docs ≈ 50% of fin QA.
FIN_CAP = 150

#: Pod-side layout. Everything the job touches lives under one root, so a single rsync
#: pulls the artifacts and a single ``rm -rf`` resets a broken run.
REMOTE_ROOT = "/workspace/latence-sota/s2"
#: The venv ``serve_ocr.sh`` creates; the worker runs under the same interpreter so
#: pypdfium2/pillow/openai are guaranteed present once serving is up.
WORKER_PYTHON = "/workspace/venvs/ocr/bin/python"

#: Client-side poll loop: progress must advance within the stall budget or the stage
#: blocks (the S1 lesson: never bare polling without freeze detection).
STALL_BUDGET_S = 1800.0
POLL_INTERVAL_S = 30.0


#: The launch: vLLM up (readiness-gated), then the SUPERVISOR (run_worker.sh) under
#: nohup with a pid file. The supervisor relaunches the worker (with ``python -u``, so
#: logs always flush) up to 5 times — the worker is skip-if-exists throughout, so a
#: relaunch continues; a single silent child death never costs the operator a manual
#: restart (the 2026-08-02 OOM field lesson).
_LAUNCH_CMD = (
    f"cd {REMOTE_ROOT} && bash pod/serve_ocr.sh && "
    f"(S2_PYTHON={WORKER_PYTHON} nohup bash pod/run_worker.sh {REMOTE_ROOT} "
    f">> supervisor.log 2>&1 & echo $! > worker.pid)"
)


# -- the transport seam ----------------------------------------------------------------------


class PodTransport(Protocol):
    """THE seam: every remote interaction of the stage goes through these three calls.

    Tests inject a fake pod behind it; the real implementation
    (:class:`SubprocessTransport`) wraps ``ssh``/``rsync`` subprocesses. Implementations
    must make ``push``/``pull`` raise :class:`StageBlockedError` on failure — a transfer
    that half-happened silently would corrupt the stage's resumability story.
    """

    def run(self, cmd: str) -> tuple[int, str]:
        """Execute ``cmd`` on the pod; return ``(exit_code, combined_output)``."""
        ...

    def push(self, local: Path, remote: str) -> None:
        """Copy a local file/dir INTO the remote directory (rsync no-trailing-slash)."""
        ...

    def pull(self, remote: str, local: Path) -> None:
        """Copy a remote path to ``local`` (a remote trailing ``/`` copies contents)."""
        ...


@dataclass(frozen=True)
class SshEndpoint:
    """A parsed ``pod_ssh`` command line: ssh argv, the target, and rsync's ``-e`` value."""

    argv: tuple[str, ...]
    target: str
    rsh: str


#: ssh options that consume a following value — needed to find the bare target token.
_SSH_VALUE_OPTS = frozenset({"-p", "-i", "-o", "-l", "-F", "-J", "-b", "-c", "-e", "-m"})


def parse_pod_ssh(pod_ssh: str) -> SshEndpoint:
    """Parse the config's full ``ssh root@IP -p PORT -i KEY`` line into an endpoint.

    The campaign config records the ssh invocation VERBATIM (it is what the operator
    types), so the stage derives both the exec argv and rsync's ``-e`` remote shell from
    the same string — one source of truth, no per-tool drift.
    """
    tokens = shlex.split(pod_ssh)
    if not tokens or Path(tokens[0]).name != "ssh":
        msg = (
            f"campaign.pod_ssh must be a full ssh command line (got {pod_ssh!r}); "
            "e.g. `ssh root@IP -p PORT -i ~/.ssh/key`."
        )
        raise StageBlockedError(msg)
    target = ""
    options: list[str] = []
    tail = iter(tokens[1:])
    for token in tail:
        if token in _SSH_VALUE_OPTS:
            value = next(tail, None)
            if value is None:
                msg = f"campaign.pod_ssh: option {token} is missing its value in {pod_ssh!r}."
                raise StageBlockedError(msg)
            options.extend([token, value])
        elif token.startswith("-"):
            options.append(token)
        elif not target:
            target = token
        else:
            msg = f"campaign.pod_ssh: unexpected extra argument {token!r} in {pod_ssh!r}."
            raise StageBlockedError(msg)
    if not target:
        msg = f"campaign.pod_ssh names no user@host target: {pod_ssh!r}."
        raise StageBlockedError(msg)
    rsh = " ".join(["ssh", *(shlex.quote(opt) for opt in options)])
    return SshEndpoint(argv=(tokens[0], *options, target), target=target, rsh=rsh)


class SubprocessTransport:
    """The real :class:`PodTransport`: ``ssh`` for commands, ``rsync -e ssh…`` for bytes."""

    def __init__(self, pod_ssh: str) -> None:
        self._endpoint = parse_pod_ssh(pod_ssh)

    def run(self, cmd: str) -> tuple[int, str]:
        # Generous timeout: the launch command installs vllm + downloads the model on the
        # first run. Everything else (cat state.json, stop script) returns in seconds.
        proc = subprocess.run(
            [*self._endpoint.argv, cmd],
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
        return proc.returncode, proc.stdout + proc.stderr

    def push(self, local: Path, remote: str) -> None:
        # --copy-links on PUSH only: campaign corpus dirs may be symlinks into shared
        # output trees (s3's `markdown` -> ../s2-output/markdown/<ds>); plain -l ships the
        # link itself, which dangles on the pod and silently yields a 0-document run (the
        # 2026-08 s3 field failure). Pull keeps -l as-is: what the pod wrote comes back
        # unchanged.
        self._rsync(str(local), f"{self._endpoint.target}:{remote}", dereference=True)

    def pull(self, remote: str, local: Path) -> None:
        self._rsync(f"{self._endpoint.target}:{remote}", str(local))

    def _rsync(self, source: str, dest: str, *, dereference: bool = False) -> None:
        # -a minus owner/group: RunPod's /workspace network FS refuses chown (rc=23) —
        # the exact failure the M6 campaign hit; -rlptz is -a without -og, plus
        # --no-owner/--no-group defensively for any -a-implying default.
        cmd = [
            "rsync", "-rlptz", "--no-owner", "--no-group", "--partial",
            *(("--copy-links",) if dereference else ()),
            "-e", self._endpoint.rsh, source, dest,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            tail = proc.stderr.strip().splitlines()[-3:]
            msg = f"rsync {source} -> {dest} failed (rc={proc.returncode}): {' | '.join(tail)}"
            raise StageBlockedError(msg)


# -- fin-cap selection -----------------------------------------------------------------------


@dataclass(frozen=True)
class FinCapSelection:
    """The recorded UDA-financial cap: which docs, and how much QA they cover."""

    selected: tuple[str, ...]
    qa_covered: int
    qa_total: int
    docs_total: int

    def as_metrics(self) -> dict[str, Any]:
        return {
            "cap": len(self.selected),
            "docs_total": self.docs_total,
            "qa_covered": self.qa_covered,
            "qa_total": self.qa_total,
            "selected": list(self.selected),
        }


def fin_cap_selection(fin_dir: Path, *, cap: int = FIN_CAP) -> FinCapSelection:
    """Count QA per ``doc_name`` across ``test_*.parquet`` and take the deterministic top.

    ``pyarrow`` is imported lazily (it is a dev/benchmark dependency, not a base one); a
    missing fin directory blocks with the fix rather than crashing.
    """
    files = sorted(fin_dir.glob("test_*.parquet"))
    if not files:
        msg = (
            f"fin cap: no test_*.parquet under {fin_dir} — run/approve s1-corpus first "
            "(the UDA acquisition provides the fin QA table)."
        )
        raise StageBlockedError(msg)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "fin cap: pyarrow is required to read the UDA fin QA parquet files."
        raise StageBlockedError(msg) from exc

    counts: dict[str, int] = {}
    for path in files:
        # ``cast`` rather than a type-ignore: pyarrow is dynamically typed in this repo's
        # mypy config (its annotations are partial), and an ignore would flip to "unused"
        # in environments where pyarrow is absent.
        column = cast("Any", pq).read_table(path, columns=["doc_name"]).column("doc_name")
        for name in column.to_pylist():
            counts[str(name)] = counts.get(str(name), 0) + 1
    selected = ocr_lib.select_capped_docs(counts, cap)
    covered, total = ocr_lib.qa_coverage(counts, selected)
    return FinCapSelection(
        selected=selected, qa_covered=covered, qa_total=total, docs_total=len(counts)
    )


# -- zip extraction --------------------------------------------------------------------------


def extract_zip(
    zip_path: Path, dest: Path, *, members: Collection[str] | None = None
) -> tuple[int, int]:
    """Extract ``zip_path`` into ``dest`` — skip-if-exists, temp+rename, zip-slip guarded.

    ``members`` restricts extraction to exactly those entry names (the fin cap); a member
    the archive does not contain blocks loudly — a silently thinner corpus would be
    measured downstream as if it were the full one. Returns ``(extracted, skipped)``.
    """
    if not zip_path.is_file():
        msg = f"expected archive {zip_path} — run/approve s1-corpus first."
        raise StageBlockedError(msg)
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        if members is not None:
            member_set = set(members)
            missing = sorted(member_set - set(names))
            if missing:
                preview = ", ".join(missing[:5])
                msg = (
                    f"{zip_path.name}: {len(missing)} selected member(s) absent from the "
                    f"archive (first: {preview}) — the fin selection and the archive "
                    "disagree; refusing a silently thinner corpus."
                )
                raise StageBlockedError(msg)
            names = [n for n in names if n in member_set]
        for name in names:
            if not ocr_lib.safe_zip_entry(name):
                msg = (
                    f"zip-slip: entry {name!r} in {zip_path.name} would escape the "
                    "extraction root; refusing the archive."
                )
                raise StageBlockedError(msg)
            target = dest.joinpath(*PurePosixPath(name).parts)
            if target.exists() and target.stat().st_size > 0:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            with archive.open(name) as src, tmp.open("wb") as out:
                shutil.copyfileobj(src, out)
            tmp.replace(target)
            extracted += 1
    return extracted, skipped


@dataclass(frozen=True)
class ZipJob:
    """One archive to extract for a dataset (repo-relative zip, optional member subset)."""

    dataset: str
    zip_relpath: str
    members: tuple[str, ...] | None = None


def zip_jobs(active: Sequence[str], fin_selected: Sequence[str]) -> tuple[ZipJob, ...]:
    """The extraction plan for the active OCR datasets (fin restricted to the cap)."""
    jobs: list[ZipJob] = []
    if "ohr_bench" in active:
        jobs.append(ZipJob(dataset="ohr_bench", zip_relpath="ohr_bench/raw/pdfs.zip"))
    if "vidoseek" in active:
        jobs.append(
            ZipJob(dataset="vidoseek", zip_relpath="vidoseek/raw/vidoseek_pdf_document.zip")
        )
    if "uda" in active:
        jobs.append(
            ZipJob(dataset="uda", zip_relpath="uda/raw/src_doc_files/paper_docs.zip")
        )
        jobs.append(
            ZipJob(
                dataset="uda",
                zip_relpath="uda/raw/src_doc_files/fin_docs.zip",
                members=tuple(f"fin_docs/{doc}.pdf" for doc in fin_selected),
            )
        )
    return tuple(jobs)


# -- the client-side poll loop ---------------------------------------------------------------


def _read_remote_state(transport: PodTransport) -> dict[str, Any] | None:
    rc, out = transport.run(f"cat {REMOTE_ROOT}/state.json")
    if rc != 0:
        return None
    try:
        state = json.loads(out)
    except ValueError:
        return None  # A torn read would be a bug pod-side (writes are atomic); treat as absent.
    return state if isinstance(state, dict) else None


def poll_pod_job(
    transport: PodTransport,
    *,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    stall_budget: float = STALL_BUDGET_S,
    interval: float = POLL_INTERVAL_S,
) -> dict[str, Any]:
    """Poll the pod's ``state.json`` until ``phase: done``; stall or failure blocks.

    Progress is the worker's MONOTONE counter (pages rendered + pages OCR'd + docs
    assembled). An unreadable state file counts as no progress — the stall budget, not the
    read error, decides when that becomes fatal, so one dropped ssh round-trip never kills
    a 10-hour run. The clock/sleep seams make the whole loop offline-testable.
    """
    best_progress = -1
    last_advance = clock()
    while True:
        state = _read_remote_state(transport)
        if state is not None:
            phase = str(state.get("phase", ""))
            if phase == "done":
                return state
            if phase == "failed":
                error = str(state.get("error", "unreported error"))
                msg = f"pod worker failed during {state.get('failed_phase', 'run')}: {error}"
                raise StageBlockedError(msg)
            progress = int(state.get("progress", 0))
            if progress > best_progress:
                best_progress = progress
                last_advance = clock()
        if clock() - last_advance > stall_budget:
            msg = (
                f"pod job stalled: no progress beyond {max(best_progress, 0)} for more "
                f"than {stall_budget:.0f}s (see `bash pod/podcheck_s2.sh` on the pod). "
                "Fix the cause, then `latence-campaign resume` re-enters the poll."
            )
            raise StageBlockedError(msg)
        sleep(interval)


# -- probes ----------------------------------------------------------------------------------


def _active_ocr_datasets(ctx: CampaignContext) -> list[str]:
    return [d for d in OCR_DATASETS if d in ctx.answers.datasets]


def _metrics_path(ctx: CampaignContext) -> Path:
    return stage_dir(ctx.campaign_root, STAGE_NAME) / "metrics.json"


def _markdown_dir(ctx: CampaignContext, dataset: str) -> Path:
    return ctx.repo_root / SOTA_CORPUS_DIR / dataset / "markdown"


def _write_metrics(ctx: CampaignContext, metrics: Mapping[str, Any]) -> None:
    tmp = _metrics_path(ctx).with_name("metrics.json.tmp")
    tmp.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(_metrics_path(ctx))


def ocr_done_probe(ctx: CampaignContext) -> bool:
    """Metrics cover exactly the selected OCR datasets AND their markdown landed locally.

    Both halves matter: metrics without markdown is a claim without bytes; markdown
    without metrics is a run that died before the gate artifact. Editing the corpus
    selection re-opens the stage, same as s1.
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
    active = _active_ocr_datasets(ctx)
    if set(datasets) != set(active):
        return False
    return all(any(_markdown_dir(ctx, ds).glob("*.md")) for ds in active)


def ocr_progress_probe(ctx: CampaignContext) -> str:
    """Markdown docs landed locally so far — the byte-backed growing number."""
    count = sum(
        1 for ds in _active_ocr_datasets(ctx) for _ in _markdown_dir(ctx, ds).glob("*.md")
    )
    return f"{count} markdown doc(s) under {SOTA_CORPUS_DIR}/*/markdown"


# -- the stage run ---------------------------------------------------------------------------


def _push_inputs(ctx: CampaignContext, transport: PodTransport, active: Sequence[str]) -> None:
    rc, out = transport.run(f"mkdir -p {REMOTE_ROOT}/input {REMOTE_ROOT}/pod")
    if rc != 0:
        msg = f"pod refused mkdir under {REMOTE_ROOT} (rc={rc}): {out.strip()[:300]}"
        raise StageBlockedError(msg)
    pod_dir = ctx.repo_root / "benchmark" / "sota-campaign" / "pod"
    if not pod_dir.is_dir():
        msg = f"pod scripts not found at {pod_dir} — the repo checkout is incomplete."
        raise StageBlockedError(msg)
    for script in sorted(pod_dir.iterdir()):
        if script.is_file():
            transport.push(script, f"{REMOTE_ROOT}/pod/")
    # The worker imports the SAME ocr_lib the offline suite proved — pushed from the
    # installed package, so there is no second copy to drift.
    lib_file = ocr_lib.__file__
    if lib_file is None:  # pragma: no cover — a namespace-package accident
        msg = "ocr_lib has no __file__; cannot push the shared library to the pod."
        raise StageBlockedError(msg)
    transport.push(Path(lib_file), f"{REMOTE_ROOT}/pod/")
    for dataset in active:
        pdfs = ctx.repo_root / SOTA_CORPUS_DIR / dataset / "pdfs"
        if not pdfs.is_dir() or not any(pdfs.rglob("*.pdf")):
            msg = f"{dataset}: no PDFs under {pdfs} after extraction — nothing to push."
            raise StageBlockedError(msg)
        rc, out = transport.run(f"mkdir -p {REMOTE_ROOT}/input/{dataset}")
        if rc != 0:
            msg = f"pod refused mkdir input/{dataset} (rc={rc}): {out.strip()[:300]}"
            raise StageBlockedError(msg)
        transport.push(pdfs, f"{REMOTE_ROOT}/input/{dataset}/")


def _launch_worker(transport: PodTransport) -> None:
    rc, out = transport.run(_LAUNCH_CMD)
    if rc != 0:
        msg = f"pod launch failed (rc={rc}): {out.strip()[-500:]}"
        raise StageBlockedError(msg)


def _pull_results(
    ctx: CampaignContext, transport: PodTransport, active: Sequence[str], out_dir: Path
) -> dict[str, Any]:
    for dataset in active:
        local = _markdown_dir(ctx, dataset)
        local.mkdir(parents=True, exist_ok=True)
        transport.pull(f"{REMOTE_ROOT}/markdown/{dataset}/", local)
    transport.pull(f"{REMOTE_ROOT}/metrics.json", out_dir / "pod-metrics.json")
    transport.pull(f"{REMOTE_ROOT}/failures.jsonl", out_dir / "failures.jsonl")
    try:
        pod_metrics = json.loads((out_dir / "pod-metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"pulled pod metrics are unreadable: {exc}"
        raise StageBlockedError(msg) from exc
    if not isinstance(pod_metrics, dict):
        msg = "pulled pod metrics are not a JSON object."
        raise StageBlockedError(msg)
    return pod_metrics


def _pull_spot_checks(
    transport: PodTransport, pod_metrics: Mapping[str, Any], out_dir: Path
) -> list[dict[str, Any]]:
    """Pull the worker-RETAINED spot pages (image + raw page markdown) for the gate.

    The streaming worker deletes each doc's page images the moment its markdown is
    assembled (the /workspace quota bound), retaining only the deterministically sampled
    spot pages under ``spot-img/`` and recording them in its metrics ``spot_check`` list
    — so the client pulls exactly what the worker kept, never a page that is gone.
    """
    raw = pod_metrics.get("spot_check")
    entries = raw if isinstance(raw, list) else []
    spot_dir = out_dir / "spot-check"
    spot_dir.mkdir(parents=True, exist_ok=True)
    recorded: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        doc = str(entry.get("doc", ""))
        page = int(entry.get("page", 0))
        if not doc or page < 1:
            continue
        stem = f"{doc.replace('/', '__')}__{page:04d}"
        image_local = spot_dir / f"{stem}.jpg"
        markdown_local = spot_dir / f"{stem}.md"
        transport.pull(f"{REMOTE_ROOT}/spot-img/{doc}/{page:04d}.jpg", image_local)
        transport.pull(f"{REMOTE_ROOT}/pages/{doc}/{page:04d}.md", markdown_local)
        recorded.append(
            {
                "doc": doc,
                "page": page,
                "image": str(image_local),
                "markdown": str(markdown_local),
            }
        )
    return recorded


def run_ocr_stage(
    ctx: CampaignContext,
    transport: PodTransport | None = None,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Fin-cap → extract → push → launch/re-enter → poll → pull → metrics.

    :param transport: The pod seam; ``None`` builds the real ssh/rsync transport from the
        campaign config's ``pod_ssh``. Tests inject fakes here.
    :param clock: Monotonic-seconds source for the stall budget (seam, tests fake it).
    :param sleep: Poll-interval sleeper (seam, tests fake it).
    """
    active = _active_ocr_datasets(ctx)
    skipped_native = [d for d in NATIVE_DATASETS if d in ctx.answers.datasets]
    out_dir = stage_dir(ctx.campaign_root, STAGE_NAME)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not active:
        # A trimmed, text-only selection is legal (the wizard invites it for fast
        # iteration): the stage completes as a recorded no-op — never a fake conversion.
        _write_metrics(
            ctx,
            {
                "stage": STAGE_NAME,
                "model": OCR_MODEL,
                "model_revision": OCR_MODEL_REVISION,
                "fin_cap": None,
                "skipped_native": skipped_native,
                "datasets": {},
                "pod": None,
                "spot_check": [],
            },
        )
        return (
            "no PDF dataset in the corpus selection — recorded "
            f"{len(skipped_native)} text-native dataset(s) as skipped, nothing to OCR."
        )

    # 1. Fin cap (deterministic selection, recorded before any byte moves).
    fin: FinCapSelection | None = None
    if "uda" in active:
        fin = fin_cap_selection(ctx.repo_root / SOTA_CORPUS_DIR / "uda" / "raw" / "fin")

    # 2. Local zip extraction (skip-if-exists; blocked on zip-slip or a thin archive).
    extraction: dict[str, dict[str, int]] = {}
    for job in zip_jobs(active, fin.selected if fin else ()):
        dest = ctx.repo_root / SOTA_CORPUS_DIR / job.dataset / "pdfs"
        extracted, skipped = extract_zip(
            ctx.repo_root / SOTA_CORPUS_DIR / job.zip_relpath, dest, members=job.members
        )
        entry = extraction.setdefault(job.dataset, {"extracted": 0, "skipped": 0})
        entry["extracted"] += extracted
        entry["skipped"] += skipped
    for dataset, entry in extraction.items():
        pdfs = ctx.repo_root / SOTA_CORPUS_DIR / dataset / "pdfs"
        entry["pdfs"] = sum(1 for _ in pdfs.rglob("*.pdf"))

    # 3. The transport (built here, late, so the offline steps above never need ssh).
    if transport is None:
        if not ctx.answers.pod_ssh:
            msg = (
                "s2-ocr runs on the GPU pod but campaign.pod_ssh is empty — set it in "
                "campaign.yaml (the full `ssh root@IP -p PORT -i KEY` line)."
            )
            raise StageBlockedError(msg)
        transport = SubprocessTransport(ctx.answers.pod_ssh)

    # 4. Launch, or RE-ENTER a job already in flight (the resume contract).
    remote_state = _read_remote_state(transport)
    resumed = remote_state is not None
    if remote_state is None:
        _push_inputs(ctx, transport, active)
        _launch_worker(transport)

    # 5. Poll until done (stall budget, monotone progress), unless already done.
    if remote_state is None or remote_state.get("phase") != "done":
        remote_state = poll_pod_job(transport, clock=clock, sleep=sleep)

    # 6. Pull markdown + worker metrics + failures; free the shared GPU.
    pod_metrics = _pull_results(ctx, transport, active, out_dir)
    stop_rc, _stop_out = transport.run(f"bash {REMOTE_ROOT}/pod/stop_ocr.sh")

    # 7. Spot-check samples — the pages the worker retained before freeing images.
    spot_check = _pull_spot_checks(transport, pod_metrics, out_dir)

    # 8. Stage metrics: the worker's numbers PLUS what measurably landed locally.
    datasets_metrics: dict[str, Any] = {}
    per_dataset = pod_metrics.get("datasets")
    for dataset in active:
        worker = per_dataset.get(dataset, {}) if isinstance(per_dataset, dict) else {}
        landed = sum(1 for _ in _markdown_dir(ctx, dataset).glob("*.md"))
        datasets_metrics[dataset] = {
            **(worker if isinstance(worker, dict) else {}),
            "markdown_docs_local": landed,
            "extraction": extraction.get(dataset, {}),
        }

    _write_metrics(
        ctx,
        {
            "stage": STAGE_NAME,
            "model": OCR_MODEL,
            "model_revision": OCR_MODEL_REVISION,
            "fin_cap": fin.as_metrics() if fin else None,
            "skipped_native": skipped_native,
            "datasets": datasets_metrics,
            "pod": {
                "remote_root": REMOTE_ROOT,
                "resumed": resumed,
                "vllm_stopped": stop_rc == 0,
                "final_state": remote_state,
            },
            "spot_check": spot_check,
        },
    )

    total_docs = sum(m["markdown_docs_local"] for m in datasets_metrics.values())
    return (
        f"{total_docs} markdown doc(s) across {len(active)} dataset(s); "
        f"{len(spot_check)} spot-check page(s) under {out_dir / 'spot-check'}"
    )


OCR_STAGE = StageSpec(
    name=STAGE_NAME,
    title="OCR the PDF corpus to markdown on the pod (LightOnOCR-2-1B, fin capped)",
    done_probe=ocr_done_probe,
    progress_probe=ocr_progress_probe,
    run=run_ocr_stage,
)
