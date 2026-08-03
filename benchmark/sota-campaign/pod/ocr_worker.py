#!/usr/bin/env python3
"""ocr_worker.py — the pod-side S2 job, streaming PER DOC: render -> OCR -> assemble -> free.

Runs on the pod under ``/workspace/venvs/ocr/bin/python -u`` (serve_ocr.sh installs the
deps: pypdfium2, pillow, openai; everything else is stdlib), supervised by
``run_worker.sh`` (up to 5 relaunches — every step below is skip-if-exists, so a relaunch
continues instead of redoing). The pure logic — HTML-table normalization, repetition
detection, manifest planning, spot sampling, provenance format — is imported from
``ocr_lib.py``, which the campaign stage pushes NEXT TO this file from the unit-tested
``latence_benchmark.campaign.ocr_lib`` module: the pod executes the exact bytes the
offline suite proved.

Layout under ``--root`` (default /workspace/latence-sota/s2)::

    input/<dataset>/pdfs/**.pdf     # pushed by the stage
    pages-img/<dataset>/<doc_id>/<page:04d>.jpg   # TRANSIENT: deleted per doc (below)
    pages/<dataset>/<doc_id>/<page:04d>.md        # OCR pass (skip-if-exists, tiny, kept)
    pages/<dataset>/<doc_id>/.pages               # sidecar: doc fully rendered, N pages
    pages/<dataset>/<doc_id>/.stats               # sidecar: assembly stats (metrics truth)
    markdown/<dataset>/<doc_id>.md  # normalized + assembled, provenance header
    spot-img/<dataset>/<doc_id>/<page:04d>.jpg    # the retained spot-check evidence
    state.json                      # atomic (temp+rename), rewritten every 30s
    metrics.json                    # written at the end of the run
    failures.jsonl                  # append-only failure log

**Streaming is a QUOTA constraint, not a style choice** (second field lesson, 2026-08-03:
/workspace has a ~70GB quota; render-all-then-OCR-all held 15GB of page images at 36,430
pages and would have breached mid-run). The pipeline is per doc: render the doc's pages
-> OCR them -> assemble ``markdown/<doc>.md`` -> DELETE the doc's page images. The 15GB
backlog of already-rendered docs only SHRINKS: each is OCR'd, assembled, and freed in
turn. Before deletion, the deterministically sampled spot-check page (12 docs, one page
each — ``ocr_lib.sample_spot_docs``/``spot_page_for``) is moved to ``spot-img/``.

**Docs are pipelined ACROSS each other** (third field lesson: strictly per-doc streaming
starved the server — docs average 7-20 pages, so within-doc concurrency never filled the
vLLM batch and render/assemble gaps added dead air; observed 1.18 pages/s with the queue
at "Running: 1-14"). ``N_DOCS_IN_FLIGHT`` doc lifecycles run concurrently, all sharing
ONE bounded OCR request pool of ``GLOBAL_CONCURRENCY`` — the global cap the server
actually sees — while ``RENDER_WORKERS`` render children keep the window fed. Each
knob is env-overridable (``S2_DOCS_IN_FLIGHT`` / ``S2_GLOBAL_CONCURRENCY`` /
``S2_RENDER_WORKERS``; the 16GB pod runs 2/12, the 96GB pod the 8/64 defaults). The disk
high-water above the input PDFs stays bounded: images exist only for in-flight plus
render-ahead docs — at most ``N_DOCS_IN_FLIGHT + RENDER_WORKERS + 1`` docs (~13 docs
~= ~650MB worst case at 260 pages/doc).

**Render stays crash-isolated** (first field lesson: pypdfium2 leaks RSS on some
malformed/huge PDFs until the container cap OOM-kills the process — silently). Each doc
renders in a disposable child subprocess (``--render-batch``); a leak dies with the
child, a dead child logs the doc to failures.jsonl and the run continues. The parent
never imports pdfium.

**Evidence on death**: faulthandler, ``-u`` logging, SIGTERM -> orderly shutdown, an
atexit deathbed stamping ``state.json`` ``phase: failed`` on any exit short of done;
parent OOM is covered by the supervisor relaunch loop and, last, the client's stall
budget.

``state.json``'s ``progress`` counter is a CUMULATIVE function of the filesystem (pages
rendered + pages OCR'd + docs assembled, counting pre-existing work exactly once per
launch — and ONLY filesystem-backed quantities: failure/retry counters deliberately do
not feed it, because they do not re-accumulate), so it is monotone ACROSS relaunches — a
supervisor restart never dips below the campaign client's high-water mark and therefore
never manufactures a false stall.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import contextlib
import faulthandler
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ocr_lib  # noqa: E402  (the stage pushes it next to this file)


def _env_int(name: str, default: int) -> int:
    """An env-overridable positive integer knob; junk or absence yields the default."""
    try:
        return max(int(os.environ[name]), 1)
    except (KeyError, ValueError):
        return default


MODEL = "lightonai/LightOnOCR-2-1B"
MODEL_REVISION = "main"
PORT = 8044
#: Doc lifecycles running concurrently (render-wait -> OCR -> assemble -> free each).
#: 16GB pod: S2_DOCS_IN_FLIGHT=2; 96GB pod: the default 8.
N_DOCS_IN_FLIGHT = _env_int("S2_DOCS_IN_FLIGHT", 8)
#: THE global cap on in-flight OCR requests, shared by every doc lifecycle — this is
#: the number the vLLM server actually sees. 16GB pod: S2_GLOBAL_CONCURRENCY=12.
GLOBAL_CONCURRENCY = _env_int("S2_GLOBAL_CONCURRENCY", 64)
#: Parallel render child processes keeping the in-flight window fed (CPU-bound).
RENDER_WORKERS = _env_int("S2_RENDER_WORKERS", 4)
TEMPERATURE = 0.2
TOP_P = 0.9
MAX_TOKENS = 4096
RENDER_DPI = 200
LONGEST_EDGE = 1540
MAX_MEGAPIXELS = 25.0
JPEG_QUALITY = 90
STATE_INTERVAL_S = 30.0
REQUEST_TIMEOUT_S = 300.0
#: Render-complete sidecar. Lives under pages/ (NOT pages-img/) so it SURVIVES the
#: per-doc image deletion — it is the durable "this doc needs no re-render" marker.
PAGES_SIDECAR = ".pages"
#: Assembly-stats sidecar (tables normalized/kept-raw, pages missing): keeps relaunch
#: metrics truthful for docs whose images are long gone.
STATS_SIDECAR = ".stats"

_FAILLOG_LOCK = threading.Lock()

#: The ONLY fields that feed ``progress``: each is filesystem-backed, so it re-counts to
#: the identical value on a relaunch. Failure/retry counters deliberately do NOT feed it
#: (they are not re-derivable from disk — including them would make a relaunch's progress
#: dip below the campaign client's high-water mark and risk a false stall).
_PROGRESS_FIELDS = frozenset({"pages_rendered", "pages_done", "docs_assembled"})


@dataclass
class State:
    """The polled contract with the campaign CLI. ``progress`` is MONOTONE by design."""

    root: Path
    phase: str = "starting"
    progress: int = 0
    pages_total: int = 0
    pages_rendered: int = 0
    pages_done: int = 0
    docs_assembled: int = 0
    failures: int = 0
    repetition_retries: int = 0
    error: str = ""
    failed_phase: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def bump(self, **deltas: int) -> None:
        with self.lock:
            for key, delta in deltas.items():
                setattr(self, key, getattr(self, key) + delta)
                if key in _PROGRESS_FIELDS:
                    self.progress += delta

    def write(self) -> None:
        with self.lock:
            payload = {
                "phase": self.phase,
                "progress": self.progress,
                "pages_total": self.pages_total,
                "pages_rendered": self.pages_rendered,
                "pages_done": self.pages_done,
                "docs_assembled": self.docs_assembled,
                "failures": self.failures,
                "repetition_retries": self.repetition_retries,
                "error": self.error,
                "failed_phase": self.failed_phase,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        path = self.root / "state.json"
        tmp = path.with_name("state.json.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)  # atomic: a reader never sees a torn state


def _state_writer_loop(state: State, stop: threading.Event) -> None:
    while not stop.wait(STATE_INTERVAL_S):
        state.write()


def _deathbed(state: State) -> None:
    """atexit guard: an exit that never reached done/failed still leaves evidence."""
    if state.phase not in ("done", "failed"):
        with state.lock:
            state.failed_phase = state.phase
            state.phase = "failed"
            state.error = "worker exited without reaching done (see worker.log)"
        state.write()


def log_failure(root: Path, entry: dict[str, object]) -> None:
    entry = {**entry, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with _FAILLOG_LOCK, (root / "failures.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(entry) + "\n")


# -- discovery -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Doc:
    dataset: str
    doc_id: str
    pdf: Path

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.doc_id}"


#: What one finished doc lifecycle hands back to the harvester:
#: (doc, page_count | None on render failure, stats deltas, spot record | None).
DocResult = tuple[Doc, "int | None", "dict[str, int]", "dict[str, object] | None"]


def discover_docs(root: Path) -> list[Doc]:
    docs: list[Doc] = []
    input_dir = root / "input"
    for dataset_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        pdf_root = dataset_dir / "pdfs"
        for pdf in sorted(pdf_root.rglob("*.pdf")):
            rel = pdf.relative_to(pdf_root)
            doc_id = "__".join(rel.with_suffix("").parts)
            docs.append(Doc(dataset=dataset_dir.name, doc_id=doc_id, pdf=pdf))
    return docs


# -- render step (CHILD process; pypdfium2, 200 DPI, longest edge 1540, JPEG) ----------------


def render_doc(root: Path, doc: Doc) -> int:
    """Render every page of ``doc`` to JPEG, skip-if-exists; write the ``.pages`` sidecar.

    Runs ONLY inside a render child. Pages, bitmaps and the document handle are closed
    explicitly (defensive memory bound — pdfium leaks on some inputs regardless, which is
    exactly why the child itself is disposable). Returns the page count.
    """
    import pypdfium2 as pdfium

    out_dir = root / "pages-img" / doc.dataset / doc.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(doc.pdf)
    try:
        page_count = len(pdf)
        for index in range(page_count):
            target = out_dir / f"{index + 1:04d}.jpg"
            if target.exists() and target.stat().st_size > 0:
                continue
            page = pdf[index]
            bitmap = None
            try:
                width_pt, height_pt = page.get_size()
                scale = RENDER_DPI / 72.0
                # Hard cap 25 MP/page (the rasterizer discipline the framework uses).
                megapixels = (width_pt * scale) * (height_pt * scale) / 1e6
                if megapixels > MAX_MEGAPIXELS:
                    scale *= (MAX_MEGAPIXELS / megapixels) ** 0.5
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil().convert("RGB")
            finally:
                for handle in (bitmap, page):
                    if handle is not None:
                        with contextlib.suppress(Exception):
                            handle.close()
            if max(image.size) > LONGEST_EDGE:
                image.thumbnail((LONGEST_EDGE, LONGEST_EDGE))
            tmp = target.with_name(target.name + ".part")
            image.save(tmp, format="JPEG", quality=JPEG_QUALITY)
            os.replace(tmp, target)
        sidecar = _sidecar_path(root, doc)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar_tmp = sidecar.with_name(sidecar.name + ".tmp")
        sidecar_tmp.write_text(f"{page_count}\n", encoding="utf-8")
        os.replace(sidecar_tmp, sidecar)
    finally:
        with contextlib.suppress(Exception):
            pdf.close()
    return page_count


def render_batch(root: Path, manifest_path: Path) -> int:
    """CHILD entrypoint (``--render-batch``): render the manifested docs sequentially."""
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in entries:
        doc = Doc(dataset=entry["dataset"], doc_id=entry["doc_id"], pdf=Path(entry["pdf"]))
        render_doc(root, doc)
    return 0


def _sidecar_path(root: Path, doc: Doc) -> Path:
    return root / "pages" / doc.key / PAGES_SIDECAR


def read_render_sidecar(root: Path, doc: Doc) -> int | None:
    """The doc's page count if (and only if) its render step completed; else ``None``.

    Honors the legacy location too (``pages-img/<doc>/.pages``, written by the previous
    worker generation) — a legacy sidecar's doc re-renders nothing, and the durable copy
    (under ``pages/``, which survives image cleanup) is written on collection.
    """
    for path in (_sidecar_path(root, doc), root / "pages-img" / doc.key / PAGES_SIDECAR):
        if not path.is_file():
            continue
        try:
            count = int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            continue
        durable = _sidecar_path(root, doc)
        if path != durable:  # migrate: the durable copy must survive image cleanup
            durable.parent.mkdir(parents=True, exist_ok=True)
            tmp = durable.with_name(durable.name + ".tmp")
            tmp.write_text(f"{count}\n", encoding="utf-8")
            os.replace(tmp, durable)
        return count
    return None


def _spawn_render_child(root: Path, docs: list[Doc]) -> bool:
    """Render ``docs`` in a disposable child process; True iff it exited cleanly."""
    manifest = [{"dataset": d.dataset, "doc_id": d.doc_id, "pdf": str(d.pdf)} for d in docs]
    tmp_dir = root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_dir / f"render-{os.getpid()}-{id(docs)}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--root",
                str(root),
                "--render-batch",
                str(manifest_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
            print(
                f"[render] child rc={proc.returncode} for {len(docs)} doc(s) "
                f"(first: {docs[0].key}): " + " | ".join(tail),
                flush=True,
            )
        return proc.returncode == 0
    finally:
        manifest_path.unlink(missing_ok=True)


def ensure_rendered(root: Path, doc: Doc, state: State) -> int | None:
    """Sidecar-or-render: the doc's page count, or ``None`` if its render child died.

    The 15GB backlog case lands here naturally: a backlog doc has no sidecar, so its
    child opens the PDF, finds every page image already present (skip-if-exists), writes
    the sidecar and exits — a cheap verification pass, not redone work.
    """
    count = read_render_sidecar(root, doc)
    if count is not None:
        return count
    if not _spawn_render_child(root, [doc]):
        log_failure(
            root,
            {"doc": doc.key, "kind": "render",
             "error": "render child died for this doc; doc skipped"},
        )
        state.bump(failures=1)
        return None
    count = read_render_sidecar(root, doc)
    if count is None:
        log_failure(
            root,
            {"doc": doc.key, "kind": "render",
             "error": "render child exited cleanly but left no completion sidecar"},
        )
        state.bump(failures=1)
    return count


# -- OCR step (per-page OpenAI-compatible requests against the local vLLM) -------------------


def _page_image_message(image_path: Path) -> list[dict[str, object]]:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            ],
        }
    ]


def ocr_page(client: object, root: Path, doc: Doc, page: int, state: State) -> None:
    """OCR one page image to ``pages/<...>/NNNN.md`` with the single repetition retry."""
    image = root / "pages-img" / doc.key / f"{page:04d}.jpg"
    target = root / "pages" / doc.key / f"{page:04d}.md"
    target.parent.mkdir(parents=True, exist_ok=True)

    def request() -> str:
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=MODEL,
            messages=_page_image_message(image),
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            timeout=REQUEST_TIMEOUT_S,
        )
        return str(response.choices[0].message.content or "")

    try:
        text = request()
        if ocr_lib.has_repetition(text):
            state.bump(repetition_retries=1)
            text = request()
            if ocr_lib.has_repetition(text):
                # Keep the second attempt (some text beats none) but log it honestly.
                log_failure(
                    root,
                    {"doc": doc.key, "page": page, "kind": "repetition",
                     "error": "still repetitive after one retry"},
                )
                state.bump(failures=1)
    except Exception as exc:  # HTTP/timeout/decode — the page is logged, never retried hot.
        log_failure(root, {"doc": doc.key, "page": page, "kind": "http", "error": str(exc)})
        state.bump(failures=1)
        return
    tmp = target.with_name(target.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)
    state.bump(pages_done=1)


# -- normalize + assemble + free -------------------------------------------------------------


def normalize_pages(root: Path, doc: Doc, page_count: int) -> tuple[list[str], int, int, int]:
    """Per-page table normalization; returns ``(pages_md, converted, kept_raw, missing)``.

    A page whose OCR permanently failed is represented by an explicit placeholder
    comment — visible, never silently absent.
    """
    pages_dir = root / "pages" / doc.key
    converted = 0
    kept_raw = 0
    missing = 0
    pages_md: list[str] = []
    for page in range(1, page_count + 1):
        path = pages_dir / ocr_lib.page_filename(page)
        if not path.is_file():
            missing += 1
            pages_md.append(f"<!-- page {page} failed OCR; see failures.jsonl -->")
            continue
        result = ocr_lib.normalize_html_tables(path.read_text(encoding="utf-8"))
        converted += result.converted
        kept_raw += result.kept_raw
        pages_md.append(result.text)
    return pages_md, converted, kept_raw, missing


def read_assembly_stats(root: Path, doc: Doc) -> dict[str, int] | None:
    path = root / "pages" / doc.key / STATS_SIDECAR
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {
            k: int(raw[k])
            for k in ("tables_normalized", "tables_kept_raw", "pages_missing")
        }
    except (ValueError, KeyError, TypeError):
        return None


def assemble_doc(root: Path, doc: Doc, page_count: int, state: State) -> dict[str, int]:
    """Normalize + write ``markdown/<ds>/<doc_id>.md`` + persist the stats sidecar."""
    pages_md, converted, kept_raw, missing = normalize_pages(root, doc, page_count)
    target = root / "markdown" / doc.dataset / f"{doc.doc_id}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    tmp.write_text(
        ocr_lib.assemble_document(
            pages_md,
            doc_id=doc.doc_id,
            dataset=doc.dataset,
            model=MODEL,
            revision=MODEL_REVISION,
        ),
        encoding="utf-8",
    )
    stats = {"tables_normalized": converted, "tables_kept_raw": kept_raw,
             "pages_missing": missing}
    stats_path = root / "pages" / doc.key / STATS_SIDECAR
    stats_tmp = stats_path.with_name(stats_path.name + ".tmp")
    stats_tmp.write_text(json.dumps(stats) + "\n", encoding="utf-8")
    # Markdown first, stats second: a crash between the two re-derives on relaunch.
    os.replace(tmp, target)
    os.replace(stats_tmp, stats_path)
    state.bump(docs_assembled=1)  # progress tick per assembled doc (monotone counter)
    return stats


def cleanup_doc_images(root: Path, doc: Doc, spot_page: int | None) -> bool:
    """Free the doc's page images — THE quota bound — retaining the one spot page.

    Called only after ``markdown/<doc>.md`` exists. Returns True when the spot page is
    retained under ``spot-img/`` (now, or already by an earlier pass).
    """
    img_dir = root / "pages-img" / doc.key
    retained = False
    if spot_page is not None:
        kept = root / "spot-img" / doc.key / f"{spot_page:04d}.jpg"
        source = img_dir / f"{spot_page:04d}.jpg"
        if kept.is_file():
            retained = True
        elif source.is_file():
            kept.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, kept)
            retained = True
    shutil.rmtree(img_dir, ignore_errors=True)
    return retained


def process_doc(
    root: Path,
    doc: Doc,
    page_count: int,
    ocr_pool: ThreadPoolExecutor,
    client: object,
    state: State,
) -> dict[str, int]:
    """OCR the doc's pending pages and assemble its markdown; returns the stats dict.

    A doc whose markdown already exists is a prior full pass: its cumulative progress is
    re-counted (never re-OCR'd — its images may already be deleted) and its persisted
    stats are reused, falling back to re-deriving them from the kept page markdown.
    """
    pages_dir = root / "pages" / doc.key
    existing = {p.name for p in pages_dir.iterdir()} if pages_dir.is_dir() else set()
    markdown = root / "markdown" / doc.dataset / f"{doc.doc_id}.md"
    if markdown.is_file():
        done_pages = page_count - len(ocr_lib.missing_pages(page_count, existing))
        state.bump(pages_done=done_pages, docs_assembled=1)
        stats = read_assembly_stats(root, doc)
        if stats is None:
            _, converted, kept_raw, missing = normalize_pages(root, doc, page_count)
            stats = {"tables_normalized": converted, "tables_kept_raw": kept_raw,
                     "pages_missing": missing}
        return stats
    pending = ocr_lib.missing_pages(page_count, existing)
    state.bump(pages_done=page_count - len(pending))  # cumulative across relaunches
    futures = [
        ocr_pool.submit(ocr_page, client, root, doc, page, state) for page in pending
    ]
    for future in as_completed(futures):
        future.result()  # surface OCR-thread exceptions loudly
    return assemble_doc(root, doc, page_count, state)


# -- main ------------------------------------------------------------------------------------


def run(root: Path) -> None:
    state = State(root=root)
    state.write()
    atexit.register(_deathbed, state)  # evidence even on exit paths that never raise here
    signal.signal(signal.SIGTERM, lambda _sig, _frm: sys.exit(143))  # orderly -> deathbed
    stop = threading.Event()
    writer = threading.Thread(target=_state_writer_loop, args=(state, stop), daemon=True)
    writer.start()
    started = time.monotonic()
    # Three bounded pools, one per pipeline concern. The ocr_pool IS the global request
    # semaphore: every doc lifecycle submits its page requests here, so the vLLM server
    # sees at most GLOBAL_CONCURRENCY in flight no matter how many docs overlap.
    render_pool = ThreadPoolExecutor(max_workers=RENDER_WORKERS)
    ocr_pool = ThreadPoolExecutor(max_workers=GLOBAL_CONCURRENCY)
    doc_pool = ThreadPoolExecutor(max_workers=N_DOCS_IN_FLIGHT)
    try:
        from openai import OpenAI

        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="EMPTY")

        docs = discover_docs(root)
        if not docs:
            raise RuntimeError(f"no PDFs under {root}/input/*/pdfs — nothing to do")
        spot_docs = set(ocr_lib.sample_spot_docs([doc.key for doc in docs]))

        state.phase = "stream"
        state.write()
        per_dataset: dict[str, dict[str, int]] = {}
        page_counts: dict[str, int] = {}
        spot_records: list[dict[str, object]] = []
        render_futures: list[Future[int | None] | None] = [None] * len(docs)

        def doc_lifecycle(index: int, doc: Doc) -> DocResult:
            """One doc, start to finish: render-wait -> OCR -> assemble -> free.

            Runs on a doc_pool thread; N_DOCS_IN_FLIGHT of these overlap, all feeding
            the shared ocr_pool. The per-doc ORDER is unchanged from the sequential
            design — cleanup still strictly follows assembly.
            """
            future = render_futures[index]
            assert future is not None  # submission order guarantees render-before-doc
            count = future.result()
            if count is None:
                return doc, None, {}, None  # render failure: logged by ensure_rendered
            state.bump(pages_rendered=count)
            with state.lock:
                state.pages_total += count
            stats = process_doc(root, doc, count, ocr_pool, client, state)
            spot_page = (
                ocr_lib.spot_page_for(doc.key, count) if doc.key in spot_docs else None
            )
            record: dict[str, object] | None = None
            if cleanup_doc_images(root, doc, spot_page) and spot_page is not None:
                record = {"doc": doc.key, "page": spot_page}
            return doc, count, stats, record

        def harvest(future: Future[DocResult]) -> None:
            """Fold one finished lifecycle into the metrics (main thread only)."""
            doc, count, stats, record = future.result()
            entry = per_dataset.setdefault(
                doc.dataset,
                {"docs": 0, "docs_render_failed": 0, "pages": 0, "tables_normalized": 0,
                 "tables_kept_raw": 0, "pages_missing": 0},
            )
            if count is None:
                entry["docs_render_failed"] += 1
                return
            if record is not None:
                spot_records.append(record)
            entry["docs"] += 1
            entry["pages"] += count
            for key, value in stats.items():
                entry[key] += value
            page_counts[doc.key] = count
            state.write()  # per-doc checkpoint: progress is visible doc by doc

        # The sliding window: renders run RENDER_WORKERS docs ahead of doc submission,
        # doc submission runs N_DOCS_IN_FLIGHT ahead of harvest — so page images exist
        # for at most N_DOCS_IN_FLIGHT + RENDER_WORKERS + 1 docs at any instant (the
        # defensible disk bound), while the GPU sees a continuously full request queue.
        next_render = 0
        in_flight: deque[Future[DocResult]] = deque()
        for index, doc in enumerate(docs):
            while next_render < len(docs) and next_render <= index + RENDER_WORKERS:
                staged = docs[next_render]
                render_futures[next_render] = render_pool.submit(
                    ensure_rendered, root, staged, state
                )
                next_render += 1
            in_flight.append(doc_pool.submit(doc_lifecycle, index, doc))
            while len(in_flight) >= N_DOCS_IN_FLIGHT:
                harvest(in_flight.popleft())
        while in_flight:
            harvest(in_flight.popleft())

        wall = time.monotonic() - started
        with state.lock:
            pages_done = state.pages_done
            failures = state.failures
            retries = state.repetition_retries
        metrics = {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "datasets": {
                ds: {
                    **entry,
                    "failures": sum(
                        1 for line in _failure_lines(root)
                        if json.loads(line).get("doc", "").startswith(f"{ds}/")
                    ),
                }
                for ds, entry in per_dataset.items()
            },
            "docs": page_counts,
            "spot_check": spot_records,
            "pages_total": sum(page_counts.values()),
            "pages_done": pages_done,
            "failures": failures,
            "repetition_retries": retries,
            "wall_clock_s": round(wall, 1),
            "pages_per_s": round(pages_done / wall, 3) if wall > 0 else 0.0,
        }
        tmp = root / "metrics.json.tmp"
        tmp.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, root / "metrics.json")

        state.phase = "done"
        state.write()
    except BaseException as exc:  # incl. SystemExit from SIGTERM — leave evidence, always
        with state.lock:
            if state.phase not in ("done", "failed"):
                state.failed_phase = state.phase
                state.phase = "failed"
                state.error = f"{type(exc).__name__}: {exc}"
        state.write()
        raise
    finally:
        for pool in (doc_pool, render_pool, ocr_pool):
            pool.shutdown(wait=False, cancel_futures=True)
        stop.set()
        writer.join(timeout=2)


def _failure_lines(root: Path) -> list[str]:
    path = root / "failures.jsonl"
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    faulthandler.enable()  # a hard crash dumps tracebacks instead of dying silently
    parser = argparse.ArgumentParser(description="S2 OCR worker (pod-side).")
    parser.add_argument("--root", type=Path, default=Path("/workspace/latence-sota/s2"))
    parser.add_argument(
        "--render-batch",
        type=Path,
        default=None,
        help="CHILD mode: render the docs in this JSON manifest, then exit "
        "(no state machinery; the parent owns state.json).",
    )
    args = parser.parse_args()
    if args.render_batch is not None:
        return render_batch(args.root, args.render_batch)
    (args.root / "failures.jsonl").touch(exist_ok=True)
    run(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
