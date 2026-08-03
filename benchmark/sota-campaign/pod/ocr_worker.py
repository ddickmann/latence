#!/usr/bin/env python3
"""ocr_worker.py — the pod-side S2 job: render -> OCR -> normalize -> assemble.

Runs on the pod under ``/workspace/venvs/ocr/bin/python -u`` (serve_ocr.sh installs the
deps: pypdfium2, pillow, openai; everything else is stdlib), supervised by
``run_worker.sh`` (up to 5 relaunches — every pass below is skip-if-exists, so a relaunch
continues instead of redoing). The pure logic — HTML-table normalization, repetition
detection, manifest planning, provenance format — is imported from ``ocr_lib.py``, which
the campaign stage pushes NEXT TO this file from the unit-tested
``latence_benchmark.campaign.ocr_lib`` module: the pod executes the exact bytes the
offline suite proved.

Layout under ``--root`` (default /workspace/latence-sota/s2)::

    input/<dataset>/pdfs/**.pdf     # pushed by the stage
    pages-img/<dataset>/<doc_id>/<page:04d>.jpg   # render pass (skip-if-exists)
    pages-img/<dataset>/<doc_id>/.pages           # sidecar: doc fully rendered, N pages
    pages/<dataset>/<doc_id>/<page:04d>.md        # OCR pass (skip-if-exists)
    markdown/<dataset>/<doc_id>.md  # normalized + assembled, provenance header
    state.json                      # atomic (temp+rename), rewritten every 30s
    metrics.json                    # written at the end of the run
    failures.jsonl                  # append-only failure log

**Render is crash-isolated** (the 2026-08-02 field lesson: pypdfium2 leaks RSS on some
malformed/huge PDFs until the container cap OOM-kills the process — silently, because
nothing flushed). Rendering therefore runs in CHILD subprocesses of at most
``RENDER_DOCS_PER_CHILD`` docs each, so a leak dies with the child and its memory comes
back; a batch whose child dies is retried doc-by-doc to isolate the poisonous PDF, which
is then logged to failures.jsonl and skipped — the run continues. The parent process
never imports pdfium at all.

**Evidence on death**: faulthandler is enabled, ``run_worker.sh`` launches with ``-u``
(no buffered, never-flushed logs), SIGTERM is turned into an orderly shutdown, and an
atexit deathbed handler stamps ``state.json`` ``phase: failed`` on any exit that did not
reach ``done``. SIGKILL/OOM of the parent itself cannot be caught — that is covered by
the supervisor's relaunch loop and, last, by the campaign client's stall budget.

Resumability is the FILE MANIFEST, nothing else. ``state.json`` carries a single
``progress`` counter that is a CUMULATIVE function of the filesystem (pages rendered +
pages OCR'd + docs assembled, counting pre-existing work), so it is monotone ACROSS
worker relaunches — a supervisor restart never dips below the campaign client's
high-water mark and therefore never manufactures a false stall.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import contextlib
import faulthandler
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ocr_lib  # noqa: E402  (the stage pushes it next to this file)

MODEL = "lightonai/LightOnOCR-2-1B"
MODEL_REVISION = "main"
PORT = 8044
CONCURRENCY = 12
TEMPERATURE = 0.2
TOP_P = 0.9
MAX_TOKENS = 4096
RENDER_DPI = 200
LONGEST_EDGE = 1540
MAX_MEGAPIXELS = 25.0
JPEG_QUALITY = 90
STATE_INTERVAL_S = 30.0
REQUEST_TIMEOUT_S = 300.0
#: Render children are recycled after this many docs, so leaked pdfium RSS dies with the
#: child instead of accumulating toward the container's memory cap.
RENDER_DOCS_PER_CHILD = 25
#: Sidecar marking a doc as FULLY rendered (written atomically after its last page).
PAGES_SIDECAR = ".pages"


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
            self.progress += sum(deltas.values())

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
    with (root / "failures.jsonl").open("a", encoding="utf-8") as out:
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


# -- render pass (CHILD process; pypdfium2, 200 DPI, longest edge 1540, JPEG) ----------------


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
        sidecar_tmp = out_dir / (PAGES_SIDECAR + ".tmp")
        sidecar_tmp.write_text(f"{page_count}\n", encoding="utf-8")
        os.replace(sidecar_tmp, out_dir / PAGES_SIDECAR)
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


def read_render_sidecar(root: Path, doc: Doc) -> int | None:
    """The doc's page count if (and only if) its render pass completed; else ``None``."""
    path = root / "pages-img" / doc.key / PAGES_SIDECAR
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
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


def render_all(root: Path, docs: list[Doc], state: State) -> dict[str, int]:
    """The parent render loop: recycled children, per-doc isolation of poisonous PDFs.

    Returns ``{doc.key: page_count}`` for every SUCCESSFULLY rendered doc. A doc whose
    dedicated child also dies is logged to failures.jsonl and excluded — the run
    continues; one bad PDF must never cost the other 56k pages. Progress counts EVERY
    completed doc's pages (including pre-existing ones), keeping the counter cumulative
    and monotone across worker relaunches.
    """
    page_counts: dict[str, int] = {}

    def collect(doc: Doc) -> bool:
        count = read_render_sidecar(root, doc)
        if count is None:
            return False
        page_counts[doc.key] = count
        state.bump(pages_rendered=count)
        return True

    pending = [doc for doc in docs if not collect(doc)]
    state.write()  # the resume sweep is visible immediately, not 30s later

    for batch in ocr_lib.chunked(pending, RENDER_DOCS_PER_CHILD):
        batch_docs = list(batch)
        if not _spawn_render_child(root, batch_docs):
            # The child died (leak/OOM/bad PDF): isolate doc-by-doc so exactly the
            # poisonous document is skipped, not its 24 neighbours.
            for doc in batch_docs:
                if read_render_sidecar(root, doc) is not None:
                    continue  # rendered before the child died
                if not _spawn_render_child(root, [doc]):
                    log_failure(
                        root,
                        {
                            "doc": doc.key,
                            "kind": "render",
                            "error": "render child died for this doc alone; doc skipped",
                        },
                    )
                    state.bump(failures=1)
        for doc in batch_docs:
            if doc.key not in page_counts:
                collect(doc)
        state.write()
    return page_counts


# -- OCR pass (per-page OpenAI-compatible requests against the local vLLM) -------------------


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


# -- normalize + assemble --------------------------------------------------------------------


def assemble_doc(
    root: Path, doc: Doc, page_count: int, state: State
) -> tuple[int, int, int]:
    """Normalize tables per page and write ``markdown/<ds>/<doc_id>.md``.

    Returns ``(tables_converted, tables_kept_raw, missing_pages)``. A page whose OCR
    permanently failed is represented by an explicit placeholder comment — visible, never
    silently absent.
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
    os.replace(tmp, target)
    state.bump(docs_assembled=1)  # progress tick per assembled doc (monotone counter)
    return converted, kept_raw, missing


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
    try:
        from openai import OpenAI

        client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="EMPTY")

        docs = discover_docs(root)
        if not docs:
            raise RuntimeError(f"no PDFs under {root}/input/*/pdfs — nothing to do")

        # Pass 1: render in disposable child processes (crash-isolated, resumable).
        state.phase = "render"
        state.write()
        page_counts = render_all(root, docs, state)
        rendered_docs = [doc for doc in docs if doc.key in page_counts]
        with state.lock:
            state.pages_total = sum(page_counts.values())
        state.write()

        # Pass 2: OCR — independent per-page requests, concurrency 12, skip-if-exists.
        state.phase = "ocr"
        state.write()
        pending: list[tuple[Doc, int]] = []
        for doc in rendered_docs:
            pages_dir = root / "pages" / doc.key
            existing = (
                {p.name for p in pages_dir.iterdir()} if pages_dir.is_dir() else set()
            )
            pending.extend(
                (doc, page)
                for page in ocr_lib.missing_pages(page_counts[doc.key], existing)
            )
        # Count already-OCR'd pages THROUGH bump so progress stays cumulative across
        # relaunches (a plain assignment would dip below the client's high-water mark).
        state.bump(pages_done=state.pages_total - len(pending))
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = [
                pool.submit(ocr_page, client, root, doc, page, state)
                for doc, page in pending
            ]
            for future in as_completed(futures):
                future.result()  # surface worker exceptions loudly

        # Pass 3: normalize HTML tables + assemble per-doc markdown with provenance.
        state.phase = "assemble"
        state.write()
        per_dataset: dict[str, dict[str, float]] = {}
        for doc in docs:
            entry = per_dataset.setdefault(
                doc.dataset,
                {"docs": 0, "docs_render_failed": 0, "pages": 0, "tables_normalized": 0,
                 "tables_kept_raw": 0, "pages_missing": 0},
            )
            if doc.key not in page_counts:
                entry["docs_render_failed"] += 1
                continue
            converted, kept_raw, missing = assemble_doc(
                root, doc, page_counts[doc.key], state
            )
            entry["docs"] += 1
            entry["pages"] += page_counts[doc.key]
            entry["tables_normalized"] += converted
            entry["tables_kept_raw"] += kept_raw
            entry["pages_missing"] += missing

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
                    **{k: int(v) for k, v in entry.items()},
                    "failures": sum(
                        1 for line in _failure_lines(root)
                        if json.loads(line).get("doc", "").startswith(f"{ds}/")
                    ),
                }
                for ds, entry in per_dataset.items()
            },
            "docs": {doc.key: page_counts[doc.key] for doc in rendered_docs},
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
