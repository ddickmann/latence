#!/usr/bin/env python3
"""ocr_worker.py — the pod-side S2 job: render -> OCR -> normalize -> assemble.

Runs on the pod under ``/workspace/venvs/ocr/bin/python`` (serve_ocr.sh installs the deps:
pypdfium2, pillow, openai; everything else is stdlib). The pure logic — HTML-table
normalization, repetition detection, manifest planning, provenance format — is imported
from ``ocr_lib.py``, which the campaign stage pushes NEXT TO this file from the
unit-tested ``latence_benchmark.campaign.ocr_lib`` module: the pod executes the exact
bytes the offline suite proved.

Layout under ``--root`` (default /workspace/latence-sota/s2)::

    input/<dataset>/pdfs/**.pdf     # pushed by the stage
    pages-img/<dataset>/<doc_id>/<page:04d>.jpg   # render pass (skip-if-exists)
    pages/<dataset>/<doc_id>/<page:04d>.md        # OCR pass (skip-if-exists)
    markdown/<dataset>/<doc_id>.md  # normalized + assembled, provenance header
    state.json                      # atomic (temp+rename), rewritten every 30s
    metrics.json                    # written at the end of the run
    failures.jsonl                  # append-only failure log

Resumability is the FILE MANIFEST, nothing else: every unit of work checks the
filesystem before doing anything, so a crash loses at most in-flight pages and a rerun
is a cheap sweep over already-done files. ``state.json`` carries a single monotone
``progress`` counter (pages rendered + pages OCR'd + docs assembled) that the campaign
CLI polls with a 1800s stall budget.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
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


# -- render pass (pypdfium2, 200 DPI, longest edge 1540, JPEG) -------------------------------


def render_doc(root: Path, doc: Doc, state: State) -> int:
    """Render every page of ``doc`` to JPEG, skip-if-exists. Returns the page count."""
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
            width_pt, height_pt = page.get_size()
            scale = RENDER_DPI / 72.0
            # Hard cap 25 MP/page (the rasterizer discipline the framework already uses).
            megapixels = (width_pt * scale) * (height_pt * scale) / 1e6
            if megapixels > MAX_MEGAPIXELS:
                scale *= (MAX_MEGAPIXELS / megapixels) ** 0.5
            image = page.render(scale=scale).to_pil().convert("RGB")
            if max(image.size) > LONGEST_EDGE:
                image.thumbnail((LONGEST_EDGE, LONGEST_EDGE))
            tmp = target.with_name(target.name + ".part")
            image.save(tmp, format="JPEG", quality=JPEG_QUALITY)
            os.replace(tmp, target)
            state.bump(pages_rendered=1)
    finally:
        pdf.close()
    return page_count


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

        # Pass 1: render (resumable; separate from inference by design).
        state.phase = "render"
        state.write()
        page_counts: dict[str, int] = {}
        for doc in docs:
            page_counts[doc.key] = render_doc(root, doc, state)
        with state.lock:
            state.pages_total = sum(page_counts.values())
        state.write()

        # Pass 2: OCR — independent per-page requests, concurrency 12, skip-if-exists.
        state.phase = "ocr"
        state.write()
        pending: list[tuple[Doc, int]] = []
        for doc in docs:
            pages_dir = root / "pages" / doc.key
            existing = (
                {p.name for p in pages_dir.iterdir()} if pages_dir.is_dir() else set()
            )
            pending.extend(
                (doc, page)
                for page in ocr_lib.missing_pages(page_counts[doc.key], existing)
            )
        with state.lock:
            state.pages_done = state.pages_total - len(pending)
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
            converted, kept_raw, missing = assemble_doc(
                root, doc, page_counts[doc.key], state
            )
            entry = per_dataset.setdefault(
                doc.dataset,
                {"docs": 0, "pages": 0, "tables_normalized": 0,
                 "tables_kept_raw": 0, "pages_missing": 0},
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
            "docs": {doc.key: page_counts[doc.key] for doc in docs},
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
    except Exception as exc:
        with state.lock:
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
    parser = argparse.ArgumentParser(description="S2 OCR worker (pod-side).")
    parser.add_argument("--root", type=Path, default=Path("/workspace/latence-sota/s2"))
    args = parser.parse_args()
    (args.root / "failures.jsonl").touch(exist_ok=True)
    run(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
