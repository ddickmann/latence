"""Full chunk-quality stats over the S3 corpus with the REAL latence chunker.

Provider parity: the generated pipelines use provider ``chunk.markdown``
(latence_core.stages.chunk.MarkdownChunker), which chunks doc.content via
``latence_core.chunking.chunk_document(text, ChunkBudget(...))``. This script
drives that same engine directly over each markdown file's text with the
pipeline budget max_tokens=640 / overlap_tokens=80 / min_tokens=8.

Outputs (all under chunk-validation/):
  metrics.json      raw numbers per dataset
  samples/<ds>.md   3 sample chunks (typical / shortest / longest)

Run: PYTHONHASHSEED=0 .venv/bin/python chunk_stats.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import re
import time
from pathlib import Path

from latence_core.chunking import ChunkBudget, chunk_document

REPO = Path("/Users/dennisdickmann/Developer/latence-framework")
CORPUS = REPO / "benchmark" / "sota-corpus"
OUTDIR = REPO / "benchmark/sota-campaign/stages/s3-pipeline/chunk-validation"
SAMPLES = OUTDIR / "samples"

BUDGET = ChunkBudget(max_tokens=640, overlap_tokens=80, min_tokens=8)
MAX_T = 640

DATASETS = ["ohr_bench", "vidoseek", "uda", "multihop_rag", "wiki2multihop"]

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SEP_ROW_RE = re.compile(
    r"^\s*\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)+\|?[ \t]*$"
)


def _is_pipe_row(line: str) -> bool:
    return line.lstrip().startswith("|")


def _is_sep_row(line: str) -> bool:
    return bool(_SEP_ROW_RE.match(line))


def _table_truncated(orig_span: str) -> bool:
    """Heuristic: chunk starts or ends inside a ``|`` table without its header separator.

    Runs on the ORIGINAL markdown span (pipes are stripped from chunk text by the
    engine's pattern 13, so the emitted text cannot be inspected for this).
    """
    lines = [ln for ln in orig_span.splitlines() if ln.strip()]
    if not lines:
        return False
    # Starts mid-table: first line is a pipe row, and it is not a header
    # (header would be immediately followed by a separator row).
    starts_in_table = _is_pipe_row(lines[0]) and not _is_sep_row(lines[0])
    if starts_in_table and (len(lines) < 2 or not _is_sep_row(lines[1])):
        return True
    # Ends mid-table: last line is a pipe row and the contiguous trailing pipe
    # block contains no separator row (header row + separator are outside).
    if _is_pipe_row(lines[-1]) and not _is_sep_row(lines[-1]):
        i = len(lines) - 1
        has_sep = False
        while i >= 0 and (_is_pipe_row(lines[i]) or _is_sep_row(lines[i])):
            if _is_sep_row(lines[i]):
                has_sep = True
            i -= 1
        if not has_sep:
            return True
    return False


def _is_placeholder_or_empty(chunk_text: str) -> bool:
    residue = _COMMENT_RE.sub("", chunk_text)
    return len(residue.strip()) < 20


def process_doc(path_str: str) -> dict:
    path = Path(path_str)
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_document(text, BUDGET)
    tokens = [c.token_count for c in chunks]
    n_placeholder = sum(1 for c in chunks if _is_placeholder_or_empty(c.text))
    n_table_trunc = sum(
        1 for c in chunks if "|" in text[c.char_start : c.char_end]
        and _table_truncated(text[c.char_start : c.char_end])
    )
    return {
        "doc": path.name,
        "tokens": tokens,
        "n_placeholder": n_placeholder,
        "n_table_trunc": n_table_trunc,
    }


def _pct(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    k = min(len(sorted_vals) - 1, max(0, int(round(p / 100 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def _dist(vals: list[int]) -> dict:
    s = sorted(vals)
    return {
        "p5": _pct(s, 5),
        "p25": _pct(s, 25),
        "p50": _pct(s, 50),
        "p75": _pct(s, 75),
        "p95": _pct(s, 95),
        "max": s[-1] if s else 0,
    }


def _extract_chunk(md_dir: Path, doc: str, idx: int) -> tuple[str, int]:
    text = (md_dir / doc).read_text(encoding="utf-8", errors="replace")
    ch = chunk_document(text, BUDGET)[idx]
    return ch.text, ch.token_count


def run_dataset(ds: str, pool: mp.pool.Pool) -> dict:
    md_dir = CORPUS / ds / "markdown"
    docs = sorted(str(p) for p in md_dir.iterdir() if p.suffix == ".md")
    t0 = time.monotonic()
    results = pool.map(process_doc, docs, chunksize=32)
    wall = time.monotonic() - t0

    all_tokens: list[int] = []
    chunks_per_doc: list[int] = []
    n_placeholder = 0
    n_table_trunc = 0
    n_zero_chunk_docs = 0
    # (token_count, doc, idx) trackers
    shortest = longest = None  # type: ignore[assignment]
    per_chunk_index: list[tuple[int, str, int]] = []
    for r in results:
        toks = r["tokens"]
        chunks_per_doc.append(len(toks))
        if not toks:
            n_zero_chunk_docs += 1
        all_tokens.extend(toks)
        n_placeholder += r["n_placeholder"]
        n_table_trunc += r["n_table_trunc"]
        for i, t in enumerate(toks):
            per_chunk_index.append((t, r["doc"], i))
            if shortest is None or t < shortest[0]:
                shortest = (t, r["doc"], i)
            if longest is None or t > longest[0]:
                longest = (t, r["doc"], i)

    n = len(all_tokens)
    median = _dist(all_tokens)["p50"] if n else 0
    typical = min(per_chunk_index, key=lambda x: abs(x[0] - median)) if n else None

    # Write samples
    SAMPLES.mkdir(parents=True, exist_ok=True)
    with (SAMPLES / f"{ds}.md").open("w", encoding="utf-8") as f:
        f.write(f"# Sample chunks — {ds}\n\n")
        f.write("Chunker: `chunk.markdown` (max_tokens=640, overlap=80, min_tokens=8)\n\n")
        for label, ref in [("Typical (closest to median)", typical),
                           ("Shortest", shortest), ("Longest", longest)]:
            if ref is None:
                continue
            t, doc, idx = ref
            text_, tc = _extract_chunk(md_dir, doc, idx)
            f.write(f"## {label} — `{doc}` chunk {idx} ({tc} tokens)\n\n")
            f.write("````\n" + text_ + "\n````\n\n")

    stats = {
        "docs": len(docs),
        "docs_zero_chunks": n_zero_chunk_docs,
        "chunks": n,
        "wall_seconds": round(wall, 1),
        "token_dist": _dist(all_tokens),
        "pct_lt_64": round(100 * sum(1 for t in all_tokens if t < 64) / n, 2) if n else 0,
        "pct_at_max_640": round(100 * sum(1 for t in all_tokens if t >= MAX_T) / n, 2) if n else 0,
        "pct_ge_608_95pct_cap": (
            round(100 * sum(1 for t in all_tokens if t >= 608) / n, 2) if n else 0
        ),
        "chunks_per_doc_dist": _dist(chunks_per_doc),
        "placeholder_or_empty_chunks": n_placeholder,
        "table_truncated_chunks": n_table_trunc,
        "pct_table_truncated": round(100 * n_table_trunc / n, 2) if n else 0,
    }
    print(f"[{ds}] {stats}", flush=True)
    return stats


def main() -> None:
    metrics: dict[str, dict] = {
        "_config": {
            "provider": "chunk.markdown (latence_core.stages.chunk.MarkdownChunker engine)",
            "max_tokens": 640,
            "overlap_tokens": 80,
            "min_tokens": 8,
            "token_counter": "latence_core.chunking.estimate_tokens (chunker-native)",
        }
    }
    with mp.Pool(processes=8) as pool:
        for ds in DATASETS:
            metrics[ds] = run_dataset(ds, pool)
            with (OUTDIR / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
