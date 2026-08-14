"""Induction sample: 12 random chunks (seed 0, 200-600 tokens) from the first ~200
docs each of ohr_bench + vidoseek + uda, chunked with the REAL latence chunker
(chunk.markdown engine: latence_core.chunking.chunk_document) at the pipeline
budget max_tokens=640 / overlap_tokens=80 / min_tokens=8.

Run: PYTHONHASHSEED=0 .venv/bin/python induction_sample.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from latence_core.chunking import ChunkBudget, chunk_document

# benchmark/sota-campaign/stages/s3-pipeline/chunk-validation/<this file> → repo root.
REPO = Path(__file__).resolve().parents[5]
CORPUS = REPO / "benchmark" / "sota-corpus"
OUT = (
    REPO
    / "benchmark"
    / "sota-campaign"
    / "stages"
    / "s3-pipeline"
    / "chunk-validation"
    / "induction-sample.jsonl"
)

BUDGET = ChunkBudget(max_tokens=640, overlap_tokens=80, min_tokens=8)
DATASETS = ["ohr_bench", "vidoseek", "uda"]
DOCS_PER_DS = 200
MIN_T, MAX_T = 200, 600
N_SAMPLES = 12


def main() -> None:
    pool: list[dict[str, str]] = []
    for ds in DATASETS:
        md_dir = CORPUS / ds / "markdown"
        docs = sorted(p for p in md_dir.iterdir() if p.suffix == ".md")[:DOCS_PER_DS]
        for doc in docs:
            text = doc.read_text(encoding="utf-8", errors="replace")
            for i, ch in enumerate(chunk_document(text, BUDGET)):
                if MIN_T <= ch.token_count <= MAX_T:
                    pool.append(
                        {
                            "dataset": ds,
                            "doc": doc.name,
                            "chunk_id": f"{doc.stem}#chunk-{i}",
                            "text": ch.text,
                        }
                    )
    rng = random.Random(0)
    sample = rng.sample(pool, min(N_SAMPLES, len(pool)))
    with OUT.open("w", encoding="utf-8") as f:
        for rec in sample:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"pool={len(pool)} wrote={min(N_SAMPLES, len(pool))} -> {OUT}")


if __name__ == "__main__":
    main()
