# S3 chunk-quality validation (local, real chunker)

Chunker: `chunk.markdown` max_tokens 640 / overlap 80 / min_tokens 8 — the exact provider +
settings the generated pipelines run. Full corpus, all docs, CPU on the Mac (2026-08-04).

| dataset | docs | chunks | median tok | %<64 tok | % at 640 | placeholder/empty | table-truncated |
|---|---|---|---|---|---|---|---|
| ohr_bench | 1,261 | 13,909 | 582 | 0.8% | 2.6% | 27 | 217 (1.6%) |
| vidoseek | 292 | 2,434 | 597 | 0.2% | 1.4% | 1 | 20 (0.8%) |
| uda | 1,917 | 220,129 | 571 | 0.04% | 0.4% | 80 | 4,673 (2.1%) |
| multihop_rag | 609 | 3,264 | 590 | 0.1% | 0.1% | 0 | 0 |
| wiki2multihop | 56,687 | 57,812 | 70 | 45.2% | 0.05% | 374 | 0 |
| **total** | 60,766 | **297,548** | | | | 482 | 4,910 |

## Verdicts

- **ohr_bench / vidoseek / uda / multihop_rag: sound.** Dense chunks near the 640 target,
  negligible tiny-chunk rate, zero-chunk docs = 0 everywhere. Table truncation is 0.8–2.1%
  (chunk boundary landing inside a GFM table) — inherent to token-capped chunking of long
  tables; acceptable at this rate.
- **ohr_bench max 1,342 tokens**: a handful of chunks exceed 640 — atomic markdown blocks
  (giant tables/fences) the chunker keeps whole rather than corrupting. Tiny fraction, correct
  behavior.
- **wiki2multihop: short by nature, not by defect.** The question-pool corpus is 56,687
  one-paragraph wiki intro docs (median 70 tokens, 1 chunk/doc) — the same shape as the M6
  musique/hotpot corpora. 45% under 64 tokens is the corpus, not the chunker. No induction
  runs here (hand-authored schema), and GLiNER2 handles short inputs natively.
- **Placeholders**: 482 chunks are OCR-failure placeholders or near-empty — 0.16% of the
  corpus, explicit and traceable to the S2 failures.jsonl residual.

## Budget-relevant finding

Actual chunk counts vs the ticket estimate: **236,472 induction-eligible chunks**
(ohr 13,909 + vidoseek 2,434 + uda 220,129) vs ~176k estimated — UDA alone is 220k. At
gpt-4.1-mini's measured per-call cost the $150 cap covers only ~64% of UDA (per-dataset call
budget 50,403); the remainder would fail open to floor labels. See the model A/B for the
cheaper alternative.
