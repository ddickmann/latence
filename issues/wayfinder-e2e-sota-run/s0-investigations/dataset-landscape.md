# S0 investigation — SOTA benchmark dataset landscape + licenses (2026-08-02)

Recommended final-corpus composition: **6 new datasets + the 2 incumbents**, covering the
multi-hop text triad, realistic-corpus retrieval, OCR-exercised documents, and enterprise
long documents. Everything below cited in the research pass; UNVERIFIED items are blocking
flags for S1, not footnotes.

## Recommended composition

| # | Dataset | Role | License handling | Published line to beat |
|---|---|---|---|---|
| 1 | **MuSiQue** (keep) | hard multi-hop text | CC BY 4.0, attribution | existing campaign numbers |
| 2 | **HotpotQA** (keep) | multi-hop breadth, supporting-fact standard | CC BY-SA → keep-local | large literature |
| 3 | **2WikiMultiHopQA** | 3rd leg of the multi-hop triad; gold **evidence triples** map directly onto the KG/hypergraph | Apache-2.0 dataset; derived wiki text share-alike → keep-local | Beam Retrieval 99.9% retrieval precision (arXiv 2308.08973) |
| 4 | **MultiHop-RAG** | only text set with a realistic SHARED corpus (609 news docs) + per-query gold evidence — exactly the recall@budget shape; non-Wikipedia register | ODC-BY annotations, news copyright → strictly local | Hits@10 0.7467 / MRR@10 0.586 (arXiv 2401.15391) |
| 5 | **OHR-Bench** | **the OCR anchor**: 8.5k+ original PDF pages, 7 domains, 8,498 QA with evidence pages + per-evidence-type scoring, published per-OCR-pipeline retrieval numbers | LICENSE CONFLICT (HF says CC BY 4.0, README says research-only) → treat research-only, keep local, resolve with opendatalab before any redistribution | GT-text retrieval 70.4 vs best-OCR 59.6 / 31.1 overall — a direct scoreboard for our OCR+parse stage |
| 6 | **ViDoSeek** | OCR + corpus-wide page retrieval, single+multi-hop over visually rich PDFs (charts/tables/layout), 1,142 q | Apache-2.0 annotations; PDF provenance UNVERIFIED → PDFs keep local | ViDoRAG baselines (arXiv 2502.18017 — fetch exact numbers at ingest) |
| 7 | **UDA** (FinHybrid + PaperTab subsets) | enterprise + academic long docs in ORIGINAL PDF form with expert evidence labels (covers FinQA/TAT-QA ground THROUGH our OCR stage) | CC BY-SA 4.0 → keep-local lane | UDA NeurIPS D&B baselines (arXiv 2406.15187) |

**Scale**: low tens of thousands of PDF pages on the doc side + the text triad's paragraph
pools + 609 news articles; **>60k QA total, ~20k with passage/page-level evidence** usable
for supporting-fact recall@budget.

## Deliberately excluded (with reasons)

- FinanceBench — CC BY-NC + only 150 open QA.
- DocBench — NO license found (UNVERIFIED) → do not ingest.
- M3DocVQA — corpus must be self-built (authors withheld PDFs over license risk); overlaps
  Wikipedia content with #1–3.
- Bamboogle (MIT) / FRAMES (Apache-2.0) — no bundled corpus/passage gold; usable later as
  answer-only smoke tests.
- BEIR/NQ/TriviaQA — single-hop, little marginal eval value, mixed per-subset licenses.

## Open verification items (S1 gate blockers where marked)

1. **OHR-Bench license conflict** (HF CC BY 4.0 vs README research-only) — handling: assume
   research-only/keep-local; BLOCKS any redistribution, not local use.
2. MoreHopQA license read from search results only — not in the recommended set anyway.
3. Exact SOTA numbers for 2Wiki EM/F1 and ViDoSeek — fetch at ingest.
4. ViDoRe V2/V3 per-subset licenses — only relevant if a visual-retriever comparison is
   added later.

## Notes for later stages

- 2Wiki's evidence triples double as **extraction gold** — a free extraction-quality
  benchmark for the datagen-quality map's instruments (S3 synergy).
- OHR-Bench ships pre-run outputs of ~10 OCR pipelines — our LightOnOCR-2 leg can be
  compared against those directly in S2.
- UDA subsumes FinQA/TAT-QA sources as original PDFs — no need to ingest those separately.

Full citations: 2wikimultihop GitHub · MultiHop-RAG GitHub/HF/paper · OHR-Bench GitHub/HF ·
MoreHopQA HF · Bamboogle (self-ask repo) · FRAMES HF · M3DocRAG GitHub · UDA GitHub/paper ·
ViDoSeek HF · ViDoRAG paper · ViDoRe V3 · REAL-MM-RAG HF · DocBench GitHub · FinQA GitHub ·
TAT-QA GitHub · FinanceBench HF · BEIR paper · Beam Retrieval paper.
