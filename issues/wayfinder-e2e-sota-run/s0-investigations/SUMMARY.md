# S0 GATE SUMMARY — all investigations complete (2026-08-02)

Five reports in this directory: [ocr-diligence.md](ocr-diligence.md) ·
[dataset-landscape.md](dataset-landscape.md) · [inventory-core-pipeline.md](inventory-core-pipeline.md) ·
[inventory-retrieval.md](inventory-retrieval.md) · [inventory-clis-benchmarks.md](inventory-clis-benchmarks.md) ·
[inventory-providers-gaps.md](inventory-providers-gaps.md).

## The five facts that shape the campaign

1. **The OCR stage is nearly free**: `parser.lighton_vllm` already defaults to
   `lightonai/LightOnOCR-2-1B` — the exact model the diligence pass ranked #1
   (olmOCR-bench SOTA, Apache-2.0 weights+code, fits the 16GB pod). S2 = serve vLLM on the
   pod + wire the existing provider. `parser.render` handles docx/pptx/xlsx.
   Only genuine S2 work: HTML inputs (no parser supports HTML) and the v2
   HTML-tables-in-markdown normalization question.
2. **The full-feature S3 spec is a composition of existing blessed pieces**:
   gpu-sota stack (gliner2 fused extraction + gliner2 PII + embedding/glinker
   disambiguation + graph_completion.ultra) + schema induction + hyperedges +
   graph_features + context enrichment + all four export signal slots (dense, sparse,
   multivector→FDE, bm25). Nothing has to be invented; everything has to be CONFIGURED
   and measured.
3. **S5 can finally run the dense-legged eval the audit demanded**: shipped
   sentence-transformers/endpoint embedders + Qdrant sink/backend + orchestrator with
   fusion + rerank + graph legs + hypergraph/CoE. This directly closes CLOSEOUT-AUDIT
   F-VEC/F-BASE. Tier-2 query NER remains harness-side (productization candidate inside
   this campaign).
4. **The campaign runner has a proven shape to productize**: supervise.sh's stage table +
   filesystem done-probes + progress-freeze stall detection + atomic STATE.json, in
   Python, wrapped by a wizard mirroring `latence setup`'s house style (pure
   byte-deterministic generator, flag>prompt>default, --non-interactive contract,
   dry-validate before write). The "campaign" name is free in the repo.
5. **The corpus recommendation is 6 new + 2 incumbent datasets** (multi-hop triad +
   MultiHop-RAG + OHR-Bench + ViDoSeek + UDA), ~low tens of thousands of PDF pages +
   text-side paragraph pools, >60k QA / ~20k with retrieval-usable evidence, each with a
   published line to beat. Two license flags: OHR-Bench conflict (treat research-only,
   keep local), DocBench excluded (no license).

## Decisions Dennis owns at this gate (S1 spec)

1. Corpus composition — confirm/trim the 6+2.
2. OCR commit — LightOnOCR-2-1B via the served vLLM provider (recommended) vs in-process v1.
3. Wizard home — new `latence-campaign` CLI (recommended: latence-benchmark package,
   argparse, suite-registry adjacency) vs extending the `latence` typer CLI.
4. Schema induction in the S3 profile — LLM-based (OpenRouter cost, strongest labels) vs
   lexicon inducer (zero-network, needs label-profiler candidates) vs hand-authored labels.

## Known risks carried forward

- HotpotQA/UDA/2Wiki derived text = share-alike lane (local, gitignored) — same discipline
  as the M6 campaign, already proven.
- HTML subset of UDA needs a route (exclude, or print-to-PDF, or plaintext).
- GLiNER2 single global threshold (no per-label) — extraction quality ceiling is known
  (datagen map); this campaign MEASURES at full strength, the datagen map IMPROVES.
- Pod: 16GB VRAM — OCR (vLLM) and extraction (gliner2) phases must be sequenced or
  co-scheduled carefully; supervise.sh precedent already does daemon start/stop per stage.
