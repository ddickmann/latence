# S3 — Full-feature pipeline data generation (chunking + GLiNER2 extraction)

Type: task
Status: claimed
Blocked by: s2-ocr (approved 2026-08-04)

## Spec (Dennis, 2026-08-04 — all four recommendations accepted)

1. **Chunking**: `chunk.markdown`, max_tokens 640 / overlap 80 / min 8 (the blessed gpu-sota
   operating point; GLiNER2 native window 768).
2. **Labels — split strategy**: `label_inducer.llm` per-chunk (label_floor per_chunk,
   OpenRouter gpt-4.1-mini, dedup on, budget guards) on the OCR'd sets (ohr_bench,
   vidoseek, uda) — est. $90–120; **hand-authored** label/relation sets for the text sets
   (2wiki, multihop_rag): standard wiki/news schemas (PERSON/ORG/LOCATION/WORK/EVENT/DATE +
   located_in/part_of/member_of/created_by/spouse_of/directed_by/founded_by/...).
3. **Text sets**: REUSE the M6 v20 exports for musique-full + hotpot-full (settings
   comparability with the official gates; ~2 pod-days saved). Newly process: 2wiki
   (question-pool corpus, M6-style), multihop_rag (609 articles), + all OCR'd markdown.
4. **Redaction**: `redaction.gliner2` with guardrails ON (full safety path exercised;
   framework financial floor always on).

Everything wizard-applicable: a new `s3:` block in campaign.yaml (chunking knobs,
extraction provider/model/threshold, induction mode + budget cap, label sets, redaction
profile) → the stage GENERATES per-dataset pipeline.yaml from it; wizard `init` gains the
corresponding prompts with these as defaults.

## Pipeline profile per newly-processed dataset

source.local_folder (the dataset's markdown/) → intake screening → parse (plaintext/markdown)
→ chunk.markdown {640,80,8} → content screening → [induce: llm per-chunk (OCR'd sets) |
none (text sets, config labels)] → fused_entity_relation.gliner2 (labels/relations from
induction or config; threshold 0.5) → redaction.gliner2 (guardrails on; extract_on
unmasked, redact AFTER extraction per default) → profiling.statistical →
type_consolidation.exact_surface → disambiguation.exact_surface (full-corpus scale — the
cascade walls past ~10⁴ surfaces, M6 lesson) → graph.canonical {hyperedges: true,
graph_features: degree} → context.kg_header → exports: jsonl_parquet (embedder:
sentence_transformers granite-311m cuda + bm25: tokenizer.regex + context_columns) +
knowledge_graph (hyperedge_kv on).

Run host: the small pod (RTX 2000 Ada 16GB) under the supervise pattern; one `latence
process` run per dataset, run_id `s3-<dataset>-v1`, resumable via same-run_id checkpoints.

## Gate artifacts

Per-dataset Quality Report (quality-report.md — incl. HyperedgeQuality section) + a
consolidated s3 metrics.json (chunks, mentions, relations, entities, hyperedges, PII,
guardrail findings, induction cost actuals vs estimate) + latence-console pointer for
interactive inspection. Dennis inspects → approve opens S4/S5.

## Estimates

~176k chunks (OCR'd) + 2wiki pool + 609 articles ≈ 200k chunks total ≈ 1.5–2 pod-days
extraction (GLiNER2 + PII both on GPU, sequential stages). Induction $90–120 (hard budget
cap in config; run blocks at cap, never silently overruns).
