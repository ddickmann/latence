# S0 investigation — providers, extraction, disambiguation, gaps (reader D of 4)

## Parser/OCR matrix (S2-decisive)

| Provider | Types | OCR | Note |
|---|---|---|---|
| `parser.lighton_vllm` | pdf + images | **YES, served vLLM** | **default model is already `lightonai/LightOnOCR-2-1B`** — exactly the OCR-diligence recommendation; official sampling (temp 0.2/top_p 0.9), dpi 200, max_longest_dim 1540, per-page retry, max_concurrency 8, per-page failure = PARSE_ERROR marker only |
| `parser.lighton` | pdf + images | YES, in-process | default LightOnOCR-1B-1025 (v1); gpu-only, ocr_batch_size 4 |
| `parser.render` | pdf, images, **docx, pptx, xlsx** | delegates to nested `ocr:` provider | the universal front-end; LibreOffice soffice→pdf preferred, text-extraction fallback |
| `parser.glm` | pdf + images | YES (GLM-OCR, MIT) | alternative VLM |
| `parser.pdfplumber` / `parser.document` | pdf text-layer + text | no | scanned pages ⇒ empty text, recorded |
| `parser.endpoint` | any (base64 to vision endpoint) | remote | default served model paddleocr-vl |

**HTML is supported by NOTHING** (grep-verified — no parser lists html/htm). UDA's
NqText/FetaTab subset ships Wikipedia HTML → S1/S2 must route those via PDF-print,
plaintext, or subset selection. Flagged as an S1 decision input.

## Extraction (S3-decisive)

- `fused_entity_relation.gliner2` (blessed): labels + relation_labels REQUIRED, model
  fastino/gliner2-multi-v1 (Apache-2.0), **single global threshold 0.5 — NO per-label
  thresholds** (matches the datagen map's ticket-02 lever), label_floor per_chunk|universal,
  microbatching + quantize/compile auto-on-cuda.
- Schema induction: `label_inducer.llm` (default gpt-4.1-mini via any OpenAI-compatible
  base_url; dedup, caps, frozen-schema delta path) **or** `label_inducer.lexicon`
  (deterministic, zero-network; needs a candidates block from `latence-label-profile`).
  NOTE: LLM induction is an OFFLINE indexing cost — the zero-LLM claim covers the online
  path; using it is a Dennis budget/scope decision at S3.
- PII: `redaction.gliner2` (Guardrails-PII-Multi; guardrails master switch + guard labels;
  label_floor FORCED universal) / `redaction.gliner_pii` / `redaction.presidio` /
  in-core `redaction.hybrid_rule`; financial floor (ssn/credit_card/iban) is
  framework-owned, un-hideable (ADR-0044).

## Disambiguation (S3/S4-decisive)

All four are registered pipeline providers: `disambiguation.cascade` (5-rung, thresholds
configurable, order fixed), `disambiguation.embedding` (real embedder rung, threshold 0.90
NOT validated for Granite — documented; **max_pairs 100k ConfigError** = the measured scale
wall), `disambiguation.glinker` (L2→L3→L4 neural linking, gpu), `disambiguation.exact_surface`
(**registered but wired into ZERO stacks** — the linear path the official run needed; the
campaign should be its first wired consumer at full-corpus scale). Type consolidation
(cascade/exact_surface) runs before resolution (ADR-0054).

## Gaps register (verified, feature-relevant to this campaign)

1. **eu_vecs/entity_vecs producer: does not exist** (caller-supplied only) — the S5 dense
   leg needs the caller ANN wired from the export's embedding column, or a new producer.
2. **Alias table = canonical names only** — GraphNode has no aliases field; member-mention
   surfaces never reach the `a|` keyspace (datagen map ticket 11 = the fix).
3. **entities.parquet sidecar** (ADR-0056 schema) not emitted.
4. `sparse.splade`: no default checkpoint (license-gated); committed stack wires
   opensearch-neural-sparse-multilingual-v1; naver/splade-v3 recorded REJECTED (NC).
5. ColBERT-family multivector encoder: seam named, not shipped (hashing reference only).
6. Community detection: only degree/pagerank+label-prop; no Louvain/Leiden package.
7. Context-enrichment situating-sentence LLM mode: contract field exists, no producer.
8. compatibility-matrix runtime cells largely UNVERIFIED (need the pinned GPU envs).

## Registry mechanics (wizard-relevant)

Entry-point group `latence.providers`, naming `<capability>.<technique>`; ProviderProfile
carries compute/license honesty; device seam = skip-with-flag; universal `device` key;
AdapterBase batching. ~60 registered providers total across core + 25 provider packages.
