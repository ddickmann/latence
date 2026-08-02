# S0 investigation — core pipeline config surface (reader A of 4)

The wizard's schema source of truth. Code is authoritative; this distills the
campaign-shaping facts (full key tables live in the reader's pass over
`packages/latence-core/src/latence_core/` — pipeline.py, stages/*, runner.py).

## Campaign-critical discoveries

1. **LightOn OCR is ALREADY a shipped provider**: `stacks/gpu-ocr.yaml` and
   `universal-parse.yaml` wire `parser.render` with the nested `ocr:` sub-provider slot →
   **`parser.lighton`**; `deploy/stacks/served-ocr.yaml` is the served-OCR client stack.
   S2 = configure + serve, zero new pipeline code. (Verify at S2 spec time which LightOnOCR
   model version the provider targets — v2's HTML-tables caveat.)
2. **The full-blown SOTA path already has a blessed stack**: `stacks/gpu-sota.yaml` =
   `fused_entity_relation.gliner2` + `redaction.gliner_pii` + `disambiguation.embedding`
   (sentence-transformers granite-311m, cuda, flash-attn) + `graph_completion.ultra` +
   endpoint embedder export + chunk {640,80,8}; `schema-induction.yaml` adds
   `label_inducer.llm` between screening and extraction. "Most advanced features" ≈
   gpu-sota + schema-induction + hyperedges + graph_features + context-enrichment +
   retrieval-signals slots — the S3 spec composes these.
3. **Hyperedges block** (graph.canonical ONLY, not graph.weighted): keys =
   weak_hyperedges(true) / max_arity(12) / min_tokens(25) / max_tokens(180) /
   max_window_sentences(3) / token_counter("tiktoken:<enc>"|null); unknown keys are loud
   errors; tau_text redaction NOT configurable (enforced at writer); enterprise-gold's
   pipeline.yaml is the only in-repo YAML wiring it today.
4. **Resume semantics**: NO --resume flag — re-invoke with the SAME run_id; stage-level
   checkpoint skip (finalized checkpoint ⇒ stage replayed from disk, flagged
   from_checkpoint). Empty checkpoints sticky. The campaign runner's per-stage resume maps
   directly onto this. Deltas need a fresh UTC-stamped run id.
5. **Storage layout** per run: `_latence/runs/<run_id>/{manifest.json, checkpoints/*.jsonl,
   wal.log, quality-report.json|.md, export/, quarantine/, screening/, type-vocabulary/}` —
   the console UI reads exactly this; the campaign's inspection gates can point at
   quality-report.md + latence-console.
6. **Quality Report** already carries per-stage metrics incl. a `HyperedgeQuality` section
   (counts, γ, arity histogram, token costs) + goldset scoring + delta churn — S3's gate
   artifact exists off the shelf.

## Pipeline schema facts (wizard validation rules)

- Root: schema_version=2, name, storage_uri (schemes file/memory/s3/gs/gcs/abfs/az),
  stages[], gold_set_uri, extract_on {unmasked|masked}. `extra="forbid"` at root AND stage
  level — unknown keys are hard errors.
- Stage: name/capability/provider/config/depends_on. 18 CapabilityKinds; `embedding` is
  sub-slot-only (standalone = ConfigError). `device: auto|cpu|cuda` valid on EVERY stage
  (skip-with-flag routing, never raises).
- Nested sub-provider slots: parse.ocr; export.{embedder, sparse_embedder,
  multivector_embedder, fde_converter, bm25}; graph_assembly.graph_features;
  type_consolidation.embedder; disambiguation.embedder.
- `extract_on: masked` structurally requires extraction stages to depend on a redaction
  stage — wizard must wire depends_on accordingly.
- 32 in-core providers (registry in latence-core pyproject); GPU/SOTA providers
  (gliner2, gliner_pii, sentence-transformers, endpoint parsers/embedders, lighton OCR)
  live in provider packages (reader D covers them).
- **`disambiguation.exact_surface` + `type_consolidation.exact_surface` are SHIPPED
  providers** — the linear exact-surface path the eval harness needed is productized;
  use it for the full-corpus S3/S4 runs.
- Export writes jsonl+parquet (+ optional embedding/sparse/multivector/FDE/bm25-postings
  columns/files, + optional Iceberg); KG export writes nodes/edges parquet + ttl + graphml
  + hyperedges parquet + LHKV1 KV (hyperedge_kv: true default).

## Key defaults worth surfacing in the wizard

chunk.markdown {max_tokens 512, overlap 64, min 16} (sentence_window IGNORES overlap);
intake screening {25 MiB, zip ratio 100, spoof check on}; content screening
quarantine_threshold 1.01 (flag-only); redaction 21 built-in detectors + actions
mask|replace|hash|tag + sensitivity policies + no-op floor counter; relation.pattern
window_chars 200 + fanout caps (10k/doc); disambiguation cascade thresholds
(min_conf .82, embedding .75, type-compat on); graph_completion.reference bounded 2-hop
{max_candidates 32}; context enrichment {4 neighbors, 6 triples}; profiling statistical
vs lightweight (skips O(n²) co-occurrence).

## Gotchas the wizard must encode

1. hyperedges only on graph.canonical; docstring omits it — code authoritative.
2. token_counter accepts ONLY "tiktoken:<enc>"; missing package = loud ConfigError.
3. keep_multivectors + fde_converter hard-require multivector_embedder.
4. ${REPO_ROOT} substitution is script-level (enterprise-gold precedent) — core has no env
   expansion; the campaign generator must substitute paths itself.
5. Secret scrubbing on manifest write; WAL is audit-only, never control flow.
6. Atomicity: publish-or-nothing per file group; S3 caveats documented in storage.py.
