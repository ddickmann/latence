# Run & validate a stack (P3-F4)

A **stack** is a complete, runnable Pipeline config for the whole AI-ready-data spine
(Source → Intake-Screening → Parse → Chunk → Content-Screening → NER → RE → Redaction →
Profiling → Disambiguation → Graph-Assembly → Export), distinct from the per-stage
`examples/*.yaml` slices. Stacks live in the top-level `stacks/` directory. Architecture:
ADR-0036 §3.

## The two blessed stacks

| Stack | What it is | Providers |
|---|---|---|
| `stacks/default.yaml` | The **blessed default** — CPU-first, offline, permissive, deterministic (ADR-0007/0012). What an adopter runs out of the box. | In-core deterministic Providers + the real CPU adapter where it adds value (`parser.pdfplumber`, `chunk.markdown`, `entity.gazetteer`, `relation.pattern`, `redaction.hybrid_rule` (comprehensive-by-default, H-C1), `profiling.statistical`, `disambiguation.cascade`, `graph.canonical`, `export.knowledge_graph` + `export.jsonl_parquet`, both Screening checkpoints). |
| `stacks/gpu-sota.yaml` | The GPU / high-accuracy stack. | ONE **fused** Entity+Relation Stage (`relation.gliner_relex` — joint NER+RE in one pass), learned `redaction.gliner_pii`, and the served `embedding.endpoint` for the RAG export. On a CPU-only host these Stages are **skipped-with-flag** — never faked. |
| `stacks/gpu-sota-pod.yaml` | The **self-contained GPU-pod** variant of `gpu-sota.yaml`. | Identical to `gpu-sota.yaml` except the RAG-corpus embedder is the **in-process** `embedding.sentence_transformers` instead of the served `embedding.endpoint` — so the whole stack runs on a single GPU pod with no external service to stand up. This is the stack the maintainer runs to validate the fused SOTA path end-to-end (`LATENCE_CUDA=1 latence stack validate stacks/gpu-sota-pod.yaml`). |

### The fused extraction Stage (why extraction is ONE Stage, not two)

`relation.gliner_relex` is a **Fused Provider**: it fulfils *both* Entity Extraction and Relation
Extraction in one pass (`extract_fused`), satisfying the `FusedEntityRelationExtractor` Capability —
**not** the standalone `RelationExtractor.relate` seam. So the SOTA stacks declare a single
`fused_entity_relation` Stage (`extract`), and the Runner routes its `FusedExtraction` (mentions +
relations) to *both* downstream consumers (Profiling, Disambiguation). Pairing it with a separate
`entity.gliner` Entity Stage would double-run NER, and declaring it under a `relation_extraction`
Stage raises `ContractError` at run (it does not satisfy that seam). A stack that wants standalone
learned NER can still wire `entity.gliner` + a non-fused RE provider; the blessed SOTA path is the
single fused pass. The CPU **capability-satisfaction guard** (`test_stack_capability_guard.py`)
statically checks this wiring on every push — see below.

`stacks/<client>.yaml` is the documented extension point: copy any stack and swap Providers.

## Running a stack directly

A stack YAML uses two placeholders so it carries no machine-specific path: `{STORAGE_URI}` (the
run's state/export root) and `{CORPUS_DIR}` (the Source folder). To run a stack against your own
data with `latence run`, replace them with a Storage URI and a folder of documents:

```bash
sed -e 's#{STORAGE_URI}#file:///tmp/my-run#' -e 's#{CORPUS_DIR}#file:///path/to/my/docs#' \
    stacks/default.yaml > /tmp/my-stack.yaml
latence run /tmp/my-stack.yaml
```

## Validating a stack — `latence stack validate`

`latence stack validate <stack.yaml>` runs a named stack **end-to-end on a small bundled corpus**
and asserts the G1-style checks in one command — the G1 gate's E2E / reliability lanes generalized
from a one-off gate into a reusable, standing gate ("Goal A: test the default stack, one command").

```bash
latence stack validate stacks/default.yaml
# → a pass/fail table per check; exit 0 when green, non-zero on any failure (gates CI/publish).
```

It is **composition** over the existing seams — `LocalRunner` + `QualityReportBuilder` + the
device/skip seam — not new pipeline logic. The checks:

| Check | Asserts |
|---|---|
| `contracts` | Every record carries Provenance + Classification; offsets align; zero drift (reuses the `QualityReportBuilder` completeness check + the ADR-0031 drift roll-up). |
| `kg` | A non-empty, content-addressed knowledge graph with per-edge Evidence is produced and exported (Parquet/TTL/GraphML). Skip-clean when the stack declares no Graph-Assembly stage. |
| `rag` | The redacted, chunked corpus is exported (JSONL/Parquet), well-formed, and carries **no raw PII** (redaction is load-bearing — a bypass FAILs here). |
| `report` | A schema-valid Quality Report is produced, counts-only (no PII), with per-stage metrics + the device/license/profile columns (P3-F1). |
| `determinism` | A second run under a fresh `run_id` produces a **byte-identical** export (the H-E1 equality discipline). |
| `resume` | Re-running the same `run_id` resumes from checkpoints and reproduces a byte-identical export. |
| `graceful_failure` | The planted malformed inputs are quarantined / PARSE_ERROR'd, never crashing the run. |
| `device_honesty` | On a CPU-only host a GPU stack's learned Stages are **skipped-with-flag** (recorded, never faked); the CPU-reachable spine still validates. |

A signed-style summary (like `G1-RESULTS.md`) is written to Storage under
`<storage_root>/_latence/stack-validation/<stack>.md`. Pin `--storage-root` to inspect the run's
artifacts after the fact:

```bash
latence stack validate stacks/default.yaml --storage-root file:///tmp/stack-run
```

## Device honesty on a CPU-only host

`latence stack validate stacks/gpu-sota.yaml` on a machine with no CUDA (or without the heavy
Provider packages installed) reports the learned Stages as **skipped-with-flag** — the fused
`extract` (`relation.gliner_relex`), `redact` (`redaction.gliner_pii`), the served-embedder RAG
export, and everything that depends on them — and still validates the CPU-reachable spine
(Source → … → Content Screening). No fabricated quality/throughput numbers ever appear (#64). On a
real GPU host with the packages installed, the learned Stages run (`device: cuda`) and the full
spine validates.

### The capability-satisfaction guard (CPU-visible wiring check)

Because a CPU-only `stack validate` *skips* every `device: cuda` Stage, the Runner's dispatch-time
`isinstance(provider, <Protocol>)` check — the one that raises `ContractError` for a mis-wired Stage
— never runs for a GPU Stage in CI. That is how the fused/standalone miswiring (a Fused Provider
under a `relation_extraction` Stage) once hid until a real pod run. `check_stack_capabilities`
(`latence_core.stacks.capability_guard`) closes the gap: it loads each Provider **class** (a
torch-free import — the heavy dep is deferred) and structurally checks it against the Protocol its
Stage's Capability declares, the SAME mapping the Runner enforces. `test_stack_capability_guard.py`
asserts the committed stacks have zero findings **and** that the guard bites on the exact miswiring
— so a fused-under-relation slip fails CI on a CPU host, never reaching a pod.

## CI

CI runs `latence stack validate stacks/default.yaml` (and `stacks/gpu-sota.yaml` /
`stacks/gpu-sota-pod.yaml`) on every push with `LATENCE_CUDA=0` — the blessed stacks are proven
end-to-end (the real pipeline, not a config-parse smoke) on a tiny corpus, offline and
deterministic, as a standing gate — plus the CPU capability-satisfaction guard over every stack's
wiring.
