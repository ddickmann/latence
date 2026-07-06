# S1 — Walking skeleton

**Parent**: none · **Phase 1 · Baseline bar**

## What to build
The thinnest complete pipeline that exercises every architectural seam. `latence run pipeline.yaml` reads documents from a local-folder **Source**, runs a passthrough **Parse** (plain-text/markdown in → records out), and **Export**s records to JSONL and Parquet on Storage — producing a skeleton **Quality Report**. This forces the whole architecture into existence at once: `latence-core` contracts, the Capability protocol + entry-point Provider registry, the declarative YAML **Pipeline**, the local **Runner** (single-node DAG execution with per-Stage checkpoint/resume), the fsspec **Storage** seam, packaging as a uv workspace, and CI.

## Ports (algorithmic assets)
- Write-Ahead Log (`dataset-intelligence/state/write_ahead_log.py`) → Runner checkpoint/commit.
- Retry with jitter → use `tenacity` (EXTRACT-ALGORITHM-ONLY).

## Acceptance criteria
- [ ] `latence run` executes a YAML pipeline of Source→Parse→Export against a folder of `.txt`/`.md`.
- [ ] Records are versioned Pydantic contracts carrying `Provenance` (file name/type/size/offsets) from the first Stage.
- [ ] Runner writes a per-Stage checkpoint; killing mid-run and re-running resumes without redoing completed Stages; re-running a completed pipeline is a no-op.
- [ ] Storage works against `file://` and (in CI) a MinIO `s3://` with no code change.
- [ ] A skeleton Quality Report (doc counts, per-Stage timing, contract-completeness check) is emitted.
- [ ] `latence-core` installs with near-zero deps; a second workspace package (`latence-parser-plaintext`) registers its Provider via entry points and is discovered dynamically.
- [ ] CI runs lint + type(strict) + tests + build and is green.
- [ ] Baseline bar met.

## Blocked by
None — can start immediately.
