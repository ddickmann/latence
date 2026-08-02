# S2 — OCR → markdown normalization (LightOnOCR-2-1B served on the pod)

Type: task
Status: claimed
Blocked by: s1-corpus (approved 2026-08-02)

## Scope decision (Dennis, 2026-08-02)

**Capped fin: ~56k pages.** OCR everything except UDA-financial, which is capped to the
top-150 most-questioned docs (covers 4,132 of 8,190 fin QA = 50%; selection computed from
`uda/raw/fin/test_*.parquet` doc_name counts, deterministic tie-break by name, recorded in
stage metrics). The uncapped remainder is resumable later. Volumes: OHR-Bench ~8.5k +
ViDoSeek ~5.5k + UDA-papers ~20k + UDA-fin-capped ~22k pages. Est. 10–31h pod wall-clock at
0.5–1.5 pages/s.

## Evidence-decided spec items (no open questions)

1. **HTML-table normalization is mandatory**: LightOnOCR-2 emits tables as HTML inside
   markdown; the chunker's `strip_markup` deletes raw HTML tags keeping inner text only, so
   unnormalized tables would lose ALL row/column structure downstream. Post-process each
   page: `<table>` → GitHub markdown table (deterministic, stdlib html.parser; colspan
   flattened with repeated cells; nested tables linearized; failures logged + page kept
   with the raw HTML intact rather than dropped).
2. **Serving**: vLLM ≥0.11.1 on the pod, model `lightonai/LightOnOCR-2-1B`, card flags
   (`--limit-mm-per-prompt '{"image": 1}' --mm-processor-cache-gb 0
   --no-enable-prefix-caching --gpu-memory-utilization 0.85 --max-model-len 8192`),
   sampling temp 0.2 / top_p 0.9.
3. **Render**: pypdfium2 at 200 DPI, longest edge 1540 px, JPEG; render is a separate
   resumable pass from inference. Hard cap 25 megapixels/page (existing rasterizer caps).
4. **Resumability**: file-based manifest — one `pages/<doc_id>/<page:04d>.md` per page,
   skip-if-exists; independent per-page HTTP requests (concurrency 8–16 on the 16GB card);
   failures log with per-page retry (1 retry on n-gram repetition flag: 8-gram repeated
   >4× in a page); a crash loses at most in-flight pages.
5. **Assembly**: per-doc `markdown/<doc_id>.md` = page join with `\n\n` + provenance
   header comment (doc id, source dataset, page count, ocr model+revision). Text-native
   datasets (musique, hotpotqa, wiki2multihop, multihop_rag) pass through untouched —
   they are already text/markdown; recorded as `skipped_native` in metrics.
6. **Gate metrics** (`stages/s2-ocr/metrics.json`): per dataset — docs, pages rendered,
   pages OCR'd, failures, repetition retries, HTML tables normalized, wall-clock,
   pages/s; plus 12 randomly-sampled page images + their markdown for Dennis's visual
   spot-check (sample list in metrics). The OHR-Bench scoreboard comparison (GT 70.4 /
   best-OCR 59.6 retrieval) lands in S5 where retrieval numbers exist — S2's gate is
   conversion completeness + spot-check quality.
7. **Supervision**: pod-side supervisor script (supervise.sh shape: done-probe = manifest
   coverage, progress-probe = page-count monotone, stall budget, atomic state), driven
   from the campaign CLI; local watchdog uses byte/count deltas (never `find -newermt` —
   the macOS false-alarm from S1 is the negative prototype).

## Deliverables

1. `latence_benchmark/campaign/ocr_stage.py` — the s2-ocr stage: fin-cap selection,
   zip extraction, transfer orchestration (rsync over the configured pod_ssh), pod job
   launch/resume, result sync-back, metrics + gate artifact. All remote interactions
   behind a seam (offline tests with fakes).
2. `benchmark/sota-campaign/pod/` — pod-side scripts: `serve_ocr.sh` (vLLM daemon with
   readiness probe), `ocr_worker.py` (render + request + normalize + assemble, pure
   file-manifest resumable), `podcheck_s2.sh` (status probe).
3. Campaign CLI: s2-ocr registered after s1-corpus (gate: only runs when s1 approved).
4. Tests: fin-cap selection determinism, HTML-table→md normalization (incl. colspan,
   nested, malformed), repetition detector, manifest resume logic, stage gating.

## Constraints

- Licenses: OCR'd markdown inherits each dataset's lane — all derived markdown stays
  under gitignored `benchmark/sota-corpus/<ds>/markdown/`; nothing committed.
- No fake green: the gate passes only on the real pod run's metrics.
- Pod is shared: check free VRAM before serving; stop the vLLM daemon after the stage
  completes (it blocks S3's GLiNER2 otherwise).
