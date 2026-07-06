# S4 — Entity Extraction

**Parent**: S3 · **Phase 1 · Baseline bar**

## What to build
An **Entity Extraction** Capability that finds typed entity mentions in chunked text (zero-shot NER), with one CPU-viable reference Provider. Mentions carry offsets and Provenance back to the source page.

## Ports (algorithmic assets)
- None to port (new); reference Provider wraps a permissively-licensed GLiNER checkpoint via plain `transformers`/ONNX (CPU path, no vLLM).

## Acceptance criteria
- [x] EntityExtractor Capability protocol defined; ≥1 CPU reference Provider (license verified).
- [x] Emits typed mentions with confidence, char offsets, and Provenance resolving to the correct source page.
- [x] Label set is configurable per Pipeline (zero-shot); no label set hardcoded in core.
- [x] Confidence distribution + per-type counts land in the Quality Report.
- [x] Empty/near-empty chunk handled without error.
- [x] Baseline bar met.

## Delivered
- Contracts: `EntityMention` (a first-class inter-Stage `Record`; `SCHEMA_VERSION` → 4),
  with Provenance-span enforcement (char + page required) at the boundary.
- Capability: `EntityExtractor` protocol; `CapabilityKind.ENTITY_EXTRACTION`; wired
  into the local Runner (execute, checkpoint/resume, Export pass-through carrier) and
  the Pipeline DAG.
- Providers: `entity.gazetteer` (in `latence-core`, deterministic zero-shot
  gazetteer/regex, no deps) and `entity.gliner` (new package `latence-ner-gliner`,
  learned GLiNER via transformers/ONNX; Apache-2.0 weights `urchade/gliner_multi-v2.1`
  + Apache-2.0 code, verified 2026-07-06 per ADR-0012). Both registered via the
  `latence.providers` entry-point group.
- Quality Report: `EntityQuality` — per-type mention counts, a fixed-bin confidence
  histogram, and the offset-resolution invariant — emitted into the report.
- Tests: gazetteer unit (happy-path + failures + determinism), EntityMention/quality
  contract unit, GLiNER-provider unit (offline via a monkeypatched fake dep), and an
  end-to-end run through the local Runner (offset round-trip → source page, empty-doc
  handling, determinism, resume). Local ruff / mypy --strict / pytest all green.
- Docs: ADR-0022; core README S4 section; `examples/entity-slice.yaml`.

## Blocked by
S3.
