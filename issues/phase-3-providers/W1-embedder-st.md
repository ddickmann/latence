# W1-embed — `latence-embedder-st`: a real sentence-transformers Embedder (Wave-1)

**Phase-3 Wave-1. GitHub issue #93 (P3-A-embed).** The first real learned **Embedder** adapter,
built through `AdapterBase` (H-A1 template) and gated by the conformance suite (F3). Fills a real
gap: today the only offline Embedder is the in-core `embedding.hashing` (a deterministic hash, not
semantic); this makes the exported RAG corpus carry **real semantic embeddings** a vector DB ingests,
and gives the F5 bake-off a real embedding matrix (`embedding.hashing` vs a learned model).

**A working plugin, not a stub:** the adapter really calls `sentence-transformers` and returns model
vectors when the dep + weights are present. Offline CI verifies it the SAME way the gliner trio is
verified — install the package `--no-deps`, monkeypatch the model with a deterministic fake for
conformance/unit tests, and gate the real-weights test on the dep being importable (skipped offline,
run where the model exists). This is the established honest boundary, not a mock-only implementation.

## What to build
Package **`latence-embedder-st`** registering `embedding.sentence_transformers`, an `AdapterBase`
subclass satisfying the `Embedder` Protocol (`dimension` property + `embed(texts) -> Iterator[list[float]]`,
capability.py:423).
- **Default model:** `intfloat/multilingual-e5-small` (**MIT**, 384-dim) — the model the `Embedder`
  docstring already names as the default. Config-selectable `model_id`; `MiniLM` (Apache-2.0) as a
  documented alternative. Heavy deps (`sentence-transformers`, which pulls torch/transformers) live
  ONLY in this package (ADR-0016) — nothing leaks into `latence-core`.
- **Exercise AdapterBase for real:** batch `encode()` over texts (input-order-preserving,
  streaming-friendly via the batching helper); device selection cpu|cuda through the shared seam
  (ST's `device=`); native exception (bad model id / OOM / encode failure) → typed `ProviderError`
  via the guard, **no raw text in the message**. If any AdapterBase facility misfits a real embedder,
  report it (as H-A1 did with `carry_provenance`) — that keeps validating the seam.
- **`dimension`** comes from the loaded model (e5-small = 384); the exported vectors are that width.
  Normalize per the model's convention (e5 wants the `query:`/`passage:` prefix + L2-normalize —
  follow the model card; document the choice).

## Profile + license + conformance (the gates)
- `ProviderProfile`: `compute="either"` (CPU-runnable, GPU-faster), honest `memory_mb`, `model_id` set,
  `deterministic=True` **only if** encode is deterministic on fixed input+seed (verify; ST inference is
  typically deterministic on CPU — confirm, else declare False), `batch=True`, `cost_per_1k=None`.
- **Evidence-bearing license (H-B1):** `license_code` = sentence-transformers SPDX (**Apache-2.0**,
  cited), `license_weights` = the model SPDX (e5-small **MIT**, cited HF model-card URL + date). Both
  verified or honestly `UNVERIFIED` — no guessing. Passes conformance **C3**.
- **Passes the P3-F3 conformance suite** (C1–C6) as a registered provider — under the offline CI it
  runs against the monkeypatched fake (deterministic vectors of the right width), exactly like the
  gliner providers. Roster count updates; the full conformance suite stays green.

## Wire-in
- CI: install `latence-embedder-st --no-deps` in the existing heavy-provider install step (mirrors the
  gliner packages); its tests monkeypatch `SentenceTransformer`. Add a real-weights test gated on
  `importlib.util.find_spec("sentence_transformers")` (skipped offline).
- Add a `matrix/embed.yaml` bake-off (`embedding.hashing` vs `embedding.sentence_transformers`); the
  ST candidate is dep/device-gated → skip-with-flag offline (never a fabricated number). Optionally add
  the learned embedder to `stacks/gpu-sota.yaml` (skip-with-flag on the CPU host).

## Definition of Done (Gate bar)
- `latence-embedder-st` builds, registers `embedding.sentence_transformers`, is an `AdapterBase`
  subclass, satisfies `Embedder`, and returns correct-width vectors (real model where present; the
  monkeypatched fake in offline CI). Heavy dep isolated (ADR-0016).
- Native failure → typed `ProviderError`, no text/PII in the message (C2); evidence-bearing license
  (C3); passes conformance C1–C6; the full conformance + stack-validate suites stay green.
- CI installs it `--no-deps` + monkeypatch tests green; a real-weights test exists, gated on the dep;
  `matrix/embed.yaml` runs (ST candidate skip-with-flag offline, hashing candidate real).
- `mypy --strict`/`ruff`/pytest green; deterministic seeded; ADR-0036 (embedder adapter) + a short
  docs note; the seam-report says whether AdapterBase fit a real embedder.

## Reviewer should check
- The adapter genuinely uses `sentence-transformers` (real `encode` path), not a permanent mock; the
  offline fake is only the CI test double (same discipline as gliner), and a real-weights test is
  present + gated.
- License is really verified for BOTH the library (Apache-2.0) and the e5-small weights (MIT) with
  cited sources; `deterministic` is declared truthfully (verified, not assumed).
- No heavy dep (torch/sentence-transformers) leaked into `latence-core`; conformance stays green.
- The `matrix/embed.yaml` ST row is skip-with-flag offline (no fabricated quality/throughput), the
  hashing row is real; e5 prefix/normalization follows the model card.
