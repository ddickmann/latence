# Public-release gates

This repository is **private**. It is not published publicly until every gate below is green.

The gates exist because the failure mode they prevent is specific and expensive: publishing a
framework whose headline claims a sophisticated evaluator can falsify in an afternoon. A capability
that is *designed* is not a capability that *works*, and the distance between them is exactly what
an evaluator is looking for.

**These are machine-checked.** `packages/latence-core/tests/test_public_release_gate.py` runs on
every build. Ordinarily it verifies the gate documents are well-formed and inventories what is
outstanding. Under `LATENCE_RELEASE_CHECK=1` it becomes strict and **fails while any gate is
unmet** — so a release cannot proceed by forgetting.

```bash
# ordinary development — inventories what is outstanding, does not fail
PYTHONHASHSEED=0 uv run pytest packages/latence-core/tests/test_public_release_gate.py -o addopts=""

# the release check — fails while any placeholder remains
LATENCE_RELEASE_CHECK=1 PYTHONHASHSEED=0 uv run pytest \
  packages/latence-core/tests/test_public_release_gate.py -o addopts=""
```

---

## Gate 1 — Measured end-to-end numbers

**Blocked on:** a real GPU run per [`issues/e2e-extended/RUNBOOK.md`](issues/e2e-extended/RUNBOOK.md).

Every `PENDING:` placeholder in [`docs/BENCHMARKS-E2E.md`](docs/BENCHMARKS-E2E.md) must be replaced
with a measured value. Specifically:

- [ ] **Pipeline throughput** — wall-clock and peak VRAM per stage, on stated hardware
- [ ] **Device-honesty check** — the JIT left enabled, FlashInfer not forced, CUDA graphs captured,
      torch.compile active. Without this the throughput numbers were measured under a handicap.
- [ ] **Retrieval per-signal + fused** — dense, SPLADE, BM25, graph, each alone and fused
- [ ] **Hop-stratified results** — by hop count *and* shape, never pooled
- [ ] **The question-set funnel** — enumerated → structural → G3 → G4 → human-audited
- [ ] **The KG ablation** — with and without the graph leg
- [ ] **Link prediction** — measured precision on our own held-out sample, and the end-to-end
      retrieval delta

**Do not fill a placeholder with a number from a different corpus, a different commit, or a
different card.** Each table states its method and its hardware; a number that does not match its
stated method is worse than a missing one, because it looks checked.

## Gate 2 — Green code, on real dependencies

- [ ] `bash scripts/verify-local.sh` exits **0** — captured by **redirect**, never by piping to
      `tail` (a pipeline returns the last command's status and will report green on a red tree)
- [ ] `mypy --strict` passes on src **and** tests for every package
- [ ] The full suite passes with `PYTHONHASHSEED=0`
- [ ] **The environment-gated lanes actually ran** — live Qdrant search, GPU providers, s3. These
      skip locally. If they still skip on the pod, the environment is wrong; a skip is not a pass.
- [ ] SPLADE proven as a **live retrieval signal** against a real Qdrant — a sparse-only query
      returning documents a dense-only query does not
- [ ] The graph leg proven to **retrieve**, not merely reorder — a gold document unreachable by
      dense and sparse, recovered by traversal

## Gate 3 — Licence coherence

- [ ] `LICENSE` is the intended licence, reproduced verbatim from its canonical source
- [ ] `COMMERCIAL.md` states how commercial terms are obtained
- [x] **All 33 `packages/*/pyproject.toml`** declare the same SPDX identifier. Met, and it stays met
      without anyone re-reading this line: `test_public_release_gate.py`'s
      `test_release_the_licence_is_declared_consistently_across_every_package` fails the release
      check on any divergent package, and it has a negative control proving the check still fires.
      Verified by hand at the time of ticking —
      `grep -h "^license = " packages/*/pyproject.toml | sort | uniq -c` → `33
      license = "PolyForm-Noncommercial-1.0.0"`
- [ ] `THIRD-PARTY-LICENSES.md` is re-verified under the new posture, including the **LGPL-3.0
      `easydict`** obligation on the opt-in `latence-linkpred-ultra[ultra]` path
- [ ] `CONTRIBUTING.md` states the CLA requirement
- [ ] **The benchmark corpus's own licence flags are coherent.**
      `latence_benchmark/redaction_suite.py` declares the synthetic PII corpus
      `dataset_license="Apache-2.0", code_license="Apache-2.0", commercial_use=True`, and those
      values render into `docs/BENCHMARKS.md`. Deferred deliberately, because it is not a mechanical
      swap: setting PolyForm while leaving `commercial_use=True` states two contradictory things,
      and setting `commercial_use=False` gates the project's OWN corpus behind
      `--accept-non-commercial` in `licensing.require_usable`. **Owner decision, not a docstring.**
- [ ] **`latence-splade`'s `license_code` resolved.** Its prose says "the package code and the
      `transformers` library are Apache-2.0". The `transformers` half is true; the package-code half
      is not, post-relicence. `ProviderProfile.license_code` is single-valued, so which claim it
      carries is a semantics decision. Pinned by `test_splade_embedder.py:309`.
- [ ] **Legal review completed** — the licence choice, the LGPL interaction, any prior PyPI
      publication, the CLA text, and the **deliberately retained Apache-2.0 subset** in
      `latence-retrieval` (7 modules keep an outbound Apache grant by decision, not oversight; a
      scanner will flag the mismatch against the PolyForm package metadata, so the exception must be
      one the reviewer has actually seen). This gate is not satisfiable by an automated check.

## Gate 4 — Documentation truthfulness

- [ ] Every command in `README.md`, `docs/quickstart.md` and `docs/tutorials/*` executes as written
      — verified against the real CLI, not assumed
- [ ] Every number is derived from the repo
- [ ] No capability is claimed that is partial or unbuilt, unless labelled as such
- [ ] The doc-truth gates in the suite pass
- [ ] `README.md` links to `docs/BENCHMARKS-E2E.md` rather than restating numbers, so results live
      in exactly one place

## Gate 5 — Corpus and data posture

- [ ] The corpus manifest records per-source licence and the **measured** file-type histogram
- [ ] Nothing derived from a fair-use-only source (IIT-CDIP, UCSF IDL) appears in any public
      artifact
- [ ] No key, token, or credential in the tree — `detect-secrets` clean
- [ ] Nothing derived from the corpus republishes third-party PII

---

## Why this file exists

Two of the four capabilities most likely to be quoted from a README — SPLADE as a fused signal, and
graph traversal as a retrieval leg — were, at the time this file was written, **structurally unable
to work**: the sparse index was never built despite the query path existing, and the graph could only
reorder a candidate pool it had not chosen. Both are now implemented; neither is yet *proven on real
data*.

That gap is the thing this file is for. A framework is judged on the claims it cannot support, not
the ones it can.
