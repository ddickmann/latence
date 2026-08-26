# Release gates

**Publication and release are two different gates** ([ADR-0066](docs/adr/0066-publication-and-release-are-separate-gates.md)).

* **The source is public**, as a pre-1.0 preview. That bar is: every claim this repository makes is
  either substantiated or explicitly labelled as unsubstantiated. It is met.
* **A `v*` release — a tag, signed artifacts, an SBOM, something people install — requires every
  gate below**, Gate 1 included. Gate 1 is **not** met, and nothing below pretends otherwise.

The gates exist because the failure mode they prevent is specific and expensive: shipping a
framework whose headline claims a sophisticated evaluator can falsify in an afternoon. A capability
that is *designed* is not a capability that *works*, and the distance between them is exactly what
an evaluator is looking for. That failure is **overclaim**, which is why the honest-labelling
discipline is what gates publication, and measurement is what gates a release.

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
- [x] `mypy --strict` passes on src **and** tests for every package
- [x] The full suite passes with `PYTHONHASHSEED=0`
- [ ] **The environment-gated lanes actually ran** — live Qdrant search, GPU providers, s3. These
      skip locally. If they still skip on the pod, the environment is wrong; a skip is not a pass.
- [ ] SPLADE proven as a **live retrieval signal** against a real Qdrant — a sparse-only query
      returning documents a dense-only query does not
- [ ] The graph leg proven to **retrieve**, not merely reorder — a gold document unreachable by
      dense and sparse, recovered by traversal

## Gate 3 — Licence coherence

- [ ] `LICENSE` is the intended licence, reproduced verbatim from its canonical source
- [ ] `COMMERCIAL.md` states the licence's commercial position accurately (post-ADR-0063: granted
      outright, no agreement) and leaves no dangling inbound link
- [x] **All 33 `packages/*/pyproject.toml`** declare the same SPDX identifier. Met, and it stays met
      without anyone re-reading this line: `test_public_release_gate.py`'s
      `test_release_the_licence_is_declared_consistently_across_every_package` fails the release
      check on any divergent package, and it has a negative control proving the check still fires.
      Verified by hand at the time of ticking —
      `grep -h "^license = " packages/*/pyproject.toml | sort | uniq -c` → `33
      license = "Apache-2.0"` (re-verified after the ADR-0063 relicence back to Apache-2.0)
- [ ] `THIRD-PARTY-LICENSES.md` is re-verified under the new posture, including the **LGPL-3.0
      `easydict`** obligation on the opt-in `latence-linkpred-ultra[ultra]` path
- [x] `CONTRIBUTING.md` states the CLA requirement
- [x] **The benchmark corpus's own licence flags are coherent.** Met by ADR-0063, which dissolved
      the contradiction rather than resolving it: `latence_benchmark/redaction_suite.py` declares the
      synthetic PII corpus `dataset_license="Apache-2.0", code_license="Apache-2.0",
      commercial_use=True`, and with the repository back on Apache-2.0 all three are true together.
      Under PolyForm this was deferred because no combination was honest — PolyForm with
      `commercial_use=True` stated two contradictory things, and `commercial_use=False` would have
      gated the project's OWN corpus behind `--accept-non-commercial`.
- [x] **`latence-splade`'s `license_code` resolved.** Same dissolution: its prose says "the package
      code and the `transformers` library are Apache-2.0", and post-ADR-0063 both halves are true, so
      the single-valued `ProviderProfile.license_code` has one correct value. Pinned by
      `test_splade_embedder.py:309`. (Its *weights* — `naver/splade-v3`, CC-BY-NC-SA-4.0 — are
      unaffected and stay behind `--accept-noncommercial-license`.)
- [ ] **Legal review completed** — the LGPL interaction, any prior PyPI publication, and the
      contributor position (whether to keep `CLA.md` or move to a DCO, now that Apache-2.0's
      inbound=outbound removes the reason the CLA was written — ADR-0063). The licence *choice* no
      longer needs review in itself: Apache-2.0 is reproduced verbatim and digest-pinned, and the
      retained-Apache-subset exception a reviewer would have had to see is gone with the
      noncommercial model that needed it. This gate is not satisfiable by an automated check.

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
