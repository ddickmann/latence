# Changelog

All notable changes to the Latence Framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project (all publishable `latence-*` distributions) adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). All packages in the
workspace are versioned and released together under one version number.

## [Unreleased]

**Everything in this file is unreleased.** No `v*` tag has been cut and no distribution has ever
been published, so this section is not a backlog sitting on top of shipped versions — it *is* the
history. It is large for that reason. When the first tag is cut (`docs/RELEASING.md`), this block
becomes that release's section and the cadence starts from there; until then there is no cadence to
report, and claiming one would be the kind of thing `RELEASE-GATES.md` exists to stop.

### Removed

- **The query-time retrieval tooling is gone (ADR-0068).** Four distributions leave the
  workspace — `latence-retrieval`, `latence-sink-qdrant`, `latence-splade` and `latence-muvera`
  — and with them everything in `latence-core` that existed only to feed them: Hyperedges
  (`GraphHyperedge`, `HyperedgeRelation`, `HyperedgeSourceRef`, `GraphScope.HYPEREDGE`, the
  hyperedge KV sidecar), the Context Enrichment Stage and `ContextHeader`, the graph-feature
  computers, the BM25 / sparse / multi-vector / FDE export signals and their Capability
  Protocols (`ContextEnricher`, `GraphFeatureComputer`, `SparseEmbedder`,
  `ComposingSparseEmbedder`, `MultiVectorEmbedder`, `FdeConverter`, `Tokenizer`), the
  `latence retrieval` CLI group, the `context_columns` / `sparse_embedder` /
  `multivector_embedder` / `fde_converter` / `keep_multivectors` / `bm25` Export keys, the
  `graph_features` / `hyperedges` Graph Assembly keys and the `hyperedge_kv` KG Export key, the
  three retrieval-only stacks, the S5 retrieval matrix and the Qdrant serving lane.
  **`SCHEMA_VERSION` is 22 and `QUALITY_SCHEMA_VERSION` is 21**: both are removals from
  `extra="forbid"` models, so a v21 chunk record (which always carries a `context_header` key)
  and a v20 Quality Report that carries `context_enrichment` or `graph.hyperedges` no longer
  validate. Graph Assembly emits nodes and edges exactly as before — same ids, same fields, same
  order. Embedder Providers stay, as a signal of the resolution phase; the corpus Export still
  accepts an opt-in `embedder` block.
- **`musique` and `hotpotqa` left the benchmark campaign's dataset registry.** The campaign had
  no pipeline for either. `KNOWN_DATASETS` now holds five names, and a test requires each of them
  to have an acquisition plan in `s1-corpus` and a pipeline in `s3-pipeline`. A `campaign.yaml`
  that lists either removed name is rejected as an unknown dataset. Gone with them: the
  `local-reference` acquisition kind, the `attribution` lane and the `not_processed` list in the
  `s3-pipeline` metrics. The committed records under `benchmark/sota-campaign/stages/` are
  unchanged and still name both.
- **Eight benchmark scripts that could no longer run.** The four under `benchmark/official/` and
  `benchmark/musique-subset/run_eval_final.py` imported `latence_retrieval`; the three under
  `benchmark/official-data/` read a hyperedge export the current pipeline does not write. Their
  result files, question sets and export data stay as dated records, and each folder carries a
  note saying so.

### Changed

- **The repository moved to `ddickmann/latence`, and 200 references to the old slug moved with
  it.** Publication is from a repository created clean rather than from one whose history had been
  rewritten, so no unreachable-but-uncollected objects survive anywhere. Two of the 71 files
  carrying the old slug were not cosmetic: `README.md`'s clone command pointed at a repository the
  reader cannot see (`git clone …/latence-framework` 404s for everyone but the maintainer) and the
  CI badge pointed at a private repository's Actions, which renders permanently broken on a public
  page. **`scripts/release-integrity.sh` was the serious one** — it defaulted the Sigstore
  verification identity to the old slug, so a release cut from the new repository would sign under
  one identity and be verified against another, failing the one workflow whose entire job is
  proving a release is authentic. Only URL-shaped occurrences and `cd` targets were rewritten: the
  product is still the Latence Framework and the 34 distributions are still `latence-*`, because
  what changed is the repository slug, not the name.
- **A logo.** `docs/assets/logo-{light,dark}.svg` and `mark.svg` — three ragged document rules
  resolving into a canonical graph, which is the pipeline in one glyph. The README serves the
  right variant per colour scheme via `<picture>`; the mark alone is the docs-site logo and
  favicon.
- **Publication and release are separate gates
  ([ADR-0066](docs/adr/0066-publication-and-release-are-separate-gates.md), amending
  `RELEASE-GATES.md`).** That page previously opened "This repository is **private**. It is not
  published publicly until every gate below is green," and Gate 1 — the end-to-end benchmark
  protocol on named hardware — is genuinely unmet, with 52 `PENDING:` tokens on
  `docs/BENCHMARKS-E2E.md`. Publishing under that sentence would have falsified the page on its
  own first line, in a repository whose distinguishing discipline is that documentation making a
  claim about the code is machine-checked. The gate's purpose is preventing *overclaim*, and that
  survives the split: **publication** of the source requires only that every claim is
  substantiated or explicitly labelled as unsubstantiated, which is met and enforced; **a `v*`
  release** still requires every gate, Gate 1 included, and the `pypi-publish` job stays disabled.
  Holding the source private was never going to make an unmeasured number measured — the two facts
  are independent, and coupling them cost the only thing publication buys a bus-factor-1 project.
- **The strict release check now runs on the path it gates.** `LATENCE_RELEASE_CHECK` was
  documented as the enforcement mechanism — `README.md` claimed publishing with an unfilled table
  "is not possible" — while appearing in **no workflow, no script and not the Makefile**. It failed
  only if a human remembered the flag. It now runs in `release.yml` before anything is built, so a
  red gate stops a tag rather than decorating a job summary after the artifacts already exist.
  Three gate items that were verifiably true but never ticked are ticked; Gate 1 is left unticked,
  because it is not met.
- **The repository is prepared for public launch: internal material out, an agent-facing entry
  point in.** Five maintainer-internal documents left the shipping tree — `docs/GO-LIVE-RUNBOOK.md`
  (which named the scrubbed competitor review *and* the commit it remains reachable at, and
  disclosed an open supply-chain gap), `docs/PORTING-MAP.md` and `docs/ASSET-INVENTORY.md` (which
  named four private predecessor repositories, a hosted product and a vendor bill-optimisation
  hack, and costed the reinvention of that IP), `docs/PITCH-SLIDES.md` (sales speaker notes,
  already out of the nav but still published on GitHub) and `docs/audits/S1-architecture-audit.md`
  (whose asks had all shipped — its `ParserInput` recommendation is ADR-0019 — so it read as a
  stalled to-do list). **Nothing was merely de-linked:** the release-process content the runbook
  carried moved into `docs/RELEASING.md` under a new *Cutting the tag* section, the three
  `test_release_integrity.py` guards that policed that prose were re-anchored to it with every
  assertion intact, and ADR and source-comment references were rewritten to stand alone — the
  borrowed fact inlined, the pointer dropped — because an ADR records why a decision was made and
  is not revised by a scrub. The signed G1 evidence was promoted out of `issues/` into
  `docs/evidence/`, where a reader can actually reach it.
- **The README is a front door again — 554 lines to 230** — and it opens with a pointer for coding
  agents, which is how most readers now arrive. Two new files serve them: `llms.txt` (the
  `llmstxt.org`-format index) and `docs/agent-briefing.md`, which is written for an LLM agent
  rather than adapted from human docs — the mental model, every Provider available at each Stage
  with the CPU-first default marked, the pipeline YAML contract, the CLI, the output schema, the
  retrieval wiring and the traps. `AGENTS.md` now routes between the two audiences; it previously
  had no use-the-framework path at all, and its one gesture at one pointed to a page whose first
  command was a broken `pip install`.
- **110 MB of unread derived data left the tracked tree.** `benchmark/musique-subset/export/` held
  eight files; `run_eval_final.py` opens exactly two of them. The other six had no reader anywhere
  in the repository and no committed configuration that reproduces them, so they were a snapshot of
  one historical run rather than a fixture. The two that are read stay tracked, so the harness still
  runs from a clean clone, and `SOURCE.md` now states which is which. Removing them made a security
  gate's own justification stale — `scripts/secret_scan_gate.py` excludes `*.graphml` and cited a
  file that no longer exists — so that reasoning was rewritten rather than left asserting something
  false about the tree.

### Fixed

- **`make sync` produced a broken environment, and had for some time.** The heavy/endpoint Provider
  list existed as three hand-synchronised copies — `Makefile`, `scripts/verify-local.sh` and
  `.github/actions/setup-workspace/action.yml`, the last carrying a comment promising it was "kept
  byte-identical" with nothing enforcing it. The Makefile copy had drifted to **9 of 21** entries,
  so a contributor who followed `AGENTS.md` (`make sync`, then `make test`) got a red suite for
  environment reasons: twelve Providers never registered their entry points, which surfaces as
  *registry* failures rather than as an import error and so does not read as a setup problem. The
  list is now single-sourced in `scripts/heavy-packages.txt`, read by all three, and gated by
  `packages/latence-core/tests/test_heavy_package_list.py` — which asserts the stronger invariant
  the copies violated: that **every** workspace package is reachable by one of the documented
  install paths, or carries a reasoned exemption whose stated mechanism is verified to still exist.

- **The outbound licence is Apache-2.0 again
  ([ADR-0063](docs/adr/0063-relicence-to-apache-2-0.md), superseding
  [ADR-0055](docs/adr/0055-relicence-to-polyform-noncommercial.md)).** The workspace relicensed
  from Apache-2.0 to PolyForm Noncommercial 1.0.0 on 2026-07-29 and back to **Apache-2.0** on
  2026-08-15. Commercial use is now granted by the licence outright: there is no agreement to
  obtain and nothing to buy. The reason is the adopter — this is enterprise data infrastructure,
  corporate SCA scanners hard-block the `PolyForm-Noncommercial-1.0.0` identifier as a matter of
  policy, and "ask us for a commercial licence" is not a step anyone takes for an unproven tool;
  even a corporate *evaluation* was, strictly, unlicensed. Apache-2.0 rather than MIT for the
  express §3 patent grant and its defensive termination, which is what enterprise legal reads
  past the identifier. The switch is clean and was cheap only right now: nothing published,
  no tag, no PyPI, sole copyright holder, no CLA signed — so nobody holds a grant under a
  superseded state. What changed concretely: `LICENSE` is the canonical Apache-2.0 text byte for
  byte (nothing added — the provenance record moved to `NOTICE`, which Apache-2.0 §4(d) carries
  downstream anyway), all 33 `packages/*/LICENSE` copies and `pyproject.toml` declarations, all
  129 `SPDX-License-Identifier` headers, and the prose in `README.md`, `NOTICE`,
  `THIRD-PARTY-LICENSES.md`, `CITATION.cff`, `AGENTS.md`, `CONTRIBUTING.md`, `CLA.md`,
  `RELEASE-GATES.md`, `mkdocs.yml` and the published docs pages.
- **The retained Apache-2.0 subset is dissolved, and the gate that pinned it is now a
  universal.** Seven query-time modules in `latence-retrieval` kept a file-level Apache-2.0
  grant as a bounded exception to the PolyForm licence; with the whole repository Apache-2.0 the
  concept has no referent. `test_publish_readiness.py` no longer pins a seven-file allow-list —
  it walks the tree and fails on **any** SPDX identifier that is not `Apache-2.0`, which is
  strictly stronger: the old gate permitted a bounded set of files to disagree with the package
  metadata, the new one permits none. `pack.py` keeps its INBOUND/OUTBOUND provenance labels,
  which matter more now that both claims name the same identifier.
- **`COMMERCIAL.md` is now a correction page, not a route.** It keeps its name and every inbound
  link so a reader arriving from ADR-0055, from `NOTICE` or from a bookmark is told the gate is
  gone — instead of hitting a 404 or a stale demand for an agreement. Its gate test was inverted
  rather than deleted: it now asserts the page *grants* in so many words and that no agreement
  requirement, price or tier has crept back in.
- **PyPI: ADR-0062's rationale is superseded and its decision is REOPENED — not reversed.**
  ADR-0062 declined PyPI entirely on the strength of the PolyForm licence (corporate `pip
  install` audience, SCA identifier block, retained-subset invisibility). Every one of those
  premises is false under Apache-2.0, and ADR-0062's own revisit condition is met. The
  `pypi-publish` job in `release.yml` nevertheless **stays disabled** behind its `false &&`:
  whether to publish is a distribution decision (name claims, cadence, support expectations, an
  irreversible first upload) that the relicence did not weigh, and it needs its own ADR.
- **`CLA.md` is no longer required for the reason it was written.** It exists to secure
  relicensing and commercial-grant rights that Apache-2.0's inbound-equals-outbound (§5) no
  longer requires; a DCO would now suffice. It is *not* retired here — that is the copyright
  holder's decision, recorded as open in ADR-0063 — and `CONTRIBUTING.md` still requires it,
  because a stated precondition that is quietly not enforced is worse than either alternative.
- **Two `RELEASE-GATES.md` Gate-3 items were dissolved rather than resolved.** The benchmark
  corpus's `dataset_license="Apache-2.0" / commercial_use=True` flags and `latence-splade`'s
  single-valued `license_code` were both blocked on a contradiction that only existed under a
  noncommercial repository licence. Both are now true as written and are ticked.

### Fixed

- **PII inside a markdown table was not masked, because the chunker welded the cells together.**
  `chunking.py`'s pattern 13 matched `[ \t]*\|[ \t]*` with a literal `" "` replacement, and a
  replacement carrying no backreference is a *deletion* (`_apply_match`, deliberately — an
  injected literal has no original offset to map). The pipe **and the whitespace on both sides of
  it** were therefore removed and adjacent cell texts abutted: `| Ada Lovelace | 123 45 6789 |
  DE89370400440532013000 |` reached the Redactor as
  `Ada Lovelace123 45 6789DE89370400440532013000`. Welded to the name, the digits were refused by
  `ssn_nodash`'s letter-glue guard and the IBAN's leading `\b` could not fire between `e` and `D`,
  so the chunk came out with an **empty `pii_spans` and `masked_content == content`** — the same
  values that were masked in a sentence were left in the clear in a table, on every one of the 11
  shipped stacks (all of them wire `chunk.markdown`) and through the non-bypassable floor as well.
  The pattern now takes its separator from the text itself: a pipe with whitespace in front of it
  (or one that opens a line) is dropped along with the padding that follows it and the preceding
  space survives as the separator; a row-closing pipe goes with its trailing padding; a pipe with
  no whitespace on its left — the compact `|a|b|` form, where there is no space to promote — is
  kept, because the grid character is then the only thing that can hold the cells apart. Pinned by
  `test_strip_keeps_table_cell_texts_separated` (unit) and
  `test_table_row_pii_is_masked_identically_to_prose` (chunk → redact, end to end). The
  `ssn_nodash` letter-glue guard was re-measured against the fix rather than deleted with the weld
  that motivated it: it now costs no recall and still rejects letter-glued part and lot codes
  (`Lot 410455470A`, `Menü410455470`), so it stays.
  **This changes chunk text, and therefore embeddings and every benchmark number measured from a
  table-bearing corpus** (13 of the 40 enterprise-gold documents carry a markdown table; the OCR'd
  campaign corpora are table-heavy by construction). Nothing has been regenerated: every committed
  result is *pre-fix* and must be re-run before it is compared with anything measured after this
  change.
- **The table-pipe pattern was quadratic on a long whitespace run.** It began with a
  variable-length `[ \t]*`, so the engine rescanned the whole run from every start position inside
  it — 200k spaces cost ~55s in this one pattern. Every branch of the replacement now starts *at*
  the pipe (the lookbehind is fixed-width), which makes it linear: ~7ms for the same input,
  guarded by `test_table_pipe_pattern_stays_linear_on_a_long_whitespace_run`. `strip_markup` as a
  whole is still **not** linear on that input — pattern 12 (the table *separator* row) overlaps
  `^\s*` with a later `[ \t]*` over the same run and costs ~3s on 20k spaces. That is a separate,
  pre-existing defect; the new test is deliberately scoped to the pattern it guards so it neither
  hides nor claims to have fixed it.
- **The pattern list claimed an order dependence it does not have.** "Order matters: fenced/inline
  code first (their contents must not be re-interpreted as markup)" was false in both halves. The
  keep mask is a union of deletions marked against the *original* text, so the order is inert —
  shuffling the list 200 times produces byte-identical output and offset maps
  (`test_strip_pattern_order_does_not_change_the_result`) — and a fence's contents *are*
  re-interpreted: `x = a | b ***c***` inside a fence strips to `x = a b *c*`. The comment now says
  what the code does, and the anatomy chapter says it too.

- **CI never left Linux while all 33 packages claimed `Operating System :: OS Independent`.**
  Every `runs-on:` in every workflow was `ubuntu-latest`. The `test` job now matrixes over
  `os` as well as interpreter — `ubuntu-latest` on every supported Python plus a
  `macos-latest` leg — and the whole suite was run on an arm64 macOS host before the lane was
  added, so the claim is evidenced rather than hoped for. **Windows is not claimed**, and the
  classifiers changed instead of the workflow: `latence-demo` defaults to the POSIX path
  `file:///tmp/latence-demo`, ~98 test modules build `file:///…` URIs with no Windows spelling,
  the entire merge gate is bash, and the pinned GLinker wheels are Linux/x86-64 only. All 33
  packages now declare `Operating System :: POSIX :: Linux` + `Operating System :: MacOS` —
  exactly what CI runs — and `test_publish_readiness.py` pins the classifier set to the OS
  matrix so the two cannot drift apart again.
- **`requires-python = ">=3.11"` was unbounded with only 3.11/3.12 tested.** Rather than
  capping the claim, the tested range was widened: 3.13 was run locally against a
  `uv sync --python 3.13` of this exact lockfile, then added to the CI matrix and to every
  package's classifiers.
- **`CHANGELOG.md` recorded a "first public release" that never happened.** The `[0.1.0]` entry
  claimed a public, Apache-2.0-licensed release; no `v*` tag exists and nothing was ever
  published. It is now labelled as what it was
  — an internal version pin — and the two link references (`compare/v0.1.0...HEAD`,
  `releases/tag/v0.1.0`), both of which would 404 on a public repository, are gone.
- **Two files routed users to a GitHub Discussions tab that does not exist.**
  `.github/ISSUE_TEMPLATE/config.yml` and `SUPPORT.md` both linked `/discussions` on an
  unpublished repository. Both were rewritten to stop claiming a channel that was not there.
  The owner decision they were waiting on has since been taken — Discussions is ON and is the
  front door (see *Changed*, below) — so both files link it again, this time with the
  repository setting that makes the link resolve written into the go-live runbook as a
  required step rather than an optional nicety.
- **The `## Install` section read as live for ten distributions that are not on PyPI.** The
  honest line was eight lines above; the block itself was not marked. `## Install` now leads
  with the not-yet-published caveat and the working from-clone path, and the PyPI list is
  explicitly labelled as the eventual distribution map.
- **The `tests 5,139 passing` badge was a hardcoded count that goes stale on the next merge.**
  Replaced with a non-numeric capability badge; the exact count, the commit it was measured at
  and the command that produces it stay in `## Status & maturity`, which is the source of truth.
- **`RELEASE-GATES.md` Gate 3 stated a licence defect that had already been fixed.** All 33
  packages have declared the same SPDX identifier for some time, and a release-mode test already
  enforces it. The item is ticked and reworded to state the invariant and name its enforcing test.
- **`test_campaign_s5_wizard.py::test_the_vocabularies_mirror_the_harness` was red in CI and
  green on developer machines.** It imports `benchmark.s5.legs`, which imports `numpy` at module
  scope, while the CPU-first dev env is deliberately numpy-free (ADR-0016) — a pristine
  `uv sync`, which is exactly what `.github/actions/setup-workspace` produces, has no numpy. It
  passed locally only where numpy had been picked up transitively (a docs sync, a heavy Provider
  install). Found while standing up the py3.13 lane, on the only truly pristine environment in
  play. Now `pytest.importorskip("numpy")` with the reason stated, so it skips where the harness
  is not installable and still asserts lockstep where it is.
- **Five ADRs (0056–0060) and `docs/BENCHMARKS-E2E.md` were published but unreachable from the
  docs nav** — the same class of defect `#189` fixed for ADR-0032/0047. `BENCHMARKS-E2E.md` is
  the page `README.md` and `RELEASE-GATES.md` both send evaluators to.

- **The production Docker image's CLI could not import (deploy/Dockerfile).** The runtime stage
  copies only the venv, but `uv sync` installs workspace members editable and a workspace-member
  path given to `uv pip install` is rerouted editable through `tool.uv.sources` — so every
  `latence_*` import in the shipped image resolved to a source tree that was not there. Fixed with
  `--no-editable` / `--no-sources`, and the build now runs `latence --help` as a smoke so a broken
  entrypoint fails the build itself.

- **Delta manifests no longer resurrect edited-away content ids (`DeltaRunner._manifest_of`).** A
  DELETED entry now shadows the ancestor Version chain; before, an in-place edit's retired content
  id came back from an ancestor manifest and was re-counted DELETED on every later delta.

- **Qdrant round-trip identity and accuracy knobs (`latence_retrieval.qdrant.QdrantBackend`).**
  Served candidates now carry the corpus `record_id` as `Candidate.id` (config `id_key`, default
  `record_id` — matching what the sink stores), so they merge with the BM25 lane, dedupe and graph
  resolution instead of floating as opaque UUIDs; and `search_params` (`hnsw_ef`, `exact`) is
  validated at construction and threaded to the live store. Both defects were found and the fixes
  validated against a live Qdrant v1.19.0 on full query sets (recall parity within −0.0004
  absolute; see `docs/guides/serving-qdrant.md`).

- **Package READMEs no longer disagree with the repository licence.** The 2026-07-29 PolyForm
  relicence had left sixteen stale prose claims saying Apache-2.0; they were corrected to
  PolyForm, and the ADR-0063 relicence back to Apache-2.0 moved all sixteen again. The gate that
  found them is kept and inverted each time — it now fails on a README claiming the framework's
  own code is PolyForm or commercially restricted — so the prose cannot drift in either
  direction. Third-party noncommercial statements (research-only corpora, CC-BY-NC-SA weights)
  are deliberately not matched: those really are noncommercial.

- **Page provenance survives the OCR handoff (ADR-0060, contracts v21).** Page-accurate citation
  silently died whenever OCR happened out of band and its markdown was re-ingested: the plaintext
  Parser had no page structure to report, so it emitted a one-span map and every downstream record
  claimed `page 1`. Measured on the S3 SOTA corpus: `provenance.page_start == page_end == 1` on all
  236k chunks of 3,470 OCR'd documents, with `page_slice` reporting a 7-page paper as one page —
  populated, plausible, and wrong. Pre-paginated markdown now carries its boundaries in a sidecar
  file discovered beside it (`<file>.pagemap.json`, a serialized `PageMap`): `LocalFolderSource`
  finds it and attaches the raw bytes as `ParserInput.page_map_sidecar` (never ingesting the
  sidecar as a document of its own), and `parser.plaintext` turns it into a real map via
  `PageMap.from_sidecar_json`, which **cross-checks `total_chars` against the decoded document** so
  a stale sidecar is a counted `PARSE_ERROR` rather than confidently wrong pages. Where no page
  information exists, the new `PageMapOrigin` on `PageMap`/`PageSlice` records
  `ASSUMED_SINGLE_PAGE` — the assumption is now a value that propagates into the export instead of
  hiding as `page 1` — and `parser.plaintext`'s `require_page_map` turns a corpus the operator
  DECLARED paginated into a per-document refusal. The S2 campaign worker emits the sidecar from the
  same function that builds the markdown (`ocr_lib.assemble_document_with_pages`, of which
  `assemble_document` is now literally the `.text`, so the artifact is byte-identical) and
  back-fills it for already-assembled documents when the reconstruction still matches byte for
  byte. Every addition defaults: a pipeline with no sidecars resolves every offset to exactly the
  page it did before.

### Added

- **GLiNER 2.5 as a new Provider package, behind an enforced `gliner2` major boundary
  ([ADR-0064](docs/adr/0064-gliner-2-5-is-a-new-provider-package-behind-an-enforced-gliner2-major-boundary.md)).**
  `packages/latence-gliner25` registers **two** entry points — `fused_entity_relation.gliner25`
  (fused zero-shot NER+RE) and `redaction.gliner25` (PII) — over `gliner2[local]>=2.0,<3` and the
  `fastino/gliner2.5-multi-v1` checkpoint (287M, mDeBERTa-v3-base, multilingual, Apache-2.0, a
  4096-token window). One package because 2.5 is one checkpoint family serving both Capabilities
  over one pin, one compat gate and one loader seam.

  **The record contract is identical across the major — measured, not inferred.** A cross-major A/B
  on a pod (RTX 2000 Ada 16 GB, 2026-08-25) ran gliner2 1.3.2 + `gliner2-multi-v1` (span head) and
  gliner2 2.0.0 + `gliner2.5-multi-v1` (boundary head) in two venvs, on the same text with the same
  labels under identical transformers 4.57.6 / torch 2.4.1+cu124, and diffed the structural
  signature: envelopes, entity row keys, relation row keys, endpoint keys and int offsets all came
  back **SAME** — only `head_class` (`GLiNER2` → `BoundaryExtractor`) and `gliner2_version`
  differed. So swapping the extractor changes the engine and nothing about the emitted records. The
  key sets are now pinned in `latence_core.conformance.stub` and **all four** offline fakes in the
  workspace are asserted against that one definition, so a future gliner2 release that moves a key
  turns tests red instead of letting a fake drift away from the API it stands in for.

  That asymmetry — the inference contract carries, the loader and checkpoints do not — is the whole
  justification for a new package rather than a version bump. The pin is **disjoint** from the
  `>=1.3,<2` the existing gliner2 Providers declare, deliberately:
  gliner2 2.0 is a major whose 2.5 checkpoints are the new `boundary` architecture head, and
  `AutoExtractor.from_pretrained` refuses to bridge span and boundary checkpoints. That refusal is
  enforced three times before anything is fetched — the package pins, an
  `E-CONSTRAINT-CONFLICT` boundary on `deploy/env/cpu-text` + `gpu-learned` (offline, on every
  commit), and `latence_gliner25._compat`, which turns an unsupported version / a missing
  `AutoExtractor` / a 1.x `span` checkpoint into a typed `ProviderError` naming the Provider that
  *does* load it. The batched API itself did **not** change across the major (both architectures mix
  in the same runtime), so no parsing changed.

  **The PII claim is zero-shot and says so.** There is no GLiNER 2.5 PII or guardrails checkpoint,
  so `redaction.gliner25` asks the *general* checkpoint for PII labels zero-shot: no parity with
  `redaction.gliner2` is claimed, no recall has been measured (`docs/BENCHMARKS.md` records where
  the number will go), and `guardrails: true` is **refused** on the default checkpoint rather than
  run unbacked. The ADR-0044 deterministic financial-PII floor applies underneath either backend.

  **The advertised 4096-token window is NOT a window of reliable extraction** — measured on a pod
  GPU (RTX 2000 Ada 16 GB, torch 2.4.1+cu124, transformers 4.57.6, gliner2 2.0.0, 2026-08-25). A
  7,562-token document holding 90 SSNs returned 32 spans covering the first 35% of the text, with
  no error and a well-formed result; 12,612 tokens with one label returned zero spans silently; and
  longer inputs OOM'd a 16 GB card inside DeBERTa's quadratic attention. The degradation is
  **non-monotonic** (2,438 tokens scored worse than 3,026), which is worse than a hard truncation
  because no threshold test catches it. The filler was synthetic, so the percentages are
  directional — the failure MODE is not. So: **1024 tokens is the decided chunk budget**, the
  Providers' `max_len` is DERIVED from it rather than typed, raising it past the safe window is
  warned about on extraction and **REFUSED** on redaction (there it is fail-open — "redacted"
  chunks that still contain PII, silently), and a chunk too long for the window is refused with a
  typed error naming the budget knob rather than truncated. `chunk.page`'s split of an over-budget
  page is now a measured requirement, not a safety argument.

  Two things a real load needed that nothing declared, both found by running it: the 2.5
  checkpoints store `extra_special_tokens` as a **list**, which transformers >= 4.57 rejects at
  load — so the guarded shim that already fixed this for a 1.x checkpoint is **promoted into
  `latence_core.providers.perf`** and called by both packages (the old claim that only one
  checkpoint was affected is corrected in the same change) — and `protobuf` + `sentencepiece` are
  declared, since `gliner2[local]` declares neither and the DeBERTa-v2 tokenizer path needs both.

  Consequences worth knowing: the package is outside the uv workspace (a workspace is one resolved
  environment, and `uv lock` correctly refuses the pair), so it carries its own committed `uv.lock`,
  the release builds by path rather than `uv build --package`, and `scripts/generate-sbom.sh` unions
  the two locks so a shipped distribution is never absent from the SBOM. All three are gated.

- **`chunk.page` — a third reference Chunker: page-preferred, 1024 otherwise
  ([ADR-0064](docs/adr/0064-gliner-2-5-is-a-new-provider-package-behind-an-enforced-gliner2-major-boundary.md)).**
  One chunk per source page when the document carries a **measured** page map, so a retrieval hit is
  citable as a page with no further resolution; the windowed budget at **1024** tokens when it does
  not. Per [ADR-0060](docs/adr/0060-page-boundaries-cross-the-ocr-handoff-in-a-sidecar-and-an-absent-page-map-is-a-value-not-a-page-1.md)
  an `assumed_single_page` map counts as "no page structure", not as page 1 — the Provider reads the
  map's origin, not merely its presence. A page over the budget still **splits**: a page-sized chunk
  past the model window is a silent truncation and a split is not. Same offset-preserving engine as
  the other two, so the chunk → offset → original page round-trip and the v19 `page_slice` stay
  exact. **The existing `chunk.markdown` / `chunk.sentence_window` 512-token defaults are
  unchanged** — they are baked into committed benchmark baselines, and 1024 is the default of the
  new strategy only.

- **`latence setup` selects an extraction ENGINE, not a Provider.** One `--engine`
  (`gliner2`, the unchanged default, or `gliner25`) drives BOTH the fused extractor and the
  redactor, plus its coherent chunking (`chunk.page` at 1024 for `gliner25`). Because both Stages
  read the same `ExtractionEngine` row, a stack pairing one family's extractor with the other's
  redactor — which validates and then cannot be installed — is **unrepresentable** rather than
  discouraged. `--chunk-strategy auto|window|page` is the escape hatch. A chunk budget whose derived
  model window exceeds the chosen engine's documented checkpoint window is refused at answer
  construction rather than silently truncating at run time.


- **The KG Export's redaction asymmetry is now loud, and settled as intended behaviour.**
  `GraphHyperedge.tau_text` is redacted at its writer
  ([ADR-0057](docs/adr/0057-hyperedge-is-a-third-graphscope-emitted-by-graph-assembly-and-tau-text-follows-corpus-redaction.md)); `GraphNode.canonical_name`
  never was, and `graph_export.py` holds no redaction gate — so under the default
  `extract_on: unmasked` the real entity names ship in `graph-nodes.parquet`
  (`canonical_name`, the `evidence` snippet, and the `description` in `properties`), in the TTL
  `rdfs:label`, in the GraphML `label`, and in the hyperedge KV's `a|` alias keys. Reviewed and
  kept: a graph of `[PERSON]` nodes is a graph about nobody, and masking the label would collapse
  every person onto the single `a|person` alias key — a gate preserving node distinctness would
  have to *pseudonymise*, which is a different feature. A span-symmetric gate could not live in
  the export either: it receives only `GraphRecord`s, so it holds none of the `pii_spans` evidence
  the `tau_text` gate acts on. What is new is the signal: the Runner now logs one WARNING per KG
  export (`runner.KG_ENTITY_NAMES_WARNING`) in the genuinely surprising configuration — a
  redaction Stage wired while `extract_on` is `unmasked`, i.e. the corpus masked and the graph
  not — naming every surface and the `extract_on: masked` lever. Neither half of the condition
  warns alone. It is keyed off the new `KnowledgeGraphExport.exports_entity_names` marker (the
  same additive duck-typed seam as `wants_document_streams`), because the Provider itself sees
  neither the extraction posture nor whether anything upstream was redacted. The posture is now
  stated at the Provider, in `examples/graph-slice.yaml` (which hits the condition), and in
  `docs/anatomy/04-enrichment-and-export.md` §4.5; `test_kg_export_name_posture.py` pins the
  banner, the cleartext surfaces, and that `extract_on: masked` puts no PII span text in any
  node label.

- **`.pre-commit-config.yaml`** — a commit-time mirror of the two fastest CI gates (the ruff lint
  scope and the `detect-secrets` audited-baseline gate). Every hook is `local` /
  `language: system` and runs the *identical* command from `ci.yml`: pulling hook code from a
  third-party `rev:` would add an unpinned execution surface to a repository that SHA-pins every
  GitHub Action, and a hook that has drifted from CI is worse than no hook. A test asserts each
  hook's `entry:` still appears verbatim in `ci.yml`. `scripts/verify-local.sh` remains the merge
  gate.

- **A single `all green` aggregate CI job.** Branch protection can now require one check instead
  of enumerating every job and matrix leg by name — the failure mode being that a newly added
  matrix leg (the macOS and py3.13 legs) creates a status nobody required. It runs under
  `if: always()` and treats `skipped` as failure, since nothing in the workflow is conditional.

- **`CITATION.cff`** — GitHub's "Cite this repository" button. It deliberately carries no
  `version` / `date-released` / `doi`, because nothing has been released, and a test enforces
  that along with agreement between its URLs and the rest of the tree.

- **`Documentation` in `[project.urls]` for all 33 packages**, pinned by test to equal
  `mkdocs.yml`'s `site_url` so the docs URL cannot rot in one place and not the other.

- **`examples/*.yaml` are validated (`test_shipped_examples_validate.py`).** All 13 shipped
  examples now parse through the real `Pipeline` model, resolve every `provider:` against the live
  registry, and topologically sort (`Pipeline._validate_dag` catches unknown and self-dependencies
  but not a longer cycle). Every example must appear in an explicit classification map, so a new
  one cannot be added silently unvalidated — the rule `stacks/*.yaml` has had for a long time and
  `examples/` did not, despite the README quoting one as a proof. That proof is pinned too:
  `demo-pipeline.yaml` and `demo-pipeline.swap.yaml` must keep an identical Stage graph and differ
  only in Providers, or "one line changes, the pipeline does not" quietly stops being true while
  the diff still looks small.

- **`CONTRIBUTING.md` §"Contributing with AI"** and a matching disclosure checkbox in the PR
  template — disclose it, you own it, write the PR prose yourself. Assistance is expected;
  undisclosed assistance is what costs trust.

- **A `Roadmap` row in the README's "Where to go next".** No dates and no invented plan: the
  roadmap is `RELEASE-GATES.md` worked gate by gate, then the *Not yet proven* list.

- **`s5-retrieval` — the retrieval feature matrix is a campaign stage
  (`latence_benchmark.campaign.retrieval_stage`).** The S5 harness already honoured the campaign
  contract (per-cell resumability, refuse-don't-fake, `REPORT.md` + `metrics.json` gate artifacts)
  through machinery no campaign could reach: an argparse `main`, a hardcoded output directory, and
  module-level corpus constants. Now `benchmark.s5.run_matrix.run(plan) -> MatrixOutcome` is the
  harness API (argparse demoted to one adapter over it, with the `PYTHONHASHSEED` refusal at that
  entry rather than in `main`), `benchmark.s5.datasets` resolves every input from a passed-in
  `CorpusLayout`, and the campaign registry gains a fourth stage — gated on `s3-pipeline`'s
  approval, done-probed on every planned cell plus the two gate artifacts, and blocked when a whole
  dataset recorded only refusals (an absent export is not a measurement). The wizard grew a
  byte-deterministic `s5:` block (datasets, legs, limit, fusion), and the campaign config the
  wizard emits was regenerated with it. One cheap cell produced through
  BOTH entries — CLI and `latence-campaign run s5-retrieval` — is identical modulo wall-clock
  latency keys.

- **Hypergraph query index — the graph lanes at serving latency (`latence_retrieval.hypergraph`,
  ADR-0056 amendment).** `HyperedgeKV.open` now attaches a transparently cached, file-identity-keyed
  `HypergraphIndex` (postings, incidence, γ-rank — decoded once per store per process; numpy-
  accelerated when importable, pure-Python otherwise), and `GraphRetriever.search` resolves
  documents tier-by-tier with a byte-identical early stop. Measured at the 677,257-hyperedge
  export: every graph lane ~100× faster (e.g. hypergraph lane median 3,570 ms → 37 ms), with
  parity counted over 21,215 real gold queries × 3 lanes across all code paths — 0 diffs.
  `cache=False` restores the scan-per-call path bit-for-bit.

- **Graph-rescue retrieval — the production graph-at-query-time composition
  (`latence_retrieval.rescue.GraphRescueRetrieval`).** Base dense+BM25 fused and preserved, a
  per-query policy gate, graph traversal + hypergraph lanes that APPEND (never fuse), and a
  token-budgeted selector (knapsack with backfill / chain-of-evidence / none) — zero LLM calls
  online, measured defaults, configurable from the campaign wizard. Retrieval and selection are
  separable: `run(bundle)` is the one-shot call and is exactly `select(text, retrieve(bundle))`,
  so one `RescueRetrieval` serves any number of budget/selector A/Bs without re-running the
  engines. Guide: `docs/guides/graph-rescue-retrieval.md`.

- **Incremental ingest through the full enrichment pipeline (ADR-0061).** The flagship
  enrichment+KG+export profile now reuses unchanged documents on churn (O(changed) extraction)
  through an order-preserving overlay fold that upgrades delta/rebuild equivalence to
  byte-identical streams — proven on a 609-document real corpus (sha256-equal exports, bitwise-
  equal embeddings; add of 61 docs: 8.4 min vs 97.4 min rebuild). Plus `latence delta --dry-run`
  (churn preview, no Stage run) and `docs/guides/incremental-ingest.md`.

- **Qdrant serving path, verified end-to-end.** A pinned single-node compose
  (`deploy/qdrant/docker-compose.yml`), and `docs/guides/serving-qdrant.md` — the measured
  Export → collection → served-queries path (`distance: dot`, `hnsw_ef: 512`) with full-set
  recall-parity tables.

- **Hyperedge sidecar KV + entity descriptions (ADR-0056/0058/0059, hypergraph-m2;
  `latence_core.hyperedge_store`).** The opted-in hyperedge path now ships its query-time face:
  `graph-hyperedges.kv`, a dependency-free, byte-reproducible single-file point-lookup store
  (sorted keys, mmap binary search, opened per call and closed — statelessness structural), always
  rebuildable from the parquet. Keyspaces: solver fields (no text), postings γ-sorted descending
  with γ inline (top-k candidate truncation without a row fetch), `tau_text` (corpus redaction
  variant), and an NFKC/casefold alias table — ADR-0059's tier-1 zero-model query entry works
  against it today. Nodes additionally carry a deterministic spec-1.4 `description`
  (name — type — top-3 relation contexts) and `hyperedge_degree` in properties: the zero-LLM
  embedding input for entity ANN. The ADR-0058 index side completes with
  `hyperedges.token_counter: "tiktoken:<encoding>"` (lazy, loud ConfigError when missing — never a
  silent fallback; identity stored on every record). Stacks without `hyperedges:` are untouched —
  no KV, no new node properties, byte-identical output.

- **Hyperedges — the m-ary retrieval unit, an OPTIONAL third Graph-Assembly projection
  (ADR-0056/0057/0058, hypergraph-m1; `latence_core.hyperedge`, `latence_core.sentences`,
  contracts v20).** Where an edge connects two nodes, a `GraphHyperedge` carries the FULL set of
  canonical entities co-occurring in one minimal sentence window plus the directed relations
  extracted from it — the atomic item a chain-of-evidence selector packs into a token budget, with
  its selection inputs precomputed on the record: `token_cost` (+ the counter identity beside the
  value, ADR-0058, so an index/query tokenizer mismatch is detectable), `gamma` (link ⊕ relation ⊕
  weakest-mention confidence, renormalised over signals actually present), `support` (corpus-wide
  dedup on identical entity-set + relation-signature), and `has_relations` (weak, entities-only
  units behind a config toggle). Segmentation runs on a new zero-dep, abbreviation-safe DE/EN
  `RuleSentenceSplitter` (`z. B.`, `Dr.`, `am 3. Mai`, decimals, initials — never boundaries), not
  the chunker's budget regex. **`tau_text` follows the corpus redaction variant, enforced at the
  writer** (ADR-0057): a Hyperedge can never carry text the exported corpus masked — the PII-bypass
  regression test pins it. Strictly opt-in: `graph_assembly` config gains `hyperedges:` (absent =
  everything byte-identical to v19 apart from the version number; the Runner feeds the extra
  chunk/mention/relation streams only to a Provider that opts in via `wants_document_streams` — no
  Capability-protocol change). Opted-in stacks additionally get `graph-hyperedges.parquet` in the
  KG Export and a `graph.hyperedges` Quality-Report section (counts, arity histogram, and the
  token-cost distribution that predicts packing behaviour). Disambiguation now PERSISTS the
  per-mention link confidence the linker always computed and dropped
  (`CanonicalEntity.member_link_confidences`, glinker `link_confidence`) — γ's data source.

- **T7 — efficient per-chunk extraction: label-set bucketing + gliner2's native batched forward
  (`fused_entity_relation.gliner2`, `redaction.gliner2`, `latence_core.providers.perf`).** Per-chunk
  induction (T1/T2) gives every chunk its own small label set, but gliner2's batched entry points
  take ONE schema per call — so the naive readings were "one native call per chunk" (correct labels,
  no batching) or "union the batch's labels" (maximal width, but back to the ~40-label degradation
  and the ~17 GiB span-scoring tensor the redesign exists to remove). Both Providers now BUCKET each
  streaming window by label set through one shared, torch-free seam
  (`perf.iter_label_buckets`, first-seen bucket order, ascending indices) and run **one native
  batched forward per bucket** at that bucket's own small label count — maximal batch width, never
  the corpus union. The fused extractor was the un-batched half: it now drives
  `batch_extract_entities` / `batch_extract_relations` (bucketed independently per slot, since a
  chunk can share the entity schema and not the relation one), scatters per-text results back **by
  index**, and inherits the token-budget micro-batch bound plus the CUDA-OOM halving backoff its
  twin already had. Output is bucket- and batch-INVARIANT (asserted record-for-record against a
  strictly per-chunk reference). **This is the root fix for the CUDA-OOM**: the tensor was
  `batch × ~768 tokens × ~40 corpus-wide labels`; per-chunk labels collapse the label dimension, so
  it never forms — the reactive backoff stays only as a last-resort net. gliner2's native
  **`compile` is unlocked** as a consequence: it was forced OFF because a fresh (label-count, batch)
  shape family per forward recompile-thrashed `torch.compile(dynamic=True)`, and bucketing +
  micro-batching pin both dimensions, so it rides the shared `auto` default (True on cuda) again,
  behind the load-time guard AND the inference-time reload-uncompiled latch. `latence-extract-gliner2`
  raises its `gliner2` pin to `>=1.3,<2` (the batched slice does not exist across the whole 1.x
  line) and checks both batched methods at LOAD, so an out-of-range install is a typed
  `ProviderError`, never a native `AttributeError`. **Honest framing:** gliner2 ships no
  self-hostable inference server — its only server surface is `from_api`, a client for fastino's
  *cloud* API — so the throughput here is the native batched forward + compile + quantize,
  in-process. The eager-vs-compiled number and the peak-VRAM / backoff-count evidence are a GPU
  measurement: the harness is `scripts/validate-gliner2-unified-on-gpu.md` §7, unmeasured (never
  fabricated) until the pod run.

- **Blackwell / CUDA-13 compatibility matrix, machine-checked (#199, ADR-0016/0036/0030/0012).**
  `deploy/env/compatibility-matrix.yaml` maps every learned Provider and serving dependency of the
  enterprise-sota stack against every pinned environment (#195/#196/#198), and
  `docs/compatibility-matrix.md` is **generated** from it and byte-compared, so the published page
  cannot drift from the checked data. The rule the new `latence env matrix` command enforces is the
  reason the file exists: **every cell is a MEASUREMENT or explicitly `UNVERIFIED`**, with no third,
  softer status. A `verified` cell without date/hardware/command/observed output is
  `E-MATRIX-EVIDENCE-MISSING`; an `UNVERIFIED` cell without a *runnable* reproduce command is
  `E-MATRIX-REPRODUCE-MISSING` / `-UNRUNNABLE`; coverage is **derived from the committed
  `environment.yaml` files**, so a new enforced boundary or learned Provider is
  `E-MATRIX-CELL-UNCOVERED` until it is mapped; a recorded version that disagrees with that
  environment's committed lock is `E-MATRIX-VERSION-DRIFT`; a named served model without its
  weights AND code licences cited separately is `E-MATRIX-LICENCE-UNCITED` (ADR-0012); a committed
  lock that pins `nvidia-*` CUDA wheels `THIRD-PARTY-LICENSES.md` does not name, or whose path that
  register never cites, is `E-LICENCE-UNDISCLOSED` — the gate that relates an *environment* to the
  published licence page, so a second pinned GPU environment cannot ship disclosed only by the
  first one's carve-out. Each cell
  carries two independent axes — **resolution** (does it install: offline, checked on every commit)
  and **runtime** (does it run on `sm_120`: needs the hardware) — because a green resolution is not
  a runtime claim. The five failure modes of the live 519-page run are matrix entries with their
  resolutions and the file that enforces each (`FM-FLASHINFER-SM120` — the `sm75` check that
  misparses compute capability 12.0; `FM-NVRTC-JIT` — the TorchScript/nvrtc crash in the mdeberta
  encoder, #190; `FM-WHEEL-ABI` — the ABI-broken `flash-attn`/`torchcodec` wheels;
  `FM-MULTIMODAL-OOM` — the multimodal OOM at the generic serve settings; `FM-TRANSFORMERS-SPLIT`);
  dropping any one of them fails the check. `scripts/verify-local.sh` runs the verifier on every
  commit; `--require-verified` (the maintainer's switch ON the Blackwell host) additionally fails
  while anything is still unmeasured.

- **The one full-stack end-to-end verification (#199).** `deploy/env/full-stack/` composes the
  served-OCR environment and the GPU learned-Provider environment into a single pipeline — the only
  artifact where both legs run in one run — as one command:
  `bash deploy/env/full-stack/verify-full-stack.sh --corpus <scans> --record`. Four gates: both
  environments' offline spec gates, the gpu-learned smoke (GPU-placement probe included), the
  composed run, then `assert_full_stack.py`, which refuses every near-miss shape the project's own
  history has produced — a **skipped** Stage (correct on a CPU host, a false green for a GPU
  verification), a Stage that silently resolved to `cpu` (the #191 signature), an empty export,
  `document_count: 0` or `contracts_complete: false`. `--record` prints the evidence block to paste
  back into the matrix, so the measurement is transcribed by the machine that made it. **Status:
  UNVERIFIED** — it has not been run: no Blackwell/CUDA-13 host was available, and the matrix says
  so in the record rather than implying otherwise.

- **Go-live runbook + a machine-checked front door (#189, #72).** `docs/GO-LIVE-RUNBOOK.md`
  is the maintainer-manual launch procedure — visibility flip, docs deploy, signed release,
  PyPI publish, announcement — each step with the check that proves it landed, a rollback
  table, and the two decisions that must be made *before* step 1 (PyPI name availability,
  and the fact that flipping visibility publishes git **history**, not just the tree).
  Backing it, `test_front_door_truthfulness.py` now fails the build if the README claims a
  Runner adapter with no package behind it, quotes an ADR count that disagrees with
  `docs/adr/`, links at a file that does not exist, or tells a reader to install a
  distribution that is not in `packages/`. The PyPI publish was rehearsed end to end: all
  32 distributions build, `twine check` passes on all 64 artifacts, and `pip install
  latence-demo` into a clean venv runs the full demo to a green Quality Report.
  **`docs/GO-LIVE-RUNBOOK.md` has since been removed from the shipping tree** as one of five
  maintainer-internal documents. This entry is left standing as the record of what shipped and
  is not a live pointer: the path named here no longer resolves in the tree.

- **Benchmark suites for Entity/Relation Extraction, Disambiguation and Redaction (#187).** The
  public benchmark harness (#181) now measures four more Stages through the same
  `BenchmarkSuite` seam, the same one command (`latence-benchmark run --suite all`) and the same
  published results page. Entity Extraction, Relation Extraction and Disambiguation run on
  **RED^FM** (`Babelscape/REDFM`, CC BY-SA 4.0 corpus + Apache-2.0 loader, verified separately
  per ADR-0012), pinned to a commit SHA, across all seven languages — so the P1 German/EU slice
  is one row of the table those suites emit. Their numbers are **not published yet** (RED^FM has
  to be downloaded first); `docs/running-the-benchmarks.md` declares them under *Not yet
  measured* with the command that produces them, and the results page carries no section for a
  Stage that has not been measured. Redaction runs on a **synthetic PII corpus that ships in
  the package**: no real personal data enters the repository, every value is a reserved or
  published example, the corpus is generated by a committed generator whose output is
  byte-checked, and it is pinned by **content digest** (`ShippedCorpusSource` refuses a corpus
  whose bytes no longer match its pin). Results gained a second reporting currency,
  `MetricScore` (TP/FP/FN → precision/recall/F1, with an undefined ratio reported as `n/a`
  rather than 0%), rendered on the same page next to the Parse pass-rate table. WikiANN/PAN-X
  was **excluded** because its dataset card carries `license: unknown` — recorded in
  `THIRD-PARTY-LICENSES.md` so the decision is not silently re-litigated.

- **GPU learned-Provider environment (#198, ADR-0016/0007/0030/0036/0045/0046).**
  `deploy/env/gpu-learned/` stands up the learned Provider set — GLiNER2 fused
  extraction, GLiNER2 financial-PII and learned GLiNER PII redaction, GLinker
  disambiguation, Granite r2 GPU-tier embeddings — pinned and reproducible on
  CUDA 13 / Blackwell (`torch==2.13.0+cu130`, the wheel line carrying `sm_120`
  kernels), from one documented `install.sh` with no trial-and-error and no
  `--break-system-packages` (PEP-668 safe). The `transformers` major split has
  exactly **one** supported resolution — the ADR-0030 endpoint boundary: the
  4.x GLiNER/mdeberta family in-process, the 5.x OCR VLMs served and reached
  over HTTP via `parser.endpoint`. A two-phase single-interpreter install was
  rejected as merely deferring the failure. The resolution is **enforced**:
  `latence env verify` now reports `E-CONSTRAINT-CONFLICT` when a package's
  declared version window is disjoint from an enforced boundary, and
  `E-PACKAGE-EXCLUDED` when an in-process OCR package enters the set directly or
  as another package's workspace dependency — both before `install.sh` runs, so
  the unsupported combination fails at dependency resolution rather than at model
  load.

- **Real GPU-placement proof (#198, closing the #191 class).** New
  `latence env placement` drives a real inference through every Provider an
  environment declares in `learned_providers` and reads torch's CUDA allocator
  around the call, reporting the allocation **delta above the Provider's resident
  footprint** (a warm-up drive plus a baseline subtraction, because real
  `reset_peak_memory_stats()` resets the peak to the *current* allocation — so
  weights parked on the GPU can never be counted as work). A forward pass that
  silently ran on the CPU allocates **zero CUDA
  bytes** above that baseline however correct its output is — which is exactly what #191 looked like
  on the pod (`device: cuda` configured, ~11.5 s/chunk, ~94 minutes for one
  Stage, the GPU idle, no warning). The assertion is behavioural and cannot be
  satisfied by a flag or a config echo; a missing CUDA device is a **failure**,
  not a skip. `latence env verify` additionally refuses a GPU environment whose
  smoke stack never instantiates a declared learned Provider
  (`E-LEARNED-UNDRIVEN`), since a Provider the gate never drives is a hole in the
  gate.

- **Optional Iceberg-table Export output (#184, ADR-0017).** The corpus Export can now
  *additionally* register the records it wrote as an Apache Iceberg table in an Iceberg **REST
  catalog** (Unity / Polaris / Glue-via-REST / Lakekeeper), so the same data becomes directly
  governable with no glue code. Enabled with one `iceberg:` config block on
  `export.jsonl_parquet`; **off by default**, and with it off `pyiceberg` is never imported and
  `records.jsonl` / `records.parquet` are byte-identical. The Iceberg schema *is* the Parquet
  schema (including opt-in signal columns) and carries the same contract version stamp; the
  registered data file is an immutable, content-addressed copy in the target table's own location
  (`<table location>/data/records-<sha256>.parquet`), so a snapshot never points at a path a later
  run rewrites — and never at one the framework's own Purge / run-retention cleanup deletes. The
  export directory itself is left byte-for-byte unchanged. Delta runs pick `mode: append` (a delta
  run's records, added as a snapshot) or `mode: overwrite` (a full re-export, replacing the
  contents in one atomic commit) — semantics documented in `docs/iceberg-export.md`. The client
  ships as the `latence-core[iceberg]` extra (pyiceberg, Apache-2.0), never as a core dependency.

- **OpenLineage run events (#185).** The Runner can emit standard OpenLineage
  `RunEvent`s — START/COMPLETE/FAIL per Stage plus a parent event for the run —
  with each Stage's input/output datasets and a `schema` facet from its carrier
  contract, so any OpenLineage consumer (Marquez, Atlan, Collibra, Dataplex) sees
  a latence run in its lineage graph with no custom integration. Off by default
  (`LocalRunner(lineage=...)` / `LATENCE_OPENLINEAGE_URL`), stdlib-only on the wire
  (no `openlineage-python` dependency), counts-only (no content, no PII, no secrets,
  and deliberately no `errorMessage` facet — only the typed error category), and an
  emission failure never fails a run. See `docs/observability.md`.

- **`latence-sink-qdrant`** — the first first-party **sink**: one command between "pipeline
  finished" and "my retrieval engine serves the corpus" (#183, the first slice of #156).
  `latence-sink-qdrant load ./out/export --collection corpus --url …` reads the Export's
  `records.parquet` (streamed in bounded batches), creates the collection if absent at the corpus's
  own vector width, and upserts with the `embedding` column as the vector and every remaining column
  as payload.
  - **Idempotent** — point ids are a deterministic UUIDv5 of the content-addressed `record_id` (a
    raw `record_id` string is not a legal Qdrant point id), and a stored content digest means an
    unchanged record is **not** re-upserted, so a re-run or an incremental/Delta run
    ([ADR-0018](docs/adr/0018-incremental-corpus-deltas.md)) writes only what changed.
  - **Retraction/Purge honoured** — records the Export no longer contains are deleted by a
    reconciling sweep, and the load **counts the stale set back to zero before reporting success**;
    `latence-sink-qdrant verify-absent <record_id>…` is the operator-side audit step (exit 1 while
    any named record is still there; `SinkVerificationError`/exit 2 rather than a false green when
    the named collection does not exist, so a typo can never pass for erasure evidence). `--no-reconcile` covers the incremental-Export case.
  - A store fault mid-load aborts **before** the sweep, so a half-uploaded corpus can never delete
    the records it did not reach.
  - The Export stays the source of truth ([ADR-0017](docs/adr/0017-ai-ready-deliverable-is-rag-corpus-plus-kg-as-files.md));
    the sink holds no index and runs no search ([ADR-0048](docs/adr/0048-retrieval-tooling-is-strictly-stateless.md)).
  - `latence-core` takes **no** new dependency: the mapping/idempotency/reconcile core is pure-Python
    over `latence-core`, and `qdrant-client` (Apache-2.0) rides the optional `[client]` extra,
    imported lazily. Ships a faithful `FakeQdrantClient` that rejects what a real Qdrant rejects
    (404/409/403/400 incl. non-UUID point ids and wrong-width vectors), plus an opt-in live smoke
    test (`LATENCE_QDRANT_URL`).

- **Reproducible environments (`deploy/env/`)** — a pinned lockfile, a one-command install recipe
  and a one-command smoke test per Provider-set slice, starting with `cpu-text` (GLiNER2 fused
  extraction, learned GLiNER PII redaction, GLinker disambiguation, Granite r2 dense embeddings,
  SPLADE sparse). The pins are **enforced constraints**: `latence env verify` cross-checks the
  generated `requirements.in`/`constraints.txt`, the solved `requirements.lock`, the Provider
  packages' real dependency declarations and the smoke stack, offline and import-free, and
  `scripts/verify-local.sh` runs it for every environment — so dependency drift that would break
  the stand-up turns the build red instead of a pod. New `latence env verify|render` commands and
  the optional `latence-core[envspec]` extra.

- **`latence-benchmark`** — the reproducible **public** benchmark harness (#181). One command
  (`latence-benchmark run --suite parse`) runs a framework Stage over a public dataset pinned to
  an exact commit SHA, under a pinned stack config, and publishes `docs/BENCHMARKS.md` plus a
  machine-readable `docs/benchmarks/results.json`. The Parse tracer slice scores
  `parser.pdfplumber` against [olmOCR-bench](https://huggingface.co/datasets/allenai/olmOCR-bench)
  (ODC-BY-1.0 corpus, Apache-2.0 toolkit — verified separately per ADR-0012). Not a pipeline
  Provider: it registers nothing under `latence.providers`, and it is distinct from the
  adopter-side gold-set harness of ADR-0028.

  Structural guarantees, not conventions: a `DatasetSpec` rejects any revision that is not a full
  40-hex SHA; a partial download is refused rather than partially scored; the license gate runs
  *before* acquisition and treats an UNVERIFIED license as a non-overridable stop; a parse failure
  counts its document's tests as failures; unimplemented test classes are reported *unsupported*
  and excluded from the pass-rate denominator; vendor numbers are labelled claims and kept in
  their own section. `latence_benchmark.suite.BenchmarkSuite` is the seam further Stage suites
  (#187) plug into.

- **`latence-console`** — the read-only run console (#186). One command,
  `latence-console <storage-location>`, serves a localhost page over the files a run already
  left on Storage: the run list (status + timestamps) and each run's Quality Report rendered
  as **headline quality verdicts a non-engineer can read** — traceability, unsafe input,
  personal data, knowledge-graph statistics, gold-set scores, and the what-changed summary of
  an incremental run — with the per-Stage drill-down beneath. Read-only and stateless by
  construction (ADR-0010/0017): it reaches only `exists`/`ls`/`read_text`/`modified` on the
  Storage seam, refuses non-read HTTP methods with `405`, caches nothing between requests,
  binds the loopback interface only, and emits self-contained pages with no CDN, script, or
  beacon. Zero new dependencies — `http.server` from the standard library plus `latence-core`.

- **The run console's debugging loop (#188).** Three deep views on top of the console, reached
  from any run page. **Held back & flagged** lists every record Screening quarantined (checkpoint,
  detector, category, recorded reason, bytes withheld) and every record it kept but marked, with
  the risk marker that rode into the corpus. **The chunk inspector** answers "why is this chunk
  wrong": each chunk resolves back through its Provenance to its source document, source *page*
  and character offsets — computed with the framework's own `OffsetIndex`/`PageOffsetIndex` over
  the record's own maps, so the console cannot disagree with the pipeline — alongside its
  Classification and its Redaction disposition, including the case where the redaction policy ran
  *no* detectors (an empty PII list then says "the control was OFF", never "clean"). **The
  knowledge-graph browser** makes entities and edges navigable and shows **every edge with its
  Evidence** — the source mentions resolved to text, file, page, offsets and confidence; a cited
  mention the run no longer holds is shown as unresolved rather than dropped. All of it reads
  JSONL the Runner already wrote (`checkpoints/`, `quarantine/`, `screening/`) through the real
  contracts, so a malformed line is rejected exactly as the pipeline would reject it. Still
  strictly read-only, still no new persistent state, still zero new dependencies, still offline —
  and only the PII-handled corpus text is ever rendered.

- **`Storage.modified(uri)`** — a read-only last-modified probe on the Storage seam,
  normalising the per-backend spellings (`mtime` / `LastModified` / `updated` / `created`) to
  an aware UTC datetime and returning `None` rather than fabricating a time when a backend
  exposes none. This is how the run console reads real run timestamps without the framework
  persisting any new state.

### Changed

- **Distribution is signed GitHub Releases only — PyPI is closed, not pending
  ([ADR-0062](docs/adr/0062-distribution-is-signed-github-releases-only-no-pypi-under-polyform-noncommercial.md)).**
  *(Superseded within the same Unreleased block by ADR-0063 — see the licence entry above. The
  decision is reopened; nothing is published either way.)* PyPI's Terms of Use permit a
  PolyForm-Noncommercial upload, so this was a choice rather than a prohibition — but `pip
  install` is overwhelmingly a corporate action, PolyForm-NC did not licence commercial use, and
  corporate SCA scanners hard-block the `PolyForm-Noncommercial-1.0.0` SPDX identifier, so
  publishing would have put the artifact in a channel whose majority consumer was by the
  licence's own terms unlicensed. The retained Apache-2.0 subset in `latence-retrieval` did not
  change that: scanners read package metadata, which said PolyForm for the whole distribution.
  `release.yml`'s `pypi-publish` job keeps its full OIDC scaffolding and its `false &&`
  short-circuit.
- **GitHub Discussions is ON, and is the front door for questions.** Blank issues stay disabled,
  Discussions is contact link #1 in `.github/ISSUE_TEMPLATE/config.yml`, and `SUPPORT.md` routes
  every question there while the three issue templates keep only structured reports. Both files
  describe a **repository setting**; enabling it is now a numbered step in
  `docs/GO-LIVE-RUNBOOK.md` step 1, because no test in this tree can turn a GitHub setting on and
  the last defect here was exactly a docs-vs-settings disagreement. (That runbook has since been
  removed from the shipping tree as maintainer-internal; the path above no longer resolves. The
  repository setting it enumerated is unaffected.)
- **A chunk describes its OWN pages — `ChunkRecord.page_map` becomes `ChunkRecord.page_slice`
  and `PageIndexResolver` becomes STATELESS (`SCHEMA_VERSION` 18 → 19, BREAKING).** v18 kept the
  payload win (below) but bought it with a SHAPE defect: the whole document's `PageMap` rode ONE
  record of the chunk stream — the document's first chunk — and the resolver adopted it from
  whichever record it saw first. That made page resolution depend on two things that are not
  properties of the chunk being resolved: that the map-carrying record was **observed first**
  (every batching extractor resolves out of stream order) and that it **survived every filtering
  Stage** (any chunk→chunk Capability may drop chunk 0, and Screening quarantining a title page
  routinely did). Both failures degraded SILENTLY to the chunk's inherited page range — a
  plausible-looking WRONG page, indistinguishable from a correct one in the output. Four audit
  rounds patched the consequences (`rehome_page_maps()`, then an LRU retention window, then that
  window's own justification) and never the shape.
  `ChunkRecord.page_map: PageMap` is therefore **replaced** by `ChunkRecord.page_slice: PageSlice`
  — exactly the document page spans overlapping that chunk's own `[char_start, char_end)`, in the
  document's original coordinates. One span for a chunk inside a page, two for a chunk straddling
  a page break. A chunk never needed the whole document's map: the only question asked of it is
  which page a SUB-CHUNK offset falls on. Order- and drop-independence are now properties of the
  record rather than of the stream, so the failure mode is not mitigated but **inexpressible** —
  and ~99.8% of the v18 payload win is kept (~2 spans, ~60 bytes, against a ~30 KB whole-document
  map). Provenance is byte-identical (ADR-0021/0031).
  **Deleted, not deprecated:** `PageIndexResolver.observe()`, its per-document memoisation, the
  `max_live_pages` retention window and its `evicted_reentries`/`documents`/`unresolved`
  counters, `PageIndexResolver.from_documents()`, `rehome_page_maps()` and
  `DEFAULT_MAX_LIVE_DOCUMENTS` — every one of them was compensating mechanism for the old shape.
  `page_for(chunk, offset) -> int` and `pages_for_span(chunk, start, end) -> tuple[int, int]` no
  longer return `None`: a chunk carrying no spans raises `PageSliceMissingError` (ADR-0034 — a
  failure is observable or it is not handled), and `AdapterBase.resolve_pages` no longer degrades
  to the chunk's inherited range. A relation's **covering span** — whose endpoints may sit in
  different chunks, so no single chunk can answer for it — is now computed by the new
  `pages_for_covering_span(head, tail)`, arithmetic on the two endpoints' already-exact pages,
  replacing the "anchor chunk" lookup that only ever worked because the anchor carried the whole
  map. **Migration:** chunk checkpoints are run-scoped artifacts and a run is re-chunked from its
  Parse checkpoint, so there is no data migration to perform — the schema bump is the notice. A
  Provider that read `chunk.page_map` must go through `PageIndexResolver`; see ADR-0022's C1
  amendment and `docs/writing-an-adapter.md`. `DocumentRecord.page_map` is **unchanged** — the
  document genuinely has a whole-document map, and it is what the slice is cut from.
- **The document page map stops riding every chunk — one `PageIndexResolver` seam replaces five
  hand-rolled per-chunk indices (`SCHEMA_VERSION` 17 → 18).** `PageMap` is DOCUMENT-scoped state,
  invariant across every chunk of its document, yet `ChunkRecord` carried it by value on all N
  chunks: measured on a 519-page document chunked at 640 tokens that is ~30 KB of JSON on each of
  ~399 records — **~92% of every chunk checkpoint line**, ~12 MB redundant per document, paid again
  on every checkpoint write, every streaming read-back and every decode. The Chunk Stage now stamps
  `page_map` on a document's **first** chunk only, and offset→page resolution moves behind
  `latence_core.page_index.PageIndexResolver` — a narrow, document-keyed seam that memoises ONE
  `PageOffsetIndex` per `document_record_id`, owns the half-open-span end clamp, and aggregates
  drift corpus-wide (previously each throwaway per-chunk index carried its own diagnostics and
  nobody summed them). Five core call sites (`stages/entity`, `stages/header_refs`,
  `stages/inline_refs`, `stages/relation`, `redaction_policy`) re-derived that knowledge; exactly
  one of them cached, and only per chunk. Provenance is unchanged — offsets and pages are exact
  (ADR-0021/0031) — and the field stays optional, so a pre-v18 stream (map on every chunk) resolves
  identically: the resolver adopts whichever record carries the map first, or is seeded straight
  from the parsed `DocumentRecord`s (`PageIndexResolver.from_documents`, ADR-0017). Degradation is
  now COUNTED (`resolver.unresolved`) instead of a silent per-caller fallback.
  **`AdapterBase.resolve_pages` — the PUBLIC helper third-party adapters are pointed at — goes
  through that seam too (BREAKING, audit R2).** It signature changed from
  `resolve_pages(doc_char_start, doc_char_end, page_map=…, fallback_page_start=…,
  fallback_page_end=…)` to `resolve_pages(chunk, doc_char_start, doc_char_end, resolver=…)`: the
  old form built a throwaway `PageOffsetIndex` from `chunk.page_map`, which under v18 is `None`
  on every chunk after a document's first, so an adapter following `docs/writing-an-adapter.md`
  verbatim silently degraded those mentions to the parent chunk's inherited page range — the
  exact regression ADR-0021/0031 forbids, on the seam that is supposed to make the invariant the
  default. The interface got *smaller* (the two fallback arguments are now derived from the
  source chunk, which is what every caller passed), the guide documents the one-resolver-per-call
  + `observe()`-every-chunk discipline, and the reference adapter demonstrates it.
- **The benchmark guide no longer describes numbers the results page does not carry (audit
  R1-docs_truth-9, #187).** `docs/running-the-benchmarks.md` said in the present tense that the
  Entity/Relation Extraction, Disambiguation and Redaction slices were published on
  `docs/BENCHMARKS.md`, which carried the Parse suite and nothing else. The **redaction** number —
  the one that makes the framework's central privacy claim externally checkable, and the one suite
  that needs no download because its synthetic corpus ships in the package — is now measured and
  published (100.0% precision / 77.1% coverage recall over 35 gold items in 20 documents, with the
  per-language, per-PII-type and per-match-criterion tables). The three RED^FM suites are declared
  under an explicit **Not yet measured** heading with the command that produces them. The
  structural cause is fixed too: a run covering a subset of the suites now **republishes** the
  other Stages' published measurements instead of overwriting the page with its own suite
  (`--no-carry-forward` opts out), refuses to republish a number whose dataset has since been
  re-pinned, and every measurement records and renders the date it was produced.
- **Package license metadata is PEP 639 SPDX (#72).** Every publishable package moved from
  the legacy `license = { text = "Apache-2.0" }` + `License ::` classifier combination —
  which strict twine/PyPI validators reject — to a bare SPDX expression plus a
  `license-files` glob, and each distribution now carries its own `LICENSE`. Wheels emit
  `License-Expression: Apache-2.0` and `License-File: LICENSE`.
- **Front-door claims corrected before publishing (#189).** The README promised "thin
  adapters run the same Stages on Databricks or Airflow" when only `latence-runner-airflow`
  exists; it quoted 32 ADRs against 52 on disk; it advertised a `latence` meta-package that
  is not publishable. All three are fixed in the README and in the matching prose in
  `CONTEXT.md`, `docs/index.md` and `docs/concepts/`. The dangling `docs/INTEROP.md`
  references (a document that was never written) now point at
  `docs/RETRIEVAL-ENGINE-INPUT-SPEC.md`. `THIRD-PARTY-LICENSES.md`'s version snapshot was
  re-read from the installed environment.
- **Docs navigation is complete (#189).** ADR-0032 and ADR-0047 were missing from both the
  site nav and `docs/adr/index.md`, and `deployment.md` / `observability.md` /
  `secrets.md` were published but unreachable from the nav. Heading slugs now use the
  GitHub-compatible slugifier, so a hand-written `#anchor` resolves on github.com *and* on
  the docs site (this fixes four broken in-page links in `deployment.md`).

## 0.1.0 — 2026-07-14 · internal version pin, never released

**This was not a release.** No `v0.1.0` tag exists (`git tag -l 'v*'` is empty), no artifact was
ever published, and nothing has ever been on PyPI — the same state `README.md` §"Status & maturity"
records. What happened on this date is the internal version pin: every publishable distribution
(`latence-core` plus the provider packages) was set to `0.1.0` so the workspace versioned as one
unit. The entry is kept because the work below is the substantive history of the project; the
heading is not a version link, because there is nothing to link to.

The licence in force at the time was Apache-2.0. The workspace was relicensed to
**PolyForm Noncommercial 1.0.0** on 2026-07-29 (ADR-0055) and back to **Apache-2.0** on
2026-08-15 (ADR-0063) — see the provenance record in [`NOTICE`](./NOTICE). Everything named
below therefore carries the same licence today that it carried in July; the noncommercial
interlude reached nobody, because nothing was ever published under it.

### Added

- **The pipeline spine** — a model-agnostic, cloud-agnostic set of Stage contracts
  and a local Runner that turn messy, siloed source data into AI-ready outputs:
  Source → Parse/OCR → Chunk → Screening → Entity/Relation Extraction →
  Schema Induction → Redaction (PII) → Profiling → Disambiguation (entity linking /
  GLinker-style resolution) → Graph Assembly → Graph Completion → Context Enrichment →
  Export (Parquet · JSONL · TTL · GraphML). Every Stage declares typed input/output
  contracts and is fulfilled by a swappable Provider discovered via entry points.

- **`latence-core`** — data contracts, Capability protocols, the Pipeline + local
  Runner, the `latence` CLI (`stack validate`, `bake-off`), the Storage seam
  (`file://` and `s3://` via fsspec), and the near-zero-dependency built-in reference
  Providers (local-folder source, markdown/sentence chunkers, gazetteer entities,
  pattern relations, hybrid-rule redaction, statistical/lightweight profiling, cascade
  and exact-surface disambiguation, canonical/weighted graph assembly, reference graph
  completion, KG-header context enrichment, hashing embedder, JSONL/Parquet + KG export).

- **The provider ecosystem** — each heavy/endpoint model isolated behind a Stage seam in
  its own package so `latence-core` stays thin (ADR-0016):
  - _Parse / OCR_ — `latence-parser-plaintext`, `latence-parser-document`,
    `latence-parser-pdfplumber`, `latence-parser-render` (universal any-file, license-clean),
    `latence-parser-lighton`, `latence-parser-lighton-vllm`, `latence-parser-glm`,
    `latence-parser-endpoint`.
  - _Entities / Relations_ — `latence-ner-gliner`, `latence-ner-endpoint`,
    `latence-relation-gliner`, `latence-relation-llm`, `latence-extract-gliner2`,
    `latence-schema-inducer`.
  - _PII / Redaction_ — `latence-pii-gliner`, `latence-pii-gliner2`, `latence-pii-presidio`.
  - _Embedders_ — `latence-embedder-st`, `latence-embedder-endpoint`.
  - _Disambiguation_ — `latence-disambig-embedding`, `latence-disambig-glinker`.
  - _Graph completion_ — `latence-linkpred-ultra`.
  - _Runners_ — `latence-runner-airflow`.
  - _Demo_ — `latence-demo` (end-to-end showcase).

- **Enterprise-readiness gates** — a Provider Conformance suite parametrized over every
  registered Provider (contract, graceful failure, license declaration, determinism),
  the Goal-A end-to-end stack validation on a bundled corpus, and the Goal-B per-Stage
  bake-off matrices (quality/latency/memory/cost/license/determinism/error-rate).

- **Packaging & metadata** — every publishable package carries full `[project]` metadata
  (description, readme, `Apache-2.0` license, authors, keywords, PyPI classifiers marked
  `Development Status :: 4 - Beta` and `Typing :: Typed`, and Homepage/Repository/Issues/
  Changelog URLs).

### Engineering

- **CI** split into a fast required lane (`ci.yml`: secret-scan, lint+type fast gate,
  test matrix on py3.11/3.12 with surfaced coverage, Goal-A stack validation, all-package
  build) and a heavy lane (`ci-heavy.yml`: MinIO `s3://` e2e, real-Airflow DAG tests, and
  the Goal-B bake-offs) that also runs nightly. Actions are pinned by commit SHA with
  least-privilege `permissions`, per-ref `concurrency`, and docs-only `paths-ignore` on the
  heavy lane. `scripts/verify-local.sh` mirrors the gates for offline, deterministic local
  verification.

- **Release engineering** — `release.yml` builds all distributions and cuts a GitHub
  Release with generated notes on a `v*` tag; a PyPI Trusted-Publishing (OIDC) job is
  scaffolded, environment-gated, and disabled pending the maintainer's go-ahead
  (see `docs/RELEASING.md`).

<!--
No version link definitions here, deliberately. Keep a Changelog links each heading to a
compare/tag view; this project has cut no `v*` tag and published no release, so
`compare/v0.1.0...HEAD` and `releases/tag/v0.1.0` would both 404 the moment the repository is
public. The link definitions go back in — one per released version — when the first real tag is
cut per `docs/RELEASING.md`, at which point the `[Unreleased]` block above becomes that release's
section.
-->
