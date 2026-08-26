# AGENTS.md

Instructions for coding agents (Claude Code, Codex, Cursor, Aider, and anything else that edits
this repository unattended). Human contributors want [`CONTRIBUTING.md`](./CONTRIBUTING.md) — it
covers the CLA, the review process and the AI-disclosure policy, and it is the authority on all
three. This page is the operational half: the repository's shape, the gates, and the specific
things that are easy to get wrong here and expensive to get wrong.

> **Are you here to *use* the framework rather than change this repository?** You want
> [`docs/agent-briefing.md`](./docs/agent-briefing.md) instead — the mental model, every Provider
> available at each Stage, the Pipeline YAML contract, the CLI, the output schema and the retrieval
> wiring. [`llms.txt`](./llms.txt) indexes the rest. This page is for agents **editing this
> codebase**: the gates, the contracts, and the traps.

Two ground rules before anything else:

1. **The tree is the authority.** Every claim below was read out of a file at the path named. If a
   command here disagrees with the repository, the repository is right and this page is a bug —
   fix it in the same change.
2. **This repository rejects plausible-sounding claims.** Documentation that makes an assertion
   about the code is machine-checked by ordinary pytest modules — see item 4 of § *Six things that
   will burn you*. A confident sentence you cannot substantiate will not merge; it will turn CI red.

---

## What this is

A model-agnostic, cloud-agnostic Python framework that turns messy enterprise documents into
AI-ready data: parse → chunk → screen → extract entities and relations → redact PII → resolve →
knowledge graph → export. The framework names **Capabilities** (narrow typed interfaces) and lets
**Providers** fulfil them; the core never names a model. [`CONTEXT.md`](./CONTEXT.md) is the
canonical glossary — use its terms exactly (Stage, Runner, Capability, Provider, Provenance,
Evidence, Quarantine) and avoid the synonyms it bans.

---

## Repository shape

A [uv](https://docs.astral.sh/uv/) workspace. `[tool.uv.workspace] members = ["packages/*"]` in the
root `pyproject.toml`; **34 packages** under `packages/`, each an independently publishable
distribution.

**One of them is deliberately NOT a workspace member.** `packages/latence-gliner25` pins a
`gliner2` major disjoint from the one the other Provider packages pin, and a uv workspace is ONE
resolved environment — `uv lock` refuses the set, correctly — so it is listed under
`[tool.uv.workspace] exclude` (ADR-0064). Everything that works by directory (ruff,
`scripts/typecheck.sh`, pytest, the contract reference, the environment specs) covers it unchanged;
three things had to be taught about it, and all three are gated: the release builds by **path**
(`uv build "$pkg"`, not `--package <name>`), the package carries its **own committed `uv.lock`**,
and `scripts/generate-sbom.sh` folds that lock into the SBOM union.

```
packages/latence-<family>-<name>/     33 of these
├── pyproject.toml                    deps, classifiers, the entry-point registration
├── LICENSE                           must match the root LICENSE
├── README.md                         this distribution's PyPI long_description
├── src/latence_<family>_<name>/      note: underscores here, hyphens in the directory name
│   ├── __init__.py
│   ├── provider.py
│   └── py.typed
└── tests/
    ├── conftest.py
    └── test_*.py
```

**The directory is hyphenated, the module is underscored.** `packages/latence-parser-pdfplumber/`
holds `src/latence_parser_pdfplumber/`. This is not cosmetic — it is the direct cause of the mypy
rule in § *Six things that will burn you*.

Everything else at the top level:

| Path | What it is |
|---|---|
| `packages/latence-core/` | contracts, Capability protocols, the plugin registry, the local Runner, Storage, the CLI, the conformance suite. Near-zero-dep and pure-Python **by policy** (ADR-0016) |
| `packages/latence-retrieval/` | the stateless query-time library. A separate downstream layer, not a Stage; it registers under a different entry-point group |
| `benchmark/` | measurement harnesses. `benchmark/s5/` is the retrieval matrix, `benchmark/serving/` the live-engine lane, `benchmark/sota-campaign/` the pipeline campaign |
| `scripts/` | `verify-local.sh` (the merge gate), `typecheck.sh`, the SBOM and secret-scan gates |
| `docs/` | the mkdocs site, including `docs/adr/` — the Decision Log |
| `stacks/`, `matrix/`, `examples/` | shipped pipeline configs, bake-off matrices, example configs — all executed by gates, none of them decoration |
| `conftest.py` (root) | the determinism guard. Read it before you run anything |

---

## Setup

```bash
uv sync                                          # the CPU-first dev env: core + the CPU reference set
uv run latence-demo                              # end-to-end smoke: CPU-only, offline, exits non-zero on drift
```

The heavy and endpoint Provider packages (the GLiNER family, the `*-endpoint` packages, the LightOn
OCR parsers, `latence-runner-airflow`, `latence-disambig-glinker`) are workspace members but are
**deliberately not installed** into the dev env — they pull torch/transformers, an inference client,
or Airflow, any of which breaks the CPU-first offline deterministic environment (ADR-0007/0016).
Their source is still type-checked, and their tests run offline against faithful fakes. To work on
one, install its source without its dependencies:

```bash
uv pip install --no-deps -e packages/latence-ner-gliner
```

`make sync` and `scripts/verify-local.sh` do this for the whole heavy set; the authoritative list is
[`scripts/heavy-packages.txt`](./scripts/heavy-packages.txt), which the Makefile, the verify script
and the CI setup action all read. It is gated by
`packages/latence-core/tests/test_heavy_package_list.py`, which also asserts that every workspace
package is reachable by one of the documented install paths — the three copies of this list had
drifted, and `make sync` was installing 9 of 21.

---

## The gates

Run these before you claim a change is done, **in this order** — cheapest first, so a lint break
costs you seconds rather than ten minutes.

```bash
# 1. lint — the scope CI uses, which is wider than `packages`
uv run ruff check packages benchmark/s5 benchmark/serving benchmark/sota-campaign scripts

# 2. strict types, src AND tests, per package (see below for why it is a script)
MYPY_CMD="uv run mypy" bash scripts/typecheck.sh

# 3. the suite — the seed is not optional, and `-q` is already in addopts (see #6 below)
PYTHONHASHSEED=0 uv run pytest packages

# 4. the docs site, if you touched anything under docs/ or mkdocs.yml
uv sync --extra docs && uv run mkdocs build --strict
```

Scoping the suite while you iterate is fine and encouraged:

```bash
PYTHONHASHSEED=0 uv run pytest packages/latence-core/tests/test_conformance.py -o addopts=""
```

**The merge gate is `scripts/verify-local.sh`.** It mirrors CI's lint · type · test · build lane
plus the secret scan, Provider conformance, nine end-to-end stack validations, the reproducible
environment specs, the compatibility matrix and seven bake-off matrices — same commands, same
environment (`PYTHONHASHSEED=0 LATENCE_CUDA=0`), same order.

```bash
bash scripts/verify-local.sh > /tmp/verify.log 2>&1; echo "REAL_EXIT=$?"
```

**Capture the exit code by redirect, never by piping.** A shell pipeline reports the status of its
*last* command, so `bash scripts/verify-local.sh | tail -20` exits `0` on a thoroughly red tree.
That mistake has been made in this repository and produced a false green that survived into a
written summary. It prints, on completion, the lanes it did **not** cover (the MinIO `s3://` e2e
lane, the Airflow DAG lane, real GPU runs) — say in your summary which of those you could not
exercise rather than implying you did.

---

## Six things that will burn you

### 1. `PYTHONHASHSEED=0` is mandatory, and it must be set before the interpreter starts

Some code paths iterate dicts and sets keyed by strings, whose order depends on the per-process hash
seed. The seed is read **once at interpreter startup**, so setting it from a fixture is a no-op that
only misleads — which is why no such fixture exists. The root `conftest.py` inspects
`sys.flags.hash_randomization` and aborts the whole run with one clear message if the seed is not
pinned. `make test` pins it for you; a bare `uv run pytest` does not.

A test that depends on dict/set ordering, or on wall-clock time, is a bug — not a flake.

### 2. The S5 benchmark harness additionally refuses without pinned BLAS threads

`benchmark/s5/run_matrix.py` refuses to run unless `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` **and**
`MKL_NUM_THREADS` are all `1`. This is a measured finding, not superstition, and the numbers are in
the module's `THREADS_REFUSAL` constant: the dense leg's BLAS `gemv` and the query embedder
reassociate their float sums per thread count, so at `OMP_NUM_THREADS=4` instead of `1`, on the same
code and the same corpus, 33 of 1,209 musique queries get a different dense pool and `recall@10`
moves from 0.3404 to 0.3406. An unpinned harness reports thread scheduling as a quality change.

Unpinned, it stops before doing any work and tells you exactly why:

```console
$ PYTHONHASHSEED=0 .venv/bin/python -m benchmark.s5.run_matrix --datasets multihop_rag --legs dense
refusing to run: pin OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1 (the dense leg's
BLAS gemv and the query embedder reassociate their float sums per thread count, which changes
rankings: measured, 33/1209 musique queries get a different dense pool at 4 threads vs 1).
Unpinned: OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS.
```

The invocation shape the harness documents for itself (`benchmark/s5/run_matrix.py`, module
docstring) is:

```bash
env PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    .venv/bin/python -m benchmark.s5.run_matrix --datasets vidoseek,multihop_rag --legs all --limit 50
```

Do not "fix" the refusal by deleting the check, and do not export the pins globally in a way that
hides which run they applied to — they are part of a recorded cell's identity.

### 3. mypy cannot be run whole-tree — use `scripts/typecheck.sh`

`uv run mypy --strict packages` **aborts before checking anything**. The package directories are
hyphenated, which mypy cannot turn into dotted module names, so every package's `tests/conftest.py`
maps onto the same bare top-level module `conftest` and mypy exits with `Duplicate module named
"conftest"`. `scripts/typecheck.sh` loops one package at a time — each `conftest` is then the only
`conftest` in its mypy process — and checks `src` **and** `tests` strictly for all of them. It is the
single command CI runs; a non-zero exit means at least one package failed.

Tests are typed to the same standard as source, deliberately (ADR-0032): an untyped test is exactly
where a contract regression hides. Do not add `# type: ignore` to make a strict error go away
without saying why in the same line.

One local-environment caveat, because it will confuse you otherwise: the root `pyproject.toml` marks
the heavy optional dependencies (`torch`, `transformers`, `qdrant_client`, …) with
`ignore_missing_imports`, which only applies while they are **absent** — which is the state CI's
CPU-first env is in. If you install torch or transformers into your local `.venv`, mypy starts
resolving their real stubs and reports mismatches against the deliberately-faithful fakes in those
Providers' tests, in packages you never touched. Those are an artefact of your environment, not a
regression you introduced. Check `git status` before you "fix" them, and do not edit a Provider you
did not otherwise change.

### 4. The doc-truth gates resolve links through `git ls-files`, not the working tree

`packages/latence-core/tests/test_front_door_truthfulness.py` checks the front-door pages —
`README.md`, `CONTEXT.md`, `CONTRIBUTING.md`, `SECURITY.md`, this file, and every **tracked** page
under `docs/` — plus `CHANGELOG.md` and every `packages/*/README.md`. It rejects:

- relative markdown links, and backticked `docs/…` / `issues/…` file references, that do not resolve;
- `#fragment` anchors that do not match a real heading under GitHub's slug rules;
- a stated size of the Decision Log that disagrees with the files in `docs/adr/`;
- any published page calling the dependency tree uniformly permissive while
  `THIRD-PARTY-LICENSES.md` discloses exceptions.

Two further checks are README-specific: an execution substrate presented as shipped must have a
matching `packages/latence-runner-*` distribution, and every `pip install` line must name a
distribution that exists.

**The trap for an agent:** existence is resolved against the git **index**, not the filesystem. A
link to a file you created but have not `git add`ed is green on your machine and red in CI's fresh
clone, and the failure message will point at the link rather than at the missing `git add`. If you
add a document and link to it, stage it in the same change.

`packages/latence-core/tests/test_continuity_statement.py` does the same for
`docs/CONTINUITY.md` and pins the release-signing and PyPI-publishing claims against the actual
workflow files. `packages/latence-core/tests/test_publish_readiness.py` polices licence pins,
contact addresses, and agreement between trove classifiers and the CI matrix — a classifier is a
claim, and the matrix is its only evidence, so widening either without the other fails.

The rule underneath all of them: **assert observable repository facts, never a flag that records
intent.** If you add a doc-truth gate, make it fail on the tree that motivated it before you make it
pass.

### 5. The licence topology is uniform, and it is pinned

The repository is **Apache-2.0**, everywhere, with no exception and no carve-out (ADR-0063, which
supersedes ADR-0055's relicence to PolyForm Noncommercial). Every source file that declares an SPDX
identifier declares `Apache-2.0`, and `LICENSE` is the canonical Apache-2.0 text byte for byte —
nothing added, not even a copyright line, because scanners identify that file by its bytes.

`test_publish_readiness.py` scans every `.py`/`.pyi`/`.rs`/`.sh` header in the tree and fails on
**any** identifier that is not `Apache-2.0`, digests `LICENSE` against the canonical text, and
checks that no owned document still grants PolyForm or claims commercial use needs an agreement.

So: if you add an SPDX header to a new file, it says `Apache-2.0` and nothing else. Do not invent a
per-file exception — if one is ever genuinely wanted it needs an ADR, a statement in the file, and a
pin in that test, in one commit, so the code, the licence document and the gate cannot disagree.

Two things are NOT licence drift and must survive:

- **`pack.py`'s inbound provenance.** It records that the QKP packer was ported from the
  maintainer's own `colsearch` project and relicensed Apache-2.0 for that use. That is *where the
  code came from*, not *what you may do with it*; both claims now name the same identifier, so the
  `INBOUND` / `OUTBOUND` labels are what keeps them apart. A gate requires them.
- **Third-party noncommercial terms.** `naver/splade-v3`'s CC-BY-NC-SA-4.0 weights and
  research-only corpora like OmniDocBench are still noncommercial and still gated behind their
  opt-in flags. The framework's outbound licence does not touch a third party's terms.

### 6. `-q` on the command line resolves to `-qq`

The root `pyproject.toml` already sets `addopts = "-q"`. Passing your own `-q` gives you `-qq`, which
suppresses the summary line entirely — you get no pass/fail counts at all. Clear it when you want
numbers:

```bash
PYTHONHASHSEED=0 uv run pytest packages/latence-core -o addopts=""
```

---

## Adding a Provider — do this instead of editing core

This is the single highest-leverage thing to know here. Swapping the OCR model, the induction LLM or
the embedder is a **config key**, not a refactor, and a new model is a **new package**, not an edit
to `latence-core` (ADR-0004/0016). An agent that "adds support for model X" by editing core has
written a fork; an agent that adds a package has written a plugin.

The full per-Capability replacement contract — what every Stage receives, what it must emit, the
wiring rules with their exact typed errors, and the swap walkthrough — is the **generated**
[`docs/reference/stage-contracts.md`](./docs/reference/stage-contracts.md) (machine-readable twin
`stage-contracts.json`, drift-gated against the descriptor table). Read that first when replacing a
component.

1. **Create `packages/latence-<capability>-<name>/`** with its own `pyproject.toml`, `LICENSE`,
   `README.md`, `src/latence_<capability>_<name>/` and `tests/`. Its heavy dependencies belong to it
   and stay out of core.
2. **Implement the Capability protocol** — the narrow typed interface the Stage depends on
   (`Source`, `Parser`, `Chunker`, `IntakeScreener`, `ContentScreener`, `EntityExtractor`,
   `RelationExtractor`, `PIIDetector`, `Disambiguator`, `GraphAssembler`, `Embedder`, `Export`, …),
   defined in `packages/latence-core/src/latence_core/capability.py`. Structural typing is enough; no
   base class is imposed. The optional `AdapterBase` factors out device resolution, batching,
   native-exception mapping and the provenance/offset carry — see
   [`docs/writing-an-adapter.md`](./docs/writing-an-adapter.md).
3. **Declare a `ProviderProfile`** — `compute` (`cpu`/`gpu`/`either`), `memory_mb`, `model_id`, the
   verified SPDX `license` (or the `UNVERIFIED` sentinel — never guess), `deterministic`, `batch`,
   `cost_per_1k`. See [`docs/provider-profile.md`](./docs/provider-profile.md). The conformance gate
   reads it; a missing profile fails, and a malformed one fails the Stage loudly rather than being
   read as "no profile".
4. **Register it with one entry-point block.** That is the entire registration — no core file
   changes and no central list to append to, because the conformance roster is *derived* from the
   workspace's package manifests precisely so it cannot drift:

   ```toml
   [project.entry-points."latence.providers"]
   "parser.pdfplumber" = "latence_parser_pdfplumber.provider:PdfPlumberParser"
   ```

   The group name is `latence.providers` (`ENTRY_POINT_GROUP` in `capability.py`). Query-time
   retrieval components use a separate group, `latence.retrieval`, kept off the Stage seam on
   purpose.
5. **Verify weights and code licences separately** (ADR-0012). A checkpoint's licence is per-weight
   and cannot be inferred from the library it runs under. Only permissive weights ship as defaults;
   a restricted licence makes the Provider opt-in and non-default, and its flow-down obligations go
   into `NOTICE` and `THIRD-PARTY-LICENSES.md`.
6. **Pass conformance.** The suite lives in
   `packages/latence-core/src/latence_core/conformance/` and is driven by
   `packages/latence-core/tests/test_conformance.py` and `test_e2e_conformance.py`. It parametrizes
   over **every** registered Provider and asserts C1–C6:

   ```bash
   LATENCE_CUDA=0 PYTHONHASHSEED=0 uv run pytest \
     packages/latence-core/tests/test_conformance.py \
     packages/latence-core/tests/test_e2e_conformance.py -o addopts=""
   ```

   C1 valid typed records (Provenance + Classification, in-range offsets, a sub-document carrier's
   page resolved through the parent's page map rather than inherited); C2 graceful typed failure on
   adversarial input, with no PII in the message; C3 a recorded, evidence-bearing licence; C4
   determinism where declared; C5 the declared device honoured (a `compute="gpu"` Provider on a CPU
   host is **skipped-with-flag**, never faked); C6 no PII in non-content fields. Details:
   [`docs/CONFORMANCE.md`](./docs/CONFORMANCE.md).

   Coverage is automatic — a Provider is mapped to its Capability by its entry-point **name prefix**
   (`entity.gliner` → Entity). Add a Provider for a Capability with no case and the suite fails
   loudly rather than silently skipping.

A **Runner** adapter is a different seam and does not go through the Provider entry point. Adding
one is still a package rather than a core change, but open an issue first: the Stage contract, not
the substrate, is what has to stay stable.

---

## Contracts you must not break

- **Provenance and Classification** are present and validated at **every** Stage boundary you touch.
  Offsets are exact half-open spans; a chunk resolves its own page slice rather than inheriting the
  document's, and a missing slice raises `PageSliceMissingError` rather than guessing a plausible
  wrong page. Never degrade to an inherited page range.
- **Inter-Stage data is versioned Pydantic.** Any contract change bumps `SCHEMA_VERSION` in
  `packages/latence-core/src/latence_core/contracts.py`.
- **The typed error taxonomy** in `packages/latence-core/src/latence_core/errors.py` —
  `LatenceError` → `ContractError` / `ConfigError` / `ProviderError` / `StorageError`, each also
  subclassing the builtin it replaces. Never swallow an error into a default, a `None`, or a quiet
  fallback. A Capability that cannot be served is a loud error; a missing Provider exits non-zero
  naming the package to install.
- **The universal PII floor.** A deterministic recogniser for `ssn` / `credit_card` / `iban` sits
  beneath every redaction Provider at the single `finalize_redacted_chunk` seam in
  `packages/latence-core/src/latence_core/redaction_policy.py`. It cannot be bypassed by choosing a
  different Provider, and a test enforces that. Do not add a code path around it.
- **Heavy dependencies never enter `latence-core`** (ADR-0016). No torch, no model libraries, no
  cloud clients. Optional integrations go behind extras and are imported lazily at first use, never
  at module import — Provider discovery and the device seam must work with none of them present.
- **Determinism.** Same input, same config, same bytes out. Emit records in a stable input order, or
  declare `deterministic=False` in your profile and mean it.

---

## No fake green

This is the prohibition that matters most, because it is the one an agent violates while trying to
be helpful.

- **A fake must be able to disagree with you.** Heavy and endpoint Providers are tested with their
  real dependency monkeypatched. That fake has to mirror the real API's shape and **reject what the
  real one rejects**. A stub that always agrees turns a passing test into a decoration. Fakes exist
  so the offline CI lane can run — they are not a substitute for the real dependency.
- **A benchmark number without the real dependency behind it is not a result.** If you cannot stand
  up the real thing, the honest outputs are a skip with a stated reason, a recorded
  `skipped-with-flag`, or a `missing-inputs` status — all three of which this codebase already
  emits. Never a zero, never an estimate, never a number carried over from a different
  configuration.
- **Never weaken a gate to pass it.** Deleting an assertion, widening an `exclude`, adding a
  `# type: ignore`, marking a test `xfail`, or relaxing a refusal check to make your change go
  green is a defect, not a fix. If a gate is genuinely wrong, say so explicitly and change it as its
  own decision.
- **Do not report a gate you did not run.** State which gates you ran, which you could not, and why.
  "Tests pass" without the command and its exit code is not a claim this project accepts from a
  human either.

---

## Before you claim done

1. **A fix lands with a test that failed before the fix** — not a test that exercises the new code,
   but one that was *red* on the tree you started from and is *green* now. If reverting your source
   change does not turn a test red, the change is unprotected.
2. Run the gates in order (§ *The gates*). Prefer `scripts/verify-local.sh` for anything beyond a
   trivial edit, and report which lanes you could not exercise.
3. `git add` anything you linked to, and check `git status` for files you created and forgot.
4. Keep the change scoped, and argue architecture rather than smuggling it in beside a bug fix. A
   decision that is surprising, hard to reverse, or that someone will re-litigate in six months
   needs an ADR in `docs/adr/`: `NNNN-kebab-case-title.md`, opening with a `**Status:**` line, and
   carrying **the alternatives considered and why they were rejected** — the part that does the work.
5. Correcting an overclaim counts as a fix, not scope creep. If a docstring, an ADR or a README line
   claims more than the code does, tighten the wording as part of your change.
6. Write your own summary: what you tried, what you rejected, and which claim you are least sure of.
   A restatement of the diff omits exactly what a reviewer needs.

---

## Where to read more

| | |
|---|---|
| Contributing, the CLA, the review process, AI disclosure | [`CONTRIBUTING.md`](./CONTRIBUTING.md) |
| The vocabulary — use these terms exactly | [`CONTEXT.md`](./CONTEXT.md) |
| Why each seam is shaped the way it is | [`docs/adr/`](./docs/adr/) |
| Writing a Provider adapter | [`docs/writing-an-adapter.md`](./docs/writing-an-adapter.md) · [`docs/provider-profile.md`](./docs/provider-profile.md) · [`docs/CONFORMANCE.md`](./docs/CONFORMANCE.md) |
| Running it on a real corpus | [`docs/getting-started/index.md`](./docs/getting-started/index.md) |
| What is proven and what is not | [`README.md`](./README.md) · [`RELEASE-GATES.md`](./RELEASE-GATES.md) |
| Security-relevant surfaces, and how to report a vulnerability privately | [`SECURITY.md`](./SECURITY.md) |
| Licence terms (Apache-2.0, uniform) and what commercial use needs (nothing) | [`LICENSE`](./LICENSE) · [`NOTICE`](./NOTICE) · [`COMMERCIAL.md`](./COMMERCIAL.md) |

**Do not open a public issue or pull request for a suspected vulnerability.** Report it privately as
described in [`SECURITY.md`](./SECURITY.md).
