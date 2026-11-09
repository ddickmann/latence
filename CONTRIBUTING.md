# Contributing to the Latence Framework

Thanks for your interest. This is a model-agnostic, cloud-agnostic framework for turning
messy enterprise data into AI-ready data. It is built to a deliberately high bar
(`issues/DEFINITION-OF-DONE.md`); contributions are held to the same bar.

Read this page before you open a pull request. Two things in it are unusual and will
otherwise cost you a review cycle: the **CLA requirement** below, and the **doc-truth
gates** — parts of the documentation are machine-checked against the code, so a prose
change can turn the suite red.

---

## Before you write code: the Contributor Licence Agreement

**Every contribution requires a signed Contributor Licence Agreement (CLA) before it can be
merged.** No exceptions, including for one-line fixes and documentation.

**Why:** the CLA was written for the licence the project carried between 2026-07-29 and
2026-08-15 — PolyForm Noncommercial, where commercial rights were reserved to the owner and
an inbound contribution had to arrive with an explicit grant or those rights became
unenforceable. The project is now **Apache-2.0** (ADR-0063), whose §5 places every
contribution under the same licence as the work, so a DCO would be sufficient and the CLA's
original reason is gone.

**Whether to replace it with a DCO is an open decision for the maintainer, recorded in
ADR-0063.** Until he takes it, this requirement stands as written — a stated precondition
that is quietly not enforced is worse than either alternative. What the CLA still buys is a
recorded, explicit grant and the ability to relicense later without tracing contributors.

It is not a claim on your other work, it does not stop you using your own contribution
however you like, and it is not a judgement about you.

Practically:

- **The agreement is [`CLA.md`](CLA.md), at the repository root.** It is an Apache ICLA
  derivative — about two pages, and its provenance section lists exactly what was changed
  and why. This page states the *requirement*; `CLA.md` is the agreement and is the only
  part of the two that grants anything.
- **Sign it once, not per pull request.** Sign electronically, or fill in the block at the
  end and email it to **ddickmann81@gmail.com** with the subject
  `CLA: latence`. The maintainer keeps a private record of who has signed.
- **Open the pull request as normal.** You do not have to sign before you start work — but
  no PR is merged until the signature is recorded, so send it early rather than at the end
  of review.
- You must have the right to grant what the CLA grants. If you wrote the change on an
  employer's time or equipment, your employer may own it — get that cleared first. Say so
  in the PR if you are unsure; it is far cheaper to resolve before review than after. In
  that case a **corporate** agreement, not this one, is usually what is needed, and there
  is not one drafted yet — so raise it before you write the code, not after.
- Contributions that cannot be covered by a CLA cannot be merged. They can still be
  valuable as a **bug report with a reproduction**, and that route is genuinely welcome —
  see the issue templates in [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/).

> `CLA.md` is a **draft that has not yet been reviewed by a lawyer**, and it says so at the
> top. The review is already an open item on `RELEASE-GATES.md` Gate 3. Read the document
> before you sign it; if something in it looks wrong to you, open an issue and say so —
> that is a genuinely useful contribution.

Administration is manual — one maintainer, an email, a written record. That is proportionate
to the number of contributors this project has today. Moving to
[cla-assistant](https://github.com/cla-assistant/cla-assistant) (a GitHub App: you comment to
sign, the bot records it and sets a status check) is the intended next step once a second
contributor makes the automation worth its own maintenance, not a promise with a date on it.

If the CLA is a blocker for you and the change matters, open an issue and say so. Some
changes can be re-implemented independently, which is a legitimate outcome.

---

## Development setup — a uv workspace

The monorepo is a [uv](https://docs.astral.sh/uv/) workspace: a thin `latence-core` plus one
installable package per Provider family (ADR-0016), so an adopter installs only what they
use and the CPU-first "runs on a laptop" promise holds (ADR-0007).

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
# 2. Sync the dev environment (core + CPU demo set only — no torch, no cloud client):
uv sync

# 3. Run the demo end-to-end (CPU-only, offline):
uv run latence-demo
```

The heavy and optional Provider packages (the GLiNER-family extractors and redactors, the
`*-endpoint` packages, the LightOn OCR parsers, `latence-runner-airflow`, …) are workspace
members but are **not** installed into the default dev env: they pull
`torch`/`transformers`, an inference client, or `apache-airflow`, any of which would break
the CPU-first, offline, deterministic test environment (ADR-0007/0016). Their *source* is
still type-checked, and their tests run offline by monkeypatching the heavy or endpoint
dependency with a faithful fake. To work on one, install its source without its heavy deps:

```bash
uv pip install --no-deps -e packages/latence-ner-gliner
```

`make sync` and `scripts/verify-local.sh` do this for the whole heavy set automatically. The
authoritative list is [`scripts/heavy-packages.txt`](scripts/heavy-packages.txt), which those two
and the CI setup action all read; `packages/latence-core/tests/test_heavy_package_list.py` asserts
that every workspace package is reachable by one of the documented install paths. That gate exists
because the list used to be three hand-synchronised copies, one of which had drifted to 9 of 21 —
so `make sync` produced an environment where twelve Providers never registered, which surfaces as
*registry* failures rather than import errors and reads like a code bug rather than a setup one.

**Optional but recommended — the commit hooks.** `.pre-commit-config.yaml` runs the two fastest
CI gates at commit time: the ruff lint (identical scope to CI) and the `detect-secrets`
audited-baseline gate. The second is the one worth having, because a secret caught after it is
committed means rewriting history.

```bash
uv run --with pre-commit --no-project pre-commit install
```

Every hook is `local` / `language: system` and invokes the *same command* as `ci.yml` — a test
asserts the strings still match, because a hook that has drifted from CI is worse than no hook.
It is not a substitute for `scripts/verify-local.sh`, which stays the merge gate.

---

## The gates

### `scripts/verify-local.sh` is the merge gate

It mirrors the CI lint · type · test · build lane plus the secret-scan, provider-conformance,
stack-validate, environment-spec, compatibility-matrix and bake-off gates — same commands,
same environment (`PYTHONHASHSEED=0`, `LATENCE_CUDA=0`), same order. Exit 0 means every
mirrored gate is green; non-zero is the first failing gate.

```bash
bash scripts/verify-local.sh > /tmp/verify.log 2>&1; echo "REAL_EXIT=$?"
```

**Capture the exit code by redirect, never by piping to `tail`.** A shell pipeline reports
the status of its *last* command, so `bash scripts/verify-local.sh | tail -20` exits `0` on
a thoroughly red tree. That mistake was made once in this repo and produced a false green
that survived into a written summary. Redirect to a file, echo `$?`, then read the log.

The script does not cover the S3-emulator `s3://` end-to-end lane, the Airflow DAG lane, or real
GPU-provider runs — those need containers, a heavy install, or a CUDA host. It prints that
list when it finishes. Say in your PR which of them you could not exercise.

### Running the tests directly

```bash
PYTHONHASHSEED=0 uv run pytest packages/latence-core -o addopts=""
```

Both halves are load-bearing:

- **`PYTHONHASHSEED=0`** — some code paths iterate dicts and sets keyed by strings, whose
  order depends on the per-process hash seed. The seed is read once at interpreter startup,
  so it must be set *before* the process launches; setting it from a fixture is a no-op that
  only misleads. The repo-root `conftest.py` is the guard: it inspects
  `sys.flags.hash_randomization` and fails the run with one clear message rather than letting
  a hash-order-dependent test flake silently. `make test` sets it for you.
- **`-o addopts=""`** — `pyproject.toml` sets `addopts = "-q"`. Pass your own `-q` on the
  command line and pytest resolves it to `-qq`, which suppresses the summary line entirely,
  so you get no pass/fail counts at all. Clearing `addopts` gives you back the numbers you
  are presumably running the suite to see.

A test that depends on dict or set ordering, or on wall-clock time, is a bug — not a flake.

### Lint and types

```bash
uv run ruff check packages
MYPY_CMD="uv run mypy" bash scripts/typecheck.sh
```

**mypy runs `--strict` over `src` *and* `tests`, in every package.** Tests are typed to the
same standard as source, deliberately (ADR-0032): an untyped test is exactly where a
contract regression hides.

`scripts/typecheck.sh` runs mypy **per-package, not whole-tree**, and that is not a style
preference. The package directories are hyphenated, so a single `mypy --strict packages`
maps every `tests/conftest.py` onto the same bare `conftest` module and aborts with a
duplicate-module error. The per-package loop keeps each `conftest` unique within its own
process while still strict-checking every `src` and `tests` file.

---

## The test-first rule

**A fix lands with a test that failed before the fix.** Not a test that exercises the new
code — a test that was *red* on the tree you started from and is *green* on the tree you are
proposing. If reverting your source change does not turn a test red, the change is
unprotected and the next refactor will quietly undo it.

Concretely:

- **Bug fix** — reproduce the bug as a test first, watch it fail, then fix it.
- **Security, DoS or ReDoS fix** — the failing test is the pathological adversarial input
  itself: a zip bomb, a type-spoofed header, a cross-chunk injection, an oversized file.
- **Performance change** — assert the new bound, or that the faster path is genuinely wired
  and taken. A benchmark number in the PR description protects nothing.
- **A claim added to a published page** — prefer adding it as a doc-truth gate (below) over
  prose that someone must remember to update.

There is a second rule that is easy to violate honestly: **a fake must be able to disagree
with you.** Heavy and endpoint Providers are tested with their real dependency
monkeypatched, and that fake has to mirror the real API's shape and reject what the real one
rejects. A stub that always agrees turns a passing test into a decoration. If you cannot
make a fake faithful, run the real dependency or mark the test skipped with a stated reason
— never fake the green.

---

## Definition of Done — declare your tier

Every change declares which bar it meets (`issues/DEFINITION-OF-DONE.md`):

- **Baseline bar** — works; strictly typed (src **and** tests); tested on the happy path
  **and** the key failure modes, with at least one end-to-end test through the local Runner;
  CI-green; correctly packaged (heavy deps isolated); licensed; secure-by-default;
  observable; documented.
- **Gate bar** — the G1 profound-testing gate: end-to-end on real messy corpora, security
  review, reliability and crash-resume, a recorded performance baseline, gold-set quality,
  determinism fingerprint.
- **Hardening bar** — Phase-2: adversarial, property-based and fuzz tests; scale validation;
  full observability; at least two Providers per Capability; benchmarks against a named
  baseline.

Whatever the tier, preserve the contracts: inter-Stage data is versioned Pydantic;
`Provenance` and `Classification` are present and validated at every boundary you touch;
`SCHEMA_VERSION` is bumped on any contract change; offsets, page provenance, Evidence and
Delta idempotency stay intact. And keep heavy dependencies out of `latence-core` (ADR-0016)
— core stays near-zero-dep and pure-Python.

---

## The doc-truth gates

Documentation that makes a claim about the code is **machine-checked**. These gates are
ordinary pytest modules, so a doc that drifts from the tree fails the build like any other
regression. If your prose edit turns one red, the gate is usually right.

The front-door set — `README.md`, `CONTEXT.md`, `CONTRIBUTING.md`, `SECURITY.md`, and every
tracked page under `docs/` — is checked for, among other things:

- **Dangling cross-references.** No relative markdown link and no backticked reference to a
  file under `docs/` may point at something that does not exist. The rule extends to
  `CHANGELOG.md` and to every `packages/*/README.md`, because each of those is a
  distribution's PyPI long description — a broken link there ships for the life of the
  release.
- **Dangling section anchors.** A `#fragment` must resolve to a real heading on the page it
  points at, under GitHub's own slug rules. A wrong anchor renders perfectly and silently
  dumps the reader at the top of a long document.
- **Counts.** A stated size of the Decision Log must equal the files on disk. That one has
  drifted before, in both directions, across several pages at once.
- **Shipped versus aspirational.** A substrate the front door says the Stages *run on* must
  correspond to a real `packages/latence-runner-*` distribution, unless the sentence
  explicitly marks it as not shipped. Every `pip install` line must name a distribution that
  actually exists.
- **Licence summaries.** No published page may describe the dependency tree as uniformly
  permissive while the disclosure files say otherwise.

Beyond the front door there are targeted gates on individual documents: the benchmarks page
against the results actually published, the third-party retrieval input spec against the
current `schema_version`, the pod tutorial against the stack the setup wizard really
generates, and the GPU compatibility matrix against its own evidence — every cell is either
measured, carrying date, hardware, command and observation, or explicitly `UNVERIFIED` with
a runnable reproduce command.

The rule underneath all of them: **assert observable repository facts, never a flag that
records intent.** If you add a doc-truth gate, make it fail on the tree that motivated it
before you make it pass.

---

## The ADR process

Hard decisions live in `docs/adr/` as numbered Architecture Decision Records. The Decision
Log is the authoritative account of *why* each seam is shaped the way it is; the concept
pages are navigational summaries over it.

**What warrants an ADR:** a decision that is surprising, non-obvious, hard to reverse, or
that someone will otherwise re-litigate in six months. Choosing a default model. Adding or
changing a Capability. A licence or dependency policy. A trade-off where the rejected option
was genuinely reasonable. Routine implementation does not need one — if the next reader would
simply nod, skip it.

**Shape.** `NNNN-kebab-case-title.md`, opening with a `**Status:**` line, then context, the
decision, and — the part that carries the weight — **the alternatives considered and why they
were rejected.** An ADR that only records what was chosen is half an ADR; the value is in the
paths it closes off. Name the existing ADRs it builds on.

**Status is honest.** `accepted`, `proposed`, `partially accepted` and `superseded` are all
in use. A decision that only half landed says so, and names which half.

**Superseding.** A later ADR supersedes an earlier one by saying so explicitly, in both
records; the superseded ADR is amended in place with a pointer forward, never deleted. Numbers
are decision order, so the history stays readable.

**Correcting an overclaim counts as a fix.** If a docstring or an ADR claims more than the
code does, tighten the wording to match reality as part of your change. That is not scope
creep here; it is the standard.

`CONTEXT.md` is the canonical language. Use its glossary terms exactly — Stage, Runner,
Capability, Provider, Provenance, Evidence, Quarantine — and avoid the synonyms it bans. A
genuinely new concept gets a glossary entry in the same PR.

---

## Adding a Provider without touching core

This is the seam the whole framework rests on (ADR-0004): the framework names **Capabilities**
and lets **Providers** fulfil them, so it never names a model. An in-process CPU model, a GPU
model, a remote inference endpoint and a cloud AI service are all just Providers of the same
Capability.

A new Provider is a **new package**, not an edit to `latence-core` (ADR-0016):

1. **Create `packages/latence-<capability>-<name>/`** with its own `pyproject.toml`. Its heavy
   dependencies belong to it and stay out of core.
2. **Implement the Capability protocol** — the narrow typed interface the Stage depends on
   (`Source`, `Parser`, `Chunker`, `IntakeScreener`, `ContentScreener`, `EntityExtractor`,
   `RelationExtractor`, `PIIDetector`, `Disambiguator`, `GraphAssembler`, `Embedder`,
   `Export`, and the rest). Structural typing is enough; no base class is imposed. The
   optional `AdapterBase` in core factors out the cross-cutting glue — device resolution,
   batching, native-exception-to-typed-error mapping — and adds no heavy dependency; see
   [`docs/writing-an-adapter.md`](docs/writing-an-adapter.md).
3. **Declare a `ProviderProfile`** — licence with its evidence trail, determinism, compute
   (`cpu` / `gpu` / `either`), memory, batching, cost. See
   [`docs/provider-profile.md`](docs/provider-profile.md). This is not metadata theatre; the
   conformance gate reads it.
4. **Register the plugin via an entry point.** One block in your package's `pyproject.toml`:

   ```toml
   [project.entry-points."latence.providers"]
   "parser.lighton" = "latence_parser_lighton.provider:LightOnOcrParser"
   ```

   That is the entire registration. No core file changes and no central list to append to —
   the conformance roster is *derived* from the workspace's package manifests precisely so it
   cannot drift from what the workspace actually contains.
5. **Clear the licence gate before it can ship as a default** (ADR-0012). Weights and code
   licences are verified **separately**: a checkpoint's licence is per-weight and cannot be
   inferred from the library it runs under. Only permissive weights (Apache-2.0 / MIT / BSD)
   ship as defaults; a restricted licence makes the Provider opt-in and non-default, and its
   flow-down obligations must be surfaced. Record the result in `NOTICE` and
   `THIRD-PARTY-LICENSES.md`.
6. **Pass conformance.** The suite in `latence-core` parametrizes over every registered
   Provider and asserts C1–C6: valid typed records with Provenance, Classification and
   in-range offsets; graceful failure on adversarial input through a *typed* error, with no
   PII in the message; a recorded, evidence-bearing licence declaration; determinism where
   declared; the declared device honoured (a `compute="gpu"` Provider on a CPU host is
   skipped-with-flag, not failed); and no secret or PII leaking into non-content fields. See
   [`docs/CONFORMANCE.md`](docs/CONFORMANCE.md). Passing conformance *is* the definition of
   enterprise-ready here.

Use the **provider request** issue template to propose one — it collects exactly what the
licence and conformance gates need, and it is much cheaper to find a blocker there than in
review.

A **Runner** adapter is a different seam and does not go through the Provider entry point.
Runners are the execution substrate the Stage library runs on (ADR-0003) — the local
streaming Runner ships in core, and `latence-runner-airflow` is a separate distribution.
Adding a third is a package, not a core change, but talk to the maintainer first: the Stage
contract, not the substrate, is what has to stay stable.

---

## Review and merge

Work is tracked as version-controlled issues under `issues/` (WIP = 1 on the critical
path). An issue is done when its declared DoD bar is met and the gates are
green — not when the change looks right.

The reviewer verifies the claim against the **code**, not against the PR description. Make
that easy: name the test that fails without your change, and state which gates you ran and
which you could not.

---

## Contributing with AI

Using an AI assistant to write code for this project is **allowed and expected**. Three rules,
and they are about *disclosure and ownership*, not about restriction.

**1. Say which parts — for the reviewer, not for the record.** One line in the PR description is
enough ("the Provider adapter and its tests were drafted with an assistant; the offset-mapping fix
is hand-written"). This is not a confession and it is not provenance accounting. **Much of this
codebase was itself written with an assistant, and the git history deliberately carries no
per-commit AI attribution** — so nothing here treats "AI-assisted" as a mark against a change, and
you are not being asked to declare something the maintainer does not. The reason to say it is
narrower and entirely practical: it tells a reviewer where to concentrate. Assistants fail in a
recognisable shape — plausible prose over an unchecked claim, a test that asserts the mock rather
than the behaviour, a docstring describing what the code was *meant* to do. Pointing at the
generated part aims the review at where that shape is likeliest.

**2. You own it.** Submitting a change asserts that you understand it, that you can explain why
it is correct, and — under the CLA above — that you have the rights to contribute it. "The model
generated it" is not an answer to a review question, and it is not a defence if the code turns
out to be copied from somewhere incompatible with this project's licence. If you cannot defend a
line under questioning, do not ship it; that is true of hand-written code too, and an assistant
just makes it much easier to produce a lot of it quickly.

**3. Write the PR description yourself.** The prose a human reviewer reads should come from the
human asking for the review. Generated summaries reliably restate the diff and reliably omit the
thing the reviewer needs — what you *tried*, what you *rejected*, and which claim you are least
sure of. The PR template asks for "the test that was red" for exactly this reason: a test that
fails before your change and passes after is a fact about the code, and no amount of confident
narration substitutes for it.

Two things follow from the way this repository is gated, and they cut in the contributor's
favour. Fabricated confidence has a short life here: `scripts/verify-local.sh`, the strict type
check, the Provider Conformance suite and the doc-truth gates are the arbiter, not the PR
description. And the doc-truth gates apply to generated prose exactly as they do to hand-written
prose — a README or docstring claim that the tests can falsify will be falsified, whoever or
whatever typed it.

---

## Security issues

**Do not open a public issue or pull request for a suspected vulnerability.** Report it
privately — GitHub Security Advisories, or the maintainer email — as described in
[`SECURITY.md`](SECURITY.md), which also documents the trust model, the built-in controls,
and what is deliberately out of scope.

If you are contributing to a security-relevant surface, two properties are non-negotiable and
are enforced by tests: ingested documents are treated as hostile, and the universal
financial-PII redaction floor cannot be disabled by choosing a different Provider.
`SECURITY.md` explains both.

## Code of Conduct

This project follows the Contributor Covenant. By participating you are expected to uphold
it: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
