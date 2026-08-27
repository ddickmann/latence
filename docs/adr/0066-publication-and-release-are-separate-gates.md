# Publication and release are separate gates

**Status: accepted (2026-08-26).** The repository is published publicly as a pre-1.0 preview
**before** Gate 1 of [`RELEASE-GATES.md`](https://github.com/ddickmann/latence/blob/main/RELEASE-GATES.md) is met. Publication of the
source and cutting a versioned release become two different gates with two different bars.
**Amends** `RELEASE-GATES.md`, which previously conflated them. No API, contract, schema
version, default or artifact changes.

## Context

`RELEASE-GATES.md` opened with: *"This repository is **private**. It is not published publicly
until every gate below is green."* Five gates sit under that sentence, and
`packages/latence-core/tests/test_public_release_gate.py` enforces them: under
`LATENCE_RELEASE_CHECK=1` it fails while any placeholder or unchecked box remains.

Run today, that check reports **27 unchecked items and 52 `PENDING:` tokens** in
[`BENCHMARKS-E2E.md`](../BENCHMARKS-E2E.md). Reading the 27 individually, they fall into two
piles that the single sentence above cannot distinguish:

- **Bookkeeping that is already true.** "The full suite passes with `PYTHONHASHSEED=0`",
  "`mypy --strict` passes on src *and* tests", "`CONTRIBUTING.md` states the CLA requirement",
  "`LICENSE` is the intended licence", and the licence-coherence items that
  [ADR-0063](0063-relicence-to-apache-2-0.md) dissolved. These are verifiable right now; the
  checkbox was simply never flipped.
- **Genuinely open.** All seven of Gate 1 — the end-to-end protocol on named hardware, per-stage
  VRAM, hop-stratified retrieval, the KG ablation, link prediction — plus legal review and the
  environment-gated lanes.

The second pile is real and is not being waved away. The question this ADR settles is narrower:
**does an unmeasured benchmark protocol justify keeping the source unreadable?**

## Decision

It does not. Publication and release are separated:

- **Publication** — the source is readable on GitHub — requires that every claim the repository
  makes is either substantiated or **labelled as unsubstantiated**. That bar is met.
- **Release** — a `v*` tag, signed artifacts, an SBOM, distributions people install — continues to
  require every gate in `RELEASE-GATES.md`, Gate 1 included. `LATENCE_RELEASE_CHECK=1` stays the
  strict check, and the `pypi-publish` job stays disabled.

`RELEASE-GATES.md` is amended to say this, and to move its checked/unchecked bookkeeping into
line with the tree.

## Why

**The gate was protecting against a specific failure, and that failure is overclaim, not
visibility.** Its own stated rationale is *"publishing a framework whose headline claims a
sophisticated evaluator can falsify in an afternoon."* A reader who clones this repository today
finds: a README whose Status section enumerates what is *not* proven before it enumerates what
is; a `BENCHMARKS-E2E.md` whose every cell reads `PENDING:` under a banner saying so; a
`PERF-RESULTS.md` carrying an explicit *not yet measured* list; and a `ROADMAP.md` with eleven
known-limitation bullets including that the zero-model default query tier does not pass the gate
on the harder multi-hop dataset. None of that is falsifiable in an afternoon, because none of it
is a claim.

**Holding source private does not make an unmeasured number measured.** Gate 1 is blocked on GPU
time and a specific reproduction protocol. Nothing about the repository being private advances it,
and nothing about it being public retards it. The two facts are independent, and coupling them
cost the project the only thing publication buys — outside review of the design, which is exactly
what a single-maintainer project (see [`CONTINUITY.md`](../CONTINUITY.md), *"Bus factor: 1"*)
most needs and can least manufacture internally.

**The stricter reading defeats itself.** Taken literally, the original sentence says the source
stays private until link-prediction precision is measured on a held-out sample. That is a
research milestone gating an engineering artifact. Held consistently it would keep the repository
private indefinitely; held inconsistently it would be broken quietly, which is worse than
amending it in the open.

**The honest-labelling discipline is load-bearing and stays.** The `PENDING:` tokens are not
decoration: `test_public_release_gate.py` asserts they are well-formed, greppable and
non-vacuous, and fails if the results page stops declaring its own status. That machinery is
untouched. What changes is only which act it gates.

## What this obliges

1. `RELEASE-GATES.md` states the publication/release split in its opening lines, and its
   checkboxes reflect the tree — items that are true get ticked, Gate 1 stays unticked.
2. **The strict check must actually run on the path it gates.** `LATENCE_RELEASE_CHECK` was
   documented as the enforcement mechanism but appeared in no workflow, no script and not the
   Makefile — so `README.md`'s claim that publishing with an unfilled table "is not possible" was
   false; it only failed if someone remembered the flag. It is wired into the release workflow,
   so a `v*` tag cannot proceed past a red gate.
3. Every page that claims a number keeps naming its method and hardware, and silence never
   implies a measurement.

## Alternatives rejected

**Hold publication until Gate 1 is measured.** The consistent reading of the original text. It
buys nothing an evaluator can perceive — a private repository and a public one with a
`PENDING:` table are equally uninformative about link-prediction precision — and it indefinitely
defers the review that would most improve the work.

**Publish and quietly leave the gate red.** Cheapest, and corrosive. `RELEASE-GATES.md` would
assert something false about the project on its first line, in a repository whose distinguishing
discipline is that documentation which makes a claim about the code is machine-checked. A project
that breaks its own stated gate at the first inconvenience has a weaker gate everywhere else.

**Delete Gate 1.** Removes the embarrassment by removing the standard. The measurements are worth
making and the protocol is worth keeping; what was wrong was which door it was bolted to.
