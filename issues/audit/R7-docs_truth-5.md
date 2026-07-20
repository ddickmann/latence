# R7-docs_truth-5 — the GPU half of PERF-RESULTS.md has no measurement date, and the guard accepts an ADR id instead

**Severity:** major
**Area:** docs truth / published-number provenance
**Files:** `PERF-RESULTS.md`, `packages/latence-core/tests/test_perf_results_doc.py`,
`docs/adr/0036-provider-ecosystem-and-bakeoff.md`

## Failure scenario (verified)

The `## GPU envelope` section of `PERF-RESULTS.md` contains **zero** ISO dates
(`re.findall(r'\d{4}-\d{2}-\d{2}', gpu)` → `[]`). Its only date-shaped token is the string
`ADR-0036`, in `**Date measured:** the W8 acceleration run recorded in ADR-0036`. Splitting
`docs/adr/0036-provider-ecosystem-and-bakeoff.md` on its
`## Status note — the full-transformers acceleration seam (W8-perf-accel)` heading and searching
that section for an ISO date **also** returns `[]` — the cross-reference points at a section that
carries no date either.

So the headline GPU sizing numbers (44.5 → 23.8 s/page, "~151 pages/hour per RTX 2000 Ada") and
the encoder numbers that claim the "same date" (19.1 → 14.0 ms/call) have no measurement date
anywhere in the tree. An operator cannot tell whether the GPU envelope was measured on a
driver/torch/vLLM combination still shipped by `deploy/env/`.

Three published pages assert the opposite:

* `PERF-RESULTS.md` opening — "measured on a named machine on a stated date";
* `PERF-RESULTS.md` § Not yet measured — "add the row with its hardware, profile, corpus, date and
  command";
* `README.md:176` — envelopes "each with hardware, stack profile, corpus shape, date and
  reproduction command"; `docs/perf-results.md:8` — "measurement date".

The guard written to enforce this,
`test_every_envelope_states_hardware_profile_corpus_date_and_repro`
(`packages/latence-core/tests/test_perf_results_doc.py:80`), is green only because its assertion is
`re.search(r"\d{4}-\d{2}-\d{2}|ADR-\d{4}", section)` — the alternation lets an ADR cross-reference
stand in for the date, defeating the check its own comment describes ("A date the reader can check,
not 'recently'").

## What to build

1. State the **real** measurement date in the GPU envelope. The evidence is in git: the accel
   ladder landed with the pod run in `a1131b4` / `2a2e588` / `24b0920` / `000e5a5`, all authored
   **2026-07-10** — `a1131b4` ("pod-correct max_new_tokens default 1024→4096 (content-safe) + real
   PERF-RESULTS.md", 2026-07-10 01:26 +0200) is the commit that first published these numbers.
2. Make the encoder envelope's "same date" resolvable by naming the date explicitly.
3. Record the date in the ADR-0036 W8 status note so the cross-reference is true at both ends.
4. Remove the `|ADR-\d{4}` escape hatch from the guard and add a non-vacuous regression test that
   an ADR cross-reference can never substitute for a date.

## Acceptance criteria

* [ ] The `## GPU envelope` section of `PERF-RESULTS.md` contains at least one ISO `YYYY-MM-DD`
      date on its `**Date measured:**` line.
* [ ] Every `**Date measured:**` line in `PERF-RESULTS.md` carries an ISO date on that line (not
      only somewhere in its section).
* [ ] The ADR-0036 W8 status note states the same date.
* [ ] `test_every_envelope_states_hardware_profile_corpus_date_and_repro` no longer accepts
      `ADR-\d{4}` in place of a date.
* [ ] A regression test fails on the pre-fix document and passes on the fixed one, and explicitly
      asserts that stripping ISO dates leaves no ADR token able to satisfy the check.
* [ ] `make test` stays green at or above 3320 passed / 29 skipped.
