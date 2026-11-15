# Task: retire the string cascade's fuzzy rungs from the text path's default

Type: task
Mode: AFK (the change), with an ADR (a default every stack reads)
Status: resolved (15 Nov): retired from the shipped stacks and the demo (ADR-0075)
Blocked by: nothing
Emerges from: 50 (measured 15 Nov: retire)

## Question

On the text-side gold set (`benchmark/resolution-modules/`), the string cascade
(`disambiguation.cascade`) raises pairwise F1 by 0.019 over exact surfaces and lowers precision
from 0.946 to 0.754: under the owner's rule (14 Nov) it is retired. `stacks/default.yaml`,
`stacks/schema-induction.yaml` and `stacks/linkpred-base.yaml` name it as their Disambiguator.

## Goal

The stacks that name it take `disambiguation.exact_surface` (exact normalised surfaces and
configured aliases: the measured "without") or the new resolution where it serves the text side;
the cascade's Provider stays registered for a Pipeline that names it, marked as measured below
the rule, or is removed: an ADR states which, with the measurement and the alternatives.

## Done check

- The ADR; the stacks changed; their end-to-end validations in `scripts/verify-local.sh` pass.
- The measurement's command and result cited in the ADR.

## Result (15 Nov)

Decided by the owner's rule of 14 Nov (L13) on 50's measurement; recorded in ADR-0075
(`docs/adr/0075-...`), with the alternatives (remove the Provider, refuse it at load, keep it
with aliases) and why each was rejected.

- `stacks/default.yaml`, `stacks/schema-induction.yaml`, `stacks/linkpred-base.yaml` and the
  demo's Pipeline name `disambiguation.exact_surface` (the same `aliases` and KB keys). All three
  stacks pass `latence stack validate` end to end.
- `latence stack check` gives a `retired` WARNING naming the measurement and the replacement for
  a stack that names `disambiguation.cascade` (`latence_core/stacks/retired.py`). The Provider
  stays registered: Pipelines that name it, the bake-off matrix (which measures it) and the
  examples that show it keep working.
- The demo's graph is now 7 nodes and 34 edges (was 6 and 33): "IBM" and "International Business
  Machines" are two nodes, both linked to KB entity Q2, because no fuzzy rung joins them; an
  `aliases` entry would. README, the quickstart, the console guide and the console's end-to-end
  test state the new numbers, read from a run of `latence-demo` and the console. Not restored
  with an alias: that would hide what the retirement changes.
- **Tests, red first:** `test_retired_providers.py` (core, 3) and `test_demo_retired_providers.py`
  (demo, 1): red on the staging head (no module `latence_core.stacks.retired`; once it existed,
  the three stacks and the demo named the cascade, and the check gave no warning); green now.
