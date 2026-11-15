# Task: retire the string cascade's fuzzy rungs from the text path's default

Type: task
Mode: AFK (the change), with an ADR (a default every stack reads)
Status: open
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
