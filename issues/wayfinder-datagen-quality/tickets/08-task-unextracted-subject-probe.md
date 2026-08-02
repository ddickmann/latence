# 08 — Task: unextracted-subject probe in the Quality Report

Type: task
Status: open
Blocked by: 01

## Problem (measured)

Pattern 7, 8/250 cases: the entity a paragraph is *about* appears in **no** Hyperedge of its
chunk (`Pittsburgh Pipers`, `Yankee Stadium`, `Kappa Fornacis`, …). These are recall holes at
the worst place — the span's own subject.

## Task

The cheap probe the audit already designed: flag any chunk whose title-position proper noun
appears in no member set of that chunk's Hyperedges. Ship it as a **Quality-Report section**
(count + exemplars), not a silent fixup — the fix (retry extraction at lower threshold for
the flagged span, or force a mention from the title match) is decided AFTER the probe
measures the miss rate on the full official corpora, per the measured-need rule.

## Outcome

Probe in the Quality Report + tests + the measured full-corpus miss rate recorded here;
a follow-up fix ticket only if the rate justifies it.
