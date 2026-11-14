# Task: measure each older resolution module and keep or retire it

Type: task
Mode: AFK for what runs on a CPU; HITL for the linker path (its model)
Status: open
Blocked by: nothing
Emerges from: 13 (decided by the owner, 14 Nov: "an older module (the disambiguation cascade,
embedding and linker paths, type consolidation) is kept only where a measurement shows it adds
value, and retired otherwise")

## Question

Four older modules resolve on the text path: the string cascade of `disambiguation.cascade`,
the embedding Disambiguator (`latence-disambig-embedding`), the GLinker linker
(`latence-disambig-glinker`) and Type Consolidation. Does each add value over what the new
resolution does?

## Goal

One measurement per module on a gold set it can be scored on, against the same set without the
module, with the decision rule stated before the run: kept if it adds at least the stated gain
at no loss of precision, retired otherwise.

## Done check

- A recorded result per module, with the command that produced it.
- A keep-or-retire line per module, and a ticket for each retirement.
