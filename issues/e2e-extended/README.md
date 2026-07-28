# Extended end-to-end GPU run

**Start here.** Three documents, in reading order.

| # | Document | Answers |
|---|---|---|
| 1 | **[RUNBOOK.md](RUNBOOK.md)** | *How do I actually run this?* Self-contained: state at handoff, environment bring-up, the corpus with exact commands and licences, the staged pod run, known traps, the iterate loop, what to harvest. **A fresh session should read this first and need nothing else.** |
| 2 | [DECISIONS.md](DECISIONS.md) | *Why is it like this?* Every choice with the citation or `file:line` that justified it — including the places where the evidence contradicted the original request. Read before challenging a decision. |
| 3 | [TICKETS.md](TICKETS.md) | *What was built, what is left?* Acceptance bars, file:line evidence, run order. |

## The 30-second version

The **data-processing pipeline is built, merged and green.** The **measurement harness is designed
but deliberately unbuilt** — it needs a real gold graph and a real extracted KG to be worth writing,
and building it against a stand-in is a known, previously-expensive mistake.

So the pod run is **staged**: parse → profile labels → extract → index → *then* build the benchmark
against real data.

## Four things that will bite you if you skip the runbook

1. **Do not export `PYTORCH_JIT=0` or `VLLM_USE_FLASHINFER_SAMPLER=0`.** Both are *Blackwell*
   remedies. On this Ampere card they silently cost throughput on the three heaviest stages.
2. **Do not pipe `verify-local.sh` to `tail`.** A pipeline returns the last command's status, so it
   reports green on a red tree.
3. **The 24 GB card cannot co-host the vLLM OCR server and GLiNER2.** Micro-batch 4 already peaked at
   21.6 GiB on a *32 GB* card.
4. **Do not build the benchmark against the RFC fixture.** That is the frozen-dataset trap.

## Status

`main`, tree clean, **39 commits ahead of `origin/main` — unpushed**. `verify-local` exit 0, mypy
strict on 33 packages, 4,590 tests passing.
