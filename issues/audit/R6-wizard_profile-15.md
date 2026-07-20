# R6-wizard_profile-15 — the go-live tutorial's §4 "exact enterprise-sota spine" is not the enterprise-sota spine

**Severity:** major
**Status:** FIXED
**Where:** `docs/TUTORIAL-fresh-pod-walkthrough.md` §3 spine sketch + §4 hand-authored stack YAML
**Against:** `packages/latence-core/src/latence_core/setup/wizard.py`
(`BLESSED_REDACT_PROVIDER`, `_sota_graph_features_block`, `_sota_bm25_block`,
`_sota_enterprise_stages`)

## What is wrong

§3 tells the operator "The generated stack is exactly the SOTA stack of §4", and §4 opens "The
wizard above generates this file for you… This is the **exact** enterprise-sota spine". Diffing
§4's YAML against the real `build_stack_config(profile=enterprise-sota)` output showed three
divergences:

1. `redact.provider` was `redaction.gliner_pii`; the wizard emits `redaction.gliner2`
   (`BLESSED_REDACT_PROVIDER`, #197 / ADR-0041 §7a). The config bodies are otherwise identical.
2. the `graph` stage carried only `config: {corpus_id: my-corpus-stack}` — **no `graph_features`
   block** (#193 / ADR-0051, `graph_features.pagerank`).
3. `export_corpus.config` carried `basename`/`context_columns`/`embedder` — **no `bm25` block**
   (#194 / ADR-0049, `tokenizer.regex`).

`grep -n 'graph_features\|bm25\|pagerank\|centrality' docs/TUTORIAL-fresh-pod-walkthrough.md`
returned ZERO hits across the whole 1000+ line go-live tutorial.

## Failure scenario

An operator follows §4 — the documented way to "understand every knob" and tune the blessed
stack — and runs `latence run`. They get:

* **no `bm25-stats.json` / `bm25-postings.parquet` at all**, so the query-side `Bm25Rescorer` has
  no corpus statistics and the hybrid fuser's sparse leg is dead;
* **no `centrality` / `community` columns** on `graph-nodes.parquet` and empty
  `kg_node_centrality` / `kg_node_community` on every exported chunk, so `Expander`'s graph
  bonuses are all `0.0` and the Knapsack packer's `centrality` / `cluster_ids` inputs are absent.

That is precisely the wired-but-dead state #193 and #194 were chartered to close. They
additionally load a second model library (`gliner` / `latence-pii-gliner`) for `redact` instead of
the single unified gliner2 engine #197 blessed.

#197's own regression tests only assert on the generated file header and on ADR-0041; **no test in
the tree referenced this tutorial's stage wiring**, so the full green gate could not see the drift.

## Acceptance criteria

* [x] §4's `redact` stage names `redaction.gliner2` (== `wizard.BLESSED_REDACT_PROVIDER`).
* [x] §4's `graph` stage carries a `graph_features` block equal to the wizard's
      `_sota_graph_features_block()`.
* [x] §4's `export_corpus` stage carries a `bm25` block equal to the wizard's
      `_sota_bm25_block()`.
* [x] §3's ASCII spine sketch and §4's spine comment agree with the emitted stack.
* [x] §1.2's provider note no longer claims §4 wires `redaction.gliner_pii`.
* [x] A **structural** regression test parses the tutorial's §4 YAML block and diffs its stage
      wiring against the live `build_stack_config(profile="enterprise-sota")` output — so any
      future wizard change that the tutorial does not track fails the gate. Vacuous string
      matching is not acceptable.
* [x] `make test` stays green at or above 3320 passed / 29 skipped.
