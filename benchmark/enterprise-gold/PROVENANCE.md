# Provenance of the enterprise-gold corpus and gold set

**Read this before citing any number produced against this set.**

## What this is, plainly

The corpus in `corpus/` and the gold set in `gold.jsonl` are **LLM-generated**. They were
written by Claude Opus (Anthropic) on **2026-08-01**, in a single session, on Dennis's
explicit direction. Nothing here was collected from a real enterprise, and nothing here was
labeled by a human.

* **Corpus** — 40 synthetic business documents (memos, minutes, supplier and terminal
  notices, contract excerpts, press releases, data-sheet extracts), mixed English/German,
  written to look like messy internal enterprise text: reference numbers, dates,
  distribution lists, tables, signature blocks, filler sections.
* **Entities** — **all invented**. Every company, person, product, and internal reference
  number is fictional and was chosen to have no real-world counterpart. Place names are
  real cities (used only as company seats). Email addresses and phone numbers are invented
  and use the reserved `.example` domain; they exist so the Redaction Stage has something
  to mask, and they belong to nobody.
* **Gold set** — 100 questions with machine-authored supporting-fact spans, derived
  mechanically from the fact table in `chains.py` by `build_gold.py`.

## What this is NOT

Ticket 08 specifies a **manually labeled** gold set — Dennis (or a delegate) writing the
questions and marking the spans by hand in the ticket-21 labeling UI, over the real
`benchmark/e2e` messy corpus. **This is not that set.** Dennis directed the LLM generation
on 2026-08-01 as a way to unblock ticket 17 without waiting on hours of hand labeling.

Consequences a reviewer must hold on to:

1. **The labels are machine-authored and pending human spot-check.** The mechanical
   guarantees below are real (spans round-trip, chains cross documents), but nobody has yet
   confirmed that each question is *answerable, unambiguous, and answerable only via the
   marked facts*. Spot-check with the resumable labeling UI:

   ```bash
   uv run latence-label gold \
     --export-dir benchmark/enterprise-gold/store/_latence/runs/gold-v0/export \
     --out benchmark/enterprise-gold/gold.jsonl
   ```

   The file is append-only and last-occurrence-wins, so a reworded or re-marked question is
   simply appended; the session resumes with all 100 already loaded.
2. **The corpus is synthetic and self-consistent by construction.** A retrieval or
   hypergraph result measured here is an upper bound relative to real enterprise text: no
   contradictions, no stale duplicates, no OCR noise, no genuinely ambiguous entity names.
   Do not report a number from this set as "enterprise performance" without that caveat.
3. **The generator knew the questions while writing the corpus.** The chains were designed
   first and the documents written around them. That is what makes the multi-hop structure
   genuine — but it also means the corpus is not an independent sample.

## What IS mechanically guaranteed

Enforced by `build_gold.py` (it refuses to write otherwise) and re-checked by
`latence-label validate`:

* every supporting-fact span satisfies `corpus_text[char_start:char_end] == text`, against
  the **exported** chunk text (`masked_content` when the Redaction Stage produced one);
* `len(supporting_facts) == hop_count` for every question;
* for every question with `hop_count >= 2`, the facts sit in **distinct documents** — no
  multi-hop question is answerable from any single document;
* every fact sentence resolves in exactly one document (an ambiguous match is refused);
* question texts are unique (they hash to the `question_id`).

`CHAINS.md` is generated from the same source table, so the honesty artifact and the gold
set cannot drift apart. It lists every planted fact, the document that asserts it, and
every chain — that is the sheet to review against.

## Reproducing

```bash
PYTHONHASHSEED=0 uv run python benchmark/enterprise-gold/run_pipeline.py --fresh
PYTHONHASHSEED=0 uv run python benchmark/enterprise-gold/build_gold.py
uv run latence-label validate \
  --export-dir benchmark/enterprise-gold/store/_latence/runs/gold-v0/export \
  --gold benchmark/enterprise-gold/gold.jsonl
```

The run is deterministic: two `--fresh` runs produced byte-identical hyperedge ids, γ, and
`tau_text` (checked, 2026-08-01). `store/` is gitignored — it is derived in ~30 s.

## Calibration judgments

`judgments.jsonl` (226 rows, one per exported Hyperedge) is the γ-calibration sample for
spec 1.5 / ticket 09, in the ticket-21 harness format
(`{"hyperedge_id", "judgment": correct|incorrect|partial, "note"?}`, append-only,
last-wins).

**These judgments are LLM-generated (Claude Opus, 2026-08-01), not human-labeled.** They
were produced by `make_judgments.py` against the planted-fact ground truth in `CHAINS.md`
and the corpus text: each Hyperedge was judged on whether its extracted entity set and
relation triples are what its own `tau_text` span actually asserts. Four structural checks
run over every row (members occur literally in the span; no other known entity occurs in
the span; relation endpoints are members; triples are distinct and head-before-tail), and
the rows the checks cannot settle are listed explicitly, with reasoning, in the script's
`EXCEPTIONS` table.

**Status: pending human spot-check.** Nothing downstream should treat this file as
human-labeled ground truth until a reviewer has sampled it — the calibration curve fitted
from it inherits whatever the model got wrong. The four non-`correct` rows carry notes and
are the obvious place to start.

```bash
uv run python benchmark/enterprise-gold/make_judgments.py
uv run latence-label validate \
  --export-dir benchmark/enterprise-gold/store/_latence/runs/gold-v0/export \
  --gold benchmark/enterprise-gold/gold.jsonl \
  --hyperedges benchmark/enterprise-gold/store/_latence/runs/gold-v0/export/graph-hyperedges.parquet \
  --calibration benchmark/enterprise-gold/judgments.jsonl
```

## Post-generation corrections (2026-08-01)

Two builder defects found BY this dataset's generation runs were fixed in the framework and the
export regenerated (byte-deterministic pipeline, run_id gold-v0): (1) stream-copy dedup — support
had counted Runner gather copies (×3), now counts distinct sources; (2) relation-triple dedup —
a fact asserted twice in one span had emitted duplicate triples. Two calibration judgments were
re-issued accordingly (see judgments.jsonl notes). Gold spans were unaffected by both fixes;
`latence-label validate` passes over gold + calibration against the regenerated export.
