# M6 — γ-calibration judgment sample, OFFICIAL MuSiQue-full export

**250 of 60,054 Hyperedges judged** in
`benchmark/official-data/musique-full/calibration-judgments.jsonl`.
Provenance is recorded in `benchmark/official-data/PROVENANCE.md`:
**LLM judgments (Claude Opus, 2026-08-02), pending human spot-check** — a curve fitted from
them inherits whatever the model got wrong. Not committed (a concurrent session owns git on
`hypergraph-m1`).

This is the counterpart to `T21-CALIBRATION-RESULT.md`, whose 226 judgments came out
224/1/1 on the template-planted enterprise corpus and therefore gave isotonic regression
nothing to bend. **This sample is the noisy corpus that result asked for.**

## How the sample was drawn (deterministic, no cherry-picking)

All 60,054 Hyperedges sorted by `(gamma, hyperedge_id)`; every `60054 // 250 = 240`-th row
taken (indices 0, 240 … 59,760). γ spans **0.2500 – 0.9325**.
`make_calibration_sample.py` (new, `benchmark/official-data/`, pyarrow only) joins
`entity_node_ids` → canonical names/types and expands `relations` → head/label/tail triples.

Two structural facts of the export were **verified in `hyperedge.py` before labeling**,
because both would otherwise have produced systematically wrong labels:

1. A span with more than `max_arity` (12) entities and **no** relations explodes into
   *singleton* Hyperedges — `_arity_groups` puts every relation-free entity in its own
   connected component. 110 of the 250 sampled Hyperedges have chunk siblings; 20 sit in a
   group of ≥4. An arity-1 Hyperedge over a 15-name cast table is a deliberate split, **not**
   a recall miss, so `extracted_elsewhere_in_chunk` is carried into the worksheet and
   "materially incomplete" is charged only when an entity central to the span is missing
   from the whole chunk (8 such cases).
2. Redundant relation labels over one asserted pair (`created_by` + `author_of`,
   `located_in` + `part_of`) are **not** charged as errors. A label is charged only when it
   asserts something the span does not.

## Distribution

| judgment | count | share |
|---|---|---|
| correct | 135 | 54.0% |
| partial | 80 | 32.0% |
| incorrect | 35 | 14.0% |

Scoring `correct=1.0 / partial=0.5 / incorrect=0.0`, the sample mean is **0.700**.

## The γ-vs-correctness trend — γ is NOT monotone, and the reason is structural

| γ quintile | n | correct | partial | incorrect | score |
|---|---|---|---|---|---|
| Q1 0.250–0.342 | 50 | 18 (36%) | 23 (46%) | 9 (18%) | 0.590 |
| Q2 0.344–0.456 | 50 | 29 (58%) | 18 (36%) | 3 (6%) | 0.760 |
| Q3 0.458–0.499 | 50 | 42 (84%) | 5 (10%) | 3 (6%) | **0.890** |
| Q4 0.499–0.662 | 50 | 25 (50%) | 17 (34%) | 8 (16%) | 0.670 |
| Q5 0.665–0.932 | 50 | 21 (42%) | 17 (34%) | 12 (24%) | 0.590 |

The curve is an **inverted U with its peak at γ ≈ 0.5**, and that is not noise. γ ≤ 0.5
⟺ `has_relations=False` and γ > 0.5 ⟺ `has_relations=True`, **exactly**, across the whole
export (19,280 relation-bearing Hyperedges; 19,280 with γ > 0.5). `_WEAK_GAMMA_FACTOR = 0.5`
caps every entities-only Hyperedge at 0.5, so γ encodes "does this Hyperedge carry a
relation" in its high-order bit — and on this corpus carrying a relation makes a Hyperedge
*less* likely to be right, not more:

| slice | n | correct | partial | incorrect | score |
|---|---|---|---|---|---|
| entities-only (weak) | 170 | 109 (64%) | 46 (27%) | 15 (9%) | **0.776** |
| relation-bearing | 80 | 26 (32%) | 34 (42%) | 20 (25%) | **0.537** |

**Within each regime γ *is* a usable monotone signal** — which is the finding that matters
for the fit:

| weak-only quintile | n | score | | relation-only quintile | n | score |
|---|---|---|---|---|---|---|
| 0.250–0.306 | 34 | 0.559 | | 0.508–0.613 | 16 | 0.500 |
| 0.308–0.385 | 34 | 0.706 | | 0.618–0.668 | 16 | 0.406 |
| 0.387–0.460 | 34 | 0.779 | | 0.672–0.719 | 16 | 0.531 |
| 0.461–0.495 | 34 | 0.868 | | 0.723–0.780 | 16 | 0.594 |
| 0.496–0.500 | 34 | **0.971** | | 0.786–0.932 | 16 | 0.656 |

Weak Hyperedges climb 0.559 → 0.971 monotonically; relation-bearing ones climb
0.500 → 0.656 with one dip. A **single** isotonic fit over all 250 cannot express this: it
is constrained to be non-decreasing, so it will flatten the entire 0.25–0.93 range to
roughly the pooled mean and throw away both usable signals.

**Ruling for M6 (recommendation, not yet a decision):** fit calibration **per regime** —
one isotonic curve over weak Hyperedges, one over relation-bearing — or equivalently
condition the fit on `has_relations`. Reporting calibrated-vs-uncalibrated as a single
global A/B on this corpus would measure the flattening, not the calibration.

Other structural slices (context, not calibration inputs):

| slice | n | score |
|---|---|---|
| arity 1 | 63 | 0.802 |
| arity 2–4 | 111 | 0.676 |
| arity ≥ 5 | 76 | 0.651 |
| support > 1 | 9 | 0.778 |
| in an arity-split group (≥3 siblings) | 20 | 0.800 |

## What GLiNER2 at threshold 0.5 actually gets wrong on wiki text

Seven patterns account for essentially every non-`correct` label. They are ordered by how
much damage they do to the graph, not by count.

**1. Relation triples asserted from co-occurrence rather than predication (the dominant
error).** 46 of the 80 relation-bearing Hyperedges (58%) carry at least one triple the span
does not assert. It is not spread evenly over the schema — some labels are almost pure
noise:

| label | triples in sample | of which I judged genuinely asserted |
|---|---|---|
| `capital of` | 11 | **1** (`Naypyidaw → Myanmar`) |
| `founded by` | 15 | 3 (and one points at a *date*: `Azarbaijan Shahid Madani University -[founded by]-> 1987`) |
| `spouse of` | 4 | 1 (the others: `Spain → France`, a mother read as a wife, a boxer and his manager) |
| `located in` | 41 | most — this is the one reliable label |

Representative fabrications: `Gulf of Mexico -[capital of]-> United States`,
`Dell -[founded by]-> Polish government` (the span says subsidies *kept* Dell — γ 0.84),
`Mr. Deeds -[directed by]-> Adam Sandler` (he starred; **γ 0.9325, the highest-γ Hyperedge
in the sample**), `Newcastle -[capital of]-> New South Wales`. The high-γ tail is not safe:
Q5 holds the *most* `incorrect` labels of any quintile (12/50).

**2. Title-only spans manufacturing relations.** A span that is just a page title gives the
extractor two co-occurring names and no predicate, and it invents one anyway:
`Walker, Texas Ranger` → four triples at once (Walker `capital of` / `country of` /
`located in` / `member of` Texas); `Billy Hill (band)` → the *disambiguation parenthetical*
becomes an organization and the band's own name becomes a PERSON that is a "member of" it.

**3. Common-noun and anaphoric promotion.** ~30 cases. Mid-prose descriptions become
permanent nodes: `study plans`, `carnivals`, `bodies of water`, `cities`, `census`,
`invading`, `ordained priest`, `public lecture`, `his or her work`, `forthcoming year`,
`double A-side physical single`, and — worst — the pronoun **`He`** as a person node, which
will silently conflate every unresolved male referent it merges with. (Kind-terms that are
the span's own topic — `Aircraft carrier`, `Sicilian pizza`, `Transistorized computers` —
were *not* charged: those are the article subject and MuSiQue asks about them.)

**4. Type errors that mis-link entities onto the wrong node class.** ~29 cases, in a few
recurring shapes: languages and demonyms → LOCATION (`Greek`, `late Latin`, `French`,
`English`, `American`, `Dutch`, `Dominican`, `South Korean`, `Chamorro`); people → other
types (`Lord North` → ORGANIZATION, `Dōkyō` → LOCATION, `Meat Loaf` → ORGANIZATION,
`Phoenicians` → PERSON); places and things → PERSON (`John W. Ferguson House`,
`Jimma Arjo`); works → the wrong slot (`Game of Thrones` → ORGANIZATION, an album → PRODUCT,
`June` — as in June Carter — → DATE). Downstream this is what feeds `country of` a demonym.

**5. Table/infobox spans welding tokens across cell boundaries.** MuSiQue paragraphs carry a
lot of infobox text and the segmenter has no cell notion, so mentions run across it:
**`DJay Heath Ledger`** (a role welded to the next actor — a person who does not exist),
**`Butler Detroit`** (two schools), **`Pub\n\nThe London Gazette`** (a title/body boundary),
`2002 *` (a date with its footnote asterisk), `Sandra` (a name fragment off "Bullock,
Sandra" that becomes a second node beside the sibling `Sandra Bullock`).

**6. Coreference/duplicate nodes.** Title form and body form of the same entity become two
nodes (`Fadel Brahami` / `Mohamed Fadel Brahami`; `Geraldine Ferraro` /
`Geraldine Anne Gerry ''Ferraro`), as do two formats of one date (`October 27, 1986` /
`1986 - 10 - 27`) and a truncated work title beside its full form (`Symphony No.` /
`Symphony No. 1 in C major`).

**7. The span's own subject going unextracted.** 8 cases where the entity the paragraph is
*about* appears in no Hyperedge of that chunk: `Pittsburgh Pipers`, `Kappa Fornacis`,
`Yankee Stadium`, `Southside Speedway`, `Legend High School`, `General Artigas Central
Station`, the song `The Lonely`, the song `Ningyo`/`人魚`. Cheap to probe for: a chunk whose
title-position proper noun is in no member set.

## Files touched

- `benchmark/official-data/musique-full/calibration-judgments.jsonl` (new, 250 lines)
- `benchmark/official-data/musique-full/calibration-sample.jsonl` (new, the worksheet)
- `benchmark/official-data/make_calibration_sample.py` (new)
- `benchmark/official-data/make_calibration_judgments.py` (new — the label table + reasons)
- `benchmark/official-data/analyze_calibration.py` (new)
- `benchmark/official-data/PROVENANCE.md` (new — § Calibration judgments (official))
- `issues/wayfinder-hypergraph-coe/M6-CALIBRATION-OFFICIAL.md` (this file)

Nothing committed; no other file touched; no new dependency (pyarrow only).
