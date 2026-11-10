# Resolution workbench (prototype)

The first clickable cut for ticket 22: the reviewer's side of "the machine prepares, the expert
decides", plus a view of the graph that resolution produces. It is prototype code, like the
spike next to it, and it has the shape of the package it would become: one React component a
Carbon host mounts in its own router, and a standalone app around the same component.

Synthetic data only. The sample comes from the spike's 12,004-record synthetic estate.

## What is in it

**Review**

- **Queue:** the review band grouped by cluster, the group that settles the most first
  (records touched × how undecided the machine is). Filters: open / decided / all, source
  system, only pairs with a conflicting field.
- **Evidence:** both records side by side with the differing words marked; where the calibrated
  probability sits between the reject and accept bands; what speaks for a match, what against,
  and what is missing; the matcher's weight per field with a one-line explanation computed from
  the numbers (no model call); the group around the pair, drawn, with a warning when the pair
  is the only link between two parts of it.
- **Decision:** one key each — `A` same, `R` different, `U` not sure, `Space` skip, `J`/`K`
  through the queue, `Esc` back. "Different" asks for a reason; changing a recorded decision
  asks for a note; both record with `⌘↵` / `Ctrl+Enter`. The log is append-only, with who and
  when on every entry.
- **The whole group at once:** `M` decides every open pair of the group as the same. `S`
  splits it: each record gets a letter, records with the same letter are one organisation, and
  the open pairs follow (same inside a letter, different across). The split starts from the
  links the machine accepted. A pair you already decided is never replaced, and the dialog
  says how many are left alone or disagree. Pairs the machine accepted that a split separates
  are not in the review queue, so they are recorded with the split as objections.
- **Undo:** a group decision can be taken back from the notice or from any pair it decided.
  The log stays append-only: undo adds a withdrawal entry per pair, and the pair is open again.
- **Numbers:** open, decided, pace, the size of the review band, and a read-only sample of the
  pairs the machine accepted on its own.

**Graph**

- **Schema map:** the concepts with their counts and the relations between them, laid out in
  layers. The schema is declared for the synthetic estate; schema induction (ADR-0073) is
  designed and not built, and the page says so.
- **Node view:** one node in the middle and its neighbours around it by relation. Every
  neighbour opens its own view, so the graph is walked a step at a time and never drawn as a
  tangle. An organisation also shows what its records say, value by value, with the number of
  records behind each.
- **Open links:** organisations the review band has not decided are drawn dashed and link
  straight into the review of the pair. A record in the evidence links back to its organisation.

Not in this cut: AI suggestions, turning a decision into a rule, four-eyes approval, version
certification, reviewer agreement, and acting on the objections a split records against
machine-accepted pairs.

## Run it

```bash
npm ci
npm run dev                 # the standalone app with hot reload
npm test                    # 38 unit and mount tests (Vitest, jsdom)
npm run build               # type check, the library (dist/), the standalone app (dist-standalone/)
npm run build:preview       # one self-contained page: preview/resolution-workbench.html
npm run e2e                 # browser check of the two builds above, with accessibility scans
```

`npm run e2e` uses a Playwright that is already installed on the machine; it is not a
dependency of this package. `.npmrc` turns off automatic peer installation because npm 10
fails while resolving the test runner's optional peers; every peer is listed explicitly.

## Fresh sample data

```bash
cd ../spike
python generate.py --entities 4000 --seed 11 --out data/estate_4k.parquet
PYTHONHASHSEED=0 python run_a.py --dataset estate --path data/estate_4k.parquet --baseline-max 0 \
  --out out/a_12k_export.json \
  --export-review out/review_12k.json --export-groups 40 \
  --export-graph out/graph_12k.json --export-graph-entities 320
```

Copy the two files, minified, to `src/standalone/sample/review-export.json` and
`graph-export.json`. Their shapes are `review-export/1` (`../spike/review_export.py`) and
`graph-export/1` (`../spike/graph_export.py`), mirrored in `src/types.ts`.

## How it is put together

| Path | What it holds |
|---|---|
| `src/index.ts` | everything a host imports |
| `src/ResolutionWorkbench.tsx` | the component; reads its place from the router it is mounted in |
| `src/host.ts` | the seams to the application around it: chrome, controls, translation |
| `src/dataSource.ts` | the two data interfaces and their static implementations |
| `src/ledger.ts`, `queue.ts`, `explain.ts`, `groupActions.ts`, `keys.ts`, `graphLayout.ts` | the logic, free of React and tested on its own |
| `src/components/` | the screens |
| `src/styles/workbench.scss` | all styles, on `--rw-*` variables, classes prefixed `rw-` |
| `src/messages/en.json` | every string |
| `src/standalone/` | the standalone shell, Carbon's styles and the sample data |

Mounting it in a host is described in [INTEGRATION.md](INTEGRATION.md).
