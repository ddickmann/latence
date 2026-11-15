# Build: the proposal writer writes `match`, `broader` and `labels` entries (`layer-proposal/3`)

Type: build
Mode: AFK
Status: built (15 Nov), round trip done on the host's tool at its head of the concepts branch; the vehicle estate's domains proposed and confirmed by the owner (15 Nov)
Blocked by: 57 (the crosswalk names the conformed key a match is written to); the host's 304
(decided and built on its side, 15 Nov)
Emerges from: the host's drafted ticket B-2 (15 Nov); ADR-0070 (the proposal file is the one
seam); 13 (every human decision a link with its actor)

## Question

The host's tool reads three new entries in a file of format `layer-proposal/3` (its 304, the
owner's decisions of 15 Nov: a reviewer approves a cross-domain match and every named domain's
data owner certifies it; a close match is never placed). The writer writes `spellings`,
`names` and `link` today (`layer-proposal/1` and `/2`).

## The shapes (the host's 304, as decided)

```
format: layer-proposal/3
- match: {key: dimension:<ns>.<local>, concepts: [{domain, object, column}, ...], close: false}
  origin: {element, claim, decision: {verdict: approved, by, at}, certified: [{domain, by, at}, ...]}
- broader: {object, column, value: <code>, broader: [<code>...], support: {rows, share}}
- labels: {object, column, language, names: <name>, synonyms: [...]}
```

A close match is written with `close: true` or left out. The host answers with a receipt
(`layer-receipt/1`, `kind` per entry).

## What is asked

`latence-proposal-writer` writes the three entries from approved (and, for a match, certified)
decisions only; the schema and the receipt reader follow; the writer's `SCHEMA_VERSION` is
bumped.

## Done when

Each shape round-trips into a released library version on the host's tool, as ticket 40 did
(where the host's tool cannot run here, the round trip is a live request and the writer is
tested against the shapes).

## Result (15 Nov)

- **The writer** (`latence-proposal-writer`, `VERSIONS = (1, 2, 3)`, `SCHEMA_VERSION = 3`,
  `schema(3)`): a file holding an entry over concepts states `format: layer-proposal/3`; one
  without is version 2 or 1, byte for byte as before (`concepts_too=False` keeps it there).
  - `match`: for an approved value set whose objects lie in two domains or more, or an approved
    equivalence of two value sets across domains, keyed to the host's key the build read from
    the host's alias table (57; `no_key` without one, never derived from a name), one
    `{domain, object, column}` per domain, `close: false` (a close match is never written), and
    `origin.certified` with every named domain's data owner's certification on the claim the
    element states now (`not_certified` otherwise: one owner, no owner, another claim, or someone
    who is not the domain's owner). Written only where the estate names its domains
    (`domains: {<name>: {schemas: [...], data_owner: <id>}}`); `two_columns_in_a_domain` and
    `one_domain` are refused by name.
  - `broader`: for an approved step between two value sets (a roll-up the data proved), one
    entry per member of the finer set: its code, the codes it rolls up to and `support: {rows,
    share}` (the rows carrying those codes and their share of the member's rows).
  - `labels`: a concept's words in one language are an element of their own,
    `labels:<concept id>@<lang>`, decided on their own claim (the concept's claim does not cover
    its words, so a concept's approval writes no labels entry); the build asks for that decision
    and the export carries `labels.claims` and `labels.decisions` per language.
  - Truth values are written as YAML booleans (`close: false`); nothing earlier wrote one.
  - The receipt reader marks what was placed: a value set or an equivalence (`match`), a step
    (`broader`), and a concept's labels per language (`labels.in_layer`).
- **`latence-ontology propose`** passes the estate file's `domains`, the build folder's
  `certifications.jsonl` and `propose.concepts` (default on).
- **Tests, red first** (`tests/test_concepts.py`, 6; red on the tree of 57: `proposal()` takes no
  `domains`, `schema(3)` is refused): the match in the host's shape; held back without every
  certification, on another claim, by someone who is not the owner, without a key, and not
  written without domains; a step's members with their support, and nothing for a rejected or
  stale step; labels only where the words were approved; a host on version 2 gets none; the
  receipt marks all three. The test caught one defect before it landed: an approved value set
  routed to a match also fell into the master list's `names` branch (`no_records`). Two tests
  that pinned the version moved with the bump (`SCHEMA_VERSION == 3`; the first unknown schema
  is now 4).
- **The round trip, on the host's own tool** (its checkout at the head of the concepts branch,
  read-only, its environment): a file this writer wrote from a synthetic export aimed at the
  host's fixture library (the one its own round-trip test uses), with one entry of each shape,
  was placed by `metis-dev propose shop-semantics --apply --release 1.1.0 --receipt ...`:
  `shop-semantics 1.0.0 -> 1.1.0: 3 placed, 0 refused`; the library holds
  `conformed: {status: dimension:retail.status}`, `north_east` rolls up to `north`, `channel` is
  `Vertriebsweg` in German with its synonyms, and it reads as a domain. The receipt read back
  here marks the equivalence, the step and the German labels with `shop-semantics 1.1.0`.
  (The script drove the host's tool from outside; nothing of it is in this repository.)

### Open

- The vehicle estate states no domains and the host has no command that writes its alias table
  (57), so a vehicle build writes no `match` yet. Both are inputs, not code: the domains are the
  owner's to name in `estates/vehicles.yaml`, the alias table a request to the host.

## Decided by the owner (15 Nov): the builder proposes the vehicle estate's domains

The builder proposes them from the estate's schemas, each with a data-owner role (synthetic
roles, no person's name); **proposed, and confirmed by the owner on 15 Nov.**

## Result: the vehicle estate's domains (15 Nov)

- **Proposed** in `estates/vehicles.yaml` (`domains`): `product_safety` (schema `clean`: the
  public safety data, complaints, recalls, investigations, bulletins) with
  `role:product-safety-data-owner`; `after_sales` (schema `internal`: claims, warranty, parts,
  dealers, campaigns) with `role:after-sales-data-owner`; `partner_channel` (schema `partner`,
  the seeded source, so nothing of it is ever written) with `role:partner-channel-data-owner`.
  `marts` is derived from the others and belongs to no domain. A finer split (dealers as a sales
  domain apart from warranty) would need domains by object, which the writer does not read
  today; it reads them by schema.
- **A vehicle build writes a `match`** (`test_a_vehicle_build_writes_a_match_across_two_domains`,
  run here with the host's mirror; skipped without it, as the same-bytes test): with a stand-in
  for the host's alias table in its shape (the make's conformed key, read from a view of each
  domain; the host has no command that writes one yet), the make value set is matched; a
  reviewer's approval and both domains' data-owner roles' certifications make `propose` write
  `match: {key: dimension:conformed.make, concepts: [after_sales ... make, product_safety ...
  make], close: false}` with both certifications in its origin, in a file of format
  `layer-proposal/3`. The test of the domains themselves (every schema in one domain, a role as
  each data owner, the derived layer in none) runs everywhere.

## Result: the seed's service-department word (15 Nov)

The host's client-terms guard lists the English word the seed used as a tail of the seeded
partner accounts' names. The tails are now "Kundendienst" (DE) and "Servicing" (GB), the
word the host's test export already uses (`latence_ontology/seed.py`, `TAIL`); the random
draws are unchanged (the same number of tails per market). Red first:
`test_ontology_seed.py::test_the_seeded_names_use_the_service_words_a_host_shows_its_clients`
failed on the tree before and passes now. No copy of the seed is left in the estate folder.

The vehicle estate rebuilt with the host's inputs (its staging head, `e4b71867`), the seed
11, the record linker (`--resolver ../spike`, `requirements-link.txt`) and the kept class
answer, before and after:

| | before | after |
|---|---|---|
| accounts linked to a dealer | 938 of 1,274 | 963 of 1,274 |
| right against the seed | 916 (0.9765) | 939 (0.9751) |
| share of the accounts that belong to a dealer found | 0.845 | 0.8662 |
| pairs accepted / rejected / for a person | 938 / 38,867 / 304 | 963 / 38,994 / 152 |
| candidate pairs | 40,109 | 40,109 |
| both lines certified (target 0.95, simulated labels, 1,300) | yes | yes |
| review items in the export | 207 | 159 |
| edges | 7,656 | 7,681 |
| concepts / relations / measures / entities | 40 / 89 / 39 / 2,514 | unchanged |

Pinned bytes (the same-bytes test, the build without the record linker): `ontology.json`
`c7ad0afb…10d6` → `aa922914…57a0`; the export as the prototype wrote it `05163c07…525f` →
`b0067ef0…b8f7`. The two exports differ only in 152 seeded account labels whose tail changed
(compared line by line); the Turtle, SHACL and PROV-O files are unchanged. The digests were
re-recorded, no check was loosened, and the before build was checked first to reproduce the
recorded digests. `HANDOFF.md` and the spike's `REPORT.md` state the new numbers with the old
beside them.
