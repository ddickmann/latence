# Build: the estate prototype's code moves into packages under the repository's gates

Type: task
Mode: AFK
Status: resolved (15 Nov)
Blocked by: nothing: the owner decided the layout on 14 Nov (one package); ticket 34's acceptance
is not a blocker for the move (its checks are on what the build does, which the move leaves byte
for byte as it was)
Emerges from: 42 (split out on 14 Nov: the fourth of its four parts)

## Question

The estate build (`.scratch/semantic-sidecar/estate/estate/`, some 25 modules) lives in a prototype
folder no gate of the repository covers. Ticket 43 moved the proposal writer into a package; the
rest (reader, induction, ledger, export, instances, resolution, lifecycle, standards) is to follow.

## Goal

The prototype's modules in one or more workspace packages, typed and linted to the repository's
standard, their tests in `pytest packages`, the prototype folder reduced to its estate files and
handoff. Proposed layout (for the owner): one package `latence-ontology` for the build, its
readers behind an extra for the Databricks client, and `latence-proposal-writer` as it is.

## Done check

- `scripts/verify-local.sh` covers the build; the vehicle estate's export is the same bytes from the
  package as from the prototype.

## Answer (the owner, 14 Nov)

One package, `latence-ontology`, its Databricks readers behind an extra;
`latence-proposal-writer` stays as it is.

## Result (15 Nov)

- **The package** is `packages/latence-ontology` (module `latence_ontology`, command
  `latence-ontology`, also `python -m latence_ontology`): the 25 modules of the prototype moved
  with `git mv` (history kept), the export's JSON Schema and the table of codes as package data
  (`latence_ontology.SCHEMAS`). It depends on `latence-core`, `latence-proposal-writer`, PyYAML
  and DuckDB, and registers no Provider. It is a root dependency of the dev environment, so
  ruff, `scripts/typecheck.sh` and `pytest packages` cover it; `rdflib` and `pyshacl` joined
  the dev group for its tests only.
- **The extra.** The Unity Catalog reader speaks the Databricks REST API over the standard
  library and needs no client, so an extra for it would install nothing; an empty extra is a
  flag that records an intent, and was not written. The one optional client is the hosted
  model's (`hosted-model`: the OpenAI client, imported at first use). DuckDB is a plain
  dependency: the mirror is read through it, and the records file for the linker and the
  seeded source are written through it whichever reader supplied the catalog. The owner's
  intent (a light install for the Databricks path) holds: the Databricks path pulls no client.
- **The gates.** The moved code is held to the repository's standard: line length, strict mypy
  on source and tests. The changes for that are renames of reused loop variables, annotations,
  `zip(..., strict=False)` where lengths were already equal by construction, and the
  `type: ignore` comments replaced by narrowing; the long prompt strings were wrapped with
  their content unchanged (the class prompt file is the same bytes). The tests were renamed
  `test_ontology_*.py` (the workspace's pytest imports a test module by its file name, and
  `test_proposal.py`, `test_host_guard.py` and `conftest.py` already existed in other
  packages); the shared fixtures live in `conftest.py`, the toy estate in `_ontology_toy.py`.
- **The same bytes.** `tests/test_ontology_same_bytes.py` holds two builds to digests recorded
  from the prototype at the commit before the move: the toy estate (runs everywhere), and the
  vehicle estate when `LATENCE_ONTOLOGY_VEHICLE_INPUTS` names a folder with the host's mirror,
  metric views, usage log and verified joins (skipped with that reason otherwise; the mirror is
  the host's and is not in this repository). Both pass here: the vehicle inputs were written
  with the host's own tools at its head (read-only checkout), the build with the seeded source,
  the kept class answer and the usage log. Every file of the build folder (export, Turtle,
  SHACL, PROV-O, ledger, class prompt, seed truth, seeded statements) is the prototype's byte
  for byte. The proposal file from two approved decisions differs in one line only: its
  opening comment names the command that wrote it, now `latence-ontology propose`.
  DuckDB resolved to 2.0.0 in the workspace (the prototype pinned 1.5.4); the bytes are the same.
- **Red first.** All 151 tests are red on the tree before the move (no module
  `latence_ontology`); the two same-bytes tests are new, and their digests are the prototype's.
- **The prototype folder** keeps `estates/`, `README.md` (now the estate files, the commands and
  the modules by their package names) and `HANDOFF.md` (block 3 now `uv sync` and the package's
  tests; every command `uv run latence-ontology`). `requirements.txt`, `estate/`, `tests/` and
  `schemas/` are gone.
- **Counts.** 32 packages in `AGENTS.md` and `docs/index.md`; the changelog names the package.
