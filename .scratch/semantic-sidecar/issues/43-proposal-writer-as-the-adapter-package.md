# Build: the proposal writer as the optional adapter package the decision record names

Type: task
Mode: AFK
Status: resolved (14 Nov): built; the package is under the repository's gates
Blocked by: 40 (a reviewer's decision gone through it once)
Emerges from: 40 ("Not done"),
`docs/adr/0070-metis-consumes-the-engine-through-one-proposal-file-and-otherwise-through-unity-catalog.md`

## Question

The decision record names an optional adapter package for the seam to the host. What exists
is a module of the estate prototype (`estate/estate/proposal.py`) with its command, its schema
(`schemas/proposal-file.schema.json`) and ten tests. The prototype lives under `.scratch`,
where no gate of the repository runs: its linter and type findings (468 and 216 on 12 Nov,
nearly all older than the writer) are checked by nobody.

## Goal

The writer, the receipt reader and the schema move into an installable optional package with
the repository's gates on it. The estate build calls it; nothing in the package imports the
host.

## Done check

- The package installs on its own and writes the vehicle estate's file with the same bytes as
  the prototype's command.
- The repository's linter, type check and test lanes cover it and pass.
- The prototype's module is gone or a thin call into the package.

## Result (14 Nov)

- **The package** is `packages/latence-proposal-writer` (module `latence_proposal_writer`): the
  writer (`proposal`, `to_yaml`, the rule for the value kept as `kept_values`), the receipt
  reader (`read_receipts`, `in_layer`) and the file's schema, shipped as package data
  (`schema()`). It depends on `latence-core` only, for `ConfigError` (a `ValueError`, so every
  caller that caught the prototype's `ValueError` still does), and registers no Provider. It is a
  root dependency of the dev environment, so ruff, `scripts/typecheck.sh` and `pytest packages`
  cover it; `jsonschema` and its stubs joined the dev group for its tests only.
- **What no longer depends on the prototype.** The writer used the build's `merge_values` for
  the value kept and the ledger's two lists of kinds; it now carries the rule and the lists
  itself. The prototype's tests check that both lists agree with the ledger, and that the file
  is the same before and after a build applied the decisions (the package's rule is the build's).
- **The prototype calls it.** `estate/estate/proposal.py` is a thin call: it re-exports the
  package's functions and names `python -m estate propose` in the file's opening comment.
  `requirements.txt` installs the core and the package from this checkout.
- **Tests.** Twelve in the package: the nine behaviours of the prototype's ten tests that need
  no build, on a synthetic export written for them (approved pair and list, rejected or
  undecided, seeded record and links, stale, same bytes, value kept, no day or claim, what is
  not carried, the receipt), the refusal of a wrong receipt, the shipped schema, and that the
  package imports only the standard library and the core. Red first: all twelve, on the tree
  without the package (no module). The prototype keeps five: the four that need a build (seeded
  records and the machine's links in a real build, same bytes across a rebuild, the receipt
  through `build --receipt`, the command line) and one new one, red on the old module, that the
  prototype calls the package. 119 prototype tests pass (124 - 6 moved + 1).
- **The done check.**
  - The vehicle estate's file, from a fresh build of the mirror the host's tools write (without
    the record linker, with the trial's decisions: the Kia and Chrysler pairs, the dealers, an
    approved seeded list and a rejected pair), is **byte for byte** the prototype's file of the
    base commit, and the printed report is the same. The same file comes from the package's
    wheel installed on its own beside the core's wheel in an empty environment.
  - The export of a build given a receipt is byte for byte the old one (3 elements carry
    `in_layer`).
  - Gates: see the workstream's report (ruff and strict mypy on the package clean; the
    doc-truth tests pass with the package counted: 31 packages in `AGENTS.md` and
    `docs/index.md`).
