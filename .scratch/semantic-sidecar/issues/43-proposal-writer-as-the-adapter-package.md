# Build: the proposal writer as the optional adapter package the decision record names

Type: task
Mode: AFK
Status: open, after v1
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
