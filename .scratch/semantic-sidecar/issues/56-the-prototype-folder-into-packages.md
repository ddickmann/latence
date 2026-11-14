# Build: the estate prototype's code moves into packages under the repository's gates

Type: task
Mode: AFK
Status: open
Blocked by: the owner's word on the package layout (one package or several), and ticket 34
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
