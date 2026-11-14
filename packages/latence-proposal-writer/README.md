# latence-proposal-writer

The optional adapter that [ADR-0070](https://github.com/ddickmann/latence/blob/main/docs/adr/0070-metis-consumes-the-engine-through-one-proposal-file-and-otherwise-through-unity-catalog.md)
names for the one seam to a host's semantic layer: a reviewable **proposal file**, written from
the approved decisions of an ontology build, and the **receipt** the host's own tool writes when
it has placed that file, read back.

The standalone engine carries none of this; a deployment beside a host installs it. It depends
on `latence-core` only, for the typed errors, and imports no code of any host.

```python
from latence_proposal_writer import in_layer, proposal, read_receipts, schema, to_yaml

document, report = proposal(export, decisions, names_of=("shop.dealers",), decided_on="2026-11-12")
text = to_yaml(document, "vehicles")          # the file's bytes: the same inputs give the same bytes
export, count = in_layer(export, read_receipts((receipt_path,)))
```

- `export` is a build's `graph-export/3` document; `decisions` are the decisions in force, by
  element id (`verdict`, `by`, and the `claim` and the day `at` the decision was made on, a date
  or a timestamp). `decided_on` is the day for decisions that state none; it never replaces a
  stated day.
- **Two kinds of entry**, from approved decisions only: `spellings` (two values of a value set a
  reviewer called the same; the value more rows carry is kept) and `names` (the names of a master
  list's records by their key, only for a list named in `names_of` or marked `"proposal": "names"`
  in the export). Each entry says where it goes
  by a catalog object and a column and carries its origin: the element, its claim, who approved
  it and on which day.
- **Versions** (`VERSIONS`, `SCHEMA_VERSION`): version 1, the default, is the shape above with no
  `version` key. Version 2 (`proposal(..., version=2)`, `schema(2)`) states `version: "2"` and
  adds `links`: a link between two records a person approved, both records by object, key
  column, key and name. Ask for it only once the host's tool reads it; a reader of version 1
  refuses such a file.
- **Never written:** a rejected or undecided element; anything seeded; a link the machine
  accepted and no person approved (one certified on simulated labels is such a link), and in
  version 1 any link between records; a decision made on another statement than the build now makes (reported as
  `stale`); a decision that states no claim, or no day while none was given. The report says, for
  every approved decision not written, why.
- **The file** is YAML in block style with every text in double quotes, so that no reader takes a
  name for a number, a date or a truth value. Its shape is `schema()`
  (`latence_proposal_writer/schema/proposal-file.schema.json`), this side's statement of what the
  host's reader takes; the host's reader is the authority on that shape.
- **The receipt** (`layer-receipt/1`): an approved element it lists as placed, on the claim the
  element still states, carries `in_layer` (the host's library and its release). An element it
  lists as already there or refused carries nothing.

Invalid input (a day that is no ISO date, a receipt of another format, an object that does not
hold the value set it is asked to state spellings on) raises `latence_core.errors.ConfigError`.
