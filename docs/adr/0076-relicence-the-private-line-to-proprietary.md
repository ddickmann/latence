# Relicence the private line to proprietary terms; the public snapshot stays Apache-2.0

**Status: accepted (2026-11-15).** From this date the repository's development line is
proprietary: `LICENSE` reads "Copyright (c) 2026 The Metis authors. All rights reserved." and
grants no licence without the copyright holders' written permission; every package declares
`license = "LicenseRef-Proprietary"`. **Supersedes [ADR-0063](0063-relicence-to-apache-2-0.md)
for the private line.** It does **not** reach the public snapshot of 2026-11-11, which was
released under Apache-2.0 and stays Apache-2.0. No API, contract, schema version, default or
artifact changes.

## Context

The repository becomes private, is renamed later, and is the ontology builder that runs beside
the private host application, Metis. The owner decided on 2026-11-15 that its terms follow the
host's: "Proprietary, like Metis". Under ADR-0063 the tree carried the Apache-2.0 text, 36
`SPDX-License-Identifier: Apache-2.0` headers, 32 package manifests declaring `Apache-2.0`
with a copy of the text in each, and doc-truth gates (`test_publish_readiness.py`,
`test_continuity_statement.py`) that pin all of it.

One fact constrains the change. Unlike the relicences of ADR-0055 and ADR-0063, which reached no
one, this one follows a publication: the public repository's history begins at an Apache-2.0
snapshot committed on 2026-11-11. A licence granted on that snapshot cannot be withdrawn from
anyone who received it.

## Decision

1. `LICENSE` is the copyright line and a short clause: no licence or other right to use, copy,
   modify, merge, publish, distribute, sublicense or sell is granted except by the written
   permission of the copyright holders; material with its own licence stays under it. The 32
   package copies match it byte for byte.
2. Every package declares `license = "LicenseRef-Proprietary"` with `license-files =
   ["LICENSE"]`, and no `License ::` classifier. Every source header that declared `Apache-2.0`
   declares `LicenseRef-Proprietary`.
3. `NOTICE` keeps the provenance record, adds this change, and says that the 2026-11-11
   snapshot remains available under Apache-2.0. Its third-party section is unchanged:
   schema.org's CC BY-SA 3.0 extract, the model weights and the research corpora keep their own
   terms, which the project's terms never reached.
4. The gates move with the decision rather than being deleted. The canonical-text digest pins
   the new `LICENSE`; the SPDX pin is `LicenseRef-Proprietary`; the NOTICE and `COMMERCIAL.md`
   facts are the new terms, including the snapshot that stays Apache-2.0. Each is as strict as
   before: an edited clause, an added preamble, one manifest or one header left behind fails.

## Alternatives considered

- **Keep Apache-2.0 (ADR-0063).** Rejected by the owner: the engine is now a sidecar to a
  proprietary host, and an outbound grant on it would grant what the host withholds.
- **Business Source Licence or PolyForm Noncommercial, as ADR-0055 tried.** Rejected: a
  source-available grant still licenses third parties, which is what the owner chose against,
  and ADR-0055's record shows the cost of a licence whose permitted uses a reader has to
  interpret.
- **Remove `LICENSE` and the SPDX headers altogether.** Rejected: absence is ambiguous to a
  scanner and to a reader, and the gates would have nothing to pin. An explicit "All rights
  reserved" is the falsifiable statement.
- **Rewrite the history so that the snapshot never carried Apache-2.0.** Rejected: it changes
  nothing in law (a grant made stays made) and it destroys the audit trail every earlier
  relicence relied on.
- **Relicence and rename the packages in one change.** Rejected for scope: the rename to
  `metis-ontology-*` is a separate change, so this one stays reviewable.

## Consequences

- A copy of the 2026-11-11 snapshot may be used under Apache-2.0 by anyone who holds it; code
  added or changed after it may not.
- `CLA.md` keeps the grant that lets the copyright holders license a contribution on other
  terms; it is now the instrument that makes a contribution usable at all.
- The publishing reasoning of ADR-0062 and ADR-0066 (PyPI, signed releases) applies to no
  public audience while the repository is private; nothing is published.
