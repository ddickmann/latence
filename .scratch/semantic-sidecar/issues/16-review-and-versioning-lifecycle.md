# Grilling: the review, certification and versioning lifecycle

Type: grilling
Mode: HITL
Status: resolved (14 Nov), by the owner's answers below
Decided by the owner (14 Nov): five states (draft, proposed, approved, certified, retired); a named reviewer approves and the data owner certifies; nothing is approved automatically until a calibration proves one class safe, and then only that class; approved changes reach the catalog as staged proposals its owner applies, never as direct writes; when the builder runs beside the host, review happens on the host's Ontology page and routes, standalone on its own command line and workbench; drift (a rename, split, retype or drop) becomes a new proposal with the old entry marked stale, and a column that turns sensitive is erased from every export and ledger, recorded as an erasure. The builds follow as their own tickets.
Blocked by: 11, 13, 22

## Question

Decide the lifecycle:

- the states: draft, proposed, approved, certified, retired;
- who approves, and which classes may be auto-approved under calibration (as Metis does for
  descriptions, synonyms, value forms and code lists);
- digest-pinned versions;
- how a human decision is stored so it survives rebuilds;
- drift handling (rename, split, retype, drop);
- UC write-back as staged proposals or pull requests versus direct writes;
- whether the review surface is standalone (CLI, console, app) or Metis's review routes when
  running as a sidecar;
- erasure when a column turns sensitive.
