# Grilling: the review, certification and versioning lifecycle

Type: grilling
Mode: HITL
Status: open
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
