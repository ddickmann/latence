# Design notes

- Every record carries **Provenance** and **Classification**.
- Storage is one fsspec seam: `file://`, `s3://`, `memory://` — no code change.
- The local Runner checkpoints each Stage and resumes without redoing work.
