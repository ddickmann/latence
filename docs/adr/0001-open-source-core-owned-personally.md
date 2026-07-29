# Open-source core, personally owned, published before Deloitte start date

**Superseded in part by [ADR-0055](0055-relicence-to-polyform-noncommercial.md) (2026-07-29): the
publication licence is PolyForm-Noncommercial-1.0.0, not Apache-2.0.** Only the licence named below
is superseded — the decision itself (publish personally, before the start date, as pre-existing IP)
stands, and the fact that it was taken under Apache-2.0 is left on the record deliberately: the
repository was never published under that grant, which is precisely what makes the relicence clean.
See the provenance record at the top of `LICENSE`.

The framework core is published by latence.ai under Apache-2.0 **before** Dennis's employment at Deloitte begins, establishing it as pre-existing, publicly available IP. At Deloitte it is consumed like any third-party OSS tool; client demos and firm accelerators are built *on top of* it, never *into* it. Alternatives considered: rebuilding it as a Deloitte-internal asset (loses ownership and reusability across employers/clients) and keeping the repos private for dual use (highest IP-conflict risk under typical Big-4 IP-assignment clauses). Consequence: every dependency must be license-audited — model *weights* and model *code* separately — and nothing from the private repos may be ported without a history-clean re-implementation or verified-clean copy.
