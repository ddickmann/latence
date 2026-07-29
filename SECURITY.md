# Security Policy

The Latence Framework turns messy, siloed enterprise data into AI-ready data. It routinely
ingests **untrusted files** — documents pulled from shared drives, object stores and source
systems — and emits data consumed by downstream AI systems. Security is a first-class
property of the design, not an afterthought. The honest scope below matters as much as the
controls.

## Reporting a vulnerability

Report security issues **privately**. Do not open a public GitHub issue, pull request, or
discussion for a suspected vulnerability, and do not include a working exploit in any public
place.

- **Preferred:** GitHub **Security Advisories** — "Report a vulnerability" on the
  repository's *Security* tab. This opens a private advisory only the maintainer can see.
- **Or email:** **ddickmann81@gmail.com**, subject `SECURITY: latence-framework`.

Please include a description, reproduction steps or a proof-of-concept, and the affected
version or commit. If the report involves real personal data, describe the *shape* of the
data rather than pasting it.

**Response targets** — best-effort for a personally maintained pre-1.0 project, and stated
as targets rather than an SLA:

- **Acknowledgement:** within 3 business days of the report.
- **Triage / initial assessment:** within 10 business days — severity, affected versions,
  and whether it falls inside the trust model below.
- **Fix or mitigation:** severity-dependent; critical issues are prioritized. You will get
  progress updates until it is resolved or explicitly closed as out of scope.
- **Disclosure:** coordinated. The default is a public advisory once a fix ships. Please
  allow a reasonable embargo window — 90 days is the target — before any public disclosure.
  Credit is gladly given unless you prefer to remain anonymous.

Please do not publicly disclose until an advisory is published.

## Supported versions

The project is **pre-1.0** and ships from `main`. Only the latest released version and
`main` receive security fixes; there are no maintained release branches yet, and no
backports.

| Version        | Supported          |
| -------------- | ------------------ |
| `main` (HEAD)  | :white_check_mark: |
| latest release | :white_check_mark: |
| older releases | :x:                |

A supported-versions table with real branches will replace this once a 1.0 line exists.
Continuity, maintainership and what happens if the project stops being maintained are set
out in [`docs/CONTINUITY.md`](docs/CONTINUITY.md).

## Trust model and honest scope

**Trusted:** the operator who authors and runs a Pipeline. Pipeline YAML, Provider
configuration, and the Runner are trusted inputs authored by a trusted operator. The
framework does **not** defend against a malicious operator who deliberately mis-configures a
pipeline.

**Untrusted:** the *documents* a Pipeline ingests. Source files are treated as hostile by
default and pass through Screening before they can do damage.

**Not in scope (by design):**

- **Credentials in URIs.** Source and Storage URIs must not carry inline credentials.
  Endpoint Providers read secrets from the environment (for example `$OPENAI_API_KEY`),
  never from URIs or inlined pipeline config values.
- **A malicious pipeline author.** An operator who explicitly allow-lists an `http://`
  scheme, or points a Provider at a hostile endpoint, is outside the threat model.
- **The security of a third-party model endpoint** an operator chooses to call.

## Processing untrusted documents

This is the property that shapes most of the rest of this page. Every document entering the
pipeline is a file someone else produced, and parsers are historically the softest target in
any document system: a decompression bomb, a container declaring one format while carrying
another, a malformed structure that drives a decoder into pathological work.

**Intake Screening runs before Parse**, over the raw bytes, and is where a hostile file is
stopped:

- **File-type spoofing** — the declared extension is checked against the leading magic
  bytes. PDFs get a stricter, structural check (`%PDF-` at offset 0, not a loose substring
  match anywhere in the file), because "contains the string `%PDF-`" is trivially forged.
- **Zip bombs** — ZIP-family containers (including the OpenXML office formats, which are
  ZIP containers) are inspected through the standard-library `zipfile` **central
  directory**, so the declared uncompressed size and compression ratio are read **without
  decompressing any member**. No extraction means no decompression bomb and no path
  traversal during the check.
- **Oversized and structurally corrupt files.**

A document that fails Intake Screening is **Quarantined**: converted into a
`QuarantineRecord` with the reason recorded, removed from the pipeline, and never handed to
a Parser. It is retained and inspectable rather than silently dropped.

**Content Screening runs after Chunk**, over chunk text, and covers prompt injection —
including an injection deliberately split **across a chunk boundary**, which per-chunk
scanning alone would miss — plus a harmful-content keyword set. A hit **flags** the chunk
with a `RiskMarker` that survives into the exported corpus, so a RAG consumer can exclude
it; a sufficiently strong in-chunk hit escalates to Quarantine at a configurable threshold.
Flagging rather than dropping is deliberate: a legitimate document may quote an injection
string, and deleting it silently would be the worse failure.

**Scope, stated plainly.** The reference Intake Screener performs magic-byte, size and
zip-bomb heuristics. It is **not** an antivirus and does no malware-signature scanning; if
you need that, run one ahead of ingest or supply a Screening Provider that wraps one — the
Screening Capability exists precisely so you can. Deeper structural file validation is
tracked as Phase-2 hardening. Neither reference Screener uses the network, downloads a
model, or deserializes untrusted data.

**If you are contributing a Parser**, treat every input as an attacker-controlled byte
string. A parser change lands with an adversarial test — the pathological input itself, not
a well-formed sample.

## PII handling posture

The pipeline's whole purpose is to move enterprise content into a form an AI system reads,
which means PII exposure is the highest-consequence failure mode in the design. Two
properties are load-bearing.

### The universal financial-PII floor

Redaction is pluggable — a rule-based in-core Redactor, GLiNER-family learned Redactors, and
Presidio are all Providers of the same `PIIDetector` Capability. That pluggability created a
real gap: a learned Redactor's coverage is only as good as its checkpoint, and an operator's
configured label set or policy could narrow it further, so a credit card, IBAN or SSN that
the model missed would flow straight into the exported corpus.

So there is a **floor** (ADR-0044). A deterministic, zero-dependency recogniser for the
high-harm financial subset — `ssn` (including the digits-only form), `credit_card` and
`iban` — runs at the single shared seam that **every** Redaction Provider funnels through,
and its matches are unioned into whatever the Provider detected before masking.

What that guarantees, precisely:

- **It cannot be bypassed by choosing a different backend.** A Redaction Provider that
  detects nothing at all still emits a `masked_content` with the IBAN masked. This is proven
  by a fake Redactor that detects nothing, driven through each shipped Provider's path.
- **It cannot be narrowed away by config.** The floor scans independently of the configured
  label set, the policy type list, and any induced schema.
- **It cannot be forgotten by the next Provider.** It lives at the choke point, not in each
  detection half, so a future Provider inherits it without doing anything.
- **The floor's regexes are pinned byte-identical** to the in-core rule Redactor's built-ins
  by a drift-guard test, so the two can never diverge.

What it does **not** claim, stated as honestly as the guarantee:

- It guarantees **detection**, not a fixed disposition. A floor match is handled with the
  chunk's resolved policy action — mask, replace, hash or tag — so operator intent is
  preserved and the outcome is always visible.
- It runs only when redaction runs on that chunk. An explicit `skip` in policy, or a
  near-empty chunk, is unchanged.
- It is a floor for the **financial** subset. It is not a claim that all PII is always
  masked.

A parallel **guardrail safety floor** unions `prompt injection` and `harmful content` onto
the label set a guardrails-capable Redaction Provider is always asked about, so those two
categories can never be subsetted away per chunk. That one is a *label* floor — the model is
always asked — not a detection guarantee; the model can still miss a hit, and the page says
so rather than implying otherwise.

### Where PII can and cannot travel

- **Extraction reads unmasked text by default**, so the knowledge graph keeps real entities,
  while the exported corpus carries the masked variant. Enterprises needing the stricter
  posture can set `extract_on: masked`, which feeds extraction the redacted chunk stream so
  PII never reaches extraction or the graph. That mode is structurally validated: an
  extraction Stage that does not transitively depend on a redaction Stage is a loud
  configuration error, never a silent fallback to unmasked text.
- **PII spans and counts** are recorded in the Quality Report; co-occurring entity pairs in
  the shareable report are salted-hash fingerprinted, not emitted verbatim.
- **Errors must not leak.** The provider-conformance gate asserts that an adversarial input
  produces a *typed* error carrying no PII, and that a non-Redactor Provider does not surface
  raw PII in non-content fields.
- **Do not put real PII in fixtures, test data, or a bug report.** Synthesize it.

## Other built-in controls

These are implemented and covered by tests, not aspirational:

- **Storage scheme allowlist** — `DEFAULT_ALLOWED_SCHEMES` permits only cloud-agnostic file
  and object stores (`file`, `memory`, `s3`, `gs`, `gcs`, `abfs`, `az`). Network-fetch
  schemes (`http`, `ftp`, `sftp`, …) are **denied by default**, so a Source URI can never be
  turned into an SSRF vector or an arbitrary remote read. Widening the allowlist is an
  explicit, per-call operator opt-in (`allowed_schemes=[...]`).
- **Secret scrubbing** — the run manifest persists the whole Pipeline verbatim, so any config
  key whose name looks like a secret (`api_key`, `token`, `secret`, `password`, `credential`,
  …) has its **value** replaced with `***REDACTED***` before the manifest is written. Inlined
  values and environment-variable references alike. No secret is written to `manifest.json`
  or to logs. Contributors: the repository also runs a `detect-secrets` gate over the tree,
  and `.env*`, `*.pem`, `*.key` and `.pypirc` are git-ignored.
- **No unsafe deserialization** — contracts are Pydantic-validated at every Stage boundary;
  there is no `pickle` or `eval` of external data; exports are Parquet, JSONL, TTL and
  GraphML written as inert files, never a live database connection.

## Posture, and what is not yet true

This is the **G1-passed** posture: a documented threat model (malicious files, prompt
injection, PII leakage, secret handling, SSRF via Source URIs) with the controls above
implemented and adversarially tested against pathological inputs — zip bombs, type-spoofed
PDFs, cross-chunk injection, oversized files, secret-laden config. See
`issues/gate/G1-RESULTS.md`; known residuals are enumerated there rather than papered over.

This is **not yet a 1.0 release.** Performance at scale, deeper structural file validation,
and dependency-CVE gating are tracked as Phase-2 hardening.

The full data-pipeline threat model — trust boundaries, per-boundary mitigations, and
honestly enumerated residual risks — is in
[`docs/THREAT-MODEL.md`](docs/THREAT-MODEL.md). Release artifacts, signing and verification
are described in [`docs/RELEASING.md`](docs/RELEASING.md).
