# Security Policy

The Latence Framework turns messy, siloed enterprise data into AI-ready data. It
routinely ingests untrusted files (documents from shared drives, object stores,
source systems) and emits data consumed by downstream AI systems. Security is a
first-class property of the design, not an afterthought — but the honest scope
below matters as much as the controls.

## Trust model and honest scope

**Trusted:** the operator who authors and runs a Pipeline. Pipeline YAML,
Provider configuration, and the Runner are trusted inputs authored by a trusted
operator. The framework does **not** defend against a malicious operator who
deliberately mis-configures a pipeline.

**Untrusted:** the *documents* a Pipeline ingests. Source files are treated as
hostile by default and pass through Screening before they can do damage.

**Not in scope (by design):**

- **Credentials in URIs.** Source and Storage URIs must not carry inline
  credentials. Endpoint Providers read secrets from the environment
  (e.g. `$OPENAI_API_KEY`), never from URIs or inlined pipeline config values.
- **A malicious pipeline author.** An operator who inlines an `http://` scheme
  they explicitly allow-list, or who points a Provider at a hostile endpoint, is
  outside the threat model.
- **The security of a third-party model endpoint** an operator chooses to call.

## Built-in controls

These are implemented and covered by tests, not aspirational:

- **Screening at two checkpoints** (see `CONTEXT.md` → *Screening*):
  - *Intake Screening* before Parse — malware signatures, zip bombs, file-type
    spoofing (structural `%PDF-` check at offset 0, not a loose substring),
    oversized/corrupt files. Dangerous records are **Quarantined** (retained and
    inspectable, excluded from every downstream Stage and Export).
  - *Content Screening* after Chunk — prompt injection (including cross-chunk),
    harmful content, sensitivity escalation. Offending records are flagged with
    risk markers that propagate downstream.
- **Redaction** — a comprehensive-by-default PII/secret redactor (email, phone,
  SSN, credit card, API keys / access keys / passwords, and more) produces masked
  variants; PII spans and counts are recorded in the Quality Report.
- **Storage scheme allowlist** — `DEFAULT_ALLOWED_SCHEMES` permits only
  cloud-agnostic file/object stores (`file`, `memory`, `s3`, `gs`, `gcs`,
  `abfs`, `az`). Network-fetch schemes (`http`, `ftp`, `sftp`, …) are **denied by
  default** so a Source URI can never be turned into an SSRF vector or an
  arbitrary remote read. Widening the allowlist is an explicit, per-call operator
  opt-in (`allowed_schemes=[...]`).
- **Secret scrubbing** — the run manifest persists the whole Pipeline verbatim.
  Any config key whose name looks like a secret (`api_key`, `token`, `secret`,
  `password`, `credential`, …) has its **value** replaced with `***REDACTED***`
  before the manifest is written — inlined values and env-var references alike.
  No secret is ever written to `manifest.json` or logs.
- **Quality Report fingerprinting** — co-occurring entity pairs in the shareable
  Quality Report are salted-hash fingerprinted, not emitted verbatim.
- **No unsafe deserialization** — contracts are Pydantic-validated at every Stage
  boundary; no `pickle`/`eval` of external data; exports are Parquet/JSONL/TTL/
  GraphML written as inert files, never a live DB connection.

## Posture

This is the **G1-passed** posture: a documented threat model (malicious files,
prompt injection, PII leakage, secret handling, SSRF via Source URIs) with the
above controls implemented and adversarially tested (pathological inputs: zip
bombs, type-spoofed PDFs, cross-chunk injection, oversized files, secret-laden
config). See `issues/gate/G1-RESULTS.md`. Known residuals and Phase-2 hardening
items are enumerated there rather than papered over.

This is **not yet a 1.0 release.** Performance-at-scale, deeper structural file
validation, and dependency-CVE gating are tracked as Phase-2 hardening.

The full data-pipeline threat model — trust boundaries, per-boundary mitigations,
and honestly-enumerated residual risks — is in
[`docs/THREAT-MODEL.md`](docs/THREAT-MODEL.md).

## Supported versions

The project is **pre-1.0** and ships from `main`. Only the latest released version
(and `main`) receives security fixes; there are no maintained release branches yet.

| Version        | Supported          |
| -------------- | ------------------ |
| `main` (HEAD)  | :white_check_mark: |
| latest release | :white_check_mark: |
| older releases | :x:                |

A supported-versions table with real branches will replace this once a 1.0 line
exists.

## Reporting a vulnerability

Please report security issues **privately** — do not open a public GitHub issue
for a suspected vulnerability.

- Preferred: GitHub **Security Advisories** — "Report a vulnerability" on the
  repository's *Security* tab (creates a private advisory).
- Or email the maintainer: **ddickmann81@gmail.com** with subject
  `SECURITY: latence-framework`.

Please include a description, reproduction steps or a proof-of-concept, and the
affected version/commit.

**Response targets** (best-effort for a personally-maintained pre-1.0 project):

- **Acknowledgement:** within 3 business days of the report.
- **Triage / initial assessment:** within 10 business days (severity, affected
  versions, whether it is in scope per the trust model above).
- **Fix or mitigation:** severity-dependent; critical issues prioritized. You will
  get progress updates until it is resolved or closed as out-of-scope.
- **Disclosure:** coordinated. The default is a public advisory once a fix ships;
  please give a reasonable embargo window (target 90 days) before any public
  disclosure, and credit is gladly given unless you prefer to remain anonymous.

Please do **not** publicly disclose, and do **not** open a public issue or PR that
reveals the vulnerability, until an advisory is published.
