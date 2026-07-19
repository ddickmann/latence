# Threat model

The Latence Framework is a data pipeline that ingests **untrusted enterprise
documents** (from shared drives, object stores, source systems) and emits AI-ready
data — a RAG corpus and a knowledge graph — consumed by downstream AI systems. That
shape is the threat model: hostile bytes in, trustworthy data out, with the LLM/OCR/
embedding machinery in between.

This document is the detailed companion to [`SECURITY.md`](https://github.com/ddickmann/latence-framework/blob/main/SECURITY.md). It maps
the trust boundaries, the mitigations **that exist in the code today** (with file
references so a reader can verify each claim against the source, not the prose), and
the residual risks that are honestly out of scope or deferred to Phase-2 hardening.

Terms are used exactly as `CONTEXT.md` defines them (Stage, Provider, Provenance,
Screening, Quarantine, Evidence, Retraction, Purge).

## Trust boundaries

```
                        TRUSTED                    |            UNTRUSTED
                                                   |
   operator ──authors──> Pipeline YAML             |   source documents (bytes)
                         Provider config           |        │
                         Runner                    |        ▼
                                                   |   ┌──────────────────┐
                                                   |   │ Intake Screening │  (before Parse)
                                                   |   └────────┬─────────┘
                                                   |            ▼
                                        Parse ─ Chunk ─ Content Screening ─ …
                                                   |            │
                          third-party LLM / OCR / embedding endpoints  (semi-trusted,
                          operator-chosen; treated as a network boundary)
                                                   |            ▼
                                            Redaction / Disambiguation / Graph Assembly
                                                   |            │
                                                   |            ▼
                                            Export (inert files) ──> downstream AI
```

**Trusted:** the operator who authors and runs a Pipeline (Pipeline YAML, Provider
configuration, the Runner). The framework does **not** defend against a malicious
operator who deliberately mis-configures a pipeline. This is the same scope
`SECURITY.md` states.

**Untrusted:** the *documents* a Pipeline ingests, and any text embedded in them.
Source bytes are hostile by default.

**Semi-trusted (network boundary):** the third-party model endpoints (OCR, LLM,
embedding) an operator points a Provider at. The framework does not vouch for their
security; it constrains how the pipeline talks to them (env-only secrets, no SSRF via
Source URIs) but the endpoint itself is the operator's choice.

## Boundary 1 — Untrusted input files

**Threats:** malware, zip bombs, file-type spoofing (a `.pdf` that is really an
executable or an HTML injection carrier), oversized/corrupt files that DoS the parser.

**Mitigations (implemented):**

- **Intake Screening before Parse** — `SignatureSizeIntakeScreener` in
  `packages/latence-core/src/latence_core/stages/screening.py`. Over the raw
  `ParserInput` bytes it enforces:
  - an **oversized cap** (`max_bytes`, default 25 MiB);
  - a **file-type spoof check** against magic-byte signatures (`_MAGIC_SIGNATURES`)
    for the structured formats a Parser will try to decode (zip/office/png/jpg/gz);
  - a **PDF structural check** (`_pdf_spoof_reason`, hardening #62): the `%PDF-`
    header must sit at **offset 0** (only a bounded 8-byte BOM/whitespace slack,
    `_PDF_LEADING_SLACK`), followed by a version digit, plus at least one real
    structural marker (`xref`/`trailer`/`startxref`/`%%EOF`/` obj`). This closes the
    old "`%PDF-` substring anywhere in the first 1 KB" hole that let an adversarial
    file smuggle the signature past the check;
  - a **zip-bomb check** inspected via the stdlib `zipfile` **central directory
    without extracting any member** (no decompression, no path traversal): a
    compression ratio cap (`max_zip_ratio`, default 100×) plus an absolute
    uncompressed-size cap (`max_zip_uncompressed`, default 512 MiB).
- **Quarantine, not silent drop** — a dangerous record becomes a `QuarantineRecord`,
  excluded from every downstream Stage and Export but **retained and inspectable**
  with the reason recorded (CONTEXT `Quarantine`).

**Residual:** deeper *structural* validation of complex formats (a well-formed PDF
that is nonetheless a parser exploit) is Phase-2. Screening is a signature/size/ratio
heuristic layer, not a sandboxed parser. Parsing itself should be run with least
privilege by the operator.

## Boundary 2 — Prompt injection in document content

**Threats:** a document embeds instructions aimed at a downstream LLM ("ignore
previous instructions", "system prompt: …"), including an injection **split across
two adjacent chunks** to evade per-chunk scanning.

**Mitigations (implemented):**

- **Content Screening after Chunk** — over each `ChunkRecord`'s text: a
  keyword/regex prompt-injection classifier **plus a cross-chunk-boundary pass** that
  catches an injection deliberately split across two adjacent chunks of the same
  document (`packages/latence-core/src/latence_core/stages/screening.py`).
- **Flag-and-propagate, escalate on strength** — a hit attaches a `RiskMarker` that
  **survives into the corpus**, so a RAG consumer can exclude the chunk, rather than
  silently dropping a document that may legitimately *quote* an injection string. A
  configurable `quarantine_threshold` escalates a sufficiently strong in-chunk hit to
  Quarantine.
- **No eval of model/document text** — see Boundary 5; the framework never executes
  text that flows through it.

**Residual:** the classifier is a deterministic heuristic, not a model — it is a
signal for downstream consumers, not a guarantee that no injection reaches an LLM the
*operator* chooses to feed the corpus into. Marking is advisory; enforcement is the
consumer's responsibility.

## Boundary 3 — The LLM / OCR / embedding endpoints

**Threats:** SSRF (a Source URI coerced into fetching an internal endpoint),
credential leakage to or via a third-party endpoint, an endpoint returning hostile
content.

**Mitigations (implemented):**

- **Storage scheme allowlist** — `DEFAULT_ALLOWED_SCHEMES` in
  `packages/latence-core/src/latence_core/storage.py` permits only cloud-agnostic
  file/object stores (`file`, `memory`, `s3`, `gs`, `gcs`, `abfs`, `az`).
  Network-fetch schemes (`http`, `ftp`, `sftp`, …) are **denied by default**, so a
  Source URI can never be turned into an SSRF vector or an arbitrary remote read.
  Widening is an explicit per-call operator opt-in (`allowed_schemes=[...]`).
- **Env-only secrets** — endpoint Providers read credentials from the environment
  (e.g. `$OPENAI_API_KEY`); secrets are never read from URIs or inlined config values
  as the intended path. Model responses are treated as data (Boundary 5).

**Residual:** the **security of a third-party endpoint an operator chooses to call is
out of scope** (`SECURITY.md`). A hostile OCR/LLM endpoint could return poisoned text;
that content re-enters at Content Screening but is not otherwise trusted.

## Boundary 4 — PII handling and leakage

**Threats:** PII (emails, phones, SSNs, cards, IBANs, secrets/keys) flowing into the
knowledge graph, the RAG corpus, exports, or the shareable Quality Report; secrets
written into the run manifest or logs.

**Mitigations (implemented):**

- **Universal redaction floor** — `redaction_policy.py` provides a deterministic,
  zero-dependency recogniser for the high-harm financial subset (`ssn`,
  `credit_card`, `iban`) that runs under every Redactor, so even a learned Provider
  that missed one still cannot leak it. Redaction produces masked variants; PII spans
  and counts are recorded in the Quality Report.
- **Chunk-level, mask-before-extract** — redaction operates at chunk granularity
  (ADR-0042, fixing the document-truncation leak) and, in `extract_on: masked` mode
  (ADR-0044), extraction reads the **redacted** chunk stream so **PII never reaches
  the KG** at all.
- **Manifest secret scrubbing** — `scrub_sensitive` /` _SENSITIVE_KEY_TOKENS` in
  `packages/latence-core/src/latence_core/runner.py`: any config key whose name looks
  like a secret (`api_key`, `token`, `secret`, `password`, `credential`, …) has its
  **value** replaced with `***REDACTED***` before `manifest.json` is written —
  inlined values and env-var references alike. No secret is written to the manifest or
  logs.
- **Quality Report fingerprinting** — co-occurring entity pairs in the shareable
  Quality Report are **salted-sha256 fingerprinted**, not emitted verbatim
  (`_co_occurrence_fingerprint` in
  `packages/latence-core/src/latence_core/quality_report_builder.py`); the salt makes
  a low-entropy dictionary attack per-deployment.
- **GDPR erasure** — `Purge` (hard deletion for right-to-be-forgotten: physical
  removal of the source, all derived records, and their Evidence, followed by
  reconciliation of the Affected set) and `Retraction` (soft tombstone, retained for
  audit) are first-class dispositions (CONTEXT; delta runner / corpus store).

**Residual:** redaction recall for the *learned* PII categories depends on the
configured Provider; the guaranteed floor is the financial subset only. Operators
handling regulated data should validate recall on their own corpus.

## Boundary 5 — Deserialization and code execution

**Threats:** unsafe deserialization of external data; execution of text controlled by
a document or a model.

**Mitigations (implemented):**

- **No unsafe deserialization** — inter-Stage contracts are Pydantic-validated at
  every Stage boundary; there is **no `pickle`/`eval`/`exec` of external data**
  (verified by grep over `packages/latence-core/src`). Exports are Parquet / JSONL /
  TTL / GraphML written as **inert files**, never a live DB connection.
- **Determinism** — the pipeline is seeded and hash-stable (`PYTHONHASHSEED=0`),
  removing a class of ordering-dependent surprises.

**Residual:** third-party Provider libraries (torch/transformers, OCR stacks) are
outside the framework's control; their deserialization behaviour is the dependency's
responsibility. Dependency-CVE gating is Phase-2 (see below).

## Boundary 6 — Supply chain

**Threats:** a compromised dependency or GitHub Action; a leaked secret committed to
the repo.

**Mitigations (implemented):**

- **Secret-scan gate** — `detect-secrets` runs in CI and `scripts/verify-local.sh`
  against an audited `.secrets.baseline`; a genuinely new secret turns the gate red
  (`scripts/secret_scan_gate.py`). `.env*`, `*.pem`, `*.key`, `.pypirc` are
  git-ignored.
- **Permissive-by-default licenses** — every default model Provider has its weights
  **and** code license verified separately (ADR-0012; recorded in `NOTICE` /
  `THIRD-PARTY-LICENSES.md`). The disclosed exceptions are never defaults: restricted-license
  models (EmbeddingGemma) are opt-in, and the pinned GPU environments under `deploy/env/`
  pull proprietary NVIDIA CUDA wheels.
- **Automation added with this workstream** — Dependabot (`pip` + `github-actions`,
  weekly, grouped minor/patch), CodeQL SAST for Python, OpenSSF Scorecard, and a
  documented CycloneDX SBOM script (`scripts/generate-sbom.sh`, see
  [`docs/SBOM.md`](SBOM.md)).

**Residual:** **dependency-CVE gating is Phase-2** (`SECURITY.md`) — Dependabot
surfaces advisories but there is no hard merge-blocking CVE gate yet. GitHub Actions
are referenced by major tag; Dependabot keeps them current but they are not yet
commit-pinned.

## Out of scope (by design)

Restated from `SECURITY.md` so this document is self-contained:

- A **malicious pipeline author/operator** — trusted by definition.
- **Credentials in URIs** — Providers read secrets from the environment, not from
  URIs or inlined config.
- The **security of a third-party model endpoint** an operator chooses to call.

## Posture summary

This is the **G1-passed, pre-1.0** posture: a documented threat model with the
controls above implemented and adversarially tested (zip bombs, type-spoofed PDFs,
cross-chunk injection, oversized files, secret-laden config) — see
`issues/gate/G1-RESULTS.md`. Performance-at-scale, deeper structural file validation,
and dependency-CVE gating are the tracked Phase-2 hardening items, enumerated rather
than papered over.
