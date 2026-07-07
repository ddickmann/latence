# H-C1 — Redaction integrity: make "PII-safe by default" actually true

**Source: hostile 360 audit, THEME C (security S1 + correctness). Highest real-world severity.**
The redactor's docstring claims "comprehensive PII+secret coverage … a corpus is PII-handled by
default," but the audit (verified) showed the default config lets **live credentials flow into the
corpus, the KG, and every export completely unredacted.** This is the framework's headline safety
claim and it is currently false. Fix the recall, fix the silent-no-op, make the claim earned or
soften it — and lock it all behind a hard recall benchmark.

All new patterns MUST be linear-time / ReDoS-safe (character classes, possessive/atomic where
needed, no nested or overlapping quantifiers). The audit confirmed the existing patterns are
linear; do not regress that. File: `packages/latence-core/src/latence_core/stages/redaction.py`.

## The confirmed gaps (each needs a fix + a regression fixture)

### 1. Modern secret/token formats are entirely unrecognised (S1)
None of these appear anywhere in the recogniser set; a document containing them yields an **empty**
`pii_spans` and `masked_content == source`:
- GitHub tokens: `ghp_…`, `gho_…`, `ghu_…`, `ghs_…`, `ghr_…`, `github_pat_…`
- Stripe: `sk_live_…`, `sk_test_…`, `rk_live_…`, `pk_live_…`
- OpenAI: `sk-…` (and `sk-proj-…`)
- Google API key: `AIza[0-9A-Za-z_\-]{35}`
- Slack: `xox[baprs]-…`
- GitLab PAT: `glpat-…`
- Private key blocks: `-----BEGIN (RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----`
- AWS **secret** access key (bare 40-char base64) — at minimum in a labelled/`aws_secret_access_key`
  context (bare is high-false-positive; document the boundary honestly rather than over-claim).
- Connection strings embedding a password: `scheme://user:PASSWORD@host` (postgres/mysql/redis/amqp…).
Add a dedicated, named recogniser per family (so the per-type recall metric is meaningful), each
matching the whole secret value as the span. Prefer high-precision prefix-anchored patterns
(low false-positive) over a single greedy "high-entropy blob" heuristic.

### 2. Compound-identifier miss on `api_key`/`token` labels (S2, verified)
`GITHUB_TOKEN=ghp_…`, `SESSION_TOKEN`, `X_AUTH_TOKEN`, `MY_API_KEY` all evade because the
`api_key` recogniser uses a plain `\b…\b` boundary and `_` is a word char, so `\btoken\b` cannot
match inside `GITHUB_TOKEN`. This is the **identical** bug the `secret` recogniser already fixed
(optional leading/trailing identifier components). Apply the same optional-affix construction to
the `api_key`/`token`/`access_token`/`session_token` family so a keyword embedded in a larger
`UPPER_SNAKE` identifier is caught. Regression: `GITHUB_TOKEN=ghp_…` is masked.

### 3. SSN-without-dashes is only caught by accidental `phone` overlap and mislabelled (S3)
`123456789` / `123 45 6789` are masked today only because the loose `phone` pattern swallows them,
tagging them `phone` — so a `types: ["ssn"]` policy or a `detectors` list excluding `phone`
**silently drops SSN redaction**, and even at defaults the per-type recall metric is corrupted
(SSN counted as phone). Add a real digits-only SSN recogniser (contiguous 9 digits with a
plausible-SSN guard: reject `000`/`666`/`9xx` area, `00` group, `0000` serial to cut false
positives) so `types:["ssn"]` genuinely covers it and the label is correct.

### 4. Redaction is silently skippable to a no-op on sensitive documents (S3, the "clean bill" lie)
A policy with `skip:true`, `types:[]`, or a narrow `detectors` list yields a `RedactionRecord`
with an empty span list that is **indistinguishable in the contract** from "genuinely no PII found,"
and the Quality Report's `documents_with_pii: 0` then reads as a clean bill of health rather than a
disabled control. Add a **floor**: when a document's `Classification` sensitivity is
`confidential`/`restricted` (or the configured sensitive set) and the applied policy would run no
detectors, that must surface as a **distinct, first-class signal** — a `redaction_disabled_for_sensitive`
count in the Quality Report and a per-record flag on the `RedactionRecord` — so "control was off"
can never masquerade as "nothing to redact." (Do not silently *force* redaction on — an operator
may intentionally skip — but the disposition must be auditable and loud, not invisible.)

## Definition of Done (Gate bar — this is a security slice)
- Named recognisers for every family in §1, each linear-time (add a ReDoS unit probe over a
  pathological input per new pattern, asserting bounded time — reuse the entity static-guard style).
- §2 compound-identifier fix; §3 real SSN recogniser + correct label; §4 sensitive-doc no-op floor
  with a Quality-Report signal + a `RedactionRecord` flag (contract change → bump the relevant
  SCHEMA_VERSION, keep backward-compatible defaults, update the schema-version test + note).
- **Hard recall benchmark:** extend the #61 recall benchmark into a gated test that asserts a
  minimum recall over a corpus containing EVERY format above (GitHub/Stripe/OpenAI/Google/Slack/
  GitLab/private-key/conn-string/SSN-no-dash/compound-label). The test FAILs if a format regresses.
- **No PII leak** discipline preserved (masked output + report stay counts-only/fingerprinted).
- Docstring/ADR corrected: describe coverage **accurately** (what is and isn't caught, and the
  documented AWS-bare boundary) — no "comprehensive" overclaim beyond what the benchmark proves.
- `mypy --strict` clean over src+tests, `ruff` clean, deterministic seeded tests, CI green.
- The P3-F3 **conformance suite still passes** for `redaction.hybrid_rule` (this must not break C1–C6).

## Reviewer should check
- Each new pattern is genuinely linear-time (no catastrophic backtracking) — proven, not assumed.
- The recall benchmark actually FAILs if a recogniser is reverted (non-tautological, it bites).
- The no-op floor makes a disabled control **loud** in the report; `documents_with_pii:0` can no
  longer hide a `skip:true` on a `confidential` doc.
- False-positive discipline on SSN/AWS-bare is reasonable and documented — recall gains don't drown
  the corpus in mask noise.
- Labels are correct (SSN tagged `ssn`, not `phone`) so the per-type recall metric is honest.
