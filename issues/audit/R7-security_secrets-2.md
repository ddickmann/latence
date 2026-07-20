# R7-security_secrets-2 — an inlined `private_key` / `certificate` / `passphrase` is persisted CLEARTEXT to `manifest.json`

**Severity:** major
**Stream:** feat/release-fleet-integration (#182 SharePoint Source, #30 manifest secret scrubbing)
**Status:** FIXED

## What was reported

> The framework scrubber's substring list misses exactly the two credential key names #182's own
> `_SECRET_CONFIG_KEYS` enumerates for Entra ID certificate auth, and the manifest is written
> BEFORE the SharePoint guard that would reject them can run — falsifying both `auth.py`'s "keeps
> it out of `manifest.json`" and `provider.py`'s "can never be written to the run manifest either".

Confirmed against the tree. Two independent halves, both real.

### Half 1 — the scrubber's blind spot

`packages/latence-core/src/latence_core/runner.py:165`

```python
_SENSITIVE_KEY_TOKENS = ("api_key", "apikey", "token", "secret", "password", "passwd", "credential")
```

matched as a **substring** of the lowercased config key. Of the 13 keys
`latence_source_sharepoint.auth.reject_inline_secrets` refuses, 11 contain one of those tokens.
`certificate` and `private_key` contain none. Neither does `passphrase`, nor `client_cert`.

Microsoft's *recommended* production auth for app-only Graph access is certificate-based, so
`private_key:` / `certificate:` inlined into a SharePoint stage config is the realistic
misconfiguration, not a contrived one.

The existing regression test `test_scrub_sensitive_redacts_nested_secret_keys`
(`packages/latence-core/tests/test_runner.py:194`) exercises only `api_key` / `auth_token` /
`client_secret` / `password` — i.e. only keys already inside the covered substring set. It was
green straight over the blind spot.

### Half 2 — the ordering claim is false

`provider.py:223-227` asserts:

> Enforced BEFORE anything else: a config that inlines a credential never gets as far as opening a
> connection, so it can never be written to the run manifest either.

`auth.py:17-21` makes the matching claim ("which is what keeps it out of `manifest.json`").

But `LocalRunner._run` writes `manifest.json` at `runner.py:896` and constructs the Provider only
at `runner.py:1021` / `:1346`. The guard lives in `SharePointSource.__init__`, so it fires
**after** the manifest bytes are already on the storage backend. For the 11 covered keys the
ordering claim is merely wrong-but-harmless (the scrubber saves it); for `certificate` /
`private_key` it is wrong **and** the scrubber did not save it.

## Failure scenario

An operator configures certificate auth the way Microsoft documents it, inlining the PEM into the
stack YAML. Observed on the branch before the fix:

```
scrub_sensitive({"config": {"private_key": "-----BEGIN PRIVATE KEY-----AAAA-----END PRIVATE KEY-----",
                            "certificate": "CERTBLOB", "passphrase": "hunter2", "client_cert": "CERT2"}})
→ {'config': {'private_key': '-----BEGIN PRIVATE KEY-----AAAA-----END PRIVATE KEY-----',
              'certificate': 'CERTBLOB', 'passphrase': 'hunter2', 'client_cert': 'CERT2'}}
```

All four verbatim. End-to-end against `LocalRunner`, a pipeline whose `source.local_folder` stage
config carried `private_key` produced `_latence/runs/r1/manifest.json` containing
`"private_key": "<the PEM body, verbatim>"` on disk.

Net effect for a SharePoint stage: the run aborts with a `ConfigError` — the guard *does* fire —
**after** the private key has been written cleartext to the storage backend (S3 or local), where
it persists in the run tree, in every backup of it, and in any bug report that attaches a run
directory. The operator who did the exact thing #182 built a guard against is told they were
protected while the credential is already on disk.

## What was built

1. **`_SENSITIVE_KEY_TOKENS` extended** (`runner.py`) with the certificate-auth family:
   `private_key`, `privatekey`, `certificate`, `passphrase`, `keystore`, `keyfile`. These are long
   enough that substring matching carries no false-positive risk.
2. **`_SENSITIVE_KEY_PARTS` added** — `{cert, certs, pem, pfx, p12}`, matched against a whole
   `[^a-z0-9]`-delimited *part* of the key rather than as a substring. This catches `client_cert`
   and `tls_pem` without redacting `certainty_threshold`, which a substring match on `cert` would
   have eaten. Over-redaction is a defect too: the manifest has to keep documenting the run.
3. **The two false guarantees corrected** in `auth.py` and `provider.py`. They now state what is
   actually true: the guard fails the run and stops the connection; `scrub_sensitive` — which runs
   before the manifest is serialised — is what protects the artefact. Neither alone is the
   guarantee, and the ordering was never the mechanism.
4. **The two lists pinned together** by a cross-package test, so adding a key to
   `_SECRET_CONFIG_KEYS` without teaching the core scrubber about it fails loudly rather than
   silently reopening the leak.

## Acceptance criteria

- [x] `scrub_sensitive` redacts `private_key`, `certificate`, `passphrase`, `client_cert`,
      `tls_pem`, `keystore`.
- [x] Every key in `latence_source_sharepoint.auth._SECRET_CONFIG_KEYS` is redacted by
      `latence_core.runner.scrub_sensitive`, asserted by a test that derives the set from the
      real frozenset rather than restating it.
- [x] An end-to-end `LocalRunner` run with an inlined PEM + passphrase writes neither value to
      `manifest.json`.
- [x] No over-redaction: `certainty_threshold` and other innocent keys survive unchanged, so
      off-by-default output stays byte-identical for any pipeline that inlines no credential.
- [x] The falsified "can never be written to the run manifest" / "keeps it out of manifest.json"
      claims are corrected to describe the real mechanism and the real ordering.
- [x] All three new tests fail on the pre-fix scrubber and pass after it.
- [x] `make test` stays green at or above 3320 passed / 29 skipped.

## Regression tests

- `packages/latence-core/tests/test_runner.py::test_scrub_sensitive_redacts_certificate_and_passphrase_keys`
- `packages/latence-core/tests/test_runner.py::test_manifest_never_persists_an_inlined_private_key`
- `packages/latence-source-sharepoint/tests/test_sharepoint_source.py::test_every_rejected_secret_key_is_redacted_by_the_core_manifest_scrubber`

Pre-fix output (scrubber reverted to the original 7-token list, everything else unchanged):

```
FAILED test_runner.py::test_scrub_sensitive_redacts_certificate_and_passphrase_keys
FAILED test_runner.py::test_manifest_never_persists_an_inlined_private_key
  AssertionError: '<the inlined private-key value>' leaked into manifest.json
FAILED test_sharepoint_source.py::test_every_rejected_secret_key_is_redacted_by_the_core_manifest_scrubber
  AssertionError: keys refused by reject_inline_secrets but NOT redacted by the core manifest
  scrubber: ['certificate', 'private_key']
```
