# Enterprise readiness of the Latence repository (review of 2026-11-15)

An independent read of the repository at `e746607` against the requirements an enterprise
buyer or a security reviewer of a public repository would hold it to. One row per requirement.
**Verdict** is the state found; **Fix** names the change that closes it on this branch, or
the decision it waits on. Sizes: S (under an hour), M (a day), L (more).

Counts as found: **6 red, 13 amber, 28 green** over 47 requirements. After the fixes on this
branch: **1 red** (R5, waiting on decision D-2), **4 amber** (D5 waiting on decision D-1; C2
accepted with reasons until the GPU live request L-1; C8 waiting on live request L-2; H7 left
to the map's owner), the rest green.

| Area | Red | Amber | Green |
|---|---|---|---|
| Security | 3 | 0 | 5 |
| Data protection | 0 | 2 | 3 |
| Reliability and contracts | 1 | 1 | 3 |
| Operability | 0 | 0 | 3 |
| Maintainability | 0 | 2 | 4 |
| Supply chain | 2 | 3 | 3 |
| Licence and notices | 0 | 2 | 3 |
| Documentation and hygiene | 0 | 3 | 4 |

## How it was read

- `bash scripts/verify-local.sh` at the start (exit code captured by redirect).
- `pip-audit` over `uv export --frozen --all-packages` of both lock files.
- `vulture --min-confidence 80` over every `packages/*/src`; `wc -l` over every source module.
- A scan of every tracked file for e-mail addresses and their domains.
- `uv lock --check` in the workspace and in `packages/latence-gliner25`.
- Reading: `SECURITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `README.md`, `NOTICE`,
  `THIRD-PARTY-LICENSES.md`, `.github/workflows/*`, `scripts/verify-local.sh`, and the
  `latence-ontology` and `latence-proposal-writer` sources that write to a host.

## Security

| # | Requirement | Verdict | Evidence | Gap | Smallest fix | Size | Fix |
|---|---|---|---|---|---|---|---|
| S1 | Credentials travel only over TLS | **red** | `reader_databricks.py:131` exchanged the service principal's secret before `:96` checked the scheme; `DATABRICKS_HOST=http://…` sent it in the clear | The check came after the use | Check the host before any request | S | `2774169` |
| S2 | A loopback exemption is exact | **red** | `reader_databricks.py:96` `host.startswith("http://127.0.0.1")` admits `http://127.0.0.1.attacker.invalid` | A prefix test on a URL | Parse the URL; compare the host name | S | `2774169` |
| S3 | A bearer token is never replayed to another host | **red** | `reader_databricks.py:69` used `urllib.request.urlopen`, whose redirect handler copies `Authorization` to the redirect target; a socket test showed the token arrive at the second server | The core's lineage transport and the SharePoint source refuse redirects; this reader did not | A no-redirect opener | S | `2774169` |
| S4 | Secret scanning in CI and locally | green | `scripts/secret_scan_gate.py`, CI job `secret-scan`, pre-commit hook | — | — | — | — |
| S5 | Static analysis and scorecard | green | `.github/workflows/codeql.yml`, `scorecard.yml` | — | — | — | — |
| S6 | Actions pinned by commit | green | every `uses:` in `.github/workflows/*.yml` carries a SHA | — | — | — | — |
| S7 | A private route to report a vulnerability | green | `SECURITY.md:9-35` (advisories, mail, response times) | — | — | — | — |
| S8 | A plaintext token to a lineage collector | green | `observability.py:744-758` refuses it unless loopback | — | — | — | — |

## Data protection

| # | Requirement | Verdict | Evidence | Gap | Smallest fix | Size | Fix |
|---|---|---|---|---|---|---|---|
| D1 | Synthetic data carries no real person's or company's address | **amber** | 57 addresses at registrable domains in tests and a script: `alice@acme.com` ×50, `bob@corp.com`, `a@b.com`, `c@d.com`, `alice@secret.com`, `ops@example.gov` | Real domains in records called synthetic | Reserved names (RFC 2606/6761) and a gate over the tracked tree | S | `79600fc` |
| D2 | No value of a sensitive column leaves the warehouse | green | `instances.py:19,86,221`; `test_a_column_tagged_sensitive_is_in_no_statement` | — | — | — | — |
| D3 | Erasure when a column turns sensitive | green | `erasure.py`, `cli.py:176-193` | — | — | — | — |
| D4 | PII floor beneath every redaction Provider | green | `redaction_policy.py` (`finalize_redacted_chunk`), its tests | — | — | — | — |
| D5 | Host-private pointers stay private | **amber** | `.scratch/` is tracked (233 files, 4.1 MB); the map's *Reference reads* names private paths of the host and says they "live only here, never in shipped docs" — the repository is public | A planning folder written for a private repository is public | Owner decision (below) | S | waits on D-1 |

## Reliability and contracts

| # | Requirement | Verdict | Evidence | Gap | Smallest fix | Size | Fix |
|---|---|---|---|---|---|---|---|
| R1 | The export is checked against its contract before it is written | **amber** | `graph-export-3.schema.json` ships as package data, but only tests validate an export against it; `cli.py:447` wrote without a check | A host reads the file by a schema nothing enforced at write | Validate, refuse before any file is written | S | `1100b3c` |
| R2 | Contract versions are explicit | green | `export.py:25` `graph-export/3`; `writer.py:89-92` `layer-proposal/1-3`; core `SCHEMA_VERSION` | — | — | — | — |
| R3 | Typed errors, no silent fallback | green | `errors.py` taxonomy; `WarehouseError` in the reader | — | — | — | — |
| R4 | Timeouts on every remote call | green | `reader_databricks.py` `timeout=60`, `give_up_after=900` | — | — | — | — |
| R5 | The proposal file (the host seam) conforms to its published schema | **red** | A runtime guard on `latence_proposal_writer.to_yaml` refuses the toy hub build: a `broader` entry's `support.share` is 1.0769 against `maximum: 1` (`proposal-file-3.schema.json`). `writer.py:556` divides edge rows (counted over every object carrying both columns, personal ones included, `structure.py:51,437`) by the member's node rows (personal objects excluded, `instances.py:179-193`) | Two populations in one ratio; the file can break its own schema | Owner decision (below); the guard is written and lands with the fix | M | waits on D-2 |

## Operability

| # | Requirement | Verdict | Evidence | Gap | Smallest fix | Size | Fix |
|---|---|---|---|---|---|---|---|
| O1 | Structured logging in library code, no `print` | green | 27 modules use `logging.getLogger`; the one `print` in `src` is a corpus-builder script | — | — | — | — |
| O2 | Lineage and tracing | green | `observability.py` (OpenLineage, OTel) | — | — | — | — |
| O3 | Readable failure for a missing environment | green | `test_a_build_from_the_catalog_says_what_is_missing_in_the_environment` | — | — | — | — |

## Maintainability

| # | Requirement | Verdict | Evidence | Gap | Smallest fix | Size | Fix |
|---|---|---|---|---|---|---|---|
| M1 | Strict typing, source and tests | green | `scripts/typecheck.sh`, `mypy --strict` per package, in CI and verify-local | — | — | — | — |
| M2 | Lint | green | `ruff check packages benchmark/sota-campaign scripts` | — | — | — | — |
| M3 | Module size | **amber** | 15 modules past 1,000 lines (`contracts.py` 2,570, `perf.py` 2,139, `runner.py` 2,057, `cli.py` 1,836 …); nothing stopped growth | No ceiling | A ratchet: 1,000 lines, the fifteen held at their size; splitting them is follow-up, one per change | S (gate) / L (splits) | `7a9af4b` |
| M4 | Dead code | green | `vulture --min-confidence 80`: 7 hits, all false positives (protocol parameters, `TYPE_CHECKING` imports used in strings) | — | — | — | — |
| M5 | Test coverage measured and held | **amber** | CI measures line coverage (`ci.yml` `--cov`) but sets no floor; `verify-local.sh` does not measure it | A coverage drop passes every gate | A floor at the measured value | S | see the report's last section |
| M6 | Tests that were red first | green | `CONTRIBUTING.md` *The test-first rule*; every fix on this branch | — | — | — | — |

## Supply chain

| # | Requirement | Verdict | Evidence | Gap | Smallest fix | Size | Fix |
|---|---|---|---|---|---|---|---|
| C1 | No known advisory in the locked set | **red** | `pip-audit`: 142 unaccepted advisories over both locks, among them Pillow ×42 and pypdf ×34 (pypdf opens untrusted PDFs), urllib3, cryptography, filelock, anyio, pyjwt, sqlparse, soupsieve, Airflow | No gate looked at advisories | A gate with an audited accepted list; refresh the locks | M | `5ab6f05` |
| C2 | Advisories that cannot be fixed by a refresh are accepted with a reason | **amber** | torch 2.12 (`PYSEC-2025-194`), transformers 4.57 (6 advisories, fixes in 5.x; `latence-disambig-glinker` caps `<5`), setuptools 81 (held by torch 2.12's cap) | Upgrading changes measured GPU results | `scripts/vuln-accepted.toml` with reasons; the upgrade is a GPU live request | S / M on a GPU | `5ab6f05`; live request L-1 |
| C3 | Lock files current, and CI installs from them | **red** | `packages/latence-gliner25/uv.lock` failed `uv lock --check`; CI and verify-local ran `uv sync` without `--locked`, so a stale lock was re-resolved silently | Nothing checked either | `uv lock --check` on both; `uv sync --locked` | S | `5ab6f05` |
| C4 | Tools fetched by the gates are pinned | **amber** | `uv run --with detect-secrets` (CI, verify-local, pre-commit), `--with sigstore` and `uvx sigstore` (release signing and its verification), `uv pip install pytest-cov`, `s3fs` — all unpinned | The same commit could gate differently on two days | Exact versions; a gate | S | `65416f3` |
| C5 | Dependency declarations are complete | green | `scripts/dep_audit.py`: no genuine gap | — | — | — | — |
| C6 | Dependabot | green | `.github/dependabot.yml` (pip weekly, actions weekly) | — | — | — | — |
| C7 | SBOM and signed releases | green | `scripts/generate-sbom.sh`, `scripts/release-integrity.sh`, `release.yml` | — | — | — | — |
| C8 | The Airflow lane's ad-hoc installs | **amber** | `ci-heavy.yml:166-186` installs pytest, pydantic, pyarrow, fsspec and typer with `python -m pip`, pydantic and three others without Airflow's constraint file | Not reproducible; cannot be run here | Install them under `--constraint "${CONSTRAINTS}"` and run that lane once | S + a CI run | not changed: live request L-2 |

## Licence and notices

| # | Requirement | Verdict | Evidence | Gap | Smallest fix | Size | Fix |
|---|---|---|---|---|---|---|---|
| L1 | One outbound licence, canonical text | green | `LICENSE` is the Apache-2.0 text byte for byte; ADR-0063; `test_publish_readiness.py` | — | — | — | — |
| L2 | Third-party material in the tree is attributed in NOTICE | **amber** | The schema.org extract (CC BY-SA 3.0) is tracked at `.scratch/semantic-sidecar/estate/estates/vehicles.vocabulary.schemaorg.json` and listed in `THIRD-PARTY-LICENSES.md:450`, but `NOTICE`, the file every redistributor carries, did not name it | Attribution only inside the file | A NOTICE section; a gate over tracked vocabulary files | S | `17d0bb0` |
| L3 | Provenance statements are true for a public repository | **amber** | `NOTICE:17` and `THIRD-PARTY-LICENSES.md:7` said the repository "has been private since creation"; the repository is public, and its history begins at a snapshot of 2026-11-11 (`445e634`) | A stale fact in a licence notice | State what holds: private development repository, public history from an Apache-2.0 snapshot | S | `17d0bb0` |
| L4 | Third-party noncommercial terms kept separate | green | `THIRD-PARTY-LICENSES.md` research corpora behind opt-in flags | — | — | — | — |
| L5 | Weight licences verified per checkpoint | green | `NOTICE:38-83`, `ProviderProfile.license`, conformance C3 | — | — | — | — |

## Documentation and repository hygiene

| # | Requirement | Verdict | Evidence | Gap | Smallest fix | Size | Fix |
|---|---|---|---|---|---|---|---|
| H1 | The front page says what is built | **amber** | `README.md` *Where this is going*: "Nothing in this section is built", while `latence-ontology` and `latence-proposal-writer` ship the ledger, the Unity Catalog reader and the W3C exports; no front-door page named either package, `latence-benchmark` or `latence-label-profiler` | An under-claim that hides the v1 engine | Say what is built and what is not; a gate that every package is named on a front-door page | S | `3507d24` |
| H2 | `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SUPPORT.md`, `CITATION.cff`, `CODEOWNERS`, issue templates | green | present and linked; doc-truth gates | — | — | — | — |
| H3 | A changelog | green | `CHANGELOG.md`, Keep a Changelog; this branch adds a *Security* section | — | — | — | — |
| H4 | Onboarding in one command | green | `README.md` *Quick start*: `uv sync`, `uv run latence-demo` | — | — | — | — |
| H5 | Documentation claims are machine-checked | green | `test_front_door_truthfulness.py`, `test_continuity_statement.py`, `test_publish_readiness.py` | — | — | — | — |
| H6 | `verify-local.sh` mirrors CI | **amber** | It ran neither an advisory audit nor a lock check | — | Added with C1 and C3 | S | `5ab6f05` |
| H7 | `ROADMAP.md` reflects the re-scope | **amber** | It names neither ontology package; the README now points at their READMEs | The roadmap's built/not-built list predates them | Bring the roadmap's list up to date with the map's route | S | not changed: the map's owner keeps the route |

## Decisions for the owner

- **D-1 (D5): the public `.scratch/` folder.**
  - (a) Keep it as it is.
  - (b) Move the planning (map, tickets, research) to the private host, and keep `estate/` and
    `spike/`, which tests read.
  - (c) Keep it public, and remove the host-private paths and names from the map and the
    tickets.
  - Recommendation: (c), by whoever keeps the map.
- **D-2 (R5): what a roll-up's `share` is counted over.**
  - (a) Record on each `broader` edge the member's rows within the step's own objects, and take
    the share from that. This adds a field to `graph-export/3` edges, so it is a contract bump,
    and the `same_bytes` digests are re-recorded.
  - (b) Count the edge and the node over the same objects, by dropping personal objects from
    the roll-up count. That changes induction, and the digests.
  - (c) Drop `share` from the entry. That is `layer-proposal/4`, and the host changes too.
  - Recommendation: (a). Then the proposal guard lands: it is 40 lines and a test, ready.

## Live requests

- **L-1 (C2): the GPU stack upgrade.** Run on a GPU host with the heavy environment. Upgrade
  torch to 2.13 or later, and transformers to 5.10 or later once `latence-disambig-glinker` is
  ported to 5.x. Then re-run the GPU validations: `scripts/validate-*-on-gpu.md` and
  `bash deploy/env/gpu-learned/smoke.sh`. After that, remove the three entries in
  `scripts/vuln-accepted.toml`; the gate then fails until they are removed.
- **L-2 (C8): the Airflow lane.** Put every `python -m pip install` in `ci-heavy.yml` under
  `--constraint "${CONSTRAINTS}"`, and run the `ci-heavy` workflow once.
