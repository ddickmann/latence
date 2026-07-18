# latence-source-sharepoint

The **SharePoint / OneDrive Source Provider** for the Latence framework — point a stack at a
SharePoint document library and run `latence process` to get the standard deliverable (RAG
corpus + KG + Quality Report) with SharePoint-rooted Provenance.

It plugs into the existing Source seam (ADR-0014) as `source.sharepoint`. **No Stage contract
changes**: it implements the same one-method `produce(storage) -> Iterator[ParserInput]` the
in-core `source.local_folder` implements, so every downstream Stage is untouched.

- **Zero third-party dependencies.** It speaks Microsoft Graph's REST/JSON API over the Python
  standard library. No Microsoft SDK, no HTTP client library, no vendored code — and
  `latence-core` takes on nothing new (ADR-0016).
- **Secrets are environment-only, and it is enforced.** A config that inlines a token or a
  client secret raises `ConfigError` before a connection is opened.
- **Delta-aware.** An incremental run skips unchanged documents **without downloading them**.

## Install

```bash
uv pip install latence-source-sharepoint
```

## Configure

```yaml
stages:
  - name: source
    capability: source
    provider: source.sharepoint
    config:
      drive_id: b!aBcDeF...            # the Graph id of the document library
      root_path: /Contracts            # optional; omit for the whole drive
      extensions: [md, txt, pdf]       # optional; defaults to text/markdown
      name_strategy: path              # 'item' (default) or 'path' — see below
      sensitivity: confidential
      auth:
        mode: client_credentials
        tenant_id: 00000000-0000-0000-0000-000000000000
        client_id_env: SHAREPOINT_CLIENT_ID
        client_secret_env: SHAREPOINT_CLIENT_SECRET
```

`auth.mode` is either:

- **`env_token`** (the default) — you already mint app tokens (MSAL, a sidecar, a
  workload-identity broker) and export the current bearer token into an environment variable
  named by `token_env` (default `LATENCE_SHAREPOINT_TOKEN`). The framework does no OAuth at all.
  The variable is re-read on every request, so a rotating token is picked up mid-run.
- **`client_credentials`** — you registered an Entra ID app and give the framework the *names*
  of the environment variables holding its client id and secret. The connector performs the
  standard OAuth 2.0 client-credentials exchange and caches the token in memory until shortly
  before it expires.

Config names environment **variables**, never values. `token`, `client_secret`, `password`,
`api_key` and friends are rejected as config keys at any nesting depth, which is what keeps a
credential out of the stack file, out of git, and out of the run manifest.

### Finding your `drive_id`

```bash
# site URL -> site id
GET https://graph.microsoft.com/v1.0/sites/contoso.sharepoint.com:/sites/Legal
# site id -> the drive (document library) ids
GET https://graph.microsoft.com/v1.0/sites/{site-id}/drives
```

The app registration needs the **application** permissions `Sites.Read.All` (or
`Files.Read.All`) with admin consent.

### `name_strategy`

Downstream keys (the gold-set harness, Provenance) assume one document per file name in a
corpus, and a real document library routinely holds `report.md` in two folders.
`name_strategy: item` (the default, matching `source.local_folder`) uses the bare file name and
**raises** on a collision rather than silently merging two documents; `name_strategy: path`
derives the name from the drive-relative path (`2026_report.md`) and is unique by construction.

## Provenance

Every fetched document carries:

| field | value |
| --- | --- |
| `source_system` | `sharepoint` |
| `source_uri` | `sharepoint://{drive_id}/{folder}/{name}?itemId={graph item id}` |
| `document_id` | the content address (see Delta) |

`source_uri` carries the human location *and* the stable Graph identity that resolves back to
the exact `driveItem`, in the one URI field the frozen Provenance contract provides. Every
downstream record inherits it unchanged, so an exported chunk still names the SharePoint item it
came from.

## Delta (ADR-0018)

Graph reports a content hash for every `driveItem` (`quickXorHash` on SharePoint Online,
`sha256Hash` on OneDrive Personal) **in the listing, before any download**. The connector mints
`document_id = sha256(<algorithm>:<hash>|<size>)` from it, so:

- an unchanged document keeps the same id across runs; the DeltaRunner marks it `UNCHANGED` and
  injects the id into this Provider's `exclude_document_ids` — and the connector then **skips it
  without transferring a byte**. On a large library that is the difference between an
  incremental run and a full re-download;
- a changed document gets a new id (`ADDED`, retiring the old one as `DELETED`) and is
  re-fetched;
- a document removed from the library stops appearing in the listing, so the manifest diff
  surfaces it as `DELETED`.

This is exactly the reference `source.local_folder` semantics — content addressing means "same
id" is "byte-identical", which is what the Runner's cross-run record reuse relies on. A
stable-identity Source would report `UPDATED` instead of `ADDED`+`DELETED`; see
`latence_core.state.delta_detector`.

If a drive reports no hash for an item, the connector falls back to downloading and hashing the
bytes — still correct, just not free. It never derives identity from mutable metadata such as
`eTag`, which would churn the corpus on a metadata-only edit.

## Offline testing and the live smoke path

The package **ships** its faithful Microsoft Graph fake in
`latence_source_sharepoint.testing`. It is not a permissive stand-in: it reproduces Graph's
refusals, so an offline pass means something.

| the fake rejects | as Graph does |
| --- | --- |
| a missing `Authorization` header | `401 InvalidAuthenticationToken` |
| an expired / unknown bearer token | `401 InvalidAuthenticationToken` |
| a token in the query string | `401` — Graph accepts it in the header only |
| `$skip` on a `driveItem` children collection | `400 invalidRequest` |
| an unknown drive id, folder path, or item id | `404 itemNotFound` |
| an `Authorization` header sent to the pre-authenticated download host | `403 AuthenticationFailed` |
| a client-credentials exchange with a wrong secret | `401 invalid_client` |

Throttling is scriptable, so the `429` + `Retry-After` path is genuinely exercised.

### Offline dry run

```yaml
config:
  drive_id: demo-drive
  transport: 'latence_source_sharepoint.testing:demo_transport'
  auth: { mode: env_token, token_env: MY_TOKEN }
```

`MY_TOKEN=demo-token latence process ...` runs the whole pipeline against a canned three-document
library with no tenant and no network. `transport` accepts any `module:factory` reference, which
is also how you supply a corporate proxy / mTLS transport without forking the Provider.

### Live smoke test (needs a tenant — deliberately not run in CI)

1. Register an Entra ID app; grant `Sites.Read.All` **application** permission; admin-consent it.
2. Export the credentials from your secret manager:
   ```bash
   export SHAREPOINT_CLIENT_ID=... SHAREPOINT_CLIENT_SECRET=...
   ```
3. Resolve the drive id with the two Graph calls above.
4. Run the smallest possible pipeline against one folder:
   ```bash
   latence process --stack stacks/sharepoint-smoke.yaml
   ```
5. Assert the deliverable: every exported row's `provenance.source_system` is `sharepoint` and
   its `source_uri` resolves — paste the `itemId` into
   `GET /drives/{drive_id}/items/{itemId}` and you should get the same document back.
6. Re-run it unchanged. The second run must report every document `UNCHANGED` and transfer no
   document bytes (watch the run's duration, or a network counter).

Live numbers are never fabricated into this repo: a tenant is required, so the shipped
conformance verdict for this Provider is flagged *verified against a stub*.

## Licenses (ADR-0012, verified 2026-07-18)

| component | code license | weights license | source |
| --- | --- | --- | --- |
| `latence-source-sharepoint` | Apache-2.0 | n/a — ships no model | this repo's `LICENSE` |
| Python standard library (`urllib`, `json`, `hashlib`) | PSF-2.0 | n/a | <https://docs.python.org/3/license.html> |

There are no other dependencies. Microsoft Graph is a **service**, not distributed code:
calling it carries no software-license obligation, and your tenant's own terms govern your use
of it — the framework makes no claim about them.
