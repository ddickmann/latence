# R5-docs_truth-7 — the Qdrant sink's only credential example is syntax with no implementation

**Severity:** major
**Where:** `packages/latence-sink-qdrant/README.md:128` (the Configuration YAML example);
implementation `packages/latence-sink-qdrant/src/latence_sink_qdrant/cli.py:95-140`
(`_load_config_file` / `_config_from`) and
`packages/latence-sink-qdrant/src/latence_sink_qdrant/sink.py:316` (`_client_kwargs`).

## What is wrong

The Qdrant sink README documents

```yaml
api_key: ${QDRANT_API_KEY}
```

as *the* way to supply a credential from a config file. Nothing in the repository expands
`${VAR}` in a sink config file. `_load_config_file` calls `yaml.safe_load` (or `json.loads`)
and returns the mapping as-is; there is no `os.path.expandvars` and no env-reference
resolution in the sink, in the CLI, or in `latence-core`. The literal 17-character string
`${QDRANT_API_KEY}` is threaded verbatim into `QdrantSink._client_kwargs` and handed to
`QdrantClient(api_key=...)`.

`${VAR}` appears in exactly one place in the whole doc set, so there is no repo-wide
convention that would make this an accepted idiom. The rest of the framework has the
*opposite* convention, and enforces it: `latence-source-sharepoint`'s `reject_inline_secrets`
refuses any config key that looks like a credential and tells the operator to name the
environment **variable** (`token_env: SHAREPOINT_TOKEN`), and `SECURITY.md` states secrets
are environment-only and are scrubbed from `manifest.json`. The sink had neither the
documented `${VAR}` expansion nor the framework's `*_env` indirection — so there was no
working way to keep a Qdrant credential out of the stack file.

## Failure scenario

An operator follows the README's only credential example, writes `qdrant.yaml` containing
`api_key: ${QDRANT_API_KEY}`, exports `QDRANT_API_KEY=s3cr3t`, and runs:

```
latence-sink-qdrant load ./out/export --config qdrant.yaml
```

`_load_config_file` returns `{'api_key': '${QDRANT_API_KEY}', ...}` and the constructed
`QdrantSink._client_kwargs` is `{'api_key': '${QDRANT_API_KEY}', 'url': ...}` — the
placeholder is handed to `QdrantClient` unchanged. Against a secured Qdrant the load fails
with a `SinkAuthError` (401/403) that names an auth problem but not the real cause. The
operator's most likely recovery is to inline the real key into the YAML — the exact outcome
`SECURITY.md` and the SharePoint Source's enforcement exist to prevent.

## Acceptance criteria

1. A sink config may name the environment variable holding the credential —
   `api_key_env: QDRANT_API_KEY` — and the sink resolves it from the process environment at
   construction, so the secret never lives in the file.
2. An `api_key_env` naming an unset or empty variable fails with a typed `SinkConfigError`
   (ADR-0034: no silent errors), and the message never contains a credential value.
3. An unexpanded `${...}` placeholder left in `api_key` is rejected at construction with a
   typed `SinkConfigError` that points at `api_key_env` — an operator who followed the old
   README gets the real cause instead of a mystery 401 from the store.
4. `api_key_env` is never threaded to `QdrantClient` as a kwarg (only the resolved `api_key`
   is), and an explicit `api_key` / `--api-key` still wins over `api_key_env`.
5. The README documents `api_key_env` (not `${VAR}`) as the credential path, with a table row,
   and no `${VAR}` config syntax survives anywhere in the doc set.
6. Regression tests cover: resolution from the environment, the unset-variable error, the
   `${...}` placeholder rejection, and that no secret appears in the error text.
