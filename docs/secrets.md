# Secrets: the env-only model + secret-manager integration

Latence follows one rule for credentials (ADR-0012): **no key or endpoint is ever embedded in code,
a stack YAML, an image, or a chart.** Every credential is read from the **environment** at runtime,
and every deploy artifact **references** a secret store rather than carrying the value.

## What the pipeline reads from the environment

| Env var | Used by | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | every endpoint Provider (`parser.lighton_vllm`, `parser.endpoint`, `entity.endpoint`, `relation.llm`, `embedding.endpoint`) | Falls back to `EMPTY`/`not-needed` for a local vLLM, which accepts any placeholder. A hosted endpoint (OpenRouter, etc.) needs the real key. |
| `HF_TOKEN` | the vLLM servers, at **model-download** time only | Only needed to pull gated/private weights on first boot. Unset in air-gapped mode (pre-cache the model). See `docs/deployment.md`. |
| `LATENCE_METRICS` / `LATENCE_METRICS_TEXTFILE` | the observability seam | Not secrets — operational toggles. |

The provider code reads these directly (e.g. `api_key = config.get("api_key") or
os.environ["OPENAI_API_KEY"]`); a `base_url` is stack config, not a secret. **Never** put a key in a
stack YAML's `config.api_key` — leave it to the environment.

## Docker Compose

Compose reads from your shell environment (or a git-ignored `.env` file). `deploy/.gitignore`-style
hygiene: never commit `.env`.

```bash
export OPENAI_API_KEY=sk-...        # only for a hosted endpoint; a local vLLM ignores it
export HF_TOKEN=hf_...              # only to download gated weights the first time
docker compose -f deploy/docker-compose.yml up --build
```

The compose file passes them through as `environment:` entries with safe defaults
(`OPENAI_API_KEY: "${OPENAI_API_KEY:-EMPTY}"`) — nothing is baked into the image.

## Kubernetes: reference a Secret, don't embed it

The Helm chart consumes credentials as **env from a Secret you control**. The recommended path
(`secrets.create=false`) never has the chart hold a secret at all:

```yaml
# values-prod.yaml
secrets:
  create: false
  existingSecret: latence-endpoint-keys   # a Secret your secret manager projects
```

The pipeline pod then gets `OPENAI_API_KEY` via `secretKeyRef` (optional, so a local-vLLM deploy
with no key still runs), and the vLLM servers get `HF_TOKEN` the same way. The chart's own
`templates/secret.yaml` is **quick-start only** (`secrets.create=true` with `secrets.data`) — a
convenience for a throwaway cluster, never a place for a real key in git.

## Integrating a secret manager

Point `existingSecret` at a Secret **materialised by your platform's secret store** — the chart does
not care how the Secret gets there:

* **HashiCorp Vault** — the [Vault Agent Injector](https://developer.hashicorp.com/vault/docs/platform/k8s/injector)
  or the [Vault Secrets Operator](https://developer.hashicorp.com/vault/docs/platform/k8s/vso) syncs a
  Vault path into a k8s Secret named `latence-endpoint-keys`. Set `existingSecret` to it.
* **External Secrets Operator** — an `ExternalSecret` pulls from AWS Secrets Manager / GCP Secret
  Manager / Azure Key Vault / Vault and writes the k8s Secret the chart references:

  ```yaml
  apiVersion: external-secrets.io/v1beta1
  kind: ExternalSecret
  metadata: {name: latence-endpoint-keys}
  spec:
    secretStoreRef: {name: my-cluster-store, kind: ClusterSecretStore}
    target: {name: latence-endpoint-keys}      # <- values.secrets.existingSecret
    data:
      - secretKey: OPENAI_API_KEY
        remoteRef: {key: prod/latence/openai_api_key}
      - secretKey: HF_TOKEN
        remoteRef: {key: prod/latence/hf_token}
  ```

* **Cloud CSI secret drivers** (AWS/GCP/Azure Secrets Store CSI) — mount the secret as a volume that
  is also synced to a k8s Secret (`secretObjects`), then reference that Secret.

## Discipline checklist

* No key in a stack YAML, an image layer, a chart default, or git history.
* Endpoint keys are `optional: true` in the `secretKeyRef` so a keyless local-vLLM deploy still runs.
* `HF_TOKEN` is a **build/first-boot** concern only; an air-gapped run has no token and no phone-home
  (`HF_HUB_OFFLINE=1`).
* Rotate at the secret store; the pod picks up the new value on its next run (a CronJob run or a Job
  re-create). No image rebuild needed.
* The Latence Quality Report, spans, and metrics are **counts only** — a rotated key or a leaked
  document never appears in observability output (ADR-0034/0047).
