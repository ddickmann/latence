# Getting Started

Latence produces two AI-ready deliverables from a folder of messy documents — a RAG-ready
corpus and a knowledge graph, both as portable files. This section gets you from an empty
environment to those outputs.

## The one-command path

Everything ships CPU-first and offline by default ([ADR-0007](../adr/0007-cpu-first-reference-providers.md)),
so you can run a full pipeline on a laptop with no GPU and no network:

```bash
# Install from a clone — nothing is published to PyPI (see the note below).
git clone https://github.com/ddickmann/latence
cd latence
uv sync

# 1. A guided wizard emits ONE opinionated, quality-baked stack config (ADR-0041).
uv run latence setup

# 2. Run that config end-to-end over your documents.
uv run latence process --config stack.yaml --input ./my-documents --out ./corpus
```

!!! note "Why a clone, and not `pip install`"
    Nothing is published to PyPI: the distribution channel is the signed GitHub Release
    ([ADR-0062](../adr/0062-distribution-is-signed-github-releases-only-no-pypi-under-polyform-noncommercial.md),
    whose rationale is superseded and whose decision is reopened — not reversed — by
    [ADR-0063](../adr/0063-relicence-to-apache-2-0.md)). A `pip install latence-core` today
    fails with `ERROR: No matching distribution found`. `uv sync` installs the whole
    CPU-first workspace from the clone, which is what every command on this page assumes;
    the [quickstart](../quickstart.md) walks the same install step by step.

`latence process` runs the wizard-produced stack and writes the RAG corpus + knowledge
graph, plus a first-class [Quality Report](../adr/0008-quality-report-as-first-class-artifact.md)
substantiating the "AI-ready" claim.

```bash
# 3. Read the results — a read-only, offline, localhost console over what the run wrote.
#    `latence-console` is already in the environment `uv sync` built.
uv run latence-console ./corpus
```

`latence-console` serves the run list and each run's Quality Report as **headline quality
verdicts a non-engineer can read**, not raw JSON. It writes nothing, holds no state, and makes
no network requests — see [Read the results](run-console.md).

## Run it stage by stage

To understand what each Stage does — and to swap in a learned or GPU Provider and see the
quality lift — follow the walkthrough:

<div class="grid cards" markdown>

- :material-flask: **[Fresh-pod walkthrough](../TUTORIAL-fresh-pod-walkthrough.md)**
  From a bare GPU pod to AI-ready data over a mixed-file corpus with the **enterprise-SOTA**
  pipeline (served-vLLM OCR, GLiNER2 extraction, the GLinker entity-linker, Granite r2
  embeddings) — every Stage in isolation, then the full end-to-end run.

- :material-expansion-card: **[GPU testing guide](../GPU-TESTING-GUIDE.md)**
  Run the whole *learned* system yourself: the GPU Providers, validated on a CUDA pod.

</div>

## The core CLI

The `latence` CLI (installed with `latence-core`) is the entry point for every workflow:

| Command | What it does |
| --- | --- |
| `latence setup` | Guided wizard → one opinionated, quality-baked stack config (ADR-0041). |
| `latence process` | Run a full stack config end-to-end over a document folder. |
| `latence run` | Run an explicit Pipeline file. |
| `latence stack validate` | Validate a stack config end-to-end (see [Run & validate a stack](../running-a-stack.md)). |
| `latence bake-off` | Compare Providers on a Capability (see [Run a bake-off](../running-a-bakeoff.md)). |
| `latence tune` | Auto-tune a stack per device (see [Tune a stack per device](../tuning.md)). |
| `latence delta` | Run an incremental corpus delta (ADR-0018). |
| `latence retract` / `latence purge` | Soft-tombstone or hard-erase documents (GDPR). |

The read-only run console ships as its own package (`pip install latence-console`) so the core
CLI stays dependency-thin:

| Command | What it does |
| --- | --- |
| `latence-console <location>` | Serve the read-only run console over a Storage location ([Read the results](run-console.md)). |

## Where to go next

- Want to show a run to a non-engineer? [Read the results](run-console.md) — the run console.
- New to the model? Read [Concepts & Architecture](../concepts/index.md) — the
  Capability / Provider / Stage / Runner / Storage vocabulary the rest of the docs assume.
- Want to see what you can plug in? Browse the [Provider catalog](../providers/catalog.md).
- Building your own Provider? Start with [Authoring a Provider](../providers/README.md).
