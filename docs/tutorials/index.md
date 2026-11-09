# Tutorials

Four tutorials, in order. Each one builds on the last and ends where the next begins. All of them
run on a laptop; only the last needs a GPU, and it says exactly where. They keep their original
numbers: Tutorial 3 (retrieval) was removed with the retrieval tooling ([ADR-0068](../adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)).

They share one convention with the rest of this documentation: every step ends in a
**`✓ look for`** block stating what a correct result looks like. If your output disagrees with one,
stop there. A mismatch is information — it tells you which step broke, which is far more useful than
discovering at the end that something did.

| | Tutorial | You will build | Needs |
| --- | --- | --- | --- |
| 1 | [Your first pipeline](01-your-first-pipeline.md) | A folder of documents turned into an AI-ready corpus + knowledge graph, stage by stage. | Nothing but the workspace. No network, no downloads, no GPU. |
| 2 | [Understanding the output](02-understanding-the-output.md) | Fluency in every emitted artifact — and the ability to trace a graph edge back to the sentence it came from. | Tutorial 1's output. |
| 4 | [Bringing your own Provider](04-bringing-your-own-provider.md) | A real Provider package of your own, registered as a plugin and green against the conformance suite. | The workspace. |
| 5 | [Production](05-production.md) | Served GPU models, staged runs, checkpoint/resume, and incremental corpus deltas. | Tutorials 1–2; a GPU for the served-model sections. |

## Before you start

Work through the [quickstart](../quickstart.md) first if you have not — it installs the workspace
and proves it end to end in one command. These tutorials assume you are inside a clone with
`uv sync` already run.

They also assume the vocabulary from [Concepts & Architecture](../concepts/index.md):

* a **Capability** is a narrow, model-agnostic interface (Parser, EntityExtractor, Embedder, …);
* a **Provider** is a concrete implementation of one, shipped as its own package and discovered as
  a plugin;
* a **Stage** binds one Provider to one node of a pipeline DAG;
* the **Runner** executes that DAG; the **Storage** seam is where every artifact lands.

If you want the reasoning rather than the mechanics — why the pipeline extracts narrow and resolves
wide — read the [pipeline anatomy](../anatomy/index.md) and the [decision log](../adr/index.md).
