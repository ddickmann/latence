# Extending the pipeline

Every model, parser and index in this framework sits behind a **Capability** — a narrow Protocol —
fulfilled by a **Provider** in its own installable package. Replacing one is a config change plus a
class; it is not a fork, and it does not require editing the framework.

This page is the map. The hands-on walkthrough is
[Bringing your own Provider](tutorials/04-bringing-your-own-provider.md); the reference for the
`AdapterBase` conveniences is [Writing an adapter](writing-an-adapter.md).

## What a Provider actually has to be

A Provider is a class with a constructor taking one `dict` and the method its Capability declares.
That is the whole contract:

```python
class AcmeChunker:
    def __init__(self, config: dict) -> None:
        self._max = int(config.get("max_tokens", 400))

    @classmethod
    def profile(cls) -> ProviderProfile:
        return ProviderProfile(
            compute="cpu", memory_mb=8,
            license="Apache-2.0", license_verified=True,
            license_source="LICENSE", license_verified_on="2026-08-18",
            deterministic=True, batch=False,
        )

    def chunk(self, documents): ...
```

It subclasses nothing from this framework. The Capabilities in
[`latence_core.capability`](reference/capability.md) are `typing.Protocol`s, so satisfaction is
**structural** — a class satisfies `Chunker` by having `chunk`, not by inheriting anything.
`AdapterBase` exists and is worth using (device routing, batching, a typed error `guard`, provenance
helpers), but it is optional and the in-core reference Providers deliberately do not all use it.

Register it from your own package's `pyproject.toml`:

```toml
[project.entry-points."latence.providers"]
"chunk.acme" = "acme_chunker.provider:AcmeChunker"
```

The prefix before the dot names the Capability. That is the only naming rule, and it is enforced: a
name whose prefix matches no Capability raises rather than being silently ignored.

## What the framework gives an AI coding agent

The reason this is tractable for an agent is that the contract is **generated from the code**, not
written by hand about it:

```bash
latence contracts                        # the whole reference, as JSON
latence contracts --capability parse     # one Capability
```

`docs/reference/stage-contracts.json` carries, per Capability: the Protocol name and every method's
full typed signature, what the executor delivers to the Stage and what wiring its input must satisfy,
the checkpoint and export carrier schemas field by field, whether it is document-level or
corpus-level, the entry-point prefixes, the conformance cases that gate it, and the Providers that
already fulfil it. [`stage-contracts.md`](reference/stage-contracts.md) is the same content rendered
for reading.

A committed drift gate asserts the artifacts are byte-identical to a fresh render, so a changed
carrier field or a new Capability turns CI red until the reference is regenerated. The document
cannot quietly go stale, which is what makes it safe to hand to a machine that will act on it.

The working loop is: point the agent at the JSON entry for the Capability, have it implement the
Protocol, declare the entry point, then run conformance.

## Conformance is the gate, not the guarantee

```bash
latence stack check ./latence.stack.yaml     # offline lint: wiring, Protocols, config keys
```

The suite runs six checks against a Provider — valid records, graceful failure on adversarial input,
licence evidence, determinism as declared, device routing, and no PII in non-content fields. A
Provider under an existing Capability prefix is classified and gets its case automatically; nothing
in the framework needs editing to accept it. See [Conformance](CONFORMANCE.md) for what each check
asserts and how to read a failure.

What conformance guarantees is that a replacement **cannot break the end-to-end flow**. It says
nothing about whether your parser reads documents better than the one it replaced. Retrieval quality
is the responsibility of whoever introduces the component, and the
[benchmark harness](running-the-benchmarks.md) is how you measure it.

## Where the boundary actually is

| you want to | core edit needed |
|---|---|
| add a Provider under an existing Capability | no |
| swap a model, a parser, an embedder, a graph store | no |
| add a genuinely **new Capability** | yes, by design |

The last row is deliberate. A Capability declares nine facts about itself in one frozen descriptor —
its Protocol, its carrier, its level, its input plan, its entry-point prefix, and more — and the table
is checked for exhaustiveness at import. A new Capability that has not declared its facts raises
`ContractError` the first time anything imports the framework, rather than silently answering
`False` to a question like "is this corpus-level?" and slipping past an invariant that depends on the
answer. Adding a Capability means declaring those facts; there is no path where it merely behaves
wrongly.

## The wizard is the single point of configuration

There is no hand-authored YAML in the supported path:

```bash
latence setup      # inspects the corpus, asks what matters, emits a validated stack config
latence process    # runs it
```

`latence setup` writes one `latence.stack.yaml` — the whole pipeline, every Provider, every knob —
and validates it before writing: DAG structure, phase-boundary invariants, Capability satisfaction,
Provider availability, nested sub-Provider slots, unknown config keys, and cross-Stage wiring. A
mistyped Provider name is reported as a typo with the nearest registered name; an uninstalled heavy
package is reported as absent, with the package to install.

Because the config is one generated file, a third-party Provider is reachable from the wizard the
moment its package is installed and its entry point is declared — it appears as a choice, not as a
YAML edit. Re-running `setup` with the same answers regenerates the same file byte for byte.

For a stack you keep under version control, `latence stack check` lints it offline in CI without
running it.

## Related

- [Bringing your own Provider](tutorials/04-bringing-your-own-provider.md) — the hands-on build
- [Writing an adapter](writing-an-adapter.md) — `AdapterBase`, config validation, error typing
- [ProviderProfile & device handling](provider-profile.md) — compute, licence evidence, determinism
- [Conformance checklist](CONFORMANCE.md) — the six checks
- [Provider catalog](providers/catalog.md) — what already exists per Capability
- [Pipeline anatomy](anatomy/index.md) — every stage of the shipped pipeline in depth
