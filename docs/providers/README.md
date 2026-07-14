# Authoring a Provider

A **Provider** is a concrete implementation of a [Capability](../concepts/index.md) — an
in-process model, a remote inference endpoint, or a cloud AI service — registered as a
plugin. This is the framework's extension point: because a Capability is a structural
`typing.Protocol` ([ADR-0004](../adr/0004-capability-protocols-with-provider-plugins.md)),
**your Provider needs no base class and no import of core** to satisfy it. You implement
one method, declare a profile, register an entry point, and pass conformance.

This guide is the end-to-end path. The mechanical checklist is
[Passing Provider Conformance](../CONFORMANCE.md); the ergonomic helpers are in
[Writing an adapter with `AdapterBase`](../writing-an-adapter.md); the real template is the
[`parser.pdfplumber`](../reference/example-provider.md) package.

## 1. Implement the Capability

Pick the Capability your Provider fulfils and implement its one method. Each Capability is a
narrow, `runtime_checkable` Protocol in
[`latence_core.capability`](../reference/capability.md) — for example the `Parser`:

```python
from collections.abc import Iterable, Iterator
from latence_core.contracts import ParserInput, DocumentRecord

class MyParser:
    """Any object with this shape satisfies the Parser Capability — no base class needed."""

    def parse(self, inputs: Iterable[ParserInput]) -> Iterator[DocumentRecord]:
        for item in inputs:
            markdown = my_native_library.convert(item.content)   # your model / library call
            yield DocumentRecord(
                provenance=item.provenance,          # carry Provenance losslessly
                classification=item.classification,  # carry Classification downstream
                markdown=markdown,
                # ...page map, offsets, disposition...
            )
```

The Capabilities you can fulfil: `Source`, `Parser`, `Chunker`, `IntakeScreener`,
`ContentScreener`, `EntityExtractor`, `RelationExtractor`,
`FusedEntityRelationExtractor`, `PIIDetector`, `Profiler`, `Disambiguator`,
`GraphAssembler`, `GraphCompleter`, `ContextEnricher`, `DeltaProcessor`, `Embedder`,
`LabelInducer`, `Export`. A **Fused Provider** may implement more than one — e.g.
`FusedEntityRelationExtractor` emits both entity and relation carriers in one pass
([ADR-0013](../adr/0013-relation-extraction-provider-chosen-empirically-fused-capability-allowed.md)).

### Four invariants every Provider must hold

- **Carry Provenance + Classification.** Every emitted record carries the immutable
  [Provenance](../concepts/index.md) chain and the document's Classification. A
  sub-document carrier (a mention, a relation) resolves its character span back to a source
  page through the parent's page map — reuse `OffsetIndex` / `PageOffsetIndex`
  ([ADR-0031](../adr/0031-entity-extraction-emits-true-original-offsets-via-carried-strip-offset-map.md)),
  never a weaker hand-rolled mapper.
- **Fail typed, leak nothing.** Map a native-library exception to a **typed**
  `LatenceError` — a `ProviderError` (runtime fault) or `ConfigError` (operator
  bad-config) — carrying the provider + op but **no record content or PII**
  ([ADR-0034](../adr/0034-observability-typed-errors-and-thin-tracing-seam.md)). Or take
  your Capability's defined graceful path (a Screening quarantine, a `PARSE_ERROR` record).
- **Be deterministic, or declare you are not.** Emit records in a stable order. If learned
  float inference makes byte-identical output impossible, declare `deterministic=False` —
  do not claim determinism you cannot hold.
- **Stay dep-light at import.** Defer your heavy `import` (torch, openai, spaCy) to first
  use, so Provider *discovery* and the CPU-only device seam never require the heavy stack to
  be present ([ADR-0016](../adr/0016-thin-core-per-provider-packages.md)).

## 2. Declare a `ProviderProfile`

Every Provider declares a frozen, typed
[`ProviderProfile`](../reference/providers.md#latence_core.providers.profile.ProviderProfile)
— the enterprise-readiness descriptor the registry, the device seam, and the bake-off all
read from **one declared object** ([ADR-0036](../adr/0036-provider-ecosystem-and-bakeoff.md)).
Declare it as a `profile` **classmethod** (preferred — you can compute it from config) or a
`PROFILE` ClassVar:

```python
from latence_core.providers.profile import ProviderProfile

class MyParser:
    @classmethod
    def profile(cls) -> ProviderProfile:
        return ProviderProfile(
            compute="cpu",              # "cpu" | "gpu" | "either"
            memory_mb=96,               # approximate peak resident MiB (declared, not measured)
            model_id=None,              # HF/model id, or None for a pure-code / endpoint client
            license="MIT",              # headline SPDX (or the UNVERIFIED sentinel)
            license_verified=True,      # True ONLY when weights AND code were verified
            license_code="MIT",         # SPDX of the library code
            license_weights=None,       # SPDX of the weights (None for a pure-code Provider)
            license_source="https://github.com/jsvine/pdfplumber/blob/master/LICENSE.txt",
            license_verified_on="2025-01-15",  # ISO date the human check happened
            deterministic=True,
            batch=True,
            cost_per_1k=None,           # per-1k cost for an endpoint Provider; None for local
        )
```

!!! warning "License honesty is enforced by construction"
    The `ProviderProfile` model *cannot be built* with a dishonest license claim
    ([ADR-0012](../adr/0012-permissive-only-reference-providers-gemma-opt-in.md), and the
    research-diligence directive):

    - **Weights and code are recorded separately.** A model-bearing Provider (`model_id`
      set) must declare `license_weights`; a pure-code Provider must not.
    - `license_verified=True` **requires its evidence** — a non-empty `license_source`
      citation (HF model-card / GitHub LICENSE / repo LICENSE path) and an ISO
      `license_verified_on` date. A bare `True` with no citation raises `ValueError`.
    - The `UNVERIFIED` sentinel can **never** coexist with `license_verified=True`. If you
      could not verify a license in-slice, declare it `UNVERIFIED` with
      `license_verified=False` — an honest gap, **never a guessed permissive claim**.
    - The headline `license` must equal the weights SPDX (model-bearing) or the code SPDX
      (pure-code) — the three fields cannot silently disagree.

!!! note "Profiler exception"
    If your Capability is **Profiling**, declare via a `PROFILE` ClassVar, **not** a
    `profile` classmethod — the `Profiler` Capability's own I/O method is also named
    `profile`, and the registry's reader deliberately treats a plain instance method named
    `profile` as "no declaration". See `profile_of` in the
    [reference](../reference/providers.md).

## 3. Reuse `AdapterBase` (optional but recommended)

`AdapterBase` is an **optional** convenience base that does the cross-cutting glue **one
right way** ([ADR-0036 §1](../adr/0036-provider-ecosystem-and-bakeoff.md)) — you may still
satisfy the Capability structurally without it. It gives you four composable pieces:

| Helper | Does |
| --- | --- |
| `self.device` | Resolves `config.device` (`auto`/`cpu`/`cuda`) against your profile via `select_device` — a `compute="gpu"` adapter on a CPU-only host sees the **skip** decision, never a fabricated device. |
| `AdapterBase.batched(...)` | Groups an input iterable into `config.batch_size` chunks and calls your `_run_batch`, degrading to one-item calls when the profile is not batch-capable — **deterministic input-order output**, no whole-corpus buffering (streaming-friendly, [ADR-0033](../adr/0033-streaming-memory-bounded-runner.md)). |
| `AdapterBase.guard("<op>")` | A context manager that catches a native exception and re-raises it as the right typed `LatenceError` (`ProviderError` / `ConfigError`), tagged with provider + op, carrying no content/PII. |
| `carry_provenance` / `rebase_offsets` | Map a native library's output back onto the input's Provenance/Classification and character offsets, reusing the `OffsetIndex` / `PageOffsetIndex` seam — never a weaker mapper. |

The real, shipping template is
[`parser.pdfplumber`](../reference/example-provider.md) — the first Wave-1 adapter built
*through* `AdapterBase`. It uses `guard` to map a corrupt-PDF `PdfminerException` to a typed
`ProviderError`, `carry_provenance` + the core `PageMap` seam to keep offsets and pages
exact, and `batched` to stream over documents. Copy it. Full walkthrough in
[Writing an adapter](../writing-an-adapter.md).

## 4. Register the entry point

A Provider is a plugin: register it under the `latence.providers` entry-point group in your
package's `pyproject.toml`, keyed by its **name** (`<capability>.<technique>`). The name
prefix is how conformance auto-maps the Provider to its Capability:

```toml
[project.entry-points."latence.providers"]
"parser.pdfplumber" = "latence_parser_pdfplumber.provider:PdfPlumberParser"
```

Ship it as its own package so its heavy deps stay out of `latence-core`
([ADR-0016](../adr/0016-thin-core-per-provider-packages.md)). The registry
(`ProviderRegistry`) loads the class by name and reads its profile off the class object —
without importing your heavy deps beyond the class it already loaded.

## 5. Pass conformance

The **Provider Conformance suite** is the enterprise-readiness gate — a pytest suite
parametrized over *every* registered Provider, asserting a fixed C1–C6 contract
([ADR-0036 §5](../adr/0036-provider-ecosystem-and-bakeoff.md)). "Passes conformance" *is* the
definition of enterprise-ready:

| Check | Asserts |
| --- | --- |
| **C1** valid typed records | Contract-valid records with Provenance + Classification and in-range, well-ordered offsets. |
| **C2** graceful failure | On adversarial input, a **typed** `LatenceError` or your defined graceful path — never a bare crash, hang, or silent-wrong output; message carries no PII. |
| **C3** license recorded | A `ProviderProfile` is present (missing → **FAIL**); a restricted license must be `license_verified=True` (opt-in). |
| **C4** determinism-or-documented | `deterministic=True` ⇒ twice yields byte-identical output; declaring `True` and not being is a **FAIL**. |
| **C5** resource/device honored | `compute="gpu"` on a CPU host is skipped-with-flag; no GPU import at module load. |
| **C6** no secret/PII leak | A non-Redactor Provider surfaces no raw PII in non-content fields, logs, or spans. |

A new Provider is **auto-covered** the moment its Capability has a `ConformanceCase`; add a
Provider for a Capability with no case and the suite **fails loudly** rather than silently
skipping. Run it:

```bash
LATENCE_CUDA=0 PYTHONHASHSEED=0 uv run pytest \
  packages/latence-core/tests/test_conformance.py \
  packages/latence-core/tests/test_e2e_conformance.py
```

The full checklist, including the endpoint/GPU stub path, is in
[Passing Provider Conformance](../CONFORMANCE.md).

## 6. Prove it against the incumbents

Once your Provider is green, compare it to the reference Providers on the same Capability
with a **bake-off** — same inputs, one table of quality / throughput / license / cost
columns, all read from the declared profiles
([ADR-0036](../adr/0036-provider-ecosystem-and-bakeoff.md)):

```bash
latence bake-off --capability entity --providers entity.gazetteer,entity.gliner,entity.mine
```

See [Run a bake-off](../running-a-bakeoff.md). That is the whole loop: implement one method,
declare an honest profile, register a plugin, pass conformance, win the bake-off.
