# Tutorial 4 — Bringing your own Provider

**What you will build.** A real Provider, in a package of your own, outside this repository:
implement one method, declare an honest profile, register an entry point, pass the conformance
suite, and wire it into a running pipeline. Then you will deliberately break it in three ways and
watch the suite catch each one.

This is the extensibility story, and it is the one that decides whether a framework is worth
adopting. The claim being tested is specific: **a Provider needs no base class and no import of the
framework's core to satisfy a Capability**, because a Capability is a structural
`typing.Protocol` ([ADR-0004](../adr/0004-capability-protocols-with-provider-plugins.md)).

**Prerequisites.** [Tutorial 1](01-your-first-pipeline.md), and a clone of the framework with
`uv sync` run. No GPU, no downloads.

**Time.** About 45 minutes.

**One convention for this tutorial.** Your package lives *outside* the framework clone, but every
`uv` command below is run *from* the clone, so that it uses the framework's environment. The
commands assume the package directory sits next to the clone, at `../acme-embedder`.

---

## 1. Pick a Capability that genuinely varies

The Capabilities you can fulfil as a Stage: `Source`, `Parser`, `Chunker`, `IntakeScreener`,
`ContentScreener`, `LabelInducer`, `EntityExtractor`, `RelationExtractor`,
`FusedEntityRelationExtractor`, `PIIDetector`, `Profiler`, `TypeConsolidator`, `Disambiguator`,
`GraphAssembler`, `GraphCompleter` and `Export` — plus two seams that are not Stages: `Embedder`,
which another Provider names in its config, and `DeltaProcessor`, which the delta engine uses.
Each is a `runtime_checkable` Protocol in [`latence_core.capability`](../reference/capability.md).

We will build an **Embedder**, because it is small enough to read in one screen and because it is a
seam that genuinely matters rather than a toy. An Embedder maps text to a fixed-width vector. In
this framework it is a signal the resolution phase weighs: `disambiguation.embedding` asks one
whether two mentions are the same entity, and a corpus export can attach one vector per row
([Embedders](../embedders.md)). Everything around it is fixed. Exactly one thing varies: **how
text becomes a vector.** That is the Capability:

```python
class Embedder(Protocol):
    @property
    def dimension(self) -> int:
        """The fixed embedding width every returned vector has."""

    def embed(self, texts: Iterable[str]) -> Iterator[list[float]]:
        """Map each input text to a ``dimension``-wide vector, in input order."""
```

One property and one method. The contract on top of them: one vector per input text, in input
order, each exactly `dimension` wide; an empty input yields nothing rather than crashing; and a
deterministic Embedder is byte-reproducible.

Three references ship, and they are deliberately *different* rather than one plus a rename:
`embedding.hashing` is the in-core reference, a bag of whole-token hashes with no model;
`embedding.sentence_transformers` loads a learned model in process; `embedding.endpoint` calls a
served one. Your Provider will be a fourth: a **character-trigram** embedder. The in-core
reference hashes whole tokens, so `Corp` and `Corporation` are unrelated terms to it. Yours hashes
three-character windows, so two spellings of one name share most of their features.

---

## 2. Scaffold the package

Next to the framework clone:

```bash
mkdir -p ../acme-embedder/src/acme_embedder ../acme-embedder/tests
```

`../acme-embedder/pyproject.toml`

```toml
[project]
name = "acme-embedder"
version = "0.1.0"
description = "An Embedder that hashes character trigrams, so spelling variants of a name stay close."
requires-python = ">=3.11"
dependencies = ["latence-core"]

[project.entry-points."latence.providers"]
"embedding.acme_chargram" = "acme_embedder.provider:CharGramEmbedder"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/acme_embedder"]
```

The entry-point block is the whole registration mechanism. The key is the Provider **name**, and its
shape is load-bearing: `<capability>.<technique>`. The prefix before the dot is how the conformance
suite maps your Provider to the Capability it must be checked against, so `embedding.acme_chargram`
is auto-covered by the Embedding case the moment it is installed. Get the prefix wrong and the suite
will tell you loudly rather than skip you.

Ship it as **its own package**, so whatever heavy dependency it needs stays out of `latence-core`
([ADR-0016](../adr/0016-thin-core-per-provider-packages.md)). This one needs nothing but the
standard library.

---

## 3. Implement the Capability

`../acme-embedder/src/acme_embedder/provider.py`

```python
"""An Embedder that hashes character trigrams, so spelling variants of a name stay close."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator
from typing import Any

from latence_core.providers.profile import ProviderProfile


def _bucket(gram: str, dimension: int) -> int:
    """A stable bucket for a trigram (sha256, so it does not depend on PYTHONHASHSEED)."""
    return int.from_bytes(hashlib.sha256(gram.encode("utf-8")).digest()[:8], "big") % dimension


class CharGramEmbedder:
    """Lowercase the text, slide a three-character window over it, hash each window to a bucket.

    ``Globex Corp`` and ``Globex Corporation`` share most of their trigrams, so their vectors are
    close. A whole-token hashing embedder sees ``corp`` and ``corporation`` as unrelated terms.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        return ProviderProfile(
            compute="cpu",
            memory_mb=8,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_weights=None,
            license_source="https://example.com/acme-embedder/blob/main/LICENSE",
            license_verified_on="2026-11-10",
            deterministic=True,
            batch=True,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._dimension = int((config or {}).get("dimension", 64))
        if self._dimension < 1:
            msg = f"dimension must be >= 1, got {self._dimension}."
            raise ValueError(msg)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        padded = f"  {' '.join(text.lower().split())} "
        for i in range(len(padded) - 2):
            vector[_bucket(padded[i : i + 3], self._dimension)] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return vector if norm == 0.0 else [round(v / norm, 8) for v in vector]

    def embed(self, texts: Iterable[str]) -> Iterator[list[float]]:
        for text in texts:
            yield self._embed_one(text)
```

Note what is *not* there. No base class. No `latence_core` import except the profile type — and even
that is only because the profile is a typed object; the Capability itself is satisfied structurally.
The class does not know it is a plugin.

Three conventions it does follow, and all three matter:

* **`__init__(self, config)`** — every Provider is constructed from one config mapping, so a stack
  YAML's `config:` block reaches it uniformly. Accept it even when you have no knobs.
* **Streaming** — `embed` is a generator over the input iterable. Providers are expected not to
  buffer a whole corpus; the Runner streams, and so should you
  ([ADR-0033](../adr/0033-streaming-memory-bounded-runner.md)).
* **A default width of 64.** The conformance case for this Capability constructs your Embedder
  with an empty config and expects vectors as wide as the in-core reference's default, which is 64.
  An Embedder that defaults to another width fails the first check. A stack can still ask for any
  width through `config`.

### The profile, and why it cannot lie

The `ProviderProfile` is the enterprise-readiness descriptor that the registry, the device seam and
the bake-off all read from **one declared object**. Declare it as a `profile` classmethod (preferred
— you can compute it from config) or a `PROFILE` ClassVar.

The license fields are enforced *by construction*, not by review. Try to build a profile that claims
verification without evidence:

```bash
uv run python -c "
from latence_core.providers.profile import ProviderProfile
ProviderProfile(compute='cpu', memory_mb=8, model_id=None, license='Apache-2.0',
                license_verified=True, deterministic=True, batch=True, cost_per_1k=None)
"
```

✓ **look for** — a refusal at construction time, before any test runs:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ProviderProfile
  Value error, license_verified=True requires a non-empty license_source (a citation for
  where the license was checked — HF model-card / GitHub LICENSE / repo LICENSE). ...
```

The rules the model enforces
([ADR-0012](../adr/0012-permissive-only-reference-providers-gemma-opt-in.md)):

* **Weights and code are recorded separately.** A model-bearing Provider (`model_id` set) must
  declare `license_weights`; a pure-code Provider must not.
* `license_verified=True` requires its evidence — a `license_source` citation **and** an ISO
  `license_verified_on` date.
* The `UNVERIFIED` sentinel can never coexist with `license_verified=True`.
* The headline `license` must equal the weights SPDX (model-bearing) or the code SPDX (pure-code) —
  the three fields cannot silently disagree.

If you could not verify a license, declare it `UNVERIFIED` with `license_verified=False`. That is an
honest gap and the framework is built to carry it. A guessed permissive claim is not.

!!! note "One exception, worth knowing before it bites"
    If your Capability is **Profiling**, declare the profile as a `PROFILE` ClassVar, **not** a
    `profile` classmethod — the `Profiler` Capability's own I/O method is also named `profile`, and
    a classmethod would shadow it. The registry's reader deliberately treats a plain instance method
    named `profile` as "no declaration", so a Capability method is never mislabelled malformed.

---

## 4. Install it and watch it be discovered

From the framework clone:

```bash
uv pip install -e ../acme-embedder
```

```bash
uv run python -c "
from latence_core.capability import ProviderRegistry
registry = ProviderRegistry()
print('discovered:', 'embedding.acme_chargram' in registry.names())
cls = registry.load('embedding.acme_chargram')
embedder = cls({'dimension': 8})
print(embedder.dimension, list(embedder.embed(['Globex Corp']))[0])
"
```

✓ **look for**

```
discovered: True
8 [0.0, 0.39223227, 0.39223227, 0.0, 0.39223227, 0.58834841, 0.19611614, 0.39223227]
```

No registration call, no import in the framework, no configuration file listing you. The entry point
*is* the registration. `ProviderRegistry` loads the class by name and reads its profile off the class
object — without importing whatever heavy dependency your Provider would pull at first use.

Now see what the trigrams buy. Compare your Embedder with the in-core reference on three pairs:

```bash
uv run python -c "
from latence_core.capability import ProviderRegistry

def cosine(a, b):
    return round(sum(x * y for x, y in zip(a, b)), 3)

registry = ProviderRegistry()
pairs = [('Globex Corp', 'Globex Corporation'), ('Alice Turner', 'Alice Turnr'), ('Globex Corp', 'Alice Turner')]
for name in ('embedding.hashing', 'embedding.acme_chargram'):
    embedder = registry.load(name)({'dimension': 256})
    print(name)
    for left, right in pairs:
        a, b = embedder.embed([left, right])
        print(f'  {left!r:16} ~ {right!r:22} {cosine(a, b)}')
"
```

✓ **look for**

```
embedding.hashing
  'Globex Corp'    ~ 'Globex Corporation'   0.5
  'Alice Turner'   ~ 'Alice Turnr'          0.5
  'Globex Corp'    ~ 'Alice Turner'         0.0
embedding.acme_chargram
  'Globex Corp'    ~ 'Globex Corporation'   0.728
  'Alice Turner'   ~ 'Alice Turnr'          0.801
  'Globex Corp'    ~ 'Alice Turner'         0.08
```

The reference scores both variant pairs at exactly 0.5: one of two tokens matches, and that is all
it can see. Yours scores them higher and still keeps the unrelated pair near zero. It is also still
only a surface measure. It knows nothing about meaning, and two different organisations with
similar names will look alike to it. That is a property to know before you make it a merge signal.

---

## 5. Pass conformance

The Provider Conformance suite is the enterprise-readiness gate: a fixed contract, C1–C6, that every
Provider must satisfy. "Passes conformance" *is* the definition of enterprise-ready in this project.

| Check | Asserts |
| --- | --- |
| **C1** valid typed records | On the canonical input, contract-valid output. For a record-producing Capability: Provenance + Classification present, offsets in range and well ordered, and a sub-document carrier resolving its span to the page its own offsets land on. For an Embedder: one vector per text, each of the expected width. |
| **C2** graceful failure | On adversarial input, a **typed** `LatenceError` or your Capability's defined graceful path — never a bare crash, hang, or silent-wrong output. The message must carry no PII or record content. |
| **C3** license recorded | A profile is present (missing → **FAIL**), evidence-bearing, internally consistent, and permissive-or-explicitly-opted-in. |
| **C4** determinism-or-documented | `deterministic=True` ⇒ two runs are byte-identical. Declaring `True` and not being is a **FAIL**; declaring `False` is recorded without assertion. |
| **C5** resource/device honored | A `compute="gpu"` Provider on a CPU-only host is skipped-with-flag, never faked. No GPU import at module load. |
| **C6** no secret/PII leak | A non-Redactor surfaces no raw PII in non-content record fields, logs, or spans. |

Run it against your Provider:

```bash
uv run python -c "
from latence_core.capability import ProviderRegistry
from latence_core.conformance import run_conformance
res = run_conformance(ProviderRegistry(), 'embedding.acme_chargram')
print(f'{res.provider} -> {res.capability}')
for c in res.checks:
    print(f'  {c.check:22} {c.status.value:8} {c.detail}')
"
```

✓ **look for** — six checks, all `pass`:

```
embedding.acme_chargram -> embedding
  C3_license             pass     permissive: Apache-2.0 (verified=True)
  C5_device              pass     device honored: cpu (provider is cpu-only)
  C1_valid_records       pass     3 contract-valid records
  C2_graceful_failure    pass     graceful (empty), no PII in surfaced text
  C4_determinism         pass     byte-identical over two runs
  C6_no_pii_leak         pass     no raw PII in non-content record fields
```

You wrote no test fixtures. The suite supplied the canonical input, the adversarial input and the
validity predicate, all from the Embedding `ConformanceCase` it mapped you to by name prefix.

### Make it your package's own gate

`../acme-embedder/tests/test_conformance.py`

```python
"""The enterprise-readiness gate for this package's Provider."""

from latence_core.capability import ProviderRegistry
from latence_core.conformance import run_conformance


def test_provider_passes_conformance() -> None:
    result = run_conformance(ProviderRegistry(), "embedding.acme_chargram")
    failed = [c for c in result.checks if c.status.value == "fail"]
    assert not failed, [(c.check, c.detail) for c in failed]
```

```bash
LATENCE_CUDA=0 PYTHONHASHSEED=0 uv run pytest ../acme-embedder/tests -o addopts="" -q
```

✓ **look for** — `1 passed`.

!!! warning "Do not run the framework's own conformance test file to check your Provider"
    `packages/latence-core/tests/test_conformance.py` parametrizes over a roster **derived from the
    workspace's package manifests**, and it asserts that the installed set matches that roster
    exactly. Your out-of-tree Provider is installed but not declared by any workspace package, so
    it shows up as `installed-but-undeclared` and the roster test fails:

    ```
    AssertionError: roster mismatch; declared-but-not-installed={...},
    installed-but-undeclared=['embedding.acme_chargram']
    ```

    That is the gate working correctly — it exists so a Provider can never go silently unconformed
    inside the workspace. Your Provider is not inside the workspace, so it carries its own gate, as
    above. If you are *contributing* a Provider to this repo, then it does belong in a
    `packages/latence-*` package, its entry point makes it part of the roster automatically, and
    that test becomes yours to keep green.

A larger roll-up is available programmatically when you ship several Providers:

```python
from latence_core.capability import ProviderRegistry
from latence_core.conformance import build_conformance_report

report = build_conformance_report(ProviderRegistry(), ["embedding.acme_chargram"])
assert report.all_passed
```

The `ConformanceReport` is counts-only — `providers_total`, `providers_passed`, `providers_failed`,
`providers_stub_scoped`, `providers_device_skipped` plus the per-Provider results — so it is safe to
persist and share, and a run's `QualityReport` has a `conformance` field that can carry it.

---

## 6. Break it three ways

A gate you have never seen fail is a gate you do not trust. Add a second module with two deliberately
broken Providers and register them temporarily:

```python
# ../acme-embedder/src/acme_embedder/broken.py
import random

from latence_core.providers.profile import ProviderProfile


class NoProfileEmbedder:
    """Declares no ProviderProfile at all."""

    dimension = 64

    def __init__(self, config=None): ...

    def embed(self, texts):
        for text in texts:
            yield [float(len(text))] + [0.0] * 63


class FlakyEmbedder:
    """Claims determinism it does not have."""

    dimension = 64

    @classmethod
    def profile(cls):
        return ProviderProfile(
            compute="cpu", memory_mb=8, model_id=None, license="MIT",
            license_verified=True, license_code="MIT", license_weights=None,
            license_source="https://example.com/LICENSE", license_verified_on="2026-11-10",
            deterministic=True, batch=True, cost_per_1k=None,
        )

    def __init__(self, config=None): ...

    def embed(self, texts):
        for _ in texts:
            yield [random.random() for _ in range(64)]
```

```toml
[project.entry-points."latence.providers"]
"embedding.acme_chargram"  = "acme_embedder.provider:CharGramEmbedder"
"embedding.acme_noprofile" = "acme_embedder.broken:NoProfileEmbedder"
"embedding.acme_flaky"     = "acme_embedder.broken:FlakyEmbedder"
```

```bash
uv pip install -e ../acme-embedder
uv run python -c "
from latence_core.capability import ProviderRegistry
from latence_core.conformance import run_conformance
for name in ('embedding.acme_noprofile', 'embedding.acme_flaky'):
    res = run_conformance(ProviderRegistry(), name)
    print(name)
    for c in res.checks:
        print(f'  {c.check:22} {c.status.value:8} {c.detail}')
"
```

✓ **look for** — one failure each, and only the relevant one:

```
embedding.acme_noprofile
  C3_license             fail     provider declares no ProviderProfile (every provider must declare one)
  C5_device              pass     device honored: cpu (auto resolved to cpu (no cuda))
  C1_valid_records       pass     3 contract-valid records
  C2_graceful_failure    pass     graceful (empty), no PII in surfaced text
  C4_determinism         skipped  declared non-deterministic (recorded, no assertion)
  C6_no_pii_leak         pass     no raw PII in non-content record fields
embedding.acme_flaky
  C3_license             pass     permissive: MIT (verified=True)
  C5_device              pass     device honored: cpu (provider is cpu-only)
  C1_valid_records       pass     3 contract-valid records
  C2_graceful_failure    pass     graceful (empty), no PII in surfaced text
  C4_determinism         fail     declared deterministic but two runs differ
  C6_no_pii_leak         pass     no raw PII in non-content record fields
```

Read the two `C4` lines together, because they are the design in miniature. The Provider with no
profile is `skipped` on C4 — no declaration means no claim means nothing to assert. The Provider that
*claimed* determinism and does not have it **fails**. The framework does not require your Provider to
be deterministic. It requires you not to say it is when it is not.

The third break is the one that matters most, and it needs no package at all — mistype the name
prefix, so the suite cannot tell which Capability you meant:

```bash
uv run python -c "
from latence_core.conformance import case_for_provider
case_for_provider('xyzzy.mine')
"
```

✓ **look for** — a raise, not a skip:

```
KeyError: "Provider 'xyzzy.mine' has an unknown Capability prefix 'xyzzy'; declare it on the
Capability's CapabilityDescriptor (or, for a non-Stage seam, in
_NON_STAGE_PREFIX_TO_CAPABILITY) — a Provider the conformance suite cannot classify is a
loud failure, never a silent skip."
```

The same happens one level up: a Provider whose Capability is known but has no `ConformanceCase`
raises rather than passing. A silent skip would be worse than a failure, because it looks like
coverage. Delete `broken.py` and its two entry points, and run `uv pip install -e ../acme-embedder`
again, before continuing.

---

## 7. Wire it into a pipeline

The point of the seam is that using your Provider is a one-line config change. Go back to Tutorial
1's directory, take its stack, and change the embedder on the corpus export:

```yaml
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [redact]
    config:
      basename: records
      embedder: {provider: embedding.acme_chargram, config: {dimension: 256}}   # yours
```

The commands in this section run inside `first-pipeline/`, with the clone's environment activated
as in Tutorial 1. That is the environment your package is installed in.

```bash
uv run latence stack check latence.stack.yaml
uv run latence run latence.stack.yaml --run-id run-0004
```

✓ **look for** — `stack check OK — 0 error(s), 0 warning(s)`, then a completed run. The offline
check inspected *your* class through the registry and confirmed it satisfies the Embedder
Capability, without running anything. Mistype the name and the same check stops you in under a
second:

```
error [sub_provider] stage 'export_corpus': sub-Provider slot 'embedder': unknown Provider 'embedding.acme_chargrm' — did you mean 'embedding.acme_chargram'? (no entry point of that name is registered)
stack check FAILED — 1 error(s), 0 warning(s)
```

Now confirm it actually did the work. `run-0001` is Tutorial 1's run, with the in-core Embedder:

```bash
uv run python -c "
import json

def rows(run):
    path = f'latence-out/_latence/runs/{run}/export/records.jsonl'
    return {r['provenance']['file_name']: r for r in map(json.loads, open(path))}

def cosine(a, b):
    return round(sum(x * y for x, y in zip(a, b)), 3)

old, new = rows('run-0001'), rows('run-0004')
print('same text exported:', all(old[k]['content'] == new[k]['content'] for k in old))
print('vector width      :', len(new['policy.md']['embedding']))
for name, run in (('embedding.hashing', old), ('embedding.acme_chargram', new)):
    print(f'{name:24} policy.md ~ incident.md:', cosine(run['policy.md']['embedding'], run['incident.md']['embedding']))
"
```

✓ **look for** — the same exported text, vectors of the width you asked for, and a different
similarity between the same two documents:

```
same text exported: True
vector width      : 256
embedding.hashing        policy.md ~ incident.md: 0.325
embedding.acme_chargram  policy.md ~ incident.md: 0.569
```

Two things to take from that. First: it worked — one line of YAML put your code into the run, and
nothing else in the export moved. Second, and more useful: **look at the two similarities.** The
same pair of documents is 0.325 under one Embedder and 0.569 under the other, and neither number
is wrong. They are two different scales. A threshold tuned on one Embedder means nothing on the
other.

That is now a real engineering decision you own. The same `embedder` block is what
`disambiguation.embedding` reads, and there the similarity decides whether two mentions merge
into one entity. Swap an Embedder under a resolver without re-measuring its threshold and you
have changed your graph. The framework's job was to make the consequence legible, and it did:
the run's `manifest.json` records which Embedder each stage was given.

Put the `embedder` line back to `embedding.hashing` before moving on, so the next tutorial's
output matches.

---

## 8. When your Provider is bigger than this

Everything above is the complete mechanism. Three things scale it up.

**`AdapterBase`** is an *optional* base class that factors out the glue every real adapter repeats —
device resolution against your declared profile, batching, native-exception mapping, and provenance
/ offset carry. It is not a new protocol layer and it names no Capability method; the shipping
in-core Providers satisfy their Capabilities without it. Its four helpers:

| Helper | Does |
| --- | --- |
| `self.device` | The `DeviceDecision` resolved once, at construction, from `config["device"]` (`auto`/`cpu`/`cuda`, default `auto`) and your profile. A `compute="gpu"` adapter on a CPU-only host sees the **skip** decision, never a fabricated device. An unknown device string is a `ConfigError`. Placement stays your job. |
| `self.batched(items, run_batch)` | Groups the input into `config["batch_size"]` chunks (default 32) and calls your `run_batch` once per chunk. With `batch=False` in your profile, or no profile, it calls `run_batch` with one item at a time. Outputs are yielded in the order `run_batch` returns them; the input is consumed one batch at a time. A `batch_size` that is not an int of at least 1 is a `ConfigError`. |
| `with self.guard("<op>"):` | Turns a native exception into a typed `LatenceError`: a `ConfigError` when its type is listed in your class's `config_exceptions` (empty by default), a `ProviderError` otherwise. A `LatenceError` raised inside the block passes through unchanged. The message carries the provider, the op and the native exception's **type name** — never its message, which could echo record content. |
| `carry_provenance` / `rebase_offsets` / `resolve_pages` | `carry_provenance` copies the source's Provenance, overriding only the offset and page fields you pass, and returns its Classification unchanged. `rebase_offsets` turns a chunk-local span into document offsets through the chunk's `OffsetMap` (`OffsetIndex`), or by adding the chunk's start when there is no map. `resolve_pages` resolves a document span to pages through `PageIndexResolver`, and raises `PageSliceMissingError` when the chunk carries no page spans. |

Full walkthrough: [Writing an adapter with `AdapterBase`](../writing-an-adapter.md). The shipping
template to copy is [`parser.pdfplumber`](../reference/example-provider.md), the first Wave-1 adapter
built through it.

**Offsets and pages** are where a sub-document Provider — a mention, a relation — earns or loses
trust, and the rules are exact. Since schema v19 every `ChunkRecord` carries its own `page_slice`, so
one `PageIndexResolver` per call resolves any chunk in any order with no per-document state. Do not
read the page field off the record and do not build your own page index: the seam owns the end clamp
and the drift accounting. A relation's covering span runs between two mentions that may live in
different chunks, so it is answered by `pages_for_covering_span(head.provenance, tail.provenance)`
instead.

**Proving it beats the incumbents** is what `latence bake-off` is for: hold every other stage
constant, run the same bundled corpus through each candidate for one stage, and read one table of
quality / latency / memory / license / determinism columns. Write a matrix file naming your
candidate beside the references — `matrix/chunk.yaml` in the repo is a two-line example, and
`matrix/embed.yaml` is the one for Embedders — and:

```bash
uv run latence bake-off matrix/<stage>.yaml
```

---

## The loop, in one line

Implement one method. Declare an honest profile. Register an entry point. Pass conformance. Win the
bake-off.

**Next:** [Production](05-production.md) — served GPU models, staged runs, checkpoint/resume, and
incremental corpus deltas.
