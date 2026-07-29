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

---

## 1. Pick a Capability that genuinely varies

The Capabilities you can fulfil: `Source`, `Parser`, `Chunker`, `IntakeScreener`,
`ContentScreener`, `LabelInducer`, `EntityExtractor`, `RelationExtractor`,
`FusedEntityRelationExtractor`, `PIIDetector`, `Profiler`, `TypeConsolidator`, `Disambiguator`,
`GraphAssembler`, `GraphCompleter`, `GraphFeatureComputer`, `ContextEnricher`, `DeltaProcessor`,
`Export` — plus the non-Stage signal seams `Embedder`, `SparseEmbedder`, `MultiVectorEmbedder`,
`FdeConverter` and `Tokenizer`. Each is a `runtime_checkable` Protocol in
[`latence_core.capability`](../reference/capability.md).

We will build a **Tokenizer**, because it is small enough to read in one screen and because it is a
seam that genuinely matters rather than a toy. BM25's whole statistical apparatus — document
frequency, term frequency, document length, average document length, the Robertson idf — is fixed
math. Exactly one thing varies: **how text is split into terms.** That is the Capability:

```python
class Tokenizer(Protocol):
    def tokenize(self, texts: Iterable[str]) -> Iterator[list[str]]:
        """One token list per input text, in input order."""
```

One method. The contract on top of it: terms in occurrence order, empty or term-free text yields
`[]` rather than crashing, and a deterministic Tokenizer is byte-reproducible.

Two references ship in core, and they are deliberately *different* rather than one plus a rename:
`tokenizer.regex` folds to lowercase Unicode alphanumeric runs (punctuation dropped);
`tokenizer.whitespace` splits only on whitespace and keeps punctuation attached, so `"foo,"` and
`"foo"` are distinct terms. Your Provider will be a third: a **suffix-folding** tokenizer, so that
`contract` and `contracts` share one BM25 term instead of competing for the same query.

---

## 2. Scaffold the package

Outside the framework clone:

```bash
mkdir -p acme-tokenizer/src/acme_tokenizer acme-tokenizer/tests && cd acme-tokenizer
```

`pyproject.toml`

```toml
[project]
name = "acme-tokenizer"
version = "0.1.0"
description = "A BM25 Tokenizer that folds English plural suffixes."
requires-python = ">=3.11"
dependencies = ["latence-core"]

[project.entry-points."latence.providers"]
"tokenizer.acme_suffix" = "acme_tokenizer.provider:SuffixFoldingTokenizer"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/acme_tokenizer"]
```

The entry-point block is the whole registration mechanism. The key is the Provider **name**, and its
shape is load-bearing: `<capability>.<technique>`. The prefix before the dot is how the conformance
suite maps your Provider to the Capability it must be checked against, so `tokenizer.acme_suffix` is
auto-covered by the Tokenizer case the moment it is installed. Get the prefix wrong and the suite
will tell you loudly rather than skip you.

Ship it as **its own package**, so whatever heavy dependency it needs stays out of `latence-core`
([ADR-0016](../adr/0016-thin-core-per-provider-packages.md)). This one needs nothing but the
standard library.

---

## 3. Implement the Capability

`src/acme_tokenizer/provider.py`

```python
"""A BM25 Tokenizer that folds a few English suffixes so plurals share one term."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import Any

from latence_core.providers.profile import ProviderProfile

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_SUFFIXES = ("ies", "es", "s")


class SuffixFoldingTokenizer:
    """Lowercase Unicode word runs, with a deterministic English plural fold.

    ``contracts`` and ``contract`` become one BM25 term; ``policies`` and ``policy`` likewise.
    ``min_stem`` guards short words so ``is`` and ``gas`` survive intact.
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
            license_source="https://example.com/acme-tokenizer/blob/main/LICENSE",
            license_verified_on="2026-07-29",
            deterministic=True,
            batch=True,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._min_stem = int((config or {}).get("min_stem", 4))

    def _fold(self, token: str) -> str:
        for suffix in _SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= self._min_stem:
                stem = token[: -len(suffix)]
                return stem + "y" if suffix == "ies" else stem
        return token

    def tokenize(self, texts: Iterable[str]) -> Iterator[list[str]]:
        for text in texts:
            yield [self._fold(token) for token in _WORD.findall(text.lower())]
```

Note what is *not* there. No base class. No `latence_core` import except the profile type — and even
that is only because the profile is a typed object; the Capability itself is satisfied structurally.
The class does not know it is a plugin.

Two conventions it does follow, and both matter:

* **`__init__(self, config)`** — every Provider is constructed from one config mapping, so a stack
  YAML's `config:` block reaches it uniformly. Accept it even when you have no knobs.
* **Streaming** — `tokenize` is a generator over the input iterable. Providers are expected not to
  buffer a whole corpus; the Runner streams, and so should you
  ([ADR-0033](../adr/0033-streaming-memory-bounded-runner.md)).

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
  where the license was checked — HF model-card / GitHub LICENSE / repo LICENSE).
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

```bash
uv pip install -e .
```

(From inside the framework's environment — `uv pip install -e /path/to/acme-tokenizer` from the
framework clone works too.)

```bash
uv run python -c "
from latence_core.capability import ProviderRegistry
registry = ProviderRegistry()
print('discovered:', 'tokenizer.acme_suffix' in registry.names())
cls = registry.load('tokenizer.acme_suffix')
print(list(cls({'min_stem': 4}).tokenize(['Policies and contracts, invoiced.'])))
"
```

✓ **look for**

```
discovered: True
[['policy', 'and', 'contract', 'invoiced']]
```

No registration call, no import in the framework, no configuration file listing you. The entry point
*is* the registration. `ProviderRegistry` loads the class by name and reads its profile off the class
object — without importing whatever heavy dependency your Provider would pull at first use.

---

## 5. Pass conformance

The Provider Conformance suite is the enterprise-readiness gate: a fixed contract, C1–C6, that every
Provider must satisfy. "Passes conformance" *is* the definition of enterprise-ready in this project.

| Check | Asserts |
| --- | --- |
| **C1** valid typed records | On the canonical input, contract-valid output: Provenance + Classification present, offsets in range and well ordered. A sub-document carrier must resolve its span to the page its own offsets land on. |
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
res = run_conformance(ProviderRegistry(), 'tokenizer.acme_suffix')
print(f'{res.provider} -> {res.capability}')
for c in res.checks:
    print(f'  {c.check:22} {c.status.value:8} {c.detail}')
"
```

✓ **look for** — six checks, all `pass`:

```
tokenizer.acme_suffix -> tokenizer
  C3_license             pass     permissive: Apache-2.0 (verified=True)
  C5_device              pass     device honored: cpu (provider is cpu-only)
  C1_valid_records       pass     3 contract-valid records
  C2_graceful_failure    pass     graceful (empty), no PII in surfaced text
  C4_determinism         pass     byte-identical over two runs
  C6_no_pii_leak         pass     no raw PII in non-content record fields
```

You wrote no test fixtures. The suite supplied the canonical input, the adversarial input and the
validity predicate, all from the Tokenizer `ConformanceCase` it mapped you to by name prefix.

### Make it your package's own gate

`tests/test_conformance.py`

```python
"""The enterprise-readiness gate for this package's Provider."""

from latence_core.capability import ProviderRegistry
from latence_core.conformance import run_conformance


def test_provider_passes_conformance() -> None:
    result = run_conformance(ProviderRegistry(), "tokenizer.acme_suffix")
    failed = [c for c in result.checks if c.status.value == "fail"]
    assert not failed, [(c.check, c.detail) for c in failed]
```

```bash
LATENCE_CUDA=0 PYTHONHASHSEED=0 uv run pytest tests -o addopts="" -q
```

✓ **look for** — `1 passed`.

!!! warning "Do not run the framework's own conformance test file to check your Provider"
    `packages/latence-core/tests/test_conformance.py` parametrizes over a roster **derived from the
    workspace's package manifests**, and it asserts that the installed set matches that roster
    exactly. Your out-of-tree Provider is installed but not declared by any workspace package, so
    it shows up as `installed-but-undeclared` and the roster test fails:

    ```
    AssertionError: roster mismatch; declared-but-not-installed={...},
    installed-but-undeclared=['tokenizer.acme_suffix']
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

report = build_conformance_report(ProviderRegistry(), ["tokenizer.acme_suffix"])
assert report.all_passed
```

The `ConformanceReport` is counts-only — `providers_total`, `providers_passed`, `providers_failed`,
`providers_stub_scoped`, `providers_device_skipped` plus the per-Provider results — so it is safe to
persist and share, and it folds into a run's `QualityReport.conformance`.

---

## 6. Break it three ways

A gate you have never seen fail is a gate you do not trust. Add a second module with two deliberately
broken Providers and register them temporarily:

```python
# src/acme_tokenizer/broken.py
import random

from latence_core.providers.profile import ProviderProfile


class NoProfileTokenizer:
    """Declares no ProviderProfile at all."""

    def __init__(self, config=None): ...

    def tokenize(self, texts):
        for text in texts:
            yield text.lower().split()


class FlakyTokenizer:
    """Claims determinism it does not have."""

    @classmethod
    def profile(cls):
        return ProviderProfile(
            compute="cpu", memory_mb=8, model_id=None, license="MIT",
            license_verified=True, license_code="MIT", license_weights=None,
            license_source="https://example.com/LICENSE", license_verified_on="2026-07-29",
            deterministic=True, batch=True, cost_per_1k=None,
        )

    def __init__(self, config=None): ...

    def tokenize(self, texts):
        for text in texts:
            tokens = text.lower().split()
            random.shuffle(tokens)
            yield tokens
```

```toml
[project.entry-points."latence.providers"]
"tokenizer.acme_suffix"    = "acme_tokenizer.provider:SuffixFoldingTokenizer"
"tokenizer.acme_noprofile" = "acme_tokenizer.broken:NoProfileTokenizer"
"tokenizer.acme_flaky"     = "acme_tokenizer.broken:FlakyTokenizer"
```

```bash
uv pip install -e .
uv run python -c "
from latence_core.capability import ProviderRegistry
from latence_core.conformance import run_conformance
for name in ('tokenizer.acme_noprofile', 'tokenizer.acme_flaky'):
    res = run_conformance(ProviderRegistry(), name)
    print(name)
    for c in res.checks:
        print(f'  {c.check:22} {c.status.value:8} {c.detail}')
"
```

✓ **look for** — one failure each, and only the relevant one:

```
tokenizer.acme_noprofile
  C3_license             fail     provider declares no ProviderProfile (every provider must declare one)
  C5_device              pass     device honored: cpu (auto resolved to cpu (no cuda))
  C1_valid_records       pass     3 contract-valid records
  C2_graceful_failure    pass     graceful (empty), no PII in surfaced text
  C4_determinism         skipped  declared non-deterministic (recorded, no assertion)
  C6_no_pii_leak         pass     no raw PII in non-content record fields

tokenizer.acme_flaky
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
KeyError: Provider 'xyzzy.mine' has an unknown Capability prefix 'xyzzy'; declare it on the
Capability's CapabilityDescriptor (or, for a non-Stage seam, in
_NON_STAGE_PREFIX_TO_CAPABILITY) — a Provider the conformance suite cannot classify is a
loud failure, never a silent skip.
```

The same happens one level up: a Provider whose Capability is known but has no `ConformanceCase`
raises rather than passing. A silent skip would be worse than a failure, because it looks like
coverage. Delete `broken.py` and its two entry points before continuing.

---

## 7. Wire it into a pipeline

The point of the seam is that using your Provider is a one-line config change. Take Tutorial 1's
stack and change the BM25 tokenizer:

```yaml
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [enrich]
    config:
      basename: records
      context_columns: true
      embedder: {provider: embedding.hashing, config: {dimension: 256}}
      bm25: {provider: tokenizer.acme_suffix, config: {min_stem: 4}}   # yours
```

```bash
uv run latence stack check latence.stack.yaml
uv run latence run latence.stack.yaml --run-id run-0004
```

✓ **look for** — `stack check OK — 0 error(s), 0 warning(s)`, then a completed run. The offline
check inspected *your* class through the registry and confirmed it satisfies the Tokenizer
Capability, without running anything.

Now confirm it actually did the work:

```bash
uv run python -c "
import json
old = {t['term'] for t in json.load(open('latence-out/_latence/runs/run-0003/export/bm25-stats.json'))['terms']}
new = json.load(open('latence-out/_latence/runs/run-0004/export/bm25-stats.json'))
print('tokenizer recorded in the artifact:', new['tokenizer'])
new_terms = {t['term'] for t in new['terms']}
print('only with tokenizer.regex:', sorted(old - new_terms))
print('only with yours          :', sorted(new_terms - old))
"
```

✓ **look for** — the artifact naming your Provider, and a vocabulary that visibly folded:

```
tokenizer recorded in the artifact: tokenizer.acme_suffix
only with tokenizer.regex: ['approves', 'employees', 'exceptions', 'invoices', 'questions',
                            'receipts', 'reimburses', 'reyes', 'services', 'works']
only with yours          : ['approv', 'employe', 'exception', 'invoic', 'question',
                            'receipt', 'reimburs', 'reye', 'servic', 'work']
```

Two things to take from that list. First: it worked — `invoices` and `invoice` now share a term, so
a query for either finds both. Second, and more useful: **look at `reyes` → `reye`.** Your Provider
just stemmed a surname, and `employees` became `employe` rather than `employee`. A naive suffix
stripper does that. It is visible here because the artifact records which Provider produced it and
because the vocabulary is a plain, diffable file — which is exactly the kind of thing that is
invisible when tokenisation is buried inside a search engine.

That is now a real engineering decision you own: guard proper nouns, use a real stemmer, or accept
the noise. The framework's job was to make the consequence legible, and it did.

---

## 8. When your Provider is bigger than this

Everything above is the complete mechanism. Three things scale it up.

**`AdapterBase`** is an *optional* base class that factors out the glue every real adapter repeats —
device resolution against your declared profile, batching, native-exception mapping, and provenance
/ offset carry. It is not a new protocol layer and it names no Capability method; the shipping
in-core Providers satisfy their Capabilities without it. Its four helpers:

| Helper | Does |
| --- | --- |
| `self.device` | Resolves `config["device"]` (`auto`/`cpu`/`cuda`) against your profile. A `compute="gpu"` adapter on a CPU-only host sees the **skip** decision, never a fabricated device. Placement stays your job. |
| `self.batched(items, run_batch)` | Groups the input into `config["batch_size"]` chunks and calls your `run_batch`, degrading to one-item calls when your profile says `batch=False`. Deterministic input-order output; the input is consumed one batch at a time. |
| `with self.guard("<op>"):` | Turns a native exception into the right typed `LatenceError`. The message carries the provider, the op and the native exception's **type name** — never its message, which could echo record content. |
| `carry_provenance` / `rebase_offsets` / `resolve_pages` | Map a native library's output back onto the input's Provenance and true document offsets, reusing the `OffsetIndex` and `PageIndexResolver` seams rather than a hand-rolled mapper. |

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
candidate beside the references — `matrix/chunk.yaml` in the repo is a two-line example — and:

```bash
uv run latence bake-off matrix/<stage>.yaml
```

---

## The loop, in one line

Implement one method. Declare an honest profile. Register an entry point. Pass conformance. Win the
bake-off.

**Next:** [Production](05-production.md) — served GPU models, staged runs, checkpoint/resume, and
incremental corpus deltas.
