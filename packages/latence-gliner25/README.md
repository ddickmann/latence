# latence-gliner25

**GLiNER 2.5** Providers for the [Latence framework](https://github.com/ddickmann/latence-framework),
over the [fastino GLiNER2](https://github.com/fastino-ai/GLiNER2) library at its **2.x** major
(`gliner2>=2.0,<3`, released 2026-08-24).

One package, **two** entry points:

| Entry point | Capability | What it does |
| --- | --- | --- |
| `fused_entity_relation.gliner25` | Entity + Relation Extraction (S4/S5) | fused zero-shot NER **and** RE in one pass |
| `redaction.gliner25` | Redaction (S6) | **zero-shot** PII detection — see the honesty note below |

Both drive `fastino/gliner2.5-multi-v1` (287M, mDeBERTa-v3-base, multilingual, Apache-2.0). Its
card advertises a 4096-token window; a real run showed that is **not** a window of reliable
extraction — see [below](#the-advertised-4096-window-is-not-a-window-of-reliable-extraction), it is
the most important thing on this page.

```yaml
- name: chunk
  capability: chunk
  provider: chunk.page          # one chunk per source page, else a 1024-token window
  config:
    max_tokens: 1024

- name: extract
  capability: fused_entity_relation
  provider: fused_entity_relation.gliner25
  config:
    labels: [organization, person]
    relation_labels: [works for, partner of]
    threshold: 0.3
    max_len: 1280               # the DEFAULT, derived from the 1024-token budget. Read the
                                # window section below before raising it.
    # microbatch_max_texts derives from max_len — a wider window buys its positions back

- name: redact
  capability: redaction
  provider: redaction.gliner25
  config:
    pii_labels: [person, email, phone number, iban, credit card number]
    max_len: 1280
```

## Why one package rather than two

GLiNER 2.5 is a **single checkpoint family** serving both Capabilities over **one** dependency pin,
**one** compatibility gate and **one** loader seam. Splitting it in two would duplicate the pin, the
gate and the loader across two distributions — and make it possible to install half a family, with
the two halves free to drift onto different `gliner2` minors. The sibling 1.x Providers are two
packages for a historical reason (they were built a release apart, against two different
checkpoints); that is not a precedent to copy. Recorded in
[ADR-0064](https://ddickmann.github.io/latence-framework/adr/0064-gliner-2-5-is-a-new-provider-package-behind-an-enforced-gliner2-major-boundary/).

## Same record contract, different engine — measured

The two families produce the **identical** record shape. That was not read off a docstring; both
were run. Cross-major A/B on a pod (RTX 2000 Ada 16 GB, driver 580.159.04, **2026-08-25**): gliner2
**1.3.2** + `fastino/gliner2-multi-v1` (span head) and gliner2 **2.0.0** +
`fastino/gliner2.5-multi-v1` (boundary head), in two venvs, on the same text with the same labels,
under identical transformers 4.57.6 / torch 2.4.1+cu124 — diffing key sets, types and nesting:

| field | verdict |
| --- | --- |
| `entities_signature` · `relations_signature` | **SAME** |
| `entity_row_keys` `["confidence","end","start","text"]` | **SAME** |
| `relation_row_keys` `["head","tail"]` | **SAME** |
| `relation_endpoint_keys` `["confidence","end","start","text"]` | **SAME** |
| top-level envelopes `["entities"]` · `["relation_extraction"]` | **SAME** |
| `offsets_are_ints` | **SAME** |
| `head_class` | **DIFF** — `GLiNER2` → `BoundaryExtractor` |
| `gliner2_version` | **DIFF** — `1.3.2` → `2.0.0` |

Every structural field is identical; the only two that differ name which model is which. So
swapping `fused_entity_relation.gliner2` → `fused_entity_relation.gliner25` in a stack changes the
engine and **nothing** about the records that come out — same mentions, relations, evidence,
provenance and offsets. The key sets are pinned in `latence_core.conformance.stub` and every offline
fake in the workspace is asserted against them, so a future gliner2 release that moves one turns
tests red rather than drifting quietly.

What is **not** interchangeable is the loader and the checkpoints — which is the next section, and
the reason this is a new package rather than a version bump.

## It cannot be installed beside `latence-extract-gliner2` / `latence-pii-gliner2`

That is deliberate and it is **enforced**, not documented. The inference contract carries across the
major (above); the *engine* does not. `gliner2` 2.0 is a major release:

- the 2.5 checkpoints are the new **`boundary`** architecture
  (`gliner2.models.boundary.engine.BoundaryExtractor`); every 1.x checkpoint is **`span`**;
- `AutoExtractor.from_pretrained` dispatches on the checkpoint's saved architecture and raises
  `ArchitectureMismatchError` rather than converting — *"Span and boundary heads are not
  checkpoint-compatible"*;
- the pins are therefore disjoint: `>=2.0,<3` here, `>=1.3,<2` there.

One resolved environment holds **one** `gliner2` major. Three things enforce it, and each is
checked:

1. the package **pin** — a resolver refuses the pair outright;
2. `deploy/env/*/environment.yaml` declares `gliner2>=1.3,<2` as an **enforced conflict boundary**
   on every environment that installs a 1.x Provider, so adding this package to one of them fails
   `latence env verify` with `E-CONSTRAINT-CONFLICT` — offline, in `scripts/verify-local.sh`, never
   at model load on a pod;
3. `latence_gliner25._compat` is the second gate for a hand-installed environment the resolver never
   saw: an unsupported `gliner2`, a missing `AutoExtractor`, or a `span` checkpoint pointed at these
   Providers is a **typed `ProviderError`** naming the supported range and the Provider that *does*
   load it — never a raw `AttributeError`.

## The PII claim, stated plainly

**There is no GLiNER 2.5 PII or guardrails checkpoint.** As of 2026-08-25 the fastino PII models
(`GLiNER2-Guardrails-PII-Multi`, `gliner2-privacy-filter-PII-multi`) are still the 2.x-era **span**
checkpoints that `redaction.gliner2` drives. So `redaction.gliner25` asks the **general** 2.5
checkpoint for PII labels **zero-shot**.

- No parity with the task-trained `redaction.gliner2` checkpoint is claimed, and **no recall number
  for this Provider has been measured**. That measurement is a separate step; see
  `docs/running-the-benchmarks.md` for where the numbers will go and what produces them.
- `guardrails: true` is **refused** on the default checkpoint. A guardrail screen is a security
  control, and running one against a model with no guardrail training would produce a control that
  reports nothing while looking enabled. Use the deterministic `screening.content_keyword` Stage,
  and/or `redaction.gliner2`. It is accepted with a warning on an operator-supplied `model`.
- The **ADR-0044 universal financial-PII floor** is unaffected either way: a deterministic
  recogniser for `ssn` / `credit_card` / `iban` sits beneath every Redaction Provider at the shared
  `finalize_redacted_chunk` seam, non-bypassable by backend. That floor is why a zero-shot PII
  Provider is shippable rather than merely plausible.

## The advertised 4096 window is NOT a window of reliable extraction

Measured on a pod GPU (RTX 2000 Ada 16 GB, driver 580.159.04, torch 2.4.1+cu124, transformers
4.57.6, gliner2 2.0.0, `fastino/gliner2.5-multi-v1`, **2026-08-25**). One call at `max_len=4096`
over a document seeded with a repeated PII sentence; `coverage` is how far into the document the
returned spans reach:

| tokens | pii_spans | coverage |
| ---: | ---: | ---: |
| 674 | 24 | **100%** |
| 1262 | 30 | **99%** |
| 1850 | 27 | 90% |
| 2438 | 13 | **34%** |
| 3026 | 25 | 72% |
| 4874 | 32 | 55% |
| 7562 | 32 | 35% |

At 7,562 tokens the document holds **90 SSNs** and the model returns **32 spans covering the first
35%** of the text — with no error and a well-formed result. At 12,612 tokens with one label it
returned **zero** spans, silently. Longer inputs OOM'd a 16 GB card inside DeBERTa's quadratic
attention.

The filler is synthetic repetitive prose, so the percentages are **directional, not a benchmark**.
The failure MODE is not in doubt: overflow is **silent, partial and non-monotonic** (2,438 tokens
scored worse than 3,026), which is worse than a hard truncation because no threshold test catches it
reliably.

**What this means for you.** The reliable chunk budget is **1024 tokens** — that is the decided
default for `chunk.page`'s windowed fallback and for the gliner2.5 wizard profile — and the
Providers' `max_len` is derived from it. `4096` remains the ceiling you *may* raise into, and doing
so is warned about on extraction and **refused** on redaction against the default checkpoint,
because a Redactor in that regime emits "redacted" chunks that still contain PII with nothing in the
run reporting it. A chunk too long for the window is refused outright rather than silently
truncated, naming the budget knob to lower.

Separately, raising `max_len` does **not** widen the model forward. gliner2 pads every text in a micro-batch to
the batch's longest sequence, so one forward's padded activation is `microbatch_max_texts × max_len`
positions. The micro-batch text cap is therefore **derived from the window**, holding that product
inside the same envelope the 768-window default implies. An explicit `microbatch_max_texts` always
wins — an operator with a measurement outranks a derivation.

No throughput number is claimed for this family: the boundary architecture's peak has never been
measured on hardware.

## Two things a real load needed that nothing declared

Both found by running the model, not by reading the wheel (2026-08-25):

- **the tokenizer-config shim.** The 2.5 checkpoints store `extra_special_tokens` as a **list** in
  `tokenizer_config.json`, and `transformers >= 4.57` dies on it with `AttributeError: 'list'
  object has no attribute 'keys'` — the load never reaches a forward. The framework already had a
  guarded, idempotent shim for the same failure on a 1.x checkpoint; it now lives in the shared
  seam (`latence_core.providers.perf.coerce_list_extra_special_tokens`) and both packages call it.
  You need do nothing: it is applied before every load here.
- **`protobuf>=7.36,<8` and `sentencepiece>=0.2.2,<0.3`.** `gliner2[local]` declares neither, and
  the DeBERTa-v2 tokenizer path needs both, so this package declares them — with the same
  major-boundary cap every other pin here carries, because both arrive through a tokenizer path
  this package does not control. Licences read from the distributions' own metadata on 2026-08-25:
  BSD-3-Clause and Apache-2.0 respectively. `peft` is **not** carried forward — 2.0.0 declares it
  under the `local` extra, unlike 1.3.2.

## Licensing (ADR-0012 — weights AND code, verified separately)

- **`gliner2` library**: **Apache-2.0** (code). The 2.0.0 wheel's `METADATA` carries
  `License-Expression: Apache-2.0` and bundles `LICENSE` — read from the wheel on **2026-08-25**.
- **`fastino/gliner2.5-multi-v1`**: **Apache-2.0** (weights), per the Hugging Face model card,
  ungated — recorded **2026-08-25**.

Both permissive; nothing downloads unless a Pipeline configures one of these Providers. The heavy
`gliner2[local]` stack ships with this package, never with the thin `latence-core` (ADR-0016).

## Install

```bash
pip install latence-gliner25
```

It is **excluded from the uv dev workspace** (`[tool.uv.workspace] exclude`) for the same reason it
cannot share an environment with the 1.x Providers: a workspace is one resolved environment, and
`uv lock` refuses the pair — correctly. Its source, types and tests are covered by the ordinary
`packages/*` gates, unchanged. Three things had to be taught about the exclusion, and all three are
gated: the release builds it by **path** (`uv build "$pkg"`, since `--package <name>` resolves
through the workspace and cannot see it), it carries its **own committed `uv.lock`**, and
`scripts/generate-sbom.sh` folds that lock into the SBOM union so a shipped distribution is never
absent from the release's component inventory.
