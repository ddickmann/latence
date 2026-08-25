# GLiNER 2.5 is a NEW Provider package behind an enforced `gliner2` major boundary

**Status:** Accepted (2026-08-25)

GLiNER 2.5 ships as **one new package**, `packages/latence-gliner25`, registering **two** entry
points — `fused_entity_relation.gliner25` and `redaction.gliner25` — pinned to
`gliner2[local]>=2.0,<3`. That window is **disjoint** from the `gliner2[local]>=1.3,<2` the existing
`latence-extract-gliner2` / `latence-pii-gliner2` declare, and the disjointness is **enforced at
dependency-resolution time in three places**, never discovered at model load. Alongside it, a third
Chunker — `chunk.page` — cuts a document on its own page boundaries when it has a measured page map
and falls back to a 1024-token window when it does not; the wizard selects the whole set as ONE
`engine` answer so a mixed pair cannot be expressed.

## Context

`gliner2` 2.0.0 was published to PyPI on **2026-08-24** (the previous line was 1.3.2, 2026-06-30).
It is a genuine major, and three facts read out of the 2.0.0 wheel decide everything below.

**1. The checkpoint architecture changed.** The GLiNER 2.5 checkpoints — `gliner2.5-multi-v1`
(287M, mDeBERTa-v3-base, multilingual), `-base-v1`, `-small-v1`, all `fastino/`, all Apache-2.0 —
are the new **`boundary`** architecture (`gliner2.models.boundary.engine.BoundaryExtractor`, a
sparse candidate decoder). Every 1.x checkpoint is **`span`**. `AutoExtractor.from_pretrained`
dispatches on the checkpoint's saved architecture and refuses to bridge them:

> "Span and boundary heads are not checkpoint-compatible, and automatic architecture conversion is
> not supported." — `gliner2/auto.py`

**2. The loader changed.** 2.5 is loaded through `gliner2.auto.AutoExtractor`. The legacy `GLiNER2`
class still exists and, by its own docstring, "remains span-only" — so it cannot load a 2.5
checkpoint at all.

**3. The batched surface did NOT change.** Both public model classes mix in
`gliner2.inference.runtime.ExtractorRuntimeMixin`, so `batch_extract_entities` /
`batch_extract_relations` keep the exact signature and the exact return envelopes the 1.x Providers
already drive:

```python
# -> [{"entities": {label: [{"text", "confidence", "start", "end"}, ...]}}, ...]
# -> [{"relation_extraction": {rel_type: [{"head": {...}, "tail": {...}}, ...]}}, ...]
```

That third fact was verified twice over, because the two architectures build those shapes in
different code — the span path through `ExtractorRuntimeMixin._extract_entities` /
`._extract_relations`, the boundary path through `BoundaryExtractor._decode_entities` /
`._decode_relations` — and the shared `format_results` is what folds either into one envelope. **No
Provider-side parsing change was needed.** Three real 2.0.0 deltas exist and none of them changes
the contract: `batch_size` is the third positional parameter (it was in 1.3.2 too, and every
argument is passed by keyword here so it cannot silently become the threshold); a new
`overlap_policy` kwarg, not passed; and new `*_long` variants taking `chunk_size`/`chunk_overlap`,
deliberately unused — the Chunk Stage owns segmentation, and a second model-internal chunker would
produce offsets nothing downstream can ground (ADR-0021/0031).

**4. There is no GLiNER 2.5 PII or guardrails checkpoint.** As of 2026-08-25 the fastino PII models
(`GLiNER2-Guardrails-PII-Multi`, `gliner2-privacy-filter-PII-multi`) are still the 2.x-era **span**
checkpoints. "PII on 2.5" therefore means driving the **general** 2.5 checkpoint with PII labels
**zero-shot**.

**5. The window is 4096** (model card), against the 768 the 1.x Providers default to.

## Decision 1 — a new package, not a version bump of the existing one

`latence-extract-gliner2` keeps `gliner2>=1.3,<2` and keeps its `span` checkpoint; GLiNER 2.5 lands
beside it.

### (a) Bump `latence-extract-gliner2` to `gliner2>=2,<3` — REJECTED

It reads like the smaller change and is the larger one. The library major, the loader class and the
model head all move together, so "a version bump" is a rewrite of the Provider's load path plus a
forced migration of every existing stack to a different checkpoint family — with no way to run the
old one. Worse, it is *irreversible for an adopter*: a pinned `latence-extract-gliner2==0.2` would
silently mean a different model than `==0.1`, which is exactly the "same name, different behaviour"
class this repository's per-Provider packaging (ADR-0016) exists to prevent. Nothing about 2.5 has
been measured yet (below), so a bump would also be a quality claim made in a version number.

### (b) One package with a `gliner2_major` config knob — REJECTED

A single package cannot declare two disjoint pins. The knob would resolve to whichever `gliner2` the
environment happened to install and fail at load on the other — i.e. it would move the failure from
resolution time to model-load time, which is the direction this repository refuses to move failures.

### (c) A new package pinned to the new major — CHOSEN

The pin is the Provider's own (ADR-0016), the checkpoint family is the Provider's own, and an
adopter installs the one they want. Both remain fully supported; neither is deprecated by this ADR.

## Decision 2 — ONE package with two entry points, not two packages

GLiNER 2.5 is a single checkpoint family serving both Capabilities over **one** dependency pin, one
compatibility gate (`latence_gliner25._compat`) and one loader seam (`_backbone.py`). Two packages
would duplicate all three across distributions that must move in lockstep, and would make it
possible to install half a family — with the two halves free to drift onto different `gliner2`
minors, which is the drift the single pin exists to prevent.

The 1.x line being two packages is not a precedent to copy: they were built a release apart, against
two genuinely different checkpoints (a general model and a dedicated guardrails+PII model). 2.5 has
one checkpoint for both.

## Decision 3 — the conflict boundary is ENFORCED, in three places

A stack naming `fused_entity_relation.gliner25` beside `redaction.gliner2` **passes**
`latence stack validate`: both Providers are real, both fulfil their Capability, and nothing in the
Stage contract is violated. It then cannot be installed. That is the worst possible place to learn
it, so the pair is refused three times before anything is fetched:

1. **The package pins.** Disjoint windows; any resolver refuses the pair outright. Asserted from the
   committed manifests in `packages/latence-gliner25/tests/test_gliner25_compat.py`.
2. **`deploy/env/*/environment.yaml`.** `cpu-text` and `gpu-learned` — the two environments that
   install a 1.x Provider — declare `gliner2>=1.3,<2` as an **enforced conflict boundary**, exactly
   as `gpu-learned` already declares the `transformers>=4.51.3,<5` window (#198). Adding
   `packages/latence-gliner25` to either fails `latence env verify` with **`E-CONSTRAINT-CONFLICT`**
   — offline, import-free, lock-independent, on every commit through
   `scripts/verify-local.sh`. `packages/latence-core/tests/test_envspec.py` injects exactly that
   mutation into a copy of each real environment and asserts the finding, plus the negative control
   that the boundary flags nothing else.
3. **`latence_gliner25._compat`.** The gate for the environment the resolver never saw — a pod with
   a hand-`pip install`ed `gliner2`, which is precisely what produced the #192 field failure on the
   1.x redactor. An unsupported version, a missing `AutoExtractor`, a `span` checkpoint, or a model
   without the batched entry points is a **typed `ProviderError`** (ADR-0034) naming the supported
   range and the Provider that *does* load it — never a raw `AttributeError`.

The `span`-checkpoint refusal deserves its own sentence: `AutoExtractor` dispatches on the
checkpoint, so pointing `model:` at a 1.x checkpoint **loads successfully** and then behaves like a
different model under the same Provider name. A silent wrong-model run is worse than a crash, so it
is refused at load.

### The consequence nobody wanted: the uv workspace

A uv workspace is **one resolved environment** — `uv lock` solves every member together — so the
three packages cannot all be members. `uv lock` says so directly:

> Because latence-extract-gliner2 depends on gliner2[local]>=1.3,<2 and latence-gliner25 depends on
> gliner2[local]>=2.0,<3, we can conclude that latence-extract-gliner2 and latence-gliner25 are
> incompatible.

That refusal is correct, and it is the same fact as (2) at a different altitude. `latence-gliner25`
is therefore listed under `[tool.uv.workspace] exclude`. Moving its dependency behind an extra was
tried and **does not help** — uv locks members' extras too — and it would have hidden the pin from
the `deploy/env` boundary check, disabling (2).

Exclusion is cheap for the checks that work by directory (ruff, `scripts/typecheck.sh`, pytest, the
contract reference, the environment specs, the publish-readiness gates) and expensive for exactly
two things, both of which were found by running the gates rather than by reasoning, and both fixed:

* **the release build.** `uv build --package <name>` resolves through the workspace and fails
  outright for an excluded package. The loops in `release.yml`, `ci.yml` and the runbooks now build
  by **path** (`uv build "$pkg"`), which is identical for members and non-members.
* **the SBOM.** `uv export --all-packages` resolves the workspace, so an excluded package's
  dependency surface would be absent from a document that is stamped, checksummed and signed as the
  release's component inventory — the exact silent false negative `release-integrity.sh` fails
  closed on. `latence-gliner25` therefore carries its **own committed `uv.lock`**, and
  `scripts/generate-sbom.sh` folds that lock's export into the requirements union (deduplicated by
  name+version) before the coverage check. It **fails closed** if an excluded package has no lock of
  its own, so the next exclusion cannot open the gap silently.

## Decision 4 — the redaction claim is zero-shot, and says so

`redaction.gliner25` drives the **general** 2.5 checkpoint with PII labels zero-shot, because no
2.5 PII checkpoint exists. That is a real capability — the GLiNER family is zero-shot span
extraction and PII types are ordinary label strings to it — and it is **not** the same thing as a
model trained on the task. Three consequences are built in rather than documented:

* **No parity is claimed and no recall number exists.** `docs/BENCHMARKS.md` records where the
  measurement will go and what produces it. The `ProviderProfile.perf_note` says "ZERO-SHOT",
  "no parity", "UNMEASURED" in the descriptor itself, and a test asserts those words are there.
* **`guardrails: true` is REFUSED** on the default checkpoint (a typed `ConfigError` naming the
  deterministic `screening.content_keyword` Stage and `redaction.gliner2` as the two things that do
  work). A security control that is enabled and unbacked is worse than one that is off. It is
  accepted with a warning on an operator-supplied `model`, because someone who fine-tuned their own
  2.5 guardrails head knows something this package does not.
* **The ADR-0044 universal financial-PII floor is unaffected.** It is a deterministic recogniser at
  the single shared `finalize_redacted_chunk` seam every Redaction Provider funnels through, so
  `ssn` / `credit_card` / `iban` are masked whatever the model does or fails to do. That floor is
  why a zero-shot PII Provider is shippable rather than merely plausible.

An inactive relation head gets the same treatment in the extractor: `BoundaryExtractor`'s decoder
returns `{}` when `enable_relations` is false and `format_results` then fills every requested type
with an empty list — a well-formed, entirely empty answer with no error. The load seam logs that
loudly and the Provider reports its relation count per batch, so "zero relations" is visible rather
than inferred.

## Decision 5 — `chunk.page`: page-preferred, 1024 otherwise

A third Chunker in `latence_core.stages.chunk`, behind the same `Chunker` seam and the **same
offset-preserving engine** as `chunk.markdown` / `chunk.sentence_window`, so the S3 round-trip
(chunk → offset → original page) and the v19 `page_slice` stay exact.

* **A document with a MEASURED page map is cut on its page boundaries** — one chunk per page. A page
  is an author-chosen unit of meaning, and a retrieval hit on one is citable as "page 7" with no
  further resolution; a window strategy cuts wherever the token budget runs out, which is a boundary
  nothing in the document agrees with.
* **A document with none falls back to the windowed budget at 1024.** Per ADR-0060 an absent page
  map is a **value**, not a `page 1`: an `ASSUMED_SINGLE_PAGE` map is the Parser saying "no page
  structure", so this Provider reads the map's **origin**, not merely its presence. Treating that
  single span as a page would emit one enormous chunk for a whole `.txt` file *while claiming it was
  a page*.
* **A page over the budget still SPLITS.** A page-sized chunk past the model window is a *silent*
  truncation (tail entities lost, offsets unreliable, and for the Redactor, PII left unmasked);
  a split is visible in the record count and every piece stays groundable. `min_tokens` folds a
  remnant back **within** a page, never across one.

**The existing 512-token defaults of `chunk.markdown` / `chunk.sentence_window` are deliberately
untouched.** They are baked into committed benchmark baselines and parity oracles; changing them
globally would silently invalidate every published number. 1024 is the default *of the new
strategy*, chosen to pair with the longer window: 1024 chunker tokens reach ~1229 mDeBERTa tokens at
the pod-measured 1.2× expansion, which `model_window_for` rounds to 1280.

## Decision 6 — the wizard selects an ENGINE, not a Provider

`SetupAnswers.engine` is one answer; `ExtractContribution` and `RedactContribution` both read their
Provider off the same `ExtractionEngine` row, and `ChunkContribution` reads that row's chunker. There
is no combination of answers that emits a mixed pair — the invariant is structural, not advisory, and
it is asserted over the whole (engine × profile) product. The row also carries the engine's chunk
budget (so `--engine gliner25` alone lands the page/1024 pairing, resolved on the *struct* so a
programmatic `build_stack_config` caller gets it too) and its documented checkpoint window, which is
a **refusal**: a budget whose derived `max_len` exceeds it is rejected at answer construction.

The 1.x engine's `max_window` is `None` — no ceiling is claimed, because none was ever verified for
that family. A first draft asserted one for symmetry and refused budgets that work today; the
negative control that keeps it honest is now a test.

## Decision 7 — the advertised 4096 window is not a window of reliable extraction

**This is the strongest finding in the integration and it changed the design.** Measured on the pod
(RTX 2000 Ada 16 GB, driver 580.159.04, torch 2.4.1+cu124, transformers 4.57.6, gliner2 2.0.0,
`fastino/gliner2.5-multi-v1`, 2026-08-25). One call at `max_len=4096` over a document seeded with a
repeated PII sentence, labels `["person name","email address","social security number"]`, threshold
0.4. `coverage` is the fraction of the document the returned spans reach:

| tokens | chars | pii_spans | max_char_seen | doc_chars | coverage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 674 | 2312 | 24 | 2310 | 2312 | **100%** |
| 1262 | 4335 | 30 | 4293 | 4335 | **99%** |
| 1850 | 6358 | 27 | 5738 | 6358 | 90% |
| 2438 | 8381 | 13 | 2869 | 8381 | **34%** |
| 3026 | 10404 | 25 | 7493 | 10404 | 72% |
| 3950 | 13583 | 32 | 9227 | 13583 | 68% |
| 4874 | 16762 | 32 | 9227 | 16762 | 55% |
| 6050 | 20808 | 32 | 9227 | 20808 | 44% |
| 7562 | 26010 | 32 | 9227 | 26010 | 35% |

At 7,562 tokens the document holds **90 SSNs** and the model returns **32 spans covering the first
35%** of the text. It raises nothing, warns nothing, and returns a well-formed result;
`max_char_seen` pins at 9,227 however much longer the document gets. Two further modes on the same
hardware: 12,612 tokens with a single label returned **zero** spans, silently; and longer inputs
with three labels hit `torch.OutOfMemoryError` inside DeBERTa's `disentangled_attention_bias`
(2.14 GiB requested, 1.63 GiB free) — disentangled attention is quadratic, so one long chunk OOMs a
16 GB card outright.

**Stated honestly:** the filler is synthetic repetitive prose, so the percentages are *directional,
not a benchmark*. What is **not** in doubt is the failure MODE — overflow is silent, partial and
**non-monotonic** (2,438 tokens scored *worse* than 3,026) — and that useful recall ends far below
4096. Non-monotonicity is what makes this worse than a hard truncation: no threshold test catches it
reliably.

### 1024 is an operator decision; the table confirms it

The reliable chunk budget is **1024 tokens** — the windowed-fallback default of `chunk.page` and
the value the wizard emits for the gliner2.5 profile. It was **decided**, not read off the curve as
"still acceptable at 1,850": the curve's job is to show that 1024 sits comfortably inside the
99–100% region and that the region ends nowhere near the advertised ceiling.

The Providers' default `max_len` is **derived** from it — `model_window_for(1024)` — rather than
typed as a literal, so the Provider default and the wizard's emitted `max_len` cannot drift apart.
`MAX_SUPPORTED_MAX_LEN = 4096` remains the library ceiling an operator may raise into; it is never
the default.

### Two guards, because the failure is asymmetric

* **The window bound.** Above the safe window, extraction is **warned** (losing entities is a recall
  trade an operator can see in the output and choose) and redaction is **REFUSED** with a typed
  `ConfigError`. For a Redactor an over-wide window is not a trade: the Stage emits "redacted"
  chunks that still contain PII with nothing in the run reporting it, and the ADR-0044 floor only
  covers the deterministic financial set — not a name or an address. That is fail-open on a security
  control, and it is not an operator's to opt into by accident. It is accepted with a loud warning
  when `model` names an operator-supplied checkpoint, because the measurement describes one
  checkpoint and someone with their own evidence outranks it — the same shape as the `guardrails`
  switch.
* **The per-chunk guard.** Bounding `max_len` bounds what the model is *asked* to read; it does not
  bound what it is *handed*. A chunk whose own `token_count` reaches past the window at the
  pod-measured 1.2× expansion is **refused** with a typed `ProviderError` naming the budget knob.
  Refused rather than trimmed or re-chunked: a Capability that cannot be served is a loud error, and
  segmentation belongs to the Chunk Stage (ADR-0021), not to a Provider quietly re-doing it. The
  guard inverts exactly the derivation the default applies, so a wizard-emitted stack cannot trip
  it, and a chunk reporting `token_count == 0` is let through — the guard refuses on evidence, never
  on its absence.

`chunk.page`'s split is now a **measured requirement**, not a safety argument: a dense page
routinely exceeds 2,000 tokens, which lands squarely in the collapse zone. A page over the budget
splits. No exceptions.

The offline fake models the truncation — it drops spans past a window and returns a well-formed
result — so the guard is what makes the fail-open test pass, not the fake's politeness. Verified by
removing the guard's evidence and watching the redaction test return `masked_content` set with the
email still in the clear.

### Exploiting the longer window without shipping an OOM

gliner2 pads every text in a micro-batch to the batch's longest sequence, so one forward's padded
activation is `microbatch_max_texts × max_len` positions. The shared
`resolve_microbatch_bounds` derives its default from the detected card against the **768**-window
reference point (4 × 768 = 3072 positions, measured at 21.6 GiB on the 32 GB pod). Taking that
unchanged under a 1280-token window would enlarge the padded forward by 1.67× — "raising `max_len`
without lowering `max_texts`" is an OOM shipped as a default. So the shared resolver still answers
(explicit knobs win, the VRAM derivation still lowers the envelope on a smaller card) and its answer
is re-expressed in positions and divided by the actual window. **Nothing assumes the boundary
architecture is cheaper**, because nobody has measured it — and the pod DID hit
`torch.OutOfMemoryError` inside DeBERTa's quadratic `disentangled_attention_bias` on a long single
chunk, which is the same envelope seen from the other side.

## Measured on the real model (2026-08-25) — two defects a fake cannot catch

The Providers were written against the wheel and tested against a faithful fake. Both were then
run for real — `gliner2==2.0.0` + `fastino/gliner2.5-multi-v1`, in an isolated venv on an Apple
Silicon macOS host (CPU) and on the CUDA pod — and the real load found two things no offline test
could have. Both are fixed here, and the second corrects a claim this repository already held.

**1. The 2.5 checkpoint needs the tokenizer-config shim, and the old "only one checkpoint is
affected" claim is false.** `AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1")` dies with
`AttributeError: 'list' object has no attribute 'keys'` at
`transformers/tokenization_utils_base.py:1210 _set_model_specific_special_tokens`, because the
checkpoint's `tokenizer_config.json` stores `extra_special_tokens` as a 10-element **list** — read
directly out of the downloaded snapshot, not inferred. That is the exact field failure ADR-0034
records for the 1.x PII checkpoint, and `latence_pii_gliner2._tokenizer_compat` already solved it —
while asserting that the sibling extract checkpoint's `{}` was "exactly why only this PII
checkpoint is affected". The 2.5 line reverts to the list shape, so the claim was a statement about
a file a third party controls, and it did not survive contact.

The fix is not a copy: `coerce_list_extra_special_tokens` is **promoted into
`latence_core.providers.perf`, beside `load_gliner2_model`**, and both packages call it — three
Providers, two packages, one shim. Its guarded / best-effort / idempotent posture is kept verbatim
(ADR-0034: no *new* silent error, only the removal of a crash), the stale sentence is corrected in
the same change, and the per-package module is deleted rather than left as a second copy. It is
applied once in `load_gliner25_model`, so both GLiNER 2.5 Providers and the uncompiled-reload path
get it without restating anything.

**2. `protobuf` and `sentencepiece` are undeclared runtime dependencies.** `gliner2[local]==2.0.0`
declares numpy / peft / safetensors / torch / transformers and no more; a clean install then fails
with transformers' `PROTOBUF_IMPORT_ERROR`. Same class of gap as the `peft` dep the 1.x packages
already record. `sentencepiece` is on the same path and was confirmed from the transformers 4.57.6
source rather than assumed — `models/deberta_v2/tokenization_deberta_v2.py` carries a top-level
`import sentencepiece as sp` and `@requires(backends=("sentencepiece",))`, and its
`SENTENCEPIECE_IMPORT_ERROR` sits beside the protobuf one in `utils/import_utils.py`; protobuf
simply raises first. Both are declared here, with their licences verified separately from the
installed distributions' own metadata on 2026-08-25 (ADR-0012): `protobuf` 7.36.0 declares
`License: 3-Clause BSD License`, `sentencepiece` 0.2.2 declares `License-Expression: Apache-2.0` —
both permissive. They were invisible in the 1.x environments because something else happened to
pull them (`deploy/env/cpu-text/requirements.lock` records `protobuf # via onnxruntime` and
`sentencepiece # via gliner`), and neither of those distributions is in this package's closure.

Both are pinned `protobuf>=7.36,<8` and `sentencepiece>=0.2.2,<0.3` — the H-G1 convention every
other pin in this workspace follows, and neither is an exception. The **floor** is the version
actually verified, not a guess at the oldest release that might work; the **cap** is the boundary
past which the API may break, which matters more than usual here because both arrive through a
tokenizer path this package does not control, and `protobuf` has broken generated-code
compatibility across majors before. `sentencepiece` is a 0.x line, so its minor is the effective
major boundary — the same shape as `gliner>=0.2.5,<0.3` and `flashdeberta>=0.0.5,<0.1`. (They
shipped uncapped in the first commit; that was an oversight, caught in review against the sibling
manifests.)

`peft` was re-checked and is deliberately **not** carried forward: 2.0.0's own `METADATA` declares
`peft<1,>=0.10` under the `local` extra, so the explicit dep the 1.x packages carry (a pod-found
1.x gap) is redundant here.

**What the real forward pass returns** — the shapes the record contract rests on, confirmed rather
than assumed. NER: `{"entities": {label: [{"text","confidence","start","end"}]}}` with char offsets
present. Relations: `{"relation_extraction": {rel: [{"head": {"text","start","end","confidence"},
"tail": {...}}]}}` — the same shape `latence-extract-gliner2` already unpacks, so the contract
carries over unchanged. Two details the fake now mirrors exactly: the head/tail key ORDER is
`text, start, end, confidence` (the 1.x docstring lists confidence second — key order only, no
contract impact), and head and tail carried the SAME score in every row observed, so nothing here
assumes they can differ without checking.

Zero-shot PII on the general 2.5 checkpoint found all four probes at high confidence (person
0.9992, email 0.9930, phone 0.9968, `social security number` 0.9875). That is a **smoke signal, not
a benchmark**: four probes on one hand-written string is not recall, and the measurement in
`docs/running-the-benchmarks.md` is still outstanding. No number from it appears anywhere as a
quality claim.

The fake was updated to be able to disagree about the first defect too: it refuses to load an
un-shimmed list-shaped config with the verbatim native `AttributeError`, so *removing the shim call
turns a test red* rather than passing silently. Verified by deleting the call and watching
`test_the_shim_is_applied_before_the_backbone_load` fail.

## What was NOT built, and what remains unverified

* **No throughput or quality number for GLiNER 2.5 exists.** The checkpoint HAS now been loaded
  and driven (above), which is what found the two defects — but a smoke run is not a measurement.
  Every test in this change runs against a faithful offline fake of `gliner2` 2.0.0 that reproduces
  the real degradations (bare strings without `include_spans`, an empty-but-well-formed relation
  result on an inactive head, a `TypeError` on an unknown load kwarg, the un-shimmed tokenizer
  config). A fake proves the Provider parses, grounds and batches the real API; it says **nothing**
  about extraction or redaction quality. `docs/running-the-benchmarks.md` names what will produce
  the numbers and where they land.
* **No `deploy/env/` environment resolves `gliner2` 2.x yet.** The compatibility matrix's cells for
  the boundary say so explicitly rather than leaving the absence to be inferred.
* **The weights licence is carried, not re-read.** The library licence (Apache-2.0) was read out of
  the 2.0.0 wheel's own `METADATA` on 2026-08-25; the checkpoint's Apache-2.0 was taken from the
  Hugging Face model card and is cited with that date in the `ProviderProfile`, per ADR-0012's
  weights-and-code-separately rule. The two newly-declared deps' licences WERE read from the
  installed distributions (above).
* **The pod hardware behind the real run is not named precisely here.** The run was reported as an
  isolated venv on an Apple Silicon macOS host (CPU) and on "the CUDA pod"; no SKU, driver or CUDA
  version was recorded with it, so none is stated. The Blackwell/CUDA-13 runtime axis of the
  compatibility matrix therefore stays `UNVERIFIED` — a smoke run whose hardware nobody wrote down
  cannot promote a matrix cell.
* **`chunk.sentence_window` is not offered by the wizard's chunk strategies.** It never was; adding
  a third strategy to the guided path is its own decision.

## Consequences

- Two new Providers, auto-covered by the conformance suite (C1–C6) through per-Provider stub-scoped
  cases; conformance is a gate, not a quality guarantee.
- A third Chunker, auto-covered by the existing Chunk conformance case.
- One new answer (`engine`) and one new escape hatch (`chunk_strategy`) in `latence setup`; the
  default engine's emitted stack is unchanged.
- `packages/latence-gliner25` is outside the uv workspace and carries its own lock; the release
  builds by path and the SBOM unions the two locks. Both are gated.
- A `gliner2>=1.3,<2` boundary now binds `cpu-text` and `gpu-learned`, with matching
  compatibility-matrix cells (resolution measured offline, runtime `UNVERIFIED`).
