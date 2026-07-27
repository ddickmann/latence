# Writing a Provider adapter with `AdapterBase` (P3-F2)

A **Provider is an adapter** over a native library; the Capability Protocol (ADR-0004) is
the stable I/O contract, unchanged. Every adapter repeats the same cross-cutting glue —
pick a device, batch its inputs, turn a native library exception into a **typed** framework
error, and carry Provenance/Classification + offsets through the native call. `AdapterBase`
(in `latence-core`) factors that glue into one **optional** base so a new adapter is small
and consistent. Architecture: ADR-0036 §1.

`AdapterBase` is **optional**. A Provider may satisfy a Capability structurally without it —
the shipping in-core reference Providers do. It adds **no** heavy dep to `latence-core`
(no torch, no model libs): it only orchestrates + maps; the adapter subclass owns its native
dep (ADR-0016). It is **not** a new protocol layer — it names no Capability method.

## The shape

```python
from latence_core import AdapterBase, ProviderProfile

class MyEntityExtractor(AdapterBase):
    # native exceptions that mean "operator handed me a bad config" -> ConfigError
    config_exceptions = (ValueError, KeyError)

    @classmethod
    def profile(cls) -> ProviderProfile:
        return ProviderProfile(
            compute="either", memory_mb=2048, model_id="vendor/model",
            license="Apache-2.0", license_verified=True,
            deterministic=False, batch=True, cost_per_1k=None,
        )

    def __init__(self, config=None):
        super().__init__(config)          # resolves the device against the declared profile
        if self.device.device is not None:
            self._model = load_model(...).to(self.device.device)   # placement is YOUR job

    def extract(self, chunks):            # the EntityExtractor Capability method (unchanged)
        yield from self.batched(list(chunks), self._run_batch)

    def _run_batch(self, chunk_batch):
        with self.guard("extract"):       # native exception -> typed taxonomy
            raw = self._model.infer([c.content for c in chunk_batch])
        return [self._to_mention(c, r) for c, r in zip(chunk_batch, raw)]
```

## The four helpers

### 1. Device resolution — `self.device`
`super().__init__(config)` resolves `config["device"]` (`auto|cpu|cuda`) against your declared
`ProviderProfile` via the P3-F1 `select_device` seam. Read `self.device.device`
(`"cpu"`/`"cuda"`/`None`) to place your model, and `self.device.skipped` to know it was
skipped-with-flag. **Placement stays your job** (ADR-0003) — the base only resolves + hands
over the `DeviceDecision`. A `compute="gpu"` adapter on a CPU-only host sees the **skip**
decision, never a fabricated device. A bad `device` string is a typed `ConfigError`.

### 2. Batching — `self.batched(items, run_batch)`
Groups `items` into `config["batch_size"]` chunks (default 32, validated `>= 1`) and calls
`run_batch` per group **when your profile declares `batch=True`**; otherwise it degrades to
one-item calls, so a non-batch adapter reuses the same code path. Output is **deterministic,
input-order** either way, and the input iterable is consumed **one batch at a time** — the
whole corpus is never buffered (the streaming Runner's friend, ADR-0033).

### 3. Native-exception → typed taxonomy — `with self.guard("<op>"): ...`
Wrap the native call. An exception raised inside becomes the right `LatenceError`
(ADR-0034): a native type listed in `config_exceptions` → `ConfigError` (operator bad
config); anything else → `ProviderError` (a runtime provider fault); a framework
`LatenceError` **passes through unwrapped**. The raised message carries only the provider,
the `op`, and the native exception's **type name** — **never the native exception's message**
(which could echo record content/PII), matching the Quality Report's no-PII discipline.

### 4. Provenance / offset carry
- `carry_provenance(src_record, char_start=…, char_end=…, page_start=…, page_end=…)` returns
  the produced record's `(Provenance, Classification)`: the source lineage is preserved,
  only the offset/page fields you pass are narrowed, and Classification is inherited
  unchanged.
- `rebase_offsets(local_start, local_end, chunk_char_start=…, offset_map=chunk.offset_map)`
  maps a chunk-local span to true original-markdown offsets, **reusing the existing
  `OffsetIndex` seam** (ADR-0031) — never a weaker hand-rolled mapper. It degrades to the
  exact `chunk_char_start + local` shift when a chunk carries no offset map.
- `resolve_pages(chunk, doc_char_start, doc_char_end, resolver=…)` resolves a document-offset
  span to its **own** source page(s) through the document-scoped `PageIndexResolver` seam (which
  owns the memoised, never-crashes, fuzzy drift-recovering `PageOffsetIndex`, the half-open end
  clamp, and the degradation counter), falling back to the chunk's inherited page range only when
  no map is available for the document at all.

These make the "offsets/page-provenance intact" invariant the **default**, not a per-adapter
reimplementation.

### The page resolver: one per call, observe every chunk

A `PageMap` is **document**-scoped, and since schema v18 the Chunk Stage stamps it on a
document's **first** chunk only (it used to ride all N chunks — ~92% of a chunk checkpoint line
on a long PDF). So `chunk.page_map` is `None` on every later chunk, and a per-chunk index built
from it would silently give every mention after the first chunk its chunk's *inherited* page
range — a plausible-looking wrong page. Create **one** `PageIndexResolver` per Capability call
and `observe()` **every** chunk you are handed, before any filter decides it is uninteresting
(a document's first chunk — the map carrier — is very often the one your `min_chars` floor
drops):

```python
from latence_core import PageIndexResolver

def extract(self, chunks):
    resolver = PageIndexResolver()          # ONE per call, across batches

    def run_batch(batch):
        out = []
        for chunk in batch:
            resolver.observe(chunk)         # BEFORE the "is this chunk interesting" filter
            out.extend(self._extract_one(chunk, resolver))
        return out

    yield from self.batched(chunks, run_batch)

# ...and inside _extract_one, per span:
page_start, page_end = self.resolve_pages(chunk, doc_start, doc_end, resolver=resolver)
```

A `Runner` that holds the parsed documents can seed the resolver authoritatively instead
(`PageIndexResolver.from_documents(documents)`, ADR-0017: files are the source of truth).

"One per call" does **not** mean "keeps the corpus": the resolver retains an index for the
`DEFAULT_MAX_LIVE_DOCUMENTS` (64) most recently used documents and releases the rest with their
page maps, so a Stage's page state stays proportional to the documents in flight rather than to
the corpus (ADR-0033). 64 is well above every shipped Provider's batch (16). If yours deliberately
observes more distinct documents than it resolves against — a very wide batch, or a two-pass shape
that resolves long after it observed — construct it as
`PageIndexResolver(max_live_documents=N)`. You will be told if you needed to and did not: a lookup
for a document whose index was already released is counted
(`resolver.evicted_reentries`) and warned once, counts-only, instead of quietly inheriting the
chunk's page range.

## Malformed-profile: fail loud (P3-F2 §2)

A Provider that declares a **malformed** profile (a `PROFILE`/`profile()` that is not a
valid `ProviderProfile`) is an author bug. The `LocalRunner` now **FAILs the stage** with a
typed `ConfigError` (a FAIL Quality Report + a `config` typed-error span), rather than
silently reading it as "no profile" and running. An **absent** profile still means "follow
the request" (`None`); an import/load error still defers to the normal execution path.

> Note: the `Profiler` Capability's own method is named `profile(self, documents, mentions)`.
> `profile_of` recognises a profile **declaration** only as a genuine `@classmethod` (or a
> `PROFILE` ClassVar / a bare `ProviderProfile` assigned to `profile`), so a Capability
> `profile` **instance** method is correctly read as "no declaration", never mislabelled
> malformed.
