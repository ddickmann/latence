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
  span to its **own** source page(s) through the `PageIndexResolver` seam (which owns the
  never-crashes, fuzzy drift-recovering `PageOffsetIndex`, the half-open end clamp, and the
  corpus-level drift roll-up). It returns `(page_start, page_end)` — there is no `None` to
  degrade on.

These make the "offsets/page-provenance intact" invariant the **default**, not a per-adapter
reimplementation.

### The page resolver: one per call, and the chunk answers for itself

Since **schema v19** every `ChunkRecord` carries its own `page_slice`: exactly the document page
spans overlapping that chunk's `[char_start, char_end)`, in the document's original coordinates.
One span for a chunk inside a page, two for a chunk straddling a page break. So resolution needs
nothing but the record in hand — any order, any subset of the stream, no `observe()`, no
per-document state:

```python
from latence_core import PageIndexResolver

def extract(self, chunks):
    resolver = PageIndexResolver()          # ONE per call — for the drift roll-up, not state

    def run_batch(batch):
        out = []
        for chunk in batch:
            out.extend(self._extract_one(chunk, resolver))
        return out

    yield from self.batched(chunks, run_batch)

# ...and inside _extract_one, per span:
page_start, page_end = self.resolve_pages(chunk, doc_start, doc_end, resolver=resolver)
```

**Do not read the page field off the record**, and never build a page index per chunk in your
own code: the seam owns the end clamp and the drift accounting, and re-deriving them is how
they drift apart. A chunk whose producing Chunker stamped no spans raises
`PageSliceMissingError` rather than returning `None` — degrading to the chunk's *inherited*
`page_start`/`page_end` emits a page that looks resolved and is not, which is the exact
provenance regression this seam exists to prevent (ADR-0021/0031, ADR-0034).

**A relation's covering span is different.** It runs from one endpoint mention to the other and
those may live in *different* chunks, so no single chunk can answer for it — and it is not a
sub-chunk offset, so `resolve_pages` is the wrong tool. Both endpoints already resolved their own
pages exactly when they were emitted, so the covering span's pages are their union:

```python
from latence_core import pages_for_covering_span

pages = pages_for_covering_span(head.provenance, tail.provenance)
```

"One per call" does **not** mean "keeps the corpus". The resolver holds no document state at
all — it accumulates only the drift roll-up (counts of what already happened), so its footprint
is constant whatever the corpus looks like (ADR-0033). Earlier shapes needed more: while the
whole document map rode the chunk stream, the resolver memoised one index per document behind an
LRU retention window, and a corpus of short documents could stream hundreds of them past while
one batch of long-document chunks accumulated, evicting a document the batch was still resolving
— whose later chunks carried no map of their own, so their spans inherited the chunk's page
range: a plausible-looking **wrong** page (audit R4). A self-describing chunk cannot express
that, so the window, its budget knob and its eviction counter are gone.

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
