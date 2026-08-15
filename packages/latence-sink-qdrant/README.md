# latence-sink-qdrant

**One command between "pipeline finished" and "my retrieval engine serves the corpus."**

```bash
pip install 'latence-sink-qdrant[client]'
latence-sink-qdrant load ./out/export --collection corpus --url http://localhost:6333
```

```
collection 'corpus' (created, dim 384): 1284 record(s) read — upserted 1284, unchanged 0,
deleted 0 stale point(s) [load_id=1f0c…]
```

That is the whole product. It reads the Export your pipeline already wrote
(`records.parquet`), creates the collection if it is absent at the corpus's own vector width,
upserts in batches with the `embedding` column as the vector and every remaining column as
payload, and leaves.

## What it is (and deliberately is not)

The Export — open Parquet + JSONL on object storage — **stays the source of truth**
([ADR-0017](../../docs/adr/0017-ai-ready-deliverable-is-rag-corpus-plus-kg-as-files.md)). A sink is
an *additional*, optional step, never a replacement. This package is a **one-shot loader**: it holds
no index, caches nothing, and runs no search
([ADR-0048](../../docs/adr/0048-retrieval-tooling-is-strictly-stateless.md)). Query-time work lives in
`latence-retrieval`; a `RetrievalBackend` over the collection you just loaded is
`latence-retrieval[backend-qdrant]`.

**Querying what you loaded.** Every exported column rides through as payload under its **Export
column name**, so the chunk text is the payload field **`content`** (and `record_id`, `provenance`,
`classification`, `media_type` are filterable beside it). That is exactly the default `text_key` of
`latence-retrieval`'s `QdrantBackend`, so a collection this sink loaded is queryable with no
override — `Candidate.text` is the chunk text. (A backend pointed at a key the payload lacks raises
`BackendSearchError` rather than returning empty text.)

Out of scope on purpose: the KG (`graph.ttl` / `graph.graphml`), which is a different store's
problem.

## Sparse vectors (SPLADE), loaded as a real index

By default the sink loads the dense `embedding` column and nothing else. If the Export was written
with a `sparse_embedder` wired, `--vector both` projects the `sparse_indices` / `sparse_values`
columns into a **named Qdrant sparse vector** instead of leaving them in the payload:

```bash
latence-sink-qdrant load ./out/export --collection corpus --vector both --url http://localhost:6333
```

```yaml
vector: both              # dense (default) | sparse | both
sparse_source: sparse     # reads <stem>_indices + <stem>_values
sparse_vector_name: sparse   # the name the query side passes as `using=`
```

Three things to know before you run it, because each of them changes what you must configure
downstream:

- **Under `vector: both` the dense vector is named** (`dense` by default, `vector_name` to change
  it). Qdrant's *unnamed* vector is the single-vector legacy shape, and a point carrying a sparse
  vector carries a vector **map** — which addresses its members by name. So set
  `latence-retrieval`'s `QdrantBackend` `vector_name: dense` and `sparse_vector_name: sparse` for a
  collection loaded this way; its default is the unnamed vector, which a hybrid collection has not
  got.
- **A collection's vector shape is fixed at creation.** Running `--vector both` over a collection an
  earlier dense-only load created fails with a named error telling you so — the fix is a new
  collection, not a config change.
- `--vector sparse` needs **no Embedder at all** on the Export: a SPLADE-only collection is a real
  shape and the sink will not demand a dense column it is not going to load.

The sparse columns are removed from the payload once they are loaded as a vector (keeping them would
store every term id twice), and the change digest folds the sparse signal in **only** on a sparse
load — so a re-embed of the sparse signal alone is correctly detected as changed, while a dense-only
collection's digests are untouched and a re-run of an unchanged corpus still upserts nothing.

The model behind the signal is the operator's choice and the operator's licence to accept — this
wires the *mechanism*, and pins no checkpoint. (`naver/splade-v3` is CC-BY-NC-SA-4.0; the in-core
`sparse.hashing` reference is Apache-2.0 and makes no retrieval-quality claim.)

## Re-running converges

Run it again — same command, same collection — and three things happen in one pass:

| the record is… | what the sink does |
|---|---|
| **unchanged** | nothing. Its stored content digest matches, so it is **not** re-upserted — this is what makes an incremental/Delta run ([ADR-0018](../../docs/adr/0018-incremental-corpus-deltas.md)) cheap without the sink having to be told it is one. |
| **new or changed** | upserted at the same derived point id, so it overwrites rather than duplicates. |
| **gone from the Export** | deleted — but only within this corpus's `id_namespace` (see below). |

Idempotency rests on **derived point ids**: Qdrant point ids must be unsigned integers or UUIDs, and
a corpus `record_id` is a content-addressed string, so the sink derives a deterministic UUIDv5 from
it (fixed namespace, overridable per corpus with `id_namespace`). The `record_id` itself rides in
the payload, so it stays filterable.

`id_namespace` is also the **sweep's ownership scope**: every point carries the namespace it was
derived under (`_latence_id_namespace`), and reconciling filters on it. Two corpora whose
`record_id` spaces could collide can therefore share one collection under two namespaces — each
load converges its own partition and cannot delete its co-tenant's points. (Points carrying no
namespace stamp at all — hand-loaded, or written by some other loader — count as this corpus's, so
the default "the collection is this Export's projection" setup still purges what it must.)

## The GDPR story crosses the boundary with the data

After a `latence retract` or `latence purge`, the pipeline re-exports the *surviving* corpus — the
deleted document's derived records are simply not in it. Upserting can never express a deletion, so
the sink stamps every point it touches with a per-run `_latence_load_id` and then deletes, by
filter, everything it owns (`_latence_id_namespace`) that is **not** stamped with this run's
marker.

**The verification step is not optional and not manual — it runs inside the load.** After the sweep
the sink counts the stale set again and *fails the load* if it is not zero:

```
error: reconcile did not converge on collection 'corpus': 3 point(s) of id_namespace
6f6c1a4e-2b3d-5f7a-9c1e-4d2f8a0b6c31 not written by this load are still present after the sweep. The collection may still serve retracted or purged records —
do not treat this load as complete.
```

For an audit trail, ask the store directly afterwards:

```bash
latence purge --doc contract-42                       # 1. erase in the corpus
latence process pipeline.yaml                          # 2. re-export the survivors
latence-sink-qdrant load ./out/export --collection corpus --url $QDRANT_URL   # 3. re-load
latence-sink-qdrant verify-absent 'contract-42#chunk-0' 'contract-42#chunk-1' \
    --collection corpus --url $QDRANT_URL              # 4. prove it
# exits 0 and prints "verified absent: 2 record id(s) are not in the collection",
# or exits 1 listing every id still present.
```

Exit 0 there is the erasure evidence, so the command will not produce it for a collection it never
looked at: if `--collection` is mistyped or the connection points at the wrong cluster, the
collection does not exist and `verify-absent` fails with a typed `SinkVerificationError` (exit 2)
instead of reporting the records absent. `load` is different on purpose — it creates the collection
it is projecting the Export into; an audit verb may neither create nor assume.

(Qdrant marks deleted vectors in its HNSW graph and reclaims them on optimisation. The guarantee
here is the one erasure needs: the point and its payload are no longer retrievable or searchable.)

The sweep filters on the `_latence_load_id` payload field. On a large collection that is a payload
scan; if it becomes the slow part of your load, add a keyword payload index on that field in Qdrant
once (`create_payload_index`) — the sink needs no change, and creating that index is left to the
operator rather than assumed here.

### When to turn reconciling off

Reconciling assumes the collection is *this* Export's projection — the normal setup, and the
default. If you load an **incremental** Export into a collection that also holds records the
increment omits, pass `--no-reconcile`; the collection converges on the next full load.

## Failure containment

A store fault anywhere in the upsert phase aborts the load **before** the sweep. A half-uploaded
corpus therefore never deletes the records it did not reach: the worst case is stale-but-present
data, never silently erased data. Native `qdrant_client` faults are mapped to typed errors
(`SinkAuthError` for 401/403, `SinkLoadError` otherwise) that carry no URL, api key, payload or
driver stack. `verify-absent` adds one more: `SinkVerificationError` when the audit could
not be *performed* at all (no such collection), kept apart from "the audit ran and passed".

## Configuration

Flags override a `--config` YAML/JSON block, so a cluster's connection details live in one file:

```yaml
collection: corpus
url: https://qdrant.internal:6333
api_key_env: QDRANT_API_KEY   # the NAME of the env var, never the key itself
batch_size: 512
distance: cosine        # only used when the collection is created
vector: dense           # dense | sparse | both
vector_name: null       # or a named vector
reconcile: true
```

| key | default | meaning |
|---|---|---|
| `collection` | — | **required** target collection |
| `basename` | `records` | Export file stem |
| `batch_size` | `256` | records per store round-trip (bounds memory *and* request size) |
| `distance` | `cosine` | `cosine` / `dot` / `euclid` / `manhattan`, used only at collection creation |
| `vector` | `dense` | which retrieval signal to load: `dense` (the Export's `embedding`), `sparse` (a SPLADE-only collection), `both` (hybrid) |
| `vector_name` | `null` | named dense vector to write; `null` ⇒ the collection's single unnamed vector. Under `vector: both` it defaults to `dense` instead, because a point carrying a sparse vector carries a vector *map* |
| `sparse_source` | `sparse` | Export column-pair stem (`<stem>_indices` + `<stem>_values`). Requires `vector: sparse`/`both` — set otherwise it is a loud error, never an inert key |
| `sparse_vector_name` | `sparse_source` | the name the sparse vector takes in the collection — what the query side passes as `using=` |
| `reconcile` | `true` | sweep this corpus's points absent from this Export (scoped to `id_namespace`) |
| `load_id` | none | audit **label** for the run, prefixed onto the per-run sweep marker (`<label>:<nonce>`). It is a label, never the marker itself: the sweep deletes everything *not* stamped with this run's marker, so a marker reused across runs would match nothing and silently keep serving retracted points |
| `id_namespace` | fixed UUID | point-id derivation namespace **and** the reconciling sweep's ownership scope — two corpora sharing one collection each get their own |
| `api_key_env` | none | **name** of the environment variable holding the api key, read at construction and resolved into `api_key`. Never passed to `QdrantClient` itself; an explicit `api_key` / `--api-key` wins |
| `url` / `host` / `port` / `grpc_port` / `prefer_grpc` / `https` / `api_key` / `timeout` / `location` | — | threaded verbatim to `QdrantClient` |

### The credential

A config file is **not** shell-expanded — `api_key: ${QDRANT_API_KEY}` would be sent to Qdrant as
the literal string and come back as an opaque 401. Name the environment variable instead
(`api_key_env: QDRANT_API_KEY`, or `--api-key-env QDRANT_API_KEY`) and export it from your secret
manager; the sink reads it at construction and the key never enters the file. This is the
framework-wide posture (`SECURITY.md`; the SharePoint Source's `token_env` / `client_secret_env`).
A variable that is unset or empty, and a `${…}` placeholder left in `api_key`, are both rejected up
front with a typed `SinkConfigError` (exit `2`) that names the *variable*, never a value.

## Testing without a Qdrant

The package ships a faithful in-memory client, so downstream wiring can be tested with nothing
running:

```python
from latence_sink_qdrant import QdrantSink
from latence_sink_qdrant.testing import FakeQdrantClient

client = FakeQdrantClient()
report = QdrantSink({"collection": "corpus"}, client=client).load("./out/export")
```

It is faithful on the **rejection** side, which is the side that matters: it returns 404 for an
unknown collection, 409 for a duplicate create, 403 for a bad api key, and 400 for a wrong-width
vector, a named/unnamed vector mismatch, a non-JSON payload — and for **a point id that is not an
unsigned integer or a UUID**, so a loader that handed Qdrant a raw `record_id` string fails here
exactly as it would against a real server.

It models named **sparse** vectors on the same terms: a sparse config that is not a named mapping, a
sparse name colliding with a dense one, indices and values of unequal length, non-integer/negative/
duplicate indices, a dense list under a sparse name — all 400, all rejected here rather than
discovered against a live server. What it deliberately does **not** do is rank: there is no search
in it (the sink never searches), so whether a real Qdrant *scores* a sparse query the way we expect
is settled by the live test below and is never asserted offline.

A live smoke test against a running Qdrant is included and skipped unless you opt in:

```bash
docker run -p 6333:6333 qdrant/qdrant
pip install 'latence-sink-qdrant[client]'
LATENCE_QDRANT_URL=http://localhost:6333 pytest packages/latence-sink-qdrant -k live
```

## Licensing

Apache-2.0, like the rest of the framework (commercial use granted by the
licence, no agreement needed). The optional `[client]` extra pulls `qdrant-client`,
**Apache-2.0** ([code](https://github.com/qdrant/qdrant-client/blob/master/LICENSE)); the Qdrant
engine it talks to is **Apache-2.0** ([code](https://github.com/qdrant/qdrant/blob/master/LICENSE)).
A vector store ships no model weights, so there is nothing to attest beyond the permissive code
license (ADR-0012; verified 2026-07-16). `latence-core` takes no new dependency from this package.
