# Validate `disambiguation.embedding` on a real GPU — turnkey pod steps (W1 rework, #94)

The offline slice is validated against the in-core **deterministic `embedding.hashing`** embedder and
a **context-bearing pod-vector stub** (the unit tests assert that at the default threshold 0.90 the
genuine duplicates merge while distinct entities — IBM vs Apple — stay SEPARATE, and that a
bare-surface control over-merges them). The **real** enterprise-safety numbers — the exact
IBM/International-Business-Machines/Apple/Apple-Inc/Berlin separation with the real
`multilingual-e5-small` embedder — come from a **CUDA pod run**. Until this runs, the GPU numbers are
explicitly **"pending pod validation"**, never fabricated.

> **The false-green catch (why this rework exists).** The FIRST-cut disambiguator embedded the **bare
> surface** and over-merged: pod-measured `IBM ~ Apple` = **0.870**, which out-ranks the margin of
> genuine duplicates — no safe threshold. This rework embeds **surface + context**; the pod run below
> must show `IBM ~ Apple` DROP (≈0.808) while genuine duplicates rise (≈0.96–0.99), so the default
> 0.90 threshold yields the CORRECT 3 clusters — `{IBM, International Business Machines}`,
> `{Apple, Apple Inc}`, `{Berlin}` — NOT the over-merged 2-cluster collapse. If the run yields fewer
> than 3 clusters, the context path is not being embedded — do not fold in a number until it is.

These steps are turnkey: run them on a single ≥8 GB CUDA GPU (an A10, an RTX 4090, a RunPod pod, …).
`multilingual-e5-small` is a 118M model, well under 1 GB VRAM.

## 0. Prereqs

```bash
nvidia-smi                     # confirm a CUDA device is visible
python --version               # 3.11+
```

## 1. Install the framework + the learned embedder WITH its heavy deps

```bash
git clone https://github.com/ddickmann/latence-framework && cd latence-framework
uv sync
# The disambiguator itself is pure-Python (installed by `uv sync`); the learned embedder it resolves
# ships the torch/sentence-transformers stack — install it WITH deps on the pod:
uv pip install -e packages/latence-embedder-st
uv pip install -e packages/latence-disambig-embedding
```

## 2. The decisive separation check — the exact 5-mention case, real e5 embedder

This is the `cosine evidence` the rework rests on. It embeds `surface + context` with the real
`multilingual-e5-small` and prints the pairwise cosines AND the resulting clusters.

```bash
PYTHONHASHSEED=0 python - <<'PY'
from latence_core.contracts import EntityMention, ChunkRecord, Classification, Provenance, PageMap
from latence_disambig_embedding.provider import EmbeddingDisambiguator, _cosine
from latence_embedder_st.provider import SentenceTransformersEmbedder

cls = Classification(language="en", category="news", sensitivity="internal")

def prov(doc, cs, ce):
    return Provenance(source_uri=f"file://{doc}", file_name=f"{doc}.txt", file_type="txt",
                      file_size=1000, document_id=doc, char_start=cs, char_end=ce,
                      page_start=1, page_end=1)

C1 = ("IBM reported strong earnings. International Business Machines is a technology company "
      "making enterprise computers, mainframes and software.")
C2 = ("Apple unveiled a new iPhone. Apple Inc is a consumer electronics company designing "
      "phones, laptops and tablets, based in Cupertino.")
C3 = "Berlin is the capital city of Germany, a major European cultural and political centre."

def chunk(rid, doc, content):
    return ChunkRecord(record_id=rid, document_record_id=f"{doc}:doc", index=0, content=content,
                       token_count=30, provenance=prov(doc, 0, len(content)), classification=cls,
                       page_map=PageMap.from_page_texts([content]).page_map)

def men(rid, doc, ch, text, label="ORG"):
    start = [c for c in chunks if c.record_id == ch][0].content.index(text)
    return EntityMention(record_id=rid, document_record_id=f"{doc}:doc", chunk_record_id=ch, index=0,
                         label=label, text=text, confidence=0.9,
                         provenance=prov(doc, start, start + len(text)), classification=cls)

chunks = [chunk("ch1", "d1", C1), chunk("ch2", "d2", C2), chunk("ch3", "d3", C3)]
mentions = [
    men("m1", "d1", "ch1", "IBM"),
    men("m2", "d1", "ch1", "International Business Machines"),
    men("m3", "d2", "ch2", "Apple"),
    men("m4", "d2", "ch2", "Apple Inc"),
    men("m5", "d3", "ch3", "Berlin", "LOC"),
]

# Real e5 embedder on cuda, threshold 0.90 (the context-safe default).
d = EmbeddingDisambiguator({
    "threshold": 0.90,
    "context_window": 160,
    "embedder": {"provider": "embedding.sentence_transformers",
                 "config": {"model": "intfloat/multilingual-e5-small", "dimension": 384,
                            "device": "cuda"}},
})
records = list(d.disambiguate(mentions, [], chunks))
clusters = {r.entity.canonical_text: sorted(r.entity.member_mention_ids)
            for r in records if r.entity}
print("CLUSTERS:", clusters)
n = len(clusters)
print("CLUSTER COUNT:", n, "(expected 3 — correct separation)" if n == 3
      else "(WRONG — over/under-merge)")

# Raw surface+context cosines (fold these into the ADR/issue table — never fabricate).
emb = SentenceTransformersEmbedder({"model": "intfloat/multilingual-e5-small",
                                    "dimension": 384, "device": "cuda"})
def ctx(m):
    ch = [c for c in chunks if c.record_id == m.chunk_record_id][0]
    return d._surface_context_text(m.text, m, {c.record_id: c for c in chunks})
vecs = {m.text: list(emb.embed([ctx(m)]))[0] for m in mentions}
for a, b in [("IBM", "International Business Machines"), ("Apple", "Apple Inc"),
             ("IBM", "Apple"), ("IBM", "Berlin")]:
    print(f"cos({a} + ctx, {b} + ctx) = {round(_cosine(vecs[a], vecs[b]), 3)}")
PY
```

**Pass criteria:** `CLUSTER COUNT: 3`; the same-entity cosines ≈0.96–0.99 and the `IBM ~ Apple`
cosine ≈0.81 (distinct). Record the printed numbers; they are the pod evidence for the ADR table.

## 3. Conformance on a CUDA host (real weights)

```bash
LATENCE_CUDA=1 PYTHONHASHSEED=0 python - <<'PY'
from latence_core.capability import ProviderRegistry
from latence_core.conformance import run_conformance
r = run_conformance(ProviderRegistry(), "disambiguation.embedding")
for c in r.checks:
    print(c.check, c.status.value, "-", c.detail)
PY
```

All C1–C6 should PASS (the offline path already passes against the hashing embedder; here the resolved
learned embedder runs for real). Fold the real numbers into the ADR note, replacing "pending pod
validation" — never a fabricated figure.
