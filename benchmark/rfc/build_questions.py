#!/usr/bin/env python3
"""Derive super-hard **fused** questions + gold chunk sets from a finished run (tutorial §5).

The RFC-retrieval-proof's whole claim is that *fusing* four retrieval signals (dense + BM25 +
SPLADE + KG multi-hop) unlocks a class of queries **no single signal can answer**. A benchmark that
proves that must be built from questions where fusion demonstrably wins — not hand-picked, not
padded. This builder earns each question the hard way:

1. **Mine gold pairs/chains from the INDEPENDENT ``_refs.jsonl`` graph** (the RFC-editor-authored
   cross-reference graph the pipeline never saw — ``build_corpus.py``). Every gold set is grounded
   in that graph and in the run's exported chunks, never invented.
2. **Emit candidates of four flavours** — ``multi-hop`` (needs the KG chain), ``lexical`` (exact
   header-field / status-code / ``MUST`` lookups), ``semantic`` (paraphrased concept, no lexical
   overlap), and ``compound`` (two distant sections linked only by a co-mentioned entity).
3. **Require literal subject presence in EVERY gold chunk.** A gold chunk that does not literally
   contain the question's subject is an index-page / incidental-mention false positive — the whole
   candidate is dropped (the §5 correctness guard).
4. **Gate on real retrieval.** Using :class:`Signals` (the real path drives ``latence_retrieval``
   against the **live** Qdrant — never the fake), keep a question **only** when
   ``fused recall@k > max(dense, bm25, splade) + min_lift`` **and** fused finds **≥ 1 gold chunk
   in top-k that no single signal finds** (`fused_only_gold`). Those two conditions together are the
   honest definition of "fusion unlocks this".

Each surviving question is emitted with its ``question``, ``type``, ``gold_chunk_ids``, the
``kg_chain`` it was mined from, and the **per-signal recall** — everything ``run_benchmark.py`` (§6)
and ``make_report.py`` (§7) need. If far fewer than ``--target`` survive, that is a true result
about corpus depth (report it), never a cue to pad with filler.

**The seam that keeps this testable — :class:`Signals`.** All first-stage retrieval enters through
one small protocol: ``rank(query, top_k) → {signal: [chunk_id, …]}``. The production
:class:`LatenceSignals` builds it from ``latence_retrieval`` over the real Qdrant + the query
encoders (imported lazily; the pod path). Offline tests inject a faithful in-memory ``Signals`` and
drive the entire mine → literal-guard → recall-gate → emit pipeline with **no pod, no network, and
no ``FakeQdrantClient``** — the gate logic is proven without a Qdrant at all.

Stdlib only for the core; ``latence_retrieval`` (+ the encoders / Qdrant client) only inside
:class:`LatenceSignals`, imported lazily so importing this module never needs a heavy dep.

Usage (the exact invocation from docs/TUTORIAL-rfc-retrieval-proof.md §5):
  python benchmark/rfc/build_questions.py \
      --run _latence/runs/<run_id> \
      --refs /workspace/corpus-rfc/_refs.jsonl \
      --target 40 --min-lift 0.24 \
      --out /workspace/report/rfc-benchmark.json

The Qdrant endpoint defaults to ``$LATENCE_QDRANT_URL`` (exported in §4.1) and the collection to
``rfc`` (§4.1's loader); override with ``--qdrant`` / ``--collection`` and the encoder / pool knobs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

# ---------------------------------------------------------------------------
# The retrieval seam — the ONE place first-stage search enters (kept injectable).
# ---------------------------------------------------------------------------

#: The single-signal keys every :class:`Signals` implementation must rank (the gate compares fused
#: against the best of these), plus the fused key the gate rewards.
SINGLE_SIGNALS: tuple[str, ...] = ("dense", "bm25", "splade")
FUSED_SIGNAL = "fused"


@runtime_checkable
class Signals(Protocol):
    """First-stage retrieval over the run's chunks — the injectable boundary the gate sits on.

    One method. Given a query string and a cut-off ``top_k``, return a ranked **chunk-id** list per
    signal, in descending relevance. The keys MUST include every name in :data:`SINGLE_SIGNALS`
    plus :data:`FUSED_SIGNAL`; ids are the corpus ``record_id`` (the chunk key the gold sets use),
    so the gate never needs to know how a signal was scored — only which chunks it surfaced.

    The production adapter (:class:`LatenceSignals`) runs ``latence_retrieval`` against the live
    Qdrant; a test injects an in-memory stand-in. Either way the gate code is identical.
    """

    def rank(self, query: str, top_k: int) -> dict[str, list[str]]:
        """Return ``{signal: [chunk_id, …]}`` — one ranked list per signal, descending relevance."""
        ...


# ---------------------------------------------------------------------------
# Data model — the small, typed shapes the miner and gate speak.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One exported chunk: its ``chunk_id`` (corpus ``record_id``), ``content``, and source RFC."""

    chunk_id: str
    content: str
    rfc: int | None


@dataclass(frozen=True)
class Hop:
    """One typed edge of a mined KG chain: ``src`` --``rel``--> ``dst`` (RFC numbers)."""

    src: int
    rel: str
    dst: int

    def as_dict(self) -> dict[str, object]:
        return {"src": self.src, "rel": self.rel, "dst": self.dst}


@dataclass(frozen=True)
class QuestionCandidate:
    """A mined-but-ungated question: subject + gold chunks + the KG chain it came from."""

    qid: str
    qtype: str
    question: str
    subject: str
    gold_chunk_ids: tuple[str, ...]
    kg_chain: tuple[Hop, ...] = ()


@dataclass(frozen=True)
class GatedQuestion:
    """A candidate that survived the literal-presence guard and the fused-vs-single recall gate."""

    candidate: QuestionCandidate
    recall: dict[str, float]
    fused_only_gold_ids: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        c = self.candidate
        return {
            "id": c.qid,
            "type": c.qtype,
            "question": c.question,
            "subject": c.subject,
            "gold_chunk_ids": list(c.gold_chunk_ids),
            "kg_chain": [hop.as_dict() for hop in c.kg_chain],
            "recall": {
                name: round(self.recall[name], 4) for name in (*SINGLE_SIGNALS, FUSED_SIGNAL)
            },
            "fused_only_gold_ids": list(self.fused_only_gold_ids),
        }


# ---------------------------------------------------------------------------
# Readers — standard library only.
# ---------------------------------------------------------------------------

_RFC_IN_URI = re.compile(r"rfc[-_]?(\d{3,5})", re.IGNORECASE)
# A header-field surface: Title-Cased hyphenated token, e.g. "Retry-After", "Content-Type".
_HEADER_FIELD = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:-[A-Z][A-Za-z0-9]+)+)\b")
# An HTTP status code (the classic exact-lexical lookup): 3 digits, 1xx–5xx.
_STATUS_CODE = re.compile(r"\b([1-5][0-9]{2})\b")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Every non-blank JSON object line of a file (``[]`` when the file is absent)."""
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def rfc_of_source_uri(source_uri: str) -> int | None:
    """Recover the RFC number from a chunk's Provenance ``source_uri`` (``…/rfc9110.txt``)."""
    match = _RFC_IN_URI.search(source_uri)
    return int(match.group(1)) if match else None


def load_chunks(run: Path) -> list[Chunk]:
    """Load the exported chunks (``export/records.jsonl``) as :class:`Chunk`\\ s.

    A record is a chunk when it carries a ``record_id`` and textual ``content``; its source RFC is
    read from the Provenance ``source_uri`` (the pipeline's own record of the file it chunked). A
    row with no content is skipped — a gold set can only be grounded in chunks that carry text.
    """
    chunks: list[Chunk] = []
    for row in _read_jsonl(run / "export" / "records.jsonl"):
        record_id = row.get("record_id")
        content = row.get("content")
        if not isinstance(record_id, str) or not record_id:
            continue
        if not isinstance(content, str) or not content:
            continue
        rfc: int | None = None
        prov = row.get("provenance")
        if isinstance(prov, dict):
            source_uri = prov.get("source_uri")
            if isinstance(source_uri, str) and source_uri:
                rfc = rfc_of_source_uri(source_uri)
        chunks.append(Chunk(chunk_id=record_id, content=content, rfc=rfc))
    return chunks


def load_refs(refs: Path) -> dict[int, dict[str, list[int]]]:
    """Load the independent cross-reference graph (``_refs.jsonl``) as ``rfc → {rel: [rfc, …]}``.

    Only the typed relations ``build_corpus.py`` records are kept (``obsoletes``/``updates``/
    ``references``). This graph is the gold signal — mined here, **never** fed to any retriever.
    """
    graph: dict[int, dict[str, list[int]]] = {}
    for row in _read_jsonl(refs):
        rfc = row.get("rfc")
        if not isinstance(rfc, int) or isinstance(rfc, bool):
            continue
        rels: dict[str, list[int]] = {}
        for rel in ("obsoletes", "updates", "references"):
            targets = row.get(rel)
            if isinstance(targets, list):
                rels[rel] = [t for t in targets if isinstance(t, int) and not isinstance(t, bool)]
            else:
                rels[rel] = []
        graph[rfc] = rels
    return graph


# ---------------------------------------------------------------------------
# Literal subject presence — the §5 false-positive guard.
# ---------------------------------------------------------------------------


def _rfc_surfaces(n: int) -> tuple[str, ...]:
    """The surface forms an RFC number appears as (``RFC 9110`` / ``RFC9110`` / ``rfc-9110``)."""
    return (f"rfc {n}", f"rfc{n}", f"rfc-{n}")


def subject_present(subject: str, content: str) -> bool:
    """Does ``subject`` literally appear in ``content`` (case-insensitive)?

    An RFC-number subject matches any of its surface forms (``RFC 9110`` / ``RFC9110`` /
    ``rfc-9110``) so a header line and a body citation both count; any other subject is a plain
    case-insensitive substring test. This is the guard that kills index-page / incidental-mention
    gold: a gold chunk that does not literally carry the subject is not evidence for it.
    """
    haystack = content.lower()
    match = re.fullmatch(r"rfc\s*(\d{3,5})", subject.strip(), re.IGNORECASE)
    if match:
        return any(surface in haystack for surface in _rfc_surfaces(int(match.group(1))))
    return subject.lower() in haystack


def _gold_with_subject(subject: str, chunks: Iterable[Chunk]) -> tuple[str, ...]:
    """The ids of exactly the chunks that literally contain ``subject`` (order-stable)."""
    return tuple(c.chunk_id for c in chunks if subject_present(subject, c.content))


# ---------------------------------------------------------------------------
# Candidate generation — grounded in the graph + the exported chunks.
# ---------------------------------------------------------------------------


def _chunks_by_rfc(chunks: Sequence[Chunk]) -> dict[int, list[Chunk]]:
    by_rfc: dict[int, list[Chunk]] = {}
    for chunk in chunks:
        if chunk.rfc is not None:
            by_rfc.setdefault(chunk.rfc, []).append(chunk)
    return by_rfc


def _first_match(pattern: re.Pattern[str], chunks: Iterable[Chunk]) -> str | None:
    """The first regex hit across ``chunks`` (deterministic — chunks are in a stable order)."""
    for chunk in chunks:
        m = pattern.search(chunk.content)
        if m:
            return m.group(1)
    return None


def gen_multi_hop(
    graph: dict[int, dict[str, list[int]]], by_rfc: dict[int, list[Chunk]]
) -> Iterator[QuestionCandidate]:
    """Two-hop chains ``a -obsoletes-> b -(updates|references)-> c`` with terminal ``c`` present.

    The subject is the **terminal** RFC ``c`` (the node the answer lives on); the question names
    the entry ``a``/``b`` and asks about ``c``'s content, so the chain — not a lexical hit on ``c``
    — is what leads there. Gold = the chunks of ``c`` that literally carry ``RFC c``.
    """
    for a in sorted(graph):
        for b in sorted(graph[a].get("obsoletes", [])):
            if b not in graph:
                continue
            for rel in ("updates", "references"):
                for c in sorted(graph[b].get(rel, [])):
                    if c not in by_rfc:
                        continue
                    subject = f"RFC {c}"
                    gold = _gold_with_subject(subject, by_rfc[c])
                    if not gold:
                        continue
                    yield QuestionCandidate(
                        qid=f"mh-{a}-{b}-{c}",
                        qtype="multi-hop",
                        question=(
                            f"RFC {a} obsoletes RFC {b}, which {rel} RFC {c}. "
                            f"What does the specification at the end of that chain define?"
                        ),
                        subject=subject,
                        gold_chunk_ids=gold,
                        kg_chain=(Hop(a, "obsoletes", b), Hop(b, rel, c)),
                    )


def gen_lexical(by_rfc: dict[int, list[Chunk]]) -> Iterator[QuestionCandidate]:
    """Exact header-field / status-code lookups — the lexical-critical flavour (BM25/SPLADE turf).

    The subject is a distinctive literal (a ``Title-Cased`` header field, else a 3-digit status
    code) drawn from an RFC's own chunks; gold = every chunk of that RFC literally carrying it.
    """
    for rfc in sorted(by_rfc):
        chunks = by_rfc[rfc]
        header = _first_match(_HEADER_FIELD, chunks)
        if header is not None:
            gold = _gold_with_subject(header, chunks)
            if gold:
                yield QuestionCandidate(
                    qid=f"lex-hdr-{rfc}-{header.lower()}",
                    qtype="lexical",
                    question=f"Which section of RFC {rfc} defines the '{header}' header field?",
                    subject=header,
                    gold_chunk_ids=gold,
                )
        code = _first_match(_STATUS_CODE, chunks)
        if code is not None:
            gold = _gold_with_subject(code, chunks)
            if gold:
                yield QuestionCandidate(
                    qid=f"lex-code-{rfc}-{code}",
                    qtype="lexical",
                    question=f"What does status code {code} mean in RFC {rfc}?",
                    subject=code,
                    gold_chunk_ids=gold,
                )


def gen_semantic(by_rfc: dict[int, list[Chunk]]) -> Iterator[QuestionCandidate]:
    """Paraphrased-concept questions with NO lexical overlap on the subject (dense turf).

    The subject is the RFC's own number (present in its chunks) but the **question deliberately
    never names it** — it paraphrases the topic, so only a semantic match can bridge the query to
    the gold. Gold = the chunks of that RFC literally carrying ``RFC n``.
    """
    for rfc in sorted(by_rfc):
        subject = f"RFC {rfc}"
        gold = _gold_with_subject(subject, by_rfc[rfc])
        if not gold:
            continue
        yield QuestionCandidate(
            qid=f"sem-{rfc}",
            qtype="semantic",
            question=(
                "Which document lays out the core rules and requirements for this protocol "
                "behaviour, without citing the document by number?"
            ),
            subject=subject,
            gold_chunk_ids=gold,
        )


def gen_compound(
    graph: dict[int, dict[str, list[int]]], by_rfc: dict[int, list[Chunk]]
) -> Iterator[QuestionCandidate]:
    """Two distinct RFCs linked only by a **co-referenced** third RFC (fusion turf).

    When two RFCs ``x`` and ``y`` both reference the same RFC ``z``, the shared entity ``RFC z`` is
    a bridge no single-section lookup spans. Subject = ``RFC z``; gold = the chunks of BOTH ``x``
    and ``y`` that literally carry it, so answering needs the two distant sections together.
    """
    citers_of: dict[int, list[int]] = {}
    for src in sorted(graph):
        cited = set(graph[src].get("references", [])) | set(graph[src].get("updates", []))
        for z in cited:
            citers_of.setdefault(z, []).append(src)
    for z in sorted(citers_of):
        citers = sorted({s for s in citers_of[z] if s in by_rfc})
        if len(citers) < 2:
            continue
        x, y = citers[0], citers[1]
        subject = f"RFC {z}"
        gold = _gold_with_subject(subject, [*by_rfc[x], *by_rfc[y]])
        # Genuinely compound only when BOTH sides carry the bridge.
        if not any(c.rfc == x for c in by_rfc[x] if subject_present(subject, c.content)):
            continue
        if not any(c.rfc == y for c in by_rfc[y] if subject_present(subject, c.content)):
            continue
        if not gold:
            continue
        yield QuestionCandidate(
            qid=f"cmp-{x}-{y}-via-{z}",
            qtype="compound",
            question=(
                f"RFC {x} and RFC {y} each build on a common referenced specification. "
                f"What does that shared specification require?"
            ),
            subject=subject,
            gold_chunk_ids=gold,
            kg_chain=(Hop(x, "references", z), Hop(y, "references", z)),
        )


def generate_candidates(
    chunks: Sequence[Chunk], graph: dict[int, dict[str, list[int]]]
) -> list[QuestionCandidate]:
    """Every mined candidate across the four flavours, in a deterministic, de-duplicated order."""
    by_rfc = _chunks_by_rfc(chunks)
    seen: set[str] = set()
    out: list[QuestionCandidate] = []
    generators: Iterable[Iterator[QuestionCandidate]] = (
        gen_multi_hop(graph, by_rfc),
        gen_compound(graph, by_rfc),
        gen_semantic(by_rfc),
        gen_lexical(by_rfc),
    )
    for gen in generators:
        for cand in gen:
            if cand.qid in seen:
                continue
            seen.add(cand.qid)
            out.append(cand)
    return out


# ---------------------------------------------------------------------------
# The gate — literal presence + fused-vs-single recall (the §5 hard gate).
# ---------------------------------------------------------------------------


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of ``gold`` chunk ids in the top-``k`` of ``ranked`` (``0.0`` when no gold)."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    top = set(ranked[:k])
    return len(top & gold_set) / len(gold_set)


def gate_candidate(
    cand: QuestionCandidate,
    signals: Signals,
    *,
    top_k: int,
    min_lift: float,
    chunk_index: dict[str, Chunk],
) -> GatedQuestion | None:
    """Return a :class:`GatedQuestion` iff ``cand`` clears both §5 gates, else ``None``.

    1. **Literal presence** — every gold chunk must literally carry the subject (recomputed here so
       a caller cannot smuggle a non-literal gold set past the guard). Also requires the gold set to
       be non-empty and every gold id to be a real exported chunk.
    2. **Fusion superiority** — with the real signals' rankings: ``fused recall@k`` must exceed the
       best single signal by more than ``min_lift`` **and** fused must surface ≥ 1 gold chunk in
       top-k that no single signal surfaces (``fused_only_gold``). Both must hold.
    """
    gold = cand.gold_chunk_ids
    if not gold:
        return None
    for gid in gold:
        chunk = chunk_index.get(gid)
        if chunk is None or not subject_present(cand.subject, chunk.content):
            return None

    ranked = signals.rank(cand.question, top_k)
    for name in (*SINGLE_SIGNALS, FUSED_SIGNAL):
        if name not in ranked:
            msg = f"Signals.rank omitted required signal {name!r}; got keys {sorted(ranked)}."
            raise KeyError(msg)

    recall = {
        name: recall_at_k(ranked[name], gold, top_k) for name in (*SINGLE_SIGNALS, FUSED_SIGNAL)
    }
    best_single = max(recall[name] for name in SINGLE_SIGNALS)

    gold_set = set(gold)
    fused_hits = set(ranked[FUSED_SIGNAL][:top_k]) & gold_set
    single_hits: set[str] = set()
    for name in SINGLE_SIGNALS:
        single_hits |= set(ranked[name][:top_k]) & gold_set
    fused_only = tuple(gid for gid in gold if gid in fused_hits and gid not in single_hits)

    if recall[FUSED_SIGNAL] > best_single + min_lift and fused_only:
        return GatedQuestion(candidate=cand, recall=recall, fused_only_gold_ids=fused_only)
    return None


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkResult:
    """The built benchmark: the surviving questions + the honest generated/survived counts."""

    survivors: list[GatedQuestion] = field(default_factory=list)
    generated: int = 0

    def to_json(
        self, *, run: Path, refs: Path, target: int, min_lift: float, top_k: int
    ) -> dict[str, object]:
        by_type: dict[str, int] = {}
        for q in self.survivors:
            by_type[q.candidate.qtype] = by_type.get(q.candidate.qtype, 0) + 1
        return {
            "meta": {
                "run": str(run),
                "refs": str(refs),
                "target": target,
                "min_lift": min_lift,
                "top_k": top_k,
                "generated": self.generated,
                "survived": len(self.survivors),
                "by_type": by_type,
            },
            "questions": [q.to_json() for q in self.survivors],
        }


def build_benchmark(
    chunks: Sequence[Chunk],
    graph: dict[int, dict[str, list[int]]],
    signals: Signals,
    *,
    target: int,
    min_lift: float,
    top_k: int,
) -> BenchmarkResult:
    """Mine → literal-guard → recall-gate; return the survivors (capped at ``target``).

    Candidates are gated in the deterministic generation order and accepted until ``target`` survive
    — so the signal (Qdrant) calls stop as soon as the goal is met, never running the whole cross
    product needlessly. Fewer than ``target`` survivors is a true corpus-depth result, returned as
    such (the caller reports the real number; padding is forbidden by §5).
    """
    candidates = generate_candidates(chunks, graph)
    chunk_index = {c.chunk_id: c for c in chunks}
    survivors: list[GatedQuestion] = []
    for cand in candidates:
        if len(survivors) >= target:
            break
        gated = gate_candidate(
            cand, signals, top_k=top_k, min_lift=min_lift, chunk_index=chunk_index
        )
        if gated is not None:
            survivors.append(gated)
    return BenchmarkResult(survivors=survivors, generated=len(candidates))


# ---------------------------------------------------------------------------
# The production retrieval adapter — latence_retrieval over the LIVE Qdrant.
# ---------------------------------------------------------------------------


class LatenceSignals:
    """The real :class:`Signals`: ``latence_retrieval`` over the running Qdrant (the pod path).

    Builds the four signals a question is gated on:

    * **dense** — the query is embedded (a sentence-transformer) and searched against the
      collection's dense vector via :class:`~latence_retrieval.qdrant.QdrantBackend`.
    * **splade** — the query is encoded to a sparse vector (the ADR-0053 SPLADE composition, reused
      query-side) and searched against the collection's **named sparse** vector.
    * **bm25** — exact Okapi BM25 (:class:`~latence_retrieval.bm25.Bm25Rescorer`) over the exported
      chunk pool (pool-local IDF; no second index — ADR-0048).
    * **fused** — :class:`~latence_retrieval.fuser.RrfFuser` over the three single signals, then a
      KG :class:`~latence_retrieval.multihop.MultiHopExpander` injection over the emitted
      ``graph-edges.parquet`` when present, so the multi-hop chain carries fused past the singles.

    Every heavy dependency (``latence_retrieval``, the encoders, ``qdrant-client``) is imported
    **lazily on first :meth:`rank`**, so importing this module — and running the offline test that
    injects a stand-in — never needs a pod, a GPU, or a network.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        qdrant_url: str,
        collection: str,
        dimension: int,
        dense_model: str,
        splade_model: str,
        graph_edges: Path | None = None,
        vector_name: str | None = None,
        sparse_vector_name: str | None = "sparse",
        pool: int = 100,
    ) -> None:
        self._chunks = list(chunks)
        self._qdrant_url = qdrant_url
        self._collection = collection
        self._dimension = dimension
        self._dense_model = dense_model
        self._splade_model = splade_model
        self._graph_edges = graph_edges
        self._vector_name = vector_name
        self._sparse_vector_name = sparse_vector_name
        self._pool = pool
        self._built = False
        # Lazily-built machinery (Any at the heavy-dep boundary; see module docstring).
        self._dense_backend: Any = None
        self._sparse_backend: Any = None
        self._dense_encoder: Any = None
        self._sparse_encoder: Any = None
        self._bm25: Any = None
        self._corpus_candidates: Any = None
        self._fuser: Any = None
        self._multihop: Any = None

    def _build(self) -> None:  # pragma: no cover - exercised only on the pod (real deps)
        from latence_retrieval import (
            Bm25Rescorer,
            Candidate,
            QdrantBackend,
            RrfFuser,
            resolve_sparse_encoder,
        )

        self._dense_backend = QdrantBackend(
            {
                "collection": self._collection,
                "capabilities": ["dense"],
                "dimension": self._dimension,
                "vector_name": self._vector_name,
                "url": self._qdrant_url,
            }
        )
        self._sparse_backend = QdrantBackend(
            {
                "collection": self._collection,
                "capabilities": ["sparse"],
                "sparse_vector_name": self._sparse_vector_name,
                "url": self._qdrant_url,
            }
        )
        from sentence_transformers import SentenceTransformer

        self._dense_encoder = SentenceTransformer(self._dense_model)
        self._sparse_encoder = resolve_sparse_encoder(
            {"provider": "query_sparse.splade", "config": {"model": self._splade_model}}
        )
        self._bm25 = Bm25Rescorer()
        self._corpus_candidates = [
            Candidate(id=c.chunk_id, score=0.0, text=c.content) for c in self._chunks
        ]
        self._fuser = RrfFuser()
        if self._graph_edges is not None and self._graph_edges.is_file():
            from latence_retrieval import DuckDBGraphSource, MultiHopExpander

            self._multihop = MultiHopExpander(
                graph_source=DuckDBGraphSource(str(self._graph_edges))
            )
        self._built = True

    def rank(self, query: str, top_k: int) -> dict[str, list[str]]:  # pragma: no cover - pod path
        """Rank the run's chunks by each signal for ``query`` (real Qdrant + encoders + fusion)."""
        if not self._built:
            self._build()
        from latence_retrieval import Candidate, Capability, RetrievalQuery
        from sparse_terms import sparse_vector_to_terms

        dense_vec = tuple(float(x) for x in self._dense_encoder.encode(query).tolist())
        dense_hits = self._dense_backend.search(
            RetrievalQuery(mode=Capability.DENSE, vector=dense_vec, top_k=self._pool)
        )
        sparse_terms = sparse_vector_to_terms(self._sparse_encoder.encode(query))
        splade_hits = self._sparse_backend.search(
            RetrievalQuery(
                mode=Capability.SPARSE, sparse_terms=sparse_terms, top_k=self._pool
            )
        )
        bm25_hits = self._bm25.process(query, self._corpus_candidates)[: self._pool]

        dense_list = self._as_chunk_candidates(dense_hits)
        splade_list = self._as_chunk_candidates(splade_hits)
        bm25_list = [Candidate(id=c.id, score=c.score, text=c.text) for c in bm25_hits]

        fused = self._fuser.fuse([dense_list, bm25_list, splade_list])
        if self._multihop is not None:
            fused = self._multihop.process(query, fused)

        return {
            "dense": [c.id for c in dense_list[:top_k]],
            "bm25": [c.id for c in bm25_list[:top_k]],
            "splade": [c.id for c in splade_list[:top_k]],
            FUSED_SIGNAL: [c.id for c in fused[:top_k]],
        }

    @staticmethod
    def _as_chunk_candidates(hits: Any) -> Any:  # pragma: no cover - pod path
        """Re-key backend hits to the corpus ``record_id`` so all signals fuse on one id space.

        A Qdrant point id is a derived UUIDv5; the chunk key the gold sets use is the ``record_id``
        the sink wrote into the payload. Re-key each hit to it (falling back to the point id only if
        the payload lacks it — a mis-loaded collection, surfaced not hidden).
        """
        from latence_retrieval import Candidate

        out = []
        for hit in hits:
            record_id = hit.metadata.get("record_id")
            cid = record_id if isinstance(record_id, str) and record_id else hit.id
            out.append(Candidate(id=cid, score=hit.score, text=hit.text))
        return out


def _default_dimension_hint() -> int:
    """The Granite r2 dense width the enterprise-SOTA export writes (§2's ``embedding`` 768-d)."""
    return 768


def build_real_signals(
    run: Path, chunks: Sequence[Chunk], args: argparse.Namespace
) -> LatenceSignals:
    """Construct the production :class:`LatenceSignals` from the CLI args + env (the pod path)."""
    graph_edges = run / "export" / "graph-edges.parquet"
    return LatenceSignals(
        chunks,
        qdrant_url=args.qdrant,
        collection=args.collection,
        dimension=args.dimension,
        dense_model=args.dense_model,
        splade_model=args.splade_model,
        graph_edges=graph_edges if graph_edges.is_file() else None,
        vector_name=args.vector_name,
        sparse_vector_name=args.sparse_vector_name,
        pool=args.pool,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    signals_factory: Any = None,
) -> int:
    """CLI entry (tutorial §5). ``signals_factory(run, chunks, args) → Signals`` is injectable.

    With no ``signals_factory`` the production :func:`build_real_signals` is used — the real
    ``latence_retrieval`` path against the live Qdrant. A test passes a stand-in factory to drive
    the whole mine → gate → emit pipeline offline.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path, help="the run dir (_latence/runs/<run_id>)")
    ap.add_argument("--refs", required=True, type=Path, help="the independent _refs.jsonl")
    ap.add_argument("--out", required=True, type=Path, help="output benchmark JSON path")
    ap.add_argument("--target", type=int, default=40, help="stop once this many questions survive")
    ap.add_argument(
        "--min-lift", type=float, default=0.24, dest="min_lift",
        help="fused-vs-best-single recall margin",
    )
    ap.add_argument("--top-k", type=int, default=10, dest="top_k", help="recall@k cut-off")
    ap.add_argument(
        "--qdrant",
        default=os.environ.get("LATENCE_QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL (default $LATENCE_QDRANT_URL)",
    )
    ap.add_argument("--collection", default="rfc", help="the Qdrant collection to search")
    ap.add_argument(
        "--dimension", type=int, default=_default_dimension_hint(), help="dense vector width",
    )
    ap.add_argument(
        "--dense-model", dest="dense_model",
        default="ibm-granite/granite-embedding-278m-multilingual", help="dense query encoder",
    )
    ap.add_argument(
        "--splade-model", dest="splade_model", default="naver/splade-v3",
        help="SPLADE query encoder",
    )
    ap.add_argument(
        "--vector-name", dest="vector_name", default=None,
        help="named dense vector (else the unnamed vector)",
    )
    ap.add_argument(
        "--sparse-vector-name", dest="sparse_vector_name", default="sparse",
        help="named sparse vector",
    )
    ap.add_argument("--pool", type=int, default=100, help="first-stage pool size per single signal")
    args = ap.parse_args(argv)

    if not args.run.is_dir():
        print(f"error: run dir not found: {args.run}", file=sys.stderr)
        return 2
    if not args.refs.is_file():
        print(f"error: _refs.jsonl not found: {args.refs}", file=sys.stderr)
        return 2

    chunks = load_chunks(args.run)
    graph = load_refs(args.refs)
    if not chunks:
        print(f"error: no exported chunks with content under {args.run}/export", file=sys.stderr)
        return 2

    factory = signals_factory if signals_factory is not None else build_real_signals
    signals = factory(args.run, chunks, args)

    result = build_benchmark(
        chunks, graph, signals, target=args.target, min_lift=args.min_lift, top_k=args.top_k
    )
    payload = result.to_json(
        run=args.run, refs=args.refs, target=args.target, min_lift=args.min_lift, top_k=args.top_k
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    survived = len(result.survivors)
    print(f"build_questions: generated {result.generated} candidates, {survived} survived the gate")
    by_type: dict[str, int] = {}
    for q in result.survivors:
        by_type[q.candidate.qtype] = by_type.get(q.candidate.qtype, 0) + 1
    for qtype, count in sorted(by_type.items()):
        print(f"  {qtype}: {count}")
    if survived < 30:
        print(
            f"NOTE: only {survived} questions survived (<30). This is a true corpus-depth result, "
            "not a failure — report it; do not pad with filler (tutorial §5).",
            file=sys.stderr,
        )
    print(f"wrote {survived} gated questions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
