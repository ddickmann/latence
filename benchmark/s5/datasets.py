# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Per-dataset query loading + gold **naming translation** — where each benchmark's idiom ends.

:mod:`benchmark.s5.goldmap` resolves references to chunk ids; this module is the layer above it
that turns each dataset's own way of naming evidence into that common
:class:`~benchmark.s5.goldmap.GoldRef` vocabulary. Every translation below is a decision that was
*verified against the data*, not assumed — the verification is recorded with it because a wrong
translation is invisible downstream:

``vidoseek``
    ``meta_info.file_name`` is ``<sha>.pdf``; S1 wrote it into the corpus as ``pdf__<sha>.md``.
    ``meta_info.reference_page`` is **1-based** — measured, not assumed: over the first 200
    examples, treating it as 1-based put the reference answer's key terms on the named page
    99/147 times versus 15/147 for the 0-based reading.

``ohr_bench``
    ``doc_name`` is ``<domain>/<doc>`` and became ``<domain>__<doc>.md``. ``page_idx`` is
    **0-based** — same measurement: the row's own ``gt_text`` opens page ``page_idx + 1``
    308/396 times versus 67/396 for the as-is reading.

``multihop_rag``
    ``evidence_list[].title`` names an article in ``corpus.json``, which S1 wrote out as
    ``{slugify(title)}-{sha256("# <title>\\n\\n<body>\\n")[:12]}.md``
    (:func:`latence_benchmark.campaign.s3_lib.multihop_rag_articles`). The filename is therefore
    *derived*, not fuzzy-matched — and ``tests/test_datasets.py`` pins the derivation against
    that very function, so a change in the prep breaks the test rather than the recall number.
    Each distinct evidence article is one chain slot.

``wiki2multihop`` (2Wiki)
    ``supporting_facts`` are ``(title, sentence_index)`` pairs. The corpus unit is the *whole
    paragraph* (one ``.md`` per distinct ``[title, [sentences]]`` in the dev contexts), so a
    sentence index cannot narrow further and the paragraph is the finest gold available. The
    filename is derived from the question's **own** context row, which is what makes the title
    unambiguous across the 12,576-question pool. The gold chain is the ``evidences``
    ``(subject, relation, object)`` triple order projected onto those paragraphs.

``uda``
    ``evidence`` is a literal sentence (or table row) from a known document
    (``fin_docs__<key>.md`` / ``paper_docs__<key>.md``), so it resolves by
    whitespace-normalised containment inside that document's chunks — the most precise mapping
    of the seven, and the only one that does not depend on an index at all.

``musique`` / ``hotpotqa``
    ``paragraphs[].file`` is already the export's ``provenance.file_name``; the supporting ones
    are the gold set, one chain slot each.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from benchmark.s5.goldmap import GoldQuery, GoldRef, RefKind

_BENCH_SRC = Path(__file__).resolve().parents[2] / "packages" / "latence-benchmark" / "src"
if str(_BENCH_SRC) not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, str(_BENCH_SRC))

from latence_benchmark.campaign.s3_lib import slugify  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SOTA_CORPUS = REPO_ROOT / "benchmark" / "sota-corpus"
S3_EXPORTS = SOTA_CORPUS / "s3-exports"
S2_OUTPUT = SOTA_CORPUS / "s2-output"
OFFICIAL_DATA = REPO_ROOT / "benchmark" / "official-data"


def prep_filename(title: str, body: str) -> str:
    """The S1 corpus-prep filename for a ``(title, body)`` document — the derivation, not a guess.

    Mirrors ``latence_benchmark.campaign.s3_lib._prep_file``: the markdown is
    ``"# {title}\\n\\n{body}\\n"`` (both stripped, empty title → ``Untitled``) and the name is
    ``{slugify(title)}-{sha256(content)[:12]}.md``. Pinned against the real prep functions by
    ``tests/test_datasets.py`` — this is the one place the harness reproduces pipeline logic, so
    it is the one place a test asserts byte-identical agreement.
    """
    content = f"# {title.strip() or 'Untitled'}\n\n{body.strip()}\n"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{slugify(title)}-{digest[:12]}.md"


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset's export, page artifacts and queries live, and how to read them."""

    name: str
    export_dir: Path
    queries: Callable[[], list[GoldQuery]]
    s2_dataset: str | None = None
    single_hop: bool = False

    @property
    def s2_root(self) -> Path | None:
        """The S2 output root when this dataset has per-page artifacts, else ``None``."""
        return S2_OUTPUT if self.s2_dataset else None


# -- vidoseek ---------------------------------------------------------------------------------


def load_vidoseek() -> list[GoldQuery]:
    """1,142 ViDoSeek queries; gold is ``(pdf__<sha>.md, 1-based reference_page)``."""
    raw = json.loads((SOTA_CORPUS / "vidoseek" / "raw" / "vidoseek.json").read_text())
    out: list[GoldQuery] = []
    for example in raw["examples"]:
        meta = example.get("meta_info") or {}
        stem = str(meta.get("file_name", "")).removesuffix(".pdf")
        if not stem:
            continue
        file_name = f"pdf__{stem}.md"
        pages = [int(p) for p in (meta.get("reference_page") or [])]
        refs = tuple(
            GoldRef(kind=RefKind.PAGE, file_name=file_name, page=page, slot=0) for page in pages
        )
        out.append(
            GoldQuery(
                query_id=str(example["uid"]),
                question=str(example["query"]),
                refs=refs,
                question_type=str(meta.get("query_type") or ""),
            )
        )
    return out


# -- multihop_rag -----------------------------------------------------------------------------


def multihop_rag_title_map() -> dict[str, str]:
    """``article title → prep filename`` over MultiHop-RAG's 609-article ``corpus.json``."""
    articles = json.loads((SOTA_CORPUS / "multihop_rag" / "raw" / "corpus.json").read_text())
    out: dict[str, str] = {}
    for article in articles:
        body = str(article.get("body", "") or "")
        if not body.strip():
            continue
        out[str(article.get("title", "") or "")] = prep_filename(
            str(article.get("title", "") or ""), body
        )
    return out


def load_multihop_rag() -> list[GoldQuery]:
    """2,556 MultiHop-RAG queries; each distinct evidence article is one gold chain slot."""
    titles = multihop_rag_title_map()
    raw = json.loads((SOTA_CORPUS / "multihop_rag" / "raw" / "MultiHopRAG.json").read_text())
    out: list[GoldQuery] = []
    for index, item in enumerate(raw):
        refs: list[GoldRef] = []
        seen: set[str] = set()
        for evidence in item.get("evidence_list") or []:
            title = str(evidence.get("title", "") or "")
            # An unmapped title keeps a reference on purpose: it MUST count against the
            # resolution rate rather than vanish from the denominator.
            file_name = titles.get(title, prep_filename(title, ""))
            if file_name in seen:
                continue
            refs.append(GoldRef(kind=RefKind.FILE, file_name=file_name, slot=len(seen)))
            seen.add(file_name)
        out.append(
            GoldQuery(
                query_id=f"mhrag-{index:05d}",
                question=str(item["query"]),
                refs=tuple(refs),
                question_type=str(item.get("question_type") or ""),
            )
        )
    return out


# -- wiki2multihop (2Wiki) --------------------------------------------------------------------


def load_wiki2multihop() -> list[GoldQuery]:
    """12,576 2Wiki dev queries; gold paragraphs come from the question's own context row."""
    table = pq.read_table(  # type: ignore[no-untyped-call]
        SOTA_CORPUS / "wiki2multihop" / "raw" / "dev.parquet",
        columns=["_id", "type", "question", "context", "supporting_facts", "evidences"],
    )
    out: list[GoldQuery] = []
    for row in table.to_pylist():
        paragraphs: dict[str, str] = {}
        for entry in json.loads(row["context"]):
            title, sentences = str(entry[0]), [str(s) for s in entry[1]]
            text = " ".join(s.strip() for s in sentences if s.strip())
            if text:
                paragraphs[title] = prep_filename(title, text)
        chain_titles: list[str] = []
        for triple in json.loads(row["evidences"]) or []:
            for endpoint in (str(triple[0]), str(triple[2])):
                if endpoint in paragraphs and endpoint not in chain_titles:
                    chain_titles.append(endpoint)
        refs: list[GoldRef] = []
        seen: set[str] = set()
        for fact in json.loads(row["supporting_facts"]) or []:
            title = str(fact[0])
            file_name = paragraphs.get(title) or prep_filename(title, "")
            if file_name in seen:
                continue
            seen.add(file_name)
            slot = chain_titles.index(title) if title in chain_titles else len(chain_titles)
            refs.append(GoldRef(kind=RefKind.FILE, file_name=file_name, slot=slot))
        out.append(
            GoldQuery(
                query_id=str(row["_id"]),
                question=str(row["question"]),
                refs=tuple(refs),
                question_type=str(row["type"]),
            )
        )
    return out


# -- ohr_bench --------------------------------------------------------------------------------


def load_ohr_bench() -> list[GoldQuery]:
    """OHR-Bench QAs; gold is ``(<domain>__<doc>.md, page_idx + 1)`` — page_idx is 0-based."""
    table = pq.read_table(  # type: ignore[no-untyped-call]
        SOTA_CORPUS / "ohr_bench" / "raw" / "OHR-Bench.parquet",
        columns=["doc_name", "page_idx", "qas"],
    )
    out: list[GoldQuery] = []
    for row in table.to_pylist():
        file_name = f"{str(row['doc_name']).replace('/', '__')}.md"
        page = int(row["page_idx"]) + 1
        qas = row["qas"] or {}
        questions = list(qas.get("questions") or [])
        ids = list(qas.get("ID") or [])
        for position, question in enumerate(questions):
            query_id = str(ids[position]) if position < len(ids) else f"{file_name}#{position}"
            out.append(
                GoldQuery(
                    query_id=query_id,
                    question=str(question),
                    refs=(GoldRef(kind=RefKind.PAGE, file_name=file_name, page=page, slot=0),),
                    question_type="single_hop",
                )
            )
    return out


# -- uda --------------------------------------------------------------------------------------


_UDA_PAGE = re.compile(r"/page_(\d+)\.pdf")


def _uda_paper_evidence(item: dict[str, object]) -> Iterator[str]:
    """The ``raw_evidence`` sentences of a UDA paper_tab QA record."""
    evidence = item.get("evidence")
    if not isinstance(evidence, list):
        return
    for entry in evidence:
        if not isinstance(entry, dict):
            continue
        for value in entry.get("raw_evidence") or []:
            if isinstance(value, str) and value.strip():
                yield value


def load_uda() -> list[GoldQuery]:
    """UDA fin + paper_tab QAs — two different gold idioms, mapped two different ways.

    **fin** gold is a PAGE, taken from the question's own ``q_uid``
    (``ADI/2009/page_49.pdf-1`` → page 49 of ``fin_docs__ADI_2009.md``, 1-based, verified
    against the reconstructed page artifacts). The record's ``evidence`` text is *not* usable
    for mapping: it is FinQA's preprocessed form (lowercased, spaced punctuation,
    ``$ 3.8 million``) and matches the OCR'd markdown essentially never — a measured 0%
    resolution, which is exactly the silent-zero failure this harness reports instead of hiding.

    **paper_tab** gold is the ``raw_evidence`` sentences, which are verbatim paper text and so
    resolve by containment inside ``paper_docs__<arxiv id>.md``.
    """
    out: list[GoldQuery] = []
    fin_path = SOTA_CORPUS / "uda" / "raw" / "extended_qa_info" / "fin_qa.json"
    if fin_path.is_file():
        for doc_key, items in json.loads(fin_path.read_text()).items():
            file_name = f"fin_docs__{doc_key}.md"
            for position, item in enumerate(items):
                q_uid = str(item.get("q_uid") or f"{doc_key}#{position}")
                match = _UDA_PAGE.search(q_uid)
                refs: tuple[GoldRef, ...] = (
                    (GoldRef(kind=RefKind.PAGE, file_name=file_name, page=int(match.group(1))),)
                    if match
                    else ()
                )
                out.append(
                    GoldQuery(
                        query_id=q_uid,
                        question=str(item.get("question", "")),
                        refs=refs,
                        question_type="fin",
                    )
                )
    paper_path = SOTA_CORPUS / "uda" / "raw" / "extended_qa_info" / "paper_tab_qa.json"
    if paper_path.is_file():
        for doc_key, items in json.loads(paper_path.read_text()).items():
            file_name = f"paper_docs__{doc_key}.md"
            for position, item in enumerate(items):
                refs = tuple(
                    GoldRef(kind=RefKind.TEXT, file_name=file_name, text=text)
                    for text in _uda_paper_evidence(item)
                )
                out.append(
                    GoldQuery(
                        query_id=str(item.get("q_uid") or f"{doc_key}#{position}"),
                        question=str(item.get("question", "")),
                        refs=refs,
                        question_type="paper_tab",
                    )
                )
    return out


# -- musique / hotpotqa (reused M6 exports) ----------------------------------------------------


def _load_supporting_paragraph_queries(path: Path, prefix: str) -> list[GoldQuery]:
    out: list[GoldQuery] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            refs = tuple(
                GoldRef(kind=RefKind.FILE, file_name=str(paragraph["file"]), slot=slot)
                for slot, paragraph in enumerate(
                    p for p in record["paragraphs"] if p.get("is_supporting")
                )
            )
            out.append(
                GoldQuery(
                    query_id=f"{prefix}-{record['id']}",
                    question=str(record["question"]),
                    refs=refs,
                )
            )
    return out


def load_musique() -> list[GoldQuery]:
    """MuSiQue dev (the reused M6 question set); gold = the supporting paragraph files."""
    return _load_supporting_paragraph_queries(
        OFFICIAL_DATA / "musique-full" / "musique-full-dev.jsonl", "musique"
    )


def load_hotpotqa() -> list[GoldQuery]:
    """HotpotQA dev (the reused M6 question set); gold = the supporting paragraph files."""
    return _load_supporting_paragraph_queries(
        OFFICIAL_DATA / "hotpot-full" / "hotpot-full-dev.jsonl", "hotpot"
    )


DATASETS: dict[str, DatasetSpec] = {
    "vidoseek": DatasetSpec(
        name="vidoseek",
        export_dir=S3_EXPORTS / "vidoseek",
        queries=load_vidoseek,
        s2_dataset="vidoseek",
        single_hop=True,
    ),
    "multihop_rag": DatasetSpec(
        name="multihop_rag",
        export_dir=S3_EXPORTS / "multihop_rag",
        queries=load_multihop_rag,
    ),
    "wiki2multihop": DatasetSpec(
        name="wiki2multihop",
        export_dir=S3_EXPORTS / "wiki2multihop",
        queries=load_wiki2multihop,
    ),
    "ohr_bench": DatasetSpec(
        name="ohr_bench",
        export_dir=S3_EXPORTS / "ohr_bench",
        queries=load_ohr_bench,
        s2_dataset="ohr_bench",
        single_hop=True,
    ),
    "uda": DatasetSpec(
        name="uda",
        export_dir=S3_EXPORTS / "uda",
        queries=load_uda,
        s2_dataset="uda",
        single_hop=True,
    ),
    "musique": DatasetSpec(
        name="musique",
        export_dir=OFFICIAL_DATA / "musique-full" / "export",
        queries=load_musique,
    ),
    "hotpotqa": DatasetSpec(
        name="hotpotqa",
        export_dir=OFFICIAL_DATA / "hotpot-full" / "export",
        queries=load_hotpotqa,
    ),
}

#: The single-hop CONTROL group: hypergraph/CoE gains here should be ~zero. A large gain is
#: evidence of leakage in the harness, not a retrieval win (S5 ticket).
CONTROL_DATASETS = frozenset(name for name, spec in DATASETS.items() if spec.single_hop)


__all__ = [
    "CONTROL_DATASETS",
    "DATASETS",
    "OFFICIAL_DATA",
    "S2_OUTPUT",
    "S3_EXPORTS",
    "SOTA_CORPUS",
    "DatasetSpec",
    "prep_filename",
]
