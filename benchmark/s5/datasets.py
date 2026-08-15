# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
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
    308/396 times versus 67/396 for the as-is reading. Two upstream naming facts are handled by
    explicit, verified aliases rather than by matching (:data:`OHR_DOMAIN_ALIASES`,
    :data:`OHR_DOCUMENT_ALIASES`), and one whole QA domain was never acquired into this corpus,
    so its queries are EXCLUDED with a stated reason (:data:`OHR_UNACQUIRED_DOMAINS`).

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
    the query sets are the ``fin`` / ``paper_tab`` **parquets**, whose ``doc_name`` column is
    already the corpus key (``ADI_2009`` → ``fin_docs__ADI_2009.md``, ``1912.01214`` →
    ``paper_docs__1912.01214.md``): a direct, total mapping, verified 8,190/8,190 and 393/393.
    The ``extended_qa_info/*.json`` samples that an earlier revision matched by evidence
    *sentence* are not the query set (``paper_tab_qa.json`` holds a single record) and their
    ``raw_evidence`` is FinQA/QASPER-preprocessed text that does not survive OCR verbatim — that
    approach is abandoned, not tuned.

``musique`` / ``hotpotqa``
    ``paragraphs[].file`` is already the export's ``provenance.file_name``; the supporting ones
    are the gold set, one chain slot each.

**Where the files are is an argument, not a constant.** Every loader and the dataset table take
a :class:`CorpusLayout`; :data:`DEFAULT_LAYOUT` is this checkout, so the module constants
(:data:`SOTA_CORPUS`, :data:`S3_EXPORTS`, :data:`S2_OUTPUT`, :data:`OFFICIAL_DATA`) and every
existing call site are unchanged to the byte — but a caller operating on a different checkout
(the ``s5-retrieval`` campaign stage, which is handed a ``repo_root``) can now say so instead of
being silently pointed back here.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
from benchmark.s5.goldmap import ExclusionReason, GoldQuery, GoldRef, RefKind

if TYPE_CHECKING:
    from collections.abc import Callable

_BENCH_SRC = Path(__file__).resolve().parents[2] / "packages" / "latence-benchmark" / "src"
if str(_BENCH_SRC) not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, str(_BENCH_SRC))

from latence_benchmark.campaign.s3_lib import slugify  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CorpusLayout:
    """The two roots every S5 input hangs off — passed in, never reached for.

    These were four module constants, which made the harness's inputs a property of *where its
    source file sits* rather than of what it was asked to measure: a campaign whose root is not
    this checkout could not redirect them at all, and the campaign stage (``s5-retrieval``) is
    exactly the caller that has to. One struct, two roots, the rest derived:

    * ``sota_corpus`` — the S1/S2/S3 campaign corpus (``raw/`` acquisitions, ``s2-output/``
      page artifacts, ``s3-exports/`` per-dataset exports),
    * ``official_data`` — the reused M6 exports + question sets (MuSiQue, HotpotQA), which
      live outside the campaign corpus because the campaign did not produce them.

    :data:`DEFAULT_LAYOUT` is this checkout, so every default-argument call site keeps reading
    byte-identical paths; the stage builds its own from ``CampaignContext.repo_root``.
    """

    sota_corpus: Path
    official_data: Path

    @classmethod
    def under(cls, repo_root: Path) -> CorpusLayout:
        """The standard layout of a latence-framework checkout at ``repo_root``."""
        return cls(
            sota_corpus=repo_root / "benchmark" / "sota-corpus",
            official_data=repo_root / "benchmark" / "official-data",
        )

    @property
    def s3_exports(self) -> Path:
        """Per-dataset S3 export directories (``corpus.parquet``, bm25, graph, hyperedges)."""
        return self.sota_corpus / "s3-exports"

    @property
    def s2_output(self) -> Path:
        """The S2 OCR output root (``pages/<dataset>/``, ``markdown/<dataset>/``)."""
        return self.sota_corpus / "s2-output"


#: This checkout — the layout every loader defaults to, so the module constants below (and the
#: behaviour of every existing caller) are unchanged.
DEFAULT_LAYOUT = CorpusLayout.under(REPO_ROOT)

SOTA_CORPUS = DEFAULT_LAYOUT.sota_corpus
S3_EXPORTS = DEFAULT_LAYOUT.s3_exports
S2_OUTPUT = DEFAULT_LAYOUT.s2_output
OFFICIAL_DATA = DEFAULT_LAYOUT.official_data


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
    """Where a dataset's export, page artifacts and queries live, and how to read them.

    Every path is resolved by :func:`dataset_specs` from one :class:`CorpusLayout`, so a spec
    never derives a location itself. ``s2_root`` is ``None`` for the datasets that have no
    per-page artifacts (the text-native sets), which is what turns the page reconstruction off.
    """

    name: str
    export_dir: Path
    queries: Callable[[], list[GoldQuery]]
    s2_root: Path | None = None
    single_hop: bool = False


# -- vidoseek ---------------------------------------------------------------------------------


def load_vidoseek(layout: CorpusLayout = DEFAULT_LAYOUT) -> list[GoldQuery]:
    """1,142 ViDoSeek queries; gold is ``(pdf__<sha>.md, 1-based reference_page)``."""
    raw = json.loads((layout.sota_corpus / "vidoseek" / "raw" / "vidoseek.json").read_text())
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


def multihop_rag_title_map(layout: CorpusLayout = DEFAULT_LAYOUT) -> dict[str, str]:
    """``article title → prep filename`` over MultiHop-RAG's 609-article ``corpus.json``."""
    articles = json.loads(
        (layout.sota_corpus / "multihop_rag" / "raw" / "corpus.json").read_text()
    )
    out: dict[str, str] = {}
    for article in articles:
        body = str(article.get("body", "") or "")
        if not body.strip():
            continue
        out[str(article.get("title", "") or "")] = prep_filename(
            str(article.get("title", "") or ""), body
        )
    return out


def load_multihop_rag(layout: CorpusLayout = DEFAULT_LAYOUT) -> list[GoldQuery]:
    """2,556 MultiHop-RAG queries; each distinct evidence article is one gold chain slot."""
    titles = multihop_rag_title_map(layout)
    raw = json.loads(
        (layout.sota_corpus / "multihop_rag" / "raw" / "MultiHopRAG.json").read_text()
    )
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


def load_wiki2multihop(layout: CorpusLayout = DEFAULT_LAYOUT) -> list[GoldQuery]:
    """12,576 2Wiki dev queries; gold paragraphs come from the question's own context row."""
    table = pq.read_table(  # type: ignore[no-untyped-call]
        layout.sota_corpus / "wiki2multihop" / "raw" / "dev.parquet",
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


#: OHR-Bench QA domain → the domain prefix S1 actually wrote into the corpus. The QA parquet
#: calls the arXiv subset ``paper/`` while the acquired PDFs live under ``academic/``; the alias
#: is exact (all 10 ``paper/`` documents have an ``academic__<doc>.md`` in the export, verified
#: 2026-08-09), so it is a key rename, not a match.
OHR_DOMAIN_ALIASES: dict[str, str] = {"paper": "academic"}

#: Individual documents whose name in ``OHR-Bench.parquet`` differs from the name in the source
#: ``pdfs.zip`` the corpus was built from — an upstream inconsistency, one document at the time
#: of writing. The pairing is not inferred: the zip entry is literally
#: ``textbook/Triangulated.pdf_145.pdf`` (the long title was truncated when the archive was
#: built) and the OCR'd markdown reproduces this row's ``gt_text`` verbatim. Keys are NFC —
#: :func:`ohr_bench_file_name` normalises before lookup because the QA parquet stores this
#: particular name decomposed (``Frédéric``).
OHR_DOCUMENT_ALIASES: dict[str, str] = {
    unicodedata.normalize(
        "NFC",
        "textbook/Triangulated_Categories_of_Mixed_Motives_"
        "(Denis-Charles_Cisinski_,_Frédéric_Déglise)_(Z-Library).pdf_145",
    ): "textbook__Triangulated.pdf_145.md",
}

#: OHR-Bench QA domains that were never acquired or OCR'd into this corpus. ``notes/`` is the
#: GNHK handwriting set — 288 documents, 498 questions — and the S3 export contains not one of
#: them (corpus domains: textbook 504, news 279, administration 146, law 95, manual 87,
#: academic 85, finance 65). Those queries are unanswerable here for a coverage reason, so they
#: are excluded WITH that reason rather than counted as retrieval misses.
OHR_UNACQUIRED_DOMAINS = frozenset({"notes"})


def ohr_bench_file_name(doc_name: str) -> str:
    """``<domain>/<doc>`` → the export's ``provenance.file_name``, aliases applied.

    A pure key derivation — no matching of any kind: an exact document alias first (compared in
    NFC, since Unicode composition is a storage detail of the parquet, not a difference in the
    name), else the domain segment is renamed through :data:`OHR_DOMAIN_ALIASES` and the
    separator becomes ``__``.
    """
    aliased = OHR_DOCUMENT_ALIASES.get(unicodedata.normalize("NFC", doc_name))
    if aliased is not None:
        return aliased
    domain, separator, rest = doc_name.partition("/")
    if not separator:
        return f"{doc_name}.md"
    return f"{OHR_DOMAIN_ALIASES.get(domain, domain)}__{rest.replace('/', '__')}.md"


def ohr_bench_exclusion(doc_name: str) -> ExclusionReason | None:
    """The reason ``doc_name`` cannot be answered against this corpus, or ``None``."""
    domain, _, _ = doc_name.partition("/")
    if domain in OHR_UNACQUIRED_DOMAINS:
        return ExclusionReason.CORPUS_MISSING_DOMAIN
    return None


def load_ohr_bench(layout: CorpusLayout = DEFAULT_LAYOUT) -> list[GoldQuery]:
    """OHR-Bench QAs; gold is ``(<domain>__<doc>.md, page_idx + 1)`` — page_idx is 0-based.

    The page is a refinement of a document reference that is itself certain, so the reference
    allows the document-level fallback: the handful of documents whose S2 page reconstruction
    does not reproduce the assembled markdown keep *coarser* gold (labelled ``document`` in the
    granularity mix) instead of being dropped from the denominator.
    """
    table = pq.read_table(  # type: ignore[no-untyped-call]
        layout.sota_corpus / "ohr_bench" / "raw" / "OHR-Bench.parquet",
        columns=["doc_name", "page_idx", "qas"],
    )
    out: list[GoldQuery] = []
    for row in table.to_pylist():
        doc_name = str(row["doc_name"])
        file_name = ohr_bench_file_name(doc_name)
        exclusion = ohr_bench_exclusion(doc_name)
        page = int(row["page_idx"]) + 1
        qas = row["qas"] or {}
        questions = list(qas.get("questions") or [])
        ids = list(qas.get("ID") or [])
        ref = GoldRef(
            kind=RefKind.PAGE,
            file_name=file_name,
            page=page,
            slot=0,
            allow_document_fallback=True,
        )
        for position, question in enumerate(questions):
            query_id = str(ids[position]) if position < len(ids) else f"{file_name}#{position}"
            out.append(
                GoldQuery(
                    query_id=query_id,
                    question=str(question),
                    refs=(ref,),
                    question_type="single_hop",
                    exclusion=exclusion,
                )
            )
    return out


# -- uda --------------------------------------------------------------------------------------


_UDA_PAGE = re.compile(r"/page_(\d+)\.pdf")

#: UDA query set → the corpus filename prefix S1 wrote for that half's source documents.
UDA_SPLIT_PREFIXES: dict[str, str] = {"fin": "fin_docs__", "paper_tab": "paper_docs__"}


def uda_file_name(split: str, doc_name: str) -> str:
    """``(split, doc_name)`` → the export's ``provenance.file_name``.

    The parquet's ``doc_name`` **is** the corpus key: ``ADI_2009`` → ``fin_docs__ADI_2009.md``,
    ``1912.01214`` → ``paper_docs__1912.01214.md``. Total on both halves (8,190/8,190 fin rows,
    393/393 paper_tab rows resolve to a file the export contains, verified 2026-08-09).
    """
    return f"{UDA_SPLIT_PREFIXES[split]}{doc_name}.md"


def uda_page_from_q_uid(q_uid: str) -> int | None:
    """The 1-based source-PDF page a UDA **fin** ``q_uid`` names, or ``None``.

    ``ADI/2009/page_49.pdf-1`` → 49. Every one of the 8,190 fin rows carries such a page; the
    paper_tab ``q_uid`` is a bare sha1 and yields ``None``, which is why that half is
    document-level gold rather than a page reference that could never resolve.
    """
    match = _UDA_PAGE.search(q_uid)
    return int(match.group(1)) if match else None


def _uda_rows(split: str, layout: CorpusLayout) -> list[dict[str, Any]]:
    """The ``(doc_name, q_uid, question)`` rows of a UDA split, in shard-name order."""
    shards = sorted((layout.sota_corpus / "uda" / "raw" / split).glob("test_*.parquet"))
    rows: list[dict[str, Any]] = []
    for shard in shards:
        table = pq.read_table(shard, columns=["doc_name", "q_uid", "question"])  # type: ignore[no-untyped-call]
        rows.extend(table.to_pylist())
    return rows


def load_uda(layout: CorpusLayout = DEFAULT_LAYOUT) -> list[GoldQuery]:
    """UDA fin + paper_tab QAs, keyed on the parquets' own ``doc_name``.

    **fin** keeps the finer PAGE gold wherever it is genuinely available: the page comes from
    the question's own ``q_uid`` (:func:`uda_page_from_q_uid`) and the reference allows the
    document-level fallback, so the 74 financial reports whose S2 page reconstruction does not
    reproduce the assembled markdown resolve at *document* granularity instead of vanishing from
    the denominator. The report states the mix; it never implies uniform page precision.

    **paper_tab** is document-level: its ``q_uid`` carries no page, and the ``raw_evidence``
    sentence matching an earlier revision used resolved 10% because that text is QASPER's
    preprocessed form, not the OCR'd markdown.
    """
    out: list[GoldQuery] = []
    for row in _uda_rows("fin", layout):
        q_uid = str(row["q_uid"])
        out.append(
            GoldQuery(
                query_id=q_uid,
                question=str(row["question"]),
                refs=(
                    GoldRef(
                        kind=RefKind.PAGE,
                        file_name=uda_file_name("fin", str(row["doc_name"])),
                        page=uda_page_from_q_uid(q_uid),
                        slot=0,
                        allow_document_fallback=True,
                    ),
                ),
                question_type="fin",
            )
        )
    for row in _uda_rows("paper_tab", layout):
        out.append(
            GoldQuery(
                query_id=str(row["q_uid"]),
                question=str(row["question"]),
                refs=(
                    GoldRef(
                        kind=RefKind.FILE,
                        file_name=uda_file_name("paper_tab", str(row["doc_name"])),
                        slot=0,
                    ),
                ),
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


def load_musique(layout: CorpusLayout = DEFAULT_LAYOUT) -> list[GoldQuery]:
    """MuSiQue dev (the reused M6 question set); gold = the supporting paragraph files."""
    return _load_supporting_paragraph_queries(
        layout.official_data / "musique-full" / "musique-full-dev.jsonl", "musique"
    )


def load_hotpotqa(layout: CorpusLayout = DEFAULT_LAYOUT) -> list[GoldQuery]:
    """HotpotQA dev (the reused M6 question set); gold = the supporting paragraph files."""
    return _load_supporting_paragraph_queries(
        layout.official_data / "hotpot-full" / "hotpot-full-dev.jsonl", "hotpot"
    )


# -- the dataset table -------------------------------------------------------------------------


def dataset_specs(layout: CorpusLayout = DEFAULT_LAYOUT) -> dict[str, DatasetSpec]:
    """Every dataset the S5 matrix knows, resolved against ``layout``.

    The one place a dataset name becomes paths. Loaders are bound to the same layout here
    (``partial``), so a spec's queries and its export can never come from different roots —
    which is the failure a campaign root redirecting only *some* of the constants would have
    produced silently.
    """
    exports = layout.s3_exports
    official = layout.official_data
    return {
        "vidoseek": DatasetSpec(
            name="vidoseek",
            export_dir=exports / "vidoseek",
            queries=partial(load_vidoseek, layout),
            s2_root=layout.s2_output,
            single_hop=True,
        ),
        "multihop_rag": DatasetSpec(
            name="multihop_rag",
            export_dir=exports / "multihop_rag",
            queries=partial(load_multihop_rag, layout),
        ),
        "wiki2multihop": DatasetSpec(
            name="wiki2multihop",
            export_dir=exports / "wiki2multihop",
            queries=partial(load_wiki2multihop, layout),
        ),
        "ohr_bench": DatasetSpec(
            name="ohr_bench",
            export_dir=exports / "ohr_bench",
            queries=partial(load_ohr_bench, layout),
            s2_root=layout.s2_output,
            single_hop=True,
        ),
        "uda": DatasetSpec(
            name="uda",
            export_dir=exports / "uda",
            queries=partial(load_uda, layout),
            s2_root=layout.s2_output,
            single_hop=True,
        ),
        "musique": DatasetSpec(
            name="musique",
            export_dir=official / "musique-full" / "export",
            queries=partial(load_musique, layout),
        ),
        "hotpotqa": DatasetSpec(
            name="hotpotqa",
            export_dir=official / "hotpot-full" / "export",
            queries=partial(load_hotpotqa, layout),
        ),
    }


#: The default-layout table — the names are the vocabulary (``--datasets``, the campaign's
#: ``s5.datasets``), so callers that only need the NAMES never build a redirected table.
DATASETS: dict[str, DatasetSpec] = dataset_specs()

#: Every known dataset name, in matrix order. Layout-independent, so plan validation (CLI and
#: campaign stage alike) never has to resolve a single path.
DATASET_NAMES: tuple[str, ...] = tuple(DATASETS)

#: The single-hop CONTROL group: hypergraph/CoE gains here should be ~zero. A large gain is
#: evidence of leakage in the harness, not a retrieval win (S5 ticket). Derived from the table's
#: own flag and layout-independent — which dataset is single-hop is a property of the benchmark,
#: never of where its files sit.
CONTROL_DATASETS = frozenset(name for name, spec in DATASETS.items() if spec.single_hop)


__all__ = [
    "CONTROL_DATASETS",
    "DATASETS",
    "DATASET_NAMES",
    "DEFAULT_LAYOUT",
    "OFFICIAL_DATA",
    "OHR_DOCUMENT_ALIASES",
    "OHR_DOMAIN_ALIASES",
    "OHR_UNACQUIRED_DOMAINS",
    "S2_OUTPUT",
    "S3_EXPORTS",
    "SOTA_CORPUS",
    "UDA_SPLIT_PREFIXES",
    "CorpusLayout",
    "DatasetSpec",
    "dataset_specs",
    "ohr_bench_exclusion",
    "ohr_bench_file_name",
    "prep_filename",
    "uda_file_name",
    "uda_page_from_q_uid",
]
