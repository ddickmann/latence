"""RED^FM — the pinned public multilingual corpus behind the NER, RE and ER suites.

One corpus serves three suites because RED^FM annotates all three things on the same text:
typed entity mentions with exact character boundaries (Entity Extraction), directed
predicate triples over those mentions (Relation Extraction), and a Wikidata URI per mention
(Disambiguation — two mentions are coreferent iff their gold URI matches). Scoring three Stages
on one pinned revision also means the three numbers describe the *same* documents, which a
reader can otherwise not assume.

Why this corpus and not the obvious alternatives:

* It is **multilingual with a German slice** (ar, de, en, es, fr, it, zh), so the P1 EU-language
  rider costs nothing — ``de`` is simply one of the seven scored slices.
* Its license is **verifiable and permissive**: CC BY-SA 4.0 for the corpus, Apache-2.0 for the
  loading script, verified separately (ADR-0012, see :data:`REDFM`).
* It ships as **plain JSONL at a pinned commit**, so the harness reads it directly with no
  dataset-script execution and no parquet dependency.

WikiANN/PAN-X, the other obvious multilingual NER set, is **excluded**: its Hugging Face card
carries ``license: unknown`` (checked 2026-07-18). An unverifiable license is excluded, not
guessed — that is the whole point of :mod:`latence_benchmark.licensing`.

The reader below is deliberately strict. Every row must carry the keys the real corpus carries,
and every annotated span must satisfy ``text[start:end] == surfaceform`` — which the real
release does, exactly, for all 24,812 entity spans and both endpoints of all 6,367 relation
triples across the seven
``test`` files at the pinned revision (checked 2026-07-18). A loader that tolerated a
mismatch would let a subtly mis-aligned corpus, or a hand-written fixture the real data would
never produce, score as if it were the benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from latence_benchmark.datasets import DatasetSpec, register_dataset
from latence_benchmark.licensing import DatasetLicense
from latence_benchmark.unittests import BenchmarkDataError

if TYPE_CHECKING:
    from pathlib import Path

#: Fixed language order — determinism, and the order the results page reads best in.
REDFM_LANGUAGES: tuple[str, ...] = ("ar", "de", "en", "es", "fr", "it", "zh")

#: The evaluation split. ``train``/``dev`` exist upstream but a published number is quoted on
#: ``test``, and scoring anything else under the same name would be misleading.
REDFM_SPLIT = "test"

REDFM = register_dataset(
    DatasetSpec(
        name="redfm",
        repo_id="Babelscape/REDFM",
        # Pinned SHA read from the Hub API on 2026-07-18. `main` would move under us.
        revision="eae75dc9bde14d1172351423bcbb82c5b9a94f66",
        homepage="https://huggingface.co/datasets/Babelscape/REDFM",
        license=DatasetLicense(
            # Corpus: the dataset card's `license:` front-matter AND its prose License section.
            dataset_license="CC-BY-SA-4.0",
            # Toolkit: the repo's own loading script `REDFM.py`, a SEPARATE artifact under a
            # SEPARATE license — verified by reading its file header, not the corpus card.
            code_license="Apache-2.0",
            commercial_use=True,
            attribution_required=True,
            verified_on="2026-07-18",
            source_url="https://huggingface.co/datasets/Babelscape/REDFM",
            notes=(
                "Corpus license verified twice over at the pinned revision: the card front-matter "
                "says `license: cc-by-sa-4.0` and the card's own License section says 'RED^FM is "
                "licensed under the CC BY-SA 4.0 license.' CC BY-SA 4.0 permits commercial use and "
                "requires attribution; its ShareAlike term binds ADAPTATIONS of the corpus, and this "
                "harness distributes none — it reads the corpus at run time from the pinned upstream "
                "revision, vendors no part of it into this repository, and publishes only aggregate "
                "metrics. The evaluation toolkit is the repository's own `REDFM.py`, whose file "
                "header carries the Apache-2.0 notice, verified separately per ADR-0012; this "
                "harness reads the JSONL directly and never executes that script. The authors' "
                "related model repository (github.com/Babelscape/rebel) declares NO license file "
                "(GitHub API, 2026-07-18) — that artifact is therefore UNVERIFIED and is not used."
            ),
        ),
        manifest=tuple(f"data/{REDFM_SPLIT}.{language}.jsonl" for language in REDFM_LANGUAGES),
        citation=(
            "RED^FM: a Filtered and Multilingual Relation Extraction Dataset (Huguet Cabot, "
            "Tedeschi, Ngonga Ngomo, Navigli; ACL 2023), Babelscape — CC BY-SA 4.0 — "
            "https://huggingface.co/datasets/Babelscape/REDFM"
        ),
    )
)


@dataclass(frozen=True)
class RedfmEntity:
    """One gold entity mention: a typed, KB-linked, exactly-bounded span."""

    uri: str
    surface: str
    entity_type: str
    start: int
    end: int

    @property
    def linked(self) -> bool:
        """Whether the annotation carries a usable KB identity.

        ``"None"`` is the corpus's literal string for an unlinkable mention, so it is not an
        identity and must never be treated as one — clustering every unlinkable mention together
        would invent a huge spurious gold cluster and wreck the Disambiguation numbers.
        """
        return self.uri not in ("None", "")

    @property
    def mentionable(self) -> bool:
        """Whether this annotation denotes an actual mention, i.e. a non-empty span.

        The pinned release contains exactly two zero-length entity annotations (of 24,812), and
        no relation touches either. A zero-length span is not a mention any extractor could
        emit, so the suites exclude such annotations from BOTH sides and report how many they
        excluded — rather than silently carrying an unreachable gold item that caps recall.
        """
        return self.end > self.start


@dataclass(frozen=True)
class RedfmRelation:
    """One gold relation triple, with both endpoints resolved to their spans."""

    predicate: str
    head: RedfmEntity
    tail: RedfmEntity


@dataclass(frozen=True)
class RedfmDocument:
    """One gold document: its text plus the entity and relation annotations over it."""

    docid: str
    language: str
    text: str
    entities: tuple[RedfmEntity, ...]
    relations: tuple[RedfmRelation, ...]


def _require(row: Any, key: str, source: str) -> Any:  # noqa: ANN401 - raw JSON is Any by nature
    if not isinstance(row, dict) or key not in row:
        msg = (
            f"{source}: row is missing the required key {key!r}. The RED^FM JSONL rows carry "
            "docid/title/uri/text/entities/lan/relations; a row without them is not this corpus, "
            "and scoring it would publish a number for something else."
        )
        raise BenchmarkDataError(msg)
    return row[key]


def _entity(raw: Any, text: str, source: str) -> RedfmEntity:  # noqa: ANN401 - raw JSON
    surface = str(_require(raw, "surfaceform", source))
    boundaries = _require(raw, "boundaries", source)
    if not isinstance(boundaries, list) or len(boundaries) != 2:
        msg = (
            f"{source}: entity {surface!r} has boundaries {boundaries!r}, but RED^FM annotates a "
            "half-open [start, end] pair. Refusing to guess."
        )
        raise BenchmarkDataError(msg)
    start, end = int(boundaries[0]), int(boundaries[1])
    if text[start:end] != surface:
        msg = (
            f"{source}: entity boundaries [{start}, {end}) slice {text[start:end]!r} but the "
            f"annotation says {surface!r}. Every span in the pinned release satisfies this "
            "equality, so a mismatch means the text and the annotations are out of alignment — "
            "scoring it would measure the misalignment, not the Stage."
        )
        raise BenchmarkDataError(msg)
    return RedfmEntity(
        uri=str(_require(raw, "uri", source)),
        surface=surface,
        entity_type=str(_require(raw, "type", source)),
        start=start,
        end=end,
    )


def _relation(raw: Any, text: str, source: str) -> RedfmRelation:  # noqa: ANN401 - raw JSON
    predicate = _require(raw, "predicate", source)
    return RedfmRelation(
        predicate=str(_require(predicate, "surfaceform", source)),
        head=_entity(_require(raw, "subject", source), text, source),
        tail=_entity(_require(raw, "object", source), text, source),
    )


def language_file(root: Path, language: str) -> Path:
    """Path of one language's evaluation file inside a materialised RED^FM root."""
    return root / "data" / f"{REDFM_SPLIT}.{language}.jsonl"


def load_language(root: Path, language: str) -> tuple[RedfmDocument, ...]:
    """Read and validate one language slice, in file order (which is the corpus's own order).

    :raises BenchmarkDataError: The file is missing, is not JSONL, or any row violates the
        corpus's own guarantees.
    """
    if language not in REDFM_LANGUAGES:
        msg = f"Unknown RED^FM language {language!r}. Available: {', '.join(REDFM_LANGUAGES)}"
        raise BenchmarkDataError(msg)
    path = language_file(root, language)
    if not path.is_file():
        msg = (
            f"RED^FM language file {path} is missing. Fetch the dataset at revision "
            f"{REDFM.revision} (all {len(REDFM_LANGUAGES)} `{REDFM_SPLIT}` files are declared in "
            "the manifest) and re-run."
        )
        raise BenchmarkDataError(msg)

    documents: list[RedfmDocument] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        source = f"{path.name}:{number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"{source}: not valid JSON ({exc.msg}). RED^FM ships one JSON object per line."
            raise BenchmarkDataError(msg) from exc
        text = str(_require(row, "text", source))
        documents.append(
            RedfmDocument(
                docid=str(_require(row, "docid", source)),
                language=str(row.get("lan", language)),
                text=text,
                entities=tuple(_entity(e, text, source) for e in _require(row, "entities", source)),
                relations=tuple(
                    _relation(r, text, source) for r in _require(row, "relations", source)
                ),
            )
        )
    return tuple(documents)
