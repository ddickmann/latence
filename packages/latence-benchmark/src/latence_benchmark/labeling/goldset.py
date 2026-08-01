# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Gold-set mode: corpus loading, question capture, span integrity, coverage (ticket 21).

The manual enterprise gold set (ticket 08) is ~100 multi-hop questions with hand-marked
supporting-fact spans over the real exported corpus. This module owns the data layer:

* :func:`load_corpus` reads an S1 export (``records.jsonl`` preferred, ``records.parquet``
  fallback) into an in-memory :class:`Corpus` of labelable chunks. The corpus text of a
  chunk is the **exported** text — ``masked_content`` when the PII Redaction Stage produced
  one, else ``content`` — exactly the text a retrieval stack indexes, so span offsets mean
  the same thing to the labeler and to the ticket-17 eval.
* :func:`append_question` writes one gold object per line, **append-only**, verifying every
  span round-trips (``chunk_text[char_start:char_end] == text``) AT WRITE TIME. A reopened
  file resumes: :func:`load_gold` returns everything already labeled.
* :func:`coverage` reports the hop-count distribution and warns when the 1-hop share
  exceeds :data:`ONE_HOP_WARNING_SHARE` — the set must not collapse into 1-hop questions.
* :func:`validate_gold` re-checks the whole file against the corpus; ticket 17 runs it
  (via ``latence-label validate``) before the eval reads the gold set.

Gold JSONL format (one object per line, sorted keys)::

    {"question_id": "q-<sha256(question)[:16]>", "question": ..., "hop_count": N,
     "supporting_facts": [{"document_id": ..., "chunk_record_id": ...,
                           "char_start": ..., "char_end": ..., "text": ...}]}
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

#: Warn when more than this share of labeled questions is 1-hop (ticket 21's guard).
ONE_HOP_WARNING_SHARE = 0.4


class SpanIntegrityError(ValueError):
    """A supporting-fact span does not round-trip against the corpus text."""


# -- corpus -----------------------------------------------------------------


@dataclass(frozen=True)
class ChunkRecord:
    """One labelable chunk of the exported corpus."""

    chunk_record_id: str
    document_id: str
    file_name: str
    text: str
    #: Stable ordering key within the document (chunk index / char offset when available).
    order: tuple[int, int]


@dataclass(frozen=True)
class Corpus:
    """The loaded export: chunks keyed by record id, grouped per source document."""

    chunks: dict[str, ChunkRecord]
    #: ``document_id`` -> chunk record ids in reading order.
    documents: dict[str, list[str]]

    def chunk(self, chunk_record_id: str) -> ChunkRecord:
        try:
            return self.chunks[chunk_record_id]
        except KeyError:
            msg = f"unknown chunk_record_id {chunk_record_id!r} (not in the loaded export)"
            raise SpanIntegrityError(msg) from None


def _row_to_chunk(row: dict[str, Any]) -> ChunkRecord | None:
    """An exported row as a labelable chunk, or ``None`` when it carries no corpus text.

    Mirrors the S1 exporter's corpus-text rule: ``masked_content`` (the PII-handled variant)
    IS the corpus text when present, else ``content``.
    """
    record_id = row.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        return None
    masked = row.get("masked_content")
    text = masked if isinstance(masked, str) else row.get("content")
    if not isinstance(text, str) or not text:
        return None
    provenance = row.get("provenance")
    if isinstance(provenance, str):  # Parquet rows JSON-encode nested fields (ADR-0006).
        try:
            provenance = json.loads(provenance)
        except json.JSONDecodeError:
            provenance = {}
    if not isinstance(provenance, dict):
        provenance = {}
    document_id = provenance.get("document_id")
    if not isinstance(document_id, str) or not document_id:
        document_id = str(row.get("document_record_id") or record_id)
    index = row.get("index")
    char_start = provenance.get("char_start")
    order = (
        index if isinstance(index, int) else 1 << 30,
        char_start if isinstance(char_start, int) else 1 << 30,
    )
    file_name = provenance.get("file_name")
    return ChunkRecord(
        chunk_record_id=record_id,
        document_id=document_id,
        file_name=file_name if isinstance(file_name, str) else "",
        text=text,
        order=order,
    )


def _iter_export_rows(path: Path) -> Iterable[dict[str, Any]]:
    """Rows of one export file — ``.jsonl`` line objects or ``.parquet`` row dicts."""
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row
        return
    import pyarrow.parquet as pq  # latence-core dependency; imported lazily.

    table = pq.read_table(path)  # type: ignore[no-untyped-call]
    yield from table.to_pylist()


def _find_export_file(root: Path) -> Path:
    """The records file inside an export directory (``records.jsonl`` preferred)."""
    for name in ("records.jsonl", "records.parquet"):
        direct = root / name
        if direct.is_file():
            return direct
    for name in ("records.jsonl", "records.parquet"):
        found = sorted(root.rglob(name))
        if found:
            return found[0]
    msg = f"no records.jsonl / records.parquet under export dir {root}"
    raise FileNotFoundError(msg)


def load_corpus(path: Path) -> Corpus:
    """Load an exported corpus from an export directory or a records file directly."""
    records_file = _find_export_file(path) if path.is_dir() else path
    chunks: dict[str, ChunkRecord] = {}
    for row in _iter_export_rows(records_file):
        chunk = _row_to_chunk(row)
        if chunk is not None:
            chunks[chunk.chunk_record_id] = chunk
    documents: dict[str, list[str]] = {}
    for chunk in sorted(chunks.values(), key=lambda c: (c.document_id, c.order, c.chunk_record_id)):
        documents.setdefault(chunk.document_id, []).append(chunk.chunk_record_id)
    return Corpus(chunks=chunks, documents=documents)


# -- gold format ------------------------------------------------------------


@dataclass(frozen=True)
class SupportingFact:
    """One hand-marked supporting-fact span, offsets into the chunk's corpus text."""

    document_id: str
    chunk_record_id: str
    char_start: int
    char_end: int
    text: str

    def to_json(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_record_id": self.chunk_record_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text": self.text,
        }


@dataclass(frozen=True)
class GoldQuestion:
    """One labeled gold question with its supporting facts."""

    question_id: str
    question: str
    hop_count: int
    supporting_facts: tuple[SupportingFact, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "hop_count": self.hop_count,
            "supporting_facts": [f.to_json() for f in self.supporting_facts],
        }


def question_id(question: str) -> str:
    """Deterministic id of a question: ``q-`` + sha256 of the stripped text (16 hex chars)."""
    digest = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()
    return f"q-{digest[:16]}"


def verify_fact(corpus: Corpus, fact: SupportingFact) -> None:
    """Raise :class:`SpanIntegrityError` unless the span round-trips against the corpus."""
    chunk = corpus.chunk(fact.chunk_record_id)
    if fact.document_id != chunk.document_id:
        msg = (
            f"fact on chunk {fact.chunk_record_id!r} names document {fact.document_id!r} "
            f"but the chunk belongs to {chunk.document_id!r}"
        )
        raise SpanIntegrityError(msg)
    if not (0 <= fact.char_start < fact.char_end <= len(chunk.text)):
        msg = (
            f"span [{fact.char_start}:{fact.char_end}] out of range for chunk "
            f"{fact.chunk_record_id!r} (len {len(chunk.text)})"
        )
        raise SpanIntegrityError(msg)
    sliced = chunk.text[fact.char_start : fact.char_end]
    if sliced != fact.text:
        msg = (
            f"span text mismatch on chunk {fact.chunk_record_id!r} "
            f"[{fact.char_start}:{fact.char_end}]: stored {fact.text!r} != corpus {sliced!r}"
        )
        raise SpanIntegrityError(msg)


def _fact_from_json(obj: dict[str, Any]) -> SupportingFact:
    return SupportingFact(
        document_id=str(obj["document_id"]),
        chunk_record_id=str(obj["chunk_record_id"]),
        char_start=int(obj["char_start"]),
        char_end=int(obj["char_end"]),
        text=str(obj["text"]),
    )


def _question_from_json(obj: dict[str, Any]) -> GoldQuestion:
    facts = tuple(_fact_from_json(f) for f in obj["supporting_facts"])
    return GoldQuestion(
        question_id=str(obj["question_id"]),
        question=str(obj["question"]),
        hop_count=int(obj["hop_count"]),
        supporting_facts=facts,
    )


def load_gold(path: Path) -> dict[str, GoldQuestion]:
    """Load an existing gold file for resume — ``{question_id: question}`` in file order.

    A re-submitted ``question_id`` later in the file wins (append-only correction).
    Missing file ⇒ empty dict (a fresh session).
    """
    questions: dict[str, GoldQuestion] = {}
    if not path.exists():
        return questions
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                question = _question_from_json(obj)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                msg = f"{path}:{lineno}: malformed gold line ({exc})"
                raise ValueError(msg) from exc
            # Last occurrence wins on duplicate ids so a correction can simply be re-appended.
            questions.pop(question.question_id, None)
            questions[question.question_id] = question
    return questions


def _required_fact_count(hop_count: int) -> int:
    """A multi-hop question needs >=2 marked spans; a 1-hop question needs >=1."""
    return 2 if hop_count >= 2 else 1


def append_question(
    path: Path,
    corpus: Corpus,
    question: str,
    hop_count: int,
    facts: Iterable[SupportingFact],
    *,
    replace: bool = False,
) -> GoldQuestion:
    """Verify + append one gold question. Every span is round-trip checked BEFORE writing.

    :raises SpanIntegrityError: a span does not round-trip against the corpus.
    :raises ValueError: empty question, bad hop count, too few facts, or a duplicate
        ``question_id`` without ``replace=True``.
    """
    text = question.strip()
    if not text:
        raise ValueError("question must be non-empty")
    if hop_count < 1:
        raise ValueError(f"hop_count must be >= 1, got {hop_count}")
    fact_list = tuple(facts)
    required = _required_fact_count(hop_count)
    if len(fact_list) < required:
        msg = (
            f"a {hop_count}-hop question needs >= {required} supporting facts, "
            f"got {len(fact_list)}"
        )
        raise ValueError(msg)
    for fact in fact_list:
        verify_fact(corpus, fact)
    qid = question_id(text)
    if not replace and qid in load_gold(path):
        msg = f"question_id {qid!r} already labeled (identical question text); pass replace"
        raise ValueError(msg)
    gold = GoldQuestion(
        question_id=qid, question=text, hop_count=hop_count, supporting_facts=fact_list
    )
    line = json.dumps(gold.to_json(), sort_keys=True, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return gold


# -- coverage ---------------------------------------------------------------


@dataclass(frozen=True)
class Coverage:
    """Progress + hop-depth distribution of the gold set."""

    total: int
    hop_distribution: dict[int, int]
    one_hop_share: float
    warning: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "hop_distribution": {str(k): v for k, v in sorted(self.hop_distribution.items())},
            "one_hop_share": self.one_hop_share,
            "warning": self.warning,
        }


def coverage(questions: Iterable[GoldQuestion]) -> Coverage:
    """Coverage summary with the 1-hop collapse guard (warn when share > 40%)."""
    distribution: dict[int, int] = {}
    total = 0
    for question in questions:
        total += 1
        distribution[question.hop_count] = distribution.get(question.hop_count, 0) + 1
    share = (distribution.get(1, 0) / total) if total else 0.0
    warning = None
    if total and share > ONE_HOP_WARNING_SHARE:
        warning = (
            f"1-hop share {share:.0%} exceeds {ONE_HOP_WARNING_SHARE:.0%} — the gold set is "
            "collapsing into 1-hop questions; label deeper multi-hop questions next."
        )
    return Coverage(
        total=total, hop_distribution=distribution, one_hop_share=share, warning=warning
    )


# -- validation (ticket 17 entry point) -------------------------------------


def validate_gold(corpus: Corpus, path: Path) -> list[str]:
    """Every problem in the gold file, as human-readable strings. Empty list = valid.

    Checks per line: JSON shape, ``question_id`` matches the question-text hash,
    hop count >= 1, enough facts for the hop count, and every span round-trips against
    the corpus. Ticket 17 calls this before the eval reads the gold set.
    """
    errors: list[str] = []
    if not path.exists():
        return [f"{path}: gold file does not exist"]
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                question = _question_from_json(obj)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"{path}:{lineno}: malformed line ({exc})")
                continue
            if question.question_id != question_id(question.question):
                errors.append(
                    f"{path}:{lineno}: question_id {question.question_id!r} does not match "
                    f"the question text hash {question_id(question.question)!r}"
                )
            if question.hop_count < 1:
                errors.append(f"{path}:{lineno}: hop_count {question.hop_count} < 1")
            elif len(question.supporting_facts) < _required_fact_count(question.hop_count):
                errors.append(
                    f"{path}:{lineno}: {question.hop_count}-hop question has only "
                    f"{len(question.supporting_facts)} supporting facts"
                )
            for fact in question.supporting_facts:
                try:
                    verify_fact(corpus, fact)
                except SpanIntegrityError as exc:
                    errors.append(f"{path}:{lineno}: {exc}")
    return errors
