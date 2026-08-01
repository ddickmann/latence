# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Local labeling server — stdlib ``http.server`` on ``127.0.0.1``, JSON API + one page.

No framework, no external assets, no new dependency: the whole interface is
:data:`latence_benchmark.labeling.ui.PAGE` at ``/`` plus a small JSON API under ``/api/``.
The server binds loopback only — this is a single-labeler local tool, never a service.

Every write goes through the verified append paths
(:func:`latence_benchmark.labeling.goldset.append_question` /
:func:`latence_benchmark.labeling.calibration.append_judgment`), so a span that does not
round-trip against the corpus is rejected with a 400 before a byte hits the gold file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from latence_benchmark.labeling.calibration import (
    HyperedgeJudgment,
    HyperedgeRow,
    append_judgment,
    load_judgments,
)
from latence_benchmark.labeling.goldset import (
    Corpus,
    GoldQuestion,
    SpanIntegrityError,
    SupportingFact,
    append_question,
    coverage,
    load_gold,
)
from latence_benchmark.labeling.ui import PAGE

if TYPE_CHECKING:
    from pathlib import Path

HOST = "127.0.0.1"


class ApiError(Exception):
    """A client-visible API failure with an HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class GoldSession:
    """Gold-set mode state: the loaded corpus + the resumable gold file."""

    corpus: Corpus
    out_path: Path
    questions: dict[str, GoldQuestion] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    @classmethod
    def open(cls, corpus: Corpus, out_path: Path) -> GoldSession:
        return cls(corpus=corpus, out_path=out_path, questions=load_gold(out_path))

    def state(self) -> dict[str, Any]:
        return {
            "mode": "gold",
            "out": str(self.out_path),
            "coverage": coverage(self.questions.values()).to_json(),
        }

    def _labeled_facts_per_document(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for question in self.questions.values():
            for fact in question.supporting_facts:
                counts[fact.document_id] = counts.get(fact.document_id, 0) + 1
        return counts

    def documents(self) -> dict[str, Any]:
        fact_counts = self._labeled_facts_per_document()
        docs = []
        for document_id, chunk_ids in self.corpus.documents.items():
            first = self.corpus.chunks[chunk_ids[0]]
            docs.append(
                {
                    "document_id": document_id,
                    "file_name": first.file_name,
                    "chunk_count": len(chunk_ids),
                    "labeled_fact_count": fact_counts.get(document_id, 0),
                }
            )
        return {"documents": docs}

    def document(self, document_id: str) -> dict[str, Any]:
        chunk_ids = self.corpus.documents.get(document_id)
        if chunk_ids is None:
            raise ApiError(404, f"unknown document_id {document_id!r}")
        chunks = [
            {
                "chunk_record_id": cid,
                "document_id": document_id,
                "text": self.corpus.chunks[cid].text,
            }
            for cid in chunk_ids
        ]
        return {"document_id": document_id, "chunks": chunks}

    def existing(self) -> dict[str, Any]:
        return {"questions": [q.to_json() for q in self.questions.values()]}

    def save(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            question = str(body["question"])
            hop_count = int(body["hop_count"])
            facts = [
                SupportingFact(
                    document_id=str(f["document_id"]),
                    chunk_record_id=str(f["chunk_record_id"]),
                    char_start=int(f["char_start"]),
                    char_end=int(f["char_end"]),
                    text=str(f["text"]),
                )
                for f in body["supporting_facts"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(400, f"malformed question payload: {exc}") from exc
        with self.lock:
            try:
                gold = append_question(
                    self.out_path,
                    self.corpus,
                    question,
                    hop_count,
                    facts,
                    replace=bool(body.get("replace", False)),
                )
            except SpanIntegrityError as exc:
                raise ApiError(400, f"span integrity: {exc}") from exc
            except ValueError as exc:
                status = 409 if "already labeled" in str(exc) else 400
                raise ApiError(status, str(exc)) from exc
            self.questions.pop(gold.question_id, None)
            self.questions[gold.question_id] = gold
            return {
                "question": gold.to_json(),
                "coverage": coverage(self.questions.values()).to_json(),
            }


@dataclass
class CalibrationSession:
    """Calibration mode state: the Hyperedge rows + the resumable judgment file."""

    hyperedges: list[HyperedgeRow]
    out_path: Path
    judgments: dict[str, HyperedgeJudgment] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    @classmethod
    def open(cls, hyperedges: list[HyperedgeRow], out_path: Path) -> CalibrationSession:
        return cls(hyperedges=hyperedges, out_path=out_path, judgments=load_judgments(out_path))

    def state(self) -> dict[str, Any]:
        return {
            "mode": "calibration",
            "out": str(self.out_path),
            "total": len(self.hyperedges),
            "judged": len(self.judgments),
        }

    def listing(self) -> dict[str, Any]:
        return {
            "hyperedges": [h.to_json() for h in self.hyperedges],
            "judgments": {k: j.to_json() for k, j in self.judgments.items()},
        }

    def save(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            hyperedge_id = str(body["hyperedge_id"])
            judgment = str(body["judgment"])
            note = str(body.get("note", ""))
        except (KeyError, TypeError) as exc:
            raise ApiError(400, f"malformed judgment payload: {exc}") from exc
        known = {h.hyperedge_id for h in self.hyperedges}
        with self.lock:
            try:
                record = append_judgment(self.out_path, known, hyperedge_id, judgment, note)
            except ValueError as exc:
                raise ApiError(400, str(exc)) from exc
            self.judgments.pop(record.hyperedge_id, None)
            self.judgments[record.hyperedge_id] = record
            return {"judgment": record.to_json(), "judged": len(self.judgments)}


Session = GoldSession | CalibrationSession


class LabelingHandler(BaseHTTPRequestHandler):
    """Routes ``/`` (the page) and ``/api/*`` (JSON) onto the session object."""

    def __init__(self, session: Session, *args: Any, **kwargs: Any) -> None:
        self.session = session
        super().__init__(*args, **kwargs)

    # Quiet: a labeling session should not scroll request noise over the terminal.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status: int, obj: dict[str, Any]) -> None:
        body = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(400, f"request body is not JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise ApiError(400, "request body must be a JSON object")
        return body

    def do_GET(self) -> None:  # noqa: N802 - http.server contract
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/state":
                self._send_json(200, self.session.state())
            elif parsed.path == "/api/documents" and isinstance(self.session, GoldSession):
                self._send_json(200, self.session.documents())
            elif parsed.path == "/api/document" and isinstance(self.session, GoldSession):
                query = parse_qs(parsed.query)
                document_id = (query.get("document_id") or [""])[0]
                self._send_json(200, self.session.document(document_id))
            elif parsed.path == "/api/questions" and isinstance(self.session, GoldSession):
                self._send_json(200, self.session.existing())
            elif parsed.path == "/api/hyperedges" and isinstance(
                self.session, CalibrationSession
            ):
                self._send_json(200, self.session.listing())
            else:
                self._send_json(404, {"error": f"no such endpoint: GET {parsed.path}"})
        except ApiError as exc:
            self._send_json(exc.status, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802 - http.server contract
        parsed = urlparse(self.path)
        try:
            body = self._read_body()
            if (parsed.path == "/api/questions" and isinstance(self.session, GoldSession)) or (
                parsed.path == "/api/judgments" and isinstance(self.session, CalibrationSession)
            ):
                self._send_json(200, self.session.save(body))
            else:
                self._send_json(404, {"error": f"no such endpoint: POST {parsed.path}"})
        except ApiError as exc:
            self._send_json(exc.status, {"error": str(exc)})


def make_server(session: Session, port: int = 0) -> ThreadingHTTPServer:
    """Bind the labeling server on loopback. ``port=0`` picks a free port (tests)."""
    handler = partial(LabelingHandler, session)
    return ThreadingHTTPServer((HOST, port), handler)
