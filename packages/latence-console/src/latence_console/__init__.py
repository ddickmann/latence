"""Latence run console — a read-only, offline, localhost view over runs on Storage.

The pipeline already writes everything a stakeholder needs: run manifests, Quality Reports,
exported corpora and graphs, all as files (ADR-0010, ADR-0017). What it lacked was a face.
This package is that face, and nothing more — it **reads** those files and renders them as
headline quality verdicts a non-engineer can act on, with the numbers underneath.

Three properties are structural, not configured:

* **Read-only.** The only Storage methods reached are ``exists`` / ``ls`` / ``read_text`` /
  ``modified``. Non-read HTTP methods are refused with 405.
* **No new state.** :class:`~latence_console.server.ConsoleApp` holds nothing between
  requests; every page is re-derived from the files at request time.
* **Offline.** Pages are single self-contained documents with inlined CSS — no CDN, no
  font host, no beacon — and the server binds the loopback interface only.
"""

from __future__ import annotations

from latence_console.runs import RunIndex, RunStatus, RunSummary
from latence_console.server import ConsoleApp, Response, serve
from latence_console.verdicts import Tone, Verdict, headline_verdicts

__all__ = [
    "ConsoleApp",
    "Response",
    "RunIndex",
    "RunStatus",
    "RunSummary",
    "Tone",
    "Verdict",
    "headline_verdicts",
    "serve",
]
