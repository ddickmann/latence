"""The port to a git host for the ontology hub (ticket 59; the owner's design of 15 Nov).

History is git's: one repository per estate holds the sources of its concept libraries, each
domain on its branch (``domains/<domain>``), which its stewards alone commit to. Anyone else
proposes through a merge request into it; the builder does, from a branch of its own. A merge
request's reviews carry the five-state review: a reviewer approves, the domain's data owner
certifies. A conflict across domains blocks the merge until the conformed library's owner
resolves it. A certified merge creates a tag, and the host builds the release (version and
digest) from it. A revert is a new release; history is never rewritten.

This module states the port (``GitHost``) and what passes through it; ``hub_local`` is a
backend over a local bare repository (the ``git`` binary, no network). A backend for a hosted
forge (Forgejo first) is another implementation of the same port.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

Role = Literal["reviewer", "data_owner", "conformed_owner"]
Verdict = Literal["approved", "certified", "changes_requested", "resolved"]
State = Literal["open", "blocked", "merged", "closed"]

DOMAIN_BRANCH = "domains/{domain}"


class HubError(RuntimeError):
    """What a git host refuses: a commit to a domain's branch by someone not its steward, a merge
    without its reviews, a merge a conflict blocks."""


@dataclass(frozen=True)
class Review:
    """One review on a merge request, with its actor."""

    by: str
    role: Role
    verdict: Verdict
    at: str
    note: str = ""


@dataclass(frozen=True)
class MergeRequest:
    """A change proposed into a domain's branch."""

    id: int
    source: str
    target: str
    title: str
    author: str
    state: State = "open"
    blocked_by: str = ""
    reviews: tuple[Review, ...] = ()
    merge_commit: str = ""
    tag: str = ""


@dataclass(frozen=True)
class Tag:
    """A tag and the commit it names; ``message`` is what its creator wrote on it."""

    name: str
    commit: str
    message: str


@dataclass(frozen=True)
class Commit:
    """One commit of a branch's history, with its actor."""

    sha: str
    author: str
    at: str
    message: str
    paths: tuple[str, ...] = field(default=())


class GitHost(Protocol):
    """What the builder needs of a git host. It never pushes a domain's branch."""

    def branches(self) -> dict[str, str]:
        """Every branch, to its head commit."""
        ...

    def commit(
        self,
        branch: str,
        files: Mapping[str, str],
        message: str,
        author: str,
        at: str,
        start_from: str | None = None,
    ) -> str:
        """Commit the files on the branch (created from ``start_from`` when new); returns the
        commit. A domain's branch takes commits from its stewards only."""
        ...

    def read(self, ref: str, path: str) -> str | None:
        """A file at a ref, or ``None`` where the ref does not hold it."""
        ...

    def diff(self, base: str, head: str) -> str:
        """What ``head`` changes since it left ``base``."""
        ...

    def history(self, ref: str, path: str | None = None) -> list[Commit]:
        """The commits of a ref (that touched ``path``), newest first."""
        ...

    def open_merge_request(self, source: str, target: str, title: str, author: str) -> MergeRequest:
        """Propose ``source`` into ``target``; the open one is returned where it exists."""
        ...

    def merge_requests(self) -> list[MergeRequest]:
        """Every merge request, oldest first."""
        ...

    def review(self, mr: int, review: Review) -> MergeRequest:
        """Add a review to a merge request."""
        ...

    def tags(self) -> list[Tag]:
        """Every tag, by name."""
        ...
