"""A git host over a local bare repository (ticket 59): the ``git`` binary in a subprocess, no
network. It stands in for a hosted forge in tests and on one machine: branches, commits, diffs
and history are the repository's own; merge requests and their reviews, which a forge keeps
beside the repository, are kept in ``hub.json`` in the repository's folder.

It holds the rules the owner set (15 Nov) where a forge would: a domain's branch takes commits
from its stewards only; a merge needs a reviewer's approval and the domain's data owner's
certification, and none while a conflict blocks it; a merge creates a tag; a revert is a new
commit and a new tag, never a rewrite. The forge's side of a merge (``merge``, ``block``,
``resolve``) is here for a test to play the host's part; the builder calls none of it.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .hub_port import DOMAIN_BRANCH, Commit, HubError, MergeRequest, Review, Tag

STATE = "hub.json"


def _when(at: str) -> str:
    """A day (or a timestamp) as git's date: the day at midnight UTC, so that a commit made on
    the same inputs is the same commit."""
    return f"{at[:10]}T00:00:00+00:00"


class LocalGitHost:
    """The port over a bare repository at ``path``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if not (path / "HEAD").exists():
            raise HubError(f"{path} is no bare git repository")

    @classmethod
    def create(
        cls,
        path: Path,
        domains: Mapping[str, Mapping[str, Any]],
        at: str,
        files: Mapping[str, Mapping[str, str]] | None = None,
    ) -> LocalGitHost:
        """A new repository with a branch per domain, each started by its first steward with
        the domain's files (``files``, by domain). ``domains`` names each domain's
        ``stewards`` and ``data_owner``; the estate's conformed library's owner is
        ``conformed_owner`` beside them, under the name ``_conformed``."""
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "--bare", "-q", "-b", "main", str(path)], check=True)
        host = cls(path)
        host._save({"domains": {k: dict(v) for k, v in domains.items()}, "merge_requests": []})
        for domain, spec in sorted(domains.items()):
            if domain.startswith("_"):
                continue
            stewards = list(spec.get("stewards") or [])
            if not stewards:
                raise HubError(f"the domain {domain} names no steward")
            host.commit(DOMAIN_BRANCH.format(domain=domain),
                        dict((files or {}).get(domain) or {"README": f"{domain}\n"}),
                        f"{domain}: started", stewards[0], at)
        return host

    # -- the repository -------------------------------------------------------------------

    def _git(self, *args: str, env: Mapping[str, str] | None = None, stdin: str | None = None,
             check: bool = True) -> str:
        done = subprocess.run(
            ["git", "--git-dir", str(self.path), *args],
            input=stdin, capture_output=True, text=True, check=False,
            env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                 **(env or {})},
        )
        if check and done.returncode != 0:
            raise HubError(f"git {args[0]}: {done.stderr.strip()}")
        return done.stdout

    def _state(self) -> dict[str, Any]:
        state: dict[str, Any] = json.loads((self.path / STATE).read_text(encoding="utf-8"))
        return state

    def _save(self, state: Mapping[str, Any]) -> None:
        (self.path / STATE).write_text(
            json.dumps(state, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _domain_of(self, branch: str) -> str | None:
        for domain in self._state()["domains"]:
            if branch == DOMAIN_BRANCH.format(domain=domain):
                return str(domain)
        return None

    def _identity(self, who: str, at: str) -> dict[str, str]:
        mail = f"{who}@hub.invalid"
        return {"GIT_AUTHOR_NAME": who, "GIT_AUTHOR_EMAIL": mail, "GIT_AUTHOR_DATE": _when(at),
                "GIT_COMMITTER_NAME": who, "GIT_COMMITTER_EMAIL": mail,
                "GIT_COMMITTER_DATE": _when(at)}

    def branches(self) -> dict[str, str]:
        out = self._git("for-each-ref", "--format=%(refname:short) %(objectname)", "refs/heads")
        return dict(line.split(" ", 1) for line in out.splitlines() if line.strip())

    def _steward_only(self, branch: str, author: str) -> None:
        domain = self._domain_of(branch)
        if domain is None:
            return
        stewards = self._state()["domains"][domain].get("stewards") or []
        if author not in stewards:
            raise HubError(f"{author} is no steward of {domain}: only its stewards commit to "
                           f"{branch}; anyone else opens a merge request into it")

    def _tree_with(self, parent: str | None, files: Mapping[str, str]) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
            if parent is not None:
                self._git("read-tree", parent, env=env)
            for name, text in sorted(files.items()):
                blob = self._git("hash-object", "-w", "--stdin", stdin=text).strip()
                self._git("update-index", "--add", "--cacheinfo", f"100644,{blob},{name}", env=env)
            return self._git("write-tree", env=env).strip()

    def commit(self, branch: str, files: Mapping[str, str], message: str, author: str, at: str,
               start_from: str | None = None) -> str:
        self._steward_only(branch, author)
        heads = self.branches()
        parent = heads.get(branch) or (heads.get(start_from) if start_from else None)
        if start_from and branch not in heads and parent is None:
            raise HubError(f"no branch {start_from} to start {branch} from")
        tree = self._tree_with(parent, files)
        if parent is not None and self._git("rev-parse", f"{parent}^{{tree}}").strip() == tree:
            return parent  # nothing changes: no empty commit
        args = ["commit-tree", tree, "-m", message] + (["-p", parent] if parent else [])
        sha = self._git(*args, env=self._identity(author, at)).strip()
        was = heads.get(branch)
        self._git("update-ref", f"refs/heads/{branch}", sha, was or "0" * 40)
        return sha

    def read(self, ref: str, path: str) -> str | None:
        probe = subprocess.run(["git", "--git-dir", str(self.path), "cat-file", "-e",
                                f"{ref}:{path}"], capture_output=True, check=False)
        return self._git("show", f"{ref}:{path}") if probe.returncode == 0 else None

    def diff(self, base: str, head: str) -> str:
        return self._git("diff", f"{base}...{head}")

    def history(self, ref: str, path: str | None = None) -> list[Commit]:
        out = self._git("log", "--topo-order", "--format=%x1e%H%x1f%an%x1f%aI%x1f%s",
                        "--name-only", ref,
                        *(["--", path] if path else []))
        commits = []
        for block in out.split("\x1e"):
            block = block.strip("\n")
            if not block.strip():
                continue
            head, _, names = block.partition("\n")
            sha, author, at, message = head.split("\x1f")
            commits.append(Commit(sha, author, at[:10], message,
                                  tuple(n for n in names.splitlines() if n.strip())))
        return commits

    def tags(self) -> list[Tag]:
        out = self._git(
            "for-each-ref",
            "--format=%(refname:short)%1f%(*objectname)%1f%(contents)%1e",
            "refs/tags",
        )
        tags = []
        for block in out.split("\x1e"):
            if not block.strip():
                continue
            name, commit, message = block.strip("\n").split("\x1f", 2)
            tags.append(Tag(name, commit, message.strip()))
        return sorted(tags, key=lambda t: t.name)

    # -- merge requests -------------------------------------------------------------------

    def _mr(self, raw: Mapping[str, Any]) -> MergeRequest:
        return MergeRequest(**{**raw, "reviews": tuple(Review(**r) for r in raw["reviews"])})

    def _put(self, mr: MergeRequest) -> MergeRequest:
        state = self._state()
        raw = {**asdict(mr), "reviews": [asdict(r) for r in mr.reviews]}
        listed = [m for m in state["merge_requests"] if m["id"] != mr.id]
        state["merge_requests"] = sorted([*listed, raw], key=lambda m: m["id"])
        self._save(state)
        return mr

    def merge_requests(self) -> list[MergeRequest]:
        return [self._mr(m) for m in self._state()["merge_requests"]]

    def _get(self, mr: int) -> MergeRequest:
        found = [m for m in self.merge_requests() if m.id == mr]
        if not found:
            raise HubError(f"no merge request {mr}")
        return found[0]

    def open_merge_request(self, source: str, target: str, title: str, author: str) -> MergeRequest:
        if self._domain_of(target) is None:
            raise HubError(f"{target} is no domain's branch")
        if source not in self.branches():
            raise HubError(f"no branch {source}")
        for m in self.merge_requests():
            if m.source == source and m.target == target and m.state in ("open", "blocked"):
                return m
        number = max((m.id for m in self.merge_requests()), default=0) + 1
        return self._put(MergeRequest(number, source, target, title, author))

    def review(self, mr: int, review: Review) -> MergeRequest:
        found = self._get(mr)
        if found.state in ("merged", "closed"):
            raise HubError(f"merge request {mr} is {found.state}")
        domain = self._domain_of(found.target) or ""
        spec = self._state()["domains"].get(domain) or {}
        if review.role == "data_owner" and review.by != spec.get("data_owner"):
            raise HubError(f"{review.by} is not the data owner of {domain}")
        if review.role == "conformed_owner":
            owner = (self._state()["domains"].get("_conformed") or {}).get("data_owner")
            if review.by != owner:
                raise HubError(f"{review.by} does not own the conformed library")
        return self._put(replace(found, reviews=(*found.reviews, review)))

    # -- the forge's side, played by a test -----------------------------------------------

    def block(self, mr: int, why: str) -> MergeRequest:
        """A conflict across domains: blocked until the conformed library's owner resolves it."""
        return self._put(replace(self._get(mr), state="blocked", blocked_by=why))

    def resolve(self, mr: int, by: str, at: str) -> MergeRequest:
        self.review(mr, Review(by, "conformed_owner", "resolved", at))
        return self._put(replace(self._get(mr), state="open", blocked_by=""))

    def merge(self, mr: int, by: str, at: str, version: str) -> MergeRequest:
        """Merge a certified merge request into the domain's branch and tag it
        ``<domain>/<version>``; refused without its reviews, while blocked, or by someone not
        the domain's steward."""
        found = self._get(mr)
        if found.state == "blocked":
            raise HubError(f"merge request {mr} is blocked: {found.blocked_by}")
        verdicts = {(r.role, r.verdict) for r in found.reviews}
        if ("reviewer", "approved") not in verdicts or ("data_owner", "certified") not in verdicts:
            raise HubError(f"merge request {mr} needs a reviewer's approval and the data "
                           "owner's certification")
        self._steward_only(found.target, by)
        heads = self.branches()
        tree = self._git("merge-tree", "--write-tree", heads[found.target], heads[found.source],
                         check=False)
        if not tree.strip() or "CONFLICT" in tree:
            raise HubError(f"merge request {mr} does not merge cleanly")
        sha = self._git("commit-tree", tree.splitlines()[0].strip(), "-p", heads[found.target],
                        "-p", heads[found.source], "-m", f"Merge !{mr}: {found.title}",
                        env=self._identity(by, at)).strip()
        self._git("update-ref", f"refs/heads/{found.target}", sha, heads[found.target])
        domain = self._domain_of(found.target)
        tag = f"{domain}/{version}"
        message = json.dumps({"merge_request": mr, "domain": domain, "version": version},
                             sort_keys=True)
        self._git("tag", "-a", tag, sha, "-m", message, env=self._identity(by, at))
        return self._put(replace(found, state="merged", merge_commit=sha, tag=tag))

    def revert(self, domain: str, to_tag: str, by: str, at: str, version: str) -> Tag:
        """A revert: a steward's new commit on the domain's branch that restores what an earlier
        tag held, tagged as a new release. Nothing of the history is rewritten."""
        branch = DOMAIN_BRANCH.format(domain=domain)
        self._steward_only(branch, by)
        head = self.branches()[branch]
        tree = self._git("rev-parse", f"{to_tag}^{{tree}}").strip()
        sha = self._git("commit-tree", tree, "-p", head, "-m", f"Revert to {to_tag}",
                        env=self._identity(by, at)).strip()
        self._git("update-ref", f"refs/heads/{branch}", sha, head)
        tag = f"{domain}/{version}"
        message = json.dumps({"domain": domain, "reverts_to": to_tag, "version": version},
                             sort_keys=True)
        self._git("tag", "-a", tag, sha, "-m", message, env=self._identity(by, at))
        return Tag(tag, sha, message)
