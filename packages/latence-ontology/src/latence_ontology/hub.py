"""The builder's side of the ontology hub (ticket 59; the owner's design of 15 Nov, the host's 309).

- **A merge request per domain.** The approved entries of a proposal file are split by the
  domain each goes to (a match to every domain it names; any other entry to the domain of its
  object, by schema), and each domain's share is committed as ``proposals/<estate>.yaml`` on a
  branch of the builder's own (``builder/<estate>/<domain>/<digest>``, started from the domain's
  branch), then proposed into the domain's branch by a merge request. The builder never commits
  to a domain's branch; the port refuses it, as a forge would.
- **The five-state review on the merge request.** The entries are approved (the file holds no
  other); the builder writes each approving reviewer's approval on the merge request, and the
  domain's data owner's certification once every element of the request carries the owner's
  certification on the claim it states (``certifications.jsonl``). A review made on the git
  host is read back: a data owner's certification there becomes a certification of every
  element of the request. A request's state among the five: ``proposed`` (open), ``approved``
  (a reviewer approved it), ``certified`` (the data owner certified it; merged and tagged by the
  host), ``retired`` (closed unmerged); ``blocked`` beside it while a conflict across domains
  waits for the conformed library's owner.
- **Tag and release.** A certified merge creates a tag; the host builds the release from it.
  The builder reads the tag and, from the host's receipt, the release's version and digest.
- **Views** (``hub.json`` in the build folder): each request with its diff against the domain's
  branch, its entries with their claims, its reviews with their actors, its tag and release;
  the history of each domain's branch with its actors.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import yaml
from latence_proposal_writer import to_yaml

from .hub_port import DOMAIN_BRANCH, GitHost, MergeRequest, Review

BUILDER = "engine:latence-ontology"
PATH = "proposals/{estate}.yaml"
COMMAND = "latence-ontology hub"
CONCEPT_ENTRIES = ("match", "broader", "labels")


def _kind(entry: Mapping[str, Any]) -> str:
    return next(k for k in entry if k != "origin")


def _domain_of(obj: str, domains: Mapping[str, Mapping[str, Any]]) -> str | None:
    schema = obj.partition(".")[0]
    found = [d for d, spec in sorted(domains.items()) if schema in (spec.get("schemas") or [])]
    return found[0] if len(found) == 1 else None


def split(
    document: Mapping[str, Any], domains: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Each domain's share of a proposal file, as a file of its own, and the entries that go to
    no named domain."""
    shares: dict[str, list[dict[str, Any]]] = {}
    unplaced = []
    for entry in document.get("entries") or []:
        kind = _kind(entry)
        body = entry[kind]
        if kind == "match":
            named = [str(c["domain"]) for c in body["concepts"]]
        elif kind == "link":
            named = [d for d in [_domain_of(str(body["record"]["object"]), domains)] if d]
        else:
            named = [d for d in [_domain_of(str(body["object"]), domains)] if d]
        named = [d for d in named if d in domains]
        if not named:
            unplaced.append({"element": str(entry["origin"]["element"]), "kind": kind})
        for domain in named:
            shares.setdefault(domain, []).append(dict(entry))
    files = {}
    for domain, entries in sorted(shares.items()):
        head = ({"format": "layer-proposal/3"} if any(_kind(e) in CONCEPT_ENTRIES for e in entries)
                else {"format": "layer-proposal/2"} if any(_kind(e) == "link" for e in entries)
                else {})
        files[domain] = {**head, "source": document.get("source", "ontology"), "entries": entries}
    return files, unplaced


def submit(
    host: GitHost,
    document: Mapping[str, Any],
    estate: str,
    domains: Mapping[str, Mapping[str, Any]],
    at: str,
) -> dict[str, Any]:
    """Commit each domain's share on a branch of the builder's own and open a merge request into
    the domain's branch. The same file gives the same branch, commit and request."""
    files, unplaced = split(document, domains)
    opened = []
    for domain, share in files.items():
        text = to_yaml(share, estate, command=COMMAND)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        branch = f"builder/{estate}/{domain}/{digest}"
        target = DOMAIN_BRANCH.format(domain=domain)
        n = len(share["entries"])
        host.commit(branch, {PATH.format(estate=estate): text},
                    f"{estate}: {n} approved entr{'y' if n == 1 else 'ies'} for {domain}",
                    BUILDER, at, start_from=target)
        earlier = [m for m in host.merge_requests() if m.source == branch and m.target == target]
        # The same share was proposed before (merged or not): its request stands for it.
        mr = earlier[-1] if earlier else host.open_merge_request(
            branch, target, f"{estate}: approved entries for {domain}", BUILDER)
        opened.append({"merge_request": mr.id, "domain": domain, "branch": branch,
                       "entries": n})
    return {"opened": opened, "unplaced": unplaced}


def _ours(host: GitHost, estate: str) -> list[MergeRequest]:
    return [m for m in host.merge_requests() if m.source.startswith(f"builder/{estate}/")]


def _entries(host: GitHost, mr: MergeRequest, estate: str) -> list[dict[str, Any]]:
    text = host.read(mr.source, PATH.format(estate=estate)) or ""
    doc = yaml.safe_load(text) or {}
    return list(doc.get("entries") or [])


def push_reviews(
    host: GitHost,
    estate: str,
    domains: Mapping[str, Mapping[str, Any]],
    certifications: Sequence[Mapping[str, Any]],
    at: str,
) -> list[dict[str, Any]]:
    """Write on each open request what the builder's review says of it: the approving reviewers'
    approvals, and the domain's data owner's certification where every element of the request
    carries it on the claim it states. Nothing is written twice."""
    written = []
    certified = {(str(c["subject"]), str(c["claim"]), str(c["by"])): c for c in certifications}
    for mr in _ours(host, estate):
        if mr.state not in ("open", "blocked"):
            continue
        domain = mr.target.removeprefix("domains/")
        owner = str((domains.get(domain) or {}).get("data_owner") or "")
        have = {(r.by, r.role, r.verdict) for r in mr.reviews}
        entries = _entries(host, mr, estate)
        wanted: list[Review] = []
        for by in sorted({str(e["origin"]["decision"]["by"]) for e in entries}):
            wanted.append(Review(by, "reviewer", "approved", at))
        elements = {(str(e["origin"]["element"]), str(e["origin"]["claim"])) for e in entries}
        if owner and elements and all((el, claim, owner) in certified for el, claim in elements):
            day = max(str(certified[(el, claim, owner)]["at"])[:10] for el, claim in elements)
            wanted.append(Review(owner, "data_owner", "certified", day))
        for review in wanted:
            if (review.by, review.role, review.verdict) not in have:
                host.review(mr.id, review)
                written.append({"merge_request": mr.id, "by": review.by, "role": review.role,
                                "verdict": review.verdict})
    return written


def pull_certifications(
    host: GitHost,
    estate: str,
    domains: Mapping[str, Mapping[str, Any]],
    certifications: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """The certifications a data owner gave on the git host that the build folder does not hold:
    one per element of the certified request, on the claim the file states."""
    held = {(str(c["subject"]), str(c["claim"]), str(c["by"])) for c in certifications}
    new: list[dict[str, str]] = []
    for mr in _ours(host, estate):
        domain = mr.target.removeprefix("domains/")
        owner = str((domains.get(domain) or {}).get("data_owner") or "")
        given = [r for r in mr.reviews if r.role == "data_owner" and r.verdict == "certified"
                 and r.by == owner]
        if not given:
            continue
        for e in _entries(host, mr, estate):
            line = {"subject": str(e["origin"]["element"]), "claim": str(e["origin"]["claim"]),
                    "by": owner, "at": given[0].at[:10]}
            key = (line["subject"], line["claim"], line["by"])
            if key not in held:
                held.add(key)
                new.append(line)
    return new


def _state(mr: MergeRequest) -> str:
    verdicts = {(r.role, r.verdict) for r in mr.reviews}
    if mr.state == "closed":
        return "retired"
    if mr.state == "merged" or ("data_owner", "certified") in verdicts:
        return "certified"
    if ("reviewer", "approved") in verdicts:
        return "approved"
    return "proposed"


def views(
    host: GitHost,
    estate: str,
    releases: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """What the Ontology page shows of the hub for an estate: each request with its diff, its
    entries, its reviews with their actors, its state, its tag and the release the host built
    from it (``releases``: by tag, the host's receipt's version and digest); and each domain's
    history with its actors."""
    requests: list[dict[str, Any]] = []
    for mr in _ours(host, estate):
        entries = _entries(host, mr, estate)
        requests.append({
            "id": mr.id,
            "domain": mr.target.removeprefix("domains/"),
            "branch": mr.source,
            "target": mr.target,
            "state": _state(mr),
            **({"blocked": mr.blocked_by} if mr.state == "blocked" else {}),
            "entries": [{"kind": _kind(e), "element": str(e["origin"]["element"]),
                         "claim": str(e["origin"]["claim"]),
                         "decided_by": str(e["origin"]["decision"]["by"])} for e in entries],
            "reviews": [{"by": r.by, "role": r.role, "verdict": r.verdict, "at": r.at,
                         **({"note": r.note} if r.note else {})} for r in mr.reviews],
            "diff": host.diff(mr.target if mr.state != "merged" else f"{mr.merge_commit}^1",
                              mr.source),
            **({"tag": mr.tag} if mr.tag else {}),
            **({"release": dict((releases or {})[mr.tag])}
               if mr.tag and mr.tag in (releases or {}) else {}),
        })
    domains = sorted({str(r["domain"]) for r in requests})
    history = {
        d: [{"commit": c.sha, "by": c.author, "at": c.at, "message": c.message}
            for c in host.history(DOMAIN_BRANCH.format(domain=d))]
        for d in domains
    }
    tags = [{"tag": t.name, "commit": t.commit, "message": t.message} for t in host.tags()
            if t.name.split("/")[0] in domains]
    return {"format": "hub-view/1", "estate": estate, "merge_requests": requests,
            "history": history, "tags": tags}
