# Build: the hub's builder side: approved entries as a merge request through a git port

Type: build
Mode: HITL (a review surface; the host's side is built on its branch)
Status: built on this side against a local bare repository (15 Nov); open: live requests (a real Forgejo remote, the host's route and page)
Blocked by: 58; the host's 309 (decided by the owner, 15 Nov; being built on the host's side)
Emerges from: the host's drafted ticket B-3 (15 Nov); the owner (15 Nov): a git-like ontology
hub

## Question

The owner decided the hub on 15 Nov (the host's 309): history in a git repository per estate
behind a port (any git remote, Forgejo first; tests on a local bare repository, no network). A
domain's stewards alone commit to its branch; anyone else proposes through a merge request into
it, the builder's proposal files too. Cross-domain conflicts are resolved by the conformed
library's owner, the merge blocked meanwhile. A certified merge creates a tag, the host builds a
release (version and digest) from it, and a revert is a new release.

## What is asked (this side)

- The builder commits a proposal file's approved entries on a branch of its own and opens a
  merge request into the domain's branch; it never pushes a domain's branch.
- Its five-state review maps onto the merge request's reviews: a reviewer approves, the data
  owner certifies; written to and read from the git host through the same port.
- It reads the tag a certified merge creates and the release the host built from it (the
  receipt names version and digest).
- Built against an emulated port on a local bare repository; what needs the host's running
  route or a real Forgejo remote is a live request.

## Done when

A reviewer approves and a data owner certifies one domain's merge request while another
domain's is open, and the history shows both; the builder reads the tag and the release back.

## Result (15 Nov)

Built in `latence-ontology`; nothing of the host's code (its 309 is not on its branch yet, and
would not be copied).

- **The port** (`hub_port.py`): `GitHost` with branches, commits, files, diffs, history with
  actors, merge requests, reviews (`reviewer` approves, `data_owner` certifies,
  `conformed_owner` resolves) and tags. **A backend over a local bare repository**
  (`hub_local.py`): the `git` binary in a subprocess with plumbing commands only, no network,
  commit dates from the day so the same inputs make the same commits; merge requests and their
  reviews in `hub.json` beside the repository, as a forge keeps them beside its repositories.
  It holds the owner's rules where a forge would: only a domain's stewards commit to
  `domains/<domain>`; a review as data owner only by the domain's data owner, a resolution only
  by the conformed library's owner; a merge only with a reviewer's approval and the data
  owner's certification, never while blocked, and only by the domain's steward, creating the
  tag `<domain>/<version>`; a revert is a steward's new commit and a new tag, nothing rewritten.
- **The builder's side** (`hub.py`, command `latence-ontology hub --config --out --repo --at
  [--receipt ...]`): the approved entries of the build's proposal file are split by domain (a
  match to each domain it names, any other entry to its object's domain by schema; an entry of
  no named domain is listed, not proposed), each share committed as
  `proposals/<estate>.yaml` on `builder/<estate>/<domain>/<digest>` from the domain's branch,
  and proposed by a merge request; the same share gives the same branch, commit and request,
  also after it was merged. The builder writes each approving reviewer's approval and, where
  every element of a request carries the domain's data owner's certification on its claim, the
  owner's certification; a certification given on the git host is read back into
  `certifications.jsonl`, one per element of the request. `hub.json` holds, per request, its
  state among the five (`proposed`, `approved`, `certified`, `retired`; `blocked` with the
  reason beside it), its diff against the domain's branch, its entries with their claims and
  deciders, its reviews with their actors, its tag and the release the host built from it (the
  host's receipt naming the tag, version and digest; the receipt reader now keeps a stated
  `digest` and `tag`), and each domain's history with its actors.
- **Tests, red first** (`tests/test_ontology_hub.py`, 7; red on the tree of 58: no module
  `latence_ontology.hub`): only stewards commit to a domain's branch, the builder included; the
  builder proposes each domain's share and pushes no domain's branch, and the same file opens
  no second request; one domain's request is approved, certified, merged by its steward and
  released while the other's stays open, and the history shows both with their actors; a
  certification given on the git host is read back; a conflict blocks the merge until the
  conformed library's owner resolves it; a revert is a new tag and rewrites nothing; the
  command end to end on the toy build (a step approved in `decisions.jsonl`, certified on the
  host, merged, its release read back with its digest). The tests caught one defect before it
  landed: after a merge, a rerun opened a second request from the merged branch.

### Live requests

- **A real Forgejo remote**: a Forgejo backend of this port (its REST API for branches, pull
  requests, reviews and tags), recorded against a real instance, needs the remote and a token
  as a network secret; the host builds its own (its 309.2). Until then only the local backend
  exists here.
- **The host's route and page**: `hub.json` is what the Ontology page would show; serving it
  through the host's route under the asker's rights, and a review made on the page reaching the
  git host, need the host's running side of 309 (309.4).
