# Build: the hub's builder side: approved entries as a merge request through a git port

Type: build
Mode: HITL (a review surface; the host's side is built on its branch)
Status: open
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
