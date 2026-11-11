# Build: the host's local backend runs on this machine

Type: task
Mode: AFK
Status: open
Blocked by: 34

## Question

The host's local binding is DuckDB with Postgres and pgvector for document search. The machine
this work runs on has the Postgres 16 server files but no pgvector, no container runtime, and
runs as root, which Postgres refuses.

## Goal

- Install `postgresql-16-pgvector` from the distribution's archive. If the host's plug-in needs
  a newer pgvector than that, build that version from its source release.
- One cluster run as the `postgres` user, on a socket with a short path, trust on the socket
  only, the way the host's own test plug-in does it; `CREATE EXTENSION vector`; the host's
  variable for the estate's Postgres set.
- The toolchain the host pins (Node, `uv`).

## Done check

The host's setup command for its TalkToData world succeeds, and the host's tests that need
Postgres and pgvector pass against this cluster. Recorded here: the versions and the commands.

## Tests that prove it

None of our own. The host's existing tests are the proof; they are run and their counts
recorded.
