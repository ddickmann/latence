# Build: the host's local backend runs on this machine

Type: task
Mode: AFK
Status: resolved
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

## Answer

Done on 2026-11-11.

- **pgvector.** The host's document search needs sparse vectors and the iterative index scan,
  so pgvector 0.8 or newer; the distribution ships 0.6.0. Built 0.8.7 for Postgres 16.15 from
  the source release on the distribution's own archive (checksum equal to the one its package
  description states), with `postgresql-server-dev-16`. The release on the code host could not
  be fetched from this machine.
- **The cluster** runs as the `postgres` user on a Unix socket only, trust on that socket,
  port 5433, user `metis`: `initdb -U metis -A trust -E UTF-8 --locale=C`, then `pg_ctl` with
  `listen_addresses=''`. The host's variable:
  `METIS_ESTATE_POSTGRES=postgresql://metis@:5433/postgres?host=/var/run/postgresql`.
- **Proof.** The host's setup of its TalkToData world succeeded against it; the host's 115
  tests of its pgvector plug-in pass against it (`METIS_TEST_POSTGRES_URL`).
- **Toolchain:** Node 24.11.0 and `uv`, as before.
- On the owner's machines Postgres 17 and pgvector come from the package manager, which ships
  a current pgvector; nothing of the above is needed there.

## From the map (moved 12 Nov)

Route table row: Postgres with pgvector on this machine | AFK | 34
