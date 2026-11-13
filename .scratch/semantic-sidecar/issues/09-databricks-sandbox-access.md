# Task: get a Databricks sandbox for the engine

Type: task
Mode: HITL
Status: open
Decided by the owner (12 Nov): Does the engine use the host's workspace, or wait for its own
sandbox; Reuse it for read now; write-back and capability checks after v1
Blocked by: 

## Question

Provision what the warehouse path needs to be judged and built:

- a workspace;
- a Unity Catalog catalog loaded with the synthetic estate (reuse Metis's synthetic estate and
  its 13 metric views if possible);
- a serverless SQL warehouse;
- a service principal with read on the catalog and, if the owner allows, on the `system`
  schemas (lineage, query history);
- a separate principal or grant for staged write-back (comments, tags, metric views).

Record where the credentials live (as `secret://` references, never values), the URLs, and the
row and table counts.

## Decision brief for the owner (12 Nov)

- **Question:** Does Latence use the Metis workspace and service principal, or wait for its own
  sandbox?
- **Why his:** access only he can grant or request.
- **Options:**
  - A. Reuse for read now; the write-back principal and capability checks after v1.
  - B. Request a separate sandbox.
  - C. Wait.
- **Facts:**
  - This ticket asks to "reuse Metis's synthetic estate and its 13 metric views if possible".
  - The host's ticket 09 (resolved 6 Nov): a shared workspace, catalog `semantic_agent_demo`, a
    serverless warehouse, one service principal, `system` tables not readable, no further
    identities.
  - Block 7 takes a client id and secret from the shell.
- **Recommendation:** A. Ticket 33's run needs only read and a warehouse, which exist; a second
  principal depends on admins who said no more identities will come.
- **Urgency:** before the switch. Blocks the build against the real catalog.
- **Cost:** exporting the existing credentials in one shell.
- **Decided by the owner (12 Nov):** Reuse it for read now; write-back and capability checks
  after v1.
