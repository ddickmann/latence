# The showable demo image (ADR-0011): `docker compose up` ingests the bundled messy
# corpus and produces parsed markdown, a browsable knowledge graph, and the Quality
# Report — CPU-only, no cloud, no GPU, no credentials. Everything is offline; the only
# network use is `uv sync` at build time (pinned by uv.lock).
FROM python:3.12-slim

# uv for a fast, lockfile-pinned, reproducible install. Pin the uv image to an exact
# version (never `latest`) so a rebuild is reproducible and cannot silently pull a
# changed uv — the same reproducibility contract `uv.lock` gives the dependency set.
COPY --from=ghcr.io/astral-sh/uv:0.8.11 /uv /uvx /bin/

WORKDIR /app

# Copy the workspace (the demo installs latence-core + latence-parser-document +
# latence-demo from source — the CPU demo set, ADR-0016). The heavy/endpoint Provider
# packages are NOT installed: the demo is pure-Python, CPU-only.
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY examples ./examples
COPY README.md CONTEXT.md ./

# Install only the CPU demo set into a venv (no torch, no cloud client, no GPU).
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"
# Hash-stable runs so the demo (and its Quality Report + KG) are byte-reproducible.
ENV PYTHONHASHSEED=0

# Artifacts land under this Storage root; the compose file mounts it to ./out on the host.
# In the container this is a dedicated, non-world-writable mount (not a shared /tmp), so the
# predictable-path/world-writable concern of the host `file:///tmp/latence-demo` default does
# not apply here. Hardening note (real deployments): point this at a private, per-run,
# non-predictable Storage root to avoid a shared-/tmp symlink/clobber race.
ENV LATENCE_DEMO_STORAGE="file:///data"

# Run the whole showable demo. Exits non-zero if a planted dangerous file was not caught
# or contract-completeness failed (a built-in smoke test), so `docker compose up` doubles
# as CI-in-a-box.
CMD ["sh", "-c", "latence-demo --storage-uri \"$LATENCE_DEMO_STORAGE\""]
