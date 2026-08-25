# deploy/Dockerfile — the PRODUCTION pipeline image (separate from demo/Dockerfile).
#
# Unlike the demo image (CPU-only, offline, in-process, ships the whole workspace to run one
# canned demo), this image is the *operator's* runtime: latence-core + the LIGHT served/endpoint
# Provider set (the OCR + LLM + embedder + NER Providers are thin HTTP clients — the heavy models
# live behind a vLLM / OpenAI-compatible endpoint, NOT in this image) + the observability extra.
# It exposes the `latence` CLI as the entrypoint, runs as a NON-ROOT user, and pins its base +
# uv so a rebuild is reproducible.
#
# Provider set (all light — no torch/transformers in the image; that is the whole point of the
# served path, ADR-0030): parser.lighton_vllm (served LightOn-OCR-2), parser.endpoint,
# entity.endpoint, relation.llm, embedding.endpoint — plus the in-core CPU spine and the CPU text
# parsers that ship in the default workspace set. The heavy in-process OCR (parser.lighton, which
# pulls torch) is installed --no-deps for its shared rasteriser ONLY, so no GPU stack lands here.
#
# Build (from the repo root, so the whole uv workspace is in the build context):
#   docker build -f deploy/Dockerfile -t latence:prod .

# ---- Stage 1: builder — resolve + install into a self-contained venv ---------------------
FROM python:3.12-slim AS builder

# uv, pinned to an exact version (never `latest`) for a reproducible build — the same
# reproducibility contract uv.lock gives the dependency set.
COPY --from=ghcr.io/astral-sh/uv:0.8.11 /uv /uvx /bin/

# Byte-compile + copy mode: a hermetic, relocatable venv we can hand to the runtime stage.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Copy the workspace. The build context is the repo root (see the build command above).
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY README.md CONTEXT.md ./

# 1) The default workspace set (latence-core + CPU text parsers + the CPU spine), lockfile-pinned.
#    --no-editable is load-bearing: a plain `uv sync` installs workspace members EDITABLE
#    (site-packages .pth files pointing at /app/packages/*/src), and the runtime stage copies
#    only the venv — so every workspace import would 404 in the final image. Measured, not
#    hypothetical: the image built green and `latence --help` died with ModuleNotFoundError
#    until this flag (ticket wayfinder-enterprise-serving/04). The smoke RUN in the runtime
#    stage keeps it from regressing.
RUN uv sync --frozen --no-dev --no-editable

# 2) The observability EXPORT extra (ADR-0047) so `LATENCE_METRICS=prometheus|otel` is live in
#    this image. Optional group on latence-core; installed here, off unless the env opts in.
#    --no-sources on every `uv pip install` below is the second half of the --no-editable fix:
#    inside the workspace, a bare path install is rerouted through tool.uv.sources and lands
#    EDITABLE again (measured: the sync was non-editable and this step alone put the
#    /app/packages .pth back). --no-sources installs the path as a plain project; its latence-*
#    dependencies are already satisfied by the sync, so nothing resolves from an index.
RUN uv pip install --no-sources "./packages/latence-core[observability]"

# 3) The LIGHT served/endpoint Provider set. The OpenAI-client Providers pull only `openai`
#    (light); parser.lighton (heavy: torch/transformers) is installed --no-deps for its
#    pure-Python rasteriser alone, and parser.lighton_vllm gets its real light deps explicitly —
#    so the served OCR client works with NO GPU stack in the image.
RUN uv pip install --no-sources \
        "./packages/latence-parser-endpoint" \
        "./packages/latence-ner-endpoint" \
        "./packages/latence-embedder-endpoint" \
        "./packages/latence-relation-llm" \
    && uv pip install --no-sources --no-deps \
        "./packages/latence-parser-lighton" \
        "./packages/latence-parser-lighton-vllm" \
    && uv pip install "openai>=1.0,<3" "pypdfium2>=4.0,<6" "pillow>=10.0,<13"

# ---- Stage 2: runtime — minimal, non-root, just the venv + the CLI ------------------------
FROM python:3.12-slim AS runtime

# A dedicated non-root user (uid 10001). The image never runs as root; the PVC/volume the operator
# mounts for corpus/output must be writable by this uid (documented in docs/deployment.md).
RUN groupadd --gid 10001 latence \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin latence

# Copy the resolved venv from the builder (no uv, no build tools, no source tree in runtime).
COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Offline/air-gapped friendly HF defaults: point the model cache at a mounted volume and never
    # phone home. Harmless for the served path (no local model load); documented for the rare
    # in-image model case in docs/deployment.md.
    HF_HOME="/data/hf-cache" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# The data root (corpus in, artifacts out, model cache) is a mounted volume/PVC at runtime.
WORKDIR /app
USER 10001

# Build-time smoke: the image is broken if its own entrypoint cannot import. This is what
# caught the editable-venv defect above — keep it, so `docker build` succeeding MEANS the
# CLI stands up (no network, no model load; just the import graph + Typer wiring).
RUN latence --help > /dev/null

# The `latence` CLI is the entrypoint; the default command prints help. A deployment overrides the
# command with e.g. `latence run /etc/latence/served-ocr.yaml --run-id run-0001`.
ENTRYPOINT ["latence"]
CMD ["--help"]
