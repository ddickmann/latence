# deploy/env/served-ocr-server/Dockerfile — the pinned vLLM LightOn-OCR serving image (#196).
#
#   docker build -f deploy/env/served-ocr-server/Dockerfile -t latence/ocr-server:0.24.0 \
#                deploy/env/served-ocr-server
#   docker run --gpus all -p 8000:8000 -v "$PWD/hf-cache:/root/.cache/huggingface" \
#                latence/ocr-server:0.24.0
#
# This image IS the environment: nothing is resolved at build time. `requirements.lock` is the
# solved set (681 exact pins), `constraints.txt` carries the enforced boundaries AND the
# exclusions, and both are committed next to this file.
#
# HOW THE EXCLUSIONS ARE ENFORCED (the acceptance criterion — pins, never "uninstall afterwards"):
#
#   1. The install below passes `-c constraints.txt`, in which `flash-attn` and `torchcodec` are
#      pinned to `<0` — a specifier no PEP 440 release can satisfy. Anything that asks for them,
#      directly or transitively, makes the resolver exit non-zero. The wheel cannot land.
#   2. That same file is baked in as PIP_CONSTRAINT / UV_CONSTRAINT, so the rule OUTLIVES the
#      build: a later `pip install flash-attn` inside a running container fails too. This is the
#      difference between an enforced exclusion and a README bullet nobody re-reads.
#   3. `verify-exclusions.py` fails the BUILD if either module is importable in the finished image
#      — the belt to (1)+(2)'s braces, so a base image that shipped one pre-installed cannot slip
#      through. It asserts absence by import, not by reading a manifest.
#
# Base: NVIDIA's CUDA 13.0 runtime, DIGEST-pinned (a tag is mutable; a digest is the artifact).
# `runtime` rather than `devel` is deliberate and sufficient: the CUDA toolchain FlashInfer/tilelang
# JIT against (nvcc, nvrtc, cuda-crt, nvjitlink) is carried by the lock as pip wheels
# (nvidia-cuda-nvcc==13.2.86, nvidia-cuda-nvrtc==13.0.88, ...), so the toolkit is pinned by the
# lock too instead of drifting with the base image.
# Digest resolved from registry-1.docker.io for nvidia/cuda:13.0.2-runtime-ubuntu24.04 on
# 2026-07-18. Re-resolve with:
#   docker buildx imagetools inspect nvidia/cuda:13.0.2-runtime-ubuntu24.04
FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04@sha256:6a0e31b59e70890446f2c17356d6efc0d54260090a8c63d6ca7c2ad049db95d2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# PEP 668: Ubuntu 24.04's system Python is externally managed, so EVERYTHING lives in a venv on
# PATH. No `--break-system-packages`, ever — that flag is how a serving box ends up with two half
# environments and an unreproducible `pip list`.
ENV VIRTUAL_ENV=/opt/latence-ocr \
    PATH=/opt/latence-ocr/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-venv ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python3.12 -m venv "$VIRTUAL_ENV" \
 && "$VIRTUAL_ENV/bin/python" -m pip install --no-cache-dir --upgrade pip

# The enforced boundaries land BEFORE the install and stay for the image's whole life.
COPY constraints.txt /etc/latence/ocr-constraints.txt
ENV PIP_CONSTRAINT=/etc/latence/ocr-constraints.txt \
    UV_CONSTRAINT=/etc/latence/ocr-constraints.txt

COPY requirements.lock /etc/latence/requirements.lock

# The solved set, installed exactly. `--extra-index-url .../cu130` is what makes the `+cu130` local
# versions in the lock (torch==2.11.0+cu130) reachable; without it pip cannot find the file it is
# pinned to and the build fails loudly rather than substituting the cu12 build.
RUN pip install --no-cache-dir \
      --requirement /etc/latence/requirements.lock \
      --constraint /etc/latence/ocr-constraints.txt \
      --extra-index-url https://download.pytorch.org/whl/cu130

# Build-time proof, not a claim: if either excluded module can be imported, the build fails here.
COPY verify-exclusions.py /etc/latence/verify-exclusions.py
RUN python /etc/latence/verify-exclusions.py

COPY serve.sh /usr/local/bin/serve-ocr
RUN chmod +x /usr/local/bin/serve-ocr

# HF weights cache. Mount a volume here so a restart (or an air-gapped run with HF_HUB_OFFLINE=1)
# does not re-download. NO token and NO weights are baked into the image (ADR-0012).
ENV HF_HOME=/root/.cache/huggingface
VOLUME ["/root/.cache/huggingface"]

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/serve-ocr"]
