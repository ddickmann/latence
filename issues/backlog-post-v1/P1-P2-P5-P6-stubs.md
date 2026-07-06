# P1, P2, P5, P6 — plugin stubs (preserved)

Concise stubs; expand to full issues when scheduled. All optional plugins behind the Capability/Provider seam.

## P1 — Retrieval-serving plugin
**Ports**: ColBERT pooler (`superpod/vllm_plugins/colbert_plugin/pooler_v2.py`, late-interaction 768→128, skiplist, vLLM shims) + fused CUDA kernels (RoPE+local-attn, GLU+matmul, dropout+residual). Verdict: pooler PORT-AS-IS, kernels PORT-WITH-HARDENING (validate vs torch.compile on target GPU). Provides ColBERT/ColPali dense retrieval as an Embedder/Retriever Provider.
**AC**: late-interaction embeddings served via endpoint Provider; kernels benchmarked for correctness+speed vs baseline; opt-in, GPU-only.

## P2 — Compression plugin
**Ports**: LLMLingua2 pipeline + Qwen3 compression plugin (learned token merging). Verdict: PORT-WITH-HARDENING (benchmark compression ratio vs quality loss). Provides a text-compression Capability for token-budget-constrained RAG.
**AC**: configurable compression ratio; digit/entity preservation; quality-vs-ratio curve reported.

## P5 — Source-system connectors
**Build**: SharePoint, Confluence, Google Drive, SQL, Slack as Source Providers (ADR-0014) — OAuth + pagination + rate-limit + incremental cursors, stamping Provenance. Verdict: BUILD (net-new). The enterprise "integrated" surface; each connector independent.
**AC**: each connector lists+fetches with Provenance and supports incremental Delta cursors; auth via adopter's mechanism; no secrets stored by the framework.

## P6 — Trace / groundedness scoring plugin
**Ports**: superpod trace service (RAG groundedness / phantom detection). Verdict: RECAST. Scores generated answers against the RAG corpus for hallucination.
**AC**: groundedness score per answer with cited evidence spans; opt-in.
