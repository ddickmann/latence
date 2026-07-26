"""Latence schema-induction Providers: per-chunk label induction behind one Capability seam.

Two adapters, genuinely different in cost and failure mode, interchangeable behind the
``schema_induction`` Capability:

* :class:`~latence_schema_inducer.provider.LLMLabelInducer` (``label_inducer.llm``) — any
  OpenAI-compatible endpoint (hosted, OpenRouter, or a LOCAL vLLM/Ollama/llama.cpp box), with the
  full untrusted-output safety-net stack;
* :class:`~latence_schema_inducer.lexicon.LexiconLabelInducer` (``label_inducer.lexicon``) — a
  deterministic, CPU-only, zero-network tagger over an operator-declared candidate type space, for
  a known domain where the per-chunk pass should cost nothing.
"""
from __future__ import annotations

from latence_schema_inducer.lexicon import LexiconLabelInducer
from latence_schema_inducer.provider import LLMLabelInducer

__all__ = ["LLMLabelInducer", "LexiconLabelInducer"]
__version__ = "0.1.0"
