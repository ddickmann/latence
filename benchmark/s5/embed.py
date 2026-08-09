# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Query-side embedding with the **same** model the S3 export used — the dense leg's precondition.

The exported corpus vectors come from ``ibm-granite/granite-embedding-311m-multilingual-r2``
under ``embedding.sentence_transformers`` (the S3 pipeline yaml). A dense leg is only meaningful
if the query vector lands in that same space, so this reproduces that provider's pipeline
exactly rather than substituting a convenient local encoder:

* **no instruction prefix** — Granite r2's card encodes text directly (unlike the e5 family),
  and the framework's own ``latence_embedder_st`` provider records the same,
* **CLS pooling** — from the model's ``1_Pooling/config.json`` (``pooling_mode_cls_token``),
* **L2 normalisation** — the provider passes ``normalize_embeddings=True``, which is why the
  exported vectors have unit norm and cosine is a plain dot product.

``sentence-transformers`` is not in this repo's CPU-first dev environment (ADR-0016), so the
three steps run on plain ``transformers``. That is a re-implementation of a shipped provider, so
it is **verified against the artifact it must match**: ``tests/test_embed.py`` re-embeds real
exported chunks and asserts each one's stored vector is its own nearest neighbour in the export.
Self-cosine is ~0.99 rather than 1.0 (the export ran on GPU in reduced precision); rank-1
self-retrieval is the property the dense leg actually depends on, so that is what is asserted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The model the S3 export embedded the corpus with (benchmark/sota-campaign/.../pipelines/*.yaml).
GRANITE_MODEL = "ibm-granite/granite-embedding-311m-multilingual-r2"
#: The export's own ``sentence_bert_config.json`` ``max_seq_length`` — matched so a long text
#: truncates at exactly the point the corpus side truncated it (queries are far shorter).
_MAX_TOKENS = 32768


class GraniteQueryEmbedder:
    """CLS-pooled, L2-normalised query vectors from the export's own embedding model.

    Loaded lazily on first :meth:`encode` so constructing a run plan — or running a bm25-only
    leg — never pays for a 1.2 GB checkpoint. Deterministic: ``eval()`` mode, no dropout, fixed
    ``float32``, no sampling.
    """

    def __init__(self, model_id: str = GRANITE_MODEL, *, max_tokens: int = _MAX_TOKENS) -> None:
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            self._model = AutoModel.from_pretrained(self._model_id, dtype=torch.float32).eval()
        return self._tokenizer, self._model

    def encode(self, texts: Sequence[str], *, batch_size: int = 16) -> Any:
        """``(len(texts), 768)`` float32 unit vectors, in input order."""
        import torch

        tokenizer, model = self._load()
        out: list[Any] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_tokens,
                return_tensors="pt",
            )
            with torch.no_grad():
                hidden = model(**encoded).last_hidden_state[:, 0]
                vectors = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
            out.append(vectors.numpy())
        if not out:
            return np.zeros((0, 768), dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32, copy=False)


__all__ = ["GRANITE_MODEL", "GraniteQueryEmbedder"]
