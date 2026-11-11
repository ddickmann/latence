"""Thin wrappers over the agreed model set (ticket 20; see NEXT.md).

Every wrapper imports its library lazily, picks the device once, and exposes a small numpy
interface so that run_a.py / run_b.py and the tests can swap in a fake with the same methods.
Nothing here decides anything: these produce scores, the calibrated bands in er.py decide.

Model ids are the owner's decision of 2026-11-08; override them with the SPIKE_* variables only
to point at a local directory.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

EMBEDDER_ID = os.environ.get("SPIKE_EMBEDDER", "google/embeddinggemma-2")
RERANKER_ID = os.environ.get("SPIKE_RERANKER", "Qwen/Qwen3-Reranker-0.6B")
RELEX_ID = os.environ.get("SPIKE_RELEX", "knowledgator/gliner-relex-large-v1.0")
DECIDE_ID = os.environ.get("SPIKE_DECIDE", "fastino/GLiNER2.5-Decide")
GLIDE_DIR = os.environ.get("SPIKE_GLIDE_DIR", "models/GLiDE-0.4B")  # `python pin_models.py --glide 0.4B` fetches it

# Instructions are part of a run's identity: changing one changes the scores.
SAME_RECORD_INSTRUCTION = (
    "Given two master-data records, judge whether they describe the same real-world organisation"
)
LINK_INSTRUCTION = (
    "Given a mention of an organisation in a document, judge whether the knowledge-base entry "
    "is the organisation the mention refers to"
)
SAME_QUESTION = "Do record A and record B describe the same real-world organisation?"
SAME_PERSON_INSTRUCTION = "Given two records about people, judge whether they describe the same real-world person"
SAME_PERSON_QUESTION = "Do record A and record B describe the same real-world person?"
LINK_QUESTION = "Is the knowledge-base entry the organisation that the marked mention refers to?"


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PairScorer(Protocol):
    """Scores (left text, right text) pairs; higher means more likely the same thing."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray: ...


class Embedder:
    """EmbeddingGemma 2, text path only, 256-d Matryoshka vectors, L2-normalised."""

    def __init__(self, model_id: str = EMBEDDER_ID, dim: int = 256, batch_size: int = 128) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        dev = device()
        # float16 silently returns NaN for this model (model card); bf16 only where supported.
        dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.model = SentenceTransformer(
            model_id,
            device=dev,
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"dtype": dtype},
        )
        self.dim, self.bs = dim, batch_size

    def encode(self, texts: Sequence[str], prompt_name: str) -> np.ndarray:
        """``prompt_name``: "SentenceSimilarity" for symmetric record matching, "SearchQuery" /
        "Document" for mention -> anchor retrieval."""
        out = self.model.encode(
            list(texts),
            prompt_name=prompt_name,
            truncate_dim=self.dim,
            normalize_embeddings=True,
            batch_size=self.bs,
            convert_to_numpy=True,
        )
        return np.asarray(out, dtype=np.float32)


class Reranker:
    """Qwen3-Reranker-0.6B as a pairwise scorer. Returns raw logit differences (yes - no)."""

    def __init__(self, instruction: str, model_id: str = RERANKER_ID, batch_size: int = 32, max_length: int = 512) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        dev = device()
        self.model = CrossEncoder(
            model_id,
            device=dev,
            # The checkpoint defaults to bf16, which is slow on a CPU; keep it on a GPU.
            model_kwargs={} if dev == "cuda" else {"dtype": torch.float32},
            prompts={"task": instruction},
            default_prompt_name="task",
            max_length=max_length,
        )
        self.bs = batch_size

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self.model.predict(list(pairs), batch_size=self.bs), dtype=np.float32)


def _pair_state(left: str, right: str) -> str:
    return f"A: {left}\nB: {right}"


class GlinerDecide:
    """GLiNER2.5-Decide as a yes/no decision model. ``score`` returns P(yes) in [0, 1].

    The question and the two labels are the whole prompt, so they are part of a run's identity.
    """

    TASK = {"same": ["yes", "no"]}

    def __init__(self, question: str, model_id: str = DECIDE_ID, batch_size: int = 16) -> None:
        from gliner2 import AutoExtractor

        self.model = AutoExtractor.from_pretrained(model_id)
        self.model.to(device())
        self.question, self.bs = question, batch_size

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        texts = [f"{self.question}\n{_pair_state(a, b)}" for a, b in pairs]
        outs = self.model.batch_classify_text(texts, self.TASK, batch_size=self.bs, include_confidence=True)
        p = []
        for o in outs:
            ans = o["same"]
            conf = float(ans["confidence"])
            p.append(conf if ans["label"] == "yes" else 1.0 - conf)
        return np.array(p, dtype=np.float32)


class GlideDecide:
    """GLiDE (a `noul` yes/no question) as a decision model. ``score`` returns P(yes).

    GLiDE ships its own `glide` package inside the downloaded model directory, with pinned
    requirements. The smallest variant (0.4B) is the default (owner, 2026-11-11: start small,
    go up only if needed); its model card states under 2 GB of GPU memory.
    """

    def __init__(self, question: str, model_dir: str = GLIDE_DIR, batch_size: int = 16) -> None:
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from glide import Glide  # type: ignore[import-not-found]  # provided by the model directory

        self.model = Glide.from_pretrained(model_dir)
        self.question, self.bs = question, batch_size

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        out: list[float] = []
        for i in range(0, len(pairs), self.bs):
            reqs = [
                {"state": _pair_state(a, b), "questions": {"same": {"type": "noul", "instructions": self.question}}}
                for a, b in pairs[i : i + self.bs]
            ]
            for r in self.model.decide_batch(reqs):
                if not isinstance(r, dict):  # a glide.RequestError: an unknown stays unknown
                    out.append(float("nan"))
                else:
                    out.append(float(r["answers"]["same"]["noul"]))
        return np.array(out, dtype=np.float32)


class RelexExtractor:
    """GLiNER-Relex: entities with character offsets, and relations between them, in one pass."""

    def __init__(self, model_id: str = RELEX_ID, batch_size: int = 8) -> None:
        from gliner import GLiNER

        self.model = GLiNER.from_pretrained(model_id).to(device())
        self.bs = batch_size

    def extract(
        self,
        texts: Sequence[str],
        entity_labels: Sequence[str] | dict[str, str],
        relation_labels: Sequence[str],
        threshold: float = 0.5,
        relation_threshold: float = 0.7,
    ) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
        ents: list[list[dict[str, Any]]] = []
        rels: list[list[dict[str, Any]]] = []
        for i in range(0, len(texts), self.bs):
            e, r = self.model.inference(
                texts=list(texts[i : i + self.bs]),
                labels=entity_labels,
                relations=list(relation_labels),
                threshold=threshold,
                relation_threshold=relation_threshold,
                return_relations=True,
                flat_ner=False,
            )
            ents.extend(e)
            rels.extend(r)
        return ents, rels


def revision(model_id: str) -> str | None:
    """The resolved Hub commit of a cached model, for results/models.json. None if not cached."""
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == model_id:
                return sorted(repo.revisions, key=lambda r: r.last_modified)[-1].commit_hash
    except Exception:  # noqa: BLE001 - a missing cache is not an error here
        return None
    return None
