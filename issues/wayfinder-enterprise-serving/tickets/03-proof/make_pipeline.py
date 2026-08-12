"""Generate the ticket-03 proof pipeline YAMLs from the committed s3 multihop_rag profile.

The committed campaign pipeline (benchmark/sota-campaign/stages/s3-pipeline/pipelines/
multihop_rag.yaml) is the single source of truth for the profile; this script only rewrites the
environment-bound knobs for a local CPU run:

* ``storage_uri`` + the Source ``path`` -> local proof directories,
* every ``device: cuda`` -> ``cpu`` (the seam has no MPS lane; CPU is bitwise-deterministic,
  which is exactly what the equivalence proof wants),
* optionally strips the dense ``embedder`` block (intermediate churn runs — embeddings are a
  pure per-run export augmentation recomputed from the final row set, so only the two compared
  runs need to pay the ~3.5 s/chunk CPU embedding cost).

Everything else — chunking, labels, thresholds, redaction, graph/hyperedge/enrichment config,
bm25, ``unsafe_unmasked_corpus`` — is carried over byte-for-byte.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[4]
PROFILE = REPO / "benchmark/sota-campaign/stages/s3-pipeline/pipelines/multihop_rag.yaml"


def build(storage_uri: str, source_path: str, *, embedder: bool, name: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    data["name"] = name
    data["storage_uri"] = storage_uri
    for stage in data["stages"]:
        config = stage.get("config") or {}
        if stage["capability"] == "source":
            config["path"] = source_path
        if config.get("device") == "cuda":
            config["device"] = "cpu"
        embed_block = config.get("embedder")
        if isinstance(embed_block, dict):
            if not embedder:
                del config["embedder"]
            elif (
                isinstance(embed_block.get("config"), dict)
                and embed_block["config"].get("device") == "cuda"
            ):
                embed_block["config"]["device"] = "cpu"
        if config:
            stage["config"] = config
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-root", required=True, help="Local data root for the proof runs.")
    args = parser.parse_args()
    root = Path(args.proof_root).resolve()
    out = Path(__file__).resolve().parent / "pipelines"
    out.mkdir(exist_ok=True)
    variants = (
        ("incr-noembed.yaml", f"file://{root}/store-incr", f"file://{root}/input/markdown", False),
        ("incr-embed.yaml", f"file://{root}/store-incr", f"file://{root}/input/markdown", True),
        ("rebuild.yaml", f"file://{root}/store-rebuild", f"file://{root}/input/markdown", True),
    )
    for fname, store, src, embed in variants:
        data = build(store, src, embedder=embed, name=f"ticket03-{fname.removesuffix('.yaml')}")
        (out / fname).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        print(f"wrote {out / fname}")


if __name__ == "__main__":
    main()
