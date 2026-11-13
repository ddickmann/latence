"""Download the agreed models and record what was downloaded (NEXT.md step 1).

Writes ``results/models.json``: for each model its id, the revision hash that was fetched, the
licence its model card declares, and its size on disk. A run's numbers are only reproducible
against these revisions.

Usage: python pin_models.py [--glide 0.4B] [--out results/models.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import models as M

ROLES = {
    "embedder": M.EMBEDDER_ID,
    "reranker": M.RERANKER_ID,
    "relex": M.RELEX_ID,
    "decide": M.DECIDE_ID,
}


def _size(path: Path) -> int:
    return sum(f.resolve().stat().st_size for f in path.rglob("*") if f.is_file())  # the hub cache links to blobs


def pin(model_id: str, local_dir: str | None = None) -> dict[str, object]:
    from huggingface_hub import model_info, snapshot_download

    info = model_info(model_id)
    card = info.card_data.to_dict() if info.card_data is not None else {}
    path = Path(snapshot_download(model_id, revision=info.sha, local_dir=local_dir))
    return {
        "id": model_id,
        "revision": info.sha,
        "licence": card.get("license_name") or card.get("license"),
        "last_modified": info.last_modified.isoformat() if info.last_modified else None,
        "bytes": _size(path),
        "path": str(path) if local_dir else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glide", default="", help="comma-separated GLiDE sizes to fetch into models/, e.g. 0.4B,2B")
    ap.add_argument("--out", default="results/models.json")
    a = ap.parse_args()
    out_path = Path(a.out)
    pinned: dict[str, object] = json.loads(out_path.read_text()) if out_path.exists() else {}
    for role, model_id in ROLES.items():
        pinned[role] = pin(model_id)
        print(role, pinned[role], flush=True)
    for size in filter(None, a.glide.split(",")):
        pinned[f"glide-{size}"] = pin(f"fastino/GLiDE-{size}", local_dir=f"models/GLiDE-{size}")
        print(size, pinned[f"glide-{size}"], flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pinned, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
