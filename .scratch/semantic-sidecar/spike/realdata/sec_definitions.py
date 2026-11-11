"""Mine short-name definitions from real annual reports and say what was found.

Source: the packaged corpus of annual reports filed with the US securities regulator
(`eloukas/edgar-corpus` on Hugging Face). The filings are public; the text is downloaded where
this runs and is never committed. The output holds counts and a random sample of mined
definitions (a short name and the name it stands for, both public company names) for a person
to check by eye: the precision of the miner on real text is a hand count over that sample.

Usage: python realdata/sec_definitions.py --filings 2000 --sample 80 --out out/sec_definitions.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from aliases import is_short_form, mine_definitions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filings", type=int, default=2000)
    ap.add_argument("--sample", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(f for f in list_repo_files("eloukas/edgar-corpus", repo_type="dataset") if f.endswith((".parquet", ".jsonl")))
    pick = next((f for f in files if "2020" in f and "test" in f), files[-1])
    path = hf_hub_download("eloukas/edgar-corpus", pick, repo_type="dataset")
    if pick.endswith(".parquet"):
        rows = pq.read_table(path).slice(0, a.filings).to_pylist()
    else:
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for _, line in zip(range(a.filings), fh)]
    found: list[dict[str, str]] = []
    with_any = 0
    chars = 0
    for r in rows:
        text = " ".join(str(r.get(k) or "") for k in ("section_1", "section_7"))
        chars += len(text)
        seen = set()
        for d in mine_definitions(text):
            key = (str(d["short"]), str(d["long"]))
            if key not in seen:
                seen.add(key)
                found.append({"filing": str(r.get("filename", "")), "short": key[0], "long": key[1]})
        with_any += bool(seen)
    rng = random.Random(a.seed)
    acr = [d for d in found if is_short_form(d["short"])]
    out = {
        "source": f"eloukas/edgar-corpus:{pick}",
        "filings_read": len(rows),
        "characters_read": chars,
        "filings_with_a_definition": with_any,
        "definitions": len(found),
        "acronyms": len(acr),
        "short_names": len(found) - len(acr),
        "most_frequent_short_forms": Counter(d["short"] for d in found).most_common(25),
        "sample": rng.sample(found, min(a.sample, len(found))),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "sample"}, indent=1, ensure_ascii=False))
    for d in out["sample"]:
        print(f"  {d['short']!r:<22} <- {d['long'][:110]!r}")


if __name__ == "__main__":
    main()
