"""Dev-only per-regime calibration A/B (ticket 09) — MuSiQue-full dev, no test contact."""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "musique-subset"))
import run_official_eval as R

def main() -> None:
    base = HERE.parent / "official-data" / "musique-full"
    dev = [json.loads(l) for l in (base / "musique-full-dev.jsonl").open()]
    flat = {str(b): 0.25 for b in R.BUDGETS}
    cfgs = [
        {"name": "tier2-raw", "tier": "tier2", "costs": "heuristic", "weak": "on",
         "calib": "off", "shares": flat},
        {"name": "tier2-calib", "tier": "tier2", "costs": "heuristic", "weak": "on",
         "calib": "on", "shares": flat},
    ]
    out = R.run(dev, cfgs, base, str(base / "export/graph-hyperedges.kv"),
                str(base / "calibration-map-per-regime.json"), tag="calib-ab")
    for name in ("tier2-raw", "tier2-calib"):
        curve = [out[name][str(b)] for b in R.BUDGETS]
        print(f"{name}: auc={sum(curve)/len(curve):.4f} curve={curve}")
    (base / "calib-ab-dev.json").write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
