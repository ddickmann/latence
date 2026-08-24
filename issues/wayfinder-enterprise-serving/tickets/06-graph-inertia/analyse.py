# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""Every arm of the S5b graph-inertia experiment, as a pure function of a captured trace.

The design in one sentence: **the expensive half already ran** (:mod:`probe`), so an arm here is
a re-ranking of ids that were already retrieved, and the whole matrix of arms × cut-offs settles
in seconds against the identical candidate lists. Two consequences worth stating, because they
are what make the numbers comparable at all:

* every arm sees the SAME base and the SAME graph lanes — a difference between two arms is the
  promotion policy and nothing else (not a different retrieval, not a different sample, not a
  different seed);
* the paired structure survives, so a delta comes with a per-query distribution and a bootstrap
  interval rather than two point estimates a reader has to eyeball.

**The ceiling arm is the one that matters.** ``oracle`` is not a policy anyone would ship: it puts
every gold chunk the graph lanes reached at the front of the list. It exists to bound the question
— no promotion policy, however clever, can beat it — so a ceiling of zero closes the investigation
without running a single tuning sweep, and a ceiling well above the shipped arm says the append
policy (not the graph) is what costs the recall.

RRF here is ``Σ_l w_l / (60 + rank_l)`` with ties broken on ascending id: the shipped fuser's own
formula and the harness's own tiebreak, recomputed from ranks because RRF depends on nothing else.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as stats
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.s5.metrics import ndcg_at_k, recall_at_k  # noqa: E402

HERE = Path(__file__).resolve().parent

#: The cut-offs every arm reports. ``10`` is the matrix's headline; ``50``/``100`` are where a
#: candidate appended below a 100-200 item base could first become visible at all.
CUTS = (1, 5, 10, 20, 50, 100)

#: The shipped RRF constant (``rank_score_k`` / the fuser's own default).
RRF_K = 60

#: The shipped rescue append cap.
RESCUE_CAP = 50

BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 20260824


# -- the arms ------------------------------------------------------------------------------------


def arm_base(row: dict) -> list[str]:
    """Control: the base ranking alone — dense+BM25 RRF, max-normalised, Expander-rescored."""
    return list(row["base"])


def arm_append(row: dict, cap: int = RESCUE_CAP) -> list[str]:
    """The shipped policy: the whole base, then the rescue candidates it did not already hold."""
    seen = set(row["base"])
    tail = [c for c in row["rescue"] if not (c in seen or seen.add(c))]
    return list(row["base"]) + tail[:cap]


def arm_oracle(row: dict, cap: int = RESCUE_CAP) -> list[str]:
    """The CEILING: the same candidate set as ``append``, with rescued gold moved to the front.

    Deliberately not shippable — it reads the gold labels. It answers the only question worth
    asking first: is there anything down there to promote? Any real policy is bounded by this.
    """
    gold = set(row["gold"])
    ranked = arm_append(row, cap)
    base = set(row["base"])
    promoted = [c for c in ranked if c in gold and c not in base]
    return promoted + [c for c in ranked if c not in set(promoted)]


def arm_fuse(row: dict, weight: float) -> list[str]:
    """RRF the graph lane INTO the base at ``weight`` — the append-vs-fuse continuum.

    ``weight=0`` is the base exactly; ``weight=1`` is the co-equal fusion the ladder's ``kg`` rung
    already measured regressing. The sweep between them is what the append policy's rationale
    ("graph signals must not displace strong lexical/dense hits") predicts to be uniformly bad,
    and the point of running it is that the prediction is testable rather than assumed.
    """
    scores: dict[str, float] = {}
    for rank, cid in enumerate(row["base"]):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    if weight:
        for rank, cid in enumerate(row["rescue"]):
            scores[cid] = scores.get(cid, 0.0) + weight / (RRF_K + rank)
    return [c for c, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def arm_protect(row: dict, keep: int, cap: int = RESCUE_CAP) -> list[str]:
    """Keep the base's top ``keep`` untouched, then interleave rescue-only candidates 1:1 below.

    The middle position the append rationale leaves open: the stated fear is a weak alias hit
    outvoting a strong dense hit, and a hard floor under the top ``keep`` makes that impossible
    by construction while still letting the graph reach a cut it otherwise cannot.
    """
    base = list(row["base"])
    seen = set(base)
    tail = [c for c in row["rescue"] if not (c in seen or seen.add(c))][:cap]
    head, rest = base[:keep], base[keep:]
    woven: list[str] = []
    for left, right in zip(rest, tail):
        woven += [left, right]
    woven += rest[len(tail) :] + tail[len(rest) :]
    return head + woven


def arm_oracle_uncapped(row: dict) -> list[str]:
    """The ceiling with ``rescue_cap`` lifted: bounds what a BIGGER appendix could ever buy.

    ``rescue_cap`` is 50 and the fused rescue lane is ~180 long, so ``arm_oracle`` leaves a door
    open — "you only looked at the first 50". This shuts it: same oracle promotion, whole lane.
    """
    return arm_oracle(row, cap=10**6)


def arm_oracle_base(row: dict) -> list[str]:
    """The ceiling of a PERFECT RERANKER over the base alone — no graph involved at all.

    The comparison the whole experiment turns on. If reordering candidates dense+BM25 already
    found is worth far more than everything the graph lanes can reach, then the retrieval budget
    belongs in a rescore stage, and the graph lanes are being asked to solve a problem they are
    not the bottleneck of.
    """
    gold = set(row["gold"])
    base = list(row["base"])
    hits = [c for c in base if c in gold]
    return hits + [c for c in base if c not in gold]


ARMS = {
    "base": arm_base,
    "append (shipped)": arm_append,
    "protect@5": lambda row: arm_protect(row, 5),
    "protect@10": lambda row: arm_protect(row, 10),
    "fuse w=0.05": lambda row: arm_fuse(row, 0.05),
    "fuse w=0.10": lambda row: arm_fuse(row, 0.10),
    "fuse w=0.25": lambda row: arm_fuse(row, 0.25),
    "fuse w=0.50": lambda row: arm_fuse(row, 0.50),
    "fuse w=1.00": lambda row: arm_fuse(row, 1.00),
    "ORACLE graph (cap 50)": arm_oracle,
    "ORACLE graph (uncapped)": arm_oracle_uncapped,
    "ORACLE rerank base only": arm_oracle_base,
}


# -- scoring -------------------------------------------------------------------------------------


def score(rows: list[dict], arm) -> dict[str, list[float]]:  # noqa: ANN001
    """Per-query metric vectors for one arm — kept per-query so deltas stay paired."""
    out: dict[str, list[float]] = {f"recall@{k}": [] for k in CUTS}
    out["ndcg@10"] = []
    out["ndcg@50"] = []
    for row in rows:
        gold = frozenset(row["gold"])
        ranked = arm(row)
        for k in CUTS:
            out[f"recall@{k}"].append(recall_at_k(ranked, gold, k))
        out["ndcg@10"].append(ndcg_at_k(ranked, gold, 10))
        out["ndcg@50"].append(ndcg_at_k(ranked, gold, 50))
    return out


def bootstrap_ci(deltas: list[float]) -> tuple[float, float]:
    """95% percentile bootstrap interval of the paired mean delta (fixed seed, reproducible)."""
    if not deltas or all(d == 0.0 for d in deltas):
        return (0.0, 0.0)
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(deltas)
    means = sorted(
        stats.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(BOOTSTRAP_ROUNDS)
    )
    lo = means[int(0.025 * BOOTSTRAP_ROUNDS)]
    hi = means[int(0.975 * BOOTSTRAP_ROUNDS) - 1]
    return (lo, hi)


def reachability(rows: list[dict]) -> dict[str, float]:
    """What the graph lanes actually reached — the diagnosis behind whatever the arms report."""
    n = len(rows)
    first_rescued, gold_new, only_trav, only_hyper, gate = [], [], [], [], 0
    base_sizes, gold_in_base, gold_total = [], [], []
    for row in rows:
        base, gold = set(row["base"]), set(row["gold"])
        tail = [c for c in row["rescue"] if c not in base][:RESCUE_CAP]
        first_rescued.append(len(row["base"]))
        new = gold & set(tail)
        gold_new.append(len(new))
        only_trav.append(len(new & set(row["traversal"]) - set(row["hyper"])))
        only_hyper.append(len(new & set(row["hyper"]) - set(row["traversal"])))
        gate += bool(row["gate_fired"])
        base_sizes.append(len(row["base"]))
        gold_in_base.append(len(gold & base))
        gold_total.append(len(gold))
    return {
        "queries": n,
        "gate_hit_rate": round(gate / n, 4),
        "base_size_mean": round(stats.fmean(base_sizes), 1),
        "base_size_min": min(base_sizes),
        "first_rescued_rank_min": min(first_rescued),
        "gold_mean": round(stats.fmean(gold_total), 2),
        "gold_in_base_mean": round(stats.fmean(gold_in_base), 2),
        "gold_rescued_only_mean": round(stats.fmean(gold_new), 4),
        "queries_with_rescued_gold": sum(1 for g in gold_new if g),
        "queries_with_rescued_gold_pct": round(
            100 * sum(1 for g in gold_new if g) / n, 2
        ),
        "rescued_gold_only_traversal": sum(only_trav),
        "rescued_gold_only_hypergraph": sum(only_hyper),
        "retrieve_ms_median": round(stats.median(r["retrieve_ms"] for r in rows), 1),
    }


def report(path: Path) -> str:
    """The full arm table for one trace file, as Markdown."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    reach = reachability(rows)
    scored = {name: score(rows, arm) for name, arm in ARMS.items()}
    control = scored["base"]

    lines = [f"### `{path.name}`", "", "**Reachability**", "", "```json"]
    lines.append(json.dumps(reach, indent=2))
    lines += ["```", ""]
    header = "| arm | " + " | ".join(f"r@{k}" for k in CUTS) + " | nDCG@10 | nDCG@50 | Δr@10 (95% CI) |"
    lines += [header, "|" + "---|" * (len(CUTS) + 4)]
    for name, metrics in scored.items():
        cells = [f"{stats.fmean(metrics[f'recall@{k}']):.4f}" for k in CUTS]
        cells.append(f"{stats.fmean(metrics['ndcg@10']):.4f}")
        cells.append(f"{stats.fmean(metrics['ndcg@50']):.4f}")
        deltas = [a - b for a, b in zip(metrics["recall@10"], control["recall@10"])]
        mean = stats.fmean(deltas)
        lo, hi = bootstrap_ci(deltas)
        cells.append("—" if name == "base" else f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:  # noqa: ANN001
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="*", help="trace files (default: every trace on disk)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    paths = (
        [Path(p) for p in args.traces]
        if args.traces
        else sorted((HERE / "traces").glob("*.jsonl"))
    )
    body = "\n".join(report(p) for p in paths)
    print(body)
    if args.out:
        Path(args.out).write_text(body + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
