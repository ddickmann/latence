# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""M6, enterprise leg (wayfinder ticket 17): supporting-fact recall@budget, CoE vs chunk-topk.

Runs entirely locally over the enterprise-gold export (no GPU, no model): the baseline is
the repo's own chunk-topk path (BM25 rescoring over all chunks → Packer at the budget), the
CoE path is tier-1 zero-model entry (question n-grams against the KV alias table) →
hypergraph signals → CoE selection → chain emission, at the SAME token budget with the SAME
counter (`estimate_tokens` both sides — budget parity is the point).

Metric: a gold supporting-fact span counts as recalled when its (whitespace-normalised) text
is contained in the emitted context. Reported per budget {1k, 2k, 4k, 8k, 16k}, overall and
per hop count, for THREE configs: baseline, CoE-pure (empty context when no anchor resolves
— the honest tier-1 floor), and CoE+fallback (baseline fills the remaining budget after CoE —
the deployable configuration). Gate arithmetic per ticket 08: AUC (mean over the grid) +
both endpoints + the no-point->2%-relative-worse rule.

Tier reporting honesty (ADR-0059): this corpus exercises TIER 1 ONLY (alias entry; no
embedder, no query NER). Tier 0/2 curves belong to the public-benchmark leg on the pod.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from latence_core.chunking import estimate_tokens
from latence_retrieval.bm25 import Bm25Rescorer
from latence_retrieval.chains import assemble_chains
from latence_retrieval.coe import select
from latence_retrieval.contracts import Candidate
from latence_retrieval.hypergraph import HyperedgeKV, hypergraph_signals
from latence_retrieval.pack import Packer

ROOT = Path(__file__).parent
EXPORT = ROOT / "store" / "_latence" / "runs" / "gold-v0" / "export"
BUDGETS = [1000, 2000, 4000, 8000, 16000]

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def load_chunks() -> dict[str, str]:
    chunks: dict[str, str] = {}
    for line in (EXPORT / "records.jsonl").open():
        r = json.loads(line)
        if "content" in r and "index" in r and "document_record_id" in r:
            text = r.get("masked_content") or r["content"]
            if text:
                chunks[r["record_id"]] = text
    return chunks


def question_ngrams(question: str, max_n: int = 4) -> list[str]:
    words = re.findall(r"[\w][\w&.\-]*", question)
    grams: list[str] = []
    for n in range(max_n, 0, -1):  # longest first: most specific alias wins
        grams.extend(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
    return grams


def baseline_context(question: str, chunks: dict[str, str], budget: int) -> str:
    candidates = [
        Candidate(id=cid, text=text, score=0.0) for cid, text in sorted(chunks.items())
    ]
    ranked = Bm25Rescorer().process(question, candidates)
    packed = Packer({"budget": budget}).process(question, ranked)
    return "\n".join(c.text for c in packed)


def coe_context(question: str, kv: HyperedgeKV, budget: int) -> str:
    signals = hypergraph_signals(kv, alias_surfaces=question_ngrams(question))
    if not signals.rho:
        return ""
    selection = select(
        signals,
        budget=budget,
        tau_lookup=kv.tau_text,
        token_counter=estimate_tokens,
        counter_identity="latence:estimate_tokens:v1",
    )
    assembly = assemble_chains(selection, signals, tau_lookup=kv.tau_text)
    return assembly.text


def evaluate(gold: list[dict], contexts: dict[str, str]) -> dict:
    per_hop: dict[int, list[float]] = {}
    span_hits = span_total = 0
    for q in gold:
        ctx = _norm(contexts[q["question_id"]])
        hits = sum(1 for f in q["supporting_facts"] if _norm(f["text"]) in ctx)
        total = len(q["supporting_facts"])
        span_hits += hits
        span_total += total
        per_hop.setdefault(q["hop_count"], []).append(hits / total)
    return {
        "recall": round(span_hits / span_total, 4),
        "per_hop": {
            str(h): round(sum(v) / len(v), 4) for h, v in sorted(per_hop.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "eval-results.json"))
    args = parser.parse_args()

    chunks = load_chunks()
    gold = [json.loads(line) for line in (ROOT / "gold.jsonl").open()]
    print(f"chunks: {len(chunks)}  questions: {len(gold)}  budgets: {BUDGETS}")

    results: dict[str, dict] = {"baseline": {}, "coe_pure": {}, "coe_fallback": {}}
    for budget in BUDGETS:
        base_ctx: dict[str, str] = {}
        pure_ctx: dict[str, str] = {}
        fb_ctx: dict[str, str] = {}
        with HyperedgeKV.open(str(EXPORT / "graph-hyperedges.kv")) as kv:
            for q in gold:
                qid, question = q["question_id"], q["question"]
                base = baseline_context(question, chunks, budget)
                coe = coe_context(question, kv, budget)
                base_ctx[qid] = base
                pure_ctx[qid] = coe
                # deployable config: CoE first, baseline fills the remaining budget
                used = estimate_tokens(coe)
                rest = max(0, budget - used)
                fb_ctx[qid] = (
                    coe + "\n" + baseline_context(question, chunks, rest)
                    if rest > 0
                    else coe
                )
        for name, ctxs in (
            ("baseline", base_ctx),
            ("coe_pure", pure_ctx),
            ("coe_fallback", fb_ctx),
        ):
            results[name][str(budget)] = evaluate(gold, ctxs)
        print(
            f"budget {budget:6d}: baseline={results['baseline'][str(budget)]['recall']:.3f}  "
            f"coe_pure={results['coe_pure'][str(budget)]['recall']:.3f}  "
            f"coe+fallback={results['coe_fallback'][str(budget)]['recall']:.3f}"
        )

    # ticket-08 gate arithmetic: AUC (grid mean) + endpoints + no point >2% relative worse
    def curve(name: str) -> list[float]:
        return [results[name][str(b)]["recall"] for b in BUDGETS]

    base, fb = curve("baseline"), curve("coe_fallback")
    auc_base, auc_fb = sum(base) / len(base), sum(fb) / len(fb)
    endpoints_ok = fb[0] >= base[0] and fb[-1] >= base[-1]
    worst_rel = min(
        (f - b) / b if b > 0 else 0.0 for f, b in zip(fb, base, strict=True)
    )
    gate = {
        "auc_baseline": round(auc_base, 4),
        "auc_coe_fallback": round(auc_fb, 4),
        "auc_dominates": auc_fb > auc_base,
        "endpoints_dominate": endpoints_ok,
        "worst_point_relative": round(worst_rel, 4),
        "no_point_worse_than_2pct": worst_rel >= -0.02,
        "tier": "tier-1 only (alias entry; tier 0/2 on the public leg)",
    }
    gate["PASS"] = bool(
        gate["auc_dominates"] and gate["endpoints_dominate"]
        and gate["no_point_worse_than_2pct"]
    )
    results["gate"] = gate
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("\nGATE:", json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
