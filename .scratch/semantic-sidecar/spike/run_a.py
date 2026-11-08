"""Part A of the ticket-20 spike: structured resolution, end to end, with metrics.

Pipeline (mirrors ../design/resolution-at-scale.md):
  normalise -> multi-pass blocking (DuckDB) -> Splink Fellegi-Sunter scoring of exactly that
  candidate set -> vetoes -> isotonic calibration -> Learn-then-Test bands -> constrained
  clustering, machine-only and machine+expert (a gold-backed simulated reviewer).
Baselines: plain transitive closure at Splink p>=0.95, and the current Latence cascade.

Usage:
  python run_a.py --dataset estate --path data/estate_40k.parquet --out out/estate_40k.json
  python run_a.py --dataset historical --path data/historical_50k.parquet --out out/hist.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
import jellyfish
import numpy as np
import pandas as pd

from er import (
    ConstrainedClusterer,
    fit_isotonic,
    human_minutes,
    learn_then_test_bands,
    pairwise_and_bcubed,
    plain_closure,
    wilson_lower,
)

LEGAL_CANON = {
    "ltd": "ltd", "limited": "ltd", "inc": "inc", "incorporated": "inc", "llc": "llc",
    "l l c": "llc", "corp": "corp", "corporation": "corp", "plc": "plc",
    "public limited company": "plc", "co": "co", "company": "co",
}
EXPAND = {
    "mfg": "manufacturing", "tech": "technologies", "eng": "engineering", "svcs": "services",
    "inds": "industries", "pharma": "pharmaceuticals", "intl": "international",
    "sys": "systems", "consult": "consulting", "hldgs": "holdings",
}
STOP = {"the"}
DUCKDB_MEM = os.environ.get("SPIKE_DUCKDB_MEM", "4GB")  # DuckDB spills to disk above this
BLOCK_CAP = 300  # a blocking key shared by more records than this is skipped (logged)


def _present(v: object) -> bool:
    """True for a real value. Parquet hands missing values back as None OR float NaN, and
    str(NaN) == "nan" would otherwise become a key shared by every record with a gap."""
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).strip() != ""


def _norm(s: str | None) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalise_org(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cores, legals, nums, firsts, acr_keys, acr_firsts = [], [], [], [], [], []
    for raw in out["name"]:
        n = _norm(raw)
        legal = ""
        for lf in sorted(LEGAL_CANON, key=len, reverse=True):
            if n.endswith(" " + lf) or n == lf:
                legal = LEGAL_CANON[lf]
                n = n[: -len(lf)].strip()
                break
        toks = [EXPAND.get(t, t) for t in n.split() if t not in STOP]
        core = " ".join(toks)
        cores.append(core)
        legals.append(legal or None)
        nums.append(" ".join(sorted(t for t in toks if t.isdigit())) or None)
        firsts.append(toks[0] if toks else None)
        words = [t for t in toks if not t.isdigit()]
        acr_keys.append("".join(w[0] for w in words).upper() if len(words) >= 2 else None)
        f = str(raw).split()[0] if str(raw).split() else ""
        acr_firsts.append(f if f.isupper() and 2 <= len(f) <= 5 and f.isalpha() else None)
    out["name_core"] = cores
    out["legal_canon"] = legals
    out["num_tokens"] = nums
    out["first_tok"] = firsts
    out["first_meta"] = [jellyfish.metaphone(f) if f else None for f in firsts]
    out["acr_key"] = acr_keys
    out["acr_first"] = acr_firsts
    out["reg_norm"] = [re.sub(r"\D", "", str(r)) or None if _present(r) else None for r in out["reg_id"]]
    out["pc_out"] = [str(p).split()[0] if _present(p) else None for p in out["postcode"]]
    out["industry"] = [v if _present(v) else None for v in out["industry"]]
    out["city_n"] = [_norm(c) or None for c in out["city"]]
    return out


def normalise_person(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["first_name", "surname", "birth_place", "occupation", "postcode_fake"]:
        out[c] = [_norm(v) or None for v in out[c]]
    def s(v: object) -> str | None:  # missing values arrive as None or NaN
        return v if isinstance(v, str) and v else None

    out["first_meta"] = [jellyfish.metaphone(x) if s(x) else None for x in out["surname"]]
    out["dob_year"] = [x[:4] if s(x) else None for x in out["dob"]]
    out["pc_out"] = [x.split()[0] if s(x) else None for x in out["postcode_fake"]]
    out["initials"] = [
        ((s(f) or " ")[0] + (s(g) or " ")[0]).strip() or None for f, g in zip(out["first_name"], out["surname"])
    ]
    return out


# --------------------------------------------------------------------------------------------
# Blocking
# --------------------------------------------------------------------------------------------

ORG_PASSES: dict[str, list[str]] = {
    "name_core": ["name_core"],
    "first_tok+city": ["first_tok", "city_n"],
    "first_meta+pc_out": ["first_meta", "pc_out"],
    "reg_id": ["reg_norm"],
    "acronym": ["@acr"],  # special: acr_first on one side == acr_key on the other
    "rare_token": ["@rare"],  # special: share a token whose document frequency is low
}
PERSON_PASSES: dict[str, list[str]] = {
    "first+surname": ["first_name", "surname"],
    "surname+dob": ["surname", "dob"],
    "surname_meta+dob_year": ["first_meta", "dob_year"],
    "first+dob": ["first_name", "dob"],
    "initials+pc_out+dob_year": ["initials", "pc_out", "dob_year"],
}


def _pairs_on(con: duckdb.DuckDBPyConnection, keys: list[str]) -> tuple[pd.DataFrame, int]:
    k = " || '|' || ".join(f"CAST({c} AS VARCHAR)" for c in keys)
    nn = " AND ".join(f"{c} IS NOT NULL" for c in keys)
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE kk AS SELECT unique_id, {k} AS bk FROM recs WHERE {nn}"
    )
    skipped = con.execute(
        f"SELECT count(*) FROM (SELECT bk FROM kk GROUP BY bk HAVING count(*) > {BLOCK_CAP})"
    ).fetchone()[0]
    p = con.execute(
        f"""WITH ok AS (SELECT bk FROM kk GROUP BY bk HAVING count(*) BETWEEN 2 AND {BLOCK_CAP})
        SELECT a.unique_id AS l, b.unique_id AS r FROM kk a JOIN kk b USING (bk)
        WHERE a.bk IN (SELECT bk FROM ok) AND a.unique_id < b.unique_id"""
    ).df()
    return p, int(skipped)


def _acronym_pairs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """SELECT least(a.unique_id, b.unique_id) AS l, greatest(a.unique_id, b.unique_id) AS r
        FROM recs a JOIN recs b ON a.acr_first = b.acr_key AND a.unique_id <> b.unique_id
        WHERE a.acr_first IS NOT NULL AND (a.city_n = b.city_n OR a.city_n IS NULL OR b.city_n IS NULL)"""
    ).df()


def _rare_token_pairs(con: duckdb.DuckDBPyConnection, max_df: int = 40) -> pd.DataFrame:
    con.execute(
        "CREATE OR REPLACE TEMP TABLE toks AS SELECT DISTINCT unique_id, unnest(string_split(name_core, ' ')) AS t FROM recs"
    )
    return con.execute(
        f"""WITH rare AS (SELECT t FROM toks WHERE length(t) >= 4 GROUP BY t HAVING count(*) BETWEEN 2 AND {max_df})
        SELECT DISTINCT a.unique_id AS l, b.unique_id AS r FROM toks a JOIN toks b USING (t)
        WHERE a.t IN (SELECT t FROM rare) AND a.unique_id < b.unique_id"""
    ).df()


def block(df: pd.DataFrame, passes: dict[str, list[str]], gold: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    con = duckdb.connect(config={"memory_limit": DUCKDB_MEM, "threads": os.cpu_count() or 4})
    con.register("recs_df", df)
    con.execute("CREATE TABLE recs AS SELECT * FROM recs_df")
    allp: list[pd.DataFrame] = []
    stats: dict[str, Any] = {}
    gold_pairs = sum(n * (n - 1) // 2 for n in pd.Series(list(gold.values())).value_counts())
    for name, keys in passes.items():
        t = time.time()
        if keys == ["@acr"]:
            p, skipped = _acronym_pairs(con), 0
        elif keys == ["@rare"]:
            p, skipped = _rare_token_pairs(con), 0
        else:
            p, skipped = _pairs_on(con, keys)
        p["pass"] = name
        tp = int(sum(gold[a] == gold[b] for a, b in zip(p.l, p.r)))
        stats[name] = {
            "pairs": len(p),
            "pair_completeness": tp / gold_pairs if gold_pairs else 1.0,
            "keys_skipped_over_cap": skipped,
            "secs": round(time.time() - t, 2),
        }
        allp.append(p)
    cand = pd.concat(allp, ignore_index=True)
    passes_of = cand.groupby(["l", "r"])["pass"].agg(lambda s: ",".join(sorted(set(s)))).reset_index()
    n = len(df)
    tp = int(sum(gold[a] == gold[b] for a, b in zip(passes_of.l, passes_of.r)))
    stats["_union"] = {
        "pairs": len(passes_of),
        "pair_completeness": tp / gold_pairs if gold_pairs else 1.0,
        "reduction_ratio": 1 - len(passes_of) / (n * (n - 1) / 2),
        "gold_pairs": int(gold_pairs),
    }
    return passes_of, stats


# --------------------------------------------------------------------------------------------
# Scoring with Splink over exactly our candidate set
# --------------------------------------------------------------------------------------------


def score(df: pd.DataFrame, cand: pd.DataFrame, kind: str) -> np.ndarray:
    import splink.comparison_library as cl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on

    cols = (
        ["unique_id", "name_core", "legal_canon", "city_n", "pc_out", "reg_norm", "industry", "num_tokens", "country"]
        if kind == "org"
        else ["unique_id", "first_name", "surname", "dob", "birth_place", "postcode_fake", "occupation", "gender"]
    )
    os.makedirs("out/duckdb_tmp", exist_ok=True)
    db = DuckDBAPI(duckdb.connect(config={"memory_limit": DUCKDB_MEM, "temp_directory": "out/duckdb_tmp"}))
    sdf = db.register(df[cols].copy())
    if kind == "org":
        comparisons = [
            cl.JaroWinklerAtThresholds("name_core", [0.97, 0.9, 0.8]).configure(term_frequency_adjustments=True),
            cl.ExactMatch("legal_canon"),
            cl.ExactMatch("city_n").configure(term_frequency_adjustments=True),
            cl.ExactMatch("pc_out"),
            cl.ExactMatch("reg_norm"),
            cl.ExactMatch("industry"),
            cl.ExactMatch("num_tokens"),
        ]
        prior_rules = [block_on("reg_norm"), block_on("name_core", "city_n")]
        em_rules = [block_on("pc_out"), block_on("name_core")]
    else:
        comparisons = [
            cl.NameComparison("first_name").configure(term_frequency_adjustments=True),
            cl.NameComparison("surname").configure(term_frequency_adjustments=True),
            cl.DateOfBirthComparison("dob", input_is_string=True),
            cl.ExactMatch("birth_place").configure(term_frequency_adjustments=True),
            cl.ExactMatch("postcode_fake"),
            cl.ExactMatch("occupation"),
            cl.ExactMatch("gender"),
        ]
        prior_rules = [block_on("first_name", "surname", "dob")]
        em_rules = [block_on("first_name", "surname"), block_on("dob")]
    settings = SettingsCreator(
        link_type="dedupe_only", comparisons=comparisons, blocking_rules_to_generate_predictions=prior_rules
    )
    linker = Linker(sdf, settings, log_level=40)
    linker.training.estimate_probability_two_random_records_match(prior_rules, recall=0.6)
    linker.training.estimate_u_using_random_sampling(max_pairs=2e6)
    for r in em_rules:
        linker.training.estimate_parameters_using_expectation_maximisation(r)
    bp = cand[["l", "r"]].rename(columns={"l": "join_key_l", "r": "join_key_r"})
    bp.insert(0, "match_key", "0")
    linker.table_management.register_blocked_pairs_for_predict(db.register(bp))
    pred = linker.inference.predict().as_pandas_dataframe()[["unique_id_l", "unique_id_r", "match_probability"]]
    m = cand.merge(pred, left_on=["l", "r"], right_on=["unique_id_l", "unique_id_r"], how="left")
    return m["match_probability"].fillna(0.0).to_numpy()


def vetoes(df: pd.DataFrame, cand: pd.DataFrame, kind: str) -> np.ndarray:
    """Hard negatives the scorer must never override (ADR-0046's guards, as vetoes)."""
    if kind != "org":
        return np.zeros(len(cand), dtype=bool)
    idx = df.set_index("unique_id")
    num = idx["num_tokens"]
    reg = idx["reg_norm"]
    et = idx["entity_type"]
    nl, nr = num.reindex(cand.l).to_numpy(), num.reindex(cand.r).to_numpy()
    rl, rr = reg.reindex(cand.l).to_numpy(), reg.reindex(cand.r).to_numpy()
    el, er_ = et.reindex(cand.l).to_numpy(), et.reindex(cand.r).to_numpy()
    def conflict(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        # Both sides present AND different. NaN/None means "absent", never "different".
        return np.array([bool(pd.notna(a) and pd.notna(b) and a != b) for a, b in zip(xs, ys)])

    return conflict(nl, nr) | conflict(rl, rr) | conflict(el, er_)


# --------------------------------------------------------------------------------------------
# Baseline: the current Latence cascade
# --------------------------------------------------------------------------------------------


def latence_cascade(df: pd.DataFrame, name_col: str, label_col: str | None, budget_s: float) -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages/latence-core/src"))
    from latence_core.disambiguation import EntityResolver, MentionInput  # noqa: PLC0415

    t = time.time()
    res = EntityResolver()
    for rid, name, lab in zip(df.unique_id, df[name_col], df[label_col] if label_col else ["entity"] * len(df)):
        res.add(MentionInput(mention_id=rid, document_id=rid, text=str(name), label=str(lab), confidence=1.0))
    clusters, _audit = res.resolve()
    secs = time.time() - t
    pred: dict[str, str] = {}
    for i, c in enumerate(clusters):
        for mid in getattr(c, "member_mention_ids", None) or getattr(c, "mention_ids", None) or []:
            pred[mid] = f"c{i}"
    for rid in df.unique_id:
        pred.setdefault(rid, f"single:{rid}")
    return {"pred": pred, "secs": secs, "over_budget": secs > budget_s}


# --------------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------------


def _stage(name: str, t0: float) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    print(f"[stage] {name:16s} t={time.time() - t0:7.1f}s peak_rss={rss} MB", file=sys.stderr, flush=True)


def run(
    dataset: str, path: Path, n_cal: int, n_ltt: int, seed: int, baseline_max: int, reject_target: float = 0.99
) -> dict[str, Any]:
    t0 = time.time()
    raw = pd.read_parquet(path)
    kind = "org" if dataset == "estate" else "person"
    df = normalise_org(raw) if kind == "org" else normalise_person(raw)
    gold = dict(zip(df.unique_id, df.cluster.astype(str)))
    ids = list(df.unique_id)
    timings: dict[str, float] = {"normalise": time.time() - t0}

    t = time.time()
    cand, block_stats = block(df, ORG_PASSES if kind == "org" else PERSON_PASSES, gold)
    timings["block"] = time.time() - t
    _stage("block", t0)

    t = time.time()
    p_raw = score(df, cand, kind)
    timings["score_splink"] = time.time() - t
    _stage("score", t0)
    veto = vetoes(df, cand, kind)
    y = np.array([gold[a] == gold[b] for a, b in zip(cand.l, cand.r)], dtype=int)

    # Labelled samples a reviewer would provide: uniform over candidate pairs, disjoint halves.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cand))
    cal_idx, ltt_idx = order[:n_cal], order[n_cal : n_cal + n_ltt]
    iso = fit_isotonic(p_raw[cal_idx], y[cal_idx])
    p = iso.predict(p_raw)
    p[veto] = 0.0
    bands = learn_then_test_bands(p[ltt_idx], y[ltt_idx], target=0.95, delta=0.05, reject_target=reject_target)
    band = np.array([bands.band(x) for x in p])
    _stage("bands", t0)
    acc, rev, rej = band == "accept", band == "review", band == "reject"

    # Realised quality of the bands on ALL candidate pairs (full gold): does the guarantee hold?
    acc_prec = float(y[acc].mean()) if acc.any() else float("nan")
    rej_match_rate = float(y[rej].mean()) if rej.any() else float("nan")

    # Constrained clustering.
    def cluster(with_expert: bool) -> tuple[dict[str, str], dict[str, int]]:
        cc = ConstrainedClusterer(ids)
        ls, rs = cand.l.to_numpy(), cand.r.to_numpy()
        for i in np.where(veto | (rej & (p < 0.02)))[0]:
            if cc.find(ls[i]) != cc.find(rs[i]):
                cc.add_cannot_link(ls[i], rs[i])
        if with_expert:  # the simulated expert resolves the review band from gold
            for i in np.where(rev & (y == 0))[0]:
                if cc.find(ls[i]) != cc.find(rs[i]):
                    cc.add_cannot_link(ls[i], rs[i])
            for i in np.where(rev & (y == 1))[0]:
                cc.try_union(ls[i], rs[i])
        for i in np.where(acc)[0][np.argsort(-p[acc], kind="stable")]:
            cc.try_union(ls[i], rs[i])
        return cc.clusters(), {"merges": cc.merges, "blocked_merges": cc.blocked_merges}

    t = time.time()
    pred_machine, cs_m = cluster(False)
    pred_expert, cs_e = cluster(True)
    pred_closure = plain_closure(ids, zip(cand.l[p_raw >= 0.95], cand.r[p_raw >= 0.95]))
    timings["cluster"] = time.time() - t
    _stage("cluster", t0)

    # Review effort: pairs, and connected groups the workbench would show as one item.
    rev_groups = len({x for x in plain_closure(
        set(cand.l[rev]) | set(cand.r[rev]), zip(cand.l[rev], cand.r[rev])
    ).values()}) if rev.any() else 0

    result: dict[str, Any] = {
        "dataset": dataset,
        "path": str(path),
        "records": len(df),
        "gold_clusters": len(set(gold.values())),
        "seed": seed,
        "blocking": block_stats,
        "labels_used": {"calibration": int(len(cal_idx)), "ltt": int(len(ltt_idx))},
        "bands": {
            "accept_at": bands.accept_at,
            "reject_below": bands.reject_below,
            "target": bands.target,
            "reject_target": reject_target,
            "delta": bands.delta,
            "share": {"accept": float(acc.mean()), "review": float(rev.mean()), "reject": float(rej.mean())},
            "counts": {"accept": int(acc.sum()), "review": int(rev.sum()), "reject": int(rej.sum())},
            "auto_accept_precision_realised": acc_prec,
            "auto_accept_precision_wilson_lower": wilson_lower(int(y[acc].sum()), int(acc.sum())),
            "auto_reject_true_match_rate": rej_match_rate,
            "auto_resolved_share": float((acc | rej).mean()),
            "vetoed_pairs": int(veto.sum()),
        },
        "review_effort": {
            "review_pairs": int(rev.sum()),
            "review_groups": rev_groups,
            "expert_minutes_at_20s_per_group": human_minutes(rev_groups),
        },
        "clustering": {
            "machine_only": {**pairwise_and_bcubed(pred_machine, gold), **cs_m},
            "machine_plus_expert": {**pairwise_and_bcubed(pred_expert, gold), **cs_e},
            "plain_closure_splink_p95": pairwise_and_bcubed(pred_closure, gold),
        },
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
    }
    if len(df) <= baseline_max:
        b = latence_cascade(df, "name" if kind == "org" else "full_name", "entity_type" if kind == "org" else None, 3600)
        result["clustering"]["latence_cascade_baseline"] = {**pairwise_and_bcubed(b["pred"], gold), "secs": round(b["secs"], 1)}
    else:
        result["clustering"]["latence_cascade_baseline"] = f"skipped: {len(df)} records > baseline_max {baseline_max}"
    result["timings_s"]["total"] = round(time.time() - t0, 1)
    result["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0)
    result["hardware"] = {"cpus": os.cpu_count(), "gpu": None}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["estate", "historical"], required=True)
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-cal", type=int, default=2000)
    ap.add_argument("--n-ltt", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--baseline-max", type=int, default=15_000)
    ap.add_argument("--reject-target", type=float, default=0.99)
    a = ap.parse_args()
    res = run(a.dataset, a.path, a.n_cal, a.n_ltt, a.seed, a.baseline_max, a.reject_target)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("records", "bands", "review_effort", "clustering", "timings_s", "peak_rss_mb")}, indent=1, default=float))


if __name__ == "__main__":
    main()
