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
from pathlib import Path
from typing import Any

import duckdb
import jellyfish
import numpy as np
import pandas as pd

from er import (
    ArrayClusterer,
    NeighbourTable,
    fit_isotonic,
    human_minutes,
    learn_then_test_bands,
    pairwise_and_bcubed,
    wilson_lower,
)
from rescore import ORG_TEXT_COLS, PERSON_TEXT_COLS, record_text, stage_two, uncertain_region

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
TRAIN_MAX = 150_000  # EM trains on at most this many records (seeded sample)
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
    "surname+birth_place": ["surname", "birth_place"],
    "postcode": ["postcode_fake"],
}


def _pairs_on(con: duckdb.DuckDBPyConnection, keys: list[str]) -> tuple[str, int]:
    """SQL selecting (l, r) for records sharing every key in ``keys``; keys over BLOCK_CAP skipped."""
    k = " || '|' || ".join(f"CAST({c} AS VARCHAR)" for c in keys)
    nn = " AND ".join(f"{c} IS NOT NULL" for c in keys)
    con.execute(f"CREATE OR REPLACE TEMP TABLE kk AS SELECT rid, {k} AS bk FROM recs WHERE {nn}")
    skipped = con.execute(
        f"SELECT count(*) FROM (SELECT bk FROM kk GROUP BY bk HAVING count(*) > {BLOCK_CAP})"
    ).fetchall()[0][0]
    sql = f"""WITH ok AS (SELECT bk FROM kk GROUP BY bk HAVING count(*) BETWEEN 2 AND {BLOCK_CAP}),
        kok AS (SELECT rid, bk FROM kk WHERE bk IN (SELECT bk FROM ok))
        SELECT a.rid AS l, b.rid AS r FROM kok a JOIN kok b USING (bk) WHERE a.rid < b.rid"""
    return sql, int(skipped)


def _acronym_pairs(con: duckdb.DuckDBPyConnection) -> tuple[str, int]:
    """An acronym surface ("NL") against records whose name's initials are that acronym, same city
    or unknown city. Capped like every other pass: an initials key shared by more than BLOCK_CAP
    records is skipped (at 1e6 records an uncapped two-letter key is a quadratic blow-up)."""
    skipped = con.execute(
        f"SELECT count(*) FROM (SELECT acr_key FROM recs WHERE acr_key IS NOT NULL GROUP BY acr_key HAVING count(*) > {BLOCK_CAP})"
    ).fetchall()[0][0]
    sql = f"""WITH okk AS (SELECT acr_key FROM recs WHERE acr_key IS NOT NULL GROUP BY acr_key HAVING count(*) <= {BLOCK_CAP})
        SELECT least(a.rid, b.rid) AS l, greatest(a.rid, b.rid) AS r
        FROM recs a JOIN recs b ON a.acr_first = b.acr_key AND a.rid <> b.rid
        WHERE a.acr_first IS NOT NULL AND b.acr_key IN (SELECT acr_key FROM okk)
          AND (a.city_n = b.city_n OR a.city_n IS NULL OR b.city_n IS NULL)"""
    return sql, int(skipped)


def _rare_token_pairs(con: duckdb.DuckDBPyConnection, max_df: int = 40) -> tuple[str, int]:
    con.execute(
        "CREATE OR REPLACE TEMP TABLE toks AS SELECT DISTINCT rid, unnest(string_split(name_core, ' ')) AS t FROM recs"
    )
    sql = f"""WITH rare AS (SELECT t FROM toks WHERE length(t) >= 4 GROUP BY t HAVING count(*) BETWEEN 2 AND {max_df}),
        tr AS (SELECT rid, t FROM toks WHERE t IN (SELECT t FROM rare))
        SELECT a.rid AS l, b.rid AS r FROM tr a JOIN tr b USING (t) WHERE a.rid < b.rid"""
    return sql, 0


def record_numbers(unique_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Number the records 0..n-1 in the order of their ids.

    Returns ``(rid, ids)``: ``rid[row]`` is the number of the record in that row, and ``ids[k]``
    is the id of record ``k``. Numbering in id order keeps every comparison the pair SQL makes
    (``l < r``, ``ORDER BY l, r``) the same as it was on the ids themselves, so the candidate
    pairs and their order do not change; they only stop being strings.
    """
    order = np.argsort(unique_ids.astype(str), kind="stable")
    rid = np.empty(len(order), dtype=np.int32)
    rid[order] = np.arange(len(order), dtype=np.int32)
    return rid, unique_ids[order]


def block(df: pd.DataFrame, passes: dict[str, list[str]]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run every blocking pass, then union them. Returns the candidate pairs as two integer
    arrays ``(l, r)`` of record numbers (``df.rid``), sorted, each pair once with ``l < r``.

    Counting and the union happen in DuckDB, and only the two integer columns come back: at 1e6
    records the candidate set is about 3 x 10^7 pairs, which as a frame of string ids took 12 GB.
    """
    os.makedirs("out/duckdb_tmp", exist_ok=True)
    con = duckdb.connect(config={"memory_limit": DUCKDB_MEM, "threads": os.cpu_count() or 4, "temp_directory": "out/duckdb_tmp"})
    con.register("recs_df", df)
    con.execute("CREATE TABLE recs AS SELECT * FROM recs_df")
    con.execute("CREATE TABLE cand_all (l INTEGER, r INTEGER)")
    stats: dict[str, Any] = {}
    gold_pairs = int(
        con.execute("SELECT coalesce(sum(n * (n - 1) / 2), 0) FROM (SELECT count(*) n FROM recs GROUP BY cluster)").fetchall()[0][0]
    )

    def completeness(table: str) -> int:
        return int(
            con.execute(
                f"""SELECT count(*) FROM {table} c JOIN recs a ON c.l = a.rid JOIN recs b ON c.r = b.rid
                WHERE a.cluster = b.cluster"""
            ).fetchall()[0][0]
        )

    for name, keys in passes.items():
        t = time.time()
        if keys == ["@acr"]:
            sql, skipped = _acronym_pairs(con)
        elif keys == ["@rare"]:
            sql, skipped = _rare_token_pairs(con)
        else:
            sql, skipped = _pairs_on(con, keys)
        con.execute(f"CREATE OR REPLACE TEMP TABLE pass_t AS SELECT DISTINCT l, r FROM ({sql})")
        con.execute("INSERT INTO cand_all SELECT l, r FROM pass_t")
        stats[name] = {
            "pairs": int(con.execute("SELECT count(*) FROM pass_t").fetchall()[0][0]),
            "pair_completeness": completeness("pass_t") / gold_pairs if gold_pairs else 1.0,
            "keys_skipped_over_cap": skipped,
            "secs": round(time.time() - t, 2),
        }
    con.execute("CREATE TABLE cand AS SELECT DISTINCT l, r FROM cand_all")
    pairs = con.execute("SELECT l, r FROM cand ORDER BY l, r").fetchnumpy()
    left, right = pairs["l"].astype(np.int32), pairs["r"].astype(np.int32)
    n = len(df)
    stats["_union"] = {
        "pairs": len(left),
        "pair_completeness": completeness("cand") / gold_pairs if gold_pairs else 1.0,
        "reduction_ratio": 1 - len(left) / (n * (n - 1) / 2),
        "gold_pairs": gold_pairs,
    }
    con.close()
    return left, right, stats


# --------------------------------------------------------------------------------------------
# Scoring with Splink over exactly our candidate set
# --------------------------------------------------------------------------------------------


def score(
    df: pd.DataFrame,
    left: np.ndarray,
    right: np.ndarray,
    kind: str,
    detail: dict[str, Any] | None = None,
    ids: np.ndarray | None = None,
) -> np.ndarray:
    """Splink match probability per candidate pair, in the order of ``left`` / ``right``.

    Splink is given the record number as its ``unique_id``, the prediction table stays in DuckDB
    (which spills to disk above its memory limit), and only the probability column is read back.
    Pass a dict as ``detail`` to also get the per-field evidence a reviewer is shown: it receives
    the trained ``model`` and the full ``pred`` frame (gamma and Bayes-factor columns per
    comparison) with its two id columns translated back through ``ids``. That frame is only
    affordable on a small estate.
    """
    import splink.comparison_library as cl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on

    cols = (
        ["unique_id", "name_core", "legal_canon", "city_n", "pc_out", "reg_norm", "industry", "num_tokens", "country"]
        if kind == "org"
        else ["unique_id", "first_name", "surname", "dob", "birth_place", "postcode_fake", "occupation", "gender"]
    )
    os.makedirs("out/duckdb_tmp", exist_ok=True)
    con = duckdb.connect(config={"memory_limit": DUCKDB_MEM, "temp_directory": "out/duckdb_tmp"})
    db = DuckDBAPI(con)

    def numbered(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[cols].copy()
        out["unique_id"] = frame["rid"].to_numpy()
        return out

    sdf = db.register(numbered(df))
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
    # Training keeps the records' own ids: it runs on at most TRAIN_MAX records, and Splink's
    # sampling during training depends on the id values, so numbering them would train a
    # slightly different model.
    train_df = df[cols]
    if len(df) > TRAIN_MAX:  # EM over a seeded sample (usual practice); scoring uses every record
        train_df = df[cols].sample(n=TRAIN_MAX, random_state=0)
    tdb = DuckDBAPI(duckdb.connect(config={"memory_limit": DUCKDB_MEM, "temp_directory": "out/duckdb_tmp"}))
    trainer = Linker(tdb.register(train_df.copy()), settings, log_level=40)
    trainer.training.estimate_probability_two_random_records_match(prior_rules, recall=0.6)
    trainer.training.estimate_u_using_random_sampling(max_pairs=2e6)
    for r in em_rules:
        trainer.training.estimate_parameters_using_expectation_maximisation(r)
    model = trainer.misc.save_model_to_json()
    if detail is not None:  # keeps gamma_* and bf_* per comparison; probabilities are unchanged
        model["retain_intermediate_calculation_columns"] = True
    else:  # the prediction table carries ids and scores only, not a copy of every compared value
        model["retain_matching_columns"] = False
    linker = Linker(sdf, model, log_level=40)
    con.register("cand_arrays", pd.DataFrame({"i": np.arange(len(left), dtype=np.int64), "l": left, "r": right}))
    con.execute("CREATE TABLE cand_i AS SELECT i, l, r FROM cand_arrays")
    con.unregister("cand_arrays")
    blocked = con.sql("SELECT '0' AS match_key, l AS join_key_l, r AS join_key_r FROM cand_i")
    linker.table_management.register_blocked_pairs_for_predict(db.register(blocked))
    pred = linker.inference.predict()
    if detail is not None:
        full = pred.as_pandas_dataframe()
        if ids is not None:
            for side in ("unique_id_l", "unique_id_r"):
                full[side] = ids[full[side].to_numpy()]
        detail["model"], detail["pred"] = model, full
    scored = con.execute(
        f"""SELECT coalesce(p.match_probability, 0.0) AS mp
        FROM cand_i c LEFT JOIN {pred.physical_name} p ON p.unique_id_l = c.l AND p.unique_id_r = c.r
        ORDER BY c.i"""
    ).fetchnumpy()["mp"]
    con.close()
    return np.asarray(scored, dtype=np.float64)


def _codes(values: np.ndarray) -> np.ndarray:
    """Each value as a small integer, -1 where it is absent."""
    codes, _ = pd.factorize(pd.Series(values, dtype=object), use_na_sentinel=True)
    return codes.astype(np.int32)


def vetoes(by_record: pd.DataFrame, left: np.ndarray, right: np.ndarray, kind: str) -> np.ndarray:
    """Hard negatives the scorer must never override (ADR-0046's guards, as vetoes).

    ``by_record`` is the normalised frame in record-number order, so row ``k`` is record ``k``.
    A veto is "both sides present and different"; an absent value is never "different".
    """
    if kind != "org":
        return np.zeros(len(left), dtype=bool)
    out = np.zeros(len(left), dtype=bool)
    for col in ("num_tokens", "reg_norm", "entity_type"):
        c = _codes(by_record[col].to_numpy())
        cl, cr = c[left], c[right]
        out |= (cl >= 0) & (cr >= 0) & (cl != cr)
    return out


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


def _gpu_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        return None


def _stage(name: str, t0: float) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    print(f"[stage] {name:16s} t={time.time() - t0:7.1f}s peak_rss={rss} MB", file=sys.stderr, flush=True)


def _load_scorers(names: list[str], question: str, instruction: str) -> dict[str, Any]:
    """The neural pair scorers of the agreed model set, loaded only when an arm asks for them."""
    import models as M

    make = {
        "rerank": lambda: M.Reranker(instruction),
        "decide": lambda: M.GlinerDecide(question),
        "glide": lambda: M.GlideDecide(question),
    }
    return {n: make[n]() for n in names}


def band_masks(p: np.ndarray, bands: Any) -> tuple[np.ndarray, np.ndarray]:
    """``Bands.band`` for a whole array: (accepted, rejected). What is neither is in review."""
    acc = np.zeros(len(p), dtype=bool) if bands.accept_at is None else p >= bands.accept_at
    rej = np.zeros(len(p), dtype=bool) if bands.reject_below is None else (p < bands.reject_below) & ~acc
    return acc, rej


def components(n: int, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Plain transitive closure over the given pairs: each record's component, named by its
    smallest member. What connected components at a threshold does, with no constraints."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix((np.ones(len(left), dtype=np.int8), (left, right)), shape=(n, n))
    _count, comp = connected_components(graph, directed=False)
    first = np.full(comp.max() + 1 if n else 0, n, dtype=np.int64)
    np.minimum.at(first, comp, np.arange(n))
    return first[comp].astype(np.int32)


def run(
    dataset: str,
    path: Path,
    n_cal: int,
    n_ltt: int,
    seed: int,
    baseline_max: int,
    reject_target: float = 0.99,
    rescore: list[str] | None = None,
    n_cal2: int = 1000,
    n_ltt2: int = 1500,
    scorers: dict[str, Any] | None = None,
    export_review: Path | None = None,
    export_groups: int = 150,
    export_graph: Path | None = None,
    export_graph_entities: int = 400,
    dump: Path | None = None,
) -> dict[str, Any]:
    """``rescore`` names the pair scorers ("rerank", "decide", "glide") applied to the pairs the
    first stage leaves uncertain; ``scorers`` injects ready objects (tests use fakes)."""
    rescore = list(rescore or [])
    delta = 0.05 / 2 if rescore else 0.05  # two accept claims share the 0.05 by the union bound
    t0 = time.time()
    raw = pd.read_parquet(path)
    kind = "org" if dataset == "estate" else "person"
    df = normalise_org(raw) if kind == "org" else normalise_person(raw)
    # Records are numbered in id order; pairs, labels and clusters are arrays over those numbers.
    # `ids[k]` is the id of record k. Ids come back only where a person or a file reads them.
    rid, ids = record_numbers(df.unique_id.to_numpy())
    df["rid"] = rid
    n = len(df)
    by_record = df.iloc[np.argsort(rid, kind="stable")]
    gold_codes = _codes(by_record.cluster.astype(str).to_numpy())
    gold = dict(zip(ids.tolist(), by_record.cluster.astype(str).tolist()))
    timings: dict[str, float] = {"normalise": time.time() - t0}

    t = time.time()
    left, right, block_stats = block(df, ORG_PASSES if kind == "org" else PERSON_PASSES)
    n_pairs = len(left)
    timings["block"] = time.time() - t
    _stage("block", t0)

    t = time.time()
    detail: dict[str, Any] | None = {} if export_review else None
    p_raw = score(df, left, right, kind, detail, ids)
    timings["score_splink"] = time.time() - t
    _stage("score", t0)
    veto = vetoes(by_record, left, right, kind)
    y = (gold_codes[left] == gold_codes[right]).astype(int)

    # Labelled samples a reviewer would provide: uniform over candidate pairs, disjoint halves.
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_pairs)
    cal_idx, ltt_idx = order[:n_cal], order[n_cal : n_cal + n_ltt]
    del order
    iso = fit_isotonic(p_raw[cal_idx], y[cal_idx])
    p = iso.predict(p_raw)
    p[veto] = 0.0
    bands = learn_then_test_bands(p[ltt_idx], y[ltt_idx], target=0.95, delta=delta, reject_target=reject_target)
    acc, rej = band_masks(p, bands)
    _stage("bands", t0)

    stage2_info: dict[str, Any] | None = None
    if rescore:
        t = time.time()
        unc = np.where(uncertain_region(p, bands, veto))[0]
        cols = ORG_TEXT_COLS if kind == "org" else PERSON_TEXT_COLS
        text_of = {r["unique_id"]: record_text(r, cols) for r in raw.to_dict("records")}
        pairs = [(text_of[a_], text_of[b_]) for a_, b_ in zip(ids[left[unc]].tolist(), ids[right[unc]].tolist())]
        if scorers is None:
            import models as M

            scorers = _load_scorers(rescore, M.SAME_QUESTION, M.SAME_RECORD_INSTRUCTION)
        sc_times: dict[str, float] = {}
        sc: dict[str, np.ndarray] = {}
        for name in rescore:
            ts = time.time()
            sc[name] = np.asarray(scorers[name].score(pairs), dtype=float)
            sc_times[name] = round(time.time() - ts, 2)
        # A reviewer labels a sample of the uncertain region only: disjoint halves again.
        order2 = np.random.default_rng(seed + 1).permutation(len(unc))
        cal2, ltt2 = order2[:n_cal2], order2[n_cal2 : n_cal2 + n_ltt2]
        s2 = stage_two(p[unc], sc, y[unc], cal2, ltt2, target=0.95, delta=delta, reject_target=reject_target)
        acc2, rej2 = band_masks(s2.p2, s2.bands)
        acc[unc], rej[unc] = acc2, rej2
        p[unc] = s2.p2  # the cluster-level check reads the better-informed probability
        timings["rescore"] = time.time() - t
        _stage("rescore", t0)
        stage2_info = {
            "scorers": rescore,
            "uncertain_pairs": int(len(unc)),
            "labels_used": {"calibration": s2.n_cal, "ltt": s2.n_ltt},
            "features": dict(zip(s2.feature_names, s2.coefficients)),
            "accept_at": s2.bands.accept_at,
            "reject_below": s2.bands.reject_below,
            "delta_per_stage": delta,
            "moved": {"accept": int(acc2.sum()), "review": int((~acc2 & ~rej2).sum()), "reject": int(rej2.sum())},
            "accept_precision_realised": float(y[unc][acc2].mean()) if acc2.any() else float("nan"),
            "reject_true_match_rate": float(y[unc][rej2].mean()) if rej2.any() else float("nan"),
            "scorer_secs": sc_times,
            "pairs_per_s": {n: (len(unc) / v if v > 0 else float("inf")) for n, v in sc_times.items()},
        }
    rev = ~acc & ~rej

    # Realised quality of the bands on ALL candidate pairs (full gold): does the guarantee hold?
    acc_prec = float(y[acc].mean()) if acc.any() else float("nan")
    rej_match_rate = float(y[rej].mean()) if rej.any() else float("nan")

    # Scored neighbours of every record: the evidence the cluster-level check reads. A merge
    # must never cross a veto or a strong reject; with the simulated expert, nor a pair the
    # expert called different.
    never = veto | (rej & (p < 0.02))
    table = NeighbourTable(n, left, right, p, {"machine": never, "expert": never | (rev & (y == 0))})
    accepted = np.where(acc)[0]
    accepted = accepted[np.argsort(-p[accepted], kind="stable")]  # first- and second-stage accepts

    # Constrained clustering.
    def cluster(with_expert: bool) -> tuple[np.ndarray, dict[str, int]]:
        cc = ArrayClusterer(n, table, "expert" if with_expert else "machine")
        if with_expert:  # the simulated expert resolves the review band from gold
            for i in np.where(rev & (y == 1))[0]:
                cc.try_union(int(left[i]), int(right[i]))
        for i in accepted:
            cc.try_union_checked(int(left[i]), int(right[i]))
        return cc.label.copy(), {
            "merges": cc.merges,
            "blocked_merges": cc.blocked_merges,
            "blocked_by_linkage": cc.blocked_by_linkage,
        }

    def as_clusters(labels: np.ndarray) -> dict[str, str]:
        """record id -> cluster name, the form the metrics and the exports read."""
        return dict(zip(ids.tolist(), ids[labels].tolist()))

    t = time.time()
    labels_machine, cs_m = cluster(False)
    labels_expert, cs_e = cluster(True)
    pred_machine, pred_expert = as_clusters(labels_machine), as_clusters(labels_expert)
    confident = p_raw >= 0.95
    pred_closure = as_clusters(components(n, left[confident], right[confident]))
    timings["cluster"] = time.time() - t
    _stage("cluster", t0)

    # Review effort: pairs, and connected groups the workbench would show as one item.
    rev_groups = int(len(np.unique(components(n, left[rev], right[rev])[left[rev]]))) if rev.any() else 0

    band = None
    if export_review is not None or dump is not None:  # names of bands and ids, for people and files
        band = np.where(acc, "accept", np.where(rej, "reject", "review"))
        cand = pd.DataFrame({"l": ids[left], "r": ids[right]})

    if export_review is not None and detail is not None and band is not None:
        from review_export import build_review_export

        export = build_review_export(
            raw=raw, cand=cand, p=p, p_raw=p_raw, band=band, y=y, veto=veto, detail=detail,
            accept_at=bands.accept_at, reject_below=bands.reject_below, kind=kind,
            max_groups=export_groups, seed=seed, source=str(path),
        )
        export_review.parent.mkdir(parents=True, exist_ok=True)
        export_review.write_text(json.dumps(export, indent=1, default=str))
        if export_graph is not None and kind == "org":
            from generate import INDUSTRY
            from graph_export import build_graph_export

            graph = build_graph_export(
                raw=raw, clusters=pred_machine, cand=cand, p=p, band=band,
                focus_records={r for g in export["groups"] for r in g["records"]},
                review_item_ids={it["id"] for it in export["items"]},
                industry_names={code: word for word, code in INDUSTRY},
                max_entities=export_graph_entities, source=str(path),
            )
            export_graph.parent.mkdir(parents=True, exist_ok=True)
            export_graph.write_text(json.dumps(graph, indent=1, default=str))

    if dump is not None and band is not None:
        # Everything a second implementation has to reproduce: each pair's probability and
        # band, and both cluster assignments. Compared by equivalence.py.
        dump.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dump,
            l=cand.l.to_numpy().astype(str), r=cand.r.to_numpy().astype(str),
            p_raw=p_raw, p=p, band=band.astype(str), veto=veto,
            ids=ids.astype(str),
            machine=ids[labels_machine].astype(str),
            expert=ids[labels_expert].astype(str),
            accept_at=np.array([np.nan if bands.accept_at is None else bands.accept_at]),
            reject_below=np.array([np.nan if bands.reject_below is None else bands.reject_below]),
        )

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
            "second_stage": stage2_info,
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
    result["hardware"] = {"cpus": os.cpu_count(), "gpu": _gpu_name()}
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
    ap.add_argument("--rescore", default="", help="comma-separated pair scorers for the uncertain pairs: rerank,decide,glide")
    ap.add_argument("--n-cal2", type=int, default=1000)
    ap.add_argument("--n-ltt2", type=int, default=1500)
    ap.add_argument("--export-review", type=Path, default=None, help="write the review-band items for the workbench")
    ap.add_argument("--export-groups", type=int, default=150, help="review groups to export, highest priority first")
    ap.add_argument("--dump", type=Path, default=None, help="write every pair's probability and band and both cluster assignments (.npz), for equivalence.py")
    ap.add_argument("--export-graph", type=Path, default=None, help="write the resolved graph for the workbench (needs --export-review)")
    ap.add_argument("--export-graph-entities", type=int, default=400, help="organisations to carry in full in the graph export")
    a = ap.parse_args()
    res = run(
        a.dataset, a.path, a.n_cal, a.n_ltt, a.seed, a.baseline_max, a.reject_target,
        rescore=[x for x in a.rescore.split(",") if x], n_cal2=a.n_cal2, n_ltt2=a.n_ltt2,
        export_review=a.export_review, export_groups=a.export_groups,
        export_graph=a.export_graph, export_graph_entities=a.export_graph_entities, dump=a.dump,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ("records", "bands", "review_effort", "clustering", "timings_s", "peak_rss_mb")}, indent=1, default=float))


if __name__ == "__main__":
    main()
