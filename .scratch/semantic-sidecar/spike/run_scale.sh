#!/usr/bin/env bash
# Part A scale runs (ticket 20). Sequential, so timings are not contended.
set -u
export PYTHONHASHSEED=0
PY=".venv/bin/python -W ignore"
mkdir -p out data
$PY generate.py --entities 4000 --seed 11 --out data/estate_4k.parquet
$PY run_a.py --dataset estate --path data/estate_4k.parquet --out out/estate_4k_with_baseline.json --baseline-max 15000 > out/estate_4k.log 2> out/estate_4k.stderr; echo "estate_4k exit=$?"
$PY run_a.py --dataset historical --path data/historical_50k.parquet --out out/historical_50k.json --baseline-max 0 > out/historical_50k.log 2> out/historical_50k.stderr; echo "historical_50k exit=$?"
$PY generate.py --entities 33000 --seed 13 --out data/estate_100k.parquet
$PY run_a.py --dataset estate --path data/estate_100k.parquet --out out/estate_100k.json --baseline-max 0 > out/estate_100k.log 2> out/estate_100k.stderr; echo "estate_100k exit=$?"
$PY generate.py --entities 330000 --seed 17 --out data/estate_1m.parquet
$PY run_a.py --dataset estate --path data/estate_1m.parquet --out out/estate_1m.json --baseline-max 0 > out/estate_1m.log 2> out/estate_1m.stderr; echo "estate_1m exit=$?"
