# 05 — Grilling: token_cost tokenizer policy

Type: grilling
Status: open
Blocked by: 01

## Question

Spec 1.7 requires token_cost precomputed "with the same tokenizer as the target generation model"; the repo deliberately pins no tokenizer (pack.py licence discipline, ADR-0012). Decide: operator-named tokenizer at `latence process` time? Multiple token_cost columns per tokenizer? Heuristic default + exact opt-in (mirroring HeuristicTokenCounter/TiktokenCounter)? What happens on index/query tokenizer mismatch — recompute, scale, or refuse? Outcome: recorded in the ADR from 03 or its own.
