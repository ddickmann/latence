# 15 — Task: M4 — knapsack chain-of-evidence selector

Type: task
Status: open
Blocked by: 14

## Question

Implement spec 3.1–3.3: UtilityModel protocol + HeuristicUtility, lazy-greedy marginal loop over v(e|A) with A_E-backed redundancy/connectivity, theta_min stop, and knapsack_refine reusing the existing Packer unchanged on frozen values. Tests per spec M4: budget never exceeded; identical relation signature never selected twice; greedy+refine ≥ greedy on value; determinism.
