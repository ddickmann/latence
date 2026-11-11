# Methods for linking two company registers

Reviewed on 2026-11-11 for one question: the first run on two real company registers accepted
36% of the true links automatically; what raises that share at a certified precision of 0.95,
with a few thousand reviewer labels, open-source code and open weights only?

**How far each statement was checked.** [P] means the primary page was read (a repository
file). [S] means a search excerpt of the primary page only. [U] means not verified. Only one
code host could be opened from the review's environment, so most papers are [S].

**What the spike then built and measured** is in
[`../spike/REPORT.md`](../spike/REPORT.md#a-scorer-fitted-on-the-reviewers-labels): items 2, 3,
4 and 7 below in one piece, which took the share from 36% to 97%. Item 1 was tried as
evidence and did not help. Items 5, 8, 9 and 10 have not been tried.

## Ranked by expected gain for the effort

1. **Link-only matching and one-to-one resolution.** Keep a pair only if it is the best for
   both records, and pass the margin to the next stage. Splink has
   `cluster_using_single_best_links(..., duplicate_free_datasets=[...])` [P, its source and
   change log 4.0.7]. Sadinle 2017 (arXiv 1601.06630) [S]: the unconstrained mixture loses
   precision as the overlap between files falls. Papadakis et al. 2022 (arXiv 2112.14030)
   [S]: eight bipartite algorithms on ten datasets, no universal winner. No source gives a
   recall gain that transfers. **Measured here:** rank and margin as evidence changed
   nothing between the two registers and cost precision on an estate with several records
   per organisation.
2. **Certify on fresh labels drawn where the line will be.** Spertus, Sridhar and Stark 2024
   (arXiv 2409.06680) [S]: valid stratified tests for a bounded mean. SUPG, Kang et al. 2020
   [P for the repository and its Apache-2.0 licence, S for the paper]: for precision targets,
   importance sampling beat uniform sampling in every case tested. Learn-then-Test's fixed
   sequence stays valid when thresholds are searched [S].
3. **Boosted trees on pair evidence.** ING's EntityMatchingModel (MIT) [P]: a supervised
   model over string, rank and legal-form evidence after TF-IDF candidate generation; no
   accuracy figures on its page. Papadakis et al. 2024 (arXiv 2307.01231) [S]: most matching
   benchmarks are easy classification tasks.
4. **Better name comparisons.** Cohen, Ravikumar and Fienberg 2003 [S]: TF-IDF with a
   Jaro-Winkler token match was best on average, and Jaro-Winkler alone is meant for short
   strings such as surnames. Splink's own page says it is not designed for one bag-of-words
   company-name column [S].
5. **Previous and other names.** The company register's free file carries previous names
   [S]; the legal-entity file carries other names. Not tried; the links still missed are
   mostly renamed companies.
6. **Frequency-aware address evidence.** A postcode thousands of companies share should
   weigh little. No published measurement [U]. **Built here** as "how many records share
   this address".
7. **Labels where they decide, for training.** Kasai et al. 2019 (arXiv 1906.08042) [S];
   Meduri et al. 2020 (arXiv 2003.13114) [S]: active selection reaches supervised quality
   with far fewer labels. Labels chosen this way are biased and must never be reused for the
   certifying test.
8. **A small cross-encoder fine-tuned on company pairs,** in place of the zero-shot model.
   Ditto (Apache-2.0) [P for the licence, S for results]. Steiner, Peeters and Bizer 2024
   (arXiv 2409.08185) [S]: fine-tuning helps in-domain and hurts across domains. Candidates
   with a permissive licence according to their model cards [S]: ModernBERT-base,
   bge-reranker-v2-m3, mxbai-rerank-base-v2. **Licence flags:** jina-reranker-v2 and the
   Jellyfish models are non-commercial [S]; AnyMatch has no licence file [P].
9. **Public training data that is not the test set.** The legal-entity file (CC0) names the
   register and the number for every country, so the same join gives true links to other
   registers; the United Kingdom has to be left out. A set of sanctions-list pairs needs a
   paid licence for business use [S].
10. **A larger model on the residual band only.** Too slow for every pair. Licences not
    checked [U].

## Sounds good, and the evidence says no

- **A complete optimal assignment.** McVeigh and Murray (arXiv 1710.10558) [S]: recall 97%
  and precision 63%. A best-for-both rule with a threshold is the safer form.
- **Certification labels in proportion to the score.** SUPG [S]: no better than uniform.
- **A language model on every pair,** or more examples in its prompt [S].
- **Name embeddings off the shelf,** without fine-tuning [S].
- **Jaro-Winkler over whole company names** [S].

## Not found

- A published recall gain for one-to-one resolution, as a number.
- A peer-reviewed benchmark of legal-entity records against a national register.
