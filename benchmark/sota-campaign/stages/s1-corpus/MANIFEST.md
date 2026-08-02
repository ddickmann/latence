# Corpus manifest

**106 files · 7.8 GB · 12 distinct extensions**

## Sources

| Source | Files | GB | Licence | Publishable |
|---|---:|---:|---|---|
| musique | 16 | 0.37 | CC-BY-4.0 | yes |
| hotpotqa | 12 | 1.25 | CC-BY-SA-4.0 | **no** |
| wiki2multihop | 15 | 0.40 | Apache-2.0 (dataset card); paragraph text CC-BY-SA Wikipedia-derived | **no** |
| multihop_rag | 12 | 0.01 | ODC-BY-1.0 (annotations); news article text under publisher copyright | **no** |
| ohr_bench | 15 | 1.71 | research-only (non-commercial) | **no** |
| vidoseek | 12 | 0.76 | Apache-2.0 (annotations); source-PDF copyright retained upstream | **no** |
| uda | 24 | 3.32 | CC-BY-SA-4.0 | **no** |

## Measured file-type histogram (top 15, by claimed extension)

| Extension | Files | Share |
|---|---:|---:|
| `lock` | 21 | 19.8% |
| `metadata` | 21 | 19.8% |
| `parquet` | 15 | 14.2% |
| `json` | 12 | 11.3% |
| `jsonl` | 10 | 9.4% |
| `(none)` | 5 | 4.7% |
| `md` | 5 | 4.7% |
| `tag` | 5 | 4.7% |
| `kv` | 4 | 3.8% |
| `zip` | 4 | 3.8% |
| `graphml` | 2 | 1.9% |
| `ttl` | 2 | 1.9% |

## Office formats — reported split, deliberately

* legacy OLE (doc/xls/ppt): **0**
* modern OOXML (docx/xlsx/pptx): **0**
* OOXML share of office files: **0.00%**

## Limits this corpus cannot support a claim about

* Modern OOXML is UNTESTED AT VOLUME for: docx (0), xlsx (0), pptx (0) — against 0 legacy OLE (doc/xls/ppt) from the 2009 GovDocs1 harvest. OOXML is a zipped-XML container parsed by a different code path than OLE, so legacy coverage says nothing about it. Do not let 'many file types' imply modern-format coverage for any format listed here.
* Not publishable, fair-use/access-only posture: hotpotqa, wiki2multihop, multihop_rag, ohr_bench, vidoseek, uda. Nothing derived from these may appear in a shareable artifact.
