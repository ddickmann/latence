# Licences of English entity-linking benchmarks

Reviewed on 2026-11-11 for one question: which benchmark may a commercial open-source project
download and evaluate on, publishing the numbers, without redistributing the data? The
project's rule is permissive by default (ADR-0012): CC0, CC BY, MIT, Apache-2.0, BSD and
public domain pass; non-commercial, research-only and LDC terms do not; share-alike needs the
owner's decision.

**How far each statement was checked.** [P] means the primary page was read (a repository
file or licence text). [S] means a search excerpt of the official page only. [U] means not
verified. Several hosts could not be opened from the review's environment, so [S] and [U]
items have to be re-read before a legal sign-off.

## Result

- **One clean candidate: Mewsli-9, English part.** 12,679 Wikinews articles, 80,242 mentions,
  linked to Wikidata ids [P].
  - Annotations: CC BY 4.0, **by inference**: the repository says "All datasets in this
    repository are released under the CC BY 4.0 International license" [P], but the
    annotation file is hosted outside the repository and its own page names no licence.
  - Text: English Wikinews, CC BY 2.5 (CC BY 4.0 for articles after December 2024) [S].
    Attribution only.
  - Target knowledge base: Wikidata, CC0 [S].
  - Its script rebuilds the text from a 2019 Wikinews dump, so nothing is redistributed.
  - It is not organisation-specific; an organisation subset has to be cut by Wikidata type.
- **Small extra: KORE50.** 50 hand-written sentences, about 144 mentions, CC BY 4.0 according
  to the GERBIL licence list [P]; the authors' own page was not reachable [U].
- **Out:** AIDA-CoNLL (Reuters text is research-and-development only under the NIST
  agreement), ACE2004, AQUAINT and TAC-KBP (LDC), WNED-CWEB (ClueWeb is research-only), the
  N3 collection and Microposts 2013 (CC BY-NC-SA).
- **Share-alike, owner's decision:** ZESHEL (CC BY-SA), WNED-WIKI and other Wikipedia-text
  sets.
- **No organisation-centric entity-linking benchmark with a clear licence was found.**

**Decision (owner, 2026-11-11):** Mewsli-9 is not wired in now. The spike's entity-linking
evidence on real text comes from its own organisation set, built from public company filings
and CC0 register and alias data ([22-real-data-sources.md](22-real-data-sources.md)).
Mewsli-9 English stays the external benchmark to add when numbers comparable with published
work are wanted; its CC BY 4.0 status should be confirmed with its authors first.

## All benchmarks reviewed

| Benchmark | Annotations | Text | Knowledge base | Commercial evaluation |
|---|---|---|---|---|
| AIDA-CoNLL-YAGO | CC BY-SA 3.0 [S] | Reuters RCV1, NIST agreement, research and development only [S] | YAGO2 / Wikipedia | no |
| MSNBC | none listed [P] | news copyright | Wikipedia | unclear |
| AQUAINT | LDC [P] | LDC | Wikipedia | no |
| ACE2004 | LDC [P] | LDC | Wikipedia | no |
| WNED-WIKI | none found [P] | Wikipedia, CC BY-SA | Wikipedia | share-alike, unclear |
| WNED-CWEB | none found | ClueWeb12, research only [S] | Wikipedia | no |
| KILT linking tasks | MIT covers the software [P] | the sources' own terms | Wikipedia snapshot, CC BY-SA | no for the AIDA and ClueWeb parts |
| **Mewsli-9 (English)** | CC BY 4.0 by repository-wide statement [P] | Wikinews, CC BY 2.5 [S] | Wikidata, CC0 | **yes, with attribution** |
| Mewsli-X (English Wikinews test) | as Mewsli-9 | Wikinews; its Wikipedia parts are CC BY-SA | its candidate set uses Wikipedia text | yes for the Wikinews part only |
| ZESHEL | CC BY-SA [P] | Fandom wikis, CC BY-SA | per-world pages, CC BY-SA | share-alike |
| KORE50 | CC BY 4.0 [P, list] | written by its authors | DBpedia / YAGO | yes, with attribution |
| DBpedia Spotlight corpus | CC BY 4.0 [P, list] | newspaper paragraphs | DBpedia | unclear (text) |
| N3 (Reuters-128, RSS-500) | CC BY-NC-SA 4.0 [P] | news | DBpedia | no |
| OKE 2015/2016 | none stated [P] | Wikipedia sentences | DBpedia | unclear |
| Derczynski tweets | CC BY 4.0 [S] | tweets, platform terms | DBpedia | annotations yes; text unclear |
| TweekiGold | data licence not found [U] | tweets | Wikidata | unclear |
| Microposts NEEL | 2013 CC BY-NC-SA; 2015–16 CC BY 4.0 [P, list] | tweets | DBpedia | 2013 no; later text unclear |
| TAC-KBP EDL | LDC [S] | LDC | LDC | no |
| Wikilinks | CC BY 3.0 for the extended set [S] | web pages, not distributed | Wikipedia | unclear |
| BLINK data | code MIT [P] | Wikipedia, CC BY-SA | Wikipedia | share-alike |
| AmbER sets | CC0 [P] | generated from Wikidata | Wikidata + Wikipedia | yes, but it tests retrieval, not linking in documents |
| Linked-DocRED | GPLv3 [P] | Wikipedia | Wikipedia | unclear |
| DWIE | repository GPL-3.0 [P] | broadcaster's articles, its copyright | Wikipedia | unclear |
| MedMentions | CC0 [P] | PubMed abstracts | UMLS, needs its own licence [U] | unclear, and not about organisations |

## Two general points

- **Evaluating on share-alike text does not, on the licence text, bind the software or the
  report.** CC BY-SA applies its condition when adapted material is shared; metrics and code
  are not adapted material. Quoting example sentences needs attribution; shipping converted
  benchmark files or a Wikipedia-derived knowledge base would make those files share-alike
  [P for the licence text, S for the publisher's guidance]. This is a reading, not legal
  advice.
- **Wikidata is CC0; Wikipedia text is CC BY-SA 4.0.** Using Wikidata as the target and
  mapping Wikipedia titles to Wikidata ids avoids depending on Wikipedia text.

## Open points for the owner or a lawyer

1. Whether the repository-wide CC BY 4.0 statement covers the Mewsli annotation file.
2. Whether the share-alike tier is acceptable under the reading above.
3. No recommended set is used for training until decided.

## Sources

- Mewsli-9: <https://github.com/google-research/google-research/blob/master/dense_representations_for_entity_retrieval/mel/mewsli-9.md>
- Repository licence statement: <https://github.com/google-research/google-research/blob/master/README.md>
- GERBIL licence list: <https://github.com/dice-group/gerbil/wiki/Licenses-for-datasets>
- NIST Reuters corpora: <https://trec.nist.gov/data/reuters/reuters.html>
- KILT licence: <https://github.com/facebookresearch/KILT/blob/main/LICENSE>
- ZESHEL: <https://github.com/lajanugen/zeshel>
- N3 collection: <https://github.com/dice-group/n3-collection>
- AmbER sets: <https://github.com/anthonywchen/AmbER-Sets>
- CC BY-SA 4.0 legal code: <https://creativecommons.org/licenses/by-sa/4.0/legalcode>
- Wikidata licensing: <https://www.wikidata.org/wiki/Wikidata:Licensing>
