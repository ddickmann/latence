# Real public data for evaluating organisation matching and linking

Reviewed on 2026-11-11. The question: which real (non-synthetic) data may a commercial
open-source project download and evaluate on? The data is fetched by a script where the
evaluation runs and is never committed; only aggregate results are published.

**How far each statement was checked.** *Read* means the publisher's own page was fetched on
2026-11-11 and the quoted words are from it. [S] means a search excerpt of the official page
only. [U] means not verified.

## Chosen (owner, 2026-11-11)

### 1. GLEIF legal-entity records joined to the UK company register

- **GLEIF golden copy.** 3,490,817 legal-entity records on 2026-11-11: legal name, other and
  transliterated names, legal and headquarters address, legal form, the register an entity is
  entered in and its number there. Bulk CSV (485 MB zipped) from the golden-copy service.
  - **Licence, read:** "The data available through the Access Service are provided under the
    CC0 licence, see CC0 1.0 Universal (CC0 1.0)."
    (<https://www.gleif.org/en/meta/lei-data-terms-of-use>)
- **Companies House free company data product.** 5,711,779 live companies in the file of
  2026-11-01: name, number, registered office, category, status, industry codes, previous
  names. One CSV (495 MB zipped).
  - **Terms, read:** the download page says "This snapshot is provided free of charge and will
    not be supported." and names no licence for the file. The register's guidance pages on
    GOV.UK carry "All content is available under the Open Government Licence v3.0, except
    where otherwise stated", and that licence allows to "exploit the Information commercially
    and non-commercially" with attribution.
    (<https://download.companieshouse.gov.uk/en_output.html>,
    <https://www.gov.uk/guidance/companies-house-data-products>,
    <https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/>)
  - **Status:** no page was found that puts the data file itself under a named licence. The
    use here is evaluation with aggregate results only, no redistribution. **Open point for
    the owner:** accept that, or ask the register.
- **Why the pair is ground truth.** GLEIF lists the UK register under the codes RA000585
  (England and Wales), RA000586 (Northern Ireland) and RA000587 (Scotland) (read:
  <https://www.gleif.org/en/lei-data/lei-linking-to-registration-authorities>). A GLEIF record
  with one of those codes carries the company number; the same number in the register's file
  is the same legal entity, recorded by a different organisation.
- **What was built** (`spike/realdata/gleif_ch.py`): 120,192 UK GLEIF records, of which
  102,732 link to a live company; plus one million other companies as distractors. The number
  is withheld from the matcher. Sole proprietors are left out; no officer or owner data is
  read. In 95.8% of the linked pairs the two names are the same apart from case.
- **Privacy:** registered offices of very small companies can be home addresses, and a
  company can be named after a person. Nothing but aggregates leaves the machine that runs
  the evaluation.

### 2. Annual reports filed with the US securities regulator

- Filings define their short forms, for example a registrant's full name followed by its
  short name in brackets and quotes. That makes them a real test for the acronym resolver:
  the filer is known, so a short form defined for the registrant has a known answer.
- **Source:** the packaged corpus of annual reports on Hugging Face (`eloukas/edgar-corpus`,
  tagged Apache-2.0 [S]; the tag covers the packaging). The filings are published by a US
  government agency for public access; no formal licence was found [U]. The regulator's own
  access page refused an automated read on 2026-11-11, so its wording is not quoted here.

### 3. Wikidata

- Real short names and aliases of organisations, and items that carry two register
  identifiers (a legal-entity identifier and a company number).
- **Licence, read:** "All structured data (i.e. the main, Property, Lexeme, and EntitySchema
  namespaces) is released into the public domain under Creative Commons Zero."
  (<https://www.wikidata.org/wiki/Wikidata:Licensing>)
- Crowd-edited, so it is a silver standard; items that are people are excluded.

## Considered and not chosen

| Source | Terms | Verdict |
|---|---|---|
| OpenCorporates | share-alike database licence for free use; paid licence otherwise [S] | not usable in a commercial product |
| OpenSanctions | non-commercial [S] | no |
| PermID | basic fields CC BY 4.0, bulk access unclear [S] | unclear |
| Australian business register bulk extract | CC BY 3.0 AU [S] | usable with attribution; sole traders are named individuals and must be filtered |
| UK financial-services register extract | paid licence for reuse [S] | no |
| Academic matching benchmarks (Magellan, Leipzig, WDC Products) | no licence or unspecified [S] | unclear |
| Enron mail | no explicit licence; private mail of employees | not for enterprise demos |
| EU legislation and Commission decisions | CC BY 4.0 for Commission-owned content [S] | usable; a second source of defined short names |
| US provider registry (organisations only) and public provider datasets | US public data [S] | usable; individuals and named officials must be dropped |
| US flood-insurance claims | open terms [S] | no organisation names, so no use for matching |
| EU single access point for company data | opens 10 July 2027 [S] | later |

## Sources

- GLEIF terms of use: <https://www.gleif.org/en/meta/lei-data-terms-of-use>
- GLEIF golden copy service: <https://goldencopy.gleif.org/api/v2/golden-copies/publishes/latest>
- Companies House download page: <https://download.companieshouse.gov.uk/en_output.html>
- Open Government Licence v3.0: <https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/>
- Wikidata licensing: <https://www.wikidata.org/wiki/Wikidata:Licensing>
- Annual-report corpus: <https://huggingface.co/datasets/eloukas/edgar-corpus>
