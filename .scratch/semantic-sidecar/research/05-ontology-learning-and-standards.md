# Research 05: ontology learning methods and the standards to emit

Ticket: [05](../issues/05-ontology-learning-and-standards.md). Feeds the grilling in
[14](../issues/14-ontology-induction-and-standards.md). Researched 2026-11-08.

Source access: GitHub was read directly. w3.org, arxiv.org, tib-op.org and databricks.com could not
be fetched from this session, so claims about them come from search excerpts. Claims I could not
check against a primary source are marked **UNVERIFIED**.

## Summary

- **No method turns types into an ontology reliably enough to auto-accept.**
  - Blind taxonomy discovery in LLMs4OL 2025 tops out around F1 0.5–0.6.
  - Non-taxonomic relation extraction stays weak at every model scale.
  - The robust design lets several cheap, explainable signals vote: data and warehouse statistics,
    lexical heads, extensional inclusion and FCA. An LLM is one optional signal among them.
- **Constraints are the strongest thing we can induce.**
  - The warehouse already declares many of them: types, NOT NULL, PK/FK, value domains.
  - QSE-style mining adds support and confidence; pruning on those removed up to 93% of spurious
    shapes in its paper.
- **Standards to emit:**
  - OWL 2 held to the **QL** profile, so it stays usable for ontology-based data access (OBDA)
    over Databricks;
  - **SHACL 1.1** for closed-world constraints (1.2 is still a Working Draft);
  - **SKOS** labels in German and English;
  - **PROV-O** for Evidence and actors;
  - **SSSOM** for alignments;
  - **named graphs** for edge annotations, laid out to match RDF 1.2 reifiers (RDF 1.2 is still a
    Candidate Recommendation).
- **Ossie is not ready to be the ontology interchange.**
  - Its only tag, `osi-0.1.1-rc1`, has no ontology. The ontology spec is `0.2.0.dev0` on `main`:
    unreleased, ORM-style rather than OWL, and its mapping documents have no parser yet.
  - A Databricks metric-view converter exists (Java, metric-view YAML v1.1).
  - Treat Ossie as an experimental, pinned writer, and keep the Metis seam on 0.1.1.
- **Replace `lg:`.** Use `rdf:type` with minted class IRIs, SKOS labels and PROV-O. Keep a tiny
  published namespace only for confidence, support and signal.

## Current state

- `graph_export.py` writes Turtle under `urn:latence:graph:`.
  - Each node is an `lg:Entity` with `lg:entityType "<string>"`.
  - Edges get an `rdf:Statement` reification that carries `lg:confidence` and `lg:evidenceCount`.
  - Types are literals, so there is no schema level at all.
- `TypeVocabulary` / `CanonicalType` (ADR-0054) is the natural seed for classes and properties.
  It holds a kind, a canonical label, aliases, a mention count and strategies, and it merges
  append-don't-churn.
- `Evidence` has no actor; the map already plans that slice.
- ADR-0005 cites mined ConceptInducer, Taxonomy and AxiomInducer assets as "preserved". A grep
  finds them only in that ADR, so locate them before writing the design doc.

## Methods

| Method | Inputs → outputs | Reported quality | Licence | Sources |
|---|---|---|---|---|
| **LLMs4OL tasks / OntoLearner** (term typing, taxonomy discovery, non-taxonomic relations; RAG learners) | terms → types, is-a pairs, relations | 2025, 11 teams. Best blind taxonomy F1 0.57 and 0.47; term typing 0.47–0.80 depending on the ontology. **UNVERIFIED** (participant excerpts). A 2026 scale study: size is a weak predictor, and relation extraction stays hard. | MIT | [overview](https://www.tib-op.org/ojs/index.php/ocp/article/view/2913), [OntoLearner](https://github.com/sciknoworg/OntoLearner), [scale](https://arxiv.org/abs/2608.31118) |
| **OLLM** (fine-tuned LLM emits whole subgraphs) | text plus a supervising taxonomy → concept DAG | NeurIPS 2024: beats subtask composition on Wikipedia, especially on semantic F1; adapts to arXiv with few examples. Numbers **UNVERIFIED**. | Apache-2.0 (Mistral-7B base) | [repo](https://github.com/andylolu2/ollm) |
| **OntoGPT / SPIRES** | LinkML schema plus text → grounded instances | relation extraction mid-range, strong at grounding. One 2026 study: P 0.58, R 0.57 against humans. It populates a schema; it does not induce one. | BSD-3 | [repo](https://github.com/monarch-initiative/ontogpt) |
| **Ontogenia / CQbyCQ** | competency questions → OWL | o1-preview met engineer requirements; errors recur. About 91% and 84% CQ coverage **UNVERIFIED**. | paper | [Lippolis 2025](https://arxiv.org/abs/2503.05388) |
| **Taxonomy induction** (Chain-of-Layer, TaxoLLaMA) | term set → hierarchy | both claim state of the art on their benchmarks. **UNVERIFIED** | **UNVERIFIED** | [CoL](https://arxiv.org/abs/2402.07386), [TaxoLLaMA](https://arxiv.org/abs/2403.09207) |
| **Formal concept analysis** | objects × attributes (entities × properties, tables × columns) → lattice of subsumptions | deterministic and explainable; blows up on noise, so prune by stability. No modern benchmark. | algorithm, in-core | [Cimiano et al.](https://www.kde.cs.uni-kassel.de/hotho/pub/2004/techOntologylearningFCA.pdf) |
| **Rule and axiom mining** (AMIE 3; DL-Learner) | graph → Horn rules with PCA confidence; class expressions | AMIE 3 is exact and 15× faster than prior work | AMIE **CC BY 4.0** (reimplement the measures); DL-Learner **GPL-3**, excluded | [AMIE](https://github.com/dig-team/amie), [DL-Learner](https://github.com/SmartDataAnalytics/DL-Learner) |
| **QSE** | N-Triples → SHACL with support and confidence | VLDB 2023: ~12× faster; up to 93% of spurious shapes pruned; DBpedia 11,916 → 809 constraints | MIT (Java) | [paper](https://www.vldb.org/pvldb/vol16/p1023-rabbani.pdf), [repo](https://github.com/dkw-aau/qse) |
| **sheXer** | RDF → ShEx or SHACL, with a per-constraint conformance ratio and an acceptance threshold | KBS 2022 | Apache-2.0 (Python) | [repo](https://github.com/DaniFdezAlvarez/shexer) |
| **LLM → SHACL** | rule text → SHACL | syntactically valid, but often not semantically equivalent. No benchmark of LLM shape induction *from data* found. | various | [NL2SHACL-Bench](https://arxiv.org/abs/2608.07530) |
| **Alignment** (retrieve then rerank or LLM: OntoAligner, MILA, LogMap with an LLM oracle; BERTMap) | two vocabularies → scored correspondences | MILA up to +17% F; LogMap with an LLM oracle top-2 in OAEI 2025 Bio-ML. **UNVERIFIED** | OntoAligner and DeepOnto Apache-2.0 (DeepOnto needs a JVM). OntoAligner writes SSSOM. | [OntoAligner](https://github.com/sciknoworg/OntoAligner), [DeepOnto](https://github.com/KRR-Oxford/DeepOnto) |

**Reading across the table:**

- The well-supported, permissive pieces are three: statistical shape mining, FCA, and
  retrieve-and-rerank alignment.
- LLMs help with naming and hierarchy proposals, but each use needs an egress decision, and an LLM
  must never be a proposal's only signal.
- The text-only literature lacks our two strongest signals:
  - **declared warehouse constraints**;
  - **extensional evidence**: row sets, value sketches, query co-usage.
  These outrank model output.

## Standards recommendation

**OWL 2: the QL profile.**

- EL fits huge terminologies, and RL fits rule materialisation.
- QL is built for query rewriting over relational data (R2RML, OBDA). Ontop implements RDFS, OWL 2
  QL and R2RML, and lists Databricks as a source since 5.0
  ([guide](https://ontop-vkg.org/guide/)).
- That makes the ontology queryable over UC tables without copying data.
- QL's gaps (functionality, cardinality, keys) are closed-world material that belongs in SHACL
  anyway.
- **Allowed constructs:**
  - `owl:Class`, `rdfs:subClassOf`;
  - properties with domain and range;
  - `owl:inverseOf`, `owl:disjointWith`;
  - superclass-side existentials;
  - annotations.
- A profile check gates every emit.

**SHACL 1.1 Core (W3C Recommendation 2017) now; track 1.2.**

- SHACL 1.2 Core is a Working Draft; the newest snapshot seen is 28 Aug 2026
  ([WD](https://www.w3.org/TR/2026/WD-shacl12-core-20260828/)).
- It adds `sh:reifierShape` and `sh:reificationRequired` for annotated edges, node-expression
  targets and a separate Rules spec. Design shapes so that adopting these is additive.
- Shapes carry datatype, min and max count, `sh:in`, pattern, class and keys.

**SKOS labels.**

- Put `skos:prefLabel` (one per language tag, integrity rule S14) and `skos:altLabel` `@de`/`@en`
  directly on classes and properties. No dual typing is needed.
- `skos:hiddenLabel` holds the raw induced labels; `skos:definition` holds definitions.
- The business-term layer is a `skos:ConceptScheme` linked to classes.

**PROV-O (W3C Recommendation 2013).** Every proposal is a `prov:Entity`:

- `prov:wasGeneratedBy` the run activity (config hash, corpus version);
- `prov:wasAttributedTo` its actor: a `prov:SoftwareAgent` for an engine (id and version) or a
  model (id, prompt hash, egress decision), or a `prov:Person`;
- `prov:wasDerivedFrom` its Locators;
- review decisions are activities, related by `prov:wasInvalidatedBy` and `prov:wasRevisionOf`.

**Edge annotations: named graphs now, RDF 1.2 later.**

- Status per search excerpts:
  - RDF 1.2 Concepts is a CR (7 Apr 2026);
  - Semantics is a CR Draft (24 Sep 2026);
  - Turtle 1.2 is a WD (12 Aug 2026);
  - no Recommendation was found
  ([CR](https://www.w3.org/TR/2026/CR-rdf12-concepts-20260407/)).
- Model each annotated assertion as a *reifier* resource with a stable IRI.
  - Serialise it today as a TriG or N-Quads named graph, where the graph IRI is the reifier IRI.
  - Add a Turtle 1.2 writer (`<< s p o ~ r >>`) at Recommendation. The mapping is 1:1, so no
    contract change.
- Drop `rdf:Statement`: it is verbose and semantically disconnected from the triple it describes.

**SSSOM for alignment.**

- Each alignment records the predicate, confidence, `mapping_justification` and author.
- Default to `skos:closeMatch` or `skos:exactMatch`. Never auto-emit `owl:equivalentClass`, which
  imports a foreign ontology's entailments.

**IRI policy.**

1. Use an adopter-configured HTTP base, for example `https://<adopter-domain>/ont/<domain>/`. Keep
   `urn:latence:` only as the offline default, with a warning.
2. Make IRIs **opaque and never label-derived.**
   - Mint `class/C<ulid>` and `prop/P<ulid>` at first acceptance, and persist them in the vocabulary
     state, so replay is byte-identical.
   - This is the ADR-0054 no-rename rule, lifted to IRIs.
3. Make proposals **content-addressed**: `proposal/<sha256>`. A human decision then re-attaches when
   the same proposal recurs after a rebuild.
4. **Never mint for UC objects.** Reference them as
   `urn:uc:<metastore_id>:<securable_type>:<object_id>`. Whether those ids are stable and exposed is
   **UNVERIFIED**; it is ticket 01's question.
5. On a merge, the losing IRI gets `owl:deprecated` and `dcterms:isReplacedBy`.
6. Each release gets an `owl:versionIRI` and an `owl:priorVersion`.

### Ossie

**Read from GitHub on 2026-11-08:**

- `apache/ossie` has one tag, `osi-0.1.1-rc1` (19 May 2026), and no releases.
- `core-spec` on `main` is `0.2.0.dev0` ("schema may change") and breaks the 0.1.1 document shape.
  Metis's mapping table records the same finding and pins 0.1.1.
- `ontology/ontology.md` (`0.2.0.dev0`, 2026-05-29) defines:
  - `EntityType` and `ValueType` concepts, with `extends`;
  - n-ary `relationships` with `roles`, `multiplicity` (ManyToOne or OneToOne) and `verbalizes`;
  - `identify_by`;
  - `derived_by` and `requires` rules;
  - optional `iri` and `prefixes`;
  - standalone mapping documents, for which parser and converter support "is not yet available".
- No OWL, SHACL or SKOS translation is specified anywhere. `ossie-converters/ontology` converts only
  to and from an internal Spec YAML, and to Palantir and RelationalAI.
- The working group is "Ontology & Semantic Interoperability" (ROADMAP). A FIBO and external-IRI
  question is unanswered ([discussion 219](https://github.com/apache/ossie/discussions/219)).
- **Databricks converter:** `ossie-converters/databricks` is bidirectional, works on metric-view
  YAML v1.1, and Java is "the maintained implementation". Databricks documents an `export_to_ossie`
  error class; that it is feature-flagged is **UNVERIFIED**. An NVIDIA Auto Ontology ↔ Ossie
  converter also exists, a sign that auto-ontology tools are converging on Ossie.

**Implications:**

1. Ossie's ontology is fact-oriented, so OWL plus SHACL is lossy:
   - n-ary relationships need reifier classes;
   - rules need SHACL-SPARQL or SHACL Rules;
   - `verbalizes` becomes an annotation.
2. Our contracts stay canonical, and Ossie is an experimental writer pinned by commit digest. Its
   `iri` field carries our IRIs, so a round trip is identifiable.
3. The Metis seam stays 0.1.1-shaped until Ossie tags 0.2.0 and Metis re-pins.

### Unity Catalog and metric views

| Ontology | UC counterpart | Direction |
|---|---|---|
| `owl:Class` | table or view bound as its extension (a Binding) | overlay; emit an R2RML mapping |
| datatype property | column (UC type → `xsd:`) | overlay |
| object property | FK, metric-view `joins`, lineage or co-usage join | overlay; propose a missing FK |
| SKOS labels @de/@en | comments; metric-view `display_name` and `synonyms` (at most 10, YAML 1.1, Public Preview) | **propose** upstream, never shadow |
| SHACL count, datatype, `sh:in` | NOT NULL, the column type, CHECK, governed-tag values | validate, then propose |
| measure | metric-view measure | UC owns it; we only bind concept to measure |
| glossary concept | UC Business Glossary ("coming soon", **UNVERIFIED**); Domains (Public Preview) | propose |

Genie Ontology is reportedly a preview learned-context layer with no export and no changelog, and
"not a full formal ontology" (**UNVERIFIED**,
[forum](https://community.databricks.com/t5/data-engineering/genie-ontology/td-p/164536)). A
traceable, exportable formal layer keyed on UC ids is still a gap. Positioning belongs to ticket 06.

## Sketch for the P4 design doc

**Inputs.**

- `TypeVocabulary`: entity types become class seeds, relation types become property seeds.
- Canonical entities and edges, with Evidence.
- The warehouse snapshot: schema, PK/FK, comments, tags, metric-view YAML, lineage, co-usage and
  value sketches.
- Schema-matching bindings (ticket 03).
- Prior decisions and the IRI registry.

**New contract: `OntologyProposal`.**

- `proposal_id` (content hash);
- `kind`, one of `class | label | subclass | property | domain | range | inverse | disjoint |
  constraint | key | alignment | binding`;
- `subject` and `object` IRIs, and a `payload`;
- `signals`: a list of `{name, value, support, confidence}`;
- `evidence`: a list of `Locator`;
- `actor`;
- `status` (`proposed | accepted | rejected | superseded`) and `decision` (actor, time, note).

This needs a `SCHEMA_VERSION` bump. LLM signals are cached by input hash, so replay never calls a
model.

**Stages.** All emit proposals only. Each is a Capability, and a deterministic reference Provider
comes first.

1. **Concepts.**
   - One class candidate per entity type, and one per bound UC table cluster.
   - Fold duplicates with the ADR-0054 cascade. An unresolved pair stays two classes.
2. **Labels.**
   - pref, alt and hidden labels come from the canonical label, aliases, comments, synonyms and raw
     labels, with language detection.
   - Missing DE/EN labels become proposals, which an LLM may draft if egress is allowed.
3. **Hierarchy.** Signals:
   - extensional inclusion, with support ≥ k and confidence ≥ τ;
   - stable FCA concepts over entities × properties and tables × columns;
   - lexical head containment, which needs a German compound splitter;
   - the parent of an external match;
   - optional LLM taxonomy discovery.

   `rdfs:subClassOf` needs two independent signals or a human; with one signal, emit
   `skos:broader`.
4. **Properties.**
   - Relation types, FKs and joins become object properties; columns become datatype properties.
   - Domain and range are the least common subsumer covering ≥ 1-ε of the observed edges. Uncovered
     edges are counter-evidence.
5. **Constraints (QSE-style).**
   - Support and confidence for datatype, class, min/max count, `sh:in` and pattern, per (class,
     property).
   - Keys come from PK, unique constraints and observed uniqueness.
   - Declared UC constraints enter with actor `uc`.
6. **Axioms.**
   - Disjointness: no shared instances, enough support, no lexical overlap.
   - Inverses from paired edges.
   - The QL profile check routes anything outside the profile to SHACL, or rejects it.
7. **Alignment.** Retrieve and rerank against configured vocabularies, emitted as SSSOM proposals.
8. **Validate before acceptance.**
   - Every shape is run against the instance graph and a warehouse sample. The violation count and
     sample focus nodes become evidence.
   - A shape over its stated tolerance cannot be auto-accepted.
9. **Review (ticket 16).**
   - Decisions are keyed by `proposal_id` and survive rebuilds.
   - Auto-accept only for actor `uc` or a declared engine rule over its threshold. Model-only
     proposals never auto-accept.
10. **Writers.**
    - OWL, SHACL and SKOS (Turtle); PROV and annotations (TriG); SSSOM; R2RML; Ossie (experimental);
    - UC proposal patches: comments, metric-view synonyms, tags;
    - the instance export moves from `lg:` to `rdf:type <class>` plus reifier graphs.

**Gates.**

- Byte-stable writers.
- The profile check.
- SHACL self-validation of the instance export.
- A round trip for every writer that has a reader. Parsers appear in tests only, per ADR-0016.

## Open questions

1. Does v1 need hierarchy and axioms, or are classes, properties, labels, SHACL and bindings
   enough? Hierarchy is the least reliable output (ticket 10).
2. What are the defaults for k, τ and ε, and are they calibrated per corpus on a gold slice
   (ticket 18)?
3. Do labels need provenance, which would mean SKOS-XL?
4. Are UC ids stable and exposed for tables, columns and metric views (ticket 01)?
5. Which UC write-back carries a binding (tags, comments, synonyms, glossary)? Which can the engine
   write rather than only propose?
6. When Ossie tags 0.2.0, does Metis re-pin, and does the seam become the Ossie mapping document?
7. Is LLM egress allowed for naming and taxonomy signals, and with which EU-only models?
8. Where are the ADR-0005 inducer assets, and do they survive review against this design?
9. Should schema.org be the default alignment target, or none? FIBO's licence and size are
   unchecked (**UNVERIFIED**).
10. Do we switch to Turtle 1.2 at RDF 1.2 Recommendation, or once Jena, RDF4J and Oxigraph ship it
    by default?
