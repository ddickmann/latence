# Research 05: ontology learning methods and the standards to emit

Ticket: [05-ontology-learning-and-standards](../issues/05-ontology-learning-and-standards.md). Feeds the
grilling in [14](../issues/14-ontology-induction-and-standards.md). Researched 2026-11-08.

Source access: GitHub (including raw files) was readable directly. w3.org, arxiv.org, tib-op.org and
databricks.com could not be fetched from this session, so claims about them come from search-result
excerpts of those pages. Claims I could not check against a primary source are marked **UNVERIFIED**.

## Summary

- **No single method turns canonical types into an ontology end to end with useful reliability.**
  In the LLM ontology-learning challenges, taxonomy discovery tops out around F1 0.5–0.6 on blind
  sets. Non-taxonomic relation extraction is still the weak task, at every model scale. The robust
  pattern is to have several cheap, explainable signals vote, and to use the LLM as one optional
  signal. That pattern is the statistics of the data and warehouse, lexical heads, extensional
  inclusion and FCA.
- **Constraints are the strongest thing we can induce, because the warehouse already declares many
  of them** (types, NOT NULL, primary and foreign keys, value domains). Statistical shape mining in
  the style of QSE adds support and confidence to every constraint, and pruning on those removed up
  to 93% of spurious shapes in its paper. Every constraint should be a proposal carrying support,
  confidence and its counter-examples.
- **Standards to emit:**
  - OWL 2 for the schema, held to the **OWL 2 QL** profile so it stays usable for ontology-based
    data access over Databricks (Ontop supports Databricks);
  - **SHACL 1.1** for the closed-world constraints. SHACL 1.2 is still a Working Draft;
  - **SKOS** labels in German and English;
  - **PROV-O** for Evidence and actors;
  - **SSSOM** for alignments to external vocabularies;
  - **named graphs** for edge annotations now, laid out to match RDF 1.2 reifiers. RDF 1.2 is still
    a Candidate Recommendation.
- **Ossie is not ready to be the ontology interchange.**
  - The only tag is `osi-0.1.1-rc1`, which has no ontology.
  - The ontology spec exists only on `main`, as `0.2.0.dev0` (unreleased and mutable). It is an
    ORM-style fact model (roles, verbalisations, multiplicities), not OWL, and its mapping
    documents have no parser or converter yet.
  - A Databricks to metric-view converter exists (Java, metric-view YAML v1.1).
  - Treat Ossie as an experimental, pinned export, and keep the Metis seam on 0.1.1 until both
    sides re-pin.
- **Replace the private `lg:` vocabulary.**
  - Use `rdf:type` with minted class IRIs, SKOS for labels, and PROV-O for Evidence.
  - Keep a tiny published namespace only for what no standard covers: confidence, support and
    signal.

## Current state (read-only)

- `graph_export.py` hand-emits Turtle under `urn:latence:graph:` as `lg:`:
  - every node is an `lg:Entity` with `lg:entityType "<string>"`;
  - edges use a minted predicate plus an `rdf:Statement` reification that carries `lg:confidence`
    and `lg:evidenceCount`.
  - Types are literals, not classes, so there is no schema level at all.
- `TypeVocabulary` / `CanonicalType` (ADR-0054) hold the induced type vocabulary:
  - each entry has a kind (entity or relation), a canonical label, aliases, a mention count and the
    clustering strategies used;
  - merging is append-don't-churn, and an unknown label maps to itself;
  - this is the natural seed for classes and properties.
- `Evidence` carries mention ids, document ids, a snippet and a confidence. It has no actor field;
  the map already lists "an actor on every decision" as a contract slice.
- ADR-0005 cites mined ConceptInducer, Taxonomy and AxiomInducer assets as "preserved verbatim".
  A grep of the tree finds them only in that ADR, so locate them before writing the design doc.

## Methods

| Method (family) | Inputs | Outputs | Reported quality | Licence | Sources |
|---|---|---|---|---|---|
| **LLMs4OL tasks via OntoLearner** (term typing, taxonomy discovery, non-taxonomic relations; retrieval-augmented LLM learners) | terms, types, optional context | type assignments, is-a pairs, typed relations | 2025 challenge, 11 teams. Best blind taxonomy-discovery F1 in participant reports is 0.57 and 0.47. Term typing runs 0.47–0.80 depending on the ontology. Non-taxonomic extraction is "a key challenge". **UNVERIFIED** (participant excerpts, not the official leaderboard). A 2026 scale study finds model size a weak predictor: dense 27B beat larger MoE models on typing, and relation extraction stays hard at every size. | OntoLearner: MIT | [overview](https://www.tib-op.org/ojs/index.php/ocp/article/view/2913), [OntoLearner](https://github.com/sciknoworg/OntoLearner), [scale study](https://arxiv.org/abs/2608.31118) |
| **OLLM** (fine-tune an LLM to emit whole subgraphs) | domain text with existing taxonomy supervision | concept DAG | NeurIPS 2024: beats subtask composition on Wikipedia, especially on semantic F1 (fuzzy and continuous); adapts to arXiv with few examples. Exact numbers **UNVERIFIED**. | code Apache-2.0; base model Mistral-7B-Instruct | [repo](https://github.com/andylolu2/ollm), [paper](https://arxiv.org/abs/2410.23584) |
| **OntoGPT / SPIRES** (schema-guided extraction with grounding) | a LinkML schema plus text | instances grounded to ontology ids; ungroundable ones get an `AUTO:` prefix | relation extraction mid-range versus specialised extractors, strong at grounding. One 2026 domain study: P 0.58, R 0.57 against human reviewers. Populates a given schema rather than inducing one. | BSD-3-Clause | [repo](https://github.com/monarch-initiative/ontogpt), [SPIRES](https://arxiv.org/abs/2304.02711) |
| **Ontogenia, memoryless CQbyCQ** (LLM drafts OWL from competency questions) | user stories and competency questions | OWL modules | o1-preview with Ontogenia met engineer requirements and beat novices; errors recur. Per-CQ figures (about 91% and 84%) **UNVERIFIED**. | paper only | [Lippolis et al. 2025](https://arxiv.org/abs/2503.05388) |
| **Taxonomy induction** (Chain-of-Layer in-context; TaxoLLaMA instruction-tuned on WordNet) | entity or term set | hierarchy | CoL claims state of the art on 4 benchmarks. TaxoLLaMA claims state of the art on 11 of 16 lexical tasks. Numbers **UNVERIFIED**. | **UNVERIFIED** | [CoL](https://arxiv.org/abs/2402.07386), [TaxoLLaMA](https://arxiv.org/abs/2403.09207) |
| **Formal concept analysis** (Cimiano, Hotho and Staab lineage) | binary context: objects × attributes (entities × properties, or tables × columns) | concept lattice, read as candidate subsumptions | deterministic and explainable; lattices blow up on noisy contexts, so prune with iceberg or stability thresholds. No modern benchmark found. | algorithm; small to implement in-core | [Cimiano et al.](https://www.kde.cs.uni-kassel.de/hotho/pub/2004/techOntologylearningFCA.pdf) |
| **Rule and axiom mining** (AMIE 3 Horn rules; DL-Learner class expressions) | instance graph | rules with PCA confidence; OWL class expressions | AMIE 3: 15× faster than prior work, exact. DL-Learner is mature but old. | AMIE: **CC BY 4.0** (odd for code; reimplement the measures, don't vendor it). DL-Learner: **GPL-3.0**, excluded. | [AMIE](https://github.com/dig-team/amie), [DL-Learner](https://github.com/SmartDataAnalytics/DL-Learner) |
| **QSE** (SHACL extraction with support and confidence) | N-Triples knowledge graph | SHACL node and property shapes, pruned by support and confidence | VLDB 2023: about 12× faster, up to 93% of spurious shapes pruned; DBpedia went from 11,916 to 809 constraints. | MIT (Java) | [paper](https://www.vldb.org/pvldb/vol16/p1023-rabbani.pdf), [repo](https://github.com/dkw-aau/qse) |
| **sheXer** | RDF graph | ShEx or SHACL, with a per-constraint conformance ratio used as "trustworthiness"; `acceptance_threshold` | KBS 2022; Python. | Apache-2.0 | [repo](https://github.com/DaniFdezAlvarez/shexer) |
| **LLM to SHACL** (NL2SHACL-Bench, text2shacl) | natural-language rules | SHACL | output is syntactically valid but often not semantically equivalent on complex patterns. No benchmark of LLM shape induction *from data* was found. | various | [NL2SHACL-Bench](https://arxiv.org/abs/2608.07530), [text2shacl](https://github.com/citiususc/text2shacl) |
| **Alignment** (retrieve, then rerank or ask an LLM: OntoAligner, OLaLa, MILA, LogMap with an LLM oracle; BERTMap in DeepOnto) | two vocabularies | correspondences with confidence | MILA reports up to +17% F-measure. LogMap with an LLM oracle was top-2 in OAEI 2025 Bio-ML. **UNVERIFIED** (excerpts). OntoAligner writes SSSOM. | OntoAligner and DeepOnto: Apache-2.0 (DeepOnto pulls in a JVM and OWLAPI) | [OntoAligner](https://github.com/sciknoworg/OntoAligner), [DeepOnto](https://github.com/KRR-Oxford/DeepOnto), [MILA](https://arxiv.org/abs/2501.11441) |

**Reading across the table:**

- Nothing in the table induces a hierarchy or axioms reliably enough to auto-accept.
- The well-supported, permissive pieces are three:
  - statistical shape mining (the QSE and sheXer measures);
  - FCA;
  - retrieve-and-rerank alignment.
- LLM methods help with naming and hierarchy proposals. Each needs an egress decision (map
  principle) and must never be the only signal.
- Two warehouse-specific advantages are missing from all the text-only literature:
  - **declared constraints** (types, nullability, keys, foreign keys, metric-view joins);
  - **extensional evidence** (row sets, value sketches, query co-usage).
  These should outrank model output.

## Standards recommendation

### OWL 2: target the QL profile, and keep the TBox small

- **EL** suits huge class hierarchies with existentials, as medical terminologies have. It is not our
  shape.
- **RL** suits rule materialisation.
- **QL** is designed for query rewriting over relational data, which is the R2RML and OBDA pattern.
  Ontop implements it (RDFS and OWL 2 QL, R2RML) and lists Databricks as a supported source since
  5.0 ([Ontop guide](https://ontop-vkg.org/guide/)). That makes the emitted ontology queryable over
  UC tables without copying data, which fits "never a parallel definition".
- QL lacks functional properties, cardinalities and keys, which is exactly the closed-world material
  that belongs in SHACL anyway.
- **Rule:** the OWL file uses only these constructs, and a profile check gates the emit:
  - `owl:Class`, `rdfs:subClassOf`;
  - object and datatype properties with `rdfs:domain` and `rdfs:range`;
  - `owl:inverseOf`, `owl:disjointWith`;
  - an existential on the superclass side;
  - annotations.

### SHACL: emit SHACL 1.1 (W3C Recommendation 2017) Core only; track 1.2

- SHACL 1.2 Core is a Working Draft. The newest snapshot seen is 28 Aug 2026.
- What 1.2 adds that matters here:
  - `sh:reifierShape` and `sh:reificationRequired`, for constraints on annotated edges;
  - node expressions for targets;
  - a separate SHACL Rules spec.
- Design shapes so that a later move to `sh:reifierShape` is additive.
- Shapes, not OWL, carry `sh:datatype`, `sh:minCount`/`sh:maxCount`, `sh:in`, `sh:pattern`,
  `sh:class`, and keys (shapes over `sh:path` plus uniqueness via SPARQL-based constraints, or a
  documented annotation).
- Sources: [SHACL 1.2 Core WD](https://www.w3.org/TR/2026/WD-shacl12-core-20260828/), and
  [the 1.2 overview](https://raw.githack.com/w3c/data-shapes/master/shacl12-overview/) for the
  feature list.

### SKOS for labels and the glossary layer

- Put `skos:prefLabel` (at most one per language tag, per SKOS integrity rule S14) and
  `skos:altLabel` with `@de` and `@en` directly on classes and properties. They are annotation
  properties, so no dual typing is needed.
- `skos:hiddenLabel` holds raw induced labels (the `CanonicalType.aliases` and `raw_label`), so
  every surface stays findable without being displayed.
- Use `skos:definition` for definitions.
- The business-term layer (a future UC Business Glossary) is a `skos:ConceptScheme` whose concepts
  link to classes. Use SKOS-XL only if labels themselves need provenance; probably they do, see
  open questions.

### PROV-O for Evidence and actors

Each accepted or proposed axiom is a `prov:Entity`:

- `prov:wasGeneratedBy` the run, a `prov:Activity` carrying the run id, config hash and corpus
  version;
- `prov:wasAttributedTo` its actor:
  - a `prov:SoftwareAgent` for an engine Provider, with id and version;
  - the same for a model, with model id, prompt hash and egress decision;
  - a `prov:Person` for a human;
- `prov:wasDerivedFrom` its Locators: a text span, a UC object, a query, a lineage edge or a value
  sketch;
- review decisions are `prov:Activity` instances, related by `prov:wasInvalidatedBy` or
  `prov:wasRevisionOf`.

PROV-O has been a Recommendation since 2013, is stable and has wide tooling.

### Edge annotations: named graphs now, RDF 1.2 reifiers later

- RDF 1.2 Concepts has been a Candidate Recommendation since 7 Apr 2026, and Semantics a CR Draft
  since 24 Sep 2026. Turtle 1.2 is a Working Draft (12 Aug 2026). No Recommendation was found.
- **Decision proposal:** model every annotated assertion as a *reifier resource* with a stable IRI.
  - Serialise it today as a named graph in TriG or N-Quads, where the graph IRI is the reifier IRI
    and PROV and confidence hang off it.
  - Add an opt-in Turtle 1.2 writer (`<< s p o ~ r >>`) when RDF 1.2 reaches Recommendation.
  - Both are 1:1, so nothing changes in the contracts.
- Drop `rdf:Statement` reification. It is verbose and has no semantics linking it to the asserted
  triple.
- Sources: [RDF 1.2 Concepts CR](https://www.w3.org/TR/2026/CR-rdf12-concepts-20260407/) and
  [W3C news](https://www.w3.org/news/2026/w3c-invites-implementations-of-rdf-1-2-concepts-and-abstract-data-model-and-rdf-1-2-semantics).
  Status is as of the search excerpts, not fetched.

### Alignment: SSSOM

- Write mappings to external vocabularies (schema.org, FIBO, others) as SSSOM, with the predicate,
  confidence, `mapping_justification` (a semapv term) and the author.
- Default to `skos:closeMatch` or `skos:exactMatch`. Never auto-emit `owl:equivalentClass`, which
  imports the other ontology's entailments.
- OntoAligner already writes SSSOM.

### IRI policy

1. **Base.** Adopter-configured HTTP namespace, for example `https://<adopter-domain>/ont/<domain>/`,
   optionally behind w3id-style redirects. Keep `urn:latence:` only as the offline default, with a
   warning that it is not dereferenceable.
2. **Opaque, never derived from labels.**
   - Mint `class/C<ulid>` and `prop/P<ulid>` at first acceptance, and persist them in the
     vocabulary state, so replay is byte-identical.
   - Labels change; IRIs don't. This is the ADR-0054 no-rename rule, lifted to IRIs.
3. **Content-addressed proposals.** `proposal/<sha256 of canonical payload>`, so a human decision
   re-attaches across rebuilds when the same proposal recurs.
4. **Never mint for what UC owns.**
   - Reference UC objects by a URN built from UC's own ids, for example
     `urn:uc:<metastore_id>:<securable_type>:<object_id>`. That table and column ids are stable and
     exposed through the API is **UNVERIFIED**; it is ticket 01's question.
   - Fall back to the three-part name plus a hash of the `CREATE` time if not.
5. **Deprecation, not deletion.** On a merge, the loser gets `owl:deprecated true` and
   `dcterms:isReplacedBy`.
6. **Versions.** Each ontology release gets an `owl:versionIRI` (`…/ont/<domain>/<version>/`) and
   an `owl:priorVersion`.

### Interchange: Ossie

**Facts read from GitHub, 2026-11-08:**

- `apache/ossie` has one tag, `osi-0.1.1-rc1` (19 May 2026), and no releases.
- `core-spec` on `main` is `0.2.0.dev0`, "in development, schema may change", and breaks the
  0.1.1 document shape. Metis's mapping table records the same finding and pins 0.1.1.
- `ontology/ontology.md` is `0.2.0.dev0` (2026-05-29). It defines:
  - `EntityType` and `ValueType` concepts, with `extends` for hierarchy;
  - n-ary `relationships` with `roles`, `multiplicity` (ManyToOne or OneToOne) and `verbalizes`
    templates;
  - `identify_by` for preferred identifiers;
  - `derived_by` and `requires` rule expressions;
  - optional `iri` and `prefixes` on concepts and relationships;
  - standalone mapping documents that bind an ontology to one semantic model, via
    `concept_mappings`, `object_mappings` and `link_mappings`.
  The spec itself says parser and converter support for mapping documents "is not yet available".
- The ontology spec names no OWL, SHACL or SKOS translation. The `ossie-converters/ontology`
  package converts only to and from an internal "Spec YAML", and to Palantir and RelationalAI.
- The ROADMAP's working group is "Ontology & Semantic Interoperability": business concepts, mappings
  from concepts to datasets and fields, and non-tabular data. A user question about FIBO alignment
  and external IRIs ([discussion 219](https://github.com/apache/ossie/discussions/219)) has no
  answer.
- **Databricks converter:** `ossie-converters/databricks` is bidirectional and offline, works on
  metric-view YAML v1.1, and Java is "the maintained implementation". Databricks documents an
  `export_to_ossie` error class; a search excerpt says it is feature-flagged and supports version
  1.1 only (**UNVERIFIED**). There is also an NVIDIA Auto Ontology to Ossie converter, a sign that
  auto-ontology tools are converging on Ossie as exchange.

**Implications:**

1. The Ossie ontology is a fact-oriented model. Its translation to OWL plus SHACL is lossy:
   - n-ary relationships need reifier classes;
   - `requires` and `derived_by` need SHACL-SPARQL or SHACL Rules;
   - `verbalizes` has no OWL home; carry it as an annotation.
2. Keep our canonical model in our contracts, and treat Ossie, OWL and SHACL as writers.
3. Make the Ossie writer experimental, pinned by commit digest the way Metis pins by schema digest.
   Its `iri` field carries our class IRIs, so a round trip is identifiable.
4. The sidecar seam to Metis stays 0.1.1-shaped (datasets, fields, metrics, relationships, plus
   synonyms) until Ossie tags 0.2.0 and Metis re-pins.

### Mapping to Unity Catalog and metric views

| Ontology element | UC / metric-view counterpart | Direction |
|---|---|---|
| `owl:Class` (entity-bearing) | table or view bound as its extension (a Binding) | overlay; R2RML or Ontop mapping emitted |
| datatype property | column, with its UC type mapped to an `xsd:` type | overlay |
| object property | FK constraint, metric-view `joins`, or a lineage or co-usage join | overlay; propose an FK upstream if missing |
| `skos:prefLabel` / `altLabel` @de/@en | column or table comment; metric-view `display_name` and `synonyms` (at most 10, at most 255 characters, YAML 1.1, Public Preview) | **propose** upstream, never shadow |
| SHACL `sh:minCount 1`, `sh:datatype`, `sh:in` | NOT NULL, the column type, a CHECK or a governed-tag value set | validate against data; propose upstream |
| measure | metric-view measure | UC owns it; the ontology only links concept to measure, as a Binding annotation |
| glossary `skos:Concept` | UC Business Glossary term ("preview coming soon", **UNVERIFIED**), or UC Domains (Public Preview) | propose |

Genie Ontology is reported as a learned context layer: preview, no export or changelog, and "not a
full formal ontology" according to staff on the community forum (**UNVERIFIED**,
[thread](https://community.databricks.com/t5/data-engineering/genie-ontology/td-p/164536)).
A traceable, exportable, formal layer keyed on UC ids is therefore still a gap. It is also a
moving competitor; positioning belongs to ticket 06.

## Sketch for the P4 design doc

**Inputs.**

- `TypeVocabulary`: entity types become class seeds, relation types become property seeds.
- Canonical entities and edges, with their Evidence.
- The warehouse snapshot:
  - tables, columns, types, nullability, PK/FK, comments and tags;
  - metric-view YAML;
  - lineage, query co-usage and value sketches.
- Schema-matching output (ticket 03) that binds text types to UC objects.
- Prior decisions and the persisted IRI registry.

**Contract (new).** An `OntologyProposal`:

- `proposal_id` (content hash);
- `kind`, one of `class | label | subclass | property | domain | range | inverse | disjoint |
  constraint | key | alignment | binding`;
- `subject` and `object` IRIs, and a `payload` (for example a SHACL fragment);
- `signals`: a list of `{name, value, support, confidence}`;
- `evidence`: a list of `Locator`;
- `actor`;
- `status` (`proposed | accepted | rejected | superseded`) and `decision` (`actor`, `at`, `note`).

A `SCHEMA_VERSION` bump is required. LLM signals are cached by input hash, so replay never calls
the model again.

**Stages.** Each stage emits proposals only, and is a Capability with a deterministic reference
Provider first.

1. **Concept candidates.**
   - One class per entity `CanonicalType`, plus one per bound UC table cluster.
   - Fold duplicates with the same audited cascade as ADR-0054, gated by type compatibility. An
     unresolved pair stays two classes, so there is no silent over-merge.
2. **Labels.**
   - prefLabel from the canonical label; altLabel from aliases, UC comments and metric-view
     synonyms; hiddenLabel from raw labels.
   - Language tag from detection, or from a DE/EN dictionary of the source. Missing language
     labels become label proposals, which an LLM may draft (an egress decision).
3. **Hierarchy.** Signals:
   - extensional inclusion: instances typed A, or rows bound to A, are a subset of B, with support
     ≥ k and confidence ≥ τ;
   - FCA over (entities × properties) and (tables × columns), using stable concepts only;
   - lexical head containment ("Geschäftskunde" ⊑ "Kunde", "customer account" ⊑ "account");
     German compounds need a splitter;
   - external alignment, where a parent's match is the subsumer;
   - optional LLM taxonomy discovery.

   **Promotion rule:** `rdfs:subClassOf` needs at least two independent signals, or a human.
   With one signal, emit `skos:broader` as a soft hierarchy.
4. **Properties.**
   - Each relation `CanonicalType` becomes an object property, and FKs and joins become object
     properties.
   - Non-key columns become datatype properties.
   - Domain and range come from the observed distribution of (subject class, object class). Propose
     the least common subsumer covering ≥ (1-ε) of the edges, and record the uncovered edges as
     counter-evidence.
5. **Constraints** (QSE-style). For each (class, property), compute support and confidence for:
   - datatype, class, min and max count, `sh:in` (low-cardinality value sets) and pattern;
   - keys, from PK, from unique constraints, and from observed uniqueness.

   Declared warehouse constraints enter with the actor `uc` at confidence 1. Mined ones carry their
   statistics.
6. **Axioms.**
   - Disjointness: zero shared instances, enough support for both classes, and no lexical overlap.
   - `owl:inverseOf` from symmetric edge pairs.
   - Every emitted axiom passes an OWL 2 QL profile check. Anything outside the profile is routed
     to SHACL or rejected.
7. **Alignment.** Retrieve and rerank against configured vocabularies (schema.org by default; FIBO
   and others opt-in), output as SSSOM proposals.
8. **Validate before acceptance.**
   - Run every candidate shape against the current instance graph and the warehouse sample, and
     attach the violation count and sample focus nodes.
   - A shape with violations above its stated tolerance cannot be auto-accepted.
   - This answers the grilling's "validated against data before acceptance": yes, always, and the
     result is evidence.
9. **Review and persistence** (ticket 16).
   - Decisions are keyed by `proposal_id` and survive rebuilds.
   - Auto-acceptance only for actor `uc`, or for a declared engine rule above its threshold.
     Model-only proposals are never auto-accepted.
10. **Writers.**
    - OWL in Turtle (accepted classes and properties only);
    - SHACL in Turtle;
    - SKOS;
    - PROV and edge annotations in TriG;
    - SSSOM;
    - the R2RML mapping (bindings);
    - Ossie, experimental;
    - UC proposals: comment and synonym patches for metric-view YAML, and tag assignments.
    - The instance KG export switches from `lg:` to `rdf:type <class IRI>` and the reifier-graph
      layout.
    - The residual `ltc:` namespace (confidence, support, signal) is documented and published.

**Gates.**

- Byte-stable writers.
- The profile check.
- A SHACL self-validation of the instance export.
- A round-trip test for every writer that has a reader: Ossie via the converter repo, OWL via a
  parser in tests only (ADR-0016 keeps core dependency-free).

## Open questions

1. **Domain-model level (ticket 10).** Does v1 need hierarchy and axioms at all, or are classes,
   properties, labels, SHACL and bindings enough? The literature suggests hierarchy is the
   least-reliable output.
2. **Thresholds.** What support and confidence defaults (k, τ, ε) apply, and are they calibrated
   per corpus against a gold slice (ticket 18)?
3. **Label provenance.** Does it require SKOS-XL, so that each label is a resource with PROV?
4. **UC ids.** Are UC securable ids stable and exposed for tables, columns and metric views? What
   is the URN shape (ticket 01)?
5. **UC write-back.** Do tags (`concept=<IRI>`), comments, metric-view synonyms or the glossary
   carry the binding upstream? Which of these can the engine write, as opposed to only propose?
6. **Ossie.** When Ossie tags 0.2.0, does Metis re-pin, and does our seam become the Ossie mapping
   document? Who maintains the n-ary to OWL translation?
7. **LLM egress.** Is it allowed for naming and taxonomy signals? With which models, and under which
   EU-only constraint?
8. **The ADR-0005 inducer assets.** Where are the ConceptInducer, Taxonomy and AxiomInducer assets,
   and does any of them survive review against this design?
9. **External vocabularies.** Should schema.org be the default alignment target given its
   web-centric scope, or should none be the default? FIBO's licence and size need a separate check
   (**UNVERIFIED**).
10. **RDF 1.2 trigger.** Do we switch to Turtle 1.2 when RDF 1.2 reaches Recommendation, or when the
    major stores (Jena, RDF4J, Oxigraph) ship it by default?
