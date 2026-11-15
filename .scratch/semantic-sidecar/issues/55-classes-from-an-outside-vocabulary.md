# Build: classes from an outside vocabulary as a third voice

Type: task
Mode: AFK
Status: resolved (15 Nov): schema.org, the owner's choice, as a curated extract; open: the licence to be confirmed against the live terms page (a live step)
Blocked by: nothing
Emerges from: 42 (split out on 14 Nov: the third of its four parts)

## Question

The class tree has two voices: attribute patterns from the catalog, and a language model. Ticket
34 names an outside vocabulary (a published class hierarchy) as a third voice. Which vocabulary,
under which licence, and how does it vote?

## Goal

A voice that reads a vocabulary file (classes, labels, parents) given in the estate file, proposes
a parent class for a concept where a label or synonym of the concept matches a class of the
vocabulary, and votes like the other two (a disagreement is contested, never settled by the
voice alone). The vocabulary's licence is checked and recorded before it is used.

## Done check

- Tests on the toy estate with a small vocabulary file: a match proposes, a disagreement with
  the other voices is contested, no match proposes nothing.
- The vocabulary for the vehicle estate is chosen with its licence recorded.

## Result (15 Nov): the parts that need no choice

- **The voice** (`latence_ontology/vocabulary.py`, `classes.vocabulary` in the estate file:
  `{file: <path beside the estate file>, licences_allowed: [...]}`): a vocabulary file states
  its name, version, licence (SPDX id), where the licence was read (`licence_source`) and its
  classes, each with its id, labels and the top class of the tree it falls under (one of the
  six). For a concept whose label or object name matches a class's label (lower-cased, `_` as a
  space, a final plural `s` dropped) it proposes that class's top class, with the statement
  `vocabulary.matched` (the vocabulary and the class's label; the class's id as the reference).
  A label two classes of different top classes carry matches nothing. Matching is on words only,
  never on values.
- **It votes like the other two.** It never settles a placement alone: the order of offers is
  the model, the patterns, then the vocabulary, and a person's approval settles as before. Where
  it disagrees with another voice the placement is contested and names the other parent; where
  it agrees, both statements are kept and nothing is contested. Without a vocabulary the build
  is byte for byte as before (the same-bytes tests pass unchanged).
- **The licence is checked before use and recorded.** A file that states no licence, or one the
  estate file does not allow, is refused with a typed error naming the licence; by default only
  public-domain dedications and attribution-only licences are allowed (CC0-1.0, CC-BY-4.0,
  Apache-2.0, MIT, BSD-3-Clause, ODC-By-1.0, PDDL-1.0); a share-alike licence only where the
  estate file names it. The export records the vocabulary's name, version, licence, licence
  source and the file's digest (`classes.voices.vocabulary`).
- **Tests, red first** (`tests/test_ontology_vocabulary.py`, 3; red on the tree of 24's commit:
  no module `latence_ontology.vocabulary`): a match proposes (the claims, unplaced before, as an
  event); an agreement keeps both voices uncontested (the dealers); a disagreement is contested
  and the older voice's placement stands (the owners: person against organisation); no match, or
  a label of two top classes, proposes nothing; nothing changes without a vocabulary; a
  share-alike licence is refused unless allowed and then recorded, an unstated licence and a
  class under no top class are refused.

## Decided by the owner (15 Nov): option (b), schema.org

The question as it was put:

The vocabulary is a published class hierarchy whose licence travels with what the build
derives from it, so it is the owner's choice. The voice reads any of these once a file in the
shape above is made from it (a curated extract: the classes the estate's concepts can match,
each mapped to one of the six top classes).

- **(a) Recommended: a curated extract of Wikidata classes** (structured data under CC0-1.0, no
  attribution or share-alike duty; labels in many languages, which serves the German step
  later). Classes such as automobile manufacturer, car dealership, motor vehicle, product
  recall, warranty claim, vehicle part, each with its Wikidata id as the class id. Cost: someone
  curates the extract (the classes the estate needs, not the whole of Wikidata) and records the
  date it was read.
- **(b) schema.org** (Vehicle, Car, AutoDealer, AutoRepair, Organization, Product, Event): the
  vocabulary most tools already read, but published under CC BY-SA 3.0 (to be confirmed against
  schema.org's terms page when the file is made): share-alike, so it must be allowed in the
  estate file (`licences_allowed`) and its terms recorded in `THIRD-PARTY-LICENSES.md` before it
  ships with anything.
- **(c) No outside vocabulary for this estate** (the owner's position of 11 Nov): the class tree
  keeps its two voices; the third stays available for an estate that names one.

Recommendation (a): no licence duty, multilingual labels, ids that stay stable. Until the owner
chooses, the vehicle estate file names no vocabulary and builds as before.

## Result: schema.org for the vehicle estate (15 Nov)

- **The extract** (`estates/vehicles.vocabulary.schemaorg.json`, written by
  `estates/schemaorg_extract.py` from `data/releases/30.1/schemaorg-current-https.jsonld` of the
  schema.org repository at commit `9004a665`, read on 15 Nov): 16 classes the estate's concepts
  can match, each with its schema.org IRI as id, its own label from the release, the labels the
  estate uses, and one of the six top classes. The script checks each class exists in the
  release. Left out by meaning, not by name: schema.org's `Claim` is a factual claim, not a
  warranty claim; it has no class for a complaint, a recall or a campaign; a part is a property
  there, not a class.
- **The licence: CC BY-SA 3.0.** Read from the source of schema.org's terms page in its own
  repository (`docs/terms.html` at the same commit): "The Sponsors' copyrights in the schema are
  licensed to website publishers and other third parties under the Creative Commons
  Attribution-ShareAlike License (version 3.0)." **schema.org itself was refused by this
  machine's network, so the live page is to be confirmed (a live step):** open
  https://schema.org/docs/terms.html and check the same sentence. The file states the licence,
  its source and the attribution; the extract is an adaptation, offered under the same licence.
- **Allowed by the estate file only:** `classes.vocabulary.licences_allowed: [CC-BY-SA-3.0]` in
  `estates/vehicles.yaml`; the terms are recorded in `THIRD-PARTY-LICENSES.md` (a new section:
  outside vocabularies in estate files).
- **Tests, red first** (`tests/test_ontology_vehicle_estate.py`): the extract is refused without
  the estate's allowance and read with it (red before: no file, no `classes.vocabulary`); the
  third-party licence file records it (red before the section was written).
- **On the vehicle estate** (a build with the host's mirror): the vocabulary speaks on 9
  concepts. It agrees with the placement on 7 (vehicles, dealers, owners, warranty contracts,
  make, model, manufacturer communications) and sides with one voice on 2 that were already
  contested (recall reports: document; the internal vehicles: product). It contests nothing that
  was settled. The same-bytes test builds without the vocabulary (as the bytes were recorded),
  and still matches.
