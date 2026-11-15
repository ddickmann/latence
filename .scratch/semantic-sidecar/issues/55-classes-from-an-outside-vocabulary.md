# Build: classes from an outside vocabulary as a third voice

Type: task
Mode: AFK
Status: the voice is built (15 Nov); open for the owner: which vocabulary for the vehicle estate
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

## Open for the owner: which vocabulary for the vehicle estate

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
