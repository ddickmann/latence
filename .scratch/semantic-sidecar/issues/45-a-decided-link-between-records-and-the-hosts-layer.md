# Question: should a link between records that a person decided reach the host's layer?

Type: question
Mode: HITL (the host's owner decides what enters its semantic layer)
Status: resolved (14 Nov): yes; this side's writer built behind version 2 of the file (14 Nov); open: the host's ticket for the entry it reads
Decided by the owner (14 Nov): yes: a person-decided link reaches the host's layer as a new entry shape, agreed on the host's side first (a host ticket for the entry it reads, then the writer here). Links certified on simulated labels are never written, as settled.
Blocked by: nothing
Emerges from: 40 ("Choices": no link between records is ever written, a person's included)

## Question

The proposal file carries two kinds of entry from this side: two spellings that are one value,
and the names of a master list's records. A link between two records (a dealer and a customer
account that are the same organisation) is written in neither, also where a person approved
it: the host's names entry keeps one origin per record, so a name decided through a link has
no place in it. A link certified on simulated labels must never be written; that part is
settled and tested.

What is open: whether an approved link is of use to the host at all (one name for both
records, a join the layer may take), and in which shape it would arrive.

## Goal

A decision by the host's owner, recorded here: no, with the reason; or yes, with the entry's
shape agreed on the host's side first.

## Done check

- The decision is written here and, if yes, the host has a ticket for the entry it would read.

## Proposal for the host's side: the entry a decided link arrives as (14 Nov)

For the host's ticket that settles what its tool reads. This side builds the writer behind a
version of the file; nothing reaches the host before its tool reads the version.

- **The file says its version.** A file that holds a link entry starts with `version: "2"`
  beside `source: "ontology"`. A file without `version` is version 1, today's shape, and is
  written byte for byte as before. A reader that does not know a version refuses the whole file
  by name, so a link is never dropped in silence.
- **The entry, one per link a person approved:**

  ```yaml
  - links:
      records:
        - object: "internal.dealers"        # a catalog object, as <schema>.<table>
          column: "dealer_id"               # its key column (a single-column key only)
          key: "D0042"
          name: "Dealer 42 GmbH"            # the record's own name in its object
        - object: "partner.service_accounts"
          column: "account_id"
          key: "SA0187"
          name: "D42 Service GmbH"
    origin:
      element: "same:internal.dealers:D0042~partner.service_accounts:SA0187"
      claim: "<16 hex digits>"
      decision:
        verdict: "approved"
        by: "<person id>"
        at: "<YYYY-MM-DD>"
  ```

  The two records in the order of the pair's id (the master list's first, where there is one).
  Each carries its own name, so the host may give both records one name or keep both; which,
  and whether a join on the two keys enters the layer, is the host's to decide.
- **What it says, and what it does not:** a person decided that these two records are one
  organisation, on what the pair stated on that day. It says nothing about any other record of
  either cluster: a link the machine accepted, or a record the machine put in the same cluster,
  is in no entry. The whole crosswalk of the engine stays where ADR-0070 puts it, in the
  catalog's tables; this entry carries only links a person decided.
- **Never written:** a link the machine accepted and no person approved; a link certified on
  simulated labels (no such link is a person's decision, so none is written); a rejected link
  (a cannot-link stays on this side, in the ledger); a pair that holds a seeded record; a pair
  decided on another statement than the build now makes (stale); a decision without a claim or
  a day; a record of an object whose key has more than one column.
- **Asked of the host's ticket:** whether it reads version 2, and what a placed link becomes in
  its library (one name, two names, a join); the receipt names the entry as today
  (`layer-receipt/1`, `placed` by element and claim), so `in_layer` needs no change.

## Result (14 Nov): the writer, behind version 2

- **`latence-proposal-writer`** writes the file in versions 1 and 2 (`VERSIONS`,
  `SCHEMA_VERSION = 2`). Version 1 is the default and is today's file, byte for byte (the
  vehicle estate's file of the 12 Nov trial is unchanged). Version 2 states `version: "2"` and
  adds a `links` entry, in the shape proposed above, for every link between two records a
  person approved. The package ships both schemas (`schema(1)`, `schema(2)`); a version-2 file
  does not validate against version 1. Any other version is refused (`ConfigError`).
- **The estate** asks for version 2 with `propose.version: 2` in the estate file; the vehicle
  estate stays at 1 until the host's tool reads 2.
- **What a link is checked on.** A person's decision is written only on the statement the build
  makes now. While the pair is listed for review that is the review item's claim; once a build
  applied the person's answer, the pair is no longer listed, so the build now writes the pair's
  claim on the link a person decided (`same_as` with `decided_by` and `claim`; the export
  schema states it, `graph-export/3` keeps its number for the reason given in ticket 44). A
  decided link whose edge states no claim is not written (`not_in_the_build`).
- **Never written, each tested:** a link the machine accepted (the yard's three, certified on
  simulated labels), a pair that holds a seeded record, a stale decision, one without a day; in
  version 1, any link (counted under `not_carried` as before).
- **On the vehicle estate** every link holds a seeded service account, so version 2 would add
  no link there today; the first real use is an estate with two real sources.
- **Tests, red first:** in the package, `test_links.py` (3; red: the package had no version);
  in the prototype,
  `test_a_link_a_person_decided_between_records_that_are_not_seeded_is_written_in_version_2_only`
  (red: the person's link carried no claim, `KeyError: 'claim'`).
- **For the orchestrator:** the host's ticket for the entry it reads is to be written on the
  host's side from the proposal above; until it reads version 2, nothing here changes what the
  host receives.
