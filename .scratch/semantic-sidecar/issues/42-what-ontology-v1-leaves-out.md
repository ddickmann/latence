# Build: what ontology v1 leaves out: document mentions, then SHACL and PROV-O exports

Type: task
Mode: AFK, after a decision on the order by the owner
Status: open, after v1
Decided by the owner (12 Nov): In which order follows what v1 leaves out, and when do the
grillings resume; Taken up after the demonstration
Blocked by: 34 (the decision that v1 is reached)
Emerges from: 34 ("What v1 does not hold, on purpose"), the map ("Destination", check 3; "Not
yet specified")

## Question

Ticket 34 lists what v1 does not hold on purpose: "Mentions from the estate's documents;
classes from an outside vocabulary; SHACL and PROV-O exports; a service route in the host; real
packages in place of the prototype folder." None of these had a ticket.

Against the destination:

- Check 3 asks that open-standard exports validate (OWL 2 QL, SHACL, SKOS). The Turtle file
  holds OWL 2 QL and SKOS; SHACL and PROV-O are not written.
- Modality (b), unstructured documents, has no part in the estate build: document mentions come
  after v1.
- The estate code lives in a prototype folder under `.scratch/`, not in packages.
- The service route is the host's to build (its own ticket).

## Goal

In the order the owner sets: mentions from the estate's documents linked to its entities; SHACL
and PROV-O exports beside the Turtle file; classes from an outside vocabulary as a third voice;
the prototype folder's code moved into packages under the repository's gates.

## Done check

Each of the four is done, or split into a ticket of its own on the map with its blockers, and
none is left only in ticket 34's last section.

## Decision brief for the owner (12 Nov)

- **Question:** In which order do the four things v1 leaves out follow, and when do the open
  grillings resume?
- **Why his:** ticket 42 is "after a decision on the order by the owner"; 13, 15 and 16 are
  HITL grillings; 08 needs his pod run.
- **Options:**
  - A. Decide now.
  - B. After the demonstration.
- **Facts:**
  - Ticket 42 (new, 12 Nov): document mentions, SHACL and PROV-O exports, an outside
    vocabulary, the prototype folder moved into packages.
  - Destination check 3 needs SHACL.
  - 13, 15, 16 and 08 are open.
- **Recommendation:** B. No recommendation on the order itself: it needs what the client reacts
  to in the Ontology chapter.
- **Urgency:** after v1.
- **Cost:** none now.
- **Decided by the owner (12 Nov):** Taken up after the demonstration.
- **Still open:** the order itself.
