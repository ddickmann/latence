"""Write `vehicles.vocabulary.schemaorg.json`: the classes of schema.org the vehicle estate's
concepts can match, each mapped to one of the class tree's six top classes (map ticket 55; the
owner chose schema.org on 15 Nov).

    python3 schemaorg_extract.py <schemaorg-current-https.jsonld> > vehicles.vocabulary.schemaorg.json

The release file is `data/releases/30.1/schemaorg-current-https.jsonld` of the schema.org
repository. Every class below is checked to exist in it, and its own label is taken from it.
The extra labels are this extract's curation (the words the estate uses), listed per class.

Deliberately left out: schema.org's `Claim` is a factual claim (a CreativeWork), not a warranty
claim; it has no class for a complaint, a recall or a campaign; a part is a property there
(`isAccessoryOrSparePartFor`), not a class. A word with no fitting class gets none."""

import json
import sys

# (class, top class, labels the estate uses beside schema.org's own)
CURATED = [
    ("Organization", "organisation", []),
    ("AutoDealer", "organisation", ["Dealer", "car dealer"]),
    ("AutoRepair", "organisation", ["workshop", "repair shop"]),
    ("Person", "person", ["Owner"]),
    ("Place", "place", []),
    ("Country", "place", []),
    ("City", "place", []),
    ("Product", "product", []),
    ("Vehicle", "product", []),
    ("Car", "product", []),
    ("ProductModel", "product", ["Model"]),
    ("Brand", "product", ["Make"]),
    ("Event", "event", []),
    ("Report", "document", ["Recall report"]),
    ("Message", "document", ["Manufacturer communication"]),
    ("WarrantyPromise", "document", ["Warranty contract", "Warranty"]),
]
RELEASE = "30.1"
COMMIT = "9004a6654f69f067e69cfe5d0b3df2b8a4c86238"  # pragma: allowlist secret (a git commit id)

graph = json.load(open(sys.argv[1], encoding="utf-8"))["@graph"]
classes = {x["@id"]: x for x in graph if "rdfs:Class" in (x["@type"] if isinstance(x["@type"], list) else [x["@type"]])}
out = []
for name, top, extra in CURATED:
    found = classes.get(f"schema:{name}")
    if found is None:
        raise SystemExit(f"schema:{name} is no class of release {RELEASE}")
    own = str(found["rdfs:label"])
    out.append({"id": f"https://schema.org/{name}", "labels": [own, *extra], "top": top})
json.dump({
    "name": "schema.org (curated extract for the vehicle estate)",
    "version": RELEASE,
    "licence": "CC-BY-SA-3.0",
    "licence_source": (
        "https://schema.org/docs/terms.html, read from its source in the schema.org repository "
        f"(github.com/schemaorg/schemaorg, docs/terms.html at {COMMIT}) on 2026-11-15: "
        "\"The Sponsors' copyrights in the schema are licensed to website publishers and other "
        "third parties under the Creative Commons Attribution-ShareAlike License (version 3.0).\" "
        "schema.org itself was refused by the network of the machine that read it: to be "
        "confirmed against the live page"
    ),
    "attribution": "Schema.org vocabulary, by the schema.org sponsors (Google, Microsoft, Yahoo, Yandex); CC BY-SA 3.0. This extract is an adaptation: a selection of classes with added labels and a mapping to six top classes, under the same licence.",
    "source": {"release": RELEASE, "file": f"data/releases/{RELEASE}/schemaorg-current-https.jsonld",
               "repository": f"github.com/schemaorg/schemaorg@{COMMIT}", "read_on": "2026-11-15"},
    "classes": out,
}, sys.stdout, indent=1, ensure_ascii=False)
sys.stdout.write("\n")
