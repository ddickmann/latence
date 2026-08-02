#!/usr/bin/env python3
"""Emit the gamma-calibration judgment file for the official MuSiQue-full export.

The judgments below are **LLM labels** (Claude Opus, 2026-08-02), read case-by-case off
`calibration-sample.jsonl` (the deterministic stratified sample produced by
`make_calibration_sample.py`). They are recorded here rather than hand-written into JSONL
so the label *and its reason* stay together and a human spot-check can argue with a
specific line. Pending human spot-check.

Rubric actually applied (spec 1.5 / ticket 09; the tau_text IS the evidence — there is no
gold chain file for this corpus):

* ``correct``  — every member is an entity the span genuinely mentions and reasonably
  typed, every relation triple is asserted by the span in the right direction.
* ``partial``  — the core extraction is right but one element of several is wrong
  (a bogus/mis-typed member among good ones, one unasserted relation among asserted ones),
  or a *central* entity of the span is missing altogether.
* ``incorrect`` — nothing survives: the sole member is not an entity / is a broken span,
  or every relation the Hyperedge asserts is unasserted or false.

Two structural facts of this export shaped the judgments, and both were verified against
`hyperedge.py` before labeling rather than assumed:

1. **A span with >``max_arity``(12) entities and no relations explodes into singleton
   Hyperedges** (``_arity_groups`` puts every relation-free entity in its own component).
   So an arity-1 Hyperedge over a 15-name cast table is a deliberate split, NOT a recall
   miss — its siblings carry the rest. `extracted_elsewhere_in_chunk` in the sample file
   records that context; "materially incomplete" was only charged when an entity central
   to the span is missing from the WHOLE chunk.
2. **Redundant relation labels over one asserted pair are not counted as errors** when
   both labels are semantically compatible (``created_by`` + ``author_of``, ``located_in``
   + ``part_of``). A label is charged only when it asserts something the span does not —
   ``capital_of``, ``spouse_of``, ``employer`` for non-employment, ``founded_by`` for
   non-founding, ``country_of`` onto a demonym.

Usage:  python3 benchmark/official-data/make_calibration_judgments.py musique-full
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# index in calibration-sample.jsonl -> (judgment, note)
# note is REQUIRED for partial/incorrect and omitted for correct.
JUDGMENTS: dict[int, tuple[str, str]] = {
    0: ("incorrect", "sole member 'study plans' is a mid-prose common noun promoted to a product; the span's actual subject (the university) is unnamed"),
    1: ("correct", ""),
    2: ("correct", ""),
    3: ("partial", "'election campaigns' is a common-noun event; the four named members (Red Shirts, Democratic, Confederates, 1872) are right"),
    4: ("correct", ""),
    5: ("partial", "'Samson &' is a title truncated at the span boundary, and most song titles the span lists (Morning Dew, Mama Tried, El Paso, The Red Rooster) are extracted nowhere"),
    6: ("incorrect", "the span's four languages (Greek, late Latin, French, English) are all extracted as LOCATIONS — systematic language->location mis-type; only 'asphalt' survives"),
    7: ("partial", "'primary campaign' is a common-noun event; Hillary Clinton and 2008 are right"),
    8: ("partial", "'census' is a common-noun event; the three places and the date are right"),
    9: ("correct", ""),
    10: ("correct", ""),
    11: ("incorrect", "sole member 'fossil-fuel burning' is a process phrase, not an entity"),
    12: ("partial", "'Dokyo' (a monk) is typed LOCATION and 'Buddhist institutions' is a common noun; Nara and the event are right"),
    13: ("partial", "'poetic adventure' and \"life's work\" are lifted out of the Nobel citation quote as creative works"),
    14: ("correct", ""),
    15: ("partial", "demonym 'American' typed LOCATION and 'private business jets' is a common noun; the four company/date members are right"),
    16: ("correct", ""),
    17: ("partial", "the pronoun 'He' is promoted to a PERSON node — a node that will conflate every unresolved male referent it merges with"),
    18: ("incorrect", "'Sandra' is a name fragment off the table's 'Bullock, Sandra' and becomes its own person node beside the sibling 'Sandra Bullock' — coreference failure"),
    19: ("correct", ""),
    20: ("incorrect", "sole member 'carnivals' is a common noun"),
    21: ("incorrect", "the title 'Breakfast at Tiffany's (film)' yields 'film' (common noun) + \"Tiffany's\" (title fragment); the actual work is never extracted"),
    22: ("incorrect", "'Madaris' (the plural of madrasa — schools) is typed PERSON"),
    23: ("correct", ""),
    24: ("partial", "the span's subject 'Pittsburgh Pipers' is extracted nowhere in the chunk, and 'league title' is a common-noun event"),
    25: ("partial", "'June' here is June Carter (a person, per 'I was June, and you were my Johnny Cash') but is typed DATE; Johnny Cash is not extracted"),
    26: ("partial", "media formats 'CD', 'LP' and 'digital download' promoted to product entities off a discography table; the albums, label and dates are right"),
    27: ("partial", "'ten-year-old person' is a generic noun phrase typed PERSON; Whitehead is right"),
    28: ("partial", "'Lord North' (a person) is typed ORGANIZATION; British Empire and Benjamin Labaree are right"),
    29: ("correct", ""),
    30: ("partial", "'invading' (a gerund) is promoted to an event entity; the two dukes and the two states are right"),
    31: ("correct", ""),
    32: ("partial", "'British expansion' is a descriptive phrase, not a named event"),
    33: ("partial", "the album 'After the Rain' is typed PRODUCT rather than creative work, and the band/song the span names are not extracted here"),
    34: ("correct", ""),
    35: ("correct", ""),
    36: ("partial", "'FDEs' (forensic document examiners — a profession) is typed ORGANIZATION; ASTM is right"),
    37: ("partial", "'muzzle-loading smoothbore guns' is a generic weapon description; 'Armstrong breech-loading guns' is a defensible product line"),
    38: ("correct", ""),
    39: ("correct", ""),
    40: ("incorrect", "the title 'Adams-Onis Treaty' is split at the en-dash into 'Adams'(person) + 'Onis Treaty'(event) — two malformed entities, and the treaty itself is never extracted"),
    41: ("correct", ""),
    42: ("partial", "demonym 'South Korean' typed LOCATION alongside the correct 'Korea'"),
    43: ("correct", ""),
    44: ("incorrect", "both members (\"Earth's surface\", 'bodies of water') are generic descriptions typed LOCATION; the span's subject (sedimentary rocks) is elsewhere"),
    45: ("correct", ""),
    46: ("partial", "'Saturday night' is part of the periodical title 'Saturday Night' but is typed DATE"),
    47: ("partial", "'forthcoming year' is a relative time phrase promoted to a date entity"),
    48: ("partial", "the span's subject star 'Kappa Fornacis' is extracted nowhere in the chunk; the two constellations are right"),
    49: ("partial", "the same date is emitted twice as two nodes ('October 27, 1986' and the infobox form '1986 - 10 - 27'), and the org name swallows the infobox word 'Agency'"),
    50: ("correct", ""),
    51: ("correct", ""),
    52: ("partial", "'public lecture' and 'his or her work' are anaphoric descriptions, not entities"),
    53: ("correct", ""),
    54: ("partial", "'Douglas County School District RE-1' is typed LOCATION and duplicates the org node 'Douglas County School District'; the span's subject Legend High School is missing from the chunk"),
    55: ("correct", ""),
    56: ("partial", "'Yuk Keep Smile' (a TV show) is typed EVENT; the four films are right"),
    57: ("incorrect", "'forested land' and 'steppelands' are generic descriptions typed LOCATION; the span's real entity (the Vyatichi) is not extracted"),
    58: ("correct", ""),
    59: ("correct", ""),
    60: ("correct", ""),
    61: ("partial", "'order' is an anaphoric definite description ('the order') promoted to an organization node that identifies nothing"),
    62: ("partial", "'cities' is a common noun typed LOCATION; the three Europe members are right"),
    63: ("partial", "'Game of Thrones' is typed ORGANIZATION rather than creative work"),
    64: ("partial", "the play 'Intimate Apparel' is typed PRODUCT, and the two films the span names (Midnight in Paris, Black Mass) are not extracted"),
    65: ("correct", ""),
    66: ("correct", ""),
    67: ("correct", ""),
    68: ("incorrect", "the member's canonical name is literally 'Pub\\n\\nThe London Gazette' — the mention runs across a title/body boundary and produces a garbage node"),
    69: ("incorrect", "'Sino-Tibetan' (a relation adjective off the title 'Sino-Tibetan relations during the Ming dynasty') is the sole member, typed LOCATION; support=22 means this node was reinforced 22x corpus-wide"),
    70: ("partial", "'Rover Triumph' is a division (org) typed PRODUCT and bare 'Rover' duplicates it; Austin Rover and the dates are right"),
    71: ("partial", "'central business district' is a common noun typed LOCATION, and the span's subject (General Artigas Central Station) is not extracted"),
    72: ("partial", "'Monday openers' is a common-noun event"),
    73: ("correct", ""),
    74: ("correct", ""),
    75: ("partial", "'Chamorro' (a language/people) is typed LOCATION; Guam is right"),
    76: ("partial", "the date is lifted out of a table with its footnote asterisk attached — the node is named '2002 *'"),
    77: ("correct", ""),
    78: ("partial", "the demonym adjectives 'Tasmanian' and 'South Australian' are typed LOCATION; Victoria and 2010 are right"),
    79: ("correct", ""),
    80: ("correct", ""),
    81: ("correct", ""),
    82: ("partial", "'double A-side physical single' is a common-noun description typed PRODUCT; the two years are right"),
    83: ("correct", ""),
    84: ("correct", ""),
    85: ("correct", ""),
    86: ("correct", ""),
    87: ("correct", ""),
    88: ("correct", ""),
    89: ("partial", "'within one months' is a relative duration (with the source's typo) promoted to a date entity"),
    90: ("correct", ""),
    91: ("correct", ""),
    92: ("partial", "'album' is an anaphoric description ('the album') promoted to a product node"),
    93: ("correct", ""),
    94: ("partial", "'Rudimental' (a band) is typed PERSON; the arity split itself is correct — the siblings carry the rest of the span"),
    95: ("partial", "'ordained priest' is a verb phrase promoted to an event entity; Galicia and the date are right"),
    96: ("correct", ""),
    97: ("correct", ""),
    98: ("correct", ""),
    99: ("correct", ""),
    100: ("correct", ""),
    101: ("correct", ""),
    102: ("correct", ""),
    103: ("correct", ""),
    104: ("partial", "the span's subject — the song 'The Lonely' — is extracted nowhere; the band is right"),
    105: ("partial", "the span's subject 'Yankee Stadium' is extracted nowhere in the chunk; the street, the year and the championships are right"),
    106: ("correct", ""),
    107: ("correct", ""),
    108: ("incorrect", "sole member 'south-west' is a compass direction typed LOCATION"),
    109: ("correct", ""),
    110: ("incorrect", "'Butler Detroit' welds two adjacent table cells (the schools 'Butler' and 'Detroit') into one organization node"),
    111: ("correct", ""),
    112: ("correct", ""),
    113: ("correct", ""),
    114: ("correct", ""),
    115: ("correct", ""),
    116: ("correct", ""),
    117: ("correct", ""),
    118: ("correct", ""),
    119: ("correct", ""),
    120: ("correct", ""),
    121: ("partial", "the span's subject 'Southside Speedway' is extracted nowhere in the chunk; 1959 and Chesterfield County are right"),
    122: ("correct", ""),
    123: ("correct", ""),
    124: ("correct", ""),
    125: ("correct", ""),
    126: ("incorrect", "'DJay Heath Ledger' welds the role 'DJay' to the next actor 'Heath Ledger' across a table row boundary — a person node for someone who does not exist"),
    127: ("correct", ""),
    128: ("correct", ""),
    129: ("correct", ""),
    130: ("correct", ""),
    131: ("correct", ""),
    132: ("correct", ""),
    133: ("correct", ""),
    134: ("correct", ""),
    135: ("correct", ""),
    136: ("correct", ""),
    137: ("correct", ""),
    138: ("correct", ""),
    139: ("correct", ""),
    140: ("correct", ""),
    141: ("correct", ""),
    142: ("correct", ""),
    143: ("correct", ""),
    144: ("partial", "'Barcelona' here denotes the football club but is typed LOCATION and mis-linked onto the city node; Neymar and Dani Alves are right"),
    145: ("correct", ""),
    146: ("correct", ""),
    147: ("correct", ""),
    148: ("partial", "the span's subject — the song 'Ningyo'/'人魚' — is not extracted; Avex Trax is right"),
    149: ("correct", ""),
    150: ("correct", ""),
    151: ("correct", ""),
    152: ("correct", ""),
    153: ("correct", ""),
    154: ("correct", ""),
    155: ("correct", ""),
    156: ("correct", ""),
    157: ("correct", ""),
    158: ("correct", ""),
    159: ("correct", ""),
    160: ("correct", ""),
    161: ("correct", ""),
    162: ("correct", ""),
    163: ("correct", ""),
    164: ("correct", ""),
    165: ("correct", ""),
    166: ("correct", ""),
    167: ("correct", ""),
    168: ("correct", ""),
    169: ("correct", ""),
    170: ("partial", "the spouse_of triple is genuinely asserted ('her husband Mike Kanellis'), but 'WWE debut appearance' is a common-noun event member"),
    171: ("partial", "'The Bold and the Beautiful -[directed by]-> CBS' is unasserted — CBS is the broadcaster, not a director; the Brooke Logan role triple is fine"),
    172: ("partial", "'Spain -[spouse of]-> France' is unasserted nonsense; 'Catalonia located_in Spain' is right, and 'French' is a demonym typed LOCATION"),
    173: ("correct", ""),
    174: ("correct", ""),
    175: ("partial", "she WON the Rapallo Carige Prize; the extraction inverts it into 'Prize -[directed by]-> Ballestra' and 'Prize -[founded by]-> Ballestra'. The birthplace triple is right"),
    176: ("partial", "'Labour government -[founded by]-> Clement Attlee' is unasserted (he led it); the directed_by triple matches 'led by'"),
    177: ("correct", ""),
    178: ("correct", ""),
    179: ("incorrect", "'Billy Hill (band)' is a disambiguated title: the parenthetical 'band' becomes an organization node and the band's own name becomes a PERSON that is then made a 'member of' it — a relation manufactured out of a disambiguation suffix"),
    180: ("partial", "'Adham ar n-athair uile -[created by]-> Adhamh Cuisin' mis-assigns the poem to the scribe who penned it into the manuscript; the commissioning triples are loose but asserted"),
    181: ("partial", "the employer triple is right, but the title form 'Fadel Brahami' and the body form 'Mohamed Fadel Brahami' become two person nodes for one man"),
    182: ("incorrect", "the Hyperedge's only relation is 'Gulf of Mexico -[capital of]-> United States' — unasserted and false"),
    183: ("incorrect", "the only relation, 'Severus -[employer]-> Emperor Pertinax', is unasserted — the span says Severus seized power AFTER Pertinax died"),
    184: ("incorrect", "both relations ('Union of the Crowns directed_by / founded_by James IV') are unasserted: the span says the 1503 marriage led to the Union a century later"),
    185: ("partial", "'Let It Be -[directed by]-> George Harrison' is unasserted (it is a Beatles album); 'Harrison member_of the Beatles' is right"),
    186: ("correct", ""),
    187: ("partial", "only 'Lodge member_of Republican Party' is asserted; 'Republican Party employer US Senate' (the span says controlled), 'Republican Party founded_by Woodrow Wilson' (he was a Democrat) and 'Wilson author_of Versailles conference' are all fabricated"),
    188: ("partial", "'Lord Lytton member_of Viceroy's Council' is an inference the span does not make (he proposed the act, the Council passed it); the located_in triple is loose but supportable"),
    189: ("partial", "the author_of triple is right, but 'Symphony No.' is a truncated duplicate of 'Symphony No. 1 in C major' and 'Swedish song'/'lieder' are common nouns"),
    190: ("partial", "'San Francisco -[capital of]-> California' is false and unasserted; the located_in triple is supportable"),
    191: ("incorrect", "the title 'Walker, Texas Ranger' yields four fabricated triples at once — Walker capital_of / country_of / located_in / member_of Texas — none of them asserted by anything but the comma in a show title"),
    192: ("partial", "'Theo van den Boogaard -[country of]-> Dutch' points a country relation at a demonym node; the Castricum birthplace triple is right"),
    193: ("partial", "'Reflections on the Revolution in France -[directed by]-> Edmund Burke' uses a film label for authorship; the underlying link is asserted"),
    194: ("partial", "'Emancipation Proclamation -[founded by]-> President Lincoln' is a category error; the author_of triple is right"),
    195: ("partial", "'Ottoman sultanate created_by / founded_by Murad I' is unasserted — the span says his conquests TRANSFORMED an existing sultanate; the 'conquests directed_by Murad I' triple is right"),
    196: ("incorrect", "both relations put William Childress INSIDE the Korean War ('member of', 'part of'); the span says he is regarded as a poet OF that war"),
    197: ("partial", "'U-751 -[employer]-> Kriegsmarine' is a category error (a boat cannot employ); the member_of triple carries the asserted 'built for' link"),
    198: ("incorrect", "'France -[country of]-> Japan' and 'France -[located in]-> Japan' invert a tour itinerary title into geography that is simply false"),
    199: ("incorrect", "both relations make Merlin Network part of YouTube; the span says the two reached a deal"),
    200: ("partial", "the landslide/Nainital triple is right, but 'north end of the town' is a descriptive phrase promoted to a location node"),
    201: ("partial", "'Ferraro -[capital of]-> United States' is nonsense and the title/body name variants become two person nodes; the House employer and Democratic Party member triples are right"),
    202: ("partial", "'Martin -[directed by]-> Peter Dinklage' is a category error, and the performed_by link itself is an infobox artifact — 'Martin' here is the author George R. R. Martin, not the character Dinklage plays"),
    203: ("correct", ""),
    204: ("partial", "'Bessie Higgenbottom -[spouse of]-> Hilary' turns her MOTHER into a spouse; the San Francisco triple is right"),
    205: ("partial", "the Maya Lin authorship and the Washington location are right, but both memorials are typed EVENT and the anaphoric 'monument' becomes a creative-work node"),
    206: ("partial", "the NRHP listing triples are right, but 'John W. Ferguson House' (a building) is typed PERSON, and 'demolished'/'added to the National Register of Historic Places' are verb phrases promoted to events"),
    207: ("partial", "the three containment triples are right, but 'Jimma Arjo' (a woreda) is typed PERSON"),
    208: ("correct", ""),
    209: ("partial", "'Musafir -[located in]-> Vidhan Parishad' uses a geographic label for sitting in a legislative chamber, which is also typed LOCATION"),
    210: ("partial", "'MLB -[founded by]-> Felipe Alou' is nonsense (he was promoted to manager); the employer triple is supportable and 'Dominican' is a demonym typed LOCATION"),
    211: ("correct", ""),
    212: ("incorrect", "the only relation, 'Tommy -[spouse of]-> Marty', reads loyalty between a boxer and his manager as a marriage"),
    213: ("incorrect", "the only relation, 'Defiance Campaign -[directed by]-> Mandela', is not asserted — the span names 'Mandela's Youth League' and the Campaign side by side"),
    214: ("incorrect", "'Southon -[author of]-> the Book of Exodus' is manufactured out of the comma list 'Southon, and the Book of Exodus.'; the Book is also duplicated as an EVENT node 'Exodus'"),
    215: ("partial", "the residence and List-society triples are right; 'Guido von List -[employer]-> Eberhard von Brockhusen' is unasserted (List was travelling to his house)"),
    216: ("partial", "'Division of Labor -[part of]-> Ministry of Employment and Labor' mis-models a PREDECESSOR agency as a component; the country/location and direction triples are right"),
    217: ("correct", ""),
    218: ("partial", "'Milan -[capital of]-> Italy' is false and 'Real Madrid -[country of]-> Italy' assigns a Spanish club to the venue's country; the San Siro/Milan triple is right"),
    219: ("correct", ""),
    220: ("incorrect", "the only relation, 'Superior Drilling Company -[employer]-> farming implements', makes a product into an employer; the span asserts manufacture"),
    221: ("incorrect", "the only relation, 'Bob Lemon -[member of]-> Major League Baseball', is not asserted — MLB appears in the span only as the holder of a record"),
    222: ("partial", "'Golden Bauhinia Square -[capital of]-> Wan Chai' is nonsense and 'Tin Hau -[employer]-> Marine Department' inverts 'managed by' (Tin Hau, a vessel, is also typed LOCATION)"),
    223: ("partial", "'Lambeau Field -[capital of]-> United States' is nonsense; the located_in triple is supportable"),
    224: ("correct", ""),
    225: ("partial", "the Palermo/Sicily triple is right, but 'Phoenicians' (a people) is typed PERSON"),
    226: ("correct", ""),
    227: ("incorrect", "the only relation, 'Labelle -[member of]-> Billboard Hot 100', reads 'held the number-one spot on the chart' as membership"),
    228: ("correct", ""),
    229: ("incorrect", "all three relations are fabricated: 'Israel founded_by Chileab', 'Chileab capital_of Israel', 'Chileab located_in Israel' — the span is about succession to a throne"),
    230: ("correct", ""),
    231: ("partial", "'Azarbaijan Shahid Madani University -[founded by]-> 1987' puts a DATE in the founder slot — a type-violating triple; the Tabriz location triple is right"),
    232: ("correct", ""),
    233: ("correct", ""),
    234: ("partial", "eight of the nine triples are right (creators, cast, setting); 'Will & Grace -[directed by]-> David Kohan' is unasserted — he created it"),
    235: ("correct", ""),
    236: ("correct", ""),
    237: ("incorrect", "the only relation, 'Newcastle -[capital of]-> New South Wales', is false and unasserted — the span says Hannell was the first Mayor of Newcastle"),
    238: ("incorrect", "all three relations tie an ENGLISH singer to the United States (capital_of, country_of, located_in) off the phrase 'especially in the United States'; 'Meat Loaf' is also typed ORGANIZATION"),
    239: ("correct", ""),
    240: ("correct", ""),
    241: ("correct", ""),
    242: ("correct", ""),
    243: ("incorrect", "both relations claim the Polish government created/founded Dell; the span says subsidies KEPT Dell — and gamma is 0.84"),
    244: ("incorrect", "'Christmas Club -[founded by]-> Arthur' is unasserted — Arthur withdrew the club's money and faked a burglary; gamma 0.86"),
    245: ("correct", ""),
    246: ("correct", ""),
    247: ("correct", ""),
    248: ("correct", ""),
    249: ("incorrect", "'Mr. Deeds -[directed by]-> Adam Sandler' misreads 'the 2002 Adam Sandler motion picture' — he starred, he did not direct. This is the HIGHEST-gamma Hyperedge in the sample (0.9325)"),
}


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "musique-full"
    base = ROOT / dataset
    sample = [json.loads(line) for line in (base / "calibration-sample.jsonl").open()]

    if len(sample) != len(JUDGMENTS):
        raise SystemExit(f"sample has {len(sample)} rows, table has {len(JUDGMENTS)}")
    # Pin the sample the labels were read against: if the export or the sampling rule
    # changes, these labels no longer describe these Hyperedges and must be re-read.
    if round(sample[0]["gamma"], 6) != 0.25 or sample[0]["hyperedge_id"] != "0edef61a916c07fc":
        raise SystemExit("sample drifted from the one these judgments were read against")

    out = base / "calibration-judgments.jsonl"
    counts: dict[str, int] = {"correct": 0, "partial": 0, "incorrect": 0}
    with out.open("w", encoding="utf-8") as fh:
        for row in sample:
            judgment, note = JUDGMENTS[row["i"]]
            if judgment not in counts:
                raise SystemExit(f"bad judgment {judgment!r} at index {row['i']}")
            if judgment != "correct" and not note:
                raise SystemExit(f"non-correct judgment without a note at {row['i']}")
            counts[judgment] += 1
            record = {"hyperedge_id": row["hyperedge_id"], "judgment": judgment}
            if note:
                record["note"] = note
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"wrote {len(sample)} judgments -> {out}")
    for k, v in counts.items():
        print(f"  {k}: {v} ({v / len(sample):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
