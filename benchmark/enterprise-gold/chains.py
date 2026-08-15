# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501 - a data table: one fact / one question per line, kept readable as a row.
"""The planted fact graph and the question chains derived from it (ticket 21, part D).

This module is the *source* of both ``gold.jsonl`` and ``CHAINS.md`` — build_gold.py
resolves every :data:`FACTS` sentence to its exported chunk + char offsets and writes both
files, so the honesty artifact and the gold set can never drift apart.

A ``FACTS`` entry is ``id -> (document number, the verbatim asserting sentence)``. The
document number is the ``NN`` prefix of the corpus file name; the sentence is matched
whitespace-insensitively against the EXPORTED chunk text (source line wrapping and the
markdown chunker's markup stripping both normalise away), and the matched substring is
what lands in the gold file — so ``corpus_text[start:end] == text`` holds by construction.

A ``QUESTIONS`` entry is ``(question, hop_count, answer, [fact ids])``. Invariants that
build_gold.py enforces, and that a reviewer can re-check:

* ``len(fact ids) == hop_count`` — one marked span per hop;
* for ``hop_count >= 2`` every fact sits in a DIFFERENT document, so the question cannot be
  answered from any single document;
* the questions never name the intermediate entities, only the anchor and the roles, so
  there is no lexical shortcut from question to answer document;
* the ``answer`` is recorded for review but is NOT part of the harness format (the gold
  format is supporting-fact recall: question + hop_count + spans).
"""

from __future__ import annotations

# -- the planted facts ------------------------------------------------------------------
# id -> (corpus document number, verbatim asserting sentence)
FACTS: dict[str, tuple[str, str]] = {
    # corporate events
    "ACQ_MERIDIAN_NORVIK": ("03", "Meridian Systems AG acquired Norvik Tools GmbH in March 2021."),
    "ACQ_MERIDIAN_ZEPHYRINE": ("12", "Meridian Systems AG übernahm Zephyrine Labs GmbH im Juni 2023."),
    "ACQ_ARDENT_CENDRIA": ("05", "Ardent Vale Ltd acquired Cendria Software Ltd in October 2022."),
    "ACQ_QUINTARA_DALGREN": ("18", "Quintara Energy plc acquired Dalgren Maritime AS in December 2020."),
    "ACQ_VANTOR_LORQUIN": ("22", "Vantor Stahlwerke AG übernahm Lorquin Metals SpA im Oktober 2019."),
    "ACQ_FERRIX_NYQUIST": ("27", "Ferrix Automation BV acquired Nyquist Sensors Oy in February 2024."),
    "ACQ_TOLMACH_RAVENPORT": ("33", "Tolmach Bergbau AG acquired Ravenport Terminals Ltd in May 2022."),
    "ACQ_HALCYON_MARNIX": ("36", "Halcyon Verpackung GmbH acquired Marnix Textil BV in July 2023."),
    "MRG_BRIGHTMOOR_ELLSWORTH": ("15", "Brightmoor Analytics Ltd merged with Ellsworth Diagnostics Inc in September 2023."),
    "INV_OAKHURST_PELMAR": ("09", "Oakhurst Capital Partners invested in Pelmar Robotics SA in 2022."),
    "INV_OAKHURST_ARDENT": ("30", "Oakhurst Capital Partners invested in Ardent Vale Ltd during the quarter."),
    # supply relationships
    "SUP_NORVIK_KALTHOFF": ("04", "Norvik Tools GmbH supplies gearboxes to Kalthoff AG under the frame agreement named above."),
    "SUP_KALTHOFF_PELMAR": ("07", "Kalthoff AG beliefert Pelmar Robotics SA seit 2018."),
    "SUP_ZEPHYRINE_WENDLIN": ("14", "Zephyrine Labs GmbH supplies Wendlin Pharma AG with coating material under this agreement."),
    "SUP_NYQUIST_TESSARO": ("20", "Nyquist Sensors Oy supplies Tessaro Logistik AG with sensor assemblies for onward distribution."),
    "SUP_SOLVEX_HALCYON": ("11", "Solvex Chemie GmbH beliefert Halcyon Verpackung GmbH im Rahmen eines Zweijahresvertrags."),
    "SUP_LORQUIN_BREKKER": ("24", "Lorquin Metals SpA supplies Brekker Antriebstechnik GmbH with forged blanks."),
    "SUP_BREKKER_NORVIK": ("26", "Brekker Antriebstechnik GmbH supplies Norvik Tools GmbH with drive units for the gearbox assembly line."),
    "SUP_MARNIX_PELMAR": ("37", "Marnix Textil BV supplies Pelmar Robotics SA with technical fabric for protective covers."),
    "SUP_CENDRIA_QUINTARA": ("29", "Cendria Software Ltd supplies Quintara Energy plc with monitoring software and the associated support services described in Schedule 1."),
    "SUP_RAVENPORT_DALGREN": ("34", "Ravenport Terminals Ltd supplies Dalgren Maritime AS with terminal handling and storage services under a five-year contract."),
    "SUP_TESSARO_KALTHOFF": ("31", "Tessaro Logistik AG beliefert Kalthoff AG mit Bauteilen aus dem Zentrallager."),
    "SUP_ELLSWORTH_WENDLIN": ("38", "Ellsworth Diagnostics Inc supplies Wendlin Pharma AG with assay reagents under the running agreement."),
    # products
    "MAN_NORVIK_TARNEX": ("02", "Norvik Tools GmbH fertigt das Tarnex-Getriebe."),
    "MAN_BREKKER_VORTAN": ("25", "Brekker Antriebstechnik GmbH fertigt den Vortan 400 Servoantrieb."),
    "MAN_SOLVEX_HALOX": ("10", "Solvex Chemie GmbH manufactures the Halox membrane."),
    "MAN_FERRIX_CORVID": ("28", "Ferrix Automation BV manufactures the Corvid X2 controller."),
    "MAN_ZEPHYRINE_SELENTA": ("13", "Zephyrine Labs GmbH fertigt die Selenta-Beschichtung."),
    "MAN_NYQUIST_NADIR": ("21", "Nyquist Sensors Oy manufactures the Nadir 12 sensor array."),
    "MAN_KALTHOFF_BRAXIS": ("08", "Kalthoff AG manufactures the Braxis hydraulic pump."),
    "MAN_HALCYON_PYRAND": ("35", "Halcyon Verpackung GmbH fertigt die Pyrand-Dämmplatte."),
    "MAN_CENDRIA_MARLIN": ("06", "Cendria Software Ltd produces the Marlin 7 telemetry unit."),
    "MAN_PELMAR_LARENT": ("09", "Pelmar Robotics SA manufactures the Larent handling arm."),
    "MAN_WENDLIN_AMBREL": ("14", "Wendlin Pharma AG manufactures the Ambrel dosing kit."),
    "MAN_VANTOR_KESTREL": ("22", "Vantor Stahlwerke AG fertigt das Kestrel-Profil."),
    "MAN_TOLMACH_VASKAL": ("32", "Tolmach Bergbau AG produziert das Vaskal-Konzentrat."),
    # seats
    "LOC_MERIDIAN": ("01", "Meridian Systems AG is headquartered in Düsseldorf."),
    "LOC_NORVIK": ("02", "Norvik Tools GmbH hat ihren Sitz in Bratislava."),
    "LOC_KALTHOFF": ("07", "Kalthoff AG hat ihren Sitz in Leipzig."),
    "LOC_DALGREN": ("19", "Dalgren Maritime AS hat ihren Sitz in Trondheim."),
    "LOC_PELMAR": ("09", "Pelmar Robotics SA is based in Porto."),
    "LOC_BRIGHTMOOR": ("16", "Brightmoor Analytics Ltd is based in Lyon."),
    "LOC_TOLMACH": ("32", "Tolmach Bergbau AG hat ihren Sitz in Kattowitz."),
    "LOC_NYQUIST": ("21", "Nyquist Sensors Oy is based in Tampere."),
    "LOC_LORQUIN": ("23", "Lorquin Metals SpA is based in Genoa."),
    "LOC_RAVENPORT": ("34", "Ravenport Terminals Ltd is based in Rotterdam."),
    "LOC_ARDENT": ("05", "Ardent Vale Ltd is based in Leeds."),
    "LOC_CENDRIA": ("06", "Cendria Software Ltd is based in Coventry."),
    "LOC_SOLVEX": ("10", "Solvex Chemie GmbH is based in Mannheim."),
    "LOC_ZEPHYRINE": ("13", "Zephyrine Labs GmbH hat ihren Sitz in Jena."),
    "LOC_WENDLIN": ("14", "Wendlin Pharma AG is based in Heidelberg."),
    "LOC_QUINTARA": ("18", "Quintara Energy plc is headquartered in Aberdeen."),
    "LOC_VANTOR": ("22", "Vantor Stahlwerke AG hat ihren Sitz in Duisburg."),
    "LOC_BREKKER": ("25", "Brekker Antriebstechnik GmbH hat ihren Sitz in Bielefeld."),
    "LOC_FERRIX": ("27", "Ferrix Automation BV is based in Utrecht."),
    "LOC_OAKHURST": ("30", "Oakhurst Capital Partners is based in London."),
    "LOC_TESSARO": ("31", "Tessaro Logistik AG hat ihren Sitz in Hamburg."),
    "LOC_HALCYON": ("35", "Halcyon Verpackung GmbH hat ihren Sitz in Bremen."),
    "LOC_MARNIX": ("37", "Marnix Textil BV is based in Eindhoven."),
    "LOC_ELLSWORTH": ("38", "Ellsworth Diagnostics Inc is based in Boston."),
    "LOC_AURENTA": ("39", "Aurenta Insurance Group hat ihren Sitz in Zürich."),
    # people
    "EMP_BRANDTNER_MERIDIAN": ("03", "Ilse Brandtner works at Meridian Systems AG and has led group finance since the 2021 reorganisation."),
    "EMP_VOSSKAMP_NORVIK": ("04", "Henrik Vosskamp works at Norvik Tools GmbH in the key accounts team and answers questions on this notice."),
    "EMP_CENDRILLO_KALTHOFF": ("08", "Marta Cendrillo works at Kalthoff AG and takes over production planning from the end of March."),
    "EMP_REINHARD_BREKKER": ("26", "Tobias Reinhard works at Brekker Antriebstechnik GmbH and looks after the account named above."),
    "EMP_OKONKWO_BRIGHTMOOR": ("16", "Adaeze Okonkwo works at Brightmoor Analytics Ltd and coordinates the site working group."),
    "EMP_LIDBERGH_DALGREN": ("19", "Sven Lidbergh arbeitet bei Dalgren Maritime AS und leitet die Technik am Standort."),
    "EMP_FALKNER_SOLVEX": ("11", "Renata Falkner arbeitet bei Solvex Chemie GmbH und betreut den Vertrag auf unserer Seite."),
    "EMP_PETRIC_ZEPHYRINE": ("13", "Jonas Petric arbeitet bei Zephyrine Labs GmbH und führt die Linienprotokolle."),
    "EMP_AUBERTIN_PELMAR": ("09", "Camille Aubertin works at Pelmar Robotics SA and is the committee's day-to-day contact for reporting questions."),
    "EMP_STRAUB_WENDLIN": ("14", "Miriam Straub works at Wendlin Pharma AG and is the contact for all technical questions arising under this agreement."),
    "EMP_KANDEMIR_TESSARO": ("20", "Yusuf Kandemir works at Tessaro Logistik AG and handles dispatch queries under this arrangement."),
    "EMP_MARCHETTI_LORQUIN": ("23", "Elena Marchetti works at Lorquin Metals SpA and chairs the energy working group."),
    "EMP_ZAWADZKI_TOLMACH": ("32", "Piotr Zawadzki leitet Tolmach Bergbau AG und berichtet dem Aufsichtsrat quartalsweise."),
    "EMP_LOEWE_HALCYON": ("35", "Hannah Loewe arbeitet bei Halcyon Verpackung GmbH und verantwortet die Linienplanung."),
    "EMP_MALIN_FERRIX": ("28", "Gregor Malin works at Ferrix Automation BV and answers application questions on the controller range."),
    "EMP_ANDRADE_CENDRIA": ("06", "Sofia Andrade works at Cendria Software Ltd and will chair the next two sessions while the product director is on leave."),
    "EMP_EKSTRAND_NYQUIST": ("21", "Nils Ekstrand works at Nyquist Sensors Oy and owns the qualification plan for the extension."),
    "EMP_HOMANN_VANTOR": ("22", "Beatrix Homann arbeitet bei Vantor Stahlwerke AG und beantwortet Fragen zur Integration."),
    "EMP_VESPUCCI_QUINTARA": ("18", "Dario Vespucci works at Quintara Energy plc and coordinates the reporting workstream for the transaction."),
    "EMP_ZERHOUNI_OAKHURST": ("30", "Fatima Zerhouni works at Oakhurst Capital Partners and covers the industrials allocation."),
    # partnerships
    "PAR_MERIDIAN_FERRIX": ("17", "Meridian Systems AG partnered with Ferrix Automation BV on a joint development programme for drive electronics."),
    "PAR_AURENTA_TESSARO": ("39", "Aurenta Insurance Group kooperiert mit Tessaro Logistik AG bei der Absicherung von Transportrisiken."),
    "PAR_QUINTARA_VANTOR": ("40", "Quintara Energy plc partnered with Vantor Stahlwerke AG on low-carbon process heat."),
}

# -- the questions ----------------------------------------------------------------------
# (question, hop_count, answer-for-review, [fact ids in chain order])
QUESTIONS: list[tuple[str, int, str, list[str]]] = [
    # ---------------- 1 hop (20) ----------------
    ("In which city does Meridian Systems AG have its registered seat?", 1, "Düsseldorf", ["LOC_MERIDIAN"]),
    ("Which gearbox does Norvik Tools GmbH build?", 1, "Tarnex-Getriebe", ["MAN_NORVIK_TARNEX"]),
    ("Which toolmaker did Meridian Systems AG buy in March 2021?", 1, "Norvik Tools GmbH", ["ACQ_MERIDIAN_NORVIK"]),
    ("Which customer receives gearboxes from Norvik Tools GmbH under frame agreement KA-2019-44?", 1, "Kalthoff AG", ["SUP_NORVIK_KALTHOFF"]),
    ("Which software house did Ardent Vale Ltd buy in October 2022?", 1, "Cendria Software Ltd", ["ACQ_ARDENT_CENDRIA"]),
    ("For which company does Sofia Andrade work?", 1, "Cendria Software Ltd", ["EMP_ANDRADE_CENDRIA"]),
    ("In which city is Kalthoff AG seated?", 1, "Leipzig", ["LOC_KALTHOFF"]),
    ("Which pump does Kalthoff AG build?", 1, "Braxis hydraulic pump", ["MAN_KALTHOFF_BRAXIS"]),
    ("In which robotics company did Oakhurst Capital Partners take a position in 2022?", 1, "Pelmar Robotics SA", ["INV_OAKHURST_PELMAR"]),
    ("Which membrane does Solvex Chemie GmbH make?", 1, "Halox membrane", ["MAN_SOLVEX_HALOX"]),
    ("Which packaging company does Solvex Chemie GmbH deliver to under a two-year contract?", 1, "Halcyon Verpackung GmbH", ["SUP_SOLVEX_HALCYON"]),
    ("Which laboratory business did Meridian Systems AG take over in June 2023?", 1, "Zephyrine Labs GmbH", ["ACQ_MERIDIAN_ZEPHYRINE"]),
    ("For which company does Jonas Petric work?", 1, "Zephyrine Labs GmbH", ["EMP_PETRIC_ZEPHYRINE"]),
    ("In which city is Wendlin Pharma AG based?", 1, "Heidelberg", ["LOC_WENDLIN"]),
    ("Which diagnostics company merged with Brightmoor Analytics Ltd in September 2023?", 1, "Ellsworth Diagnostics Inc", ["MRG_BRIGHTMOOR_ELLSWORTH"]),
    ("Which company does Adaeze Okonkwo work for?", 1, "Brightmoor Analytics Ltd", ["EMP_OKONKWO_BRIGHTMOOR"]),
    ("Which company runs the joint drive-electronics development programme with Meridian Systems AG?", 1, "Ferrix Automation BV", ["PAR_MERIDIAN_FERRIX"]),
    ("For which company does Dario Vespucci work?", 1, "Quintara Energy plc", ["EMP_VESPUCCI_QUINTARA"]),
    ("In which city is Dalgren Maritime AS seated?", 1, "Trondheim", ["LOC_DALGREN"]),
    ("Which sensor array does Nyquist Sensors Oy build?", 1, "Nadir 12 sensor array", ["MAN_NYQUIST_NADIR"]),

    # ---------------- 2 hops (45) ----------------
    ("Which product is built by the toolmaker that Meridian Systems AG bought in 2021?", 2, "Tarnex-Getriebe", ["ACQ_MERIDIAN_NORVIK", "MAN_NORVIK_TARNEX"]),
    ("In which city does the toolmaker acquired by Meridian Systems AG have its seat?", 2, "Bratislava", ["ACQ_MERIDIAN_NORVIK", "LOC_NORVIK"]),
    ("Which manufacturer receives gearboxes from the company Meridian Systems AG took over in 2021?", 2, "Kalthoff AG", ["ACQ_MERIDIAN_NORVIK", "SUP_NORVIK_KALTHOFF"]),
    ("Which coating comes out of the business Meridian Systems AG bought in 2023?", 2, "Selenta-Beschichtung", ["ACQ_MERIDIAN_ZEPHYRINE", "MAN_ZEPHYRINE_SELENTA"]),
    ("Which pharmaceutical company is served by the subsidiary Meridian Systems AG took over in 2023?", 2, "Wendlin Pharma AG", ["ACQ_MERIDIAN_ZEPHYRINE", "SUP_ZEPHYRINE_WENDLIN"]),
    ("Which telemetry product comes from the software house Ardent Vale Ltd purchased?", 2, "Marlin 7 telemetry unit", ["ACQ_ARDENT_CENDRIA", "MAN_CENDRIA_MARLIN"]),
    ("Which energy group buys monitoring software from the company Ardent Vale Ltd purchased?", 2, "Quintara Energy plc", ["ACQ_ARDENT_CENDRIA", "SUP_CENDRIA_QUINTARA"]),
    ("In which city is the software house bought by Ardent Vale Ltd located?", 2, "Coventry", ["ACQ_ARDENT_CENDRIA", "LOC_CENDRIA"]),
    ("Who is named as an employee of the software house Ardent Vale Ltd acquired?", 2, "Sofia Andrade", ["ACQ_ARDENT_CENDRIA", "EMP_ANDRADE_CENDRIA"]),
    ("Who runs the technical department at the maritime company Quintara Energy plc acquired?", 2, "Sven Lidbergh", ["ACQ_QUINTARA_DALGREN", "EMP_LIDBERGH_DALGREN"]),
    ("Which terminal operator provides handling services to the maritime firm Quintara Energy plc bought?", 2, "Ravenport Terminals Ltd", ["ACQ_QUINTARA_DALGREN", "SUP_RAVENPORT_DALGREN"]),
    ("In which city is the metals producer that Vantor Stahlwerke AG took over in 2019 based?", 2, "Genoa", ["ACQ_VANTOR_LORQUIN", "LOC_LORQUIN"]),
    ("Which drive-systems manufacturer receives forged blanks from the company Vantor Stahlwerke AG took over?", 2, "Brekker Antriebstechnik GmbH", ["ACQ_VANTOR_LORQUIN", "SUP_LORQUIN_BREKKER"]),
    ("Who chairs the energy working group at the business Vantor Stahlwerke AG acquired?", 2, "Elena Marchetti", ["ACQ_VANTOR_LORQUIN", "EMP_MARCHETTI_LORQUIN"]),
    ("Which sensor product is built by the company Ferrix Automation BV bought in 2024?", 2, "Nadir 12 sensor array", ["ACQ_FERRIX_NYQUIST", "MAN_NYQUIST_NADIR"]),
    ("In which city is the sensor maker acquired by Ferrix Automation BV based?", 2, "Tampere", ["ACQ_FERRIX_NYQUIST", "LOC_NYQUIST"]),
    ("Which logistics company receives sensor assemblies from the firm Ferrix Automation BV acquired?", 2, "Tessaro Logistik AG", ["ACQ_FERRIX_NYQUIST", "SUP_NYQUIST_TESSARO"]),
    ("In which port city is the terminal operator that Tolmach Bergbau AG bought registered?", 2, "Rotterdam", ["ACQ_TOLMACH_RAVENPORT", "LOC_RAVENPORT"]),
    ("Which shipping company is served by the terminal operator Tolmach Bergbau AG acquired?", 2, "Dalgren Maritime AS", ["ACQ_TOLMACH_RAVENPORT", "SUP_RAVENPORT_DALGREN"]),
    ("Which robotics company receives technical fabric from the business Halcyon Verpackung GmbH bought in 2023?", 2, "Pelmar Robotics SA", ["ACQ_HALCYON_MARNIX", "SUP_MARNIX_PELMAR"]),
    ("In which city is the textile company acquired by Halcyon Verpackung GmbH based?", 2, "Eindhoven", ["ACQ_HALCYON_MARNIX", "LOC_MARNIX"]),
    ("Which pharmaceutical company buys assay reagents from the diagnostics firm that merged with Brightmoor Analytics Ltd?", 2, "Wendlin Pharma AG", ["MRG_BRIGHTMOOR_ELLSWORTH", "SUP_ELLSWORTH_WENDLIN"]),
    ("In which city is the diagnostics business that merged with Brightmoor Analytics Ltd based?", 2, "Boston", ["MRG_BRIGHTMOOR_ELLSWORTH", "LOC_ELLSWORTH"]),
    ("Which manufacturer has been delivering since 2018 to the robotics company Oakhurst Capital Partners backed in 2022?", 2, "Kalthoff AG", ["INV_OAKHURST_PELMAR", "SUP_KALTHOFF_PELMAR"]),
    ("In which city is the industrial group Oakhurst Capital Partners backed in the first quarter of 2023 based?", 2, "Leeds", ["INV_OAKHURST_ARDENT", "LOC_ARDENT"]),
    ("Which hydraulic product is built by the company that has supplied Pelmar Robotics SA since 2018?", 2, "Braxis hydraulic pump", ["SUP_KALTHOFF_PELMAR", "MAN_KALTHOFF_BRAXIS"]),
    ("Who takes over production planning at the company that has delivered to Pelmar Robotics SA since 2018?", 2, "Marta Cendrillo", ["SUP_KALTHOFF_PELMAR", "EMP_CENDRILLO_KALTHOFF"]),
    ("In which city is the company that supplies Wendlin Pharma AG with coating material based?", 2, "Jena", ["SUP_ZEPHYRINE_WENDLIN", "LOC_ZEPHYRINE"]),
    ("Which membrane is made by the company that delivers to Halcyon Verpackung GmbH under a two-year contract?", 2, "Halox membrane", ["SUP_SOLVEX_HALCYON", "MAN_SOLVEX_HALOX"]),
    ("In which city is the chemicals company that supplies Halcyon Verpackung GmbH based?", 2, "Mannheim", ["SUP_SOLVEX_HALCYON", "LOC_SOLVEX"]),
    ("Which servo product is built by the company that supplies drive units to Norvik Tools GmbH?", 2, "Vortan 400 Servoantrieb", ["SUP_BREKKER_NORVIK", "MAN_BREKKER_VORTAN"]),
    ("In which city is the supplier of drive units to Norvik Tools GmbH seated?", 2, "Bielefeld", ["SUP_BREKKER_NORVIK", "LOC_BREKKER"]),
    ("Who handles dispatch queries at the logistics company that delivers parts to Kalthoff AG from its central warehouse?", 2, "Yusuf Kandemir", ["SUP_TESSARO_KALTHOFF", "EMP_KANDEMIR_TESSARO"]),
    ("Which mining group owns the terminal operator that handles cargo for Dalgren Maritime AS?", 2, "Tolmach Bergbau AG", ["SUP_RAVENPORT_DALGREN", "ACQ_TOLMACH_RAVENPORT"]),
    ("In which city is the company that employs Ilse Brandtner headquartered?", 2, "Düsseldorf", ["EMP_BRANDTNER_MERIDIAN", "LOC_MERIDIAN"]),
    ("Which laboratory business was taken over by the group Ilse Brandtner works for?", 2, "Zephyrine Labs GmbH", ["EMP_BRANDTNER_MERIDIAN", "ACQ_MERIDIAN_ZEPHYRINE"]),
    ("Which automation company runs a joint development programme with the group that employs Ilse Brandtner?", 2, "Ferrix Automation BV", ["EMP_BRANDTNER_MERIDIAN", "PAR_MERIDIAN_FERRIX"]),
    ("Which textile company supplies the employer of Camille Aubertin?", 2, "Marnix Textil BV", ["EMP_AUBERTIN_PELMAR", "SUP_MARNIX_PELMAR"]),
    ("Which steel producer owns the employer of Elena Marchetti?", 2, "Vantor Stahlwerke AG", ["EMP_MARCHETTI_LORQUIN", "ACQ_VANTOR_LORQUIN"]),
    ("Which automation company acquired the employer of Nils Ekstrand?", 2, "Ferrix Automation BV", ["EMP_EKSTRAND_NYQUIST", "ACQ_FERRIX_NYQUIST"]),
    ("In which city is the employer of Yusuf Kandemir seated?", 2, "Hamburg", ["EMP_KANDEMIR_TESSARO", "LOC_TESSARO"]),
    ("Which insurance group works with the employer of Yusuf Kandemir on transport risks?", 2, "Aurenta Insurance Group", ["EMP_KANDEMIR_TESSARO", "PAR_AURENTA_TESSARO"]),
    ("In which robotics company did the employer of Fatima Zerhouni take a position in 2022?", 2, "Pelmar Robotics SA", ["EMP_ZERHOUNI_OAKHURST", "INV_OAKHURST_PELMAR"]),
    ("Which energy company runs a joint process-heat programme with the employer of Beatrix Homann?", 2, "Quintara Energy plc", ["EMP_HOMANN_VANTOR", "PAR_QUINTARA_VANTOR"]),
    ("Which software house supplies the employer of Dario Vespucci?", 2, "Cendria Software Ltd", ["EMP_VESPUCCI_QUINTARA", "SUP_CENDRIA_QUINTARA"]),

    # ---------------- 3 hops (25) ----------------
    ("Which coating is produced by the business acquired by the group that employs Ilse Brandtner?", 3, "Selenta-Beschichtung", ["EMP_BRANDTNER_MERIDIAN", "ACQ_MERIDIAN_ZEPHYRINE", "MAN_ZEPHYRINE_SELENTA"]),
    ("Which pump is built by the customer of the toolmaker Meridian Systems AG acquired?", 3, "Braxis hydraulic pump", ["ACQ_MERIDIAN_NORVIK", "SUP_NORVIK_KALTHOFF", "MAN_KALTHOFF_BRAXIS"]),
    ("In which city is the customer of the toolmaker Meridian Systems AG acquired seated?", 3, "Leipzig", ["ACQ_MERIDIAN_NORVIK", "SUP_NORVIK_KALTHOFF", "LOC_KALTHOFF"]),
    ("Which servo drive is built by the company that receives forged blanks from the metals producer Vantor Stahlwerke AG took over?", 3, "Vortan 400 Servoantrieb", ["ACQ_VANTOR_LORQUIN", "SUP_LORQUIN_BREKKER", "MAN_BREKKER_VORTAN"]),
    ("In which city is the company seated that receives forged blanks from the metals producer Vantor Stahlwerke AG took over?", 3, "Bielefeld", ["ACQ_VANTOR_LORQUIN", "SUP_LORQUIN_BREKKER", "LOC_BREKKER"]),
    ("Which toolmaker receives drive units from the company that is supplied with forged blanks by the metals producer Vantor Stahlwerke AG acquired?", 3, "Norvik Tools GmbH", ["ACQ_VANTOR_LORQUIN", "SUP_LORQUIN_BREKKER", "SUP_BREKKER_NORVIK"]),
    ("In which city is the energy group headquartered that buys monitoring software from the company Ardent Vale Ltd acquired?", 3, "Aberdeen", ["ACQ_ARDENT_CENDRIA", "SUP_CENDRIA_QUINTARA", "LOC_QUINTARA"]),
    ("Which maritime company was bought by the energy group that buys monitoring software from the firm Ardent Vale Ltd acquired?", 3, "Dalgren Maritime AS", ["ACQ_ARDENT_CENDRIA", "SUP_CENDRIA_QUINTARA", "ACQ_QUINTARA_DALGREN"]),
    ("Which telemetry unit is produced by the software house owned by the company Oakhurst Capital Partners backed in early 2023?", 3, "Marlin 7 telemetry unit", ["INV_OAKHURST_ARDENT", "ACQ_ARDENT_CENDRIA", "MAN_CENDRIA_MARLIN"]),
    ("In which city is the software house located that is owned by the company Oakhurst Capital Partners backed in early 2023?", 3, "Coventry", ["INV_OAKHURST_ARDENT", "ACQ_ARDENT_CENDRIA", "LOC_CENDRIA"]),
    ("Which manufacturer receives parts from the logistics company that is supplied by the sensor maker Ferrix Automation BV acquired?", 3, "Kalthoff AG", ["ACQ_FERRIX_NYQUIST", "SUP_NYQUIST_TESSARO", "SUP_TESSARO_KALTHOFF"]),
    ("In which city is the logistics company seated that is supplied by the sensor maker Ferrix Automation BV acquired?", 3, "Hamburg", ["ACQ_FERRIX_NYQUIST", "SUP_NYQUIST_TESSARO", "LOC_TESSARO"]),
    ("In which city is the shipping company seated that is served by the terminal operator Tolmach Bergbau AG acquired?", 3, "Trondheim", ["ACQ_TOLMACH_RAVENPORT", "SUP_RAVENPORT_DALGREN", "LOC_DALGREN"]),
    ("Who leads the technical department at the shipping company served by the terminal operator Tolmach Bergbau AG bought?", 3, "Sven Lidbergh", ["ACQ_TOLMACH_RAVENPORT", "SUP_RAVENPORT_DALGREN", "EMP_LIDBERGH_DALGREN"]),
    ("In which city is the robotics company based that receives technical fabric from the textile firm Halcyon Verpackung GmbH acquired?", 3, "Porto", ["ACQ_HALCYON_MARNIX", "SUP_MARNIX_PELMAR", "LOC_PELMAR"]),
    ("Which handling product is built by the robotics company supplied by the textile firm Halcyon Verpackung GmbH bought?", 3, "Larent handling arm", ["ACQ_HALCYON_MARNIX", "SUP_MARNIX_PELMAR", "MAN_PELMAR_LARENT"]),
    ("Which dosing product is made by the pharmaceutical company supplied by the diagnostics firm that merged with Brightmoor Analytics Ltd?", 3, "Ambrel dosing kit", ["MRG_BRIGHTMOOR_ELLSWORTH", "SUP_ELLSWORTH_WENDLIN", "MAN_WENDLIN_AMBREL"]),
    ("In which city is the pharmaceutical company based that is supplied by the diagnostics firm that merged with Brightmoor Analytics Ltd?", 3, "Heidelberg", ["MRG_BRIGHTMOOR_ELLSWORTH", "SUP_ELLSWORTH_WENDLIN", "LOC_WENDLIN"]),
    ("Which pharmaceutical company is supplied by the diagnostics firm that merged with the analytics business based in Lyon?", 3, "Wendlin Pharma AG", ["LOC_BRIGHTMOOR", "MRG_BRIGHTMOOR_ELLSWORTH", "SUP_ELLSWORTH_WENDLIN"]),
    ("Which diagnostics company also delivers to the pharmaceutical customer of the business Meridian Systems AG acquired in 2023?", 3, "Ellsworth Diagnostics Inc", ["ACQ_MERIDIAN_ZEPHYRINE", "SUP_ZEPHYRINE_WENDLIN", "SUP_ELLSWORTH_WENDLIN"]),
    ("Which sensor array is built by the company acquired by the automation partner of Meridian Systems AG?", 3, "Nadir 12 sensor array", ["PAR_MERIDIAN_FERRIX", "ACQ_FERRIX_NYQUIST", "MAN_NYQUIST_NADIR"]),
    ("In which city is the company based that was acquired by the automation partner of Meridian Systems AG?", 3, "Tampere", ["PAR_MERIDIAN_FERRIX", "ACQ_FERRIX_NYQUIST", "LOC_NYQUIST"]),
    ("In which city is the terminal operator based that was bought by the company Piotr Zawadzki heads?", 3, "Rotterdam", ["EMP_ZAWADZKI_TOLMACH", "ACQ_TOLMACH_RAVENPORT", "LOC_RAVENPORT"]),
    ("Which manufacturer has delivered since 2018 to the robotics company that the employer of Fatima Zerhouni backed?", 3, "Kalthoff AG", ["EMP_ZERHOUNI_OAKHURST", "INV_OAKHURST_PELMAR", "SUP_KALTHOFF_PELMAR"]),
    ("Which gearbox is built by the toolmaker that receives drive units from the company supplied with forged blanks by Lorquin Metals SpA?", 3, "Tarnex-Getriebe", ["SUP_LORQUIN_BREKKER", "SUP_BREKKER_NORVIK", "MAN_NORVIK_TARNEX"]),

    # ---------------- 4 hops (10) ----------------
    ("Vantor Stahlwerke AG took over a metals producer. That producer delivers forged blanks to a drive-systems firm, which in turn delivers drive units to a toolmaker. Which product does that toolmaker build?", 4, "Tarnex-Getriebe", ["ACQ_VANTOR_LORQUIN", "SUP_LORQUIN_BREKKER", "SUP_BREKKER_NORVIK", "MAN_NORVIK_TARNEX"]),
    ("Which steel producer runs a joint programme with the owner of the shipping company that is served by the terminal operator Tolmach Bergbau AG bought?", 4, "Vantor Stahlwerke AG", ["ACQ_TOLMACH_RAVENPORT", "SUP_RAVENPORT_DALGREN", "ACQ_QUINTARA_DALGREN", "PAR_QUINTARA_VANTOR"]),
    ("In which city is the maritime company seated that is owned by the energy group served by the software firm Ardent Vale Ltd purchased?", 4, "Trondheim", ["ACQ_ARDENT_CENDRIA", "SUP_CENDRIA_QUINTARA", "ACQ_QUINTARA_DALGREN", "LOC_DALGREN"]),
    ("Which pump is built by the manufacturer that receives parts from the logistics company supplied by the sensor maker Ferrix Automation BV acquired?", 4, "Braxis hydraulic pump", ["ACQ_FERRIX_NYQUIST", "SUP_NYQUIST_TESSARO", "SUP_TESSARO_KALTHOFF", "MAN_KALTHOFF_BRAXIS"]),
    ("Who is named as an employee of the energy group that buys software from the firm owned by the company Oakhurst Capital Partners backed in early 2023?", 4, "Dario Vespucci", ["INV_OAKHURST_ARDENT", "ACQ_ARDENT_CENDRIA", "SUP_CENDRIA_QUINTARA", "EMP_VESPUCCI_QUINTARA"]),
    ("Which investor took a position in the robotics company that is supplied by the customer of the toolmaker Meridian Systems AG acquired?", 4, "Oakhurst Capital Partners", ["ACQ_MERIDIAN_NORVIK", "SUP_NORVIK_KALTHOFF", "SUP_KALTHOFF_PELMAR", "INV_OAKHURST_PELMAR"]),
    ("In which city is the robotics company based that is supplied by the textile firm bought by the packaging maker Solvex Chemie GmbH delivers to?", 4, "Porto", ["SUP_SOLVEX_HALCYON", "ACQ_HALCYON_MARNIX", "SUP_MARNIX_PELMAR", "LOC_PELMAR"]),
    ("Who is named as the technical contact at the pharmaceutical company supplied by the diagnostics firm that merged with the analytics business based in Lyon?", 4, "Miriam Straub", ["LOC_BRIGHTMOOR", "MRG_BRIGHTMOOR_ELLSWORTH", "SUP_ELLSWORTH_WENDLIN", "EMP_STRAUB_WENDLIN"]),
    ("In which city is the shipping company seated that buys terminal services from the operator acquired by the company Piotr Zawadzki heads?", 4, "Trondheim", ["EMP_ZAWADZKI_TOLMACH", "ACQ_TOLMACH_RAVENPORT", "SUP_RAVENPORT_DALGREN", "LOC_DALGREN"]),
    ("In which city is the group headquartered that runs a joint development programme with the acquirer of the sensor maker employing Nils Ekstrand?", 4, "Düsseldorf", ["EMP_EKSTRAND_NYQUIST", "ACQ_FERRIX_NYQUIST", "PAR_MERIDIAN_FERRIX", "LOC_MERIDIAN"]),
]
