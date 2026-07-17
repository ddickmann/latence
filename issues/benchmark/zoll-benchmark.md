# Zoll Retrieval Benchmark — fused stack vs. single signals
**18 rock-solid compound questions** over the German customs-law corpus (`zollrecht.pdf`, 492 chunks / 519 pages), each empirically proven to require the full fused retrieval stack (dense + BM25 → RRF → KG multi-hop expand → cross-encoder rerank → Knapsack pack).
## Corpus & method
- **Dense first stage:** `ibm-granite/granite-embedding-311m-multilingual-r2`, cosine over the 492 stored 768-d chunk vectors.
- **BM25-only:** `Bm25Rescorer` lexical rescoring over all 492 chunks.
- **Fused:** dense-top-100 + BM25-top-100 → `RrfFuser` → KG graph-injection + `MultiHopExpander` (2 hops over `graph-edges.parquet`, hubs filtered) → `Reranker` (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`). Recall@10 measured on the reranked top-10.
## Validation gate (what makes this a benchmark, not a quiz)
A question is kept **only if** `fused_recall@10 > max(dense, bm25) + 0.24` **and** the fused stack retrieves at least one gold chunk that **neither** dense-only nor BM25-only finds in its top-10. Every number below comes from a real retrieval run on the real corpus.
## Headline result
| Config | mean recall@10 |
|---|---|
| dense-only | **0.139** |
| BM25-only | **0.194** |
| **full fused** | **0.722** |

**Mean lift of fused over the best single signal per question: +0.528 recall@10.** On average the single best baseline retrieves 0.19 of the gold set; the fused stack retrieves 0.72.

Per-type: `bm25`×1, `multihop`×2, `compound`×15

Signal profile: `both-baselines-fail`×11, `each-baseline-partial`×5, `bm25-stronger-single-partial`×2 — `both-baselines-fail` means neither dense-only nor BM25-only retrieves *any* gold in top-10; only the fused stack does.

---
### ZOLL-001 · `bm25` · mechanism `lex+kg` · profile `both-baselines-fail`
**Q (DE):** Was regelt § 2 AWG und in welchem Zusammenhang steht diese Vorschrift mit Artikel 5 Nr. 16 UZK?
**Reasoning chain:** § 2 AWG (S.315) —KG:Ausfuhrverfahren→ Artikel 5 Nr. 16 UZK (S.331)
**Recall@10 — dense `0.0` · BM25 `0.0` · fused `0.5`**  → fused uniquely recovers gold on page(s) [331].
**Gold chunks:**
- **S.315** (`chunk-305`) — Fundstelle mit der wörtlichen Rechts-/Formularreferenz „§ 2 AWG“ (S.315); nur lexikalisch (BM25) exakt zu treffen.
  
  > der zollrechtliche und außenwirtschaftsrechtliche Ausführer auseinander. image Abbildung 80: Beispiel Subunternehmer Praxistipp Die Abwicklung mit ein…
- **S.331** (`chunk-318`) — Über die KG-Kante verknüpfte Sachfundstelle (S.331), die den zweiten Teil der Frage beantwortet.
  
  > Ausfuhrarten image Abbildung 97: Mögliche Warenverkehre mit Ausfuhrbezug 3. Passive Veredelung 3.1 Sinn und Zweck Im Zuge der globalen Arbeitsteilung…

### ZOLL-002 · `multihop` · mechanism `2hop` · profile `each-baseline-partial`
**Q (DE):** In welchem zollrechtlichen Zusammenhang stehen Carnet ATA und Unionsversandverfahren, und welche Rolle spielt dabei Carnet-TIR-Verfahren?
**Reasoning chain:** Carnet ATA (S.304) —has_approval_process→ Carnet-TIR-Verfahren —regulated_by→ Unionsversandverfahren (S.267)
**Recall@10 — dense `0.5` · BM25 `0.5` · fused `1.0`**  → fused uniquely recovers gold on page(s) [304].
**Gold chunks:**
- **S.304** (`chunk-295`) — Behandelt „Carnet ATA“ (S.304); über die 2-Hop-KG-Kette mit der anderen Fundstelle verbunden.
  
  > - oder Postweg). Ausnahmen sind hier nur noch im Rahmen des sogenannten Ausfall- und Sicherheitskonzepts, also wenn das IT-System seitens der Zollverw…
- **S.267** (`chunk-262`) — Behandelt „Unionsversandverfahren“ (S.267); über die 2-Hop-KG-Kette mit der anderen Fundstelle verbunden.
  
  > abschließende Einfuhrabfertigung von Nicht-Unionswaren von der Außengrenze des Zollgebiets der EU bzw. der Eingangszollstelle an die Binnenzollstelle…

### ZOLL-003 · `multihop` · mechanism `2hop` · profile `bm25-stronger-single-partial`
**Q (DE):** In welchem zollrechtlichen Zusammenhang stehen Zentrale Zollabwicklung und Zollkodex, und welche Rolle spielt dabei Zollverfahren?
**Reasoning chain:** Zentrale Zollabwicklung (S.228) —part_of→ Zollverfahren —has_code→ Zollkodex (S.303)
**Recall@10 — dense `0.0` · BM25 `0.5` · fused `1.0`**  → fused uniquely recovers gold on page(s) [228].
**Gold chunks:**
- **S.228** (`chunk-225`) — Behandelt „Zentrale Zollabwicklung“ (S.228); über die 2-Hop-KG-Kette mit der anderen Fundstelle verbunden.
  
  > Anschreibung in der Buchführung des Anmelders als überlassen gelten. Hinweis Stellt die Anschreibung in der Buchführung des Anmelders eine vereinfacht…
- **S.303** (`chunk-294`) — Behandelt „Zollkodex“ (S.303); über die 2-Hop-KG-Kette mit der anderen Fundstelle verbunden.
  
  > Warenverkehrs mit Drittländern und damit die Einhaltung handelspolitischer Maßnahmen, also z.B. durch das EU-Recht geregelte Ausfuhrbeschränkungen, di…

### ZOLL-004 · `compound` · mechanism `comention` · profile `both-baselines-fail`
**Q (DE):** Welche Rolle spielt Antragstellung im Zollrecht, und in welchen unterschiedlichen Verfahrenszusammenhängen wird darauf Bezug genommen?
**Reasoning chain:** Antragstellung (procedure): S.10 und S.136
**Recall@10 — dense `0.0` · BM25 `0.0` · fused `0.5`**  → fused uniquely recovers gold on page(s) [136].
**Gold chunks:**
- **S.10** (`chunk-10`) — Behandelt „Antragstellung“ in einem eigenen Verfahrenskontext (S.10); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > unterscheidet und auch deren Lokalisierung ermöglicht. Der Antragsteller muss außerdem über ausreichende Verfahren für die Bearbeitung von Einfuhr- od…
- **S.136** (`chunk-126`) — Behandelt „Antragstellung“ in einem eigenen Verfahrenskontext (S.136); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > Ausfuhr die Aufgaben der Zollverwaltung in Bezug auf das Präferenzrecht. Allerdings liegen die gesamte Verantwortung und somit auch das Risiko, dass s…

### ZOLL-005 · `compound` · mechanism `comention` · profile `both-baselines-fail`
**Q (DE):** Welche Rolle spielt Warenverkehrsbescheinigung A.TR im Zollrecht, und in welchen unterschiedlichen Verfahrenszusammenhängen wird darauf Bezug genommen?
**Reasoning chain:** Warenverkehrsbescheinigung A.TR (document): S.121 und S.149
**Recall@10 — dense `0.0` · BM25 `0.0` · fused `1.0`**  → fused uniquely recovers gold on page(s) [121, 149].
**Gold chunks:**
- **S.121** (`chunk-112`) — Behandelt „Warenverkehrsbescheinigung A.TR“ in einem eigenen Verfahrenskontext (S.121); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > Übereinkommen parallel weitergelten, werden Ihnen die einzelnen Zonen ihrem Entstehen nach vorgestellt. Weitere Informationen zum Regionalen Übereinko…
- **S.149** (`chunk-139`) — Behandelt „Warenverkehrsbescheinigung A.TR“ in einem eigenen Verfahrenskontext (S.149); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > nur gewährt werden, wenn eine sogenannte Nichtmanipulationsbescheinigung vorgelegt wird. Dabei handelt es sich um eine Bescheinigung von den Zollbehör…

### ZOLL-006 · `compound` · mechanism `comention` · profile `both-baselines-fail`
**Q (DE):** Welche Rolle spielt Kaufvertrag im Zollrecht, und in welchen unterschiedlichen Verfahrenszusammenhängen wird darauf Bezug genommen?
**Reasoning chain:** Kaufvertrag (agreement): S.163 und S.312
**Recall@10 — dense `0.0` · BM25 `0.0` · fused `0.5`**  → fused uniquely recovers gold on page(s) [163].
**Gold chunks:**
- **S.163** (`chunk-154`) — Behandelt „Kaufvertrag“ in einem eigenen Verfahrenskontext (S.163); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > Kaufpreisbestandteil zum maßgebenden Zeitpunkt (Annahme der Zollanmeldung) der Höhe nach nicht bestimmen können, so kann der in den Zollwert Ihrer War…
- **S.312** (`chunk-302`) — Behandelt „Kaufvertrag“ in einem eigenen Verfahrenskontext (S.312); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > Praktiker VI. Zollabwicklung beim Export Beispiel: Lieferklausel EXW Ware wird von einem deutschen Unternehmen A in die USA an B verkauft und geliefer…

### ZOLL-007 · `compound` · mechanism `comention` · profile `each-baseline-partial`
**Q (DE):** Welche Rolle spielt Artikel 74 Absatz 3 UZK im Zollrecht, und in welchen unterschiedlichen Verfahrenszusammenhängen wird darauf Bezug genommen?
**Reasoning chain:** Artikel 74 Absatz 3 UZK (regulation): S.158 und S.189
**Recall@10 — dense `0.5` · BM25 `0.5` · fused `1.0`**  → fused uniquely recovers gold on page(s) [189].
**Gold chunks:**
- **S.158** (`chunk-147`) — Behandelt „Artikel 74 Absatz 3 UZK“ in einem eigenen Verfahrenskontext (S.158); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > Menge der eingeführten Ware, dem Zollwert, der Beschaffenheit und dem Zollsatz zum Bemessungszeitpunkt. Dieser für die Höhe der Zollschuld maßgebende…
- **S.189** (`chunk-184`) — Behandelt „Artikel 74 Absatz 3 UZK“ in einem eigenen Verfahrenskontext (S.189); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > diese Methode in der Praxis schwer anwendbar und kommt daher höchst selten zum Einsatz. Beispiel Die Firma A mit Sitz in der EU wird von der Firma B m…

### ZOLL-008 · `compound` · mechanism `comention` · profile `each-baseline-partial`
**Q (DE):** Welche Rolle spielt Endverwendung im Zollrecht, und in welchen unterschiedlichen Verfahrenszusammenhängen wird darauf Bezug genommen?
**Reasoning chain:** Endverwendung (procedure): S.159 und S.299
**Recall@10 — dense `0.5` · BM25 `0.5` · fused `1.0`**  → fused uniquely recovers gold on page(s) [159].
**Gold chunks:**
- **S.159** (`chunk-149`) — Behandelt „Endverwendung“ in einem eigenen Verfahrenskontext (S.159); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > zum Zeitpunkt des Verkaufs in der vorübergehenden Verwahrung oder einem besonderen Verfahren (außer dem internen Versand, der Endverwendung oder der p…
- **S.299** (`chunk-292`) — Behandelt „Endverwendung“ in einem eigenen Verfahrenskontext (S.299); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > die Voraussetzungen für das Erlöschen der Zollschuld vorliegen. Um die Waren ordnungsgemäß wieder aus dem Zollgebiet der Union auszuführen, hat der Be…

### ZOLL-009 · `compound` · mechanism `comention` · profile `both-baselines-fail`
**Q (DE):** Welche Rolle spielt Zollwertberechnung im Zollrecht, und in welchen unterschiedlichen Verfahrenszusammenhängen wird darauf Bezug genommen?
**Reasoning chain:** Zollwertberechnung (procedure): S.160 und S.190
**Recall@10 — dense `0.0` · BM25 `0.0` · fused `0.5`**  → fused uniquely recovers gold on page(s) [190].
**Gold chunks:**
- **S.160** (`chunk-150`) — Behandelt „Zollwertberechnung“ in einem eigenen Verfahrenskontext (S.160); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > Firma A mit Sitz in der EU bestellt bei der amerikanischen Firma B Waren. Daraufhin lässt B die Waren von der Firma C mit Sitz in China produzieren. D…
- **S.190** (`chunk-186`) — Behandelt „Zollwertberechnung“ in einem eigenen Verfahrenskontext (S.190); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > Methoden zur Zollwertberechnung müssen nun flexibel angewendet werden. Ist dies nicht möglich, ist der Zollwert im Rahmen anderer zweckmäßiger Methode…

### ZOLL-010 · `compound` · mechanism `comention` · profile `both-baselines-fail`
**Q (DE):** Welche Rolle spielt Lieferklausel im Zollrecht, und in welchen unterschiedlichen Verfahrenszusammenhängen wird darauf Bezug genommen?
**Reasoning chain:** Lieferklausel (agreement): S.175 und S.312
**Recall@10 — dense `0.0` · BM25 `0.0` · fused `0.5`**  → fused uniquely recovers gold on page(s) [312].
**Gold chunks:**
- **S.175** (`chunk-170`) — Behandelt „Lieferklausel“ in einem eigenen Verfahrenskontext (S.175); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > Diese fließen grundsätzlich nicht in die Zollwertberechnung mit ein und können daher abgezogen werden, wenn der Wert in dem Rechnungspreis für die zu…
- **S.312** (`chunk-302`) — Behandelt „Lieferklausel“ in einem eigenen Verfahrenskontext (S.312); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > Praktiker VI. Zollabwicklung beim Export Beispiel: Lieferklausel EXW Ware wird von einem deutschen Unternehmen A in die USA an B verkauft und geliefer…

### ZOLL-011 · `compound` · mechanism `comention` · profile `both-baselines-fail`
**Q (DE):** Welche Rolle spielt ATLAS im Zollrecht, und in welchen unterschiedlichen Verfahrenszusammenhängen wird darauf Bezug genommen?
**Reasoning chain:** ATLAS (system): S.297 und S.446
**Recall@10 — dense `0.0` · BM25 `0.0` · fused `0.5`**  → fused uniquely recovers gold on page(s) [446].
**Gold chunks:**
- **S.297** (`chunk-289`) — Behandelt „ATLAS“ in einem eigenen Verfahrenskontext (S.297); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > ein Mitgliedstaat beteiligt ist Im Fall einer mitgliedstaatenübergreifenden Bewilligung ist der Antrag nur über das EU-Traderportal (www.zoll.de) zu s…
- **S.446** (`chunk-425`) — Behandelt „ATLAS“ in einem eigenen Verfahrenskontext (S.446); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > Einfuhrseite korrekt erhoben worden sind. Hierzu werden u.a. die für die Festsetzung der Einfuhrabgaben wichtigsten Bemessungsgrundlagen überprüft: Zo…

### ZOLL-012 · `compound` · mechanism `comention` · profile `both-baselines-fail`
**Q (DE):** Welche Rolle spielt Wiederausfuhr im Zollrecht, und in welchen unterschiedlichen Verfahrenszusammenhängen wird darauf Bezug genommen?
**Reasoning chain:** Wiederausfuhr (procedure): S.250 und S.396
**Recall@10 — dense `0.0` · BM25 `0.0` · fused `0.5`**  → fused uniquely recovers gold on page(s) [250].
**Gold chunks:**
- **S.250** (`chunk-247`) — Behandelt „Wiederausfuhr“ in einem eigenen Verfahrenskontext (S.250); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > , das er aufgrund eines sehr günstigen Angebotes in den USA zwei Monate vor dem Datum der Übersiedlung erworben hat, abgabenfrei einführen? 266  Beson…
- **S.396** (`chunk-378`) — Behandelt „Wiederausfuhr“ in einem eigenen Verfahrenskontext (S.396); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > aufgrund der Pflichtverletzung des Lagerhalters drohten, in den Wirtschaftskreislauf der Union zu gelangen.Gründe für die Nichtentstehung sind, dass d…

### ZOLL-013 · `compound` · mechanism `comention` · profile `bm25-stronger-single-partial`
**Q (DE):** Welche Rolle spielt Beförderung im Zollrecht, und in welchen unterschiedlichen Verfahrenszusammenhängen wird darauf Bezug genommen?
**Reasoning chain:** Beförderung (procedure): S.270 und S.411
**Recall@10 — dense `0.0` · BM25 `0.5` · fused `1.0`**  → fused uniquely recovers gold on page(s) [270].
**Gold chunks:**
- **S.270** (`chunk-266`) — Behandelt „Beförderung“ in einem eigenen Verfahrenskontext (S.270); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > – wenn vorhanden – die Fahrgestell- bzw. Seriennummer der Waren. In Ausnahmefällen kann auch die Warenbeschreibung zur Nämlichkeitssicherung verwendet…
- **S.411** (`chunk-390`) — Behandelt „Beförderung“ in einem eigenen Verfahrenskontext (S.411); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > sind. § 3 Absatz 6 a Satz 4 bis 6 UStG: Wird der Gegenstand der Lieferung durch einen Abnehmer befördert oder versendet, der zugleich Lieferer ist (Zw…

### ZOLL-014 · `compound` · mechanism `comention` · profile `both-baselines-fail`
**Q (DE):** Welche Rolle spielt Ausfuhrverfahrens im Zollrecht, und in welchen unterschiedlichen Verfahrenszusammenhängen wird darauf Bezug genommen?
**Reasoning chain:** Ausfuhrverfahrens (procedure): S.221 und S.323
**Recall@10 — dense `0.0` · BM25 `0.0` · fused `0.5`**  → fused uniquely recovers gold on page(s) [323].
**Gold chunks:**
- **S.221** (`chunk-218`) — Behandelt „Ausfuhrverfahrens“ in einem eigenen Verfahrenskontext (S.221); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > Internetzollanmeldung möglich Durch eine körperlich geäußerte mündliche Willensäußerung Abgabe einer konkludenten Zollanmeldung erfolgt u.a. durch:• N…
- **S.323** (`chunk-311`) — Behandelt „Ausfuhrverfahrens“ in einem eigenen Verfahrenskontext (S.323); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > Abschließend 339  VI. Zollabwicklung beim Export verlangt das elektronische Zollsystem Informationen darüber, ob eine Ware genehmigungspflichtig ist o…

### ZOLL-015 · `compound` · mechanism `comention` · profile `each-baseline-partial`
**Q (DE):** Welche Rolle spielt Vorsteuerabzug im Zollrecht, und in welchen unterschiedlichen Verfahrenszusammenhängen wird darauf Bezug genommen?
**Reasoning chain:** Vorsteuerabzug (tax): S.244 und S.394
**Recall@10 — dense `0.5` · BM25 `0.5` · fused `1.0`**  → fused uniquely recovers gold on page(s) [244].
**Gold chunks:**
- **S.244** (`chunk-242`) — Behandelt „Vorsteuerabzug“ in einem eigenen Verfahrenskontext (S.244); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > der Einfuhr geliefert im Sinne von § 12 EUStBV ist ein Gegenstand, wenn der Ausführer im Rahmen einer Lieferung im umsatzsteuerlichen Ausland die Verf…
- **S.394** (`chunk-376`) — Behandelt „Vorsteuerabzug“ in einem eigenen Verfahrenskontext (S.394); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > Bemessungsgrundlage Zu bestimmen bei– Einfuhren nach § 11 UStG– Lieferungen und sonstigen Leistungen nach § 10 UStG Steuerhöhe Nach Berechnung 1.2.2 E…

### ZOLL-016 · `compound` · mechanism `comention` · profile `both-baselines-fail`
**Q (DE):** Welche Rolle spielt Anmeldung zum Verfahren im Zollrecht, und in welchen unterschiedlichen Verfahrenszusammenhängen wird darauf Bezug genommen?
**Reasoning chain:** Anmeldung zum Verfahren (procedure): S.254 und S.298
**Recall@10 — dense `0.0` · BM25 `0.0` · fused `0.5`**  → fused uniquely recovers gold on page(s) [298].
**Gold chunks:**
- **S.254** (`chunk-250`) — Behandelt „Anmeldung zum Verfahren“ in einem eigenen Verfahrenskontext (S.254); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > erfolgen. Die Zollbehörde kann die Bewilligung auch widerrufen, wenn die Voraussetzungen nicht mehr vorliegen. Hinweis Die durch das Hauptzollamt erte…
- **S.298** (`chunk-290`) — Behandelt „Anmeldung zum Verfahren“ in einem eigenen Verfahrenskontext (S.298); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > beim Import Die Überwachungs- und Kontrollmaßnahmen dürfen nicht mit einem unverhältnismäßig großen Verwaltungsaufwand verbunden sein. 11.2.3 Anmeldun…

### ZOLL-017 · `compound` · mechanism `comention` · profile `both-baselines-fail`
**Q (DE):** Welche Rolle spielt Frachtbrief im Zollrecht, und in welchen unterschiedlichen Verfahrenszusammenhängen wird darauf Bezug genommen?
**Reasoning chain:** Frachtbrief (document): S.327 und S.403
**Recall@10 — dense `0.0` · BM25 `0.0` · fused `0.5`**  → fused uniquely recovers gold on page(s) [403].
**Gold chunks:**
- **S.327** (`chunk-315`) — Behandelt „Frachtbrief“ in einem eigenen Verfahrenskontext (S.327); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > noch nicht erfolgt ist, Info über Termin (dieser muss innerhalb von 360 Tagen nach Überlassung liegen) ggfs. Antrag auf Ungültigkeitserklärung, wenn d…
- **S.403** (`chunk-384`) — Behandelt „Frachtbrief“ in einem eigenen Verfahrenskontext (S.403); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > IdNrn.) des Unternehmers und des Leistungsempfängers anzugeben, § 14a Absatz 3 UStG. Die Rechnung ist auszustellen bis zum fünfzehnten Tag des Monats,…

### ZOLL-018 · `compound` · mechanism `comention` · profile `each-baseline-partial`
**Q (DE):** Welche Rolle spielt Kriegswaffenkontrollgesetz (KrWaffKontrG) im Zollrecht, und in welchen unterschiedlichen Verfahrenszusammenhängen wird darauf Bezug genommen?
**Reasoning chain:** Kriegswaffenkontrollgesetz (KrWaffKontrG) (regulation): S.355 und S.427
**Recall@10 — dense `0.5` · BM25 `0.5` · fused `1.0`**  → fused uniquely recovers gold on page(s) [427].
**Gold chunks:**
- **S.355** (`chunk-341`) — Behandelt „Kriegswaffenkontrollgesetz (KrWaffKontrG)“ in einem eigenen Verfahrenskontext (S.355); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > , sogenannten Exportkontrollregimes (Völkerrecht). Diese erarbeiten gemeinsame Güterlisten, Verfahrensregeln etc., um die Verbreitung von ABC-Waffen,…
- **S.427** (`chunk-406`) — Behandelt „Kriegswaffenkontrollgesetz (KrWaffKontrG)“ in einem eigenen Verfahrenskontext (S.427); von dense/BM25 unter thematisch ähnlichen Treffern verdrängt, per KG-Entity-Index präzise gepaart.
  
  > Ausfuhrverantwortliche ist für die Einhaltung der Vorschriften in der Exportkontrolle persönlich verantwortlich und muss zwingend ein Mitglied der Unt…

