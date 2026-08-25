"""The KG Export's redaction posture: node labels are NOT redacted, and the Runner says so.

`GraphHyperedge.tau_text` follows the corpus redaction, enforced at its writer (ADR-0057);
`GraphNode.canonical_name` does not, and `graph_export.py` holds no redaction gate at all. That
asymmetry is DELIBERATE — a graph of `[PERSON]` nodes is a graph about nobody, and masking the
label would collapse every person onto the single `a|person` alias key in the ADR-0056 KV — so
these tests pin it as behaviour rather than propose a fix:

* the surprising configuration (a redaction Stage wired while `extract_on` is `unmasked` — the
  corpus masked, the graph not) logs `KG_ENTITY_NAMES_WARNING` exactly once, before a byte is
  written, and the real name is verifiably in every surface the banner names;
* the LEVER works: the SAME DAG with `extract_on: masked` puts no PII span text in any node
  label, and warns about nothing;
* neither half of the condition alone warns.

The two pipelines differ in exactly one line (`extract_on`), so the A/B is the flag and nothing
else.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from latence_core.runner import KG_ENTITY_NAMES_WARNING, LocalRunner
from latence_core.stages.export import JsonlParquetExport
from latence_core.stages.graph_export import KnowledgeGraphExport
from latence_core.storage import Storage

#: The logger the Runner routes its KG entity-name risk WARNING through.
_RUNNER_LOGGER = "latence_core.runner"

#: The PII the corpus redaction masks out — the exact span text a masked-mode export's node
#: labels must not contain anywhere.
_PII_NAMES = ("Alice", "Bob")


def _pipeline_yaml(store: str, docs_dir: Path, *, extract_on: str, redact: bool) -> str:
    """The graph slice, wired chunk → redact → entities/relations → graph → KG export.

    Extraction depends on the redaction Stage in BOTH postures (the ordering `extract_on:
    masked` requires), so switching the flag is the only difference between the two runs.
    """
    upstream = "redact" if redact else "chunk"
    redact_stage = (
        """
  - name: redact
    capability: redaction
    provider: redaction.hybrid_rule
    depends_on: [chunk]
    config:
      gazetteer:
        name: [Alice, Bob]
      default_action: mask
      salt: test-salt
"""
        if redact
        else ""
    )
    return f"""
name: kg-name-posture
storage_uri: {store}
extract_on: {extract_on}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
      extensions: [txt]
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
{redact_stage}
  - name: entities
    capability: entity_extraction
    provider: entity.gazetteer
    depends_on: [{upstream}]
    config:
      labels:
        ORG:
          terms: [Acme Corporation, Acme]
        PERSON:
          terms: [Alice, Bob]
  - name: relations
    capability: relation_extraction
    provider: relation.pattern
    depends_on: [{upstream}, entities]
    config:
      window_chars: 200
      relations:
        works_for:
          head: PERSON
          tail: ORG
          triggers: [works at, joined]
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [entities, relations]
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
    config:
      corpus_id: kg-name-posture
  - name: export_kg
    capability: export
    provider: export.knowledge_graph
    depends_on: [graph]
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [{upstream}]
    config:
      basename: corpus
"""


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text(
        "Alice works at Acme. Bob joined Acme Corporation last year.\n", encoding="utf-8"
    )
    return docs


class _Run:
    """One executed pipeline plus every KG surface read back off Storage."""

    def __init__(self, store: str, run_id: str) -> None:
        s = Storage.for_uri(store)
        base = f"{store}/_latence/runs/{run_id}/export"
        with s.open(f"{base}/graph-nodes.parquet", "rb") as fh:
            self.nodes = pq.read_table(fh).to_pylist()  # type: ignore[no-untyped-call]
        self.ttl = s.read_text(f"{base}/graph.ttl")
        self.graphml = s.read_text(f"{base}/graph.graphml")
        self.corpus = s.read_text(f"{base}/corpus.jsonl")

    @property
    def names(self) -> list[str]:
        return [n["canonical_name"] for n in self.nodes]

    @property
    def graphml_labels(self) -> list[str]:
        ns = "{http://graphml.graphdrawing.org/xmlns}"
        graph_el = ET.fromstring(self.graphml).find(f"{ns}graph")
        assert graph_el is not None
        return [
            (d.text or "")
            for node in graph_el.findall(f"{ns}node")
            for d in node.findall(f"{ns}data")
            if d.get("key") == "label"
        ]

    @property
    def ttl_labels(self) -> list[str]:
        return [
            line.split("rdfs:label", 1)[1].strip().rstrip(";").strip()
            for line in self.ttl.splitlines()
            if "rdfs:label" in line
        ]

    @property
    def evidence_snippets(self) -> list[str]:
        return [json.loads(n["evidence"]).get("snippet") or "" for n in self.nodes]


def _run(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    *,
    extract_on: str = "unmasked",
    redact: bool = True,
) -> tuple[_Run, list[str]]:
    store = f"file://{tmp_path}/store"
    yaml_text = _pipeline_yaml(store, _docs(tmp_path), extract_on=extract_on, redact=redact)
    from latence_core.pipeline import Pipeline

    with caplog.at_level(logging.WARNING, logger=_RUNNER_LOGGER):
        LocalRunner().run(Pipeline.from_yaml_text(yaml_text), run_id="kgnames")
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.name == _RUNNER_LOGGER and r.levelno == logging.WARNING
    ]
    return _Run(store, "kgnames"), warnings


def test_export_declares_that_it_writes_entity_names() -> None:
    """The Runner-checked opt-in marker: on the KG Export, absent on the corpus Export."""
    assert KnowledgeGraphExport().exports_entity_names is True
    assert getattr(JsonlParquetExport({}), "exports_entity_names", False) is False


def test_masked_corpus_with_unmasked_extraction_warns_once_and_ships_the_real_names(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The documented (WONTFIX) behaviour: the corpus is masked, the graph is not — and says so.

    Pins BOTH halves. The banner fires exactly once, and each surface it names is checked to
    actually carry the real name, so the warning can never drift into describing a leak that is
    no longer there (or miss one that is).
    """
    run, warnings = _run(tmp_path, caplog)

    assert warnings.count(KG_ENTITY_NAMES_WARNING) == 1, "one banner per KG export, not per row"

    # Non-vacuity: the redaction Stage really did mask the names out of the exported corpus.
    assert "[NAME]" in run.corpus
    for name in _PII_NAMES:
        assert name not in run.corpus, f"{name} must not survive in the masked corpus"

    # ...and every surface the banner names carries it anyway.
    assert "Alice" in run.names
    assert any("Alice" in label for label in run.ttl_labels)
    assert any("Alice" in label for label in run.graphml_labels)
    assert any("Alice" in snippet for snippet in run.evidence_snippets)


def test_masked_extraction_puts_no_pii_span_text_in_any_node_label(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The LEVER: `extract_on: masked` — same DAG, one flag — keeps the names out of the graph.

    PII never reaches extraction, so it reaches none of the six surfaces. Checked against the
    whole serialized TTL/GraphML, not just the parsed labels, so a name leaking through an IRI,
    a description property or an evidence snippet fails here too.
    """
    run, warnings = _run(tmp_path, caplog, extract_on="masked")

    assert warnings == [], "nothing to warn about — PII never reached extraction"
    for name in _PII_NAMES:
        assert name not in run.names
        assert name not in run.ttl
        assert name not in run.graphml
        assert not any(name in json.dumps(n) for n in run.nodes), (
            f"{name} leaked into a nodes-parquet column (evidence/properties/description)"
        )
    # Non-vacuous: the graph itself is still built — the ORG entities, which no redactor
    # touched, resolved normally. An empty graph would pass the assertions above for free.
    assert "Acme" in run.names


def test_no_redaction_stage_does_not_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Nothing is masked anywhere, so there is no asymmetry — and the names still ship."""
    run, warnings = _run(tmp_path, caplog, redact=False)

    assert warnings == []
    assert "Alice" in run.names
