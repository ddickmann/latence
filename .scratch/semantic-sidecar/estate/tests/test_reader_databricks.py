"""The reader over Unity Catalog, against a stand-in that can disagree (ticket 33)."""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
import yaml

from estate.reader_databricks import DatabricksReader, WarehouseError
from estate.reader_duckdb import DuckDBReader

from .conftest import VIEWS
from .test_dialect import _all
from .uc_standin import StandIn

TAGS = {("acme_shop", "dealers", "contact_email"): ["pii"], ("acme_shop", "owners", "name"): ["PII", "contact"], ("acme_shop", "vehicles", "vin"): ["pii"]}


def _metric_view() -> dict[str, Any]:
    """The toy's metric view as the catalog stores one: the definition keeps names and
    expressions, and comments, display names and synonyms sit on the listed columns."""
    full = yaml.safe_load(VIEWS.split("\n", 1)[1])
    stored = {"version": full["version"], "source": full["source"], "joins": full["joins"],
              "dimensions": [{"name": d["name"], "expr": d["expr"]} for d in full["dimensions"]], "measures": [{"name": m["name"], "expr": m["expr"]} for m in full["measures"]]}
    columns = [{"name": f["name"], "type_text": "string", "comment": f.get("comment"),
                "metadata": json.dumps({k: f[k] for k in ("display_name", "synonyms") if k in f}) if i % 2 else {k: f[k] for k in ("display_name", "synonyms") if k in f}}
               for i, f in enumerate(full["dimensions"] + full["measures"])]
    return {"name": "shop_claims", "schema_name": "acme_semantics", "catalog_name": "demo", "table_type": "METRIC_VIEW", "table_id": "mv1",
            "comment": full["comment"], "view_definition": yaml.safe_dump(stored, sort_keys=False), "columns": columns}


def _reader(standin: StandIn, token: str = "good", schemas: tuple[str, ...] = ("shop", "marts", "semantics")) -> DatabricksReader:
    return DatabricksReader("https://workspace.invalid", token, "demo", schemas, "wh1", schema_prefix="acme_", sensitive_tags=("pii",), poll_seconds=0, http=standin)


def test_the_same_model_comes_through_the_catalog_as_through_the_mirror(reader: DuckDBReader) -> None:
    standin = StandIn(reader, tags=TAGS, metric_views=[_metric_view()])
    uc = _reader(standin)
    ids = {o.id: o.full_name for o in uc.objects()} | {o.id: o.full_name for o in reader.objects()}

    def named(text: str) -> str:
        """The same model whichever warehouse it came from: objects by name where each has its
        own id, and without the spelling of a column's type, which is the warehouse's own."""
        for oid, name in ids.items():
            text = text.replace(json.dumps(oid), json.dumps(name))
        return re.sub(r'"type": "[^"]*"', '"type": "-"', text)

    assert named(_all(uc)) == named(_all(reader))
    assert all(o.id.startswith("uc:") for o in uc.objects())  # an object is keyed on the catalog's own id
    # It disagreed on the way and the reader coped: asked again after "too many requests", followed the pages,
    # waited for pending statements and fetched the further parts of long results.
    assert not standin.busy_once and any(path.endswith("/result/chunks/1") for _m, path in standin.calls)
    assert sum(m == "GET" and path.count("/") == 5 and path.startswith("/api/2.0/sql/statements/") for m, path in standin.calls) > 10


def test_a_column_tagged_sensitive_is_in_no_statement(reader: DuckDBReader) -> None:
    standin = StandIn(reader, tags=TAGS, metric_views=[_metric_view()])
    uc = _reader(standin)
    _all(uc)
    flagged = {(o.full_name, c.name) for o in uc.objects() for c in o.columns if c.sensitive}
    assert flagged == {("shop.dealers", "contact_email"), ("shop.owners", "name"), ("shop.vehicles", "vin")}
    assert len(standin.statements) > 50
    assert not [s for s in standin.statements if "`contact_email`" in s or "`vin`" in s]
    assert not [s for s in standin.statements if "`owners`" in s and "`name`" in s]


def test_a_metric_view_is_put_together_again_from_the_definition_and_the_listed_columns(reader: DuckDBReader) -> None:
    view = _reader(StandIn(reader, metric_views=[_metric_view()])).metric_views()[0]
    mirror = reader.metric_views()[0]
    assert (view.name, view.source, view.comment, view.joins) == (mirror.name, mirror.source, mirror.comment, mirror.joins)
    assert view.dimensions == mirror.dimensions and view.measures == mirror.measures  # display names, comments and synonyms are back


def test_declared_keys_are_read_from_each_tables_own_entry(reader: DuckDBReader) -> None:
    constraints = {"demo.acme_shop.dealers": [{"primary_key_constraint": {"name": "pk", "child_columns": ["dealer_id"]}}],
                   "demo.acme_shop.claims": [{"foreign_key_constraint": {"name": "fk", "child_columns": ["dealer_id"], "parent_table": "demo.acme_shop.dealers", "parent_columns": ["dealer_id"]}}]}
    keys = _reader(StandIn(reader, constraints=constraints), schemas=("shop",)).declared_keys()
    assert [(k.object, k.columns, k.kind, k.references, k.referenced_columns) for k in keys] == [
        ("shop.claims", ("dealer_id",), "foreign", "shop.dealers", ("dealer_id",)), ("shop.dealers", ("dealer_id",), "primary", None, ())]


def test_a_refused_token_a_failed_statement_and_one_that_never_ends_each_stop_with_a_clear_word(reader: DuckDBReader) -> None:
    standin = StandIn(reader)
    with pytest.raises(WarehouseError, match="not allowed .HTTP 401."):
        _reader(standin, token="stale").query("SELECT 1")
    with pytest.raises(WarehouseError, match="not allowed .HTTP 403."):
        _reader(standin, token="no-rights").objects()
    assert len(standin.calls) == 2  # no second try with a token that was refused
    uc = _reader(standin)
    with pytest.raises(WarehouseError, match=r"statement s\d+ ended as FAILED: .*no_such_table"):
        uc.query("SELECT count(*) FROM `demo`.`acme_shop`.`no_such_table`")
    with pytest.raises(WarehouseError, match="PARSE_SYNTAX_ERROR"):
        uc.query('SELECT count("dealer_id") FROM `demo`.`acme_shop`.`dealers`')  # another dialect's quoting is an error here, not a wrong count
    standin.hang = True
    slow = DatabricksReader("https://workspace.invalid", "good", "demo", ("shop",), "wh1", schema_prefix="acme_", poll_seconds=0, give_up_after=0.0, http=standin)
    with pytest.raises(WarehouseError, match="was cancelled"):
        slow.query("SELECT 1")
    assert len(standin.cancelled) == 1


def test_values_come_back_in_the_types_the_result_states(reader: DuckDBReader) -> None:
    uc = _reader(StandIn(reader), schemas=("shop",))
    rows = uc.query("SELECT count(*), sum(`paid`), min(`claim_ref`), CAST(NULL AS STRING), count(*) > 0 FROM `demo`.`acme_shop`.`claims`")
    assert rows == [(2000, 3001500.0, "C1", None, True)] and isinstance(rows[0][0], int)
    assert len(uc.query("SELECT `claim_ref` FROM `demo`.`acme_shop`.`claims` ORDER BY 1 LIMIT 12")) == 12  # three parts of five, five and two


def test_the_reader_speaks_to_a_real_socket(reader: DuckDBReader) -> None:
    standin = StandIn(reader)

    class Handler(BaseHTTPRequestHandler):
        def _answer(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or None
            status, headers, raw = standin(self.command, self.path, dict(self.headers), body)
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        do_GET = do_POST = _answer

        def log_message(self, *args: Any) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        uc = DatabricksReader(f"http://127.0.0.1:{server.server_port}", "good", "demo", ("shop",), "wh1", schema_prefix="acme_", poll_seconds=0)
        assert {o.full_name: o.rows for o in uc.objects()}["shop.dealers"] == 40
    finally:
        server.shutdown()


def test_a_build_from_the_catalog_says_what_is_missing_in_the_environment(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from estate.cli import build, compare

    for name in ("DATABRICKS_HOST", "DATABRICKS_WAREHOUSE_ID", "DATABRICKS_TOKEN", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    cfg = tmp_path / "toy.yaml"
    cfg.write_text(yaml.safe_dump({"name": "toy", "schemas": ["shop"]}))
    with pytest.raises(WarehouseError, match="DATABRICKS_HOST, DATABRICKS_WAREHOUSE_ID"):
        build(cfg, None, None, None, tmp_path / "out", catalog="demo")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.invalid")
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "wh1")
    with pytest.raises(WarehouseError, match="DATABRICKS_TOKEN, or DATABRICKS_CLIENT_ID"):
        build(cfg, None, None, None, tmp_path / "out", catalog="demo")
    with pytest.raises(SystemExit, match="--duckdb FILE for a mirror, or --catalog NAME"):
        build(cfg, None, None, None, tmp_path / "out")
    one = {"concepts": [{"id": "concept:s.a", "kind": "entity", "key": ["id"], "status": "proposed", "label": "A"}], "relations": [], "measures": [], "structure": {}}
    other = {**one, "concepts": [{**one["concepts"][0], "key": ["code"]}, {"id": "concept:s.b", "kind": "fact", "key": [], "status": "proposed", "label": "B"}]}
    assert compare(one, other) == {"only_in_a": [], "only_in_b": ["concept:s.b"], "same": 0,
                                   "differ": {"concept:s.a": {"a": ["entity", ["id"], "proposed", "A"], "b": ["entity", ["code"], "proposed", "A"]}}}
