# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The pure S3 building blocks: corpus preps, budget model, and — the load-bearing half —
every generated pipeline.yaml variant validated through the REAL latence-core Pipeline
model (``extra="forbid"`` at root and stage level, so a typo'd key fails HERE) plus the
real ``latence stack check``."""

from __future__ import annotations

import json

import pytest
from latence_benchmark.campaign import s3_lib
from latence_benchmark.campaign.pipeline_stage import PROCESS_ORDER
from latence_benchmark.campaign.s3_lib import (
    CHUNK_ESTIMATES,
    CHUNK_HEADROOM,
    EST_USD_PER_CALL_UNKNOWN,
    PrepFile,
    est_usd_per_call,
    generate_pipeline_yaml,
    induction_call_budgets,
    induction_usd_actual,
    multihop_rag_articles,
    slugify,
    wiki2multihop_paragraphs,
)
from latence_benchmark.campaign.stages import StageBlockedError
from latence_benchmark.campaign.wizard import S3Answers
from latence_core.pipeline import CapabilityKind, Pipeline
from latence_core.stacks.check import check_stack

# -- corpus preps ----------------------------------------------------------------------------


def article(title: str, body: str) -> dict[str, str]:
    return {"title": title, "body": body, "source": "Fixture News", "url": "https://x"}


class TestMultihopRagPrep:
    def test_articles_become_content_addressed_markdown(self) -> None:
        files = multihop_rag_articles(
            json.dumps([article("Big Deal!", "Alpha met Beta."), article("Two", "Gamma.")])
        )
        assert len(files) == 2
        by_name = {f.filename: f for f in files}
        assert any(name.startswith("big-deal-") for name in by_name)
        first = files[0]
        assert first.content.startswith("# ")
        assert first.content.endswith("\n")

    def test_duplicate_content_is_deduped(self) -> None:
        rows = [article("Same", "Body."), article("Same", "Body."), article("Other", "B2.")]
        assert len(multihop_rag_articles(json.dumps(rows))) == 2

    def test_output_is_deterministic_and_order_independent(self) -> None:
        rows = [article("B", "two"), article("A", "one")]
        a = multihop_rag_articles(json.dumps(rows))
        b = multihop_rag_articles(json.dumps(list(reversed(rows))))
        assert a == b  # sorted by content-addressed filename, not input order
        assert [f.filename for f in a] == sorted(f.filename for f in a)

    def test_empty_bodies_are_skipped(self) -> None:
        rows = [article("Empty", "   "), article("Real", "text")]
        files = multihop_rag_articles(json.dumps(rows))
        assert len(files) == 1

    def test_malformed_json_blocks(self) -> None:
        with pytest.raises(StageBlockedError, match="not valid JSON"):
            multihop_rag_articles("{nope")

    def test_all_empty_blocks(self) -> None:
        with pytest.raises(StageBlockedError, match="zero non-empty"):
            multihop_rag_articles(json.dumps([article("a", "")]))


def context_cell(*paragraphs: tuple[str, list[str]]) -> str:
    return json.dumps([[title, sentences] for title, sentences in paragraphs])


class TestWiki2Prep:
    def test_one_file_per_distinct_paragraph_across_questions(self) -> None:
        # The same paragraph appears under two questions (the question-pool overlap).
        shared = ("Xawery", ["Director.", "Son of an actress."])
        cells = [
            context_cell(shared, ("Polish-Russian War", ["A 2009 film."])),
            context_cell(shared, ("Maheen Khan", ["A designer."])),
        ]
        files = wiki2multihop_paragraphs(cells)
        assert len(files) == 3
        assert [f.filename for f in files] == sorted(f.filename for f in files)

    def test_title_is_the_heading_and_sentences_join(self) -> None:
        files = wiki2multihop_paragraphs([context_cell(("T", ["S1.", " S2. "]))])
        assert files[0].content == "# T\n\nS1. S2.\n"

    def test_malformed_paragraph_shape_blocks(self) -> None:
        with pytest.raises(StageBlockedError, match="shape"):
            wiki2multihop_paragraphs([json.dumps([["only-title"]])])

    def test_malformed_json_blocks(self) -> None:
        with pytest.raises(StageBlockedError, match="not valid JSON"):
            wiki2multihop_paragraphs(["<html>"])


class TestSlug:
    def test_slugs_are_filename_safe(self) -> None:
        assert slugify("Polish-Russian War (Film)!") == "polish-russian-war-film"
        assert slugify("Überraschung / 100%") == "berraschung-100"
        assert slugify("!!!") == "doc"


# -- the budget model ------------------------------------------------------------------------


DEEPSEEK = "deepseek/deepseek-v4-flash-0731"
GPT41MINI = "openai/gpt-4.1-mini"


class TestBudget:
    def test_per_model_call_estimates_are_measured_plus_headroom(self) -> None:
        assert est_usd_per_call(DEEPSEEK) == 0.00015  # measured 0.000118 + headroom
        assert est_usd_per_call(GPT41MINI) == 0.00050  # measured 0.00042 + headroom
        # An unknown model falls back to the old conservative scaffold estimate.
        assert est_usd_per_call("acme/mystery-model") == EST_USD_PER_CALL_UNKNOWN
        assert est_usd_per_call("acme/mystery-model") == pytest.approx(0.000992)

    def test_allocation_is_proportional_to_the_measured_chunk_counts(self) -> None:
        # Under a roomy cap every known dataset gets its full chunk estimate × headroom —
        # uda (220k chunks) is no longer starved down to an even third of the cap.
        budgets = induction_call_budgets(150.0, DEEPSEEK, ["ohr_bench", "vidoseek", "uda"])
        assert budgets == {
            ds: int(CHUNK_ESTIMATES[ds] * CHUNK_HEADROOM)
            for ds in ("ohr_bench", "vidoseek", "uda")
        }
        assert budgets["uda"] > budgets["ohr_bench"] > budgets["vidoseek"]

    def test_allocations_scale_down_proportionally_to_the_cap(self) -> None:
        # A cap far below the full-corpus estimate: every allocation shrinks by the same
        # factor and the summed spend estimate never exceeds the cap (the hard bound).
        cap = 1.0
        budgets = induction_call_budgets(cap, DEEPSEEK, ["ohr_bench", "vidoseek", "uda"])
        per_call = est_usd_per_call(DEEPSEEK)
        assert sum(budgets.values()) * per_call <= cap
        full = {ds: int(CHUNK_ESTIMATES[ds] * CHUNK_HEADROOM) for ds in budgets}
        assert all(budgets[ds] < full[ds] for ds in budgets)  # every dataset scaled
        # Proportionality survives the scaling (within integer flooring).
        assert budgets["uda"] > budgets["ohr_bench"] > budgets["vidoseek"] > 0

    def test_unknown_dataset_gets_an_even_share_of_the_remainder(self) -> None:
        budgets = induction_call_budgets(150.0, DEEPSEEK, ["ohr_bench", "musique"])
        per_call = est_usd_per_call(DEEPSEEK)
        known = int(CHUNK_ESTIMATES["ohr_bench"] * CHUNK_HEADROOM)
        assert budgets["ohr_bench"] == known
        remaining = 150.0 - known * per_call
        assert budgets["musique"] == int(remaining / 1 / per_call)
        assert sum(budgets.values()) * per_call <= 150.0

    def test_no_induced_datasets_means_no_allocations(self) -> None:
        assert induction_call_budgets(150.0, DEEPSEEK, []) == {}

    def test_generation_is_never_blocked_by_a_tiny_cap(self) -> None:
        # The cap is a hard bound, not a gate: even a near-zero cap yields allocations
        # (possibly 0 calls), never an exception.
        budgets = induction_call_budgets(0.001, DEEPSEEK, ["ohr_bench", "uda"])
        assert set(budgets) == {"ohr_bench", "uda"}
        assert sum(budgets.values()) * est_usd_per_call(DEEPSEEK) <= 0.001

    def test_usd_actuals_price_by_the_model_card(self) -> None:
        # deepseek: 100k prompt @ $0.09/M + 20k completion @ $0.18/M = $0.0126
        assert induction_usd_actual(100_000, 20_000, DEEPSEEK) == pytest.approx(0.0126)
        # gpt-4.1-mini: 100k @ $0.40/M + 20k @ $1.60/M = $0.072
        assert induction_usd_actual(100_000, 20_000, GPT41MINI) == pytest.approx(0.072)
        # Unknown/empty model prices at the conservative gpt-4.1-mini card, never zero.
        assert induction_usd_actual(100_000, 20_000) == pytest.approx(0.072)
        assert induction_usd_actual(100_000, 20_000, "acme/x") == pytest.approx(0.072)


# -- the generated pipelines -----------------------------------------------------------------

VARIANTS: list[tuple[str, S3Answers, bool]] = [
    # (dataset, s3 config, induce) — every dataset shape × every induction mode.
    *[(ds, S3Answers(), ds in ("ohr_bench", "vidoseek", "uda")) for ds in PROCESS_ORDER],
    ("ohr_bench", S3Answers(induction_mode="doc"), True),
    ("ohr_bench", S3Answers(induction_mode="none", induction_budget_usd_cap=0.0), False),
]


class TestGeneratedPipelines:
    @pytest.mark.parametrize(("dataset", "s3", "induce"), VARIANTS)
    def test_every_variant_validates_through_the_real_core_model(
        self, dataset: str, s3: S3Answers, induce: bool
    ) -> None:
        text = generate_pipeline_yaml(dataset, s3, induce=induce, max_induction_calls=1000)
        pipeline = Pipeline.from_yaml_text(text)  # extra="forbid": typos die here
        assert pipeline.name == f"s3-{dataset}"
        check = check_stack(text)
        assert not check.errors, check.render()

    def test_byte_determinism(self) -> None:
        s3 = S3Answers()
        a = generate_pipeline_yaml("uda", s3, induce=True, max_induction_calls=5)
        b = generate_pipeline_yaml("uda", s3, induce=True, max_induction_calls=5)
        assert a == b

    def test_induced_wiring(self) -> None:
        text = generate_pipeline_yaml(
            "ohr_bench", S3Answers(), induce=True, max_induction_calls=1234
        )
        pipeline = Pipeline.from_yaml_text(text)
        by_name = {s.name: s for s in pipeline.stages}

        # Content screening BEFORE induction: never spend LLM calls on quarantined text.
        induce = by_name["induce"]
        assert induce.depends_on == ["content_screen"]
        assert induce.config["granularity"] == "chunk"
        assert induce.config["max_induction_calls"] == 1234
        assert induce.config["base_url"] == "https://openrouter.ai/api/v1"
        # The default ruling: deepseek with the OpenRouter reasoning knob pinned OFF.
        assert induce.config["model"] == "deepseek/deepseek-v4-flash-0731"
        assert induce.config["reasoning"] is False
        # The campaign-scale fan-out (provider default is 8; 32 keeps UDA's 242k calls ~5h).
        assert induce.config["max_concurrency"] == 32

        # Extraction + redaction ride the induced chunk stream (induced ∪ floor labels).
        assert by_name["extract"].depends_on == ["induce"]
        assert by_name["redact"].depends_on == ["induce"]
        # Lean floor under induction — the induced schema does the heavy lifting.
        assert by_name["extract"].config["labels"] == ["person", "organization"]

    def test_reasoning_none_omits_the_knob_entirely(self) -> None:
        # None must not emit `reasoning: null` — the inducer then sends no extra_body at
        # all, the safe shape for a strict vLLM endpoint that rejects unknown body fields.
        text = generate_pipeline_yaml(
            "ohr_bench",
            S3Answers(induction_reasoning=None),
            induce=True,
            max_induction_calls=10,
        )
        pipeline = Pipeline.from_yaml_text(text)
        induce = next(s for s in pipeline.stages if s.name == "induce")
        assert "reasoning" not in induce.config

    def test_reasoning_true_flows_through(self) -> None:
        text = generate_pipeline_yaml(
            "ohr_bench",
            S3Answers(induction_reasoning=True),
            induce=True,
            max_induction_calls=10,
        )
        pipeline = Pipeline.from_yaml_text(text)
        induce = next(s for s in pipeline.stages if s.name == "induce")
        assert induce.config["reasoning"] is True

    def test_doc_mode_maps_to_doc_granularity(self) -> None:
        text = generate_pipeline_yaml(
            "uda", S3Answers(induction_mode="doc"), induce=True, max_induction_calls=10
        )
        pipeline = Pipeline.from_yaml_text(text)
        induce = next(s for s in pipeline.stages if s.name == "induce")
        assert induce.config["granularity"] == "doc"

    def test_text_native_wiring_uses_the_hand_authored_schema(self) -> None:
        s3 = S3Answers()
        text = generate_pipeline_yaml("multihop_rag", s3, induce=False)
        pipeline = Pipeline.from_yaml_text(text)
        by_name = {s.name: s for s in pipeline.stages}
        assert "induce" not in by_name
        assert by_name["extract"].depends_on == ["content_screen"]
        assert by_name["extract"].config["labels"] == list(s3.entity_labels)
        assert by_name["extract"].config["relation_labels"] == list(s3.relation_labels)

    def test_the_structural_invariants_of_the_inventory(self) -> None:
        text = generate_pipeline_yaml("ohr_bench", S3Answers(), induce=True)
        pipeline = Pipeline.from_yaml_text(text)
        by_name = {s.name: s for s in pipeline.stages}

        # extract_on unmasked: extraction reads raw chunks, masking hits the export only.
        assert pipeline.extract_on.value == "unmasked"
        # Hyperedges ride graph.canonical ONLY (the inventory gotcha), with the KV export.
        graph = by_name["graph"]
        assert graph.provider == "graph.canonical"
        assert graph.config["hyperedges"]["max_arity"] == 12
        assert graph.config["graph_features"] == {"provider": "graph_features.degree"}
        assert by_name["export_kg"].config["hyperedge_kv"] is True
        # The linear full-corpus resolution path (the M6 cascade-wall lesson).
        assert by_name["consolidate_types"].provider == "type_consolidation.exact_surface"
        assert by_name["disambiguate"].provider == "disambiguation.exact_surface"
        assert by_name["disambiguate"].depends_on == ["consolidate_types"]
        # Corpus export: masked+enriched rows, cuda ST embedder, BM25, context columns.
        corpus = by_name["export_corpus"]
        assert corpus.depends_on == ["enrich"]
        assert corpus.config["embedder"]["config"]["device"] == "cuda"
        assert corpus.config["bm25"] == {"provider": "tokenizer.regex"}
        assert corpus.config["context_columns"] is True
        # Redaction: learned gliner2 with guardrails ON + the PII floor.
        redact = by_name["redact"]
        assert redact.provider == "redaction.gliner2"
        assert redact.config["guardrails"] is True
        assert "credit card number" in redact.config["pii_labels"]
        # Every stage that is not the source declares explicit depends_on wiring.
        for stage in pipeline.stages:
            if stage.capability is not CapabilityKind.SOURCE:
                assert stage.depends_on, f"{stage.name} has no depends_on"

    def test_chunk_knobs_flow_from_the_campaign_config(self) -> None:
        s3 = S3Answers(chunk_max_tokens=512, chunk_overlap_tokens=64, chunk_min_tokens=16)
        pipeline = Pipeline.from_yaml_text(generate_pipeline_yaml("uda", s3, induce=False))
        chunk = next(s for s in pipeline.stages if s.name == "chunk")
        assert chunk.config == {"max_tokens": 512, "overlap_tokens": 64, "min_tokens": 16}

    def test_storage_and_source_are_pod_rooted_per_dataset(self) -> None:
        pipeline = Pipeline.from_yaml_text(
            generate_pipeline_yaml("vidoseek", S3Answers(), induce=True)
        )
        assert pipeline.storage_uri == f"file://{s3_lib.REMOTE_ROOT}/vidoseek"
        source = next(s for s in pipeline.stages if s.name == "source")
        assert source.config["path"] == f"file://{s3_lib.REMOTE_ROOT}/input/vidoseek/markdown"


# -- quality-report readers ------------------------------------------------------------------


def fixture_report(*, with_induction: bool = True) -> dict[str, object]:
    stages: list[dict[str, object]] = [
        {"name": "chunk", "capability": "chunk", "provider": "chunk.markdown"},
    ]
    if with_induction:
        stages.append(
            {
                "name": "induce",
                "capability": "schema_induction",
                "provider": "label_inducer.llm",
                "induction_calls": 100,
                "induction_cache_hits": 40,
                "induction_prompt_tokens": 100_000,
                "induction_completion_tokens": 20_000,
                "induction_budget_exhausted": False,
            }
        )
    return {
        "pipeline_name": "s3-x",
        "run_id": "s3-x-v1",
        "document_count": 3,
        "stages": stages,
        "chunk": {"chunks_total": 42, "chunks_flagged": 1},
        "entities": {"mentions_total": 120},
        "relations": {"relations_total": 30},
        "redaction": {"pii_total": 5, "per_type": {"email": 5}},
        "graph": {
            "nodes_total": 80,
            "edges_total": 25,
            "hyperedges": {"hyperedges_total": 12},
        },
    }


class TestReportReader:
    def test_the_gate_numbers_are_lifted_from_the_typed_sections(self) -> None:
        metrics = s3_lib.dataset_metrics_from_report(fixture_report())
        assert metrics["documents"] == 3
        assert metrics["chunks"] == 42
        assert metrics["mentions"] == 120
        assert metrics["relations"] == 30
        assert metrics["entities"] == 80
        assert metrics["edges"] == 25
        assert metrics["hyperedges"] == 12
        assert metrics["pii_total"] == 5
        induction = metrics["induction"]
        assert induction is not None
        assert induction["calls"] == 100
        assert induction["cache_hits"] == 40
        # No model passed: the conservative fallback card (gpt-4.1-mini prices).
        assert induction["usd_actual"] == pytest.approx(0.072)

    def test_the_usd_actuals_use_the_campaign_models_price_card(self) -> None:
        metrics = s3_lib.dataset_metrics_from_report(
            fixture_report(), induction_model="deepseek/deepseek-v4-flash-0731"
        )
        induction = metrics["induction"]
        assert induction is not None
        assert induction["usd_actual"] == pytest.approx(0.0126)

    def test_a_run_without_induction_reports_none(self) -> None:
        metrics = s3_lib.dataset_metrics_from_report(fixture_report(with_induction=False))
        assert metrics["induction"] is None

    def test_missing_sections_read_as_zero_not_a_crash(self) -> None:
        metrics = s3_lib.dataset_metrics_from_report({"document_count": 1})
        assert metrics["chunks"] == 0
        assert metrics["hyperedges"] == 0


class TestPrepFileShape:
    def test_prep_files_are_frozen_value_objects(self) -> None:
        prep = PrepFile(filename="a.md", content="# A\n")
        with pytest.raises(AttributeError):
            prep.filename = "b.md"  # type: ignore[misc]
