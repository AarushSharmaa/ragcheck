"""Tests for v1.0.0 core.py additions:
failure modes, token/cost tracking, to_markdown(), save_baseline(), assert_no_regression().
All tests use mock llm_fn — no real LLM needed.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

import ragcheck
from ragcheck.core import EvalResult, assert_no_regression, evaluate, PRICING

QUESTION = "What is the capital of France?"
ANSWER = "The capital of France is Paris."
CONTEXTS = ["France is a country in Western Europe. Its capital city is Paris."]


def make_mock_llm(score, reasoning="ok"):
    def mock(prompt):
        return json.dumps({"score": score, "reasoning": reasoning})
    return mock


def make_result(faithfulness=0.9, answer_relevance=0.9, context_precision=0.9,
                context_recall=None, failure_modes=None, tokens_used=0,
                estimated_cost_usd=0.0, extra_metrics=None):
    return EvalResult(
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_precision=context_precision,
        reasoning={
            "faithfulness": "ok",
            "answer_relevance": "ok",
            "context_precision": "ok",
        },
        context_recall=context_recall,
        failure_modes=failure_modes or [],
        tokens_used=tokens_used,
        estimated_cost_usd=estimated_cost_usd,
        extra_metrics=extra_metrics or {},
    )


# ---------------------------------------------------------------------------
# failure modes
# ---------------------------------------------------------------------------

def test_hallucination_label_when_faithfulness_low():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.3))
    assert "hallucination" in result.failure_modes


def test_off_topic_label_when_relevance_low():
    """Low answer_relevance → off_topic label."""
    call_count = [0]
    def mock(prompt):
        call_count[0] += 1
        # faithfulness=high, answer_relevance=low, context_precision=high
        scores = [0.9, 0.3, 0.9]
        idx = min(call_count[0] - 1, 2)
        return json.dumps({"score": scores[idx], "reasoning": "ok"})
    result = evaluate(QUESTION, ANSWER, CONTEXTS, mock)
    assert "off_topic" in result.failure_modes


def test_retrieval_miss_label_when_precision_low():
    """Low context_precision → retrieval_miss label."""
    call_count = [0]
    def mock(prompt):
        call_count[0] += 1
        scores = [0.9, 0.9, 0.3]
        idx = min(call_count[0] - 1, 2)
        return json.dumps({"score": scores[idx], "reasoning": "ok"})
    result = evaluate(QUESTION, ANSWER, CONTEXTS, mock)
    assert "retrieval_miss" in result.failure_modes


def test_context_noise_label_when_recall_low():
    """Low context_recall → context_noise label (only when include_context_recall=True)."""
    call_count = [0]
    def mock(prompt):
        call_count[0] += 1
        scores = [0.9, 0.9, 0.9, 0.3]
        idx = min(call_count[0] - 1, 3)
        return json.dumps({"score": scores[idx], "reasoning": "ok"})
    result = evaluate(QUESTION, ANSWER, CONTEXTS, mock, include_context_recall=True)
    assert "context_noise" in result.failure_modes


def test_multiple_labels_when_multiple_scores_low():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.2))
    assert "hallucination" in result.failure_modes
    assert "off_topic" in result.failure_modes
    assert "retrieval_miss" in result.failure_modes


def test_no_failure_modes_when_all_scores_high():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9))
    assert result.failure_modes == []


def test_failure_modes_empty_list_by_default():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    assert isinstance(result.failure_modes, list)


def test_failure_modes_in_to_dict():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.2))
    d = result.to_dict()
    assert "failure_modes" in d
    assert isinstance(d["failure_modes"], list)


def test_failure_modes_in_to_markdown():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.2))
    md = result.to_markdown()
    assert "failure_modes" in md.lower() or any(m in md for m in result.failure_modes)


def test_failure_modes_absent_from_to_dict_when_empty():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9))
    assert "failure_modes" not in result.to_dict()


# ---------------------------------------------------------------------------
# token + cost
# ---------------------------------------------------------------------------

def test_tokens_used_positive_after_eval():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    assert result.tokens_used > 0


def test_estimated_cost_zero_when_no_model():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    assert result.estimated_cost_usd == 0.0


def test_estimated_cost_correct_for_known_model():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), model="gpt-4o-mini")
    assert result.estimated_cost_usd > 0.0


def test_custom_pricing_overrides_builtin():
    r1 = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), model="gpt-4o")
    r2 = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8),
                  pricing={"input": 0.000001, "output": 0.000002})
    # Custom pricing should produce a different cost from built-in gpt-4o pricing
    assert r2.estimated_cost_usd != r1.estimated_cost_usd


def test_cost_scales_with_more_llm_calls():
    r_no_recall = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), model="gpt-4o-mini")
    r_with_recall = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8),
                             model="gpt-4o-mini", include_context_recall=True)
    assert r_with_recall.tokens_used > r_no_recall.tokens_used


def test_tokens_in_to_dict_when_nonzero():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), model="gpt-4o-mini")
    d = result.to_dict()
    assert "tokens_used" in d
    assert "estimated_cost_usd" in d


def test_cost_in_to_markdown_when_nonzero():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), model="gpt-4o-mini")
    md = result.to_markdown()
    assert "Cost" in md or "cost" in md


def test_unknown_model_name_does_not_crash():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), model="my-custom-model-v99")
    assert result.estimated_cost_usd == 0.0
    assert result.tokens_used > 0


# ---------------------------------------------------------------------------
# to_markdown()
# ---------------------------------------------------------------------------

def test_to_markdown_contains_all_core_metric_names():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    md = result.to_markdown()
    assert "faithfulness" in md
    assert "answer_relevance" in md
    assert "context_precision" in md


def test_to_markdown_contains_score_values():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.75))
    md = result.to_markdown()
    assert "0.75" in md


def test_to_markdown_contains_reasoning():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8, "looks good"))
    md = result.to_markdown()
    assert "looks good" in md


def test_to_markdown_contains_failure_modes_when_present():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.2))
    md = result.to_markdown()
    assert len(result.failure_modes) > 0
    assert any(m in md for m in result.failure_modes)


def test_to_markdown_omits_failure_modes_when_empty():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9))
    md = result.to_markdown()
    assert "hallucination" not in md
    assert "off_topic" not in md


def test_to_markdown_contains_cost_when_present():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), model="gpt-4o-mini")
    md = result.to_markdown()
    assert "Cost" in md or "$" in md


def test_to_markdown_includes_context_recall_when_present():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8), include_context_recall=True)
    md = result.to_markdown()
    assert "context_recall" in md


def test_to_markdown_includes_extra_metrics():
    result = evaluate(
        QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8),
        extra_metrics={"conciseness": lambda q, a, c: "rate conciseness"},
    )
    md = result.to_markdown()
    assert "conciseness" in md


def test_to_markdown_is_valid_markdown_table():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    md = result.to_markdown()
    assert "| Metric |" in md
    assert "|---|" in md


# ---------------------------------------------------------------------------
# save_baseline() + assert_no_regression()
# ---------------------------------------------------------------------------

def test_save_baseline_writes_json_file():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        result.save_baseline(path)
        assert os.path.exists(path)
    finally:
        os.unlink(path)


def test_save_baseline_file_is_valid_json():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        result.save_baseline(path)
        with open(path) as fh:
            data = json.load(fh)
        assert "faithfulness" in data
    finally:
        os.unlink(path)


def test_assert_no_regression_passes_when_equal():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    baseline = result.to_dict()
    assert_no_regression(baseline, result)  # should not raise


def test_assert_no_regression_passes_within_tolerance():
    baseline = make_result(faithfulness=0.8, answer_relevance=0.8, context_precision=0.8)
    new_result = make_result(faithfulness=0.76, answer_relevance=0.76, context_precision=0.76)
    assert_no_regression(baseline.to_dict(), new_result, tolerance=0.05)  # drop=0.04, within tolerance


def test_assert_no_regression_raises_when_beyond_tolerance():
    baseline = make_result(faithfulness=0.9)
    new_result = make_result(faithfulness=0.7)
    with pytest.raises(AssertionError):
        assert_no_regression(baseline.to_dict(), new_result, tolerance=0.05)


def test_assert_no_regression_error_message_names_metric():
    baseline = make_result(faithfulness=0.9)
    new_result = make_result(faithfulness=0.7)
    with pytest.raises(AssertionError, match="faithfulness"):
        assert_no_regression(baseline.to_dict(), new_result, tolerance=0.05)


def test_assert_no_regression_accepts_dict_input():
    baseline_dict = {"faithfulness": 0.8, "answer_relevance": 0.8, "context_precision": 0.8}
    new_result = make_result(faithfulness=0.8, answer_relevance=0.8, context_precision=0.8)
    assert_no_regression(baseline_dict, new_result)  # should not raise


def test_assert_no_regression_checks_context_recall_if_in_baseline():
    baseline = make_result(context_recall=0.9)
    new_result = make_result(context_recall=0.7)
    with pytest.raises(AssertionError, match="context_recall"):
        assert_no_regression(baseline.to_dict(), new_result, tolerance=0.05)


def test_save_and_reload_round_trips_correctly():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        result.save_baseline(path)
        assert_no_regression(path, result)  # load from file path — should not raise
    finally:
        os.unlink(path)
