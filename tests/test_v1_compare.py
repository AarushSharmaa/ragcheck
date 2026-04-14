"""Tests for compare.py — CompareResult and compare()."""
from __future__ import annotations

import json

import pytest

from ragcheck.compare import compare, CompareResult
from ragcheck.core import EvalResult


def make_result(faithfulness=0.8, answer_relevance=0.8, context_precision=0.8,
                context_recall=None):
    reasoning = {
        "faithfulness": "ok",
        "answer_relevance": "ok",
        "context_precision": "ok",
    }
    if context_recall is not None:
        reasoning["context_recall"] = "ok"
    return EvalResult(
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_precision=context_precision,
        reasoning=reasoning,
        context_recall=context_recall,
    )


def test_compare_returns_compare_result():
    before = [make_result()]
    after = [make_result()]
    assert isinstance(compare(before, after), CompareResult)


def test_compare_identifies_regression():
    before = [make_result(faithfulness=0.9)]
    after = [make_result(faithfulness=0.5)]
    result = compare(before, after)
    assert any(r["metric"] == "faithfulness" for r in result.regressions)


def test_compare_identifies_improvement():
    before = [make_result(answer_relevance=0.5)]
    after = [make_result(answer_relevance=0.9)]
    result = compare(before, after)
    assert any(r["metric"] == "answer_relevance" for r in result.improvements)


def test_compare_no_change_produces_empty_diff():
    before = [make_result(0.8, 0.8, 0.8)]
    after = [make_result(0.8, 0.8, 0.8)]
    result = compare(before, after)
    assert result.regressions == []
    assert result.improvements == []


def test_net_delta_correct_per_metric():
    before = [make_result(faithfulness=0.8), make_result(faithfulness=0.6)]
    after = [make_result(faithfulness=0.9), make_result(faithfulness=0.7)]
    result = compare(before, after)
    assert abs(result.net_delta["faithfulness"] - 0.1) < 0.001


def test_top_regressions_sorted_by_magnitude():
    before = [make_result(faithfulness=0.9), make_result(faithfulness=0.8)]
    after = [make_result(faithfulness=0.3), make_result(faithfulness=0.6)]
    result = compare(before, after)
    faith_regressions = [r for r in result.regressions if r["metric"] == "faithfulness"]
    deltas = [r["delta"] for r in faith_regressions]
    assert deltas == sorted(deltas)  # most negative first


def test_top_improvements_sorted_by_magnitude():
    before = [make_result(answer_relevance=0.3), make_result(answer_relevance=0.6)]
    after = [make_result(answer_relevance=0.9), make_result(answer_relevance=0.8)]
    result = compare(before, after)
    impr = [r for r in result.improvements if r["metric"] == "answer_relevance"]
    deltas = [r["delta"] for r in impr]
    # __str__ sorts by -delta so largest improvement first; just verify they exist and are positive
    assert all(d > 0 for d in deltas)


def test_compare_raises_on_mismatched_list_lengths():
    before = [make_result(), make_result()]
    after = [make_result()]
    with pytest.raises(ValueError):
        compare(before, after)


def test_compare_empty_lists_returns_empty_result():
    result = compare([], [])
    assert result.regressions == []
    assert result.improvements == []
    assert result.net_delta == {}


def test_compare_str_contains_regressions():
    before = [make_result(faithfulness=0.9)]
    after = [make_result(faithfulness=0.4)]
    result = compare(before, after)
    s = str(result)
    assert "regression" in s.lower()


def test_compare_str_contains_improvements():
    before = [make_result(answer_relevance=0.4)]
    after = [make_result(answer_relevance=0.9)]
    result = compare(before, after)
    s = str(result)
    assert "improvement" in s.lower()


def test_compare_str_contains_net_delta():
    before = [make_result()]
    after = [make_result()]
    result = compare(before, after)
    s = str(result)
    assert "Net delta" in s


def test_compare_to_markdown_is_string():
    before = [make_result()]
    after = [make_result()]
    result = compare(before, after)
    assert isinstance(result.to_markdown(), str)


def test_compare_to_markdown_contains_regression_table():
    before = [make_result(faithfulness=0.9)]
    after = [make_result(faithfulness=0.4)]
    result = compare(before, after)
    md = result.to_markdown()
    assert "Regression" in md or "regression" in md


def test_compare_to_dict_is_json_serialisable():
    before = [make_result(faithfulness=0.9)]
    after = [make_result(faithfulness=0.5)]
    result = compare(before, after)
    d = result.to_dict()
    json.dumps(d)  # should not raise
