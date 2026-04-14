"""Tests for _async.py — aevaluate() and aevaluate_batch()."""
from __future__ import annotations

import asyncio
import json

import pytest

from ragcheck._async import aevaluate, aevaluate_batch
from ragcheck.core import EvalResult

QUESTION = "What is the capital of France?"
ANSWER = "The capital of France is Paris."
CONTEXTS = ["France is a country in Western Europe. Its capital city is Paris."]


def make_mock_llm(score=0.8, reasoning="ok"):
    def llm(prompt):
        return json.dumps({"score": score, "reasoning": reasoning})
    return llm


def run(coro):
    return asyncio.run(coro)


def test_aevaluate_returns_evalresult():
    result = run(aevaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm()))
    assert isinstance(result, EvalResult)


def test_aevaluate_scores_match_sync_on_same_mock():
    from ragcheck.core import evaluate
    llm = make_mock_llm(0.75)
    sync_result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    async_result = run(aevaluate(QUESTION, ANSWER, CONTEXTS, llm))
    assert sync_result.faithfulness == async_result.faithfulness
    assert sync_result.answer_relevance == async_result.answer_relevance
    assert sync_result.context_precision == async_result.context_precision


def test_aevaluate_propagates_llm_exception():
    def bad_llm(prompt):
        raise RuntimeError("LLM down")

    with pytest.raises(RuntimeError, match="LLM down"):
        run(aevaluate(QUESTION, ANSWER, CONTEXTS, bad_llm))


def test_aevaluate_batch_returns_correct_count():
    items = [
        {"question": QUESTION, "answer": ANSWER, "contexts": CONTEXTS},
        {"question": QUESTION, "answer": ANSWER, "contexts": CONTEXTS},
        {"question": QUESTION, "answer": ANSWER, "contexts": CONTEXTS},
    ]
    results = run(aevaluate_batch(items, make_mock_llm()))
    assert len(results) == 3


def test_aevaluate_batch_concurrency_1_is_sequential():
    results = run(aevaluate_batch(
        [{"question": QUESTION, "answer": ANSWER, "contexts": CONTEXTS}],
        make_mock_llm(),
        concurrency=1,
    ))
    assert len(results) == 1
    assert isinstance(results[0], EvalResult)


def test_aevaluate_batch_concurrency_n_returns_same_results():
    items = [{"question": QUESTION, "answer": ANSWER, "contexts": CONTEXTS}] * 5
    results = run(aevaluate_batch(items, make_mock_llm(0.6), concurrency=3))
    assert all(isinstance(r, EvalResult) for r in results)
    assert all(r.faithfulness == 0.6 for r in results)


def test_aevaluate_batch_respects_include_context_recall():
    items = [{"question": QUESTION, "answer": ANSWER, "contexts": CONTEXTS}]
    results = run(aevaluate_batch(items, make_mock_llm(), include_context_recall=True))
    assert results[0].context_recall is not None


def test_aevaluate_batch_empty_list_returns_empty():
    results = run(aevaluate_batch([], make_mock_llm()))
    assert results == []
