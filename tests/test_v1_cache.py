"""Tests for cache.py — make_cached_llm() with :memory: and file-backed SQLite."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from ragcheck.cache import make_cached_llm
from ragcheck.core import evaluate

QUESTION = "What is the capital of France?"
ANSWER = "The capital of France is Paris."
CONTEXTS = ["France is a country in Western Europe. Its capital city is Paris."]


def make_counting_llm(score=0.8):
    """Returns (llm_fn, call_count_list). call_count_list[0] increments per call."""
    call_count = [0]

    def llm(prompt):
        call_count[0] += 1
        return json.dumps({"score": score, "reasoning": "ok"})

    return llm, call_count


def test_cache_hit_does_not_call_llm_fn_again():
    llm, count = make_counting_llm()
    cached = make_cached_llm(llm, ":memory:")
    cached("same prompt")
    cached("same prompt")
    assert count[0] == 1  # only called once


def test_cache_miss_calls_llm_fn():
    llm, count = make_counting_llm()
    cached = make_cached_llm(llm, ":memory:")
    cached("prompt A")
    cached("prompt B")
    assert count[0] == 2  # different prompts → two calls


def test_different_prompts_get_different_entries():
    llm, count = make_counting_llm()
    cached = make_cached_llm(llm, ":memory:")
    r1 = cached("prompt one")
    r2 = cached("prompt two")
    assert count[0] == 2
    assert r1 == r2  # same mock response, but both were fetched


def test_in_memory_cache_works_within_session():
    llm, count = make_counting_llm(0.75)
    cached = make_cached_llm(llm, ":memory:")
    r1 = cached("hello")
    r2 = cached("hello")
    assert r1 == r2
    assert count[0] == 1


def test_file_cache_persists_across_calls():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    os.unlink(db_path)  # let cache create it fresh
    try:
        llm, count = make_counting_llm()
        cached1 = make_cached_llm(llm, db_path)
        cached1("persistent prompt")

        # New wrapped instance, same db file
        cached2 = make_cached_llm(llm, db_path)
        cached2("persistent prompt")

        assert count[0] == 1  # second call was a cache hit
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_cache_does_not_affect_score_values():
    llm, _ = make_counting_llm(0.65)
    cached = make_cached_llm(llm, ":memory:")
    result1 = evaluate(QUESTION, ANSWER, CONTEXTS, cached)
    result2 = evaluate(QUESTION, ANSWER, CONTEXTS, cached)
    assert result1.faithfulness == result2.faithfulness == 0.65


def test_cache_end_to_end_with_evaluate():
    llm, count = make_counting_llm()
    cached = make_cached_llm(llm, ":memory:")
    evaluate(QUESTION, ANSWER, CONTEXTS, cached)
    evaluate(QUESTION, ANSWER, CONTEXTS, cached)
    # 3 LLM calls for first evaluate, all cached on second
    assert count[0] == 3


def test_file_cache_creates_db_if_not_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "new_cache.db")
        assert not os.path.exists(db_path)
        llm, _ = make_counting_llm()
        cached = make_cached_llm(llm, db_path)
        cached("test")
        assert os.path.exists(db_path)


def test_memory_cache_cleared_between_separate_instances():
    """Each make_cached_llm(:memory:) call creates an isolated cache."""
    llm, count = make_counting_llm()
    cached1 = make_cached_llm(llm, ":memory:")
    cached2 = make_cached_llm(llm, ":memory:")
    cached1("same prompt")
    cached2("same prompt")  # different instance → cache miss
    assert count[0] == 2
