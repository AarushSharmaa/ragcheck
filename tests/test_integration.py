"""
Integration tests — run against a real LLM (Groq).

Skipped automatically if GROQ_API_KEY is not set. Run explicitly with:

    pytest tests/test_integration.py -v

These tests do NOT assert exact scores (LLM output is non-deterministic).
They assert structural correctness and sanity-check that scores move in
the expected direction given clearly good or clearly bad inputs.
"""

import os

import pytest

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

requires_groq = pytest.mark.skipif(
    not GROQ_API_KEY,
    reason="GROQ_API_KEY not set — skipping integration tests",
)


def make_groq_llm_fn():
    """Build an llm_fn backed by Groq (llama-3.1-8b-instant)."""
    import urllib.request
    import json as _json

    def llm_fn(prompt: str) -> str:
        payload = _json.dumps({
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode()

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
                "User-Agent": "ragcheck-integration-tests/0.1.0",
            },
        )
        with urllib.request.urlopen(req) as resp:
            body = _json.loads(resp.read())
        return body["choices"][0]["message"]["content"]

    return llm_fn


# ---------------------------------------------------------------------------
# Structural correctness
# ---------------------------------------------------------------------------

@requires_groq
def test_real_llm_returns_evalresult_with_valid_scores():
    """Core contract: real LLM produces parseable scores in [0, 1]."""
    from ragcheck import evaluate

    llm_fn = make_groq_llm_fn()
    result = evaluate(
        question="What is photosynthesis?",
        answer="Photosynthesis is the process by which plants convert sunlight into food.",
        contexts=["Plants use sunlight, water, and carbon dioxide to produce glucose and oxygen through photosynthesis."],
        llm_fn=llm_fn,
    )

    assert 0.0 <= result.faithfulness <= 1.0
    assert 0.0 <= result.answer_relevance <= 1.0
    assert 0.0 <= result.context_precision <= 1.0
    assert isinstance(result.reasoning["faithfulness"], str)
    assert isinstance(result.reasoning["answer_relevance"], str)
    assert isinstance(result.reasoning["context_precision"], str)


@requires_groq
def test_real_llm_no_parse_errors_on_clean_input():
    """A well-formed eval should produce zero parse errors from a real LLM."""
    from ragcheck import evaluate

    llm_fn = make_groq_llm_fn()
    result = evaluate(
        question="What is photosynthesis?",
        answer="Photosynthesis is the process by which plants convert sunlight into food.",
        contexts=["Plants use sunlight, water, and carbon dioxide to produce glucose and oxygen through photosynthesis."],
        llm_fn=llm_fn,
    )

    assert result.parse_errors == [], (
        f"Unexpected parse errors from real LLM: {result.parse_errors}"
    )


# ---------------------------------------------------------------------------
# Score sanity — clearly good input should score high
# ---------------------------------------------------------------------------

@requires_groq
def test_faithful_answer_scores_high_faithfulness():
    """An answer that directly quotes the context should score high on faithfulness."""
    from ragcheck import evaluate

    llm_fn = make_groq_llm_fn()
    result = evaluate(
        question="What year did World War II end?",
        answer="World War II ended in 1945.",
        contexts=["World War II ended in 1945 with the surrender of Germany and Japan."],
        llm_fn=llm_fn,
    )

    assert result.faithfulness >= 0.7, (
        f"Expected faithfulness >= 0.7 for a grounded answer, got {result.faithfulness:.2f}. "
        f"Reasoning: {result.reasoning['faithfulness']}"
    )


@requires_groq
def test_relevant_answer_scores_high_relevance():
    """A direct, on-topic answer should score high on answer_relevance."""
    from ragcheck import evaluate

    llm_fn = make_groq_llm_fn()
    result = evaluate(
        question="What is the boiling point of water?",
        answer="The boiling point of water is 100 degrees Celsius at standard atmospheric pressure.",
        contexts=["Water boils at 100°C (212°F) at sea level."],
        llm_fn=llm_fn,
    )

    assert result.answer_relevance >= 0.7, (
        f"Expected answer_relevance >= 0.7 for a direct answer, got {result.answer_relevance:.2f}. "
        f"Reasoning: {result.reasoning['answer_relevance']}"
    )


@requires_groq
def test_relevant_context_scores_high_precision():
    """Context that directly answers the question should score high on context_precision."""
    from ragcheck import evaluate

    llm_fn = make_groq_llm_fn()
    result = evaluate(
        question="What is the speed of light?",
        answer="The speed of light is approximately 299,792 kilometres per second.",
        contexts=["The speed of light in a vacuum is exactly 299,792,458 metres per second."],
        llm_fn=llm_fn,
    )

    assert result.context_precision >= 0.7, (
        f"Expected context_precision >= 0.7 for highly relevant context, got {result.context_precision:.2f}. "
        f"Reasoning: {result.reasoning['context_precision']}"
    )


# ---------------------------------------------------------------------------
# Score sanity — clearly bad input should score low
# ---------------------------------------------------------------------------

@requires_groq
def test_hallucinated_answer_scores_low_faithfulness():
    """An answer that contradicts the context should score low on faithfulness."""
    from ragcheck import evaluate

    llm_fn = make_groq_llm_fn()
    result = evaluate(
        question="Who invented the telephone?",
        answer="The telephone was invented by Thomas Edison in 1877.",
        contexts=["Alexander Graham Bell is credited with inventing the telephone in 1876."],
        llm_fn=llm_fn,
    )

    assert result.faithfulness <= 0.4, (
        f"Expected faithfulness <= 0.4 for a hallucinated answer, got {result.faithfulness:.2f}. "
        f"Reasoning: {result.reasoning['faithfulness']}"
    )


@requires_groq
def test_off_topic_answer_scores_low_relevance():
    """An answer that doesn't address the question should score low on answer_relevance."""
    from ragcheck import evaluate

    llm_fn = make_groq_llm_fn()
    result = evaluate(
        question="What is the capital of Japan?",
        answer="Photosynthesis is the process by which plants make food from sunlight.",
        contexts=["Tokyo is the capital city of Japan."],
        llm_fn=llm_fn,
    )

    assert result.answer_relevance <= 0.4, (
        f"Expected answer_relevance <= 0.4 for an off-topic answer, got {result.answer_relevance:.2f}. "
        f"Reasoning: {result.reasoning['answer_relevance']}"
    )


@requires_groq
def test_irrelevant_context_scores_low_precision():
    """Context that doesn't relate to the question should score low on context_precision."""
    from ragcheck import evaluate

    llm_fn = make_groq_llm_fn()
    result = evaluate(
        question="What is the capital of Japan?",
        answer="Tokyo is the capital of Japan.",
        contexts=[
            "The Amazon rainforest produces 20% of the world's oxygen.",
            "Penguins are flightless birds native to the Southern Hemisphere.",
        ],
        llm_fn=llm_fn,
    )

    assert result.context_precision <= 0.4, (
        f"Expected context_precision <= 0.4 for irrelevant context, got {result.context_precision:.2f}. "
        f"Reasoning: {result.reasoning['context_precision']}"
    )


# ---------------------------------------------------------------------------
# passed() works end-to-end with real scores
# ---------------------------------------------------------------------------

@requires_groq
def test_passed_returns_bool_with_real_llm():
    """passed() should work correctly regardless of actual score values."""
    from ragcheck import evaluate

    llm_fn = make_groq_llm_fn()
    result = evaluate(
        question="What is photosynthesis?",
        answer="Photosynthesis is the process by which plants convert sunlight into food.",
        contexts=["Plants use sunlight, water, and carbon dioxide to produce glucose and oxygen."],
        llm_fn=llm_fn,
    )

    assert isinstance(result.passed(), bool)
    assert isinstance(result.passed(threshold=0.5), bool)
    assert isinstance(result.passed(threshold=0.99), bool)
