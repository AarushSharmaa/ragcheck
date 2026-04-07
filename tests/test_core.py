import json
from unittest.mock import Mock

from ragcheck import EvalResult, evaluate


def make_mock_llm(score: float, reasoning: str = "ok"):
    """Returns an llm_fn that always responds with the given score and reasoning."""
    def mock_llm(prompt: str) -> str:
        return json.dumps({"score": score, "reasoning": reasoning})
    return mock_llm


QUESTION = "What is the capital of France?"
ANSWER = "The capital of France is Paris."
CONTEXTS = ["France is a country in Western Europe. Its capital city is Paris."]


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_evaluate_returns_evalresult():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.9))
    assert isinstance(result, EvalResult)
    assert hasattr(result, "faithfulness")
    assert hasattr(result, "answer_relevance")
    assert hasattr(result, "context_precision")
    assert hasattr(result, "reasoning")


# ---------------------------------------------------------------------------
# Score values and clamping
# ---------------------------------------------------------------------------

def test_perfect_scores():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(1.0))
    assert result.faithfulness == 1.0
    assert result.answer_relevance == 1.0
    assert result.context_precision == 1.0


def test_zero_scores():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.0))
    assert result.faithfulness == 0.0
    assert result.answer_relevance == 0.0
    assert result.context_precision == 0.0


def test_scores_clamped_above_one():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(1.5))
    assert result.faithfulness == 1.0
    assert result.answer_relevance == 1.0
    assert result.context_precision == 1.0


def test_scores_clamped_below_zero():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(-0.2))
    assert result.faithfulness == 0.0
    assert result.answer_relevance == 0.0
    assert result.context_precision == 0.0


def test_score_as_string_is_coerced():
    """LLMs sometimes return {"score": "0.8"} instead of {"score": 0.8}."""
    llm = lambda prompt: '{"score": "0.8", "reasoning": "string score"}'
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    assert result.faithfulness == 0.8


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------

def test_reasoning_has_all_keys():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.8, "good answer"))
    assert set(result.reasoning.keys()) == {"faithfulness", "answer_relevance", "context_precision"}


def test_reasoning_values_match_llm_output():
    result = evaluate(QUESTION, ANSWER, CONTEXTS, make_mock_llm(0.5, "partial match"))
    assert result.reasoning["faithfulness"] == "partial match"
    assert result.reasoning["answer_relevance"] == "partial match"
    assert result.reasoning["context_precision"] == "partial match"


def test_missing_reasoning_key_in_response():
    """If LLM omits 'reasoning', the field should be an empty string, not an error."""
    llm = lambda prompt: '{"score": 0.7}'
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    assert result.faithfulness == 0.7
    assert result.reasoning["faithfulness"] == ""


def test_missing_score_key_defaults_to_zero():
    """If LLM omits 'score', default to 0.0."""
    llm = lambda prompt: '{"reasoning": "no score field here"}'
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    assert result.faithfulness == 0.0
    assert result.reasoning["faithfulness"] == "no score field here"


# ---------------------------------------------------------------------------
# JSON parse resilience
# ---------------------------------------------------------------------------

def test_json_parse_failure_returns_zero_with_error_message():
    llm = lambda prompt: "this is not json at all"
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    assert result.faithfulness == 0.0
    assert "JSON parse error" in result.reasoning["faithfulness"]


def test_embedded_json_extracted_from_prose():
    """LLMs often wrap JSON in explanation text — the fallback regex should handle this."""
    llm = lambda prompt: 'Here is my evaluation: {"score": 0.8, "reasoning": "well supported"}'
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    assert result.faithfulness == 0.8
    assert result.reasoning["faithfulness"] == "well supported"


def test_empty_response_returns_zero():
    llm = lambda prompt: ""
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    assert result.faithfulness == 0.0
    assert "JSON parse error" in result.reasoning["faithfulness"]


def test_bare_number_response_returns_zero():
    """A bare number is valid JSON but not a dict — should fall back to error."""
    llm = lambda prompt: "0.9"
    result = evaluate(QUESTION, ANSWER, CONTEXTS, llm)
    assert result.faithfulness == 0.0


# ---------------------------------------------------------------------------
# llm_fn call behaviour
# ---------------------------------------------------------------------------

def test_llm_called_exactly_three_times():
    mock = Mock(return_value=json.dumps({"score": 0.7, "reasoning": "ok"}))
    evaluate(QUESTION, ANSWER, CONTEXTS, mock)
    assert mock.call_count == 3


def test_llm_fn_exception_propagates():
    """Errors inside llm_fn should not be silently swallowed."""
    def failing_llm(prompt: str) -> str:
        raise RuntimeError("API call failed")

    try:
        evaluate(QUESTION, ANSWER, CONTEXTS, failing_llm)
        assert False, "Expected RuntimeError to propagate"
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Context variations
# ---------------------------------------------------------------------------

def test_multiple_contexts():
    contexts = [
        "Paris is the capital of France.",
        "France is known for the Eiffel Tower.",
        "The Seine river flows through Paris.",
    ]
    result = evaluate(QUESTION, ANSWER, contexts, make_mock_llm(0.9))
    assert isinstance(result, EvalResult)


def test_single_context():
    result = evaluate(QUESTION, ANSWER, ["Paris is in France."], make_mock_llm(0.9))
    assert isinstance(result, EvalResult)


def test_empty_contexts_does_not_raise():
    result = evaluate(QUESTION, ANSWER, [], make_mock_llm(0.5))
    assert isinstance(result, EvalResult)
