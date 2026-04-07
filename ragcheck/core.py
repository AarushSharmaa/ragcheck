from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional


@dataclass
class EvalResult:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    reasoning: dict[str, str]
    parse_errors: list[str] = field(default_factory=list)
    context_recall: Optional[float] = None
    extra_metrics: dict[str, float] = field(default_factory=dict)

    def passed(self, threshold: float = 0.7) -> bool:
        """Return True if all scored metrics meet the threshold.

        Includes context_recall and extra_metrics if present.
        Useful as a boolean gate in CI pipelines:

            if not result.passed(threshold=0.8):
                raise ValueError("RAG quality below threshold")
        """
        core = [self.faithfulness, self.answer_relevance, self.context_precision]
        if self.context_recall is not None:
            core.append(self.context_recall)
        core.extend(self.extra_metrics.values())
        return all(s >= threshold for s in core)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of all scores and metadata."""
        d = {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "reasoning": self.reasoning,
            "parse_errors": self.parse_errors,
        }
        if self.context_recall is not None:
            d["context_recall"] = self.context_recall
        if self.extra_metrics:
            d["extra_metrics"] = self.extra_metrics
        return d

    def to_json(self, **kwargs) -> str:
        """Return a JSON string of to_dict(). kwargs are passed to json.dumps."""
        return json.dumps(self.to_dict(), **kwargs)

    def __str__(self) -> str:
        lines = [
            f"  faithfulness      {self.faithfulness:.2f}  {self.reasoning.get('faithfulness', '')}",
            f"  answer_relevance  {self.answer_relevance:.2f}  {self.reasoning.get('answer_relevance', '')}",
            f"  context_precision {self.context_precision:.2f}  {self.reasoning.get('context_precision', '')}",
        ]
        if self.context_recall is not None:
            lines.append(f"  context_recall    {self.context_recall:.2f}  {self.reasoning.get('context_recall', '')}")
        for name, score in self.extra_metrics.items():
            lines.append(f"  {name:<18}{score:.2f}  {self.reasoning.get(name, '')}")
        if self.parse_errors:
            lines.append(f"  parse_errors      {self.parse_errors}")
        return "\n".join(lines)


def _format_contexts(contexts: List[str]) -> str:
    if not contexts:
        return "(no context provided)"
    return "\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts))


def _parse_llm_json(response: str) -> tuple[float, str, str | None]:
    """Parse LLM response into (score, reasoning, error).

    Returns a non-None error string if parsing failed — callers use this
    to populate EvalResult.parse_errors so silent failures are visible.
    """
    def extract(obj: dict) -> tuple[float, str, str | None]:
        score = float(obj.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        reasoning = str(obj.get("reasoning", ""))
        return score, reasoning, None

    try:
        parsed = json.loads(response.strip())
        if isinstance(parsed, dict):
            return extract(parsed)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return extract(parsed)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    error = f"JSON parse error: {response[:200]!r}"
    return 0.0, error, error


def _faithfulness_prompt(question: str, answer: str, contexts: List[str]) -> str:
    return f"""You are evaluating whether an answer is faithful to the provided context.

Question: {question}

Answer: {answer}

Context:
{_format_contexts(contexts)}

Score 1.0 if every claim in the answer is directly supported by the context.
Score 0.0 if the answer contains claims not found in the context.
Score between 0 and 1 proportionally for partial support.

Respond ONLY with valid JSON: {{"score": <float 0-1>, "reasoning": "<one sentence>"}}"""


def _answer_relevance_prompt(question: str, answer: str) -> str:
    return f"""You are evaluating whether an answer is relevant to the question asked.

Question: {question}

Answer: {answer}

Score 1.0 if the answer directly and completely addresses the question.
Score 0.0 if the answer is completely off-topic or refuses to answer.
Score between 0 and 1 for partial relevance.

Respond ONLY with valid JSON: {{"score": <float 0-1>, "reasoning": "<one sentence>"}}"""


def _context_precision_prompt(question: str, contexts: List[str]) -> str:
    return f"""You are evaluating whether the retrieved context is relevant and useful for answering the question.

Question: {question}

Context:
{_format_contexts(contexts)}

Score 1.0 if all context chunks are highly relevant to answering the question.
Score 0.0 if none of the context chunks are relevant.
Score between 0 and 1 proportionally based on the fraction of relevant chunks.

Respond ONLY with valid JSON: {{"score": <float 0-1>, "reasoning": "<one sentence>"}}"""


def _context_recall_prompt(question: str, answer: str, contexts: List[str]) -> str:
    return f"""You are evaluating whether the retrieved context contains enough information to support the answer.

Question: {question}

Answer: {answer}

Context:
{_format_contexts(contexts)}

Score 1.0 if the context covers all the information needed to produce the answer.
Score 0.0 if the context is missing most of the information needed.
Score between 0 and 1 proportionally for partial coverage.

Respond ONLY with valid JSON: {{"score": <float 0-1>, "reasoning": "<one sentence>"}}"""


def evaluate(
    question: str,
    answer: str,
    contexts: List[str],
    llm_fn: Callable[[str], str],
    *,
    include_context_recall: bool = False,
    extra_metrics: Optional[dict[str, Callable[[str, str, List[str]], str]]] = None,
) -> EvalResult:
    """
    Evaluate a RAG response across three core metrics (plus optional extras).

    Args:
        question: The user's original question.
        answer: The answer generated by the RAG system.
        contexts: The list of retrieved context chunks.
        llm_fn: Any callable that takes a prompt string and returns a string.
        include_context_recall: If True, also score context_recall (does the
            context contain enough info to produce the answer?). Adds one
            extra LLM call.
        extra_metrics: Optional dict mapping metric name → prompt function.
            Each prompt function receives (question, answer, contexts) and
            must return a prompt string. The LLM response is parsed with the
            same JSON parser as core metrics. Example:

                def my_prompt(q, a, c): return f"Rate conciseness... {a}"
                result = evaluate(..., extra_metrics={"conciseness": my_prompt})

    Returns:
        EvalResult with faithfulness, answer_relevance, context_precision scores (0-1),
        optional context_recall, optional extra_metrics scores, per-metric reasoning
        strings, and a parse_errors list.
    """
    if not callable(llm_fn):
        raise TypeError(f"llm_fn must be callable, got {type(llm_fn).__name__!r}")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    if not isinstance(contexts, list):
        raise TypeError(f"contexts must be a list, got {type(contexts).__name__!r}")

    faith_score, faith_reason, faith_err = _parse_llm_json(
        llm_fn(_faithfulness_prompt(question, answer, contexts))
    )
    relevance_score, relevance_reason, relevance_err = _parse_llm_json(
        llm_fn(_answer_relevance_prompt(question, answer))
    )
    precision_score, precision_reason, precision_err = _parse_llm_json(
        llm_fn(_context_precision_prompt(question, contexts))
    )

    parse_errors = [e for e in [faith_err, relevance_err, precision_err] if e is not None]
    reasoning = {
        "faithfulness": faith_reason,
        "answer_relevance": relevance_reason,
        "context_precision": precision_reason,
    }

    recall_score = None
    if include_context_recall:
        recall_score, recall_reason, recall_err = _parse_llm_json(
            llm_fn(_context_recall_prompt(question, answer, contexts))
        )
        reasoning["context_recall"] = recall_reason
        if recall_err:
            parse_errors.append(recall_err)

    extra_scores: dict[str, float] = {}
    if extra_metrics:
        for name, prompt_fn in extra_metrics.items():
            score, reason, err = _parse_llm_json(
                llm_fn(prompt_fn(question, answer, contexts))
            )
            extra_scores[name] = score
            reasoning[name] = reason
            if err:
                parse_errors.append(err)

    return EvalResult(
        faithfulness=faith_score,
        answer_relevance=relevance_score,
        context_precision=precision_score,
        reasoning=reasoning,
        parse_errors=parse_errors,
        context_recall=recall_score,
        extra_metrics=extra_scores,
    )


def evaluate_batch(
    items: List[dict],
    llm_fn: Callable[[str], str],
    **evaluate_kwargs,
) -> List[EvalResult]:
    """
    Evaluate a list of RAG responses.

    Each item in `items` must be a dict with keys: "question", "answer", "contexts".
    Any kwargs are forwarded to evaluate() (e.g. include_context_recall=True).

    Example:
        results = evaluate_batch([
            {"question": "...", "answer": "...", "contexts": [...]},
            {"question": "...", "answer": "...", "contexts": [...]},
        ], llm_fn=my_llm)

        passing = [r for r in results if r.passed()]
    """
    results = []
    for item in items:
        result = evaluate(
            question=item["question"],
            answer=item["answer"],
            contexts=item["contexts"],
            llm_fn=llm_fn,
            **evaluate_kwargs,
        )
        results.append(result)
    return results
