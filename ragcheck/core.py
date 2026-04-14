from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional


# Pricing table: (input_price_per_token, output_price_per_token)
# Estimates only — token counts derived from character count (4 chars ≈ 1 token).
# Pin your judge model version the same way you pin library versions.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o":                  (0.000005,   0.000015),
    "gpt-4o-mini":             (0.00000015, 0.0000006),
    "gpt-4-turbo":             (0.00001,    0.00003),
    "claude-opus-4-6":         (0.000015,   0.000075),
    "claude-sonnet-4-6":       (0.000003,   0.000015),
    "claude-haiku-4-5-20251001": (0.00000025, 0.00000125),
    "gemini-1.5-flash":        (0.000000075, 0.0000003),
    "gemini-1.5-pro":          (0.00000125, 0.000005),
    "gemini-2.0-flash":        (0.0000001,  0.0000004),
    "llama3":                  (0.0,        0.0),
    "ollama":                  (0.0,        0.0),
    "groq":                    (0.0,        0.0),
}

_FAILURE_THRESHOLDS = {
    "hallucination":  ("faithfulness",      0.5),
    "off_topic":      ("answer_relevance",  0.5),
    "retrieval_miss": ("context_precision", 0.5),
    "context_noise":  ("context_recall",    0.5),
}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class EvalResult:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    reasoning: dict[str, str]
    parse_errors: list[str] = field(default_factory=list)
    context_recall: Optional[float] = None
    extra_metrics: dict[str, float] = field(default_factory=dict)
    failure_modes: list[str] = field(default_factory=list)
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0

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
        if self.failure_modes:
            d["failure_modes"] = self.failure_modes
        if self.tokens_used:
            d["tokens_used"] = self.tokens_used
        if self.estimated_cost_usd:
            d["estimated_cost_usd"] = self.estimated_cost_usd
        return d

    def to_markdown(self) -> str:
        """Return a markdown report of this eval result."""
        rows = [
            "| Metric | Score | Reasoning |",
            "|---|---|---|",
            f"| faithfulness | {self.faithfulness:.2f} | {self.reasoning.get('faithfulness', '')} |",
            f"| answer_relevance | {self.answer_relevance:.2f} | {self.reasoning.get('answer_relevance', '')} |",
            f"| context_precision | {self.context_precision:.2f} | {self.reasoning.get('context_precision', '')} |",
        ]
        if self.context_recall is not None:
            rows.append(f"| context_recall | {self.context_recall:.2f} | {self.reasoning.get('context_recall', '')} |")
        for name, score in self.extra_metrics.items():
            rows.append(f"| {name} | {score:.2f} | {self.reasoning.get(name, '')} |")
        lines = ["\n".join(rows)]
        if self.failure_modes:
            lines.append(f"\n**Failure modes:** {', '.join(self.failure_modes)}")
        if self.estimated_cost_usd:
            lines.append(f"\n**Cost:** ${self.estimated_cost_usd:.6f} ({self.tokens_used} tokens)")
        return "\n".join(lines)

    def save_baseline(self, path: str) -> None:
        """Save this result as a JSON baseline file."""
        Path(path).write_text(self.to_json(indent=2), encoding="utf-8")

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
        if self.failure_modes:
            lines.append(f"  failure_modes     {self.failure_modes}")
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


def _compute_failure_modes(
    faithfulness: float,
    answer_relevance: float,
    context_precision: float,
    context_recall: Optional[float],
) -> list[str]:
    scores = {
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    modes = []
    for label, (metric, threshold) in _FAILURE_THRESHOLDS.items():
        value = scores.get(metric)
        if value is not None and value < threshold:
            modes.append(label)
    return modes


def _compute_cost(
    prompts: list[str],
    responses: list[str],
    model: Optional[str],
    pricing: Optional[dict[str, float]],
) -> tuple[int, float]:
    tokens = sum(_estimate_tokens(t) for t in prompts + responses)
    if pricing:
        input_price = pricing.get("input", 0.0)
        output_price = pricing.get("output", 0.0)
    elif model and model in PRICING:
        input_price, output_price = PRICING[model]
    else:
        return tokens, 0.0
    # Rough split: prompts are input tokens, responses are output tokens
    input_tokens = sum(_estimate_tokens(p) for p in prompts)
    output_tokens = sum(_estimate_tokens(r) for r in responses)
    cost = input_tokens * input_price + output_tokens * output_price
    return tokens, cost


def evaluate(
    question: str,
    answer: str,
    contexts: List[str],
    llm_fn: Callable[[str], str],
    *,
    include_context_recall: bool = False,
    extra_metrics: Optional[dict[str, Callable[[str, str, List[str]], str]]] = None,
    model: Optional[str] = None,
    pricing: Optional[dict[str, float]] = None,
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
        model: Model name for cost estimation (e.g. "gpt-4o-mini"). See PRICING.
        pricing: Custom pricing dict with "input" and "output" keys (price per token).
            Overrides built-in PRICING table.

    Returns:
        EvalResult with faithfulness, answer_relevance, context_precision scores (0-1),
        optional context_recall, optional extra_metrics scores, per-metric reasoning
        strings, parse_errors list, failure_modes list, tokens_used, estimated_cost_usd.
    """
    if not callable(llm_fn):
        raise TypeError(f"llm_fn must be callable, got {type(llm_fn).__name__!r}")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    if not isinstance(contexts, list):
        raise TypeError(f"contexts must be a list, got {type(contexts).__name__!r}")

    faith_prompt = _faithfulness_prompt(question, answer, contexts)
    relevance_prompt = _answer_relevance_prompt(question, answer)
    precision_prompt = _context_precision_prompt(question, contexts)

    faith_response = llm_fn(faith_prompt)
    relevance_response = llm_fn(relevance_prompt)
    precision_response = llm_fn(precision_prompt)

    faith_score, faith_reason, faith_err = _parse_llm_json(faith_response)
    relevance_score, relevance_reason, relevance_err = _parse_llm_json(relevance_response)
    precision_score, precision_reason, precision_err = _parse_llm_json(precision_response)

    parse_errors = [e for e in [faith_err, relevance_err, precision_err] if e is not None]
    reasoning = {
        "faithfulness": faith_reason,
        "answer_relevance": relevance_reason,
        "context_precision": precision_reason,
    }

    all_prompts = [faith_prompt, relevance_prompt, precision_prompt]
    all_responses = [faith_response, relevance_response, precision_response]

    recall_score = None
    if include_context_recall:
        recall_prompt = _context_recall_prompt(question, answer, contexts)
        recall_response = llm_fn(recall_prompt)
        recall_score, recall_reason, recall_err = _parse_llm_json(recall_response)
        reasoning["context_recall"] = recall_reason
        if recall_err:
            parse_errors.append(recall_err)
        all_prompts.append(recall_prompt)
        all_responses.append(recall_response)

    extra_scores: dict[str, float] = {}
    if extra_metrics:
        for name, prompt_fn in extra_metrics.items():
            ep = prompt_fn(question, answer, contexts)
            er = llm_fn(ep)
            score, reason, err = _parse_llm_json(er)
            extra_scores[name] = score
            reasoning[name] = reason
            if err:
                parse_errors.append(err)
            all_prompts.append(ep)
            all_responses.append(er)

    failure_modes = _compute_failure_modes(faith_score, relevance_score, precision_score, recall_score)
    tokens_used, estimated_cost_usd = _compute_cost(all_prompts, all_responses, model, pricing)

    return EvalResult(
        faithfulness=faith_score,
        answer_relevance=relevance_score,
        context_precision=precision_score,
        reasoning=reasoning,
        parse_errors=parse_errors,
        context_recall=recall_score,
        extra_metrics=extra_scores,
        failure_modes=failure_modes,
        tokens_used=tokens_used,
        estimated_cost_usd=estimated_cost_usd,
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


def assert_no_regression(
    baseline: str | dict,
    result: EvalResult,
    *,
    tolerance: float = 0.05,
) -> None:
    """Assert that result has not regressed below a saved baseline.

    Raises AssertionError with a clear message naming the regressed metric
    if any score drops more than `tolerance` below the baseline value.

    Args:
        baseline: Path to a JSON baseline file (str) or a dict from to_dict().
        result: The new EvalResult to check.
        tolerance: Maximum allowed score drop per metric. Default 0.05.

    Example (inside a pytest file):
        ragcheck.assert_no_regression("baseline.json", new_result, tolerance=0.05)
    """
    if isinstance(baseline, str):
        baseline_dict = json.loads(Path(baseline).read_text(encoding="utf-8"))
    else:
        baseline_dict = baseline

    metrics_to_check = ["faithfulness", "answer_relevance", "context_precision"]
    if "context_recall" in baseline_dict:
        metrics_to_check.append("context_recall")

    result_dict = result.to_dict()
    for metric in metrics_to_check:
        baseline_score = baseline_dict.get(metric)
        new_score = result_dict.get(metric)
        if baseline_score is None or new_score is None:
            continue
        drop = baseline_score - new_score
        if drop > tolerance:
            raise AssertionError(
                f"Regression detected: {metric} dropped from {baseline_score:.3f} "
                f"to {new_score:.3f} (drop={drop:.3f}, tolerance={tolerance:.3f})"
            )
