# evalops

LLM output evaluation for teams that ship. Failure modes, regression gates, cost tracking, and quality history. No ground truth required, no framework lock-in.

> *Because "it seemed fine in testing" is not an ops strategy.*

```
pip install evalsystem
```

---

## What it does

RAG pipelines fail in three specific ways: the answer invents facts not in the context, the answer ignores the actual question, or the retrieval pulled the wrong documents to begin with. Most teams discover which one only after a user complains.

evalops scores all three with a single function call, using whatever LLM you already have. You get a label for what went wrong, not just a number.

---

## Quickstart

```python
import evalops

result = evalops.evaluate(
    question="What causes the northern lights?",
    answer="Charged particles from the sun collide with gases in Earth's atmosphere.",
    contexts=["Aurora borealis occurs when solar particles interact with the upper atmosphere."],
    llm_fn=lambda prompt: your_llm.generate(prompt),
)

print(result.faithfulness)       # float 0-1
print(result.answer_relevance)   # float 0-1
print(result.context_precision)  # float 0-1
print(result.reasoning)          # one-sentence explanation per metric
print(result.parse_errors)       # empty list if all metrics parsed cleanly

# readable summary
print(result)

# boolean gate for CI pipelines
if not result.passed(threshold=0.7):
    raise ValueError("RAG quality below threshold")

# export for logging or dashboards
print(result.to_json(indent=2))
```

---

## Metrics

| Metric | What it measures | Low score means |
|---|---|---|
| `faithfulness` | Every claim in the answer is grounded in the context | Hallucination |
| `answer_relevance` | The answer addresses the question | Off-topic response |
| `context_precision` | The retrieved context was useful | Bad retrieval |
| `context_recall` (optional) | The context contains enough to produce the answer | Incomplete retrieval |

All scores are floats from 0.0 to 1.0.

`parse_errors` is a list of error strings for any metric where the LLM response could not be parsed. A score of 0.0 with a parse error means the response was unreadable, not that the answer was genuinely bad.

`passed(threshold=0.7)` returns `True` if all scored metrics meet the threshold.

---

## Optional: context_recall

```python
result = evalops.evaluate(
    question="...",
    answer="...",
    contexts=[...],
    llm_fn=llm_fn,
    include_context_recall=True,  # adds one extra LLM call
)
print(result.context_recall)  # float 0-1
```

---

## Custom metrics

Extend evalops with your own evaluations without modifying the library:

```python
def conciseness_prompt(question: str, answer: str, contexts: list) -> str:
    return (
        f"Rate how concise this answer is. 1.0 = very concise, 0.0 = very verbose.\n\n"
        f"Answer: {answer}\n\n"
        f'Respond ONLY with valid JSON: {{"score": <float 0-1>, "reasoning": "<one sentence>"}}'
    )

result = evalops.evaluate(
    question="...",
    answer="...",
    contexts=[...],
    llm_fn=llm_fn,
    extra_metrics={"conciseness": conciseness_prompt},
)

print(result.extra_metrics["conciseness"])  # float 0-1
print(result.reasoning["conciseness"])      # one-sentence explanation
```

Custom metrics are included in `passed()`, `to_dict()`, and `to_json()` automatically.

---

## Batch evaluation

```python
results = evalops.evaluate_batch(
    items=[
        {"question": "...", "answer": "...", "contexts": [...]},
        {"question": "...", "answer": "...", "contexts": [...]},
    ],
    llm_fn=llm_fn,
)

passing = [r for r in results if r.passed(threshold=0.7)]
```

---

## Export

```python
# dict, useful for logging or writing to a database
d = result.to_dict()

# JSON string, useful for files or HTTP
json_str = result.to_json(indent=2)
```

---

## LLM examples

**OpenAI**
```python
from openai import OpenAI
client = OpenAI()

def llm_fn(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

**Anthropic**
```python
import anthropic
client = anthropic.Anthropic()

def llm_fn(prompt: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
```

**Google Gemini**
```python
import google.generativeai as genai
genai.configure(api_key="YOUR_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

def llm_fn(prompt: str) -> str:
    return model.generate_content(prompt).text
```

**Ollama (local)**
```python
import requests

def llm_fn(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False},
    )
    return response.json()["response"]
```

---

## Dev loop: compare + cache + baseline

Use evalops as quality infrastructure, not just a one-off checker.

```python
import evalops

# 1. Wrap your LLM with a cache - repeated CI runs are free
cached_llm = evalops.make_cached_llm(llm_fn, cache=":memory:")         # in-process
# or: evalops.make_cached_llm(llm_fn, cache="evalops.cache.db")       # persists across runs

# 2. Evaluate and save a quality baseline
result = evalops.evaluate(question, answer, contexts, cached_llm, model="gpt-4o-mini")
result.save_baseline("baseline.json")

# 3. After a change (new prompt, model swap, retrieval tweak) - compare
new_result = evalops.evaluate(question, new_answer, contexts, cached_llm)
diff = evalops.compare([result], [new_result])
print(diff)
# Top regressions:
#   Q1: faithfulness 0.90 -> 0.45  (d-0.45)
# Net delta: faithfulness -0.45  answer_relevance +0.02  context_precision -0.01

# 4. Block CI on quality drops - same mental model as jest snapshots
evalops.assert_no_regression("baseline.json", new_result, tolerance=0.05)
# raises AssertionError with metric name if any score drops > 0.05
```

---

## Understand failures

evalops tells you what went wrong, not just what the score was.

```python
result = evalops.evaluate(question, answer, contexts, llm_fn)

# Human-readable failure labels - no extra LLM calls
print(result.failure_modes)   # ["hallucination", "retrieval_miss"]

# Know what your eval suite costs per CI run
result = evalops.evaluate(..., model="gpt-4o-mini")
print(result.tokens_used)          # int
print(result.estimated_cost_usd)   # float

# Paste into a GitHub PR description
print(result.to_markdown())

# Track quality trends over time
history = evalops.History("evalops.db")
history.log(results, label="after-prompt-v3")
history.trend("faithfulness", days=30)   # [(timestamp, avg_score), ...]
history.regressions(since="2026-04-01") # runs where scores dropped
history.summary()                        # {metric: latest_avg}

# Async evaluation for large test suites
results = await evalops.aevaluate_batch(items, llm_fn, concurrency=5)
```

---

## Claim decomposition

For higher-fidelity faithfulness scoring, evalops can break the answer into individual atomic claims and verify each one separately. Scores come with an audit trail showing exactly which claims were not supported.

```python
result = evalops.evaluate(
    question="...",
    answer="...",
    contexts=[...],
    llm_fn=llm_fn,
    decompose_claims=True,  # adds 2 extra LLM calls for faithfulness
)

print(result.faithfulness)                  # fraction of supported claims, e.g. 0.75
print(result.reasoning["faithfulness"])     # "3/4 claims supported. Unsupported: 'temperatures reach 500C'"
```

When decomposition fails (unparseable LLM response), evalops falls back to the standard single-prompt faithfulness scoring automatically.

---

## Confidence intervals

A single eval score is a point estimate. `evaluate_with_confidence` runs evaluation `n` times and reports how stable the scores are.

```python
result = evalops.evaluate_with_confidence(
    question="...",
    answer="...",
    contexts=[...],
    llm_fn=llm_fn,
    n=3,  # number of runs (default 3)
)

print(result.faithfulness)  # mean across runs

ci = result.confidence["faithfulness"]
print(ci["mean"])      # 0.82
print(ci["std"])       # 0.04
print(ci["ci_lower"])  # 0.74
print(ci["ci_upper"])  # 0.90
print(ci["scores"])    # [0.8, 0.85, 0.8]

# Confidence also appears in to_markdown() and to_dict()
print(result.to_markdown())  # includes a Score Stability table
```

All kwargs supported by `evaluate()` work here too (e.g. `include_context_recall`, `model`, `decompose_claims`).

---

## Compared to RAGAS

| | evalops | RAGAS |
|---|---|---|
| Ground truth required | No | Some metrics yes |
| Mandatory dependencies | None | Several |
| LLM flexibility | Any callable | OpenAI-first |
| Custom metrics | Pass a prompt function | Subclass-based |
| Batch evaluation | `evaluate_batch()` | Built-in dataset support |
| Install | `pip install evalops` | Framework adoption |

---

## How scoring works

evalops uses LLM-as-a-judge: a capable language model evaluates outputs along dimensions that are too nuanced to specify as rules. Instead of matching strings or requiring labeled ground truth, the judge asks whether claims are grounded, whether the answer addresses the question, and whether the retrieved context was useful.

This approach is backed by research. [Zheng et al. (2023)](https://arxiv.org/abs/2306.05685) showed LLM judges achieve over 80% agreement with human evaluators on open-ended tasks, comparable to human-human agreement. The [RAGAS paper (Es et al., 2023)](https://arxiv.org/abs/2309.15217) extended this to RAG-specific metrics.

One practical benefit: as you upgrade the judge model, your evaluations get sharper without any changes to your eval suite. Better judge, better signal, same code.

The prompts include structural defenses against common failure modes (truncated responses, parse errors, prompt injection). The [CHANGELOG](CHANGELOG.md) tracks every scoring-relevant change.

---

## A note from the author

This is my first open-source contribution, and it comes from a practical place. The AI use cases I have worked on that went into production all ran into the same problem: no good way to know if quality was holding after a change. evalops is what I wanted to have.

The core idea is simple: teams should be able to start evaluating today - not after a signup, not after adopting a framework, not after configuring a dashboard. `pip install evalsystem`, write a test, ship. If eval has friction, people skip it. And when people skip eval, bad outputs reach users.

There is a growing idea in the LLM community called [Eval Driven Development](https://vadim.blog/eval-driven-development) - write an eval before you change a prompt, the same way you'd write a test before you change a function. evalops is my attempt to make that practical: low friction, no setup, just a function call and a quality gate.

It is a work in progress. If something is unclear, missing, or broken for your use case, please open an issue. Feedback is genuinely welcome.

---

## License

MIT
