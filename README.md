# ragcheck

Lightweight, reference-free RAG evaluation. Drop it into any project with no ground truth required and no framework lock-in.

```
pip install ragcheck
```

---

## What it does

RAG systems fail in three main ways: hallucination, off-topic answers, and poor retrieval. ragcheck scores all three with a single function call, using whatever LLM you already have.

---

## Quickstart

```python
import ragcheck

result = ragcheck.evaluate(
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
result = ragcheck.evaluate(
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

Extend ragcheck with your own evaluations without modifying the library:

```python
def conciseness_prompt(question: str, answer: str, contexts: list) -> str:
    return (
        f"Rate how concise this answer is. 1.0 = very concise, 0.0 = very verbose.\n\n"
        f"Answer: {answer}\n\n"
        f'Respond ONLY with valid JSON: {{"score": <float 0-1>, "reasoning": "<one sentence>"}}'
    )

result = ragcheck.evaluate(
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
results = ragcheck.evaluate_batch(
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

Use ragcheck as quality infrastructure, not just a one-off checker.

```python
import ragcheck

# 1. Wrap your LLM with a cache — repeated CI runs are free
cached_llm = ragcheck.make_cached_llm(llm_fn, cache=":memory:")         # in-process
# or: ragcheck.make_cached_llm(llm_fn, cache="ragcheck.cache.db")       # persists across runs

# 2. Evaluate and save a quality baseline
result = ragcheck.evaluate(question, answer, contexts, cached_llm, model="gpt-4o-mini")
result.save_baseline("baseline.json")

# 3. After a change (new prompt, model swap, retrieval tweak) — compare
new_result = ragcheck.evaluate(question, new_answer, contexts, cached_llm)
diff = ragcheck.compare([result], [new_result])
print(diff)
# Top regressions:
#   Q1: faithfulness 0.90 → 0.45  (Δ-0.45)
# Net delta: faithfulness -0.45  answer_relevance +0.02  context_precision -0.01

# 4. Block CI on quality drops — same mental model as jest snapshots
ragcheck.assert_no_regression("baseline.json", new_result, tolerance=0.05)
# raises AssertionError with metric name if any score drops > 0.05
```

---

## Understand failures

ragcheck tells you what went wrong, not just what the score was.

```python
result = ragcheck.evaluate(question, answer, contexts, llm_fn)

# Human-readable failure labels — no extra LLM calls
print(result.failure_modes)   # ["hallucination", "retrieval_miss"]

# Know what your eval suite costs per CI run
result = ragcheck.evaluate(..., model="gpt-4o-mini")
print(result.tokens_used)          # int
print(result.estimated_cost_usd)   # float

# Paste into a GitHub PR description
print(result.to_markdown())

# Track quality trends over time
history = ragcheck.History("ragcheck.db")
history.log(results, label="after-prompt-v3")
history.trend("faithfulness", days=30)   # [(timestamp, avg_score), ...]
history.regressions(since="2026-04-01") # runs where scores dropped
history.summary()                        # {metric: latest_avg}

# Async evaluation for large test suites
results = await ragcheck.aevaluate_batch(items, llm_fn, concurrency=5)
```

---

## Compared to RAGAS

| | ragcheck | RAGAS |
|---|---|---|
| Ground truth required | No | Some metrics yes |
| Mandatory dependencies | None | Several |
| LLM flexibility | Any callable | OpenAI-first |
| Custom metrics | Pass a prompt function | Subclass-based |
| Batch evaluation | `evaluate_batch()` | Built-in dataset support |
| Install | `pip install ragcheck` | Framework adoption |

---

## License

MIT
