# ragcheck

Lightweight, reference-free RAG evaluation. Drop it into any project — no ground truth, no framework lock-in.

```
pip install ragcheck
```

---

## What & Why

RAG systems fail in three ways: hallucination, off-topic answers, and bad retrieval. ragcheck scores all three with a single function call, using whatever LLM you already have. No reference answers needed, no new framework to adopt.

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

print(result.faithfulness)       # float 0–1
print(result.answer_relevance)   # float 0–1
print(result.context_precision)  # float 0–1
print(result.reasoning)          # {"faithfulness": "...", "answer_relevance": "...", "context_precision": "..."}
print(result.parse_errors)       # [] if all metrics parsed cleanly

# readable summary
print(result)

# boolean gate — useful in CI
if not result.passed(threshold=0.7):
    raise ValueError("RAG quality below threshold")

# export to JSON for logging or dashboards
print(result.to_json(indent=2))
```

---

## Metrics

| Metric | What it measures | Low score means |
|---|---|---|
| `faithfulness` | Is every claim in the answer grounded in the context? | Hallucination |
| `answer_relevance` | Does the answer actually address the question? | Off-topic response |
| `context_precision` | Was the retrieved context useful? | Bad retrieval |
| `context_recall` *(optional)* | Does the context contain enough to produce the answer? | Incomplete retrieval |

All scores are `float` from 0.0 to 1.0. `EvalResult.reasoning` gives a one-sentence explanation per metric.

`EvalResult.parse_errors` is a list of error strings for any metric where the LLM response couldn't be parsed. A score of `0.0` with a parse error means the response was unreadable — not that the answer was actually bad.

`EvalResult.passed(threshold=0.7)` returns `True` if **all** scored metrics meet the threshold — a one-liner CI gate.

---

## Optional: context_recall

```python
result = ragcheck.evaluate(
    question="...", answer="...", contexts=[...], llm_fn=llm_fn,
    include_context_recall=True,  # adds one extra LLM call
)
print(result.context_recall)  # float 0–1
```

---

## Custom Metrics

Extend ragcheck with your own domain-specific evaluations without forking the library:

```python
def conciseness_prompt(question: str, answer: str, contexts: list) -> str:
    return (
        f"Rate how concise this answer is (1.0 = very concise, 0.0 = very verbose).\n\n"
        f"Answer: {answer}\n\n"
        f'Respond ONLY with valid JSON: {{"score": <float 0-1>, "reasoning": "<one sentence>"}}'
    )

result = ragcheck.evaluate(
    question="...", answer="...", contexts=[...], llm_fn=llm_fn,
    extra_metrics={"conciseness": conciseness_prompt},
)
print(result.extra_metrics["conciseness"])  # float 0–1
print(result.reasoning["conciseness"])      # one-sentence explanation
```

Custom metrics are included in `passed()`, `to_dict()`, and `to_json()` automatically.

---

## Batch Evaluation

```python
results = ragcheck.evaluate_batch(
    items=[
        {"question": "...", "answer": "...", "contexts": [...]},
        {"question": "...", "answer": "...", "contexts": [...]},
    ],
    llm_fn=llm_fn,
)

# filter for passing responses
passing = [r for r in results if r.passed(threshold=0.7)]
```

---

## Export

```python
# dict — pipe into logging, databases, dashboards
d = result.to_dict()

# JSON string — write to file, send over HTTP
json_str = result.to_json(indent=2)
```

---

## LLM Examples

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

## vs RAGAS

| | ragcheck | RAGAS |
|---|---|---|
| Ground truth required | No | Some metrics yes |
| Mandatory dependencies | None | Several |
| LLM flexibility | Any callable | OpenAI-first |
| Install | `pip install ragcheck` | Framework adoption |
| Custom metrics | Yes — pass a prompt function | Subclass-based |
| Batch evaluation | `evaluate_batch()` | Built-in dataset support |
| API surface | One function | Full framework |

RAGAS is a gym membership. ragcheck is a pushup.

---

## License

MIT
