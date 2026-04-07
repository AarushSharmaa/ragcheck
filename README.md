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
```

---

## Metrics

| Metric | What it measures | Low score means |
|---|---|---|
| `faithfulness` | Is every claim in the answer grounded in the context? | Hallucination |
| `answer_relevance` | Does the answer actually address the question? | Off-topic response |
| `context_precision` | Was the retrieved context useful? | Bad retrieval |

All scores are `float` from 0.0 to 1.0. `EvalResult.reasoning` gives a one-sentence explanation per metric.

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
| API surface | One function | Full framework |

RAGAS is a gym membership. ragcheck is a pushup.

---

## License

MIT
