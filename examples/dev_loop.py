"""
ragcheck dev loop example — compare + cache + baseline

Shows how to use ragcheck as infrastructure for eval-driven development:
  1. Evaluate a RAG response and save it as a quality baseline
  2. Re-evaluate after a change (e.g. new prompt or model swap)
  3. Compare before/after to see exactly what regressed or improved
  4. Use caching so repeated CI runs don't cost extra

Requirements:
    pip install google-generativeai python-dotenv

Set GEMINI_API_KEY in a .env file or your environment.
Run: python examples/dev_loop.py
"""

import os
import tempfile

from dotenv import load_dotenv
import google.generativeai as genai

import ragcheck

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")


def gemini_llm_fn(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text


# ------------------------------------------------------------------
# Sample RAG scenario: two versions of an answer to the same question
# ------------------------------------------------------------------

question = "What causes the northern lights?"
contexts = [
    "Aurora borealis occurs when charged particles ejected by the sun travel to Earth "
    "and interact with gases in the upper atmosphere, producing colorful light displays.",
    "Earth's magnetic field channels solar wind particles toward the poles, where they "
    "collide with oxygen and nitrogen atoms, releasing energy as visible light.",
]

# v1: a well-grounded answer
answer_v1 = (
    "The northern lights are caused by charged particles from the sun colliding with "
    "gases in Earth's atmosphere near the magnetic poles."
)

# v2: same answer but with an unsupported claim added (introduces hallucination risk)
answer_v2 = (
    "The northern lights are caused by solar wind. They appear mostly in winter "
    "because Earth is closer to the sun and the nights are longer. Scientists "
    "believe they also affect animal migration."
)

# ------------------------------------------------------------------
# Step 1: Wrap llm_fn with a cache so repeated CI runs don't cost money
# ------------------------------------------------------------------
cached_llm = ragcheck.make_cached_llm(gemini_llm_fn, cache=":memory:")

# ------------------------------------------------------------------
# Step 2: Evaluate v1 and save as baseline
# ------------------------------------------------------------------
print("Evaluating v1 answer...")
result_v1 = ragcheck.evaluate(question, answer_v1, contexts, cached_llm, model="gemini-1.5-flash")

print("\n--- v1 result ---")
print(result_v1.to_markdown())

# Save baseline to a temp file (in real use: save to repo as "baseline.json")
with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
    baseline_path = f.name
result_v1.save_baseline(baseline_path)
print(f"\nBaseline saved to: {baseline_path}")

# ------------------------------------------------------------------
# Step 3: Evaluate v2 and compare
# ------------------------------------------------------------------
print("\nEvaluating v2 answer...")
result_v2 = ragcheck.evaluate(question, answer_v2, contexts, cached_llm, model="gemini-1.5-flash")

print("\n--- v2 result ---")
print(result_v2.to_markdown())

# ------------------------------------------------------------------
# Step 4: Compare before/after
# ------------------------------------------------------------------
diff = ragcheck.compare([result_v1], [result_v2])
print("\n--- Before/After comparison ---")
print(diff)

# ------------------------------------------------------------------
# Step 5: Assert no regression (this would block CI if quality dropped)
# ------------------------------------------------------------------
print("\nRunning regression check (tolerance=0.10)...")
try:
    ragcheck.assert_no_regression(baseline_path, result_v2, tolerance=0.10)
    print("Regression check passed.")
except AssertionError as e:
    print(f"Regression detected: {e}")

# ------------------------------------------------------------------
# Step 6: Log to history
# ------------------------------------------------------------------
history = ragcheck.History("ragcheck_history.db")
history.log([result_v1], label="answer-v1")
history.log([result_v2], label="answer-v2")

print("\n--- Faithfulness trend ---")
for ts, score in history.trend("faithfulness"):
    print(f"  {ts[:19]}  faithfulness={score:.2f}")

print("\n--- Summary (latest run) ---")
for metric, score in history.summary().items():
    print(f"  {metric}: {score:.2f}")

# cleanup
os.unlink(baseline_path)
if os.path.exists("ragcheck_history.db"):
    os.unlink("ragcheck_history.db")
