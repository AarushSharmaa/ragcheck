# Changelog

All notable changes to evalops are documented here.

This project follows [Semantic Versioning](https://semver.org/) and
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions.

---

## [Unreleased] — on `dev`, not yet on PyPI

Work in progress toward the next release. See the `dev` branch.

---

## [1.0.0] — 2026-04-15

The first production-ready release. evalops graduates from a one-off eval function
into infrastructure for eval-driven development: regression gates, cost tracking,
comparison diffs, caching, async evaluation, and a persistent eval history — all
with zero mandatory dependencies.

### Added

#### Failure mode classification
- `EvalResult.failure_modes` — list of human-readable labels derived from score
  thresholds, no extra LLM calls
- Labels: `hallucination` (low faithfulness), `off_topic` (low answer_relevance),
  `retrieval_miss` (low context_precision), `context_noise` (low context_recall)
- Included in `to_dict()`, `to_json()`, and `to_markdown()`; omitted when empty
- Motivation: "faithfulness: 0.38" is a number — "hallucination" is a decision

#### Token and cost tracking
- `EvalResult.tokens_used` — estimated token count across all LLM calls in a run
- `EvalResult.estimated_cost_usd` — cost estimate using a built-in pricing table
- `PRICING` dict exported from `evalops` — 11 models including GPT-4o, Claude Sonnet,
  Gemini Flash, Llama 3 via Ollama/Groq (free tier)
- `pricing={"input": ..., "output": ...}` keyword on `evaluate()` for custom or
  private model pricing
- Token estimation: 4 chars ≈ 1 token, documented as an estimate — exact counts
  require provider tokenisers which would break the zero-dependency guarantee
- Motivation: "how much does our eval suite cost per CI run?" should be a field read,
  not a back-of-envelope calculation

#### Before/after comparison
- New module `evalops/compare.py`
- `compare(before, after) -> CompareResult` — diffs two lists of `EvalResult`
- `CompareResult` fields: `net_delta`, `regressions`, `improvements`
- `CompareResult.__str__()` — readable diff with top regressions, top improvements,
  and net delta per metric
- `CompareResult.to_markdown()` — table suitable for pasting into a GitHub PR
- `CompareResult.to_dict()` — JSON-serialisable for logging or dashboards
- Motivation: a prompt change that drops faithfulness 0.12 and lifts relevance 0.08
  should produce a before/after story by default, not require manual comparison

#### Markdown report on EvalResult
- `EvalResult.to_markdown()` — formatted table with scores, reasoning, failure modes,
  and a cost line
- Conditionally includes `context_recall`, `extra_metrics`, and cost only when present
- Motivation: paste into GitHub PR descriptions to make quality visible during review

#### Baseline snapshots and regression gates
- `EvalResult.save_baseline(path)` — write a result to a JSON file as a quality snapshot
- `evalops.assert_no_regression(baseline, result, *, tolerance=0.05)` — raises
  `AssertionError` naming the metric if any score drops beyond the tolerance
- `baseline` accepts a file path string or a pre-loaded dict
- Designed to drop into a pytest file and block CI on quality regressions
- Motivation: same mental model as jest/vitest snapshots — gate deploys on eval
  quality, not just code correctness

#### LLM prompt caching
- New module `evalops/cache.py`
- `make_cached_llm(llm_fn, cache) -> callable` — wraps any `llm_fn` with
  prompt-level caching
- `cache=":memory:"` — in-process dict cache, isolated per `make_cached_llm` call
- `cache="path/to/file.db"` — SQLite-backed, persists across processes and CI runs
- Cache keys use SHA-256 hashes of the prompt string — deterministic across processes
  (Python's built-in `hash()` is randomised per-process via `PYTHONHASHSEED` and
  cannot be used for cross-run caching)
- Motivation: 20 eval items × 3 metrics = 60 LLM calls per CI run; unchanged inputs
  are free on the second run

#### SQLite eval history
- New module `evalops/history.py`
- `History(db_path)` — creates the database on first use, no setup required
- `History.log(results, label=None)` — persist a batch of results with an optional
  run label (e.g. `"after-prompt-v3"`)
- `History.trend(metric, days=None)` — `[(timestamp, avg_score), ...]` for charting
  quality over time
- `History.regressions(since=None)` — runs where any metric average dropped vs the
  preceding run, optionally filtered by date
- `History.summary()` — `{metric: latest_avg}` snapshot of the most recent run
- Motivation: turns evalops from a checker into a monitor — catch slow drift, not
  just sudden drops

#### Async evaluation
- New module `evalops/_async.py`
- `aevaluate(question, answer, contexts, llm_fn, **kwargs) -> EvalResult`
- `aevaluate_batch(items, llm_fn, *, concurrency=5, **kwargs) -> List[EvalResult]`
- Implemented via `asyncio.to_thread` — works with any synchronous `llm_fn`; users
  do not need to provide a native async function
- `concurrency` controls parallelism via `asyncio.Semaphore` — avoids overwhelming
  rate-limited APIs
- Motivation: 50 items × 3 LLM calls = 150 serial calls synchronously; with
  `concurrency=5` that becomes ~30 batches, practical for large overnight test suites

#### Rubric-based prompts with chain-of-thought
- All four metric prompts rewritten with 5-level scoring rubrics, CoT instructions,
  and two calibration examples per metric (one high-score, one low-score)
- Research backing: Prometheus (rubric anchors reduce score variance 15-25%), G-Eval
  (CoT before scoring improves human correlation ~0.05 Spearman), few-shot calibration
  (2-3 examples reduce run-to-run variance 10-20%)
- No API change — `_parse_llm_json` already handles CoT prose before JSON via regex
  fallback; weaker models that skip CoT still parse correctly

#### Claim decomposition for faithfulness
- `decompose_claims=True` keyword on `evaluate()` — two-step faithfulness path
- Step 1: decompose the answer into atomic claims (one extra LLM call)
- Step 2: verify each claim against the retrieved context (one extra LLM call)
- `result.faithfulness` becomes the fraction of supported claims (still 0.0–1.0)
- `result.reasoning["faithfulness"]` contains an audit trail: "3/4 claims supported.
  Unsupported: 'temperatures reach 500°C'"
- Falls back silently to standard single-prompt faithfulness if decomposition returns
  no parseable claims — no errors raised on bad LLM output
- Research backing: FActScore (claim decomposition achieves ~0.89 human correlation
  vs ~0.68 for holistic scoring); RAGAS uses the same two-step approach

#### Confidence intervals
- New function `evalops.evaluate_with_confidence(question, answer, contexts, llm_fn,
  *, n=3, **kwargs) -> EvalResult`
- Runs `evaluate()` n times, returns mean scores with per-metric confidence stats
- `EvalResult.confidence` — new optional field, `None` by default (standard evaluate)
- Per-metric dict: `mean`, `std`, `ci_lower`, `ci_upper` (95% CI), `scores` list
- 95% CI formula: `mean ± 1.96 * (std / sqrt(n))`, clamped to [0, 1]
- `to_dict()` includes `confidence` when present; `to_markdown()` renders a Score
  Stability table with mean ± std and CI bounds
- `tokens_used` and `estimated_cost_usd` are summed across all n runs
- Research backing: ARES (calibrated confidence intervals dramatically improve trust
  in automated evals)

### Changed
- `EvalResult` gains three new optional fields: `failure_modes`, `tokens_used`,
  `estimated_cost_usd` — all default to empty/zero, no existing code breaks
- `EvalResult` gains one additional optional field: `confidence` — `None` by default
- `evaluate()` gains two new keyword-only params: `model=None`, `pricing=None`
- `evaluate()` gains one new keyword-only param: `decompose_claims=False`
- `to_dict()` conditionally includes `failure_modes`, `tokens_used`,
  `estimated_cost_usd`, `confidence` only when non-empty/non-zero/non-None
- `to_markdown()` conditionally renders a Score Stability table when `confidence` set
- `evalops.__init__` now exports: `compare`, `CompareResult`, `make_cached_llm`,
  `History`, `aevaluate`, `aevaluate_batch`, `assert_no_regression`, `PRICING`,
  `evaluate_with_confidence`

### Fixed
- SQLite on Windows holds file handles open even after exiting a
  `with sqlite3.connect()` block — the context manager commits/rolls back
  transactions but does **not** close the connection. All SQLite usage in
  `cache.py` and `history.py` now uses explicit `try/finally conn.close()` to
  prevent `WinError 32` file lock errors on Windows

---

## [0.2.0] — 2026-04-08 — first public release on PyPI

The first version published to PyPI. Established the core evaluation loop and the
`llm_fn` contract that all future versions maintain.

### Added

#### Core evaluation
- `evaluate(question, answer, contexts, llm_fn) -> EvalResult` — evaluate a RAG
  response against three reference-free metrics with a single function call
- `llm_fn: callable[[str], str]` — bring any LLM; evalops never imports a provider
- Three metrics scored 0.0–1.0:
  - `faithfulness` — are all claims in the answer grounded in the context?
  - `answer_relevance` — does the answer address the question?
  - `context_precision` — was the retrieved context useful for answering the question?
- `EvalResult.reasoning` — one-sentence explanation per metric from the LLM judge
- `EvalResult.parse_errors` — visible list of parse failures; a score of 0.0 with a
  parse error means the LLM response was unreadable, not that the answer was bad

#### Resilient JSON parsing
- `_parse_llm_json` tries `json.loads` first, then falls back to a regex that
  extracts `{"score": ..., "reasoning": "..."}` from surrounding prose
- Handles LLMs that prefix JSON with explanation text
- Handles `"score": "0.8"` (string coercion to float)
- Handles missing `score` or `reasoning` keys with safe defaults
- All scores clamped to `[0.0, 1.0]`

#### Optional context recall metric
- `include_context_recall=True` on `evaluate()` — adds a fourth LLM call
- `EvalResult.context_recall` — `None` by default, float 0–1 when requested
- Measures whether the retrieved context contains enough information to answer;
  distinct from context_precision (which measures whether what was fetched was useful)

#### Custom metrics
- `extra_metrics={"name": prompt_fn}` on `evaluate()` — add domain-specific
  evaluations without modifying the library
- `prompt_fn(question, answer, contexts) -> str` — same contract as built-in metrics
- `EvalResult.extra_metrics` — `{"name": score}` dict
- Custom metrics participate in `passed()`, `to_dict()`, and `to_json()`

#### Batch evaluation
- `evaluate_batch(items, llm_fn, **kwargs) -> List[EvalResult]`
- `items` is a list of dicts with `question`, `answer`, `contexts` keys
- All `evaluate()` kwargs forwarded to each item; results preserve input order

#### Result utilities
- `EvalResult.passed(threshold=0.7)` — boolean gate for CI pipelines
- `EvalResult.__str__()` — human-readable summary for logging
- `EvalResult.to_dict()` — full result as a Python dict, JSON-serialisable
- `EvalResult.to_json(**kwargs)` — JSON string, kwargs forwarded to `json.dumps`
- Input validation with clear `TypeError` / `ValueError` messages

---

*evalops is MIT licensed. Source: [github.com/AarushSharmaa/evalops](https://github.com/AarushSharmaa/evalops)*
