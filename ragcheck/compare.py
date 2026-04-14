from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from ragcheck.core import EvalResult

_CORE_METRICS = ["faithfulness", "answer_relevance", "context_precision"]


@dataclass
class CompareResult:
    """Result of comparing two lists of EvalResult (before vs after)."""

    net_delta: dict[str, float]
    regressions: list[dict]   # [{index, metric, before, after, delta}, ...]
    improvements: list[dict]  # [{index, metric, before, after, delta}, ...]

    def __str__(self) -> str:
        lines = []
        if self.regressions:
            lines.append("Top regressions:")
            for r in sorted(self.regressions, key=lambda x: x["delta"])[:5]:
                lines.append(
                    f"  Q{r['index']}: {r['metric']} {r['before']:.2f} → {r['after']:.2f}"
                    f"  (Δ{r['delta']:+.2f})"
                )
        if self.improvements:
            lines.append("Top improvements:")
            for r in sorted(self.improvements, key=lambda x: -x["delta"])[:5]:
                lines.append(
                    f"  Q{r['index']}: {r['metric']} {r['before']:.2f} → {r['after']:.2f}"
                    f"  (Δ{r['delta']:+.2f})"
                )
        lines.append("Net delta: " + "  ".join(
            f"{m} {d:+.2f}" for m, d in self.net_delta.items()
        ))
        return "\n".join(lines)

    def to_markdown(self) -> str:
        sections = []
        if self.regressions:
            rows = ["| # | Metric | Before | After | Δ |", "|---|---|---|---|---|"]
            for r in sorted(self.regressions, key=lambda x: x["delta"])[:10]:
                rows.append(
                    f"| Q{r['index']} | {r['metric']} | {r['before']:.2f}"
                    f" | {r['after']:.2f} | {r['delta']:+.2f} |"
                )
            sections.append("**Regressions**\n\n" + "\n".join(rows))
        if self.improvements:
            rows = ["| # | Metric | Before | After | Δ |", "|---|---|---|---|---|"]
            for r in sorted(self.improvements, key=lambda x: -x["delta"])[:10]:
                rows.append(
                    f"| Q{r['index']} | {r['metric']} | {r['before']:.2f}"
                    f" | {r['after']:.2f} | {r['delta']:+.2f} |"
                )
            sections.append("**Improvements**\n\n" + "\n".join(rows))
        net = "  ".join(f"{m}: {d:+.2f}" for m, d in self.net_delta.items())
        sections.append(f"**Net delta:** {net}")
        return "\n\n".join(sections)

    def to_dict(self) -> dict:
        return {
            "net_delta": self.net_delta,
            "regressions": self.regressions,
            "improvements": self.improvements,
        }


def compare(before: List[EvalResult], after: List[EvalResult]) -> CompareResult:
    """Diff two lists of EvalResult and return a CompareResult.

    Args:
        before: EvalResults from before a change (e.g. old prompt).
        after: EvalResults from after a change (e.g. new prompt).

    Returns:
        CompareResult with per-question regressions, improvements, and net delta.

    Raises:
        ValueError: If the two lists have different lengths.
    """
    if len(before) != len(after):
        raise ValueError(
            f"before and after must have the same length, "
            f"got {len(before)} and {len(after)}"
        )
    if not before:
        return CompareResult(net_delta={}, regressions=[], improvements=[])

    metrics = list(_CORE_METRICS)
    # Include context_recall if present in either list
    if any(r.context_recall is not None for r in before + after):
        metrics.append("context_recall")

    regressions = []
    improvements = []
    metric_deltas: dict[str, list[float]] = {m: [] for m in metrics}

    for i, (b, a) in enumerate(zip(before, after)):
        b_dict = b.to_dict()
        a_dict = a.to_dict()
        for metric in metrics:
            bv = b_dict.get(metric)
            av = a_dict.get(metric)
            if bv is None or av is None:
                continue
            delta = av - bv
            metric_deltas[metric].append(delta)
            entry = {"index": i + 1, "metric": metric, "before": bv, "after": av, "delta": delta}
            if delta < 0:
                regressions.append(entry)
            elif delta > 0:
                improvements.append(entry)

    net_delta = {
        m: round(sum(deltas) / len(deltas), 4) if deltas else 0.0
        for m, deltas in metric_deltas.items()
        if deltas
    }

    return CompareResult(
        net_delta=net_delta,
        regressions=regressions,
        improvements=improvements,
    )
