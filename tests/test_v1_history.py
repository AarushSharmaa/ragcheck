"""Tests for history.py — History log/trend/regressions/summary."""
from __future__ import annotations

import os
import tempfile

import pytest

from ragcheck.core import EvalResult
from ragcheck.history import History


def make_result(faithfulness=0.8, answer_relevance=0.8, context_precision=0.8,
                context_recall=None):
    reasoning = {"faithfulness": "ok", "answer_relevance": "ok", "context_precision": "ok"}
    if context_recall is not None:
        reasoning["context_recall"] = "ok"
    return EvalResult(
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_precision=context_precision,
        reasoning=reasoning,
        context_recall=context_recall,
    )


def tmp_db():
    """Return a path to a temp db file that doesn't exist yet."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = f.name
    f.close()
    os.unlink(path)
    return path


def test_history_log_persists_to_disk():
    path = tmp_db()
    try:
        h = History(path)
        h.log([make_result()])
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_history_log_accepts_label():
    path = tmp_db()
    try:
        h = History(path)
        h.log([make_result()], label="my-label")
        # Query directly to verify label was stored
        import sqlite3
        conn = sqlite3.connect(path)
        row = conn.execute("SELECT label FROM runs").fetchone()
        conn.close()
        assert row[0] == "my-label"
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_history_trend_returns_list_of_tuples():
    path = tmp_db()
    try:
        h = History(path)
        h.log([make_result(faithfulness=0.8)])
        trend = h.trend("faithfulness")
        assert isinstance(trend, list)
        assert len(trend) == 1
        assert isinstance(trend[0], tuple)
        assert len(trend[0]) == 2
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_history_trend_respects_days_filter():
    path = tmp_db()
    try:
        h = History(path)
        h.log([make_result(faithfulness=0.7)])
        # days=0 should return nothing (timestamp is from "now" but filter is strict <)
        # Use days=1 — the just-logged entry should be included
        trend = h.trend("faithfulness", days=1)
        assert len(trend) == 1
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_history_trend_returns_correct_metric_values():
    path = tmp_db()
    try:
        h = History(path)
        h.log([make_result(faithfulness=0.65)])
        trend = h.trend("faithfulness")
        assert abs(trend[0][1] - 0.65) < 0.001
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_history_regressions_returns_runs_with_score_drops():
    path = tmp_db()
    try:
        h = History(path)
        h.log([make_result(faithfulness=0.9)], label="run1")
        h.log([make_result(faithfulness=0.5)], label="run2")
        regs = h.regressions()
        assert any(r["metric"] == "faithfulness" for r in regs)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_history_regressions_respects_since_filter():
    path = tmp_db()
    try:
        h = History(path)
        h.log([make_result(faithfulness=0.9)])
        h.log([make_result(faithfulness=0.5)])
        # Filter to a future date — no regressions should be found
        regs = h.regressions(since="2099-01-01")
        assert regs == []
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_history_summary_returns_latest_scores():
    path = tmp_db()
    try:
        h = History(path)
        h.log([make_result(faithfulness=0.5)])
        h.log([make_result(faithfulness=0.9)])
        summary = h.summary()
        assert abs(summary["faithfulness"] - 0.9) < 0.001
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_history_multiple_labels_stored_and_queryable():
    path = tmp_db()
    try:
        h = History(path)
        h.log([make_result()], label="alpha")
        h.log([make_result()], label="beta")
        trend = h.trend("faithfulness")
        assert len(trend) == 2
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_history_db_created_if_not_exists():
    path = tmp_db()
    assert not os.path.exists(path)
    try:
        History(path)
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_history_readable_across_separate_instances_same_file():
    path = tmp_db()
    try:
        h1 = History(path)
        h1.log([make_result(faithfulness=0.77)])

        h2 = History(path)
        trend = h2.trend("faithfulness")
        assert len(trend) == 1
        assert abs(trend[0][1] - 0.77) < 0.001
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_history_empty_db_returns_empty_results():
    path = tmp_db()
    try:
        h = History(path)
        assert h.trend("faithfulness") == []
        assert h.regressions() == []
        assert h.summary() == {}
    finally:
        if os.path.exists(path):
            os.unlink(path)
