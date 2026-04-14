from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional

from ragcheck.core import EvalResult

_CORE_METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]


class History:
    """Log eval runs to a local SQLite database and query trends over time.

    No server required. All data stays in a local file you control.

    Example:
        history = ragcheck.History("ragcheck.db")
        history.log(results, label="after-prompt-v3")
        history.trend("faithfulness", days=30)
        history.summary()
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT,
                    timestamp TEXT NOT NULL,
                    result_json TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def log(
        self,
        results: List[EvalResult] | EvalResult,
        label: Optional[str] = None,
    ) -> None:
        """Log one or more EvalResult objects to the history database.

        Args:
            results: A single EvalResult or a list of EvalResults.
            label: Optional human-readable label (e.g. "after-prompt-v3").
        """
        if isinstance(results, EvalResult):
            results = [results]
        ts = datetime.now(timezone.utc).isoformat()
        payload = json.dumps([r.to_dict() for r in results])
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO runs (label, timestamp, result_json) VALUES (?, ?, ?)",
                (label, ts, payload),
            )
            conn.commit()
        finally:
            conn.close()

    def trend(
        self,
        metric: str,
        days: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """Return average metric score per run, optionally filtered by recency.

        Args:
            metric: One of the core metric names (e.g. "faithfulness").
            days: If set, only include runs from the last N days.

        Returns:
            List of (timestamp, avg_score) tuples, oldest first.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT timestamp, result_json FROM runs ORDER BY id ASC"
            ).fetchall()
        finally:
            conn.close()

        result = []
        cutoff = None
        if days is not None:
            from datetime import timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        for row in rows:
            ts_str = row["timestamp"]
            ts = datetime.fromisoformat(ts_str)
            if cutoff and ts < cutoff:
                continue
            items = json.loads(row["result_json"])
            scores = [r.get(metric) for r in items if r.get(metric) is not None]
            if scores:
                avg = sum(scores) / len(scores)
                result.append((ts_str, avg))

        return result

    def regressions(
        self,
        since: Optional[str] = None,
    ) -> list[dict]:
        """Return runs where any core metric average dropped vs the previous run.

        Args:
            since: ISO date string (e.g. "2026-04-01"). Only check runs after this date.

        Returns:
            List of dicts: {label, timestamp, metric, before, after, delta}
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT label, timestamp, result_json FROM runs ORDER BY id ASC"
            ).fetchall()
        finally:
            conn.close()

        if since:
            cutoff = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
            rows = [r for r in rows if datetime.fromisoformat(r["timestamp"]) >= cutoff]

        found = []
        prev_avgs: dict[str, float] = {}

        for row in rows:
            items = json.loads(row["result_json"])
            cur_avgs: dict[str, float] = {}
            for metric in _CORE_METRICS:
                scores = [r.get(metric) for r in items if r.get(metric) is not None]
                if scores:
                    cur_avgs[metric] = sum(scores) / len(scores)

            for metric, cur in cur_avgs.items():
                prev = prev_avgs.get(metric)
                if prev is not None and cur < prev:
                    found.append({
                        "label": row["label"],
                        "timestamp": row["timestamp"],
                        "metric": metric,
                        "before": prev,
                        "after": cur,
                        "delta": cur - prev,
                    })

            prev_avgs = cur_avgs

        return found

    def summary(self) -> dict[str, float]:
        """Return the latest average score per metric across all tracked metrics.

        Returns:
            Dict mapping metric name → average score from the most recent run.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT result_json FROM runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()

        if not row:
            return {}

        items = json.loads(row["result_json"])
        out: dict[str, float] = {}
        for metric in _CORE_METRICS:
            scores = [r.get(metric) for r in items if r.get(metric) is not None]
            if scores:
                out[metric] = sum(scores) / len(scores)
        return out
