"""
database.py — SQLite persistence layer for the Assessment AI project.

Why SQLite instead of CSV / JSONL?
  - Queryable: find all predictions for a crop, partner, year with SQL
  - Concurrent-safe: WAL mode allows multiple readers + one writer
  - Atomic writes: partial failures don't corrupt the file
  - Zero extra dependencies: sqlite3 is in Python's standard library
  - Portable: single file, no server to run

Tables
------
  prediction_runs   — one row per POST /predict call (plan context + summary)
  prediction_items  — one row per indicator predicted (linked to prediction_runs)
  feedback          — user corrections to AI predictions (replaces feedback.jsonl)
  model_registry    — record of every saved model checkpoint
"""

import sqlite3
import json
import datetime
import logging
import os
import threading
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# One connection per thread (sqlite3 connections are not thread-safe)
_local = threading.local()


class AssessmentDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_schema()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Return a per-thread connection with WAL mode for concurrency."""
        if not hasattr(_local, 'conn') or _local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            _local.conn = conn
        return _local.conn

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._connect()

    def _init_schema(self):
        """Create all tables if they don't already exist."""
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS prediction_runs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id          TEXT UNIQUE NOT NULL,   -- UUID from caller
                    created_at      TEXT NOT NULL,
                    country         TEXT,
                    crop            TEXT,
                    partner         TEXT,
                    subpartner      TEXT,
                    irrigation      TEXT,
                    hired_workers   TEXT,
                    area            REAL,
                    plan_year       TEXT,
                    total_indicators INTEGER,
                    auto_filled     INTEGER,
                    needs_review    INTEGER,
                    avg_confidence  REAL,
                    sustainability_score REAL,
                    sustainability_band  TEXT
                );

                CREATE TABLE IF NOT EXISTS prediction_items (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id          TEXT NOT NULL REFERENCES prediction_runs(run_id),
                    indicator       TEXT NOT NULL,
                    section         TEXT,
                    predicted_value TEXT,
                    confidence      REAL,
                    ui_status       TEXT,
                    policy          TEXT,
                    source          TEXT,
                    model_accuracy  REAL
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at      TEXT NOT NULL,
                    run_id          TEXT,
                    indicator       TEXT NOT NULL,
                    predicted_value TEXT,
                    actual_value    TEXT NOT NULL,
                    was_helpful     INTEGER,          -- 1=yes, 0=no, NULL=unknown
                    country         TEXT,
                    crop            TEXT,
                    partner         TEXT,
                    subpartner      TEXT,
                    irrigation      TEXT,
                    hired_workers   TEXT,
                    area            REAL,
                    plan_year       TEXT
                );

                CREATE TABLE IF NOT EXISTS model_registry (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    version         TEXT UNIQUE NOT NULL,
                    created_at      TEXT NOT NULL,
                    model_path      TEXT NOT NULL,
                    total_models    INTEGER,
                    avg_accuracy    REAL,
                    training_rows   INTEGER,
                    notes           TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_runs_crop    ON prediction_runs(crop);
                CREATE INDEX IF NOT EXISTS idx_runs_partner ON prediction_runs(partner);
                CREATE INDEX IF NOT EXISTS idx_runs_year    ON prediction_runs(plan_year);
                CREATE INDEX IF NOT EXISTS idx_items_run    ON prediction_items(run_id);
                CREATE INDEX IF NOT EXISTS idx_items_ind    ON prediction_items(indicator);
                CREATE INDEX IF NOT EXISTS idx_fb_indicator ON feedback(indicator);
            """)
        logger.info(f"Database ready: {self.db_path}")

    # ── Prediction Runs ────────────────────────────────────────────────────────

    def save_prediction_run(
        self,
        run_id: str,
        input_data: dict,
        predictions: dict,
        summary: dict,
        sustainability_score: Optional[dict],
        question_meta: dict,
    ) -> None:
        """
        Persist one complete prediction run.
        Inserts one row into prediction_runs and N rows into prediction_items.
        """
        import uuid
        now = datetime.datetime.utcnow().isoformat()
        score_val = sustainability_score.get("estimated_score") if sustainability_score else None
        score_band = sustainability_score.get("band") if sustainability_score else None

        try:
            with self._conn:
                self._conn.execute("""
                    INSERT OR REPLACE INTO prediction_runs
                        (run_id, created_at, country, crop, partner, subpartner,
                         irrigation, hired_workers, area, plan_year,
                         total_indicators, auto_filled, needs_review,
                         avg_confidence, sustainability_score, sustainability_band)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    run_id, now,
                    input_data.get('country_code') or input_data.get('country'),
                    input_data.get('crop_name') or input_data.get('crop'),
                    input_data.get('partner'),
                    input_data.get('subpartner'),
                    str(input_data.get('irrigation', '')),
                    str(input_data.get('hired_workers', '')),
                    input_data.get('area'),
                    str(input_data.get('planYear') or input_data.get('plan_year', '')),
                    summary.get('total_indicators', len(predictions)),
                    summary.get('auto_filled_count', 0),
                    summary.get('needs_review_count', 0),
                    summary.get('average_confidence', 0),
                    score_val,
                    score_band,
                ))

                rows = []
                for indicator, res in predictions.items():
                    meta = question_meta.get(indicator, {})
                    value = res.get('value') or str(res.get('suggestedOptions', [''])[0])
                    rows.append((
                        run_id,
                        indicator,
                        meta.get('section', 'General'),
                        str(value) if value is not None else None,
                        res.get('confidence'),
                        res.get('ui_status'),
                        res.get('policy'),
                        res.get('source'),
                        res.get('model_accuracy'),
                    ))

                self._conn.executemany("""
                    INSERT INTO prediction_items
                        (run_id, indicator, section, predicted_value, confidence,
                         ui_status, policy, source, model_accuracy)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, rows)

            logger.debug(f"Saved prediction run {run_id} ({len(predictions)} indicators)")
        except Exception as e:
            logger.warning(f"DB: failed to save prediction run: {e}")

    def get_prediction_history(self, limit: int = 50, crop: str = None, partner: str = None) -> list:
        """Return recent prediction runs, optionally filtered."""
        query = "SELECT * FROM prediction_runs WHERE 1=1"
        params = []
        if crop:
            query += " AND crop = ?"
            params.append(crop)
        if partner:
            query += " AND partner = ?"
            params.append(partner)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_prediction_items(self, run_id: str) -> list:
        """Return all indicator predictions for a specific run."""
        rows = self._conn.execute(
            "SELECT * FROM prediction_items WHERE run_id = ? ORDER BY indicator",
            (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Feedback ───────────────────────────────────────────────────────────────

    def save_feedback(self, payload: dict) -> int:
        """
        Store one feedback record.  Returns the new row id.
        Also writes to feedback.jsonl for backward compatibility with
        apply_feedback_to_training().
        """
        now = datetime.datetime.utcnow().isoformat()
        inp = payload.get('input', {})
        with self._conn:
            cur = self._conn.execute("""
                INSERT INTO feedback
                    (created_at, run_id, indicator, predicted_value, actual_value,
                     was_helpful, country, crop, partner, subpartner,
                     irrigation, hired_workers, area, plan_year)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                now,
                payload.get('run_id'),
                payload.get('indicator'),
                str(payload.get('predicted', '')),
                str(payload.get('actual', '')),
                1 if payload.get('was_helpful') is True else (0 if payload.get('was_helpful') is False else None),
                inp.get('country_code') or inp.get('country'),
                inp.get('crop_name') or inp.get('crop'),
                inp.get('partner'),
                inp.get('subpartner'),
                str(inp.get('irrigation', '')),
                str(inp.get('hired_workers', '')),
                inp.get('area'),
                str(inp.get('planYear') or inp.get('plan_year', '')),
            ))
            return cur.lastrowid

    def get_feedback_for_indicator(self, indicator: str) -> list:
        """Return all feedback rows for a given indicator."""
        rows = self._conn.execute(
            "SELECT * FROM feedback WHERE indicator = ? ORDER BY created_at DESC",
            (indicator,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_feedback_counts(self) -> list:
        """Return count of feedback entries grouped by indicator."""
        rows = self._conn.execute("""
            SELECT indicator,
                   COUNT(*) AS total,
                   SUM(CASE WHEN was_helpful = 1 THEN 1 ELSE 0 END) AS helpful,
                   SUM(CASE WHEN was_helpful = 0 THEN 1 ELSE 0 END) AS not_helpful
            FROM feedback
            GROUP BY indicator
            ORDER BY total DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def get_all_feedback_as_jsonl_entries(self) -> list:
        """
        Return feedback records formatted the same way as feedback.jsonl entries
        so that apply_feedback_to_training() can use the DB directly.
        """
        rows = self._conn.execute(
            "SELECT * FROM feedback WHERE actual_value IS NOT NULL AND actual_value != ''"
        ).fetchall()
        entries = []
        for r in rows:
            r = dict(r)
            entries.append({
                "indicator": r["indicator"],
                "actual": r["actual_value"],
                "input": {
                    "country_code": r.get("country"),
                    "crop_name":    r.get("crop"),
                    "partner":      r.get("partner"),
                    "subpartner":   r.get("subpartner"),
                    "irrigation":   r.get("irrigation"),
                    "hired_workers":r.get("hired_workers"),
                    "area":         r.get("area"),
                    "planYear":     r.get("plan_year"),
                },
            })
        return entries

    # ── Model Registry ─────────────────────────────────────────────────────────

    def register_model_version(
        self,
        version: str,
        model_path: str,
        total_models: int = 0,
        avg_accuracy: float = 0.0,
        training_rows: int = 0,
        notes: str = "",
    ) -> None:
        """Register a model checkpoint in the registry."""
        now = datetime.datetime.utcnow().isoformat()
        try:
            with self._conn:
                self._conn.execute("""
                    INSERT OR REPLACE INTO model_registry
                        (version, created_at, model_path, total_models, avg_accuracy, training_rows, notes)
                    VALUES (?,?,?,?,?,?,?)
                """, (version, now, model_path, total_models, avg_accuracy, training_rows, notes))
            logger.info(f"Model version registered: {version}")
        except Exception as e:
            logger.warning(f"DB: failed to register model version: {e}")

    def get_model_versions(self) -> list:
        """Return all registered model versions, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM model_registry ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Stats / Analytics ──────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return quick summary stats from the database."""
        try:
            total_runs = self._conn.execute("SELECT COUNT(*) FROM prediction_runs").fetchone()[0]
            total_feedback = self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            total_models = self._conn.execute("SELECT COUNT(*) FROM model_registry").fetchone()[0]
            crops = self._conn.execute(
                "SELECT crop, COUNT(*) as cnt FROM prediction_runs WHERE crop IS NOT NULL GROUP BY crop ORDER BY cnt DESC LIMIT 5"
            ).fetchall()
            return {
                "total_prediction_runs": total_runs,
                "total_feedback_entries": total_feedback,
                "registered_model_versions": total_models,
                "top_crops": [dict(r) for r in crops],
            }
        except Exception as e:
            return {"error": str(e)}
