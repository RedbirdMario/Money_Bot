"""SQLite-based trading journal."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "journal.db"


class JournalDB:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    strategy TEXT,
                    dataset TEXT,
                    entry_type TEXT,
                    tags TEXT,
                    sharpe REAL,
                    max_dd REAL,
                    notes TEXT,
                    params TEXT
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def add_entry(
        self,
        strategy: str = "",
        dataset: str = "",
        entry_type: str = "backtest",
        tags: str = "",
        sharpe: float = 0.0,
        max_dd: float = 0.0,
        notes: str = "",
        params: str = "{}",
    ):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO entries (created_at, strategy, dataset, entry_type, tags, sharpe, max_dd, notes, params)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.utcnow().isoformat(), strategy, dataset, entry_type, tags, sharpe, max_dd, notes, params),
            )

    def get_entries(
        self,
        entry_type: list[str] | None = None,
        tag: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        query = "SELECT * FROM entries WHERE 1=1"
        params = []

        if entry_type:
            placeholders = ",".join("?" * len(entry_type))
            query += f" AND entry_type IN ({placeholders})"
            params.extend(entry_type)

        if tag:
            query += " AND tags LIKE ?"
            params.append(f"%{tag}%")

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def delete_entry(self, entry_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
