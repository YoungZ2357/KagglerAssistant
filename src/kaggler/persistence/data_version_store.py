"""数据版本导出目录（catalog）：记录每次「版本 → 文件」导出的应用层元数据。

与 ConversationStore 同构：独立 SQLite 文件（.kaggler/data_versions.sqlite），
不混入 LangGraph checkpoint DB。仅登记导出产物（版本号、落盘路径、格式、谱系描述、
行列数、时间戳），供审计 / 未来的 /exports 浏览；本身不持有数据字节。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS exports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT,
    version     INTEGER NOT NULL,
    file_path   TEXT    NOT NULL,
    format      TEXT    NOT NULL,
    description TEXT    NOT NULL,
    rows        INTEGER NOT NULL,
    cols        INTEGER NOT NULL,
    created_at  TEXT    NOT NULL
);
"""


@dataclass
class ExportRecord:
    id: int
    thread_id: str | None
    version: int
    file_path: str
    format: str
    description: str
    rows: int
    cols: int
    created_at: str


class DataVersionStore:
    """导出目录的 SQLite CRUD 层（照 ConversationStore 范式）。"""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            # check_same_thread=False：连接可能在 worker 线程惰性创建、被 UI 主线程复用。
            # 无内部锁，依赖「单消费者、串行调用」假设（导出为低频、逐个触发）。
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_SCHEMA)
            self._conn.commit()
        return self._conn

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def record(
        self,
        *,
        version: int,
        file_path: str,
        format: str,
        description: str,
        rows: int,
        cols: int,
        thread_id: str | None = None,
    ) -> ExportRecord:
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO exports "
            "(thread_id, version, file_path, format, description, rows, cols, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (thread_id, version, file_path, format, description, rows, cols, self._now()),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM exports WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return self._row_to_record(row)

    def list_by_thread(self, thread_id: str) -> list[ExportRecord]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM exports WHERE thread_id = ? ORDER BY created_at DESC",
            (thread_id,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_all(self) -> list[ExportRecord]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM exports ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row: sqlite3.Row) -> ExportRecord:
        return ExportRecord(
            id=row["id"],
            thread_id=row["thread_id"],
            version=row["version"],
            file_path=row["file_path"],
            format=row["format"],
            description=row["description"],
            rows=row["rows"],
            cols=row["cols"],
            created_at=row["created_at"],
        )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
