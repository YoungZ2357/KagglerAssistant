"""对话元数据存储：管理对话的名称、thread_id、工作区等应用层元数据。

使用独立 SQLite 文件，与 SqliteSaver 的 checkpoint DB 分离——
checkpoint 的 schema 由 LangGraph 管理，不混入应用层表。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT    NOT NULL UNIQUE,
    name        TEXT    NOT NULL,
    csv_path    TEXT    NOT NULL,
    workspace_path TEXT NOT NULL,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
"""


@dataclass
class ConversationRecord:
    id: int
    thread_id: str
    name: str
    csv_path: str
    workspace_path: str
    created_at: str
    updated_at: str


class ConversationStore:
    """对话元数据的 SQLite CRUD 层。"""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            db_path_str = str(self._db_path)
            # check_same_thread=False：连接可能在某个 worker 线程惰性创建，之后被
            # UI 主线程调用（如 /conversations 的 rename/delete）。本 store 无内部锁，
            # 依赖「单消费者、串行调用」假设——TUI 的 worker 逐个启动，不并发写。
            self._conn = sqlite3.connect(db_path_str, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_SCHEMA)
            self._conn.commit()
        return self._conn

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create(
        self,
        name: str,
        thread_id: str,
        csv_path: str,
        workspace_path: str,
    ) -> ConversationRecord:
        now = self._now()
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO conversations (thread_id, name, csv_path, workspace_path, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (thread_id, name, csv_path, workspace_path, now, now),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM conversations WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        return self._row_to_record(row)

    def get_by_thread_id(self, thread_id: str) -> ConversationRecord | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM conversations WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_name(self, name: str) -> ConversationRecord | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM conversations WHERE name = ?", (name,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def list_all(self, workspace_path: str | None = None) -> list[ConversationRecord]:
        conn = self._get_conn()
        if workspace_path is not None:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE workspace_path = ? ORDER BY updated_at DESC",
                (workspace_path,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def update_timestamp(self, thread_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE thread_id = ?",
            (self._now(), thread_id),
        )
        conn.commit()

    def rename(self, thread_id: str, new_name: str) -> bool:
        # 用 cursor.rowcount（本次语句影响的行数），不是 conn.total_changes——后者是
        # 连接开启以来的累计值，任意一次 create 之后恒为正，会对不存在的 thread_id 谎报成功。
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE conversations SET name = ?, updated_at = ? WHERE thread_id = ?",
            (new_name, self._now(), thread_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def delete(self, thread_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM conversations WHERE thread_id = ?", (thread_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def _row_to_record(self, row: sqlite3.Row) -> ConversationRecord:
        return ConversationRecord(
            id=row["id"],
            thread_id=row["thread_id"],
            name=row["name"],
            csv_path=row["csv_path"],
            workspace_path=row["workspace_path"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
