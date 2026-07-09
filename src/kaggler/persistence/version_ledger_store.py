"""数据版本操作账本（version ledger）：逐版本持久化血缘 + 惰性操作代码片段。

与 DataVersionStore / ConversationStore 同构：独立 SQLite 文件
（.kaggler/version_ledger.sqlite），不混入 LangGraph checkpoint DB。

用途：DataProvider 的版本图本活在内存，恢复对话会丢失全部派生版本。本账本按
(thread_id, version) 记录每个版本的 parent / tool / description / reproducible 及其
Polars 代码片段（source 存读取表达式、派生版本存操作 ``lf`` 的语句），使恢复时能
重放片段重建整棵版本树。HEAD/当前版本不在此——由 LangGraph checkpoint 的
``data_version`` 唯一持有。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id     TEXT    NOT NULL,
    version       INTEGER NOT NULL,
    parent        INTEGER,
    kind          TEXT    NOT NULL,
    tool          TEXT,
    description   TEXT    NOT NULL,
    reproducible  INTEGER NOT NULL DEFAULT 1,
    code          TEXT,
    snapshot_path TEXT,
    created_at    TEXT    NOT NULL,
    UNIQUE(thread_id, version)
);
"""


@dataclass
class VersionRecord:
    id: int
    thread_id: str
    version: int
    parent: int | None
    kind: str  # 'source' | 'derived'
    tool: str | None
    description: str
    reproducible: bool
    code: str | None
    snapshot_path: str | None
    created_at: str


class VersionLedgerStore:
    """版本操作账本的 SQLite CRUD 层（照 DataVersionStore 范式）。"""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            # check_same_thread=False：连接可能在 worker 线程惰性创建、被 UI 主线程复用。
            # 无内部锁，依赖「单消费者、串行调用」假设（版本写入为低频、逐个触发）。
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
        thread_id: str,
        version: int,
        parent: int | None,
        kind: str,
        tool: str | None,
        description: str,
        reproducible: bool = True,
        code: str | None,
        snapshot_path: str | None = None,
    ) -> VersionRecord:
        """登记一个版本。(thread_id, version) 幂等：重放/重复写以最新一条为准。"""
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO versions "
            "(thread_id, version, parent, kind, tool, description, reproducible, "
            "code, snapshot_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(thread_id, version) DO UPDATE SET "
            "parent=excluded.parent, kind=excluded.kind, tool=excluded.tool, "
            "description=excluded.description, reproducible=excluded.reproducible, "
            "code=excluded.code, snapshot_path=excluded.snapshot_path, "
            "created_at=excluded.created_at",
            (
                thread_id, version, parent, kind, tool, description,
                int(reproducible), code, snapshot_path, self._now(),
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM versions WHERE thread_id = ? AND version = ?",
            (thread_id, version),
        ).fetchone()
        _ = cursor  # lastrowid 在 UPSERT 更新分支下不可靠，改按 (thread_id, version) 回选
        return self._row_to_record(row)

    def list_by_thread(self, thread_id: str) -> list[VersionRecord]:
        """按 version 升序返回某对话的全部版本（重建时依序重放）。"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM versions WHERE thread_id = ? ORDER BY version ASC",
            (thread_id,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def delete_by_thread(self, thread_id: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM versions WHERE thread_id = ?", (thread_id,))
        conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> VersionRecord:
        return VersionRecord(
            id=row["id"],
            thread_id=row["thread_id"],
            version=row["version"],
            parent=row["parent"],
            kind=row["kind"],
            tool=row["tool"],
            description=row["description"],
            reproducible=bool(row["reproducible"]),
            code=row["code"],
            snapshot_path=row["snapshot_path"],
            created_at=row["created_at"],
        )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
