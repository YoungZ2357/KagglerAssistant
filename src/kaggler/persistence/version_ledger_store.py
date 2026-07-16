"""数据版本操作账本（version ledger）：逐版本持久化血缘 + IR（中间表示）。

与 DataVersionStore / ConversationStore 同构：独立 SQLite 文件
（.kaggler/version_ledger.sqlite），不混入 LangGraph checkpoint DB。

用途：DataProvider 的版本图本活在内存，恢复对话会丢失全部派生版本。本账本按
(thread_id, version) 记录每个版本的 parent / tool / description / reproducible 及其
IR JSON（kaggler.ir.dumps_ir 产出），使恢复时能经同一 interpreter 重建整棵版本树。
HEAD/当前版本不在此——由 LangGraph checkpoint 的 ``data_version`` 唯一持有。
code 列为 IR 重构前的历史遗留，只读保留、不再写入。
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
    ir            TEXT,
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
    ir: str | None  # IR 节点的 JSON 文本(kaggler.ir.dumps_ir 产出),恢复真相
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
            # 幂等补列:CREATE TABLE IF NOT EXISTS 不会改老表,IR 重构前建的库
            # 缺 ir 列,在此 ALTER 补上(旧行该列为 NULL,恢复时响亮报错)。
            cols = {row[1] for row in self._conn.execute("PRAGMA table_info(versions)")}
            if "ir" not in cols:
                self._conn.execute("ALTER TABLE versions ADD COLUMN ir TEXT")
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
        ir: str | None = None,
        snapshot_path: str | None = None,
    ) -> VersionRecord:
        """登记一个版本。(thread_id, version) 幂等：重放/重复写以最新一条为准。

        code 物理列保留(旧库兼容、免表重建迁移)但**不再写入**——IR 重构后
        持久化真相是 ir 列;``VersionRecord.code`` 仅只读(旧行残值)。
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO versions "
            "(thread_id, version, parent, kind, tool, description, reproducible, "
            "ir, snapshot_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(thread_id, version) DO UPDATE SET "
            "parent=excluded.parent, kind=excluded.kind, tool=excluded.tool, "
            "description=excluded.description, reproducible=excluded.reproducible, "
            "ir=excluded.ir, snapshot_path=excluded.snapshot_path, "
            "created_at=excluded.created_at",
            (
                thread_id, version, parent, kind, tool, description,
                int(reproducible), ir, snapshot_path, self._now(),
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
            ir=row["ir"],
            snapshot_path=row["snapshot_path"],
            created_at=row["created_at"],
        )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
