# -*- coding: utf-8 -*-
"""IR 持久化与跨进程恢复验证(阶段 3 gate,§8 经验主义)。

- 跨进程 round-trip(阻断性):subprocess 构建版本树并落账本(IR)→ 本进程
  rebuild_into 经 IR→interpreter 重建 → 逐版本与子进程写下的 parquet 精确比对。
  这同时是「运行时闭包 vs 恢复路径」的跨进程差分,强度最高。
- State 通道跨进程 round-trip:file-backed SqliteSaver(非 MemorySaver)。
- SqliteSaver serde 对 IR 形态 payload 的最小实测(§7.3 信息性:本设计 IR 不入
  State,只为把「能否接纳」从推断变实测)。
- 账本 schema migration:旧库(无 ir 列)自动补列,旧行 ir 为 NULL。
"""
import json
import math
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from kaggler.persistence.data_provider import DataProvider
from kaggler.persistence.pipeline_replay import rebuild_into
from kaggler.persistence.version_ledger_store import VersionLedgerStore

_CHILD = Path(__file__).with_name("_ir_subprocess_child.py")
_SRC = Path(__file__).resolve().parent.parent / "src"


def _run_child(*args) -> str:
    env = {**os.environ, "PYTHONPATH": str(_SRC)}
    proc = subprocess.run(
        [sys.executable, str(_CHILD), *map(str, args)],
        capture_output=True, text=True, env=env, timeout=180,
    )
    assert proc.returncode == 0, (
        f"子进程失败:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    return proc.stdout


class TestCrossProcessRoundTrip:
    def test_child_persists_parent_restores_frames_equal(self, tmp_path):
        csv = tmp_path / "train.csv"
        pl.DataFrame({
            "age": [20, 30, 40, 50, 60, 35],
            "income": [1.0, 2.0, 3.0, 4.0, 5.0, 2.5],
            "bonus": [10.0, None, 30.0, None, 50.0, 20.0],
            # 分组 mode 平票无关:city=x 组 note 众数 a;y 组全空回落全局;全局众数 a
            "note": ["a", None, "b", "a", None, "a"],
            "city": ["x", "y", "x", "z", "y", "x"],
            "target": [0, 1, 0, 1, 1, 0],
        }).write_csv(csv)
        ledger = tmp_path / "ledger.sqlite"
        outdir = tmp_path / "frames"
        outdir.mkdir()

        stdout = _run_child("tree", csv, ledger, "t-xproc", outdir)
        versions = json.loads(stdout.strip().splitlines()[-1])["versions"]

        store = VersionLedgerStore(ledger)
        records = store.list_by_thread("t-xproc")
        store.close()
        assert [r.version for r in records] == sorted(versions)
        assert all(r.ir is not None for r in records), "每条账本记录都应携带 IR"

        dp = DataProvider()
        rebuild_into(dp, records)
        for v in versions:
            expected = pl.read_parquet(outdir / f"v{v}.parquet")
            assert_frame_equal(dp.get(v), expected, check_exact=True)
        # 续号不撞
        assert dp._next_version == max(versions) + 1


class TestStateCrossProcess:
    def test_state_channels_roundtrip_via_file_saver(self, tmp_path):
        db = tmp_path / "ckpt.sqlite"
        _run_child("state", db, "t-state")

        from langgraph.graph import END, START, StateGraph

        from kaggler.graph.assembly import make_sqlite_saver
        from kaggler.graph.state import CommonState
        from kaggler.shared.types import Mode

        saver = make_sqlite_saver(db)
        g = StateGraph(CommonState)
        g.add_node("noop", lambda state: {})
        g.add_edge(START, "noop")
        g.add_edge("noop", END)
        graph = g.compile(checkpointer=saver)
        values = graph.get_state({"configurable": {"thread_id": "t-state"}}).values
        saver.conn.close()

        assert values["current_mode"] == Mode.FEAT_ENG
        assert values["file_path"] == "train.csv"
        assert values["explored_schema"] == "age:Int64"
        assert values["turn"] == 3
        assert values["memory"] == {"goal": "验证", "findings": ["a", "b"]}
        assert values["data_version"] == 5
        assert values["todos"] == [{"id": 1, "content": "todo-1", "status": "open"}]
        assert values["plans"] == [
            {"id": 1, "title": "p", "content": "c", "status": "draft"},
        ]
        assert values["context_usage"] == {"total": 123}
        assert values["messages"][0].content == "跨进程持久化验证"


class TestSqliteSaverSerdeProbe:
    def test_saver_accepts_ir_shaped_payload(self, tmp_path):
        """§7.3 信息性实测:含 nested array + nan/inf 的 IR 形态 dict 能被
        JsonPlusSerializer + file-backed SqliteSaver 写入并读回。
        本设计 IR 走账本、不入 State——此测试仅钉死结论,非依赖。
        """
        from langgraph.checkpoint.base import empty_checkpoint

        from kaggler.graph.assembly import make_sqlite_saver

        payload = {
            "kind": "dim_reduct",
            "components": [
                {"bias": float("nan"), "weights": [1.0, float("inf"), -2.5]},
            ],
        }
        config = {"configurable": {"thread_id": "t-serde", "checkpoint_ns": ""}}
        saver = make_sqlite_saver(tmp_path / "c.sqlite")
        ckpt = empty_checkpoint()
        ckpt["channel_values"] = {"ir_probe": payload}
        saver.put(config, ckpt, {}, {})
        saver.conn.close()

        verify = make_sqlite_saver(tmp_path / "c.sqlite")
        got = verify.get_tuple(config).checkpoint["channel_values"]["ir_probe"]
        verify.conn.close()
        assert math.isnan(got["components"][0]["bias"])
        assert got["components"][0]["weights"] == [1.0, float("inf"), -2.5]


_OLD_SCHEMA = """
CREATE TABLE versions (
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


class TestLedgerMigration:
    def test_old_table_without_ir_column_gets_migrated(self, tmp_path):
        db = tmp_path / "old.sqlite"
        conn = sqlite3.connect(str(db))
        conn.execute(_OLD_SCHEMA)
        conn.execute(
            "INSERT INTO versions (thread_id, version, parent, kind, tool, "
            "description, reproducible, code, snapshot_path, created_at) "
            "VALUES ('t', 0, NULL, 'source', NULL, 'src', 1, "
            "'pl.read_csv(''x.csv'')', NULL, '2026-01-01T00:00:00')"
        )
        conn.commit()
        conn.close()

        store = VersionLedgerStore(db)
        rows = store.list_by_thread("t")
        assert len(rows) == 1
        assert rows[0].ir is None  # 旧行补列后为 NULL
        assert rows[0].code == "pl.read_csv('x.csv')"  # 旧行 code 残值可读
        store.record(  # 补列后新行可正常写 ir
            thread_id="t", version=1, parent=0, kind="derived", tool="x",
            description="d", ir='{"probe": true}',
        )
        assert store.list_by_thread("t")[1].ir == '{"probe": true}'
        store.close()

    def test_rebuild_old_ledger_without_ir_raises(self, tmp_path):
        """旧账本(记录无 IR、只有 code)恢复时响亮报错——已拍板不兼容。"""
        db = tmp_path / "old.sqlite"
        conn = sqlite3.connect(str(db))
        conn.execute(_OLD_SCHEMA)
        conn.executemany(
            "INSERT INTO versions (thread_id, version, parent, kind, tool, "
            "description, reproducible, code, snapshot_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, '2026-01-01T00:00:00')",
            [
                ("t", 0, None, "source", None, "src", 1, "pl.read_csv('x.csv')"),
                ("t", 1, 0, "derived", "standardize", "std", 1, "lf = lf"),
            ],
        )
        conn.commit()
        conn.close()

        store = VersionLedgerStore(db)
        records = store.list_by_thread("t")
        store.close()

        dp = DataProvider()
        with pytest.raises(ValueError, match="无 IR 记录"):
            rebuild_into(dp, records)
