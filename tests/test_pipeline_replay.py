# -*- coding: utf-8 -*-
"""pipeline_replay（persistence/pipeline_replay.py）单测。

核心证明：把持久化的 IR 经 interpreter 重建 op 后，重建的每个版本逐值等于原
DataProvider（含 fork 分支树），即「IR 重建 op == 原闭包」。真实读写临时文件（无网络）。
"""
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from kaggler.ir import IRNode, dumps_ir
from kaggler.modes.feature_engineering import compute
from kaggler.persistence.data_provider import DataProvider
from kaggler.persistence.pipeline_replay import rebuild_into
from kaggler.persistence.version_ledger_store import VersionLedgerStore


class _Sink:
    """把 DataProvider 的版本登记转发进 VersionLedgerStore（测试用）。"""

    def __init__(self, db_path: Path, thread_id: str) -> None:
        self._db_path = db_path
        self._thread_id = thread_id

    def record_version(self, version, *, ir=None, **kw):
        s = VersionLedgerStore(self._db_path)
        try:
            s.record(thread_id=self._thread_id, version=version,
                     ir=dumps_ir(ir) if ir is not None else None, **kw)
        finally:
            s.close()


def _make_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "train.csv"
    pl.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "income": [1.0, 2.0, 3.0, 4.0, 5.0],
        "city": ["a", "b", "a", "c", "b"],
        "target": [0, 1, 0, 1, 1],
    }).write_csv(csv)
    return csv


def _build_tree(dp: DataProvider, root: int) -> dict[int, pl.DataFrame]:
    """构造一棵含 fork 的版本树，返回 {version: 期望帧}。"""
    def step(fn, parent, tool, *args):
        r = fn(dp.get(parent), *args)
        assert "error" not in r, (tool, r)
        return dp.add_version(r["op"], parent=parent, tool=tool, description=tool,
                              ir=r["ir"])

    v1 = step(compute.exec_standardize, root, "standardize", ["income"])
    v2 = step(compute.exec_encode, v1, "encode", [{"column": "city", "action": "label"}])
    v3 = step(compute.exec_transform_mono, v2, "mono", [{"column": "age", "method": "square"}])
    v4 = step(compute.exec_dim_reduct, v3, "pca", "pca", 2)
    # fork：回到 v2 派生另一分支（parent 非最新版）
    dp.set_head(v2)
    v5 = step(compute.exec_drop_columns, v2, "drop", ["target"])
    v6 = step(
        compute.exec_filter_rows, v5, "filter",
        [{"logic": "and", "conditions": [{"column": "age", "op": "gt", "value": 25}]}],
        "and", "keep",
    )
    versions = [root, v1, v2, v3, v4, v5, v6]
    return {v: dp.get(v) for v in versions}


class TestRebuildRoundTrip:
    def test_rebuild_reproduces_every_version_incl_fork(self, tmp_path):
        csv = _make_csv(tmp_path)
        db = tmp_path / "version_ledger.sqlite"
        tid = "t1"

        dp = DataProvider(sink=_Sink(db, tid))
        root = dp.load_initial(str(csv))
        expected = _build_tree(dp, root)

        store = VersionLedgerStore(db)
        records = store.list_by_thread(tid)
        store.close()
        # 账本按 version 升序，且 fork 分支的 parent 指针被保留。
        assert [r.version for r in records] == sorted(expected)
        assert {r.version: r.parent for r in records}[5] == 2  # v5 fork 自 v2

        dp2 = DataProvider()
        rebuild_into(dp2, records)

        for v, exp in expected.items():
            assert_frame_equal(
                dp2.get(v), exp, check_dtypes=False, rel_tol=1e-6, abs_tol=1e-8
            )
        # next_version 抬到 max+1，续写不会撞号。
        assert dp2._next_version == max(expected) + 1

    def test_code_export_works_after_rebuild(self, tmp_path):
        csv = _make_csv(tmp_path)
        db = tmp_path / "version_ledger.sqlite"
        tid = "t1"
        dp = DataProvider(sink=_Sink(db, tid))
        root = dp.load_initial(str(csv))
        expected = _build_tree(dp, root)
        head = max(expected)

        store = VersionLedgerStore(db)
        records = store.list_by_thread(tid)
        store.close()

        dp2 = DataProvider()
        rebuild_into(dp2, records)
        code = dp2.generate_pipeline_code(head)
        assert code.startswith("import polars as pl")
        assert "lf = lf" in code

    def test_derived_without_ir_raises(self, tmp_path):
        """派生记录无 IR(旧账本 / eager_op 桥)时响亮报错,不产出残缺版本树。"""
        csv = _make_csv(tmp_path)
        db = tmp_path / "version_ledger.sqlite"
        tid = "t1"
        src_ir = dumps_ir(IRNode(version=0, kind="source", parents=[],
                                 params={"format": "csv", "path": str(csv)}))
        store = VersionLedgerStore(db)
        store.record(thread_id=tid, version=0, parent=None, kind="source",
                     tool=None, description="src", ir=src_ir)
        store.record(thread_id=tid, version=1, parent=0, kind="derived",
                     tool="mystery", description="x", ir=None)
        records = store.list_by_thread(tid)
        store.close()

        dp = DataProvider()
        with pytest.raises(ValueError, match="无 IR 记录"):
            rebuild_into(dp, records)
