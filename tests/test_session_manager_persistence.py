# -*- coding: utf-8 -*-
"""持久化后端的线程与生命周期回归测试（无网络）。

- A1：make_sqlite_saver 的连接须能跨线程使用（check_same_thread=False）。
       TUI 在 "init" 线程建连接、在 "stream" 线程跑图；缺此则抛 ProgrammingError。
- A3：SessionManager.delete_conversation 须连同 checkpoint 一并清除，不留孤儿。
"""
import threading

import pytest
from langgraph.checkpoint.base import empty_checkpoint

from kaggler.graph.assembly import make_sqlite_saver
from kaggler.persistence.version_ledger_store import VersionLedgerStore
from kaggler.shared.session_manager import SessionManager
from kaggler.workspace import manager as ws_manager


@pytest.fixture(autouse=True)
def _isolate_last_workspace(tmp_path, monkeypatch):
    """把「上次工作区」状态文件重定向到 tmp，避免测试污染用户级 ~/.kaggler。"""
    state_dir = tmp_path / "_user_state"
    monkeypatch.setattr(ws_manager, "_USER_STATE_DIR", state_dir)
    monkeypatch.setattr(ws_manager, "_LAST_WORKSPACE_FILE", state_dir / "last_workspace")


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def _put_checkpoint(saver, thread_id: str) -> None:
    saver.put(_config(thread_id), empty_checkpoint(), {}, {})


class TestCrossThreadSaver:
    def test_saver_usable_from_another_thread(self, tmp_path):
        # 主线程建连接，另一线程写入——check_same_thread=False 下不应抛异常。
        saver = make_sqlite_saver(tmp_path / "checkpoints.sqlite")
        errors: list[Exception] = []

        def worker() -> None:
            try:
                _put_checkpoint(saver, "tid")
                assert saver.get_tuple(_config("tid")) is not None
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert errors == [], f"跨线程访问 saver 抛异常：{errors}"
        saver.conn.close()


class TestDeleteConversationPurgesCheckpoint:
    def test_delete_removes_metadata_and_checkpoint(self, tmp_path):
        mgr = SessionManager(tmp_path)
        thread_id = "tid-1"

        # 直接落一条元数据 + 一个 checkpoint，模拟一个已发生过对话的 thread。
        mgr._store.create(
            name="x", thread_id=thread_id, csv_path="/a.csv",
            workspace_path=str(mgr.workspace.path),
        )
        saver = make_sqlite_saver(mgr.workspace.checkpoint_db)
        _put_checkpoint(saver, thread_id)
        assert saver.get_tuple(_config(thread_id)) is not None
        saver.conn.close()

        mgr.delete_conversation(thread_id)

        # 元数据行已删。
        assert mgr._store.get_by_thread_id(thread_id) is None
        # checkpoint 也已被 delete_thread 清除（新开连接确认不是缓存假象）。
        verify = make_sqlite_saver(mgr.workspace.checkpoint_db)
        assert verify.get_tuple(_config(thread_id)) is None
        verify.conn.close()


class TestDeleteConversationPurgesVersionLedger:
    def test_delete_removes_version_ledger_rows(self, tmp_path):
        mgr = SessionManager(tmp_path)
        thread_id = "tid-led"

        mgr._store.create(
            name="x", thread_id=thread_id, csv_path="/a.csv",
            workspace_path=str(mgr.workspace.path),
        )
        ledger = VersionLedgerStore(mgr.workspace.version_ledger_db)
        ledger.record(
            thread_id=thread_id, version=0, parent=None, kind="source",
            tool=None, description="原始数据集", code="pl.read_csv('a.csv')",
        )
        ledger.record(
            thread_id=thread_id, version=1, parent=0, kind="derived",
            tool="standardize", description="std", code="lf = lf",
        )
        ledger.close()

        mgr.delete_conversation(thread_id)

        verify = VersionLedgerStore(mgr.workspace.version_ledger_db)
        assert verify.list_by_thread(thread_id) == []
        verify.close()
