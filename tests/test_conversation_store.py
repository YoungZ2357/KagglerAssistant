# -*- coding: utf-8 -*-
"""ConversationStore（persistence/conversation_store.py）单测。

真实读写临时 SQLite 文件（无网络）。重点回归：rename/delete 用 cursor.rowcount，
对不存在的 thread_id 必须返回 False——历史 bug 是用 conn.total_changes（连接累计值），
任意一次 create 之后恒为真。
"""
import pytest

from kaggler.persistence.conversation_store import ConversationStore


@pytest.fixture
def store(tmp_path):
    s = ConversationStore(tmp_path / "conversations.sqlite")
    yield s
    s.close()


def _create(store, thread_id="t1", name="对话一"):
    return store.create(
        name=name, thread_id=thread_id, csv_path="/data/a.csv", workspace_path="/ws"
    )


class TestCrud:
    def test_create_and_get(self, store):
        rec = _create(store)
        assert rec.thread_id == "t1"
        assert rec.name == "对话一"
        assert store.get_by_thread_id("t1").csv_path == "/data/a.csv"

    def test_get_missing_returns_none(self, store):
        assert store.get_by_thread_id("nope") is None

    def test_list_filters_by_workspace(self, store):
        store.create(name="x", thread_id="t1", csv_path="/a.csv", workspace_path="/ws1")
        store.create(name="y", thread_id="t2", csv_path="/b.csv", workspace_path="/ws2")
        assert [r.thread_id for r in store.list_all("/ws1")] == ["t1"]
        assert len(store.list_all()) == 2


class TestRenameDeleteReturnValue:
    """A2 回归：返回值必须反映本次语句是否真的改到行。"""

    def test_rename_existing_returns_true(self, store):
        _create(store)
        assert store.rename("t1", "新名") is True
        assert store.get_by_thread_id("t1").name == "新名"

    def test_rename_missing_returns_false_even_after_create(self, store):
        # 先 create（使连接累计 total_changes>0），再对不存在的 thread_id rename。
        _create(store)
        assert store.rename("does-not-exist", "x") is False

    def test_delete_existing_returns_true(self, store):
        _create(store)
        assert store.delete("t1") is True
        assert store.get_by_thread_id("t1") is None

    def test_delete_missing_returns_false_even_after_create(self, store):
        _create(store)
        assert store.delete("does-not-exist") is False
