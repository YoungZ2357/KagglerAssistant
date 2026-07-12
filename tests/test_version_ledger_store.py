# -*- coding: utf-8 -*-
"""VersionLedgerStore（persistence/version_ledger_store.py）单测。

真实读写临时 SQLite 文件（无网络）。覆盖 record + list_by_thread 升序、
(thread_id, version) 幂等 UPSERT、多 thread 隔离、delete_by_thread。
"""
import pytest

from kaggler.persistence.version_ledger_store import VersionLedgerStore


@pytest.fixture
def store(tmp_path):
    s = VersionLedgerStore(tmp_path / "version_ledger.sqlite")
    yield s
    s.close()


def _record(store, *, version, thread_id="t1", parent=None, kind="derived",
            tool="standardize", code="lf = lf", reproducible=True):
    return store.record(
        thread_id=thread_id, version=version, parent=parent, kind=kind,
        tool=tool, description=f"step {version}", reproducible=reproducible, code=code,
    )


class TestRecord:
    def test_record_source_and_roundtrip(self, store):
        rec = _record(store, version=0, parent=None, kind="source",
                      tool=None, code="pl.read_csv('x.csv')")
        assert rec.id is not None
        assert rec.version == 0 and rec.parent is None
        assert rec.kind == "source" and rec.tool is None
        assert rec.reproducible is True
        assert rec.code == "pl.read_csv('x.csv')"

    def test_list_by_thread_orders_by_version_asc(self, store):
        _record(store, version=2)
        _record(store, version=0, parent=None, kind="source", tool=None)
        _record(store, version=1, parent=0)
        assert [r.version for r in store.list_by_thread("t1")] == [0, 1, 2]

    def test_upsert_is_idempotent_on_thread_version(self, store):
        _record(store, version=1, code="lf = lf.drop(['a'])")
        _record(store, version=1, code="lf = lf.drop(['b'])")  # 同 (t1,1) 覆盖
        rows = store.list_by_thread("t1")
        assert len(rows) == 1
        assert rows[0].code == "lf = lf.drop(['b'])"

    def test_thread_isolation(self, store):
        _record(store, version=0, thread_id="t1", parent=None, kind="source", tool=None)
        _record(store, version=0, thread_id="t2", parent=None, kind="source", tool=None)
        _record(store, version=1, thread_id="t2")
        assert [r.version for r in store.list_by_thread("t1")] == [0]
        assert [r.version for r in store.list_by_thread("t2")] == [0, 1]

    def test_delete_by_thread(self, store):
        _record(store, version=0, thread_id="t1", parent=None, kind="source", tool=None)
        _record(store, version=1, thread_id="t1")
        _record(store, version=0, thread_id="t2", parent=None, kind="source", tool=None)
        store.delete_by_thread("t1")
        assert store.list_by_thread("t1") == []
        assert [r.version for r in store.list_by_thread("t2")] == [0]

    def test_reproducible_persisted_as_bool(self, store):
        _record(store, version=1, reproducible=False)
        assert store.list_by_thread("t1")[0].reproducible is False
