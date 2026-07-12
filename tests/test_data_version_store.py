# -*- coding: utf-8 -*-
"""DataVersionStore（persistence/data_version_store.py）单测。

真实读写临时 SQLite 文件（无网络）。覆盖 record + list_by_thread/list_all。
"""
import pytest

from kaggler.persistence.data_version_store import DataVersionStore


@pytest.fixture
def store(tmp_path):
    s = DataVersionStore(tmp_path / "data_versions.sqlite")
    yield s
    s.close()


def _record(store, *, version=0, thread_id="t1", path="/ws/.kaggler/exports/v0.csv"):
    return store.record(
        version=version,
        file_path=path,
        format="csv",
        description="原始数据集",
        rows=3,
        cols=2,
        thread_id=thread_id,
    )


class TestRecord:
    def test_record_and_list_all(self, store):
        rec = _record(store)
        assert rec.id is not None
        assert rec.version == 0
        assert rec.thread_id == "t1"
        assert rec.format == "csv"
        assert store.list_all()[0].file_path.endswith("v0.csv")

    def test_thread_id_nullable(self, store):
        rec = store.record(
            version=0, file_path="/x.csv", format="csv",
            description="d", rows=1, cols=1, thread_id=None,
        )
        assert rec.thread_id is None

    def test_list_by_thread_filters(self, store):
        _record(store, version=0, thread_id="t1")
        _record(store, version=1, thread_id="t1", path="/ws/exports/v1.csv")
        _record(store, version=0, thread_id="t2")
        assert {r.version for r in store.list_by_thread("t1")} == {0, 1}
        assert [r.version for r in store.list_by_thread("t2")] == [0]
        assert len(store.list_all()) == 3
