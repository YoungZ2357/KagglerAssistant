# -*- coding: utf-8 -*-
"""data_export（persistence/data_export.py）单测。

真实读写临时文件与 SQLite（无网络）。覆盖：格式分派、CSV/parquet 落盘往返、
坏版本/坏格式的异常、export_and_record 的「落盘 + 登记目录」编排。
"""
from pathlib import Path

import polars as pl
import pytest

from kaggler.persistence.data_export import (
    ExportResult,
    export_and_record,
    export_version,
    resolve_format,
)
from kaggler.persistence.data_provider import DataProvider
from kaggler.persistence.data_version_store import DataVersionStore


@pytest.fixture
def data() -> DataProvider:
    dp = DataProvider()
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    dp.add_source(lambda: df, description="原始数据集")
    return dp


class TestResolveFormat:
    def test_explicit_fmt_wins(self):
        assert resolve_format(Path("out.csv"), "parquet") == "parquet"

    def test_infer_from_suffix(self):
        assert resolve_format(Path("out.parquet"), None) == "parquet"
        assert resolve_format(Path("out.pq"), None) == "parquet"
        assert resolve_format(Path("out.csv"), None) == "csv"

    def test_default_when_no_suffix(self):
        assert resolve_format(Path("out"), None) == "csv"

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="不支持的导出格式"):
            resolve_format(Path("out.xml"), None)
        with pytest.raises(ValueError, match="不支持的导出格式"):
            resolve_format(Path("out.csv"), "xml")


class TestExportVersion:
    def test_csv_roundtrip(self, data, tmp_path):
        target = tmp_path / "sub" / "out.csv"
        result = export_version(data, 0, target)
        assert isinstance(result, ExportResult)
        assert result.format == "csv"
        assert result.rows == 3 and result.cols == 2
        assert target.exists()  # 父目录被自动创建
        assert pl.read_csv(target)["a"].to_list() == [1, 2, 3]

    def test_parquet_roundtrip(self, data, tmp_path):
        target = tmp_path / "out.parquet"
        result = export_version(data, 0, target)
        assert result.format == "parquet"
        assert pl.read_parquet(target)["b"].to_list() == ["x", "y", "z"]

    def test_explicit_fmt_overrides_suffix(self, data, tmp_path):
        target = tmp_path / "out.data"
        result = export_version(data, 0, target, fmt="parquet")
        assert result.format == "parquet"
        assert pl.read_parquet(target).height == 3

    def test_nonexistent_version_raises(self, data, tmp_path):
        with pytest.raises(RuntimeError, match="不存在"):
            export_version(data, 99, tmp_path / "out.csv")

    def test_bad_format_raises(self, data, tmp_path):
        with pytest.raises(ValueError):
            export_version(data, 0, tmp_path / "out.xml")


class TestExportAndRecord:
    def test_writes_file_and_records(self, data, tmp_path):
        target = tmp_path / "exports" / "v0.csv"
        db_path = tmp_path / "data_versions.sqlite"
        result = export_and_record(
            data, 0, target, None,
            db_path=db_path, thread_id="t1", description="原始数据集",
        )
        assert target.exists()
        store = DataVersionStore(db_path)
        try:
            rows = store.list_all()
        finally:
            store.close()
        assert len(rows) == 1
        assert rows[0].version == 0
        assert rows[0].file_path == result.path
        assert rows[0].thread_id == "t1"
        assert rows[0].description == "原始数据集"
        assert rows[0].rows == 3 and rows[0].cols == 2

    def test_skips_record_when_no_db(self, data, tmp_path):
        target = tmp_path / "out.csv"
        result = export_and_record(data, 0, target, None, db_path=None)
        assert target.exists()
        assert result.rows == 3
