# -*- coding: utf-8 -*-
"""data_export（persistence/data_export.py）单测。

真实读写临时文件与 SQLite（无网络）。覆盖：格式分派、CSV/parquet 落盘往返、
坏版本/坏格式的异常、export_and_record 的「落盘 + 登记目录」编排。
"""
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from kaggler.modes.feature_engineering import compute
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

    def test_pipeline_py_format(self):
        assert resolve_format(Path("pipeline.py"), None) == "py"
        assert resolve_format(Path("out.txt"), "py") == "py"


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


def _build_multistep(tmp_path: Path) -> tuple[DataProvider, int]:
    """构造一个覆盖 standardize/encode/mono/combo/filter/drop/pca 的多步版本链。

    用 load_initial(真实 CSV) 建 source，使源版本带有可生成的读取代码。
    """
    csv = tmp_path / "train.csv"
    pl.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "income": [1.0, 2.0, 3.0, 4.0, 5.0],
        "city": ["a", "b", "a", "c", "b"],
        "target": [0, 1, 0, 1, 1],
    }).write_csv(csv)

    dp = DataProvider()
    v = dp.load_initial(str(csv))
    steps = [
        ("standardize", compute.exec_standardize, [["income"]]),
        ("encode", compute.exec_encode, [[{"column": "city", "action": "label"}]]),
        ("mono", compute.exec_transform_mono, [[{"column": "age", "method": "square"}]]),
        ("combo", compute.exec_transform_combination, [["age", "income"], "product", "age_x_income"]),
        ("filter", compute.exec_filter_rows,
         [[{"logic": "and", "conditions": [{"column": "age", "op": "gt", "value": 25}]}], "and", "keep"]),
        ("drop", compute.exec_drop_columns, [["target"]]),
        ("pca", compute.exec_dim_reduct, ["pca", 2]),
    ]
    for tool, fn, args in steps:
        r = fn(dp.get(v), *args)
        assert "error" not in r, (tool, r)
        v = dp.add_version(r["op"], parent=v, tool=tool, description=tool, ir=r["ir"])
    return dp, v


class TestPipelineCodeExport:
    def test_generated_script_reproduces_version(self, tmp_path):
        """硬证明「可复现」：真跑生成脚本，其产物须逐值等于 data.get(version)。"""
        dp, v = _build_multistep(tmp_path)
        out_csv = tmp_path / "out.csv"
        code = dp.generate_pipeline_code(v, output_path=str(out_csv).replace("\\", "/"))
        script = tmp_path / "pipeline.py"
        script.write_text(code, encoding="utf-8")

        res = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, cwd=str(tmp_path),
        )
        assert res.returncode == 0, res.stderr
        assert_frame_equal(
            pl.read_csv(out_csv), dp.get(v),
            check_dtypes=False, rel_tol=1e-6, abs_tol=1e-8,
        )

    def test_export_version_py_writes_script_and_reports_shape(self, tmp_path):
        dp, v = _build_multistep(tmp_path)
        target = tmp_path / "exports" / "pipeline.py"
        result = export_version(dp, v, target)
        assert result.format == "py"
        assert target.exists()
        text = target.read_text(encoding="utf-8")
        assert text.startswith("import polars as pl")
        assert "lf = lf" in text
        # rows/cols 统一取该版本形状用于登记
        expected = dp.get(v)
        assert result.rows == expected.height and result.cols == expected.width

    def test_export_and_record_py(self, tmp_path):
        dp, v = _build_multistep(tmp_path)
        target = tmp_path / "exports" / "pipeline.py"
        db_path = tmp_path / "data_versions.sqlite"
        export_and_record(
            dp, v, target, None, db_path=db_path, thread_id="t1", description="管道脚本",
        )
        store = DataVersionStore(db_path)
        try:
            rows = store.list_all()
        finally:
            store.close()
        assert len(rows) == 1 and rows[0].format == "py"

    def test_non_reproducible_step_raises(self, tmp_path):
        dp, v = _build_multistep(tmp_path)
        # 追加一个无 IR 的步骤（模拟 eager_op 桥 / 无种子随机）。
        v_bad = dp.add_version(lambda lf: lf, parent=v, tool="mystery", description="x")
        with pytest.raises(ValueError, match="无 IR 记录"):
            dp.generate_pipeline_code(v_bad)

    def test_source_without_ir_raises(self, tmp_path):
        # add_source 未传 ir（如持久化重载未记录来源）→ 无法作为脚本链首。
        dp = DataProvider()
        v = dp.add_source(lambda: pl.DataFrame({"a": [1]}), description="no-ir source")
        with pytest.raises(ValueError, match="无 IR 记录"):
            dp.generate_pipeline_code(v)
