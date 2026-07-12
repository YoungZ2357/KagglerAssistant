"""工具回传体积上限的测试：验证各 compute 层在宽表 / 高基数 / 长文本 / 大目录
场景下正确截断，且截断只影响回传摘要、不影响实际数据。"""

import polars as pl

from kaggler.modes.common.compute import list_files
from kaggler.modes.eda.compute import (
    get_correlation,
    get_descriptive_statistics,
    get_schema_report,
)
from kaggler.modes.feature_engineering.compute import exec_drop_columns, exec_encode
from kaggler.shared.limits import (
    MAX_CORRELATION_PAIRS,
    MAX_MAPPING_ENTRIES,
    MAX_SCHEMA_COLUMNS,
    MAX_STR_VALUE_LEN,
    MAX_WORKSPACE_ENTRIES,
    cap_list,
    truncate_str,
)
from kaggler.shared.serialization import safe_val


# --- 纯 helper 边界 ---------------------------------------------------------
class TestTruncateStr:
    def test_short_string_unchanged(self):
        assert truncate_str("abc") == "abc"

    def test_exact_limit_unchanged(self):
        s = "x" * MAX_STR_VALUE_LEN
        assert truncate_str(s) == s

    def test_over_limit_truncated_with_marker(self):
        s = "x" * (MAX_STR_VALUE_LEN + 50)
        out = truncate_str(s)
        assert out.startswith("x" * MAX_STR_VALUE_LEN)
        assert "+50 chars" in out


class TestCapList:
    def test_under_limit_returns_none_info(self):
        items = [1, 2, 3]
        capped, info = cap_list(items, 5)
        assert capped == items
        assert info is None

    def test_exact_limit_not_truncated(self):
        items = list(range(5))
        capped, info = cap_list(items, 5)
        assert capped == items
        assert info is None

    def test_over_limit_truncated_with_counts(self):
        items = list(range(10))
        capped, info = cap_list(items, 4)
        assert capped == [0, 1, 2, 3]
        assert info == {"truncated": True, "shown": 4, "total": 10}


# --- safe_val 字符串截断 ----------------------------------------------------
class TestSafeValTruncation:
    def test_long_string_truncated(self):
        s = "y" * (MAX_STR_VALUE_LEN + 10)
        assert safe_val(s) == truncate_str(s)

    def test_numeric_and_none_unaffected(self):
        assert safe_val(3.1415926535) == round(3.1415926535, 6)
        assert safe_val(None) is None


# --- explore_schema 宽表截列 + 长文本样本截断 -------------------------------
class TestSchemaReport:
    def test_wide_df_truncates_columns(self):
        n = MAX_SCHEMA_COLUMNS + 20
        df = pl.DataFrame({f"c{i}": [i] for i in range(n)})
        report = get_schema_report(df)
        assert report["total_columns"] == n
        assert report["columns_truncated"] is True
        assert report["columns_shown"] == MAX_SCHEMA_COLUMNS
        assert len(report["columns"]) == MAX_SCHEMA_COLUMNS

    def test_narrow_df_not_truncated(self):
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        report = get_schema_report(df)
        assert "columns_truncated" not in report
        assert len(report["columns"]) == 2

    def test_long_sample_value_truncated(self):
        long_text = "z" * (MAX_STR_VALUE_LEN + 100)
        df = pl.DataFrame({"text": [long_text, "short", "s"]})
        report = get_schema_report(df)
        sample = report["columns"][0]["sample_values"][0]
        assert len(sample) < len(long_text)
        assert "chars" in sample


# --- correlation pair 截断 --------------------------------------------------
class TestCorrelationCap:
    def test_many_numeric_columns_capped(self):
        # 列数 n → pearson pair 数 = n*(n-1)/2；取 n 使 pair 数超过上限。
        n = 20  # C(20,2) = 190 > 100
        df = pl.DataFrame({f"n{i}": [float(i), float(i + 1), float(i + 2)] for i in range(n)})
        result = get_correlation(df, list(df.columns))
        assert len(result["results"]["pearson"]) == MAX_CORRELATION_PAIRS
        assert result["truncated"]["pearson"]["total"] == n * (n - 1) // 2

    def test_few_columns_not_capped(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
        result = get_correlation(df, ["a", "b"])
        assert "truncated" not in result


# --- descriptive_analysis 列数截断 -----------------------------------------
class TestDescribeCap:
    def test_many_columns_capped(self):
        from kaggler.shared.limits import MAX_DESCRIBE_COLUMNS

        n = MAX_DESCRIBE_COLUMNS + 5
        df = pl.DataFrame({f"n{i}": [float(i), float(i + 1)] for i in range(n)})
        result = get_descriptive_statistics(df, list(df.columns))
        assert len(result["stats"]) == MAX_DESCRIBE_COLUMNS
        assert result["columns_truncated"]["total"] == n


# --- label 编码高基数：映射截断但数据无损 -----------------------------------
class TestEncodeMappingCap:
    def test_high_cardinality_label_mapping_truncated_but_data_intact(self):
        n = MAX_MAPPING_ENTRIES + 30
        # 每个取值唯一 → n 个类别
        df = pl.DataFrame({"hi_card": [f"v{i:04d}" for i in range(n)]})
        result = exec_encode(df, [{"column": "hi_card", "action": "label"}])

        col_summary = result["summary"][0]
        assert col_summary["n_categories"] == n
        assert len(col_summary["mapping"]) == MAX_MAPPING_ENTRIES
        assert col_summary["mapping_truncated"]["total"] == n

        # 关键回归：完整映射仍被应用，编码后的列无空值、覆盖 0..n-1。
        encoded = result["op"](df.lazy()).collect()
        vals = encoded["hi_card"].to_list()
        assert encoded["hi_card"].null_count() == 0
        assert set(vals) == set(range(n))


# --- drop_columns remaining_columns 截断 ------------------------------------
class TestDropColumnsCap:
    def test_remaining_columns_truncated(self):
        from kaggler.shared.limits import MAX_COLUMN_LIST

        n = MAX_COLUMN_LIST + 30
        df = pl.DataFrame({f"c{i}": [i] for i in range(n)})
        result = exec_drop_columns(df, ["c0"])
        s = result["summary"][0]
        assert s["n_remaining_columns"] == n - 1
        assert len(s["remaining_columns"]) == MAX_COLUMN_LIST
        assert s["remaining_columns_truncated"]["total"] == n - 1


# --- list_workspace_files 目录条目截断 -------------------------------------
class TestListFilesCap:
    def test_large_directory_truncated(self, tmp_path):
        n = MAX_WORKSPACE_ENTRIES + 15
        for i in range(n):
            (tmp_path / f"f{i:05d}.txt").write_text("x")
        out = list_files(tmp_path)
        lines = out.splitlines()
        # MAX 个条目 + 1 行「未显示」提示
        assert len(lines) == MAX_WORKSPACE_ENTRIES + 1
        assert "未显示" in lines[-1]
        assert str(n) in lines[-1]

    def test_small_directory_not_truncated(self, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.txt").write_text("y")
        out = list_files(tmp_path)
        assert "未显示" not in out
