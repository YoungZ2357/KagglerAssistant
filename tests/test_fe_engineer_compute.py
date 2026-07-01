import numpy as np
import polars as pl
import pytest

from kaggler.modes.feature_engineering.compute import (
    exec_dim_reduct,
    exec_drop_columns as execute_drop_columns,
    exec_empty as execute_empty_value,
    exec_encode as execute_encode,
    exec_filter_rows as execute_filter_rows,
    exec_standardize as execute_standardize,
    standardize_numeric,
    ONE_HOT_CARDINALITY_WARN,
)


class TestExecuteEmptyValue:
    def test_fill_zero_numeric(self):
        df = pl.DataFrame({"a": [1.0, None, 3.0]})
        result = execute_empty_value(df, [{"column": "a", "action": "zero"}])
        assert "error" not in result
        assert result["processed_df"]["a"].to_list() == [1.0, 0.0, 3.0]
        assert result["rows_before"] == 3
        assert result["rows_after"] == 3

    def test_fill_zero_string(self):
        df = pl.DataFrame({"s": ["hello", None, "world"]})
        result = execute_empty_value(df, [{"column": "s", "action": "zero"}])
        assert result["processed_df"]["s"].to_list() == ["hello", "0", "world"]

    def test_fill_zero_boolean(self):
        df = pl.DataFrame({"b": [True, None, False]})
        result = execute_empty_value(df, [{"column": "b", "action": "zero"}])
        assert result["processed_df"]["b"].to_list() == [True, False, False]

    def test_fill_avg(self):
        df = pl.DataFrame({"a": [2.0, None, 4.0]})
        result = execute_empty_value(df, [{"column": "a", "action": "avg"}])
        assert result["processed_df"]["a"].to_list() == [2.0, 3.0, 4.0]

    def test_fill_median(self):
        df = pl.DataFrame({"a": [1.0, None, 100.0]})
        result = execute_empty_value(df, [{"column": "a", "action": "median"}])
        assert result["processed_df"]["a"].null_count() == 0

    def test_fill_mode(self):
        df = pl.DataFrame({"a": ["x", None, "x", "y"]})
        result = execute_empty_value(df, [{"column": "a", "action": "mode"}])
        assert result["processed_df"]["a"].to_list() == ["x", "x", "x", "y"]

    def test_fill_mode_all_nulls_warns(self):
        df = pl.DataFrame({"a": [None, None]})
        result = execute_empty_value(df, [{"column": "a", "action": "mode"}])
        assert "error" not in result
        assert result["processed_df"]["a"].null_count() == 2
        assert any(
            "全部为空值" in w
            for s in result["summary"]
            for w in s.get("warnings", [])
        )

    def test_delete_rows(self):
        df = pl.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        result = execute_empty_value(df, [{"column": "a", "action": "delete"}])
        assert result["processed_df"].height == 2
        assert result["rows_before"] == 3
        assert result["rows_after"] == 2

    def test_mixed_actions(self):
        df = pl.DataFrame({
            "num": [10.0, None, None, 40.0],
            "cat": ["a", None, "b", "b"],
        })
        result = execute_empty_value(df, [
            {"column": "num", "action": "avg"},
            {"column": "cat", "action": "mode"},
        ])
        processed = result["processed_df"]
        assert processed["num"].null_count() == 0
        assert processed["cat"].null_count() == 0
        assert processed["num"].to_list() == [10.0, 25.0, 25.0, 40.0]
        assert processed["cat"].to_list() == ["a", "b", "b", "b"]

    def test_fill_then_delete(self):
        df = pl.DataFrame({
            "num": [10.0, None, None, 40.0],
            "cat": ["a", None, "b", "b"],
        })
        result = execute_empty_value(df, [
            {"column": "num", "action": "avg"},
            {"column": "cat", "action": "delete"},
        ])
        processed = result["processed_df"]
        assert processed["num"].null_count() == 0
        assert processed["cat"].null_count() == 0
        assert processed.height == 3

    def test_unknown_column(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = execute_empty_value(df, [{"column": "zzz", "action": "zero"}])
        assert "error" in result

    def test_incompatible_dtype_avg_on_string(self):
        df = pl.DataFrame({"s": ["a", None]})
        result = execute_empty_value(df, [{"column": "s", "action": "avg"}])
        assert "error" in result

    def test_incompatible_dtype_median_on_string(self):
        df = pl.DataFrame({"s": ["a", None]})
        result = execute_empty_value(df, [{"column": "s", "action": "median"}])
        assert "error" in result

    def test_unknown_action(self):
        df = pl.DataFrame({"a": [1, None]})
        result = execute_empty_value(df, [{"column": "a", "action": "unknown_method"}])
        assert "error" in result

    def test_preview_contains_three_rows(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = execute_empty_value(df, [{"column": "a", "action": "zero"}])
        assert len(result["preview"]) == 3

    def test_preview_less_than_three_rows(self):
        df = pl.DataFrame({"a": [1, 2]})
        result = execute_empty_value(df, [{"column": "a", "action": "zero"}])
        assert len(result["preview"]) == 2

    def test_summary_reports_nulls_filled(self):
        df = pl.DataFrame({"a": [1.0, None, None, 4.0, None]})
        result = execute_empty_value(df, [{"column": "a", "action": "zero"}])
        summary_a = next(s for s in result["summary"] if s["column"] == "a")
        assert summary_a["nulls_before"] == 3
        assert summary_a["nulls_filled"] == 3

    def test_no_nulls_no_fill(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = execute_empty_value(df, [{"column": "a", "action": "zero"}])
        summary_a = next(s for s in result["summary"] if s["column"] == "a")
        assert summary_a["nulls_before"] == 0
        assert summary_a["nulls_filled"] == 0

    def test_preview_serializes_safe(self):
        df = pl.DataFrame({"a": [float("nan"), float("inf"), 1.0]})
        result = execute_empty_value(df, [{"column": "a", "action": "zero"}])
        assert len(result["preview"]) == 3
        assert result["preview"][0]["a"] is None
        assert result["preview"][1]["a"] is None
        assert result["preview"][2]["a"] == 1.0

    def test_processes_in_order_of_pairs(self):
        df = pl.DataFrame({"a": [None, None], "b": [None, None]})
        result = execute_empty_value(df, [
            {"column": "a", "action": "zero"},
            {"column": "b", "action": "zero"},
        ])
        summary = result["summary"]
        assert len(summary) == 2
        assert all("不支持" in w for s in summary for w in s.get("warnings", []))

    def test_fill_zero_unsupported_dtype_skips(self):
        df = pl.DataFrame({"d": [None, None]}, schema={"d": pl.Datetime})
        result = execute_empty_value(df, [{"column": "d", "action": "zero"}])
        assert any(
            w and "不支持" in w
            for s in result["summary"]
            for w in s.get("warnings", [])
        )

    def test_fill_mode_on_numeric(self):
        df = pl.DataFrame({"a": [1.0, None, 1.0, 2.0]})
        result = execute_empty_value(df, [{"column": "a", "action": "mode"}])
        assert result["processed_df"]["a"].null_count() == 0
        assert result["processed_df"]["a"].to_list() == [1.0, 1.0, 1.0, 2.0]

    def test_multiple_delete_columns(self):
        df = pl.DataFrame({
            "a": [1, None, 3, None],
            "b": [4, 5, None, None],
            "c": [7, 8, 9, 10],
        })
        result = execute_empty_value(df, [
            {"column": "a", "action": "delete"},
            {"column": "b", "action": "delete"},
        ])
        assert result["processed_df"].height == 1


class TestExecEncode:
    def test_one_hot_basic(self):
        df = pl.DataFrame({"color": ["red", "blue", "red", "green"]})
        result = execute_encode(df, [{"column": "color", "action": "one_hot"}])
        assert "error" not in result
        processed = result["processed_df"]
        assert "color" not in processed.columns
        assert len(processed.columns) == 2
        assert processed["color_blue"].to_list() == [False, True, False, False]
        assert processed["color_green"].to_list() == [False, False, False, True]
        assert result["rows_after"] == 4

    def test_one_hot_binary_drop_first(self):
        df = pl.DataFrame({"x": ["a", "b", "a", "a"]})
        result = execute_encode(df, [{"column": "x", "action": "one_hot"}])
        processed = result["processed_df"]
        assert "x" not in processed.columns
        assert len(processed.columns) == 1
        assert processed["x_b"].to_list() == [False, True, False, False]

    def test_one_hot_single_value_warns(self):
        df = pl.DataFrame({"x": ["a", "a", "a"]})
        result = execute_encode(df, [{"column": "x", "action": "one_hot"}])
        summary = result["summary"][0]
        assert "仅有一个唯一值" in str(summary["warnings"])
        assert summary["new_columns"] == []
        assert "x" not in result["processed_df"].columns

    def test_one_hot_with_nulls(self):
        df = pl.DataFrame({"x": ["a", "b", None, "a"]})
        result = execute_encode(df, [{"column": "x", "action": "one_hot"}])
        processed = result["processed_df"]
        assert "x" not in processed.columns
        assert processed.row(2, named=True) == {"x_b": None}

    def test_one_hot_all_nulls(self):
        df = pl.DataFrame({"x": [None, None]})
        result = execute_encode(df, [{"column": "x", "action": "one_hot"}])
        summary = result["summary"][0]
        assert "全部为空值" in str(summary["warnings"])
        assert summary["new_columns"] == []
        assert "x" not in result["processed_df"].columns

    def test_one_hot_high_cardinality_warns(self):
        vals = [str(i) for i in range(ONE_HOT_CARDINALITY_WARN + 1)]
        df = pl.DataFrame({"x": vals})
        result = execute_encode(df, [{"column": "x", "action": "one_hot"}])
        assert any("稀疏" in w for w in result["summary"][0]["warnings"])
        assert "error" not in result

    def test_label_basic(self):
        df = pl.DataFrame({"size": ["medium", "small", "large", "small"]})
        result = execute_encode(df, [{"column": "size", "action": "label"}])
        processed = result["processed_df"]
        assert processed["size"].dtype == pl.Int64
        assert processed["size"].to_list() == [1, 2, 0, 2]
        assert result["summary"][0]["mapping"] == {"large": 0, "medium": 1, "small": 2}

    def test_label_with_nulls(self):
        df = pl.DataFrame({"x": ["b", None, "a", "b"]})
        result = execute_encode(df, [{"column": "x", "action": "label"}])
        processed = result["processed_df"]
        assert processed["x"].to_list() == [1, None, 0, 1]
        assert result["summary"][0]["mapping"] == {"a": 0, "b": 1}

    def test_label_numeric_column(self):
        df = pl.DataFrame({"x": [30, 10, 20, 10]})
        result = execute_encode(df, [{"column": "x", "action": "label"}])
        processed = result["processed_df"]
        assert processed["x"].dtype == pl.Int64
        assert processed["x"].to_list() == [2, 0, 1, 0]
        assert result["summary"][0]["mapping"] == {"10": 0, "20": 1, "30": 2}

    def test_label_boolean_column(self):
        df = pl.DataFrame({"x": [True, False, True, False]})
        result = execute_encode(df, [{"column": "x", "action": "label"}])
        processed = result["processed_df"]
        assert processed["x"].dtype == pl.Int64
        assert processed["x"].to_list() == [1, 0, 1, 0]
        assert result["summary"][0]["mapping"] == {"False": 0, "True": 1}

    def test_label_all_nulls(self):
        df = pl.DataFrame({"x": [None, None]})
        result = execute_encode(df, [{"column": "x", "action": "label"}])
        assert "error" not in result
        assert "全部为空值" in str(result["summary"][0]["warnings"])
        assert result["processed_df"]["x"].null_count() == 2

    def test_mixed_encodings(self):
        df = pl.DataFrame({
            "color": ["red", "blue", "red"],
            "size": ["L", "M", "S"],
        })
        result = execute_encode(df, [
            {"column": "color", "action": "one_hot"},
            {"column": "size", "action": "label"},
        ])
        processed = result["processed_df"]
        assert "color" not in processed.columns
        assert "size" in processed.columns
        assert processed["size"].dtype == pl.Int64
        assert result["rows_after"] == 3

    def test_unknown_column(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = execute_encode(df, [{"column": "zzz", "action": "one_hot"}])
        assert "error" in result

    def test_unknown_action(self):
        df = pl.DataFrame({"a": [1, 2]})
        result = execute_encode(df, [{"column": "a", "action": "unknown"}])
        assert "error" in result

    def test_preview_contains_three_rows(self):
        df = pl.DataFrame({"x": ["a", "b", "c", "d", "e"]})
        result = execute_encode(df, [{"column": "x", "action": "one_hot"}])
        assert len(result["preview"]) == 3

    def test_preview_serializes_safe(self):
        df = pl.DataFrame({"x": ["a", "b", "c"]})
        result = execute_encode(df, [{"column": "x", "action": "one_hot"}])
        for row in result["preview"]:
            for v in row.values():
                assert isinstance(v, (str, int, type(None)))

    def test_rows_unchanged(self):
        df = pl.DataFrame({"x": ["a", "b", "c", "a"]})
        result = execute_encode(df, [{"column": "x", "action": "label"}])
        assert result["rows_before"] == 4
        assert result["rows_after"] == 4


class TestStandardizeNumeric:
    def test_basic_single_column(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = standardize_numeric(df, ["a"])
        assert result["a"].to_list() == pytest.approx(
            [-1.414214, -0.707107, 0.0, 0.707107, 1.414214], rel=1e-5
        )
        assert "a" in result.columns

    def test_multiple_columns(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        result = standardize_numeric(df, ["a", "b"])
        vals_a = result["a"].to_list()
        vals_b = result["b"].to_list()
        assert vals_a == pytest.approx(
            [-1.224745, 0.0, 1.224745], rel=1e-5
        )
        assert vals_b == pytest.approx(
            [-1.224745, 0.0, 1.224745], rel=1e-5
        )
        assert result.height == 3

    def test_preserves_non_standardized_columns(self):
        df = pl.DataFrame(
            {"x": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"], "y": [4.0, 5.0, 6.0]}
        )
        result = standardize_numeric(df, ["x"])
        assert result["cat"].to_list() == ["a", "b", "c"]
        assert result["y"].to_list() == [4.0, 5.0, 6.0]
        assert result.columns == ["x", "cat", "y"]

    def test_constant_column(self):
        """标准化常数列不会报错（sklearn 内部处理，结果归零）"""
        df = pl.DataFrame({"a": [5.0, 5.0, 5.0]})
        result = standardize_numeric(df, ["a"])
        assert result["a"].to_list() == pytest.approx([0.0, 0.0, 0.0])

    def test_empty_columns_list(self):
        df = pl.DataFrame({"a": [1.0, 2.0]})
        result = standardize_numeric(df, [])
        assert result.equals(df)


class TestExecStandardize:
    def test_success_basic(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
        result = execute_standardize(df, ["x", "y"])
        assert "error" not in result
        assert result["rows_before"] == 3
        assert result["rows_after"] == 3
        assert len(result["preview"]) == 3
        assert result["summary"][0]["columns"] == ["x", "y"]

    def test_unknown_column(self):
        df = pl.DataFrame({"a": [1.0, 2.0]})
        result = execute_standardize(df, ["b"])
        assert "error" in result
        assert "列名不存在" in result["error"]

    def test_non_numeric_column(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
        result = execute_standardize(df, ["a", "b"])
        assert "error" in result
        assert "不是数值类型" in result["error"]

    def test_null_in_column(self):
        df = pl.DataFrame({"x": [1.0, None, 3.0]})
        result = execute_standardize(df, ["x"])
        assert "error" in result
        assert "空值" in result["error"]

    def test_single_column(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = execute_standardize(df, ["x"])
        assert "error" not in result
        mean_val = np.mean(result["processed_df"]["x"].to_list())
        assert mean_val == pytest.approx(0.0, abs=1e-6)

    def test_multiple_errors_reports_all(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
        result = execute_standardize(df, ["b", "a"])
        assert "error" in result
        assert "b" in result["error"]


class TestExecDropColumns:
    def test_success_basic(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        result = execute_drop_columns(df, ["b"])
        assert "error" not in result
        processed = result["processed_df"]
        assert "b" not in processed.columns
        assert processed.columns == ["a", "c"]
        assert result["summary"][0]["remaining_columns"] == ["a", "c"]
        assert result["rows_before"] == 3
        assert result["rows_after"] == 3

    def test_unknown_column(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = execute_drop_columns(df, ["zzz"])
        assert "error" in result
        assert "列名不存在" in result["error"]

    def test_empty_columns_list(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = execute_drop_columns(df, [])
        assert "error" in result

    def test_duplicate_columns_in_list(self):
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = execute_drop_columns(df, ["a", "a"])
        assert "error" not in result
        assert result["processed_df"].columns == ["b"]
        assert result["summary"][0]["dropped_columns"] == ["a"]

    def test_drop_all_columns(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = execute_drop_columns(df, ["a", "b"])
        assert "error" not in result
        assert result["processed_df"].width == 0
        assert result["rows_before"] == 3
        assert result["rows_after"] == 0
        assert "全部列" in str(result["summary"][0]["warnings"])

    def test_preview_contains_three_rows(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})
        result = execute_drop_columns(df, ["b"])
        assert len(result["preview"]) == 3

    def test_preview_less_than_three_rows(self):
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = execute_drop_columns(df, ["b"])
        assert len(result["preview"]) == 2

    def test_summary_reports_remaining_columns(self):
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = execute_drop_columns(df, ["a", "c"])
        summary = result["summary"][0]
        assert summary["dropped_columns"] == ["a", "c"]
        assert summary["remaining_columns"] == ["b"]
        assert summary["warnings"] == []


def _cond(column, op, value):
    return {"column": column, "op": op, "value": value}


def _group(logic, conditions):
    return {"logic": logic, "conditions": conditions}


class TestExecFilterRows:
    def test_keep_basic(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = execute_filter_rows(
            df,
            groups=[_group("and", [_cond("a", "gt", 2)])],
            group_logic="and",
            action="keep",
        )
        assert "error" not in result
        assert result["processed_df"]["a"].to_list() == [3, 4, 5]
        assert result["rows_before"] == 5
        assert result["rows_after"] == 3

    def test_delete_basic(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = execute_filter_rows(
            df,
            groups=[_group("and", [_cond("a", "gt", 2)])],
            group_logic="and",
            action="delete",
        )
        assert "error" not in result
        assert result["processed_df"]["a"].to_list() == [1, 2]
        assert result["rows_after"] == 2

    def test_group_inner_and(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = execute_filter_rows(
            df,
            groups=[_group("and", [_cond("a", "gt", 1), _cond("a", "lt", 5)])],
            group_logic="and",
            action="keep",
        )
        assert result["processed_df"]["a"].to_list() == [2, 3, 4]

    def test_group_inner_or(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = execute_filter_rows(
            df,
            groups=[_group("or", [_cond("a", "lt", 2), _cond("a", "gt", 4)])],
            group_logic="and",
            action="keep",
        )
        assert result["processed_df"]["a"].to_list() == [1, 5]

    def test_top_level_and_across_groups(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "x", "y", "y", "y"]})
        result = execute_filter_rows(
            df,
            groups=[
                _group("and", [_cond("a", "gt", 1)]),
                _group("and", [_cond("b", "eq", "y")]),
            ],
            group_logic="and",
            action="keep",
        )
        assert result["processed_df"]["a"].to_list() == [3, 4, 5]

    def test_top_level_or_across_groups(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "x", "y", "y", "y"]})
        result = execute_filter_rows(
            df,
            groups=[
                _group("and", [_cond("a", "lt", 2)]),
                _group("and", [_cond("b", "eq", "y")]),
            ],
            group_logic="or",
            action="keep",
        )
        assert result["processed_df"]["a"].to_list() == [1, 3, 4, 5]

    def test_unknown_column(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = execute_filter_rows(
            df,
            groups=[_group("and", [_cond("zzz", "gt", 1)])],
            group_logic="and",
            action="keep",
        )
        assert "error" in result
        assert "列名不存在" in result["error"]

    def test_multiple_dtype_errors_reported_together(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = execute_filter_rows(
            df,
            groups=[_group("and", [_cond("a", "gt", "notanumber"), _cond("b", "eq", 1)])],
            group_logic="and",
            action="keep",
        )
        assert "error" in result
        assert len(result["details"]) == 2

    def test_bool_rejected_for_numeric_column(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = execute_filter_rows(
            df,
            groups=[_group("and", [_cond("a", "gt", True)])],
            group_logic="and",
            action="keep",
        )
        assert "error" in result

    def test_empty_groups(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = execute_filter_rows(df, groups=[], group_logic="and", action="keep")
        assert "error" in result

    def test_group_with_empty_conditions(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = execute_filter_rows(
            df,
            groups=[_group("and", [])],
            group_logic="and",
            action="keep",
        )
        assert "error" in result

    def test_null_handling_keep_excludes_null_row(self):
        df = pl.DataFrame({"a": [1.0, None, 3.0]})
        result = execute_filter_rows(
            df,
            groups=[_group("and", [_cond("a", "gt", 0)])],
            group_logic="and",
            action="keep",
        )
        assert result["processed_df"]["a"].to_list() == [1.0, 3.0]

    def test_null_handling_delete_retains_null_row(self):
        df = pl.DataFrame({"a": [1.0, None, 3.0]})
        result = execute_filter_rows(
            df,
            groups=[_group("and", [_cond("a", "gt", 2)])],
            group_logic="and",
            action="delete",
        )
        assert result["processed_df"]["a"].to_list() == [1.0, None]

    def test_preview_contains_three_rows(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        result = execute_filter_rows(
            df,
            groups=[_group("and", [_cond("a", "ge", 1)])],
            group_logic="and",
            action="keep",
        )
        assert len(result["preview"]) == 3

    def test_summary_content(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = execute_filter_rows(
            df,
            groups=[_group("and", [_cond("a", "gt", 1)])],
            group_logic="and",
            action="delete",
        )
        summary = result["summary"][0]
        assert summary["action"] == "delete"
        assert summary["rows_kept"] == result["rows_after"]
        assert summary["rows_removed"] == result["rows_before"] - result["rows_after"]
        assert "a > 1" in summary["condition_description"]


class TestExecDimReduct:
    # --- PCA tests ---
    def test_pca_basic(self):
        df = pl.DataFrame(
            {"f1": [1.0, 2.0, 3.0, 4.0, 5.0], "f2": [5.0, 4.0, 3.0, 2.0, 1.0]}
        )
        result = exec_dim_reduct(df, method="pca", n_components=2)
        assert "error" not in result
        assert "PC1" in result["processed_df"].columns
        assert "PC2" in result["processed_df"].columns
        assert result["rows_after"] == 5
        assert result["summary"][0]["n_components"] == 2
        assert len(result["summary"][0]["explained_variance_ratio"]) == 2

    def test_pca_with_non_numeric_columns(self):
        df = pl.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "f2": [5.0, 4.0, 3.0, 2.0, 1.0],
                "label": ["a", "b", "a", "b", "a"],
            }
        )
        result = exec_dim_reduct(df, method="pca", n_components=2)
        assert "label" in result["processed_df"].columns
        assert "PC1" in result["processed_df"].columns
        assert "f1" not in result["processed_df"].columns

    def test_pca_no_standardize(self):
        df = pl.DataFrame(
            {"f1": [1.0, 2.0, 3.0, 4.0, 5.0], "f2": [5.0, 4.0, 3.0, 2.0, 1.0]}
        )
        result = exec_dim_reduct(df, method="pca", n_components=2, standardize=False)
        assert "error" not in result
        assert result["summary"][0]["standardized"] is False

    def test_pca_single_component(self):
        df = pl.DataFrame(
            {"f1": [1.0, 2.0, 3.0, 4.0, 5.0], "f2": [2.0, 3.0, 4.0, 5.0, 6.0]}
        )
        result = exec_dim_reduct(df, method="pca", n_components=1)
        assert "PC1" in result["processed_df"].columns
        assert "PC2" not in result["processed_df"].columns

    def test_pca_no_numeric_columns(self):
        df = pl.DataFrame({"a": ["x", "y", "z"]})
        result = exec_dim_reduct(df, method="pca", n_components=1)
        assert "error" in result
        assert "没有数值列" in result["error"]

    def test_pca_n_components_too_large(self):
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = exec_dim_reduct(df, method="pca", n_components=5)
        assert "error" in result
        assert "超过数值列数" in result["error"]

    def test_pca_n_components_zero_or_negative(self):
        df = pl.DataFrame({"x": [1.0, 2.0]})
        result = exec_dim_reduct(df, method="pca", n_components=0)
        assert "error" in result
        assert "必须为正整数" in result["error"]

    def test_pca_nan_in_data(self):
        df = pl.DataFrame({"x": [1.0, None, 3.0], "y": [4.0, 5.0, 6.0]})
        result = exec_dim_reduct(df, method="pca", n_components=1)
        assert "error" in result
        assert "NaN" in result["error"]

    def test_pca_preview_structure(self):
        df = pl.DataFrame(
            {"f1": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "c", "d"]}
        )
        result = exec_dim_reduct(df, method="pca", n_components=1)
        assert len(result["preview"]) == 3
        assert "cat" in result["preview"][0]
        assert "PC1" in result["preview"][0]

    # --- LDA tests ---
    def test_lda_basic(self):
        df = pl.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 6.0, 7.0, 8.0],
                "f2": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                "f3": [2.0, 3.0, 1.0, 8.0, 9.0, 7.0],
                "label": ["a", "a", "a", "b", "b", "b"],
            }
        )
        result = exec_dim_reduct(df, method="lda", n_components=1, target="label")
        assert "error" not in result
        assert "LD1" in result["processed_df"].columns
        assert "label" in result["processed_df"].columns
        assert "f1" not in result["processed_df"].columns
        assert result["summary"][0]["method"] == "lda"
        assert result["summary"][0]["n_classes"] == 2

    def test_lda_no_standardize(self):
        df = pl.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 6.0, 7.0, 8.0],
                "f2": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                "f3": [2.0, 3.0, 1.0, 8.0, 9.0, 7.0],
                "target": ["a", "a", "a", "b", "b", "b"],
            }
        )
        result = exec_dim_reduct(
            df, method="lda", n_components=1, target="target", standardize=False
        )
        assert "error" not in result
        assert result["summary"][0]["standardized"] is False

    def test_lda_numeric_target_column(self):
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
                "y": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "cls": [0, 0, 0, 1, 1, 1],
            }
        )
        result = exec_dim_reduct(df, method="lda", n_components=1, target="cls")
        assert "error" not in result
        assert "cls" in result["processed_df"].columns
        assert "x" not in result["processed_df"].columns

    def test_lda_multi_class(self):
        df = pl.DataFrame(
            {
                "f1": [1.0, 2.0, 5.0, 6.0, 9.0, 10.0, 1.0, 2.0, 5.0],
                "f2": [9.0, 8.0, 5.0, 4.0, 1.0, 0.0, 8.5, 7.5, 4.5],
                "label": ["a", "a", "b", "b", "c", "c", "a", "b", "c"],
            }
        )
        result = exec_dim_reduct(df, method="lda", n_components=2, target="label")
        assert "error" not in result
        assert result["summary"][0]["n_classes"] == 3
        assert "LD1" in result["processed_df"].columns
        assert len(result["summary"][0]["new_columns"]) >= 1

    def test_lda_keeps_non_numeric_except_target(self):
        df = pl.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 6.0, 7.0, 8.0],
                "f2": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                "cat": ["x", "y", "z", "x", "y", "z"],
                "label": ["a", "a", "a", "b", "b", "b"],
            }
        )
        result = exec_dim_reduct(df, method="lda", n_components=1, target="label")
        assert "label" in result["processed_df"].columns
        assert "cat" in result["processed_df"].columns
        assert "f1" not in result["processed_df"].columns

    def test_lda_missing_target(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        result = exec_dim_reduct(df, method="lda", n_components=1)
        assert "error" in result
        assert "需要指定 target" in result["error"]

    def test_lda_target_not_exist(self):
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = exec_dim_reduct(df, method="lda", n_components=1, target="z")
        assert "error" in result
        assert "不存在" in result["error"]

    def test_lda_single_class(self):
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "label": ["a", "a", "a"],
            }
        )
        result = exec_dim_reduct(df, method="lda", n_components=1, target="label")
        assert "error" in result
        assert "至少 2 个类别" in result["error"]

    def test_lda_n_components_too_large(self):
        df = pl.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0],
                "f2": [5.0, 6.0, 7.0, 8.0],
                "label": ["a", "a", "b", "b"],
            }
        )
        result = exec_dim_reduct(df, method="lda", n_components=5, target="label")
        assert "error" in result
        assert "超过最大允许值" in result["error"]

    def test_lda_no_numeric_features(self):
        df = pl.DataFrame(
            {"label": ["a", "b", "c"], "cat": ["x", "y", "z"]}
        )
        result = exec_dim_reduct(df, method="lda", n_components=1, target="label")
        assert "error" in result
        assert "没有数值特征列" in result["error"]

    def test_lda_nan_in_features(self):
        df = pl.DataFrame(
            {
                "x": [1.0, None, 3.0, 4.0],
                "label": ["a", "b", "a", "b"],
            }
        )
        result = exec_dim_reduct(df, method="lda", n_components=1, target="label")
        assert "error" in result
        assert "NaN" in result["error"]

    def test_lda_target_with_nulls(self):
        df = pl.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "f2": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                "label": ["a", "b", None, "a", "b", "a"],
            }
        )
        result = exec_dim_reduct(df, method="lda", n_components=1, target="label")
        assert "error" not in result
        assert result["rows_before"] == 6
        assert result["rows_after"] == 5
        assert result["summary"][0]["rows_dropped"] == 1

    def test_lda_preview_structure(self):
        df = pl.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 6.0, 7.0, 8.0],
                "f2": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                "cat": ["x", "y", "z", "x", "y", "z"],
                "label": ["a", "a", "a", "b", "b", "b"],
            }
        )
        result = exec_dim_reduct(df, method="lda", n_components=1, target="label")
        assert len(result["preview"]) == 3
        preview = result["preview"][0]
        assert "cat" in preview
        assert "label" in preview
        assert "LD1" in preview

    # --- General error tests ---
    def test_unknown_method(self):
        df = pl.DataFrame({"x": [1.0, 2.0]})
        result = exec_dim_reduct(df, method="tsne", n_components=1)
        assert "error" in result
        assert "未知的降维方法" in result["error"]

    def test_n_components_negative(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = exec_dim_reduct(df, method="pca", n_components=-1)
        assert "error" in result
        assert "必须为正整数" in result["error"]
