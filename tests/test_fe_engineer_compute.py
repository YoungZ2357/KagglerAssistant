import polars as pl

from kaggler.modes.feature_engineering.compute import exec_empty as execute_empty_value


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
