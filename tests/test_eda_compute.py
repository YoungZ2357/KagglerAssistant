
import polars as pl
import pytest

from kaggler.modes.eda.compute import (
    _safe_val,
    _pearson_batch,
    _cramers_v,
    _eta_squared,
    _box_data_raw,
    BinnedColumn,
    get_correlation,
    get_schema_report,
    get_descriptive_statistics,
    get_boxed_data,
    distribution_evaluation,
)


class TestSafeVal:
    def test_none(self):
        assert _safe_val(None) is None

    def test_nan(self):
        assert _safe_val(float("nan")) is None

    def test_inf(self):
        assert _safe_val(float("inf")) is None
        assert _safe_val(float("-inf")) is None

    def test_float_normal(self):
        assert _safe_val(3.1415926535) == 3.141593
        assert _safe_val(1.0) == 1.0

    def test_int(self):
        assert _safe_val(42) == 42
        assert isinstance(_safe_val(42), int)

    def test_bool(self):
        assert _safe_val(True) == "True"
        assert _safe_val(False) == "False"

    def test_str(self):
        assert _safe_val("hello") == "hello"


class TestPearsonBatch:
    def test_empty_pairs(self):
        result = _pearson_batch(pl.DataFrame({"x": [1]}), [])
        assert result == []

    def test_perfect_negative(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
        result = _pearson_batch(df, [("a", "b")])
        assert result[0] == pytest.approx(-1.0)

    def test_perfect_positive(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
        result = _pearson_batch(df, [("a", "b")])
        assert result[0] == pytest.approx(1.0)

    def test_multiple_pairs(self):
        df = pl.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0], "c": [1.0, 2.0, 3.0]}
        )
        result = _pearson_batch(df, [("a", "b"), ("a", "c")])
        assert len(result) == 2
        assert result[0] == pytest.approx(-1.0)
        assert result[1] == pytest.approx(1.0)


class TestCramersV:
    def test_strong_association(self):
        df = pl.DataFrame(
            {
                "col_a": ["A", "A", "A", "A", "B"],
                "col_b": ["X", "X", "X", "X", "Y"],
            }
        )
        result = _cramers_v(df, "col_a", "col_b")
        assert result is not None
        assert result == pytest.approx(0.824621, abs=1e-4)

    def test_independence(self):
        df = pl.DataFrame(
            {
                "col_a": ["A", "A", "B", "B"],
                "col_b": ["X", "Y", "X", "Y"],
            }
        )
        result = _cramers_v(df, "col_a", "col_b")
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_all_nulls_drop(self):
        df = pl.DataFrame(
            {
                "col_a": [None, None],
                "col_b": ["X", "Y"],
            }
        )
        assert _cramers_v(df, "col_a", "col_b") is None

    def test_single_unique_min_dim_zero(self):
        df = pl.DataFrame(
            {
                "col_a": ["A", "A", "A"],
                "col_b": ["X", "Y", "Z"],
            }
        )
        assert _cramers_v(df, "col_a", "col_b") is None


class TestEtaSquared:
    def test_strong_group_difference(self):
        df = pl.DataFrame(
            {
                "cat": ["A", "A", "B", "B", "C", "C"],
                "num": [1.0, 2.0, 10.0, 11.0, 20.0, 21.0],
            }
        )
        result = _eta_squared(df, "cat", "num")
        assert result is not None
        assert result > 0.95

    def test_no_group_difference(self):
        df = pl.DataFrame(
            {
                "cat": ["A", "A", "B", "B"],
                "num": [5.0, 5.0, 5.0, 5.0],
            }
        )
        result = _eta_squared(df, "cat", "num")
        assert result is None

    def test_all_nulls(self):
        df = pl.DataFrame(
            {
                "cat": [None, None],
                "num": [1.0, 2.0],
            }
        )
        assert _eta_squared(df, "cat", "num") is None

    def test_single_row(self):
        df = pl.DataFrame(
            {
                "cat": ["A"],
                "num": [5.0],
            }
        )
        assert _eta_squared(df, "cat", "num") is None


class TestGetCorrelation:
    def test_less_than_two_columns(self, df_mixed):
        result = get_correlation(df_mixed, ["age"])
        assert "error" in result
        assert "至少需要 2 列" in result["error"]

    def test_unknown_columns(self, df_mixed):
        result = get_correlation(df_mixed, ["age", "nonexistent"])
        assert "error" in result
        assert "不存在" in result["error"]

    def test_pure_numeric(self, df_numeric_only):
        result = get_correlation(df_numeric_only, ["x", "y"])
        assert "error" not in result
        assert "pearson" in result["results"]
        assert "cramers_v" not in result["results"]
        assert "eta_squared" not in result["results"]
        assert result["column_types"] == {"x": "numeric", "y": "numeric"}

    def test_pure_categorical(self, df_categorical_only):
        result = get_correlation(df_categorical_only, ["cat_a", "cat_b"])
        assert "error" not in result
        assert "cramers_v" in result["results"]
        assert "pearson" not in result["results"]
        assert "eta_squared" not in result["results"]

    def test_mixed_types(self, df_mixed):
        result = get_correlation(df_mixed, ["age", "score", "city"])
        assert "error" not in result
        assert "pearson" in result["results"]
        assert "eta_squared" in result["results"]
        assert result["column_types"] == {
            "age": "numeric",
            "score": "numeric",
            "city": "categorical",
        }

    def test_single_numeric_single_categorical(self, df_mixed):
        result = get_correlation(df_mixed, ["age", "city"])
        assert "error" not in result
        assert "eta_squared" in result["results"]
        assert "pearson" not in result["results"]
        assert "cramers_v" not in result["results"]

    def test_sorted_by_abs_value(self, df_numeric_only):
        df = pl.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [-1.0, -2.0, -3.0], "c": [1.0, 2.0, 3.0]}
        )
        result = get_correlation(df, ["a", "b", "c"])
        pearson_results = result["results"]["pearson"]
        assert abs(pearson_results[0]["value"]) >= abs(pearson_results[1]["value"])

    def test_output_structure(self, df_numeric_only):
        result = get_correlation(df_numeric_only, ["x", "y"])
        assert "column_types" in result
        assert "method_descriptions" in result
        assert "results" in result
        assert "pearson" in result["method_descriptions"]
        assert "cramers_v" in result["method_descriptions"]
        assert "eta_squared" in result["method_descriptions"]


class TestSchemaReport:
    def test_output_structure(self, df_mixed):
        report = get_schema_report(df_mixed)
        assert report["total_rows"] == 5
        assert report["total_columns"] == 5
        assert len(report["columns"]) == 5

    def test_column_info(self, df_mixed):
        report = get_schema_report(df_mixed)
        city_col = next(c for c in report["columns"] if c["name"] == "city")
        assert city_col["null_count"] == 0
        assert city_col["null_rate"] == 0.0
        assert city_col["n_unique"] == 3
        assert len(city_col["sample_values"]) == 3

    def test_with_nulls(self, df_with_nulls):
        report = get_schema_report(df_with_nulls)
        a_col = next(c for c in report["columns"] if c["name"] == "a")
        assert a_col["null_count"] == 1
        assert a_col["null_rate"] == 0.2
        assert a_col["n_unique"] == 5

    def test_empty_df(self, df_empty):
        report = get_schema_report(df_empty)
        assert report["total_rows"] == 0
        assert report["total_columns"] == 2


class TestDescriptiveStatistics:
    def test_empty_columns(self, df_mixed):
        result = get_descriptive_statistics(df_mixed, [])
        assert "error" in result

    def test_normal_numeric(self, df_mixed):
        result = get_descriptive_statistics(df_mixed, ["age"])
        assert "error" not in result
        stats = result["stats"][0]
        assert stats["column"] == "age"
        assert stats["count"] == 5
        assert stats["null_count"] == 0
        assert stats["mean"] == 35.0
        assert stats["median"] == 35.0
        assert stats["min"] == 25
        assert stats["max"] == 45

    def test_skips_non_numeric(self, df_mixed):
        result = get_descriptive_statistics(df_mixed, ["age", "city"])
        assert "error" not in result
        assert "skipped_non_numeric" in result
        assert "city" in result["skipped_non_numeric"]
        assert len(result["stats"]) == 1

    def test_all_non_numeric(self, df_categorical_only):
        result = get_descriptive_statistics(df_categorical_only, ["cat_a", "cat_b"])
        assert "error" in result
        assert "non_numeric_columns" in result

    def test_with_nulls(self, df_with_nulls):
        result = get_descriptive_statistics(df_with_nulls, ["a"])
        stats = result["stats"][0]
        assert stats["column"] == "a"
        assert stats["count"] == 4
        assert stats["null_count"] == 1


class TestBoxedData:
    def test_numeric_binning(self, df_mixed):
        result = get_boxed_data(df_mixed, "age")
        assert result["column"] == "age"
        assert result["dtype"] == "numeric"
        assert result["total"] == 5
        assert result["null_count"] == 0
        assert "bins" in result
        assert len(result["bins"]) > 0

    def test_custom_bins(self, df_mixed):
        result = get_boxed_data(df_mixed, "age", bins=3)
        assert len(result["bins"]) <= 3

    def test_categorical_frequency(self, df_mixed):
        result = get_boxed_data(df_mixed, "city")
        assert result["dtype"] == "categorical"
        assert result["n_unique"] == 3
        assert "frequencies" in result
        assert len(result["frequencies"]) == 3

    def test_categorical_truncated(self):
        values = [str(i) for i in range(30)]
        df = pl.DataFrame({"col": values})
        result = get_boxed_data(df, "col")
        assert result["truncated"] is True
        assert len(result["frequencies"]) == 20

    def test_single_value_numeric(self):
        df = pl.DataFrame({"val": [5.0, 5.0, 5.0, 5.0, 5.1]})
        result = get_boxed_data(df, "val", bins=4)
        assert result["dtype"] == "numeric"
        assert result["min_val"] == 5.0
        assert result["max_val"] == 5.1
        assert len(result["bins"]) > 0

    def test_all_nulls_numeric(self, df_with_nulls):
        df = pl.DataFrame({"val": [None, None, None]})
        result = get_boxed_data(df, "val", bins=3)
        assert result["null_count"] == 3
        assert result["total"] == 3


class TestBoxDataRaw:
    def test_fixed_length_includes_empty_bins(self):
        # 10,12,13,20,50 → 4 个箱，中间两箱计数为 0，但不可被丢弃。
        df = pl.DataFrame({"x": [10, 12, 13, 20, 50]})
        raw = _box_data_raw(df, "x", bins=4)
        assert isinstance(raw, BinnedColumn)
        assert len(raw.counts) == 4
        assert len(raw.edges) == 5
        assert raw.counts == [4, 0, 0, 1]

    def test_invariant_counts_exclude_nulls(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, None]})
        raw = _box_data_raw(df, "x", bins=3)
        assert raw.total == 5
        assert raw.null_count == 1
        assert sum(raw.counts) == raw.total - raw.null_count == 4

    def test_constant_column_single_bin(self):
        df = pl.DataFrame({"x": [7.0, 7.0, 7.0]})
        raw = _box_data_raw(df, "x", bins=5)
        assert raw.min_val == raw.max_val == 7.0
        assert raw.counts == [3]

    def test_all_null_empty(self):
        df = pl.DataFrame({"x": [None, None]}, schema={"x": pl.Float64})
        raw = _box_data_raw(df, "x")
        assert raw.edges == [] and raw.counts == []
        assert raw.min_val is None

    def test_rejects_non_numeric(self):
        df = pl.DataFrame({"c": ["a", "b"]})
        with pytest.raises(ValueError):
            _box_data_raw(df, "c")

    def test_rejects_bad_bins(self):
        df = pl.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(ValueError):
            _box_data_raw(df, "x", bins=0)


class TestDistributionEvaluation:
    def test_normal_data_not_rejected_beats_uniform(self):
        rng = __import__("numpy").random.default_rng(42)
        df = pl.DataFrame({"v": rng.normal(100, 15, 400).tolist()})
        result = distribution_evaluation(df, "v")
        pvals = {c["distribution"]: c["p_value"] for c in result["fit"]["candidates"]}
        # 正态数据：正态候选不应被拒绝，且 p 值显著高于均匀/指数候选。
        # （注：全正数据下 gamma/lognormal 亦可良好拟合，故不断言 best_fit 恰为 normal。）
        assert pvals["normal"] > 0.05
        assert pvals["normal"] > pvals["uniform"]
        assert pvals["normal"] > pvals["exponential"]

    def test_includes_observation(self):
        df = pl.DataFrame({"v": [float(i) for i in range(50)]})
        result = distribution_evaluation(df, "v", bins=5)
        assert result["observation"]["dtype"] == "numeric"
        assert len(result["observation"]["bins"]) == 5

    def test_positive_only_skipped_on_nonpositive(self):
        rng = __import__("numpy").random.default_rng(0)
        df = pl.DataFrame({"v": rng.normal(0, 5, 200).tolist()})  # 含负值
        result = distribution_evaluation(df, "v")
        skipped = {s["distribution"] for s in result["fit"].get("skipped", [])}
        assert {"lognormal", "gamma"} <= skipped

    def test_non_numeric_error(self):
        df = pl.DataFrame({"c": ["a", "b", "c"]})
        result = distribution_evaluation(df, "c")
        assert "error" in result

    def test_missing_column_error(self):
        df = pl.DataFrame({"v": [1.0, 2.0]})
        result = distribution_evaluation(df, "nope")
        assert "error" in result

    def test_insufficient_sample_skips_fit(self):
        df = pl.DataFrame({"v": [1.0, 2.0, 3.0]})
        result = distribution_evaluation(df, "v")
        assert "error" in result["fit"]
        assert "observation" in result  # 观测数据仍然返回
