import json

import pytest

from kaggler.modes.eda.tools import make_tools
from kaggler.persistence.data_provider import DataProvider


@pytest.fixture
def data(df_mixed) -> DataProvider:
    dp = DataProvider()
    dp._frames[0] = df_mixed
    return dp


def _by_name(tools):
    return {t.name: t for t in tools}


class TestMakeEdaTools:
    def test_returns_five_named_tools(self, data):
        tools = make_tools(data)
        assert set(_by_name(tools)) == {
            "explore_schema",
            "correlation_analysis",
            "descriptive_analysis",
            "distribution_analysis_raw",
            "distribution_fit",
        }

    def test_explore_schema_returns_json_report(self, data):
        tool = _by_name(make_tools(data))["explore_schema"]
        result = json.loads(tool.func(state={"data_version": 0}))
        assert result["total_rows"] == 5
        assert result["total_columns"] == 5

    def test_correlation_analysis_wraps_compute(self, data):
        tool = _by_name(make_tools(data))["correlation_analysis"]
        result = json.loads(
            tool.func(state={"data_version": 0}, columns=["age", "score"])
        )
        assert "results" in result
        assert "pearson" in result["results"]

    def test_descriptive_analysis_wraps_compute(self, data):
        tool = _by_name(make_tools(data))["descriptive_analysis"]
        result = json.loads(
            tool.func(state={"data_version": 0}, columns=["age"])
        )
        assert result["stats"][0]["column"] == "age"

    def test_distribution_analysis_wraps_compute(self, data):
        tool = _by_name(make_tools(data))["distribution_analysis_raw"]
        result = json.loads(
            tool.func(state={"data_version": 0}, column="city")
        )
        assert result["column"] == "city"
        assert result["dtype"] == "categorical"

    def test_tool_reads_versioned_frame(self, data):
        # 工具按 state 的 data_version 取数 —— 未知版本应触发 DataProvider 错误
        tool = _by_name(make_tools(data))["explore_schema"]
        with pytest.raises(RuntimeError):
            tool.func(state={"data_version": 7})
