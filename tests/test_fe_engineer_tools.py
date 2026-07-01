import json

import polars as pl
import pytest

from kaggler.modes.feature_engineering.tools import make_tools
from kaggler.modes.feature_engineering.types import EncodePair, FillPair
from kaggler.persistence.data_provider import DataProvider


@pytest.fixture
def data() -> DataProvider:
    dp = DataProvider()
    dp._frames[0] = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat": ["a", "b", "c", "d", "e"],
        }
    )
    return dp


@pytest.fixture
def data_multi() -> DataProvider:
    dp = DataProvider()
    dp._frames[0] = pl.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 6.0, 7.0, 8.0],
            "f2": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            "f3": [2.0, 3.0, 1.0, 8.0, 9.0, 7.0],
            "label": ["a", "a", "a", "b", "b", "b"],
        }
    )
    return dp


def _by_name(tools):
    return {t.name: t for t in tools}


class TestMakeFeatEngTools:
    def test_returns_four_named_tools(self, data):
        tools = make_tools(data)
        names = set(_by_name(tools))
        assert names == {
            "execute_empty_value",
            "encode_columns",
            "standardize_columns",
            "execute_dim_reduct",
        }

    # --- standardize_columns ---
    def test_standardize_columns_success(self, data):
        tool = _by_name(make_tools(data))["standardize_columns"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_1",
            columns=["x", "y"],
        )
        update = result_cmd.update
        assert update["data_version"] == 1
        msg = json.loads(update["messages"][0].content)
        assert msg["rows_before"] == 5
        assert msg["rows_after"] == 5
        assert len(msg["preview"]) == 3
        assert "columns" in msg["summary"][0]

    def test_standardize_columns_error(self, data):
        tool = _by_name(make_tools(data))["standardize_columns"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_2",
            columns=["z"],
        )
        msg = json.loads(result_cmd.update["messages"][0].content)
        assert "error" in msg

    # --- execute_dim_reduct ---
    def test_dim_reduct_pca_success(self, data):
        tool = _by_name(make_tools(data))["execute_dim_reduct"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_3",
            method="pca",
            n_components=1,
        )
        update = result_cmd.update
        assert update["data_version"] == 1
        msg = json.loads(update["messages"][0].content)
        assert msg["rows_before"] == 5
        assert msg["rows_after"] == 5
        assert msg["summary"][0]["method"] == "pca"

    def test_dim_reduct_lda_success(self, data_multi):
        tool = _by_name(make_tools(data_multi))["execute_dim_reduct"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_4",
            method="lda",
            n_components=1,
            target="label",
        )
        update = result_cmd.update
        assert update["data_version"] == 1
        msg = json.loads(update["messages"][0].content)
        assert msg["summary"][0]["method"] == "lda"

    def test_dim_reduct_error(self, data):
        tool = _by_name(make_tools(data))["execute_dim_reduct"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_5",
            method="tsne",
            n_components=1,
        )
        msg = json.loads(result_cmd.update["messages"][0].content)
        assert "error" in msg

    # --- execute_empty_value success path ---
    def test_execute_empty_value_success(self, data):
        tool = _by_name(make_tools(data))["execute_empty_value"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_6",
            pairs=[FillPair(column="x", action="zero")],
        )
        update = result_cmd.update
        assert update["data_version"] == 1
        msg = json.loads(update["messages"][0].content)
        assert msg["rows_before"] == 5

    # --- encode_columns success path ---
    def test_encode_columns_success(self, data):
        tool = _by_name(make_tools(data))["encode_columns"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_7",
            pairs=[EncodePair(column="cat", action="label")],
        )
        update = result_cmd.update
        assert update["data_version"] == 1
        msg = json.loads(update["messages"][0].content)
        assert msg["rows_before"] == 5
