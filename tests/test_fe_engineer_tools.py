import json

import polars as pl
import pytest

from kaggler.modes.feature_engineering.tools import make_tools
from kaggler.modes.feature_engineering.types import (
    Condition,
    ConditionGroup,
    EncodePair,
    FillPair,
    MonoSpec,
)
from kaggler.persistence.data_provider import DataProvider


@pytest.fixture
def data() -> DataProvider:
    dp = DataProvider()
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat": ["a", "b", "c", "d", "e"],
        }
    )
    dp.add_source(lambda: df, description="test")
    return dp


@pytest.fixture
def data_multi() -> DataProvider:
    dp = DataProvider()
    df = pl.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 6.0, 7.0, 8.0],
            "f2": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            "f3": [2.0, 3.0, 1.0, 8.0, 9.0, 7.0],
            "label": ["a", "a", "a", "b", "b", "b"],
        }
    )
    dp.add_source(lambda: df, description="test")
    return dp


def _by_name(tools):
    return {t.name: t for t in tools}


class TestMakeFeatEngTools:
    def test_returns_nine_named_tools(self, data):
        tools = make_tools(data)
        names = set(_by_name(tools))
        assert names == {
            "execute_empty_value",
            "encode_columns",
            "standardize_columns",
            "drop_columns",
            "filter_rows",
            "create_indicator_column",
            "execute_dim_reduct",
            "transform_column_mono",
            "transform_column_combination",
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
        info = data.get_version_info(1)
        assert info.parent == 0
        assert info.tool == "standardize_columns"
        assert "x" in info.description and "y" in info.description

    def test_standardize_columns_error(self, data):
        tool = _by_name(make_tools(data))["standardize_columns"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_2",
            columns=["z"],
        )
        msg = json.loads(result_cmd.update["messages"][0].content)
        assert "error" in msg

    # --- drop_columns ---
    def test_drop_columns_success(self, data):
        tool = _by_name(make_tools(data))["drop_columns"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_8",
            columns=["cat"],
        )
        update = result_cmd.update
        assert update["data_version"] == 1
        msg = json.loads(update["messages"][0].content)
        assert msg["rows_before"] == 5
        assert msg["rows_after"] == 5
        info = data.get_version_info(1)
        assert info.parent == 0
        assert info.tool == "drop_columns"
        assert "cat" in info.description

    def test_drop_columns_error(self, data):
        tool = _by_name(make_tools(data))["drop_columns"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_9",
            columns=["z"],
        )
        msg = json.loads(result_cmd.update["messages"][0].content)
        assert "error" in msg

    # --- filter_rows ---
    def test_filter_rows_keep_success(self, data):
        tool = _by_name(make_tools(data))["filter_rows"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_10",
            groups=[ConditionGroup(logic="and", conditions=[Condition(column="x", op="gt", value=2.0)])],
            group_logic="and",
            action="keep",
        )
        update = result_cmd.update
        assert update["data_version"] == 1
        msg = json.loads(update["messages"][0].content)
        assert msg["rows_before"] == 5
        assert msg["rows_after"] == 3
        info = data.get_version_info(1)
        assert info.parent == 0
        assert info.tool == "filter_rows"
        assert "keep" in info.description

    def test_filter_rows_delete_success(self, data):
        tool = _by_name(make_tools(data))["filter_rows"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_11",
            groups=[ConditionGroup(logic="and", conditions=[Condition(column="x", op="gt", value=2.0)])],
            group_logic="and",
            action="delete",
        )
        update = result_cmd.update
        assert update["data_version"] == 1
        msg = json.loads(update["messages"][0].content)
        assert msg["rows_after"] == 2
        info = data.get_version_info(1)
        assert info.parent == 0
        assert info.tool == "filter_rows"
        assert "delete" in info.description

    def test_filter_rows_error_does_not_bump_version(self, data):
        tool = _by_name(make_tools(data))["filter_rows"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_12",
            groups=[ConditionGroup(logic="and", conditions=[Condition(column="zzz", op="gt", value=2.0)])],
            group_logic="and",
            action="keep",
        )
        update = result_cmd.update
        assert "data_version" not in update
        msg = json.loads(update["messages"][0].content)
        assert "error" in msg

    # --- create_indicator_column ---
    def test_create_indicator_success(self, data):
        tool = _by_name(make_tools(data))["create_indicator_column"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_ind_1",
            groups=[ConditionGroup(logic="and", conditions=[Condition(column="x", op="gt", value=2.0)])],
            group_logic="or",
            output_name="x_gt2",
        )
        update = result_cmd.update
        assert update["data_version"] == 1
        msg = json.loads(update["messages"][0].content)
        assert msg["rows_before"] == 5
        assert msg["rows_after"] == 5
        assert msg["summary"][0]["output_column"] == "x_gt2"
        assert msg["summary"][0]["rows_flagged"] == 3
        info = data.get_version_info(1)
        assert info.parent == 0
        assert info.tool == "create_indicator_column"
        assert "x_gt2" in info.description
        # 新列真的写入了
        new_df = data.get(1)
        assert "x_gt2" in new_df.columns
        assert new_df["x_gt2"].dtype == pl.Int8

    def test_create_indicator_name_conflict_does_not_bump_version(self, data):
        tool = _by_name(make_tools(data))["create_indicator_column"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_ind_2",
            groups=[ConditionGroup(logic="and", conditions=[Condition(column="x", op="gt", value=2.0)])],
            group_logic="or",
            output_name="x",  # 与已有列冲突
        )
        update = result_cmd.update
        assert "data_version" not in update
        msg = json.loads(update["messages"][0].content)
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
        info = data.get_version_info(1)
        assert info.parent == 0
        assert info.tool == "execute_dim_reduct"
        assert "pca" in info.description

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
        info = data_multi.get_version_info(1)
        assert info.parent == 0
        assert info.tool == "execute_dim_reduct"
        assert "lda" in info.description
        assert "label" in info.description

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
        info = data.get_version_info(1)
        assert info.parent == 0
        assert info.tool == "execute_empty_value"
        assert "x" in info.description

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
        info = data.get_version_info(1)
        assert info.parent == 0
        assert info.tool == "encode_columns"
        assert "cat" in info.description

    # --- transform_column_mono ---
    def test_transform_column_mono_success(self, data):
        tool = _by_name(make_tools(data))["transform_column_mono"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_13",
            specs=[MonoSpec(column="x", method="square")],
        )
        update = result_cmd.update
        assert update["data_version"] == 1
        msg = json.loads(update["messages"][0].content)
        assert msg["rows_before"] == 5
        assert msg["rows_after"] == 5
        assert msg["summary"][0]["output_column"] == "square_x"
        info = data.get_version_info(1)
        assert info.parent == 0
        assert info.tool == "transform_column_mono"
        assert "x" in info.description
        # 新列已加入版本数据，原列保留
        new_df = data.get(1)
        assert "square_x" in new_df.columns
        assert "x" in new_df.columns

    def test_transform_column_mono_error(self, data):
        tool = _by_name(make_tools(data))["transform_column_mono"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_14",
            specs=[MonoSpec(column="zzz", method="cos")],
        )
        update = result_cmd.update
        assert "data_version" not in update
        msg = json.loads(update["messages"][0].content)
        assert "error" in msg

    # --- transform_column_combination ---
    def test_transform_column_combination_success(self, data):
        tool = _by_name(make_tools(data))["transform_column_combination"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_15",
            columns=["x", "y"],
            method="product",
            output_name="x_times_y",
        )
        update = result_cmd.update
        assert update["data_version"] == 1
        msg = json.loads(update["messages"][0].content)
        assert msg["rows_before"] == 5
        assert msg["summary"][0]["output_column"] == "x_times_y"
        info = data.get_version_info(1)
        assert info.parent == 0
        assert info.tool == "transform_column_combination"
        assert "x_times_y" in info.description
        new_df = data.get(1)
        assert "x_times_y" in new_df.columns
        assert new_df["x_times_y"].to_list() == [10.0, 40.0, 90.0, 160.0, 250.0]

    def test_transform_column_combination_error(self, data):
        tool = _by_name(make_tools(data))["transform_column_combination"]
        result_cmd = tool.func(
            state={"data_version": 0},
            tool_call_id="call_16",
            columns=["x", "cat"],
            method="product",
            output_name="bad",
        )
        update = result_cmd.update
        assert "data_version" not in update
        msg = json.loads(update["messages"][0].content)
        assert "error" in msg
