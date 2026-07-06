import json

import polars as pl
import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from kaggler.modes.common.tools import make_tools
from kaggler.persistence.data_provider import DataProvider
from kaggler.shared.types import Mode


@pytest.fixture
def data() -> DataProvider:
    dp = DataProvider()
    df = pl.DataFrame({"a": [1, 2, 3]})
    dp.add_source(lambda: df, description="原始数据集")
    return dp


@pytest.fixture
def data_multi_version(data) -> DataProvider:
    data.add_version(
        DataProvider.eager_op(lambda _: pl.DataFrame({"a": [1, 2]})),
        parent=0, tool="drop_columns", description="删除了1行",
    )
    data.add_version(
        DataProvider.eager_op(lambda _: pl.DataFrame({"a": [1]})),
        parent=1, tool="filter_rows", description="再删除了1行",
    )
    return data


def _by_name(tools):
    return {t.name: t for t in tools}


class TestMakeCommonTools:
    def test_returns_three_tools(self, data):
        tools = make_tools(data)
        assert len(tools) == 3
        assert {t.name for t in tools} == {"switch_mode", "switch_data_version", "list_data_versions"}

    def test_tool_has_docstring(self, data):
        assert make_tools(data)[0].description

    def test_switch_mode_returns_command(self, data):
        switch_mode = _by_name(make_tools(data))["switch_mode"]
        cmd = switch_mode.func(new_mode=Mode.EDA, tool_call_id="tc1")
        assert isinstance(cmd, Command)
        assert cmd.update["current_mode"] == Mode.EDA

    def test_switch_mode_emits_tool_message(self, data):
        switch_mode = _by_name(make_tools(data))["switch_mode"]
        cmd = switch_mode.func(new_mode=Mode.EDA, tool_call_id="tc-abc")
        msgs = cmd.update["messages"]
        assert len(msgs) == 1
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].tool_call_id == "tc-abc"
        assert "eda" in msgs[0].content.lower()

    def test_switch_mode_invoke_injects_tool_call_id(self, data):
        """走 .invoke() 真实运行时路径：InjectedToolCallId 注入的是字符串，
        若 tool_call_id 注解写成 dict 会在 args_schema 校验时报错。
        直调 .func 无法覆盖此路径，故用 ToolCall dict 触发注入 + 校验。
        """
        switch_mode = _by_name(make_tools(data))["switch_mode"]
        cmd = switch_mode.invoke({
            "name": "switch_mode",
            "args": {"new_mode": Mode.FEAT_ENG},
            "id": "tc-xyz",
            "type": "tool_call",
        })
        assert isinstance(cmd, Command)
        assert cmd.update["current_mode"] == Mode.FEAT_ENG
        assert cmd.update["messages"][0].tool_call_id == "tc-xyz"


class TestSwitchDataVersion:
    def test_switch_to_existing_version_success(self, data_multi_version):
        tool = _by_name(make_tools(data_multi_version))["switch_data_version"]
        cmd = tool.func(version=1, tool_call_id="tc1")
        assert isinstance(cmd, Command)
        assert cmd.update["data_version"] == 1
        payload = json.loads(cmd.update["messages"][0].content)
        assert payload["current_data_version"] == 1
        assert payload["parent"] == 0
        assert payload["tool"] == "drop_columns"
        assert payload["description"] == "删除了1行"

    def test_switch_to_nonexistent_version_returns_error(self, data_multi_version):
        tool = _by_name(make_tools(data_multi_version))["switch_data_version"]
        cmd = tool.func(version=99, tool_call_id="tc2")
        assert "data_version" not in cmd.update
        payload = json.loads(cmd.update["messages"][0].content)
        assert "error" in payload
        assert "不存在" in payload["error"]

    def test_switch_data_version_invoke_injects_tool_call_id(self, data_multi_version):
        tool = _by_name(make_tools(data_multi_version))["switch_data_version"]
        cmd = tool.invoke({
            "name": "switch_data_version",
            "args": {"version": 0},
            "id": "tc-xyz",
            "type": "tool_call",
        })
        assert isinstance(cmd, Command)
        assert cmd.update["data_version"] == 0
        assert cmd.update["messages"][0].tool_call_id == "tc-xyz"


class TestListDataVersions:
    def test_returns_str(self, data_multi_version):
        tool = _by_name(make_tools(data_multi_version))["list_data_versions"]
        assert isinstance(tool.func(), str)

    def test_content_and_order(self, data_multi_version):
        tool = _by_name(make_tools(data_multi_version))["list_data_versions"]
        versions = json.loads(tool.func())
        assert [v["version"] for v in versions] == [0, 1, 2]
        assert versions[1]["parent"] == 0
        assert versions[1]["tool"] == "drop_columns"
        assert versions[2]["description"] == "再删除了1行"

    def test_empty_provider_returns_empty_list(self):
        empty_data = DataProvider()
        tool = _by_name(make_tools(empty_data))["list_data_versions"]
        assert json.loads(tool.func()) == []
