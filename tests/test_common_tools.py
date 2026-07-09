import json

import polars as pl
import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

import kaggler.workspace.manager as wsm
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
    def test_returns_all_common_tools(self, data):
        tools = make_tools(data)
        assert len(tools) == 7
        assert {t.name for t in tools} == {
            "switch_mode", "switch_data_version", "list_data_versions",
            "list_workspace_files", "export_data_version",
            "add_todo", "complete_todo",
        }

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


class TestTodoTools:
    def test_add_todo_returns_command_without_id(self, data):
        # add_todo 只提交 content + status，不带 id——id 由 _merge_todos reducer 分配
        add_todo = _by_name(make_tools(data))["add_todo"]
        cmd = add_todo.func(content="稍后做特征缩放", tool_call_id="tc1")
        assert isinstance(cmd, Command)
        todos = cmd.update["todos"]
        assert todos == [{"content": "稍后做特征缩放", "status": "open"}]
        assert "id" not in todos[0]

    def test_add_todo_emits_tool_message(self, data):
        add_todo = _by_name(make_tools(data))["add_todo"]
        cmd = add_todo.func(content="记得导出结果", tool_call_id="tc-add")
        msg = cmd.update["messages"][0]
        assert isinstance(msg, ToolMessage)
        assert msg.tool_call_id == "tc-add"
        payload = json.loads(msg.content)
        assert payload["added_todo"] == "记得导出结果"

    def test_complete_todo_marks_done(self, data):
        complete_todo = _by_name(make_tools(data))["complete_todo"]
        state = {"todos": [{"id": 3, "content": "缩放", "status": "open"}]}
        cmd = complete_todo.func(todo_id=3, tool_call_id="tc2", state=state)
        assert isinstance(cmd, Command)
        assert cmd.update["todos"] == [{"id": 3, "status": "done"}]
        payload = json.loads(cmd.update["messages"][0].content)
        assert payload["completed_todo"] == 3
        assert payload["content"] == "缩放"

    def test_complete_todo_unknown_id_returns_error(self, data):
        complete_todo = _by_name(make_tools(data))["complete_todo"]
        state = {"todos": [{"id": 1, "content": "x", "status": "open"}]}
        cmd = complete_todo.func(todo_id=99, tool_call_id="tc3", state=state)
        assert "todos" not in cmd.update
        payload = json.loads(cmd.update["messages"][0].content)
        assert "error" in payload
        assert "99" in payload["error"]

    def test_complete_todo_empty_state_returns_error(self, data):
        complete_todo = _by_name(make_tools(data))["complete_todo"]
        cmd = complete_todo.func(todo_id=1, tool_call_id="tc4", state={})
        payload = json.loads(cmd.update["messages"][0].content)
        assert "error" in payload


@pytest.fixture
def active_ws(tmp_path, monkeypatch):
    """设置一个临时活跃工作区(已建 .kaggler 布局);避免污染真实 ~/.kaggler。"""
    ws = wsm.Workspace(tmp_path)
    ws.ensure_layout()
    monkeypatch.setattr(wsm, "_active", ws)
    return ws


class TestExportDataVersion:
    _CFG = {"configurable": {"thread_id": "t-export"}}

    def test_export_current_version_success(self, data, active_ws):
        tool = _by_name(make_tools(data))["export_data_version"]
        out = tool.func(
            filename="out.csv", tool_call_id="tc", state={"data_version": 0},
            config=self._CFG,
        )
        payload = json.loads(out)
        assert payload["exported_version"] == 0
        assert payload["format"] == "csv"
        assert payload["rows"] == 3
        target = active_ws.path / "exports" / "out.csv"
        assert target.exists()
        assert pl.read_csv(target)["a"].to_list() == [1, 2, 3]

    def test_export_records_catalog_row(self, data, active_ws):
        from kaggler.persistence.data_version_store import DataVersionStore

        tool = _by_name(make_tools(data))["export_data_version"]
        tool.func(
            filename="out.csv", tool_call_id="tc", state={"data_version": 0},
            config=self._CFG,
        )
        store = DataVersionStore(active_ws.data_version_db)
        try:
            rows = store.list_all()
        finally:
            store.close()
        assert len(rows) == 1
        assert rows[0].version == 0
        assert rows[0].thread_id == "t-export"
        assert rows[0].format == "csv"

    def test_export_no_workspace_returns_error(self, data, monkeypatch):
        monkeypatch.setattr(wsm, "_active", None)
        tool = _by_name(make_tools(data))["export_data_version"]
        out = tool.func(
            filename="out.csv", tool_call_id="tc", state={"data_version": 0},
            config=self._CFG,
        )
        assert "未设置工作区" in out

    def test_export_escape_path_returns_error(self, data, active_ws):
        # 足够多的 ../ 逃出工作区根 -> resolve_within 拒绝(安全边界是工作区，非 exports 子目录)。
        tool = _by_name(make_tools(data))["export_data_version"]
        out = tool.func(
            filename="../../../evil.csv", tool_call_id="tc", state={"data_version": 0},
            config=self._CFG,
        )
        assert "工作区之外" in out

    def test_export_nonexistent_version_returns_error(self, data, active_ws):
        tool = _by_name(make_tools(data))["export_data_version"]
        out = tool.func(
            filename="out.csv", tool_call_id="tc", state={"data_version": 0},
            config=self._CFG, version=99,
        )
        payload = json.loads(out)
        assert "error" in payload
        assert "不存在" in payload["error"]

    def test_export_parquet_by_suffix(self, data, active_ws):
        tool = _by_name(make_tools(data))["export_data_version"]
        out = tool.func(
            filename="out.parquet", tool_call_id="tc", state={"data_version": 0},
            config=self._CFG,
        )
        payload = json.loads(out)
        assert payload["format"] == "parquet"
        target = active_ws.path / "exports" / "out.parquet"
        assert pl.read_parquet(target)["a"].to_list() == [1, 2, 3]
