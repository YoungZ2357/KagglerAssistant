from langchain_core.messages import ToolMessage
from langgraph.types import Command

from kaggler.modes.common.tools import make_tools
from kaggler.shared.types import Mode


class TestMakeCommonTools:
    def test_returns_single_switch_mode_tool(self):
        tools = make_tools()
        assert len(tools) == 1
        assert tools[0].name == "switch_mode"

    def test_tool_has_docstring(self):
        assert make_tools()[0].description

    def test_switch_mode_returns_command(self):
        switch_mode = make_tools()[0]
        cmd = switch_mode.func(new_mode=Mode.EDA, tool_call_id="tc1")
        assert isinstance(cmd, Command)
        assert cmd.update["current_mode"] == Mode.EDA

    def test_switch_mode_emits_tool_message(self):
        switch_mode = make_tools()[0]
        cmd = switch_mode.func(new_mode=Mode.EDA, tool_call_id="tc-abc")
        msgs = cmd.update["messages"]
        assert len(msgs) == 1
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].tool_call_id == "tc-abc"
        assert "eda" in msgs[0].content.lower()
