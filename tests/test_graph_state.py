from typing import Annotated

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from kaggler.graph.state import CommonState, _add_turns, _take_latest
from kaggler.shared.types import Mode


class TestAddTurns:
    def test_default_increment(self):
        assert _add_turns(0) == 1
        assert _add_turns(4) == 5

    def test_explicit_increment(self):
        assert _add_turns(2, 3) == 5

    def test_accumulates(self):
        turn = 0
        for _ in range(3):
            turn = _add_turns(turn)
        assert turn == 3


class TestCommonState:
    def test_dict_literal_construction(self):
        # CommonState 本质是 TypedDict，运行时即普通 dict
        state: CommonState = {
            "messages": [],
            "current_mode": Mode.EDA,
            "file_path": "data.csv",
            "explored_schema": "",
            "turn": 0,
            "summary": "",
            "data_version": 0,
        }
        assert state["current_mode"] == Mode.EDA
        assert state["data_version"] == 0


class TestTakeLatest:
    def test_returns_update_ignoring_current(self):
        # 语义：后写覆盖先写，与当前值无关
        assert _take_latest(Mode.EDA, Mode.FEAT_ENG) == Mode.FEAT_ENG
        assert _take_latest(0, 7) == 7

    def test_single_write_replaces(self):
        assert _take_latest(3, 3) == 3


def _make_multiwrite_graph(field: str):
    """构造一个 ToolNode 图：工具返回 Command 写 ``field``。同一 AIMessage 里放多个
    tool_call 即可让该字段在一个 super-step 内被多次写入——正是触发 InvalidUpdateError
    的场景，用于验证 reducer 已让多写收敛为「取最后一个」而非报错。
    """

    @tool
    def write_field(value: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """写入受测字段。"""
        payload = Mode(value) if field == "current_mode" else int(value)
        return Command(update={
            field: payload,
            "messages": [ToolMessage("ok", tool_call_id=tool_call_id)],
        })

    builder = StateGraph(CommonState)
    builder.add_node("tools", ToolNode([write_field], handle_tool_errors=True))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile(), write_field


class TestReducerAppliedInGraph:
    """端到端：编译后的真实 CommonState 上，同一 step 多写不再抛 InvalidUpdateError，
    且结果取最后一个写入值（reducer 已正确绑定到 channel）。"""

    def test_multiple_current_mode_writes_take_last(self):
        graph, _ = _make_multiwrite_graph("current_mode")
        ai = AIMessage(content="", tool_calls=[
            {"name": "write_field", "args": {"value": Mode.FEAT_ENG.value}, "id": "1"},
            {"name": "write_field", "args": {"value": Mode.EDA.value}, "id": "2"},
        ])
        out = graph.invoke({"messages": [ai], "current_mode": Mode.FEAT_ENG})
        assert out["current_mode"] == Mode.EDA

    def test_multiple_data_version_writes_take_last(self):
        graph, _ = _make_multiwrite_graph("data_version")
        ai = AIMessage(content="", tool_calls=[
            {"name": "write_field", "args": {"value": "1"}, "id": "1"},
            {"name": "write_field", "args": {"value": "2"}, "id": "2"},
        ])
        out = graph.invoke({"messages": [ai], "data_version": 0})
        assert out["data_version"] == 2

    def test_single_write_still_replaces(self):
        graph, _ = _make_multiwrite_graph("data_version")
        ai = AIMessage(content="", tool_calls=[
            {"name": "write_field", "args": {"value": "5"}, "id": "1"},
        ])
        out = graph.invoke({"messages": [ai], "data_version": 0})
        assert out["data_version"] == 5
