from typing import Annotated

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from kaggler.graph.state import CommonState, _add_turns, _merge_todos, _take_latest
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
            "memory": {},
            "data_version": 0,
            "todos": [],
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


class TestMergeTodos:
    def test_none_inputs_yield_empty(self):
        assert _merge_todos(None, None) == []

    def test_new_todo_gets_id_one_when_empty(self):
        out = _merge_todos([], [{"content": "a", "status": "open"}])
        assert out == [{"content": "a", "status": "open", "id": 1}]

    def test_new_todo_id_continues_from_max(self):
        current = [{"id": 5, "content": "old", "status": "open"}]
        out = _merge_todos(current, [{"content": "new", "status": "open"}])
        assert {t["id"] for t in out} == {5, 6}

    def test_update_existing_merges_fields(self):
        current = [{"id": 2, "content": "x", "status": "open"}]
        out = _merge_todos(current, [{"id": 2, "status": "done"}])
        assert out == [{"id": 2, "content": "x", "status": "done"}]

    def test_two_new_in_one_update_do_not_collide(self):
        # 同一 update 批内两条新待办应拿到不同 id
        out = _merge_todos([], [
            {"content": "a", "status": "open"},
            {"content": "b", "status": "open"},
        ])
        assert sorted(t["id"] for t in out) == [1, 2]

    def test_unknown_id_inserted_as_new(self):
        out = _merge_todos([], [{"id": 7, "content": "z", "status": "done"}])
        assert out == [{"id": 7, "content": "z", "status": "done"}]


class TestMergeTodosInGraph:
    """端到端：真实 CommonState 上，一个 super-step 内两次 add_todo（各返回 Command
    写 todos）应逐条 fold、各得独立 id，而非抛 InvalidUpdateError 或互相覆盖。"""

    def test_two_adds_in_one_step_both_persisted(self):
        @tool
        def add(content: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
            """追加一条待办。"""
            return Command(update={
                "todos": [{"content": content, "status": "open"}],
                "messages": [ToolMessage("ok", tool_call_id=tool_call_id)],
            })

        builder = StateGraph(CommonState)
        builder.add_node("tools", ToolNode([add], handle_tool_errors=True))
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        graph = builder.compile()

        ai = AIMessage(content="", tool_calls=[
            {"name": "add", "args": {"content": "first"}, "id": "1"},
            {"name": "add", "args": {"content": "second"}, "id": "2"},
        ])
        out = graph.invoke({"messages": [ai], "todos": []})
        contents = {t["content"] for t in out["todos"]}
        ids = {t["id"] for t in out["todos"]}
        assert contents == {"first", "second"}
        assert len(ids) == 2  # 两条各得独立 id


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
