from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from kaggler.graph.edges import entry_condition, route_after_agent
from kaggler.graph.types import Node
from kaggler.shared.config import GraphConfig


class TestEntryCondition:
    def test_below_threshold_goes_react(self):
        state = {"messages": [HumanMessage(content="q")]}
        cfg = GraphConfig(summary_trigger_count=3)
        assert entry_condition(state, graph_config=cfg) == Node.REACT

    def test_at_threshold_goes_summarize(self):
        state = {"messages": [HumanMessage(content="q")] * 3}
        cfg = GraphConfig(summary_trigger_count=3)
        assert entry_condition(state, graph_config=cfg) == Node.SUMMARIZE

    def test_above_threshold_goes_summarize(self):
        state = {"messages": [HumanMessage(content="q")] * 5}
        cfg = GraphConfig(summary_trigger_count=3)
        assert entry_condition(state, graph_config=cfg) == Node.SUMMARIZE

    def test_empty_messages_goes_react(self):
        state = {"messages": []}
        cfg = GraphConfig(summary_trigger_count=1)
        assert entry_condition(state, graph_config=cfg) == Node.REACT

    def test_at_threshold_but_nothing_deletable_goes_react(self):
        # #2：达阈值但仅 1 个进行中的巨型回合（cutoff=0）→ 跳过总结、直接 react，不空转
        state = {
            "messages": [
                HumanMessage(content="q"),
                AIMessage(content="a"),
                ToolMessage(content="r", tool_call_id="t"),
            ]
        }
        cfg = GraphConfig(summary_trigger_count=3, summary_keep_recent=4)
        assert entry_condition(state, graph_config=cfg) == Node.REACT


class TestRouteAfterAgent:
    def test_ai_with_tool_calls_goes_tools(self):
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "f", "args": {}, "id": "tc1"}],
        )
        state = {"messages": [HumanMessage(content="q"), ai]}
        assert route_after_agent(state) == Node.TOOLS

    def test_ai_without_tool_calls_goes_finish(self):
        ai = AIMessage(content="done")
        state = {"messages": [HumanMessage(content="q"), ai]}
        assert route_after_agent(state) == Node.FINISH

    def test_non_ai_last_message_goes_finish(self):
        # 末条不是 AIMessage（例如 ToolMessage）时进入收尾
        tm = ToolMessage(content="result", tool_call_id="tc1")
        state = {"messages": [HumanMessage(content="q"), tm]}
        assert route_after_agent(state) == Node.FINISH
