from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.tools import tool

from kaggler.graph.nodes import finish_turn, react_node, summarize_conversation
from kaggler.shared.config import GraphConfig
from kaggler.shared.types import Mode


@tool
def _common_tool(x: int) -> str:
    """通用工具占位。"""
    return str(x)


@tool
def _eda_tool(x: int) -> str:
    """EDA 工具占位。"""
    return str(x)


class TestFinishTurn:
    def test_returns_turn_increment(self):
        assert finish_turn({"messages": []}) == {"turn": 1}


class TestReactNode:
    def _state(self, **extra):
        base = {
            "messages": [HumanMessage(content="你好")],
            "current_mode": Mode.EDA,
        }
        base.update(extra)
        return base

    def test_binds_common_and_mode_tools(self, fake_llm):
        react_node(
            self._state(),
            llm=fake_llm,
            tools_by_mode={Mode.EDA: [_eda_tool]},
            prompt_templates={Mode.EDA: "模板 {schema}"},
            common_tools=[_common_tool],
        )
        names = {t.name for t in fake_llm.bound_tools}
        assert names == {"_common_tool", "_eda_tool"}

    def test_schema_filled_into_system_prompt(self, fake_llm):
        react_node(
            self._state(explored_schema="SCHEMA_X"),
            llm=fake_llm,
            tools_by_mode={Mode.EDA: [_eda_tool]},
            prompt_templates={Mode.EDA: "前缀 {schema} 后缀"},
            common_tools=[],
        )
        system = fake_llm.invoked_with[0]
        assert isinstance(system, SystemMessage)
        assert system.content == "前缀 SCHEMA_X 后缀"

    def test_missing_schema_defaults_empty(self, fake_llm):
        react_node(
            self._state(),  # 无 explored_schema
            llm=fake_llm,
            tools_by_mode={Mode.EDA: [_eda_tool]},
            prompt_templates={Mode.EDA: "[{schema}]"},
            common_tools=[],
        )
        assert fake_llm.invoked_with[0].content == "[]"

    def test_system_prepended_history_preserved(self, fake_llm):
        state = self._state()
        react_node(
            state,
            llm=fake_llm,
            tools_by_mode={Mode.EDA: [_eda_tool]},
            prompt_templates={Mode.EDA: "{schema}"},
            common_tools=[],
        )
        # system 临时置于最前，原始历史紧随其后
        assert isinstance(fake_llm.invoked_with[0], SystemMessage)
        assert fake_llm.invoked_with[1:] == state["messages"]

    def test_returns_response_in_messages(self, make_fake_llm):
        resp = AIMessage(content="模型回复")
        llm = make_fake_llm(resp)
        out = react_node(
            self._state(),
            llm=llm,
            tools_by_mode={Mode.EDA: [_eda_tool]},
            prompt_templates={Mode.EDA: "{schema}"},
            common_tools=[],
        )
        assert out == {"messages": [resp]}

    def test_common_tools_none_ok(self, fake_llm):
        react_node(
            self._state(),
            llm=fake_llm,
            tools_by_mode={Mode.EDA: [_eda_tool]},
            prompt_templates={Mode.EDA: "{schema}"},
            common_tools=None,
        )
        assert {t.name for t in fake_llm.bound_tools} == {"_eda_tool"}


class TestSummarizeConversation:
    def test_writes_summary_content(self, make_fake_llm):
        llm = make_fake_llm(AIMessage(content="新摘要"))
        state = {"messages": [HumanMessage(content="q", id="h1")]}
        out = summarize_conversation(
            state, llm=llm, graph_config=GraphConfig(summary_keep_recent=4)
        )
        assert out["summary"] == "新摘要"

    def test_initial_prompt_branch(self, fake_llm):
        # 无 summary → 使用 INITIAL 提示词（不含「已有摘要」字样）
        state = {"messages": [HumanMessage(content="q", id="h1")]}
        summarize_conversation(
            state, llm=fake_llm, graph_config=GraphConfig(summary_keep_recent=4)
        )
        prompt = fake_llm.invoked_with[-1].content
        assert "已有摘要" not in prompt
        assert "{template}" not in prompt  # 占位符已被替换

    def test_merge_prompt_branch(self, fake_llm):
        # 有 summary → 使用 MERGE 提示词，旧摘要被注入
        state = {
            "messages": [HumanMessage(content="q", id="h1")],
            "summary": "旧的摘要内容",
        }
        summarize_conversation(
            state, llm=fake_llm, graph_config=GraphConfig(summary_keep_recent=4)
        )
        prompt = fake_llm.invoked_with[-1].content
        assert "已有摘要" in prompt
        assert "旧的摘要内容" in prompt
        assert "{summary}" not in prompt

    def test_no_deletion_when_within_keep(self, fake_llm):
        # HumanMessage 数量未超过 keep → 不删除任何消息
        msgs = [HumanMessage(content=f"q{i}", id=f"h{i}") for i in range(2)]
        out = summarize_conversation(
            {"messages": msgs}, llm=fake_llm, graph_config=GraphConfig(summary_keep_recent=4)
        )
        assert out["messages"] == []

    def test_deletes_old_messages_at_human_boundary(self, fake_llm):
        # 6 条 Human，keep=2 → 截断点落在倒数第 2 个 Human，删除其前的所有带 id 消息
        msgs = [HumanMessage(content=f"q{i}", id=f"h{i}") for i in range(6)]
        out = summarize_conversation(
            {"messages": msgs}, llm=fake_llm, graph_config=GraphConfig(summary_keep_recent=2)
        )
        deleted = out["messages"]
        assert all(isinstance(m, RemoveMessage) for m in deleted)
        # 保留最近 2 个 Human（h4, h5），删除 h0..h3
        assert {m.id for m in deleted} == {"h0", "h1", "h2", "h3"}

    def test_skips_messages_without_id(self, fake_llm):
        # 缺 id 的消息无法 RemoveMessage，应被跳过
        msgs = [
            HumanMessage(content="q0"),  # 无 id
            HumanMessage(content="q1", id="h1"),
            HumanMessage(content="q2", id="h2"),
            HumanMessage(content="q3", id="h3"),
        ]
        out = summarize_conversation(
            {"messages": msgs}, llm=fake_llm, graph_config=GraphConfig(summary_keep_recent=1)
        )
        # keep=1 → cutoff 在 h3，删除 q0/h1/h2，但 q0 无 id 被跳过
        assert {m.id for m in out["messages"]} == {"h1", "h2"}
