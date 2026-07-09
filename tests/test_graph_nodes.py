from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.tools import tool

from kaggler.graph.nodes import (
    finish_turn,
    react_node,
    summarize_conversation,
    summary_cutoff,
)
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
        # 模板填充后作为前缀；工具声明块追加在其后
        assert system.content.startswith("前缀 SCHEMA_X 后缀")

    def test_missing_schema_defaults_empty(self, fake_llm):
        react_node(
            self._state(),  # 无 explored_schema
            llm=fake_llm,
            tools_by_mode={Mode.EDA: [_eda_tool]},
            prompt_templates={Mode.EDA: "[{schema}]"},
            common_tools=[],
        )
        assert fake_llm.invoked_with[0].content.startswith("[]")

    def test_memory_injected_when_present(self, fake_llm):
        # (a) state 里有结构化记忆 → 分区渲染后回注进系统提示词
        react_node(
            self._state(memory={"user_goal": "预测房价", "progress": "历史要点摘要"}),
            llm=fake_llm,
            tools_by_mode={Mode.EDA: [_eda_tool]},
            prompt_templates={Mode.EDA: "{schema}"},
            common_tools=[],
        )
        content = fake_llm.invoked_with[0].content
        assert "Agent对之前对话的已知信息" in content
        assert "预测房价" in content
        assert "历史要点摘要" in content

    def test_no_memory_block_when_absent(self, fake_llm):
        # 无记忆 → 不出现记忆块
        react_node(
            self._state(),
            llm=fake_llm,
            tools_by_mode={Mode.EDA: [_eda_tool]},
            prompt_templates={Mode.EDA: "{schema}"},
            common_tools=[],
        )
        assert "Agent对之前对话的已知信息" not in fake_llm.invoked_with[0].content

    def test_open_todos_injected(self, fake_llm):
        # (c) 未完成待办逐字注入，已完成的不出现
        react_node(
            self._state(todos=[
                {"id": 1, "content": "做特征缩放", "status": "open"},
                {"id": 2, "content": "已完成项", "status": "done"},
            ]),
            llm=fake_llm,
            tools_by_mode={Mode.EDA: [_eda_tool]},
            prompt_templates={Mode.EDA: "{schema}"},
            common_tools=[],
        )
        content = fake_llm.invoked_with[0].content
        assert "待办管理" in content
        assert "[#1] 做特征缩放" in content
        assert "已完成项" not in content

    def test_todo_guidance_always_present(self, fake_llm):
        # 即使无待办，仍附「用 add_todo 登记」的轻量指引
        react_node(
            self._state(),
            llm=fake_llm,
            tools_by_mode={Mode.EDA: [_eda_tool]},
            prompt_templates={Mode.EDA: "{schema}"},
            common_tools=[],
        )
        assert "add_todo" in fake_llm.invoked_with[0].content

    def test_available_tools_declared_in_prompt(self, fake_llm):
        # (b) 当前可调用工具（common + 当前模式）显式写入提示词
        react_node(
            self._state(),
            llm=fake_llm,
            tools_by_mode={Mode.EDA: [_eda_tool]},
            prompt_templates={Mode.EDA: "{schema}"},
            common_tools=[_common_tool],
        )
        content = fake_llm.invoked_with[0].content
        assert "当前可调用工具" in content
        assert "_eda_tool" in content
        assert "_common_tool" in content

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
        assert out["messages"] == [resp]
        assert "context_usage" in out

    def test_context_usage_emitted(self, fake_llm):
        out = react_node(
            self._state(),
            llm=fake_llm,
            tools_by_mode={Mode.EDA: [_eda_tool]},
            prompt_templates={Mode.EDA: "模板 {schema}"},
            common_tools=[_common_tool],
        )
        cu = out["context_usage"]
        assert set(cu["categories"]) == {
            "system", "summary", "tools", "user", "assistant", "tool_results",
        }
        assert cu["recommended"] == 256_000 and cu["limit"] == 1_000_000
        # 无 usage_metadata（假 LLM）→ 无实测、系数不动
        assert cu["actual_total"] is None
        assert cu["calibration_factor"] == 1.0

    def test_context_usage_captures_actual_and_calibrates(self, make_fake_llm):
        resp = AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 1234, "output_tokens": 5, "total_tokens": 1239},
        )
        llm = make_fake_llm(resp)
        out = react_node(
            self._state(memory={"progress": "较长的中文摘要内容片段" * 20}),
            llm=llm,
            tools_by_mode={Mode.EDA: [_eda_tool]},
            prompt_templates={Mode.EDA: "模板 {schema}"},
            common_tools=[_common_tool],
        )
        cu = out["context_usage"]
        assert cu["actual_total"] == 1234
        assert cu["total"] == 1234  # 有实测取实测
        assert cu["calibration_factor"] != 1.0  # 已按实测校准

    def test_common_tools_none_ok(self, fake_llm):
        react_node(
            self._state(),
            llm=fake_llm,
            tools_by_mode={Mode.EDA: [_eda_tool]},
            prompt_templates={Mode.EDA: "{schema}"},
            common_tools=None,
        )
        assert {t.name for t in fake_llm.bound_tools} == {"_eda_tool"}


class TestSummaryCutoff:
    def _convo(self, turns: int, per_turn: int):
        """构造 turns 个回合，每回合 1 条 Human + (per_turn-1) 条 AI，模拟工具密集程度。"""
        msgs = []
        for t in range(turns):
            msgs.append(HumanMessage(content=f"h{t}", id=f"h{t}"))
            for j in range(per_turn - 1):
                msgs.append(AIMessage(content=f"a{t}_{j}", id=f"a{t}_{j}"))
        return msgs

    def test_caps_retained_below_trigger_on_dense_turns(self):
        # #1：工具密集回合下，保留 keep=4 回合会超阈值 → 消息数上限进一步收紧
        msgs = self._convo(turns=6, per_turn=8)  # 48 条
        cut = summary_cutoff(msgs, keep=4, trigger=20)
        retained = msgs[cut:]
        assert len(retained) < 20  # 总结后必然低于阈值，不会下一轮立刻再触发
        assert isinstance(retained[0], HumanMessage)  # 截断落在 Human 边界

    def test_turn_budget_applies_when_light(self):
        # 轻量回合（未达阈值）：回合预算生效，保留最近 keep 个 Human 起
        msgs = self._convo(turns=6, per_turn=3)  # 18 条 < trigger
        cut = summary_cutoff(msgs, keep=4, trigger=20)
        humans = [m for m in msgs[cut:] if isinstance(m, HumanMessage)]
        assert len(humans) == 4

    def test_single_mega_turn_returns_zero(self):
        # #2 支撑：仅 1 个 Human 的巨型回合，无法在不割裂回合下压缩 → 0
        msgs = [HumanMessage(content="h", id="h")] + [
            AIMessage(content=f"a{i}", id=f"a{i}") for i in range(30)
        ]
        assert summary_cutoff(msgs, keep=4, trigger=20) == 0

    def test_no_deletion_within_keep(self):
        msgs = self._convo(turns=2, per_turn=2)  # 2 个 Human <= keep
        assert summary_cutoff(msgs, keep=4, trigger=20) == 0


class TestSummarizeConversation:
    def test_parses_json_into_structured_memory(self, make_fake_llm):
        llm = make_fake_llm(AIMessage(
            content='{"用户目标": "预测房价", "关键发现": ["f1"], "进展": "p1"}'
        ))
        state = {"messages": [HumanMessage(content="q", id="h1")]}
        out = summarize_conversation(
            state, llm=llm, graph_config=GraphConfig(summary_keep_recent=4)
        )
        assert out["memory"] == {
            "user_goal": "预测房价", "key_findings": ["f1"], "progress": "p1",
        }

    def test_unparseable_output_falls_back_to_progress(self, make_fake_llm):
        # 非 JSON 输出 → 回退：并入进展，不丢信息
        llm = make_fake_llm(AIMessage(content="这不是JSON"))
        state = {"messages": [HumanMessage(content="q", id="h1")]}
        out = summarize_conversation(
            state, llm=llm, graph_config=GraphConfig(summary_keep_recent=4)
        )
        assert out["memory"]["progress"] == "这不是JSON"

    def test_sticky_user_goal_preserved_on_merge(self, make_fake_llm):
        # 模型未给出新目标 → 沿用既有目标（锚定不漂移）
        llm = make_fake_llm(AIMessage(
            content='{"用户目标": "", "关键发现": ["新发现"], "进展": "新进展"}'
        ))
        state = {
            "messages": [HumanMessage(content="q", id="h1")],
            "memory": {"user_goal": "原目标", "key_findings": [], "progress": ""},
        }
        out = summarize_conversation(
            state, llm=llm, graph_config=GraphConfig(summary_keep_recent=4)
        )
        assert out["memory"]["user_goal"] == "原目标"
        assert out["memory"]["key_findings"] == ["新发现"]

    def test_key_findings_accumulate_and_dedup(self, make_fake_llm):
        llm = make_fake_llm(AIMessage(
            content='{"用户目标": "g", "关键发现": ["a", "b"], "进展": "p"}'
        ))
        state = {
            "messages": [HumanMessage(content="q", id="h1")],
            "memory": {"user_goal": "g", "key_findings": ["a"], "progress": ""},
        }
        out = summarize_conversation(
            state, llm=llm, graph_config=GraphConfig(summary_keep_recent=4)
        )
        assert out["memory"]["key_findings"] == ["a", "b"]

    def test_initial_prompt_branch(self, fake_llm):
        # 无记忆 → 使用 INITIAL 提示词（不含合并标记，占位符已消解）
        state = {"messages": [HumanMessage(content="q", id="h1")]}
        summarize_conversation(
            state, llm=fake_llm, graph_config=GraphConfig(summary_keep_recent=4)
        )
        prompt = fake_llm.invoked_with[-1].content
        assert "已有的结构化记忆" not in prompt
        assert "{prev}" not in prompt
        assert "用户目标" in prompt  # 结构化 JSON 模式说明

    def test_merge_prompt_branch(self, fake_llm):
        # 有记忆 → 使用 MERGE 提示词，既有记忆 JSON 被注入
        state = {
            "messages": [HumanMessage(content="q", id="h1")],
            "memory": {"user_goal": "旧目标", "key_findings": [], "progress": "旧的进展内容"},
        }
        summarize_conversation(
            state, llm=fake_llm, graph_config=GraphConfig(summary_keep_recent=4)
        )
        prompt = fake_llm.invoked_with[-1].content
        assert "已有的结构化记忆" in prompt
        assert "旧的进展内容" in prompt
        assert "{prev}" not in prompt

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
