# -*- coding: utf-8 -*-
"""HITL 审批门测试。

策略层（needs_approval / 副作用分类）用纯函数直接断言；门节点行为用一张
「与真实拓扑同构、但 react/tools 为桩节点」的小图跑通——无需 DeepSeek，
覆盖：只读放行、高风险触发断点、批准/拒绝/始终允许的 resume、全局开关、
会话 allowlist 记忆。
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from kaggler.graph.edges import route_after_agent, route_after_approval
from kaggler.graph.hitl import (
    Effect,
    build_effects_map,
    make_approval_gate,
    mark_effects,
    needs_approval,
)
from kaggler.graph.nodes import finish_turn
from kaggler.graph.state import CommonState
from kaggler.graph.types import Node

RISKY = "drop_columns"
READONLY = "list_data_versions"
EFFECTS_MAP = {
    RISKY: frozenset({Effect.TRIGGERS_MATERIALIZE, Effect.WRITES_DISK}),
    READONLY: frozenset(),
}


# ── 策略纯函数 ──────────────────────────────────────────────────────────
def test_needs_approval_truth_table():
    empty = frozenset()
    assert needs_approval(RISKY, EFFECTS_MAP, allowlist=empty) is True
    assert needs_approval(READONLY, EFFECTS_MAP, allowlist=empty) is False
    # 未知工具（无声明）→ 不拦截
    assert needs_approval("unknown", EFFECTS_MAP, allowlist=empty) is False


def test_allowlist_suppresses_approval():
    # 两类副作用都进 allowlist → 不再需要审批
    al = frozenset({Effect.TRIGGERS_MATERIALIZE, Effect.WRITES_DISK})
    assert needs_approval(RISKY, EFFECTS_MAP, allowlist=al) is False
    # 只放行其中一类 → 仍需审批（还有未放行的受控副作用）
    partial = frozenset({Effect.WRITES_DISK})
    assert needs_approval(RISKY, EFFECTS_MAP, allowlist=partial) is True


def test_gated_subset_limits_scope():
    # 仅把 WRITES_DISK 纳入断点集：只触发 materialize 的工具不再被拦
    gated = frozenset({Effect.WRITES_DISK})
    only_mat = {"pca": frozenset({Effect.TRIGGERS_MATERIALIZE})}
    assert needs_approval("pca", only_mat, allowlist=frozenset(), gated=gated) is False


def test_mark_effects_and_build_map():
    from langchain_core.tools import tool

    @mark_effects(Effect.WRITES_DISK)
    @tool
    def w(x: int) -> int:
        "doc"
        return x

    @tool
    def r(x: int) -> int:
        "doc"
        return x

    em = build_effects_map([w, r])
    assert em["w"] == frozenset({Effect.WRITES_DISK})
    assert em["r"] == frozenset()


# ── 门节点：与真实拓扑同构的小图 ─────────────────────────────────────────
def _stub_react(state: CommonState) -> dict:
    """桩 react：产出一条无 tool_calls 的终结回复（使 route_after_agent → FINISH）。"""
    return {"messages": [AIMessage(content="完成")]}


def _stub_tools(state: CommonState) -> dict:
    """桩 tools：为最后一条 AIMessage 的每个 tool_call 产出一条 ToolMessage。"""
    last = state["messages"][-1]
    return {
        "messages": [
            ToolMessage(content=f"ran:{tc['name']}", tool_call_id=tc["id"])
            for tc in last.tool_calls
        ]
    }


def _build(hitl_enabled: bool = True, effects_map: dict | None = None):
    b = StateGraph(CommonState)
    b.add_node(Node.REACT, _stub_react)
    b.add_node(Node.APPROVAL, make_approval_gate(
        hitl_enabled=hitl_enabled,
        effects_map=effects_map if effects_map is not None else EFFECTS_MAP,
    ))
    b.add_node(Node.TOOLS, _stub_tools)
    b.add_node(Node.FINISH, finish_turn)
    # START 直达 APPROVAL：种子里已带一条 AIMessage(tool_calls) 作为待审批调用。
    b.add_edge(START, Node.APPROVAL)
    b.add_conditional_edges(Node.APPROVAL, route_after_approval, [Node.TOOLS, Node.REACT])
    b.add_edge(Node.TOOLS, Node.REACT)
    b.add_conditional_edges(Node.REACT, route_after_agent, [Node.APPROVAL, Node.FINISH])
    b.add_edge(Node.FINISH, END)
    return b.compile(checkpointer=MemorySaver())


def _seed(tool_name: str, call_id: str = "c1") -> dict:
    ai = AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": {"columns": ["x"]}, "id": call_id}],
    )
    return {"messages": [HumanMessage(content="go"), ai], "turn": 0}


def _cfg(tid: str) -> dict:
    return {"configurable": {"thread_id": tid}}


def _tool_msgs(state) -> list[ToolMessage]:
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


def test_readonly_passes_through():
    g = _build()
    cfg = _cfg("ro")
    state = g.invoke(_seed(READONLY), config=cfg)
    assert "__interrupt__" not in state
    assert g.get_state(cfg).next == ()  # 跑到 END
    assert [m.content for m in _tool_msgs(state)] == [f"ran:{READONLY}"]


def test_risky_triggers_interrupt():
    g = _build()
    cfg = _cfg("risk")
    state = g.invoke(_seed(RISKY), config=cfg)
    assert "__interrupt__" in state
    intr = state["__interrupt__"]
    pending = intr[0].value["pending"]
    assert pending[0]["name"] == RISKY
    assert set(pending[0]["effects"]) == {"triggers_materialize", "writes_disk"}
    assert g.get_state(cfg).next == (Node.APPROVAL,)
    # 断点期间工具未执行
    assert _tool_msgs(state) == []


def test_approve_resume_runs_tool():
    g = _build()
    cfg = _cfg("appr")
    g.invoke(_seed(RISKY), config=cfg)
    state = g.invoke(Command(resume={"action": "approve"}), config=cfg)
    assert "__interrupt__" not in state
    assert [m.content for m in _tool_msgs(state)] == [f"ran:{RISKY}"]
    assert g.get_state(cfg).next == ()


def test_reject_resume_skips_tool():
    g = _build()
    cfg = _cfg("rej")
    g.invoke(_seed(RISKY), config=cfg)
    state = g.invoke(Command(resume={"action": "reject"}), config=cfg)
    # 工具未执行；触发的 AIMessage 被同 id 覆盖、移除了 tool_calls，正文含拒绝说明
    assert _tool_msgs(state) == []
    ai = [m for m in state["messages"] if isinstance(m, AIMessage)]
    triggering = ai[0]
    assert not triggering.tool_calls
    assert "拒绝" in triggering.content
    assert g.get_state(cfg).next == ()


def test_always_allow_records_allowlist():
    g = _build()
    cfg = _cfg("always")
    g.invoke(_seed(RISKY), config=cfg)
    state = g.invoke(Command(resume={"action": "always"}), config=cfg)
    # 放行并执行；副作用写入会话 allowlist
    assert [m.content for m in _tool_msgs(state)] == [f"ran:{RISKY}"]
    assert set(state["hitl_allowlist"]) == {"triggers_materialize", "writes_disk"}
    # 记忆生效：带该 allowlist 后同类不再需要审批
    al = frozenset(Effect(e) for e in state["hitl_allowlist"])
    assert needs_approval(RISKY, EFFECTS_MAP, allowlist=al) is False


def test_global_toggle_off_never_interrupts():
    g = _build(hitl_enabled=False)
    cfg = _cfg("off")
    state = g.invoke(_seed(RISKY), config=cfg)
    assert "__interrupt__" not in state
    assert [m.content for m in _tool_msgs(state)] == [f"ran:{RISKY}"]


def test_seeded_allowlist_suppresses_interrupt():
    g = _build()
    cfg = _cfg("preallow")
    seed = _seed(RISKY)
    seed["hitl_allowlist"] = ["triggers_materialize", "writes_disk"]
    state = g.invoke(seed, config=cfg)
    assert "__interrupt__" not in state
    assert [m.content for m in _tool_msgs(state)] == [f"ran:{RISKY}"]
