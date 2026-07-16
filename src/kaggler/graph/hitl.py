# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: KagglerAssistant
# FileName: hitl.py
# -------------------------------------------------------------------------
"""Human-in-the-loop（HITL）断点：在高风险工具调用执行前暂停请求人工审批。

设计（见 docs 与 plan）：
- **按副作用分类**：每个工具用 ``metadata["effects"]`` 声明自己的副作用标签
  （见 ``mark_effects``）；策略据标签而非工具名判定是否需要断点，故新增工具只要
  声明副作用即自动纳入，无需维护中心白名单。
- **图层 interrupt()**：断点是一个位于 ``REACT`` 与 ``TOOLS`` 之间的**无副作用**门
  节点（``make_approval_gate``）。它只读最后一条 AIMessage 的 tool_calls、按策略
  分流、必要时 ``interrupt()`` 暂停。因无副作用，LangGraph 在 resume 时重跑该节点
  完全安全——这也是为什么断点放在 TOOLS **之前**而非塞进各工具内部
  （在 ToolNode 内部 interrupt 会导致同一 super-step 里已完成的工具被重跑）。
- **只拦「系统以外」触发**：interrupt 只在图节点边界触发，而节点边界恰好等价于
  Agent/用户发起的工具调用。DataProvider 内部的读时重算 / LRU 驱逐永不跨越该边界，
  故天然被排除，无需改动 DataProvider。

resume 契约：TUI/CLI 用 ``Command(resume=decision)`` 恢复，``decision`` 形如
``{"action": "approve" | "reject" | "always"}``：
- ``approve``：本次放行待批调用；
- ``always`` ：放行并把这些调用命中的副作用并入会话 allowlist（此后同类不再断点）；
- ``reject`` ：从触发的 AIMessage 中移除被拒调用（同 id 覆盖），并在其正文追加一行
  拒绝说明供模型知悉——**不**产生游离 ToolMessage（ToolNode 不会跳过已应答的调用，
  见实测；保留被拒调用只会被再次执行）。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.types import interrupt

from kaggler.graph.state import CommonState


class Effect(str, Enum):
    """工具副作用标签（与 Node / Mode 同为 str-Enum 风格，取值即 JSON 友好字符串）。

    对应用户请求的三类高风险行为：
    - WRITES_DISK          —— 磁盘写操作（导出 CSV/parquet/.py、落 IR 到账本）
    - TRIGGERS_MATERIALIZE —— 外部触发的 materialize/惰性化存储（新版本落地、切头重物化）
    - MUTATES_VERSION      —— 数据版本变更（切换/回滚工作指针）
    """

    WRITES_DISK = "writes_disk"
    TRIGGERS_MATERIALIZE = "triggers_materialize"
    MUTATES_VERSION = "mutates_version"


# 默认纳入断点的副作用集合：全部三类。GraphConfig 可裁剪（见 gated_effects）。
DEFAULT_GATED_EFFECTS: frozenset[Effect] = frozenset(Effect)

# interrupt payload / resume decision 的类型别名（结构化字典，跨线程/进程传递安全）。
ApprovalPayload = dict[str, Any]
ApprovalDecision = dict[str, Any]


def mark_effects(*effects: Effect) -> Callable[[BaseTool], BaseTool]:
    """把副作用标签写入工具的 ``metadata["effects"]``。用作 ``@tool`` 之上的装饰器：

    >>> @mark_effects(Effect.WRITES_DISK)
    ... @tool
    ... def export_data_version(...): ...

    装饰器自下而上应用：``@tool`` 先产出 StructuredTool，本装饰器再给它挂 metadata。
    """

    frozen = frozenset(effects)

    def deco(tool_obj: BaseTool) -> BaseTool:
        tool_obj.metadata = {**(tool_obj.metadata or {}), "effects": frozen}
        return tool_obj

    return deco


def effects_of(tool_obj: BaseTool) -> frozenset[Effect]:
    """读取一个工具声明的副作用标签集合（未声明则为空集）。"""
    return frozenset((tool_obj.metadata or {}).get("effects", ()))


def build_effects_map(tools: list[BaseTool]) -> dict[str, frozenset[Effect]]:
    """从工具列表汇总 ``工具名 → 副作用集合`` 映射，供门节点判定。"""
    return {t.name: effects_of(t) for t in tools}


def _gated_effects(
    tool_effects: frozenset[Effect],
    gated: frozenset[Effect],
) -> frozenset[Effect]:
    """某工具命中的、且属于「需断点」集合的副作用。空集表示该工具无需审批。"""
    return tool_effects & gated


def needs_approval(
    tool_name: str,
    effects_map: dict[str, frozenset[Effect]],
    *,
    allowlist: frozenset[Effect],
    gated: frozenset[Effect] = DEFAULT_GATED_EFFECTS,
) -> bool:
    """判定某工具调用是否需要人工审批。

    需审批 ⇔ 该工具命中的受控副作用非空，且其中存在尚未被会话 allowlist 放行的项。
    （allowlist 承载「本会话始终允许此类」的记忆。）
    """
    gated_hit = _gated_effects(effects_map.get(tool_name, frozenset()), gated)
    return bool(gated_hit) and not gated_hit <= allowlist


def make_approval_gate(
    *,
    hitl_enabled: bool,
    effects_map: dict[str, frozenset[Effect]],
    gated: frozenset[Effect] = DEFAULT_GATED_EFFECTS,
) -> Callable[[CommonState], dict]:
    """构造审批门节点（闭包注入配置与副作用映射，仿 react_node 的注入风格）。

    返回的节点：
    1. 取最后一条 AIMessage 的 tool_calls，按策略分出「待批」子集；
    2. 无待批 → 返回 {}（放行到 TOOLS）；
    3. 有待批 → ``interrupt()`` 暂停，等 ``Command(resume=decision)``；
    4. resume 后据 decision 构造 state 更新（见模块文档的 resume 契约）。
    """

    def approval_gate(state: CommonState) -> dict:
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", None) or [])
        if not (hitl_enabled and calls):
            return {}

        allowlist = frozenset(
            Effect(e) for e in (state.get("hitl_allowlist") or [])
        )
        pending = [
            {
                "id": tc["id"],
                "name": tc["name"],
                "args": tc.get("args", {}),
                "effects": sorted(
                    e.value
                    for e in _gated_effects(
                        effects_map.get(tc["name"], frozenset()), gated
                    )
                ),
            }
            for tc in calls
            if needs_approval(tc["name"], effects_map, allowlist=allowlist, gated=gated)
        ]
        if not pending:
            return {}

        decision: ApprovalDecision = interrupt({"pending": pending})
        return _apply_decision(last, calls, pending, decision, gated, effects_map)

    return approval_gate


def _apply_decision(
    triggering: AIMessage,
    calls: list[dict],
    pending: list[dict],
    decision: ApprovalDecision,
    gated: frozenset[Effect],
    effects_map: dict[str, frozenset[Effect]],
) -> dict:
    """据 resume 决策构造 state 更新（纯函数，便于单测）。"""
    action = (decision or {}).get("action", "reject")
    pending_ids = {p["id"] for p in pending}

    if action in ("approve", "always"):
        update: dict = {}
        if action == "always":
            # 把待批调用命中的受控副作用并入会话 allowlist（去重、存字符串值）。
            allow: set[str] = set()
            for p in pending:
                allow.update(p["effects"])
            if allow:
                update["hitl_allowlist"] = sorted(allow)
        return update

    # reject：移除被拒调用（= 全部待批），保留其余调用；不产生游离 ToolMessage。
    kept = [tc for tc in calls if tc["id"] not in pending_ids]
    rejected_names = [p["name"] for p in pending]
    note = "\n（注：以下高风险操作已被用户拒绝，未执行：" + "、".join(rejected_names) + "）"
    new_ai = AIMessage(
        content=(triggering.content or "") + note,
        tool_calls=kept,
        id=triggering.id,
    )
    return {"messages": [new_ai]}


__all__ = [
    "Effect",
    "DEFAULT_GATED_EFFECTS",
    "ApprovalPayload",
    "ApprovalDecision",
    "mark_effects",
    "effects_of",
    "build_effects_map",
    "needs_approval",
    "make_approval_gate",
]
