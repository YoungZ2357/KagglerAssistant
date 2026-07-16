# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: KagglerAssistant
# FileName: edges.py
# Date: 2026/6/25 12:19
# -------------------------------------------------------------------------
from typing import Literal

from langchain_core.messages import AIMessage

from kaggler.graph.nodes import summary_cutoff
from kaggler.graph.state import CommonState
from kaggler.graph.types import Node
from kaggler.shared.config import GraphConfig


def entry_condition(
        state: CommonState,
        *,
        graph_config: GraphConfig,
) -> Literal[Node.SUMMARIZE, Node.REACT]:
    """turn 入口路由：历史消息数达到阈值且本次总结确有可删消息时先压缩，否则直接 react。

    ``graph_config`` 经 assembly 用 partial 绑定（仅关键字，避开 LangGraph
    保留名 ``config``）。返回 Node 成员而非裸字符串，使路由目标可被静态追溯。

    达阈值但 ``summary_cutoff`` 为 0（如单个进行中的巨型回合，无法在不割裂回合的
    前提下压缩）时跳过总结、直接 react，避免空转出一次无效的摘要 LLM 调用。
    """
    messages = state["messages"]
    if len(messages) < graph_config.summary_trigger_count:
        return Node.REACT
    cutoff = summary_cutoff(
        messages,
        keep=graph_config.summary_keep_recent,
        trigger=graph_config.summary_trigger_count,
    )
    return Node.SUMMARIZE if cutoff > 0 else Node.REACT


def route_after_agent(state: CommonState) -> Literal[Node.APPROVAL, Node.FINISH]:
    """react 之后路由：带 tool_calls 先过审批门（HITL 断点），否则进入收尾。

    审批门是无副作用节点：不需断点时直接放行到 TOOLS，需断点时 interrupt 暂停。
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return Node.APPROVAL
    return Node.FINISH


def route_after_approval(state: CommonState) -> Literal[Node.TOOLS, Node.REACT]:
    """审批门之后路由：仍有待执行的 tool_calls 则去 TOOLS，否则（全被拒）回 REACT。

    审批门在「拒绝」时会用同 id 覆盖那条 AIMessage、移除被拒调用；若因此不再有
    tool_calls，则回 REACT 让模型据正文中的拒绝说明重新规划。
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return Node.TOOLS
    return Node.REACT
