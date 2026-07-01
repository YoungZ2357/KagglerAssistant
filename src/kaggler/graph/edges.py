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


def route_after_agent(state: CommonState) -> Literal[Node.TOOLS, Node.FINISH]:
    """react 之后路由：最后一条 AIMessage 带 tool_calls 则执行工具，否则进入收尾。"""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return Node.TOOLS
    return Node.FINISH
