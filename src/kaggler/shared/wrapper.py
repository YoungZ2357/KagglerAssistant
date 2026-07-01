# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: KagglerAssistant
# FileName: wrapper.py
# Date: 2026/6/26 10:08
# -------------------------------------------------------------------------
"""高层会话封装：把「加载数据 → 建图 → 种子注入 → 流式问答」收口为一个对象，
让上层（TUI / CLI）保持纯 UI，不直接接触 graph 细节。

会话记忆由编译图的 checkpointer 按 ``thread_id`` 持久化，故仅首轮注入种子 state
（current_mode / file_path / data_version），后续仅传新问题——与 cli.py 同源。
"""
from typing import Any, Iterator
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from kaggler.graph.assembly import build_graph
from kaggler.graph.types import Node
from kaggler.shared.types import Mode
from kaggler.persistence.data_provider import DataProvider


class AgentSession:
    """单数据集、多轮对话的会话句柄。线程安全性由调用方保证（TUI 在单 worker 串行调用）。"""

    def __init__(self, csv_path: str) -> None:
        self._csv_path = csv_path
        data = DataProvider()
        data.load_initial(csv_path)
        self._graph = build_graph(data)
        self._config = {"configurable": {"thread_id": uuid4().hex}}
        self._seeded = False

    def _seed_payload(self, question: str) -> dict:
        """构造本轮入图 payload；种子 state 仅首轮注入，其余由 checkpointer 续写。"""
        payload: dict = {"messages": [HumanMessage(content=question)]}
        if not self._seeded:
            payload |= {
                "current_mode": Mode.EDA,
                "file_path": self._csv_path,
                "data_version": 0,
            }
            self._seeded = True
        return payload

    def stream_events(self, question: str) -> Iterator[dict[str, Any]]:
        """逐事件产出本轮过程，供 UI 同时驱动「回答」与「Agent 行为追溯」。

        事件类型：
        - ``{"type": "node_active", "node": str}``  —— 进入某节点（react/tools/…）
        - ``{"type": "token", "content": str}``     —— react 节点的回答文本片段
        - ``{"type": "node_done", "node": str, "tool_calls": list[dict]}``
                                                    —— 节点产出一批 state 更新

        仅透出节点流转与 tool_calls（Agent 决策了什么 / 调了什么工具）；**不**透出
        tool_result 等数据呈现内容——那是另一类需求，不在追溯范围内。
        """
        current_node: str | None = None
        for mode, data in self._graph.stream(
            self._seed_payload(question),
            config=self._config,
            stream_mode=["updates", "messages"],
        ):
            if mode == "messages":
                chunk, metadata = data
                node = metadata.get("langgraph_node")
                if node and node != current_node:
                    current_node = node
                    yield {"type": "node_active", "node": node}
                if (
                    node == Node.REACT
                    and isinstance(chunk, AIMessageChunk)
                    and chunk.content
                ):
                    yield {"type": "token", "content": chunk.content}
            elif mode == "updates":
                # updates 批次边界：清空当前节点，使下一次 messages 事件能重新报 active
                current_node = None
                for node_name, state_update in data.items():
                    tool_calls: list[dict] = []
                    for msg in (state_update or {}).get("messages", []):
                        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                            tool_calls = [
                                {"name": tc["name"], "args": tc["args"]}
                                for tc in msg.tool_calls
                            ]
                    new_mode = (state_update or {}).get("current_mode")
                    if new_mode is not None:
                        yield {"type": "mode_change", "mode": str(new_mode)}
                    yield {
                        "type": "node_done",
                        "node": node_name,
                        "tool_calls": tool_calls,
                    }

    def stream(self, question: str) -> Iterator[str]:
        """便捷封装：仅逐 token 产出回答文本（不关心追溯的调用方用它）。"""
        for ev in self.stream_events(question):
            if ev["type"] == "token":
                yield ev["content"]
