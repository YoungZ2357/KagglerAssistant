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
from typing import Iterator
from uuid import uuid4

from langchain_core.messages import AIMessageChunk, HumanMessage

from kaggler.graph.assembly import build_graph
from kaggler.graph.types import Node
from kaggler.shared.types import Mode
from kaggler.workspace.data_provider import DataProvider


class AgentSession:
    """单数据集、多轮对话的会话句柄。线程安全性由调用方保证（TUI 在单 worker 串行调用）。"""

    def __init__(self, csv_path: str) -> None:
        self._csv_path = csv_path
        data = DataProvider()
        data.load_initial(csv_path)
        self._graph = build_graph(data)
        self._config = {"configurable": {"thread_id": uuid4().hex}}
        self._seeded = False

    def stream(self, question: str) -> Iterator[str]:
        """逐 token 产出本轮回答的文本片段。

        只透出 react 节点的文本 token：仅带 ``tool_calls``、content 为空的中间轮
        会被 ``chunk.content`` 过滤掉，因此工具调用不外泄给 UI。
        """
        payload: dict = {"messages": [HumanMessage(content=question)]}
        if not self._seeded:
            # 种子 state 仅首轮注入；其余 channel 由 checkpointer 按 thread_id 续写
            payload |= {
                "current_mode": Mode.EDA,
                "file_path": self._csv_path,
                "data_version": 0,
            }
            self._seeded = True

        for chunk, metadata in self._graph.stream(
            payload, config=self._config, stream_mode="messages"
        ):
            if (
                metadata.get("langgraph_node") == Node.REACT
                and isinstance(chunk, AIMessageChunk)
                and chunk.content
            ):
                yield chunk.content
