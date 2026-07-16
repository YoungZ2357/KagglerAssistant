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
from pathlib import Path
from typing import Any, Iterator, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from kaggler.graph.assembly import build_graph
from kaggler.graph.types import Node
from kaggler.ir import IRNode, dumps_ir
from kaggler.shared.types import Mode
from kaggler.persistence.data_export import ExportResult, export_and_record
from kaggler.persistence.data_provider import DataProvider
from kaggler.persistence.pipeline_replay import rebuild_into
from kaggler.persistence.version_ledger_store import VersionLedgerStore, VersionRecord


class _LedgerSink:
    """DataProvider 的持久化端口实现：每次登记版本就开-写-关一次短连接。

    沿用 export_and_record 里「store 短生命周期」的约定——版本写入低频（一次/工具调用），
    无需 AgentSession 持有长连接与生命周期管理。
    """

    def __init__(self, db_path: Path, thread_id: str) -> None:
        self._db_path = db_path
        self._thread_id = thread_id

    def record_version(
        self,
        version: int,
        *,
        parent: int | None,
        kind: str,
        tool: str | None,
        description: str,
        reproducible: bool,
        ir: IRNode | None,
    ) -> None:
        store = VersionLedgerStore(self._db_path)
        try:
            store.record(
                thread_id=self._thread_id,
                version=version,
                parent=parent,
                kind=kind,
                tool=tool,
                description=description,
                reproducible=reproducible,
                ir=dumps_ir(ir) if ir is not None else None,
            )
        finally:
            store.close()


def _read_ledger(db_path: Path, thread_id: str) -> list[VersionRecord]:
    store = VersionLedgerStore(db_path)
    try:
        return store.list_by_thread(thread_id)
    finally:
        store.close()


class AgentSession:
    """单数据集、多轮对话的会话句柄。线程安全性由调用方保证（TUI 在单 worker 串行调用）。

    支持两种构造模式：
    - **新建对话**：不传 ``thread_id``，内部自动生成。
    - **恢复对话**：传入既有 ``thread_id``，由 SqliteSaver 按 thread 恢复状态。
      ``checkpointer`` 参数允许调用方显式指定持久化后端。
    """

    def __init__(
        self,
        csv_path: str,
        *,
        thread_id: Optional[str] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        version_ledger_db: Optional[Path] = None,
    ) -> None:
        self._csv_path = csv_path
        tid = thread_id or uuid4().hex
        self._config = {"configurable": {"thread_id": tid}}

        # 持久化端口：给了账本 DB 才落盘/可恢复；否则纯内存（CLI / 裸会话）。
        sink = _LedgerSink(version_ledger_db, tid) if version_ledger_db is not None else None
        data = DataProvider(sink=sink)

        # 有账本记录即为「恢复」——此判定在 load_initial 写入之前做，故可靠。
        records = _read_ledger(version_ledger_db, tid) if version_ledger_db is not None else []
        if records:
            rebuild_into(data, records)
            self._seeded = True  # 用 checkpoint 里的 data_version，勿再把种子打回 0
        else:
            data.load_initial(csv_path)  # 全新：经 sink 落 v0
            self._seeded = False

        self._data = data  # 供 /export 指令通道直达版本存储（工具通道走图内闭包）
        self._graph = build_graph(data, checkpointer=checkpointer)

        if records:
            # 把 HEAD 物化到 checkpoint 的当前版本（恢复点），使后续读取/分析即时可用。
            try:
                data.set_head(self.current_data_version)
            except RuntimeError:
                pass  # 恢复点异常缺失时不阻断会话，留待首个工具调用时报错

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

    def history(self) -> list[dict[str, str]]:
        """从 checkpoint 读出可展示的历史消息，供恢复对话时重绘左栏对话窗。

        只回放「用户提问」与「助手有正文的回复」两类，跳过 ToolMessage、system 与
        仅含 tool_calls 的空 AIMessage。**局限**：被 summarize 节点压缩掉的更早历史
        已从 messages 移除（仅存于结构化记忆 memory），故此处只能回放最近一段幸存的消息。
        """
        state = self._graph.get_state(self._config)
        out: list[dict[str, str]] = []
        for m in state.values.get("messages", []):
            if isinstance(m, HumanMessage):
                out.append({"role": "user", "content": str(m.content)})
            elif isinstance(m, AIMessage) and m.content:
                out.append({"role": "assistant", "content": str(m.content)})
        return out

    @property
    def thread_id(self) -> str:
        return self._config["configurable"]["thread_id"]

    @property
    def current_data_version(self) -> int:
        """当前工作数据版本;未播种/无 state 时回退到 0(load_initial 产生的 root)。"""
        return self._graph.get_state(self._config).values.get("data_version", 0)

    def export_data_version(
        self,
        version: int | None,
        target: Path,
        fmt: str | None = None,
        *,
        db_path: Path | None = None,
        description: str | None = None,
    ) -> ExportResult:
        """确定性导出(Channel A):把指定版本落盘到 target,并(若给 db_path)登记导出目录。

        version 为 None 时导出当前版本;description 缺省取该版本的谱系描述。
        供 TUI 的 /export 指令调用——指令通道有 AgentSession 句柄但够不到图内的 DataProvider,
        故在此暴露一个直达导出入口。
        """
        v = version if version is not None else self.current_data_version
        desc = description if description is not None else self._data.get_version_info(v).description
        return export_and_record(
            self._data, v, target, fmt,
            db_path=db_path,
            thread_id=self.thread_id,
            description=desc,
        )

    def set_mode(self, mode: Mode) -> None:
        """确定性切换模式（Channel A）：不经 LLM，直接写入图 state。

        供 TUI 的 ``/switch`` 指令调用。若尚未首轮播种，则一并注入种子字段并翻转
        ``_seeded``——否则下一次 ``_seed_payload`` 会再次注入 ``current_mode=EDA``
        覆盖本次切换。``MemorySaver`` 下 ``update_state`` 会在该 thread 上创建/续写
        checkpoint，之后 ``stream_events`` 从该 checkpoint 恢复，模式即生效。
        """
        values: dict = {"current_mode": mode}
        if not self._seeded:
            values |= {"file_path": self._csv_path, "data_version": 0}
            self._seeded = True
        self._graph.update_state(self._config, values)

    def stream_events(self, question: str) -> Iterator[dict[str, Any]]:
        """逐事件产出本轮过程，供 UI 同时驱动「回答」与「Agent 行为追溯」。

        事件类型：
        - ``{"type": "node_active", "node": str}``  —— 进入某节点（react/tools/…）
        - ``{"type": "token", "content": str}``     —— react 节点的回答文本片段
        - ``{"type": "node_done", "node": str, "tool_calls": list[dict]}``
                                                    —— 节点产出一批 state 更新
        - ``{"type": "context", "usage": dict}``    —— react 节点的上下文 token 分类
                                                       拆分（含校准系数），供占用面板可视化

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
                    # 单个节点在同一 super-step 内产生多条写入时，state_update 是
                    # list[dict] 而非 dict——典型触发：ToolNode 一次执行多个工具且其中
                    # 至少一个返回 Command（本项目的 switch_mode / switch_data_version）。
                    # 详见 langgraph ToolNode._combine_tool_outputs。统一成 list 再逐条处理，
                    # 否则对 list 调 .get 会抛 "'list' object has no attribute 'get'"。
                    updates = state_update if isinstance(state_update, list) else [state_update]
                    tool_calls: list[dict] = []
                    new_mode = None
                    context_usage = None
                    for upd in updates:
                        if not isinstance(upd, dict):
                            continue
                        for msg in upd.get("messages", []):
                            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                                tool_calls = [
                                    {"name": tc["name"], "args": tc["args"]}
                                    for tc in msg.tool_calls
                                ]
                        mode_upd = upd.get("current_mode")
                        if mode_upd is not None:
                            new_mode = mode_upd
                        cu = upd.get("context_usage")
                        if cu:
                            context_usage = cu
                    if new_mode is not None:
                        yield {"type": "mode_change", "mode": str(new_mode)}
                    if context_usage is not None:
                        yield {"type": "context", "usage": context_usage}
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
