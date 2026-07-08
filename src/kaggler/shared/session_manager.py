"""会话管理器：将工作区、对话存储、SqliteSaver 组装为对话生命周期管理入口。

作为 composition root，本模块负责：
- 确定当前工作区（适用哪个 .kaggler 目录）
- 创建／恢复 AgentSession（含正确的 thread_id 与 SqliteSaver）
- 对话 CRUD（创建、列表、删除、重命名）
- LLM 自动命名：新对话未指定名称时，基于数据集特征自动生成
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import polars as pl
from langchain_core.messages import HumanMessage

from kaggler.graph.assembly import make_sqlite_saver
from kaggler.persistence.conversation_store import ConversationRecord, ConversationStore
from kaggler.shared.config import DeepSeekModel, make_llm_raw
from kaggler.shared.wrapper import AgentSession
from kaggler.workspace.manager import (
    Workspace,
    get_active_workspace,
    set_active_workspace,
)

_SAMPLE_ROWS = 5
_NAME_PROMPT = (
    "请为以下数据集生成一个简洁的中文名称（10字以内），描述其主题。"
    "仅返回名称本身，不要任何解释、标点或引号。\n"
    "文件名：{filename}\n"
    "列名：{columns}\n"
    "样本行：\n{sample}"
)


def _generate_name(csv_path: str) -> str:
    """基于 CSV 的结构与样本数据，调用轻量 LLM 生成对话名称。"""
    filename = Path(csv_path).stem
    try:
        df = pl.read_csv(csv_path)
    except Exception:
        return filename

    columns = ", ".join(df.columns[:15])
    if len(df.columns) > 15:
        columns += f" ...（共{len(df.columns)}列）"

    sample = "\n".join(
        str(row) for row in df.head(_SAMPLE_ROWS).rows()
    )

    prompt = _NAME_PROMPT.format(filename=filename, columns=columns, sample=sample)

    llm = make_llm_raw(DeepSeekModel.FLASH, temperature=0.3)
    response = llm.invoke([HumanMessage(content=prompt)])
    name = response.content if isinstance(response.content, str) else str(response.content)
    name = name.strip().strip("\"'").strip("。，,.；;：:！!？?")
    return name[:20] or filename


class SessionManager:
    """对话生命周期管理入口。

    用法::

        mgr = SessionManager("/path/to/workspace")
        session = mgr.create_conversation("data.csv")
        # 多轮问答...
        session = mgr.resume_conversation(session._config["configurable"]["thread_id"])
    """

    def __init__(self, workspace_path: Path | str | None = None) -> None:
        if workspace_path is not None:
            self._workspace = set_active_workspace(workspace_path)
        else:
            existing = get_active_workspace()
            self._workspace = existing or set_active_workspace(Path.cwd())

        self._store = ConversationStore(self._workspace.conversation_db)

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def create_conversation(
        self,
        csv_path: str,
        name: str | None = None,
    ) -> AgentSession:
        """创建新对话，返回可立即使用的 AgentSession。

        ``name`` 为 None 时自动调用 LLM 生成描述数据集主题的中文名称。
        """
        thread_id = uuid4().hex
        resolved_csv = str(Path(csv_path).resolve())
        workspace_path_str = str(self._workspace.path)

        if name is None:
            name = _generate_name(csv_path)

        self._store.create(
            name=name,
            thread_id=thread_id,
            csv_path=resolved_csv,
            workspace_path=workspace_path_str,
        )

        saver = make_sqlite_saver(self._workspace.checkpoint_db)
        return AgentSession(resolved_csv, thread_id=thread_id, checkpointer=saver)

    def resume_conversation(self, thread_id: str) -> AgentSession:
        """恢复已有对话，返回 AgentSession。

        由 SqliteSaver 按 thread_id 恢复完整 state（messages、summary 等），
        Agent 重启时自动获得对话历史的感知（通过 summary 字段注入 system prompt）。
        """
        record = self._store.get_by_thread_id(thread_id)
        if record is None:
            raise KeyError(f"未找到 thread_id={thread_id[:8]} 的对话记录")

        self._store.update_timestamp(thread_id)
        saver = make_sqlite_saver(self._workspace.checkpoint_db)
        return AgentSession(record.csv_path, thread_id=thread_id, checkpointer=saver)

    def list_conversations(self) -> list[ConversationRecord]:
        return self._store.list_all(str(self._workspace.path))

    def delete_conversation(self, thread_id: str) -> None:
        record = self._store.get_by_thread_id(thread_id)
        if record is None:
            raise KeyError(f"未找到 thread_id={thread_id[:8]} 的对话记录")

        self._store.delete(thread_id)

    def rename_conversation(self, thread_id: str, new_name: str) -> None:
        record = self._store.get_by_thread_id(thread_id)
        if record is None:
            raise KeyError(f"未找到 thread_id={thread_id[:8]} 的对话记录")

        self._store.rename(thread_id, new_name)
