# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: KagglerAssistant
# FileName: state.py
# Date: 2026/6/25 12:19
# -------------------------------------------------------------------------
from typing import Annotated

from langgraph.graph import MessagesState

from kaggler.shared.types import Mode

def _add_turns(current: int, update: int=1) -> int:
    return current + update

def _take_latest(current, update):
    """取最新写入值（后写覆盖先写）。

    无 reducer 的字段是 LastValue channel，同一 super-step 内出现多次写入
    （例如 LLM 一轮里同时调用两个 switch_mode / switch_data_version）会抛
    InvalidUpdateError。加此 reducer 后多写不再报错——单次写入行为与原先一致。
    """
    return update

def _upsert_by_id(current: list[dict] | None, update: list[dict] | None) -> list[dict]:
    """按 id upsert 合并列表通道（todos / plans 共用；逻辑与字段名无关）。

    - update 中 id 缺省（None）的项视为新建，由 reducer 依据 ``max(existing)+1`` 分配 id；
      同一 super-step 内多次 add 也不会撞号——LangGraph 对同一通道的多次写入逐条 fold，
      每次 fold 都基于最新的 current 重算下一个 id。
    - update 中带已知 id 的项与既有项合并字段（如 status → done，或仅改 content）；
      id 未知则按新建插入。未在 update 里出现的字段保持原值（部分更新）。

    这样容忍 LLM 一轮内多次调用同类工具（与 switch_mode 多写同源的问题）。
    """
    by_id: dict[int, dict] = {t["id"]: dict(t) for t in (current or [])}
    next_id = (max(by_id) + 1) if by_id else 1
    for u in (update or []):
        tid = u.get("id")
        if tid is None:
            tid = next_id
            next_id += 1
            by_id[tid] = {**u, "id": tid}
        elif tid in by_id:
            by_id[tid].update(u)
        else:
            by_id[tid] = {**u, "id": tid}
    return list(by_id.values())

class CommonState(MessagesState):
    current_mode: Annotated[Mode, _take_latest]
    file_path: str
    explored_schema: str
    turn: Annotated[int, _add_turns]
    # 结构化即时记忆（AgentMemory.to_dict()）：用户目标粘性锚定、关键发现累积、进展滚动压缩。
    memory: Annotated[dict, _take_latest]
    data_version: Annotated[int, _take_latest]
    # 待办挂起列表：{"id": int, "content": str, "status": "open"|"done"}。永不进摘要压缩。
    todos: Annotated[list[dict], _upsert_by_id]
    # 方案挂起列表：{"id": int, "title": str, "content": str,
    #   "status": "draft"|"active"|"archived"}。可反复修订，永不进摘要压缩。
    plans: Annotated[list[dict], _upsert_by_id]
    # 最近一次 react invoke 的上下文 token 分类拆分（含校准系数），供 TUI 可视化。
    context_usage: Annotated[dict, _take_latest]


