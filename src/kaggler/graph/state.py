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

class CommonState(MessagesState):
    current_mode: Annotated[Mode, _take_latest]
    file_path: str
    explored_schema: str
    turn: Annotated[int, _add_turns]
    summary: str
    data_version: Annotated[int, _take_latest]
    # 最近一次 react invoke 的上下文 token 分类拆分（含校准系数），供 TUI 可视化。
    context_usage: Annotated[dict, _take_latest]


