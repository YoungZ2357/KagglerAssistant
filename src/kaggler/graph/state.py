# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: KagglerAssistant
# FileName: state.py
# Date: 2026/6/25 12:19
# -------------------------------------------------------------------------
from langgraph.graph import MessagesState

from typing import Annotated

from kaggler.shared.types import Mode

def _add_turns(current: int, update: int=1) -> int:
    return current + update

class CommonState(MessagesState):
    current_mode: Mode
    file_path: str
    explored_schema: str
    turn: Annotated[int, _add_turns]
    summary: str
    data_version: int


