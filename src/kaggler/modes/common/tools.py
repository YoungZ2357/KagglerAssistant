from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langgraph.types import Command

from kaggler.shared.types import Mode


def make_common_tools() -> list[BaseTool]:

    @tool
    def switch_mode(
            new_mode: Mode,
            tool_call_id: Annotated[dict, InjectedToolCallId]
            # state: Annotated[dict, InjectedState]
    ) -> Command:
        return Command(update={
            "current_mode": new_mode,
            "messages": [
                ToolMessage(f"代理已切换至新模式: {new_mode}", tool_call_id=tool_call_id),
            ]
        })
    return [switch_mode]