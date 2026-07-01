from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langgraph.types import Command

from kaggler.shared.types import Mode


def make_tools() -> list[BaseTool]:

    @tool
    def switch_mode(
            new_mode: Mode,
            tool_call_id: Annotated[str, InjectedToolCallId]
            # state: Annotated[dict, InjectedState]
    ) -> Command:
        """切换当前工作模式（mode）。

        当用户的需求超出当前模式能力范围、需要进入另一能力切片时调用，
        例如从 EDA 探索切换到建模。new_mode 必须是受支持的模式枚举值。
        """
        return Command(update={
            "current_mode": new_mode,
            "messages": [
                ToolMessage(f"代理已经切换至新模式: {new_mode}", tool_call_id=tool_call_id),
            ]
        })
    return [switch_mode]