import json
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from kaggler.modes.feature_engineering.compute import exec_empty
from kaggler.workspace.data_provider import DataProvider


def make_tools(data: DataProvider) -> list[BaseTool]:

    @tool
    def execute_empty_value(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        pairs: list[dict],
    ) -> Command:
        """对指定列执行空值填充或删除操作。

        pairs 是一个列表，每个元素为 {"column": <列名>, "action": <填充方法>}。
        支持的 action 值：
        - "zero"：用零值填充（数值填 0，字符串填 "0"，布尔填 False）
        - "avg"：用均值填充（仅限数值列）
        - "median"：用中位数填充（仅限数值列）
        - "mode"：用众数填充
        - "delete"：删除包含空值的行

        使用情景：
        - 用户指定某些列存在空值并要求处理时
        - 用户可以混合使用多种填充方法，例如某列用均值、另一列删除
        - 当你拥有足够自主权，且认为需要堆数据进行相关处理
        """
        df = data.get(state["data_version"])
        result = exec_empty(df, pairs)

        if "error" in result:
            return Command(update={
                "messages": [
                    ToolMessage(
                        json.dumps(result, ensure_ascii=False),
                        tool_call_id=tool_call_id,
                    )
                ],
            })

        new_version = data.add_version(result["processed_df"])
        return Command(update={
            "data_version": new_version,
            "messages": [
                ToolMessage(
                    json.dumps({
                        "new_data_version": new_version,
                        "rows_before": result["rows_before"],
                        "rows_after": result["rows_after"],
                        "preview": result["preview"],
                        "summary": result["summary"],
                    }, ensure_ascii=False),
                    tool_call_id=tool_call_id,
                ),
            ],
        })

    @tool
    def encode_columns() -> str:
        return "<PLACEHOLDER>"

    return [
        execute_empty_value,
        encode_columns,
    ]
