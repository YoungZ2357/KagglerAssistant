import json
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from kaggler.modes.feature_engineering.compute import exec_empty, exec_encode
from kaggler.persistence.data_provider import DataProvider


def make_tools(data: DataProvider) -> list[BaseTool]:

    @tool
    def execute_empty_value(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        pairs: list[dict],
    ) -> Command:
        """对指定列执行空值填充或删除操作。如果你的信息有限，你可以尝试少量多次使用。

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
        - 当你拥有足够自主权，且认为需要对数据进行相关处理
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
    def encode_columns(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        pairs: list[dict],
    ) -> Command:
        """对指定列执行编码操作。如果你的信息有限，你可以尝试少量多次使用。

        pairs 是一个列表，每个元素为 {"column": <列名>, "action": <编码方法>}。
        支持的 action 值：
        - "one_hot"：独热编码，强制丢弃第一类（drop_first），n 个唯一值生成 n-1 列。
          如果唯一值过多会给出警告但仍然执行。
        - "label"：标签编码，将类别值映射为整数。

        使用情景：
        - 用户要求对分类/字符串列进行编码转换以便模型训练时
        - 你可以根据列的属性自行判断使用哪种编码（低基数用 one_hot，高基数用 label）
        - 一次可以同时对多列使用不同编码方法
        """
        df = data.get(state["data_version"])
        result = exec_encode(df, pairs)

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

    return [
        execute_empty_value,
        encode_columns,
    ]
