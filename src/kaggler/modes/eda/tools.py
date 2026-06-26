# LangGraph/LangChain组件
from langgraph.prebuilt import InjectedState
from langchain_core.tools import BaseTool, tool

# 额外注释类型
from typing import Annotated

# 内部包引用
from kaggler.workspace.data_provider import DataProvider
from kaggler.modes.eda.compute import (
    get_correlation,
    get_schema_report,
    get_descriptive_statistics
)

# 序列化
import json

def make_eda_tools(data: DataProvider) -> list[BaseTool]:
    """

    Args:
        data:

    Returns:

    """
    @tool
    def explore_schema(state: Annotated[dict, InjectedState]) -> str:
        """
        获取数据集的完整结构信息：列名称、数据类型、缺失数量、唯一值数量和示例值。

        使用情景：
        - 用户要求获取列名、数据类型或者数据集结构
        - 需要进行统计分析但不确定具体列名或数据类型时，先调用此工具确认

        不要使用此工具进行实际的统计分析或分布计算。
        """
        df = data.get(state["data_version"])
        result = json.dumps(get_schema_report(df), ensure_ascii=False)
        return result

    def correlation_analysis(state: Annotated[dict, InjectedState], columns: list[str]) -> str:
        """
        分析指定列之间的相关性。自动根据列类型选择统计方法：
        - 连续 vs 连续：Pearson 相关系数（-1 到 1）
        - 分类 vs 分类：Cramér's V（0 到 1）
        - 分类 vs 连续：Eta²（0 到 1）

        columns 必须是精确列名，至少 2 列。

        使用情景：
        - 用户询问列之间的相关性、关联程度或相关系数
        - 用户想了解两个或多个变量之间的关系

        不要使用此工具进行描述性统计或分布分析。
        """
        df = data.get(state["data_version"])
        result = json.dumps(get_correlation(df, columns), ensure_ascii=False)
        return result

    def descriptive_analysis(state: Annotated[dict, InjectedState], columns: list[str]) -> str:
        """
        对指定列生成描述性统计。columns 必须是精确列名，且为数值类型。

        使用情景：
        - 用户询问具体统计量时，如均值、中位数、标准差或分位数

        不要使用此工具进行分布分析或相关性分析。
        """
        df = data.get(state["data_version"])
        result = json.dumps(get_descriptive_statistics(df, columns), ensure_ascii=False)
        return result

    return [explore_schema, correlation_analysis, descriptive_analysis]