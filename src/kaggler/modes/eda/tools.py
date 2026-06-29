import json
from typing import Annotated, Literal

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from kaggler.modes.eda.compute import (
    distribution_evaluation,
    get_boxed_data,
    get_correlation,
    get_descriptive_statistics,
    get_schema_report,
)
from kaggler.workspace.data_provider import DataProvider

def make_tools(data: DataProvider) -> list[BaseTool]:
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

    @tool
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

    @tool
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


    @tool
    def distribution_analysis_raw(state: Annotated[dict, InjectedState], column: str) -> str:
        """
        分析指定列的分布情况。数值列返回分箱统计，分类列返回频率表。

        - 用户询问某列的分布、频率、最常出现的值或者取值范围

        不要使用此工具进行描述性统计（均值、标准差等）或相关性分析。
        """
        df = data.get(state["data_version"])
        result = json.dumps(get_boxed_data(df, column), ensure_ascii=False)
        return result

    @tool
    def distribution_fit(
        state: Annotated[dict, InjectedState],
        column: str,
        method: Literal["chi2", "monte_carlo"] = "chi2",
    ) -> str:
        """
        检验某个数值列是否服从常见分布（正态、均匀、指数、对数正态、伽马）。
        返回分箱观测数据 + 各候选分布的拟合优度（检验统计量、p 值、估计参数）。

        使用情景：
        - 用户询问某列是否服从正态分布 / 某种分布，或需要判断分布形态以选择建模/变换策略

        参数 method：
        - "chi2"（默认）：基于分箱的卡方拟合优度，毫秒级，绝大多数情况下用它即可。
        - "monte_carlo"：KS + 蒙特卡洛 p 值，统计上更严格但慢得多（大列可达数十秒），
          仅当用户明确要求更严格/更精确的检验时才使用。

        说明：候选分布固定由后端决定，调用方只需给出列名（必要时指定 method）；
        p 值越大越无法拒绝该分布假设。不要使用此工具进行描述性统计或相关性分析。
        """
        df = data.get(state["data_version"])
        result = json.dumps(
            distribution_evaluation(df, column, method=method), ensure_ascii=False
        )
        return result

    return [
        explore_schema,
        correlation_analysis,
        descriptive_analysis,
        distribution_analysis_raw,
        distribution_fit,
    ]