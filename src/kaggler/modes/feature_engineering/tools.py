from typing import Annotated

from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from kaggler.modes.feature_engineering.compute import (
    exec_dim_reduct,
    exec_empty,
    exec_encode,
    exec_standardize,
)
from kaggler.modes.feature_engineering.types import (
    DimReductMethod,
    EncodePair,
    FillPair,
)
from kaggler.persistence.data_provider import DataProvider
from kaggler.shared.tool_helpers import commit_mutation


def make_tools(data: DataProvider) -> list[BaseTool]:

    @tool
    def execute_empty_value(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        pairs: list[FillPair],
    ) -> Command:
        """对指定列执行空值填充或删除操作。如果你的信息有限，你可以尝试少量多次使用。

        pairs 是列-方法对的列表，每项指定一列的处理方式。支持的 action：
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
        result = exec_empty(df, [p.model_dump(mode="json") for p in pairs])
        return commit_mutation(data, result, tool_call_id)

    @tool
    def encode_columns(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        pairs: list[EncodePair],
    ) -> Command:
        """对指定列执行编码操作。如果你的信息有限，你可以尝试少量多次使用。

        pairs 是列-方法对的列表，每项指定一列的编码方式。支持的 action：
        - "one_hot"：独热编码，强制丢弃第一类（drop_first），n 个唯一值生成 n-1 列。
          如果唯一值过多会给出警告但仍然执行。
        - "label"：标签编码，将类别值映射为整数。

        使用情景：
        - 用户要求对分类/字符串列进行编码转换以便模型训练时
        - 你可以根据列的属性自行判断使用哪种编码（低基数用 one_hot，高基数用 label）
        - 一次可以同时对多列使用不同编码方法
        """
        df = data.get(state["data_version"])
        result = exec_encode(df, [p.model_dump(mode="json") for p in pairs])
        return commit_mutation(data, result, tool_call_id)

    @tool
    def standardize_columns(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        columns: list[str],
    ) -> Command:
        """对指定的数值列执行 z-score 标准化（均值=0，标准差=1）。

        columns 是一个列名字符串列表，所有列必须是数值类型且不含空值。
        标准化会改变列值的尺度和分布，使不同量纲的特征可以直接比较。

        使用情景：
        - 在降维（PCA/LDA）或建模前，将不同量纲的特征统一到同一尺度
        - 用户要求对某些列进行标准化时
        - 特征量纲差异较大时，有助于提升模型表现
        """
        df = data.get(state["data_version"])
        result = exec_standardize(df, columns)
        return commit_mutation(data, result, tool_call_id)

    @tool
    def execute_dim_reduct(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        method: DimReductMethod,
        n_components: int,
        target: str | None = None,
        standardize: bool = True,
    ) -> Command:
        """执行数据降维，将多个数值特征压缩为少数主成分或判别分量。

        支持两种方法，应用场景截然不同：
        - "pca"：无监督主成分分析。适用于无标签数据，用于探索数据结构、
          去噪、特征压缩和可视化。它寻找方差最大的投影方向，不利用任何标签信息。
        - "lda"：有监督线性判别分析。适用于有标签/目标列的分类数据，在有监督
          场景下寻找能最大化类间分离、最小化类内散布的投影方向。需要提供 target
          参数指定目标列，且目标列需为分类列（至少 2 个类别）。

        参数说明：
        - method: "pca" 或 "lda"
        - n_components: 降维后的维度数（正整数）
        - target: LDA 必需的目标列名，PCA 时忽略
        - standardize: 是否先对数值列做标准化，默认 True（推荐）

        注意事项：
        - 降维会替换原有的数值列为新生成的分量列（PC1, PC2, ... 或 LD1, LD2, ...）
        - 非数值列以及 LDA 的 target 列会被保留
        - 数值列中不能有 NaN，建议先使用 execute_empty_value 处理空值
        - PCA 的 n_components 不能超过数值列数
        - LDA 的 n_components 不能超过 min(类别数-1, 数值特征列数)

        使用情景：
        - 特征数量太多导致模型过拟合或训练缓慢时
        - 需要消除特征间的多重共线性时
        - 希望在保留主要信息的同时降低数据维度时
        - 数据有标签且希望利用标签信息优化降维效果时（选 lda）
        """
        df = data.get(state["data_version"])
        result = exec_dim_reduct(
            df,
            method=method,
            n_components=n_components,
            target=target,
            standardize=standardize,
        )
        return commit_mutation(data, result, tool_call_id)

    return [
        execute_empty_value,
        encode_columns,
        standardize_columns,
        execute_dim_reduct,
    ]
