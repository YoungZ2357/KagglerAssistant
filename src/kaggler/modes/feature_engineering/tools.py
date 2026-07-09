from typing import Annotated

from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from kaggler.modes.feature_engineering.compute import (
    exec_dim_reduct,
    exec_drop_columns,
    exec_empty,
    exec_encode,
    exec_filter_rows,
    exec_standardize,
    exec_transform_combination,
    exec_transform_mono,
)
from kaggler.modes.feature_engineering.types import (
    CombineMethod,
    ConditionGroup,
    DimReductMethod,
    EncodePair,
    FillPair,
    MonoSpec,
    RowAction,
    RowLogic,
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

        缺失本身即信息：若某列的“是否缺失”本身可能与目标相关，请将该 pair 的
        add_indicator 设为 true。工具会在填充前先生成缺失标识列 <列名>_is_missing
        （1=原本缺失，0=非缺失），再执行填充，从而保留这一信号。该参数对 action=delete
        或本身无缺失的列无效（会跳过并在 summary 中说明）。

        使用情景：
        - 用户指定某些列存在空值并要求处理时
        - 用户可以混合使用多种填充方法，例如某列用均值、另一列删除
        - 当你拥有足够自主权，且认为需要对数据进行相关处理
        - 当缺失可能携带信息时，优先对相关列开启 add_indicator 再填充
        """
        df = data.get(state["data_version"])
        result = exec_empty(df, [p.model_dump(mode="json") for p in pairs])
        _indicated = [p.column for p in pairs if p.add_indicator]
        description = "空值处理: " + "; ".join(f"{p.column}→{p.action.value}" for p in pairs)
        if _indicated:
            description += f"（缺失标识列: {_indicated}）"
        return commit_mutation(
            data, result, tool_call_id,
            parent_version=state["data_version"],
            tool_name="execute_empty_value",
            description=description,
        )

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
        description = "编码: " + "; ".join(f"{p.column}→{p.action.value}" for p in pairs)
        return commit_mutation(
            data, result, tool_call_id,
            parent_version=state["data_version"],
            tool_name="encode_columns",
            description=description,
        )

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
        return commit_mutation(
            data, result, tool_call_id,
            parent_version=state["data_version"],
            tool_name="standardize_columns",
            description=f"标准化列: {columns}",
        )

    @tool
    def drop_columns(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        columns: list[str],
    ) -> Command:
        """删除指定的列。

        columns 是一个列名字符串列表，将从数据集中直接移除这些列。

        使用情景：
        - 用户明确要求删除某些不需要的列（如 ID、冗余或已被其他特征替代的列）时
        - 清理编码/降维前不需要的原始列
        - 注意：删除全部列会得到空数据集；该操作不改变行数（除非删空所有列）
        """
        df = data.get(state["data_version"])
        result = exec_drop_columns(df, columns)
        return commit_mutation(
            data, result, tool_call_id,
            parent_version=state["data_version"],
            tool_name="drop_columns",
            description=f"删除列: {columns}",
        )

    @tool
    def filter_rows(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        groups: list[ConditionGroup],
        group_logic: RowLogic,
        action: RowAction,
    ) -> Command:
        """根据条件筛选行，保留或删除满足条件的行。

        条件采用两层结构：
        - 每个 ConditionGroup 内部用 logic（and/or）组合若干叶子条件（column, op, value）
        - 多个 ConditionGroup 之间再用顶层 group_logic（and/or）组合
        例如 (age > 60 且 income < 1000) 或 (flag == "invalid")，应拆分为两个 group：
        [{"logic": "and", "conditions": [age>60, income<1000]}, {"logic": "and", "conditions": [flag=="invalid"]}]，
        并将顶层 group_logic 设为 or。

        action 决定整体语义：
        - "keep"：只保留组合条件为真的行
        - "delete"：删除组合条件为真的行，其余行（含条件涉及列为空值、无法判断真假的行）保留

        注意：条件值的类型必须与对应列的数据类型匹配（数值列传数字，字符串列传字符串，布尔列传布尔值）。
        注意：如果你缺乏必要信息，切换至eda模式并使用描述性数据分析
        使用情景：
        - 用户要求剔除异常值或明显错误的样本（如年龄为负数、某列超出合理范围）
        - 用户要求只保留满足特定业务条件的子集数据
        - 需要组合多个条件（且/或混合）来定位需要处理的行
        """
        df = data.get(state["data_version"])
        result = exec_filter_rows(
            df,
            groups=[g.model_dump(mode="json") for g in groups],
            group_logic=group_logic,
            action=action,
        )
        action_value = getattr(action, "value", action)
        group_logic_value = getattr(group_logic, "value", group_logic)
        return commit_mutation(
            data, result, tool_call_id,
            parent_version=state["data_version"],
            tool_name="filter_rows",
            description=f"按条件筛选行 (action={action_value}, group_logic={group_logic_value})",
        )

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
        method_value = getattr(method, "value", method)
        description = f"降维 method={method_value} n_components={n_components}"
        if target:
            description += f" target={target}"
        return commit_mutation(
            data, result, tool_call_id,
            parent_version=state["data_version"],
            tool_name="execute_dim_reduct",
            description=description,
        )

    @tool
    def transform_column_mono(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        specs: list[MonoSpec],
    ) -> Command:
        """对单个数值列应用一元变换，产出新特征列附加到数据中（保留原列，不替换）。

        专用于「只需一个原始特征」的新特征制作，例如线性模型的基函数变换和进阶特征工程。
        specs 是变换规格的列表，每个规格对一列应用一种变换并生成一个新列。支持的 method：
        - "cos"/"sin"/"tan"：三角函数
        - "exp"：指数 e^x
        - "log"：对数（默认自然对数；可用 base 指定底数）
        - "sqrt"：平方根
        - "square"：平方 x^2
        - "power"：幂运算 x^exponent（用 exponent 指定指数）
        - "linear"：线性变换 y = a*x + b（用 a 指定斜率、b 指定截距）
        - "reciprocal"：倒数 1/x
        - "abs"：绝对值

        每个规格可用 output_name 指定新列名；省略时自动生成（如 cos_<列名>）。
        注意：新列名不能与已有列或本批其它新列重名。变换若超出定义域（如对负数取对数、
        对 0 取倒数）会产生 NaN/无穷值，工具会照常执行并在 summary 中给出警告。

        使用情景：
        - 为线性/多项式模型构造基函数特征（如取平方、对数、三角变换）
        - 用户要求对某列做数学变换以改善其分布或线性关系时
        - 一次可对多列分别应用不同变换
        """
        df = data.get(state["data_version"])
        result = exec_transform_mono(
            df, [s.model_dump(mode="json") for s in specs]
        )
        description = "一元变换: " + "; ".join(
            f"{s.column}→{s.method.value}" for s in specs
        )
        return commit_mutation(
            data, result, tool_call_id,
            parent_version=state["data_version"],
            tool_name="transform_column_mono",
            description=description,
        )

    @tool
    def transform_column_combination(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        columns: list[str],
        method: CombineMethod,
        output_name: str,
    ) -> Command:
        """将多个数值列按算术方式组合为一个新特征列，附加到数据中（保留原列，不替换）。

        专用于构造交叉特征（cross feature）。columns 是参与组合的列名列表（至少 2 列），
        按传入顺序依次归约；output_name 为生成的新列名。支持的 method：
        - "product"：各列相乘（即交叉特征，col1 * col2 * ...）
        - "sum"：各列相加
        - "mean"：各列求平均
        - "difference"：依次相减（col1 - col2 - ...）
        - "ratio"：依次相除（col1 / col2 / ...）

        注意：所有列必须是数值类型；output_name 不能与已有列重名。ratio 遇到除零会产生
        无穷值，工具会照常执行并在 summary 中给出警告。

        使用情景：
        - 用户希望用两个或多个特征的乘积/比值等构造交叉特征时
        - 特征之间存在交互作用，需要显式建模（如 面积 = 长 * 宽、单价 = 总价 / 数量）
        """
        df = data.get(state["data_version"])
        result = exec_transform_combination(
            df,
            columns=columns,
            method=method,
            output_name=output_name,
        )
        method_value = getattr(method, "value", method)
        description = f"交叉特征: {columns} →({method_value}) {output_name}"
        return commit_mutation(
            data, result, tool_call_id,
            parent_version=state["data_version"],
            tool_name="transform_column_combination",
            description=description,
        )

    return [
        execute_empty_value,
        encode_columns,
        standardize_columns,
        drop_columns,
        filter_rows,
        execute_dim_reduct,
        transform_column_mono,
        transform_column_combination,
    ]
