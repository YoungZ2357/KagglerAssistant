from enum import Enum

from pydantic import BaseModel, Field


class FillMethod(str, Enum):
    ZERO = "zero"
    MEDIAN = "median"
    AVG = "avg"
    MODE = "mode"
    DELETE = "delete"


class EncodeMethod(str, Enum):
    ONE_HOT = "one_hot"
    LABEL = "label"


class DimReductMethod(str, Enum):
    PCA = "pca"
    LDA = "lda"


class ComparisonOp(str, Enum):
    GT = "gt"
    LT = "lt"
    GE = "ge"
    LE = "le"
    EQ = "eq"
    NE = "ne"


class RowLogic(str, Enum):
    AND = "and"
    OR = "or"


class RowAction(str, Enum):
    KEEP = "keep"
    DELETE = "delete"


class MonoTransform(str, Enum):
    COS = "cos"
    SIN = "sin"
    TAN = "tan"
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"
    SQUARE = "square"
    POWER = "power"
    LINEAR = "linear"
    RECIPROCAL = "reciprocal"
    ABS = "abs"


class CombineMethod(str, Enum):
    PRODUCT = "product"  # 交叉特征：各列相乘
    SUM = "sum"
    MEAN = "mean"
    DIFFERENCE = "difference"  # col1 - col2 - ...
    RATIO = "ratio"  # col1 / col2 / ...


class Condition(BaseModel):
    """行筛选的单个条件。"""

    column: str = Field(description="列名")
    op: ComparisonOp = Field(
        description="比较运算符：gt(大于)/lt(小于)/ge(大于等于)/le(小于等于)/eq(等于)/ne(不等于)"
    )
    value: int | float | str | bool = Field(description="用于比较的值，类型需与列的数据类型匹配")


class ConditionGroup(BaseModel):
    """一组条件，组内条件通过 logic 组合。"""

    logic: RowLogic = Field(description="组内条件的组合方式：and(全部满足)/or(任一满足)")
    conditions: list[Condition] = Field(description="该组包含的条件列表")


class FillPair(BaseModel):
    """空值处理的列-方法对。"""

    column: str = Field(description="列名")
    action: FillMethod = Field(description="填充方法")
    add_indicator: bool = Field(
        default=False,
        description="是否在填充前先生成缺失标识列 <列名>_is_missing（1=原本缺失，0=非缺失），"
        "用于保留“缺失本身即信息”。仅在该列确有缺失且 action 非 delete 时生效。",
    )


class EncodePair(BaseModel):
    """编码的列-方法对。"""

    column: str = Field(description="列名")
    action: EncodeMethod = Field(description="编码方法")


class MonoSpec(BaseModel):
    """单列一元变换规格，每个规格产出一个新特征列。"""

    column: str = Field(description="源列名（必须是数值列）")
    method: MonoTransform = Field(description="一元变换方法")
    output_name: str | None = Field(
        default=None, description="新列名；省略时自动生成，如 cos_<列名>"
    )
    a: float = Field(default=1.0, description="linear 的斜率 y=a*x+b；其他方法忽略")
    b: float = Field(default=0.0, description="linear 的截距 y=a*x+b；其他方法忽略")
    exponent: float = Field(default=2.0, description="power 的指数；其他方法忽略")
    base: float | None = Field(
        default=None, description="log 的底数；省略时为自然对数；其他方法忽略"
    )