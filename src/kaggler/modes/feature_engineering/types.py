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


class EncodePair(BaseModel):
    """编码的列-方法对。"""

    column: str = Field(description="列名")
    action: EncodeMethod = Field(description="编码方法")