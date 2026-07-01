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


class FillPair(BaseModel):
    """空值处理的列-方法对。"""

    column: str = Field(description="列名")
    action: FillMethod = Field(description="填充方法")


class EncodePair(BaseModel):
    """编码的列-方法对。"""

    column: str = Field(description="列名")
    action: EncodeMethod = Field(description="编码方法")