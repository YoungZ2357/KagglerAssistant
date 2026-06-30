from enum import Enum


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