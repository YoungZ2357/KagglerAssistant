from enum import Enum


class FillMethod(str, Enum):
    ZERO = "zero"
    MEDIAN = "median"
    AVG = "avg"
    MODE = "mode"
    DELETE = "delete"