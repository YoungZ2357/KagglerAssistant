import polars as pl
import pytest


@pytest.fixture
def df_mixed() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "age": [25, 30, 35, 40, 45],
            "score": [88.5, 92.0, 76.3, 85.0, 90.1],
            "city": ["A", "B", "A", "B", "C"],
            "flag": ["X", "X", "Y", "Y", "Y"],
        }
    )


@pytest.fixture
def df_numeric_only() -> pl.DataFrame:
    return pl.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [5.0, 4.0, 3.0, 2.0, 1.0]}
    )


@pytest.fixture
def df_categorical_only() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cat_a": ["A", "A", "B", "B", "C"],
            "cat_b": ["X", "Y", "X", "Y", "X"],
        }
    )


@pytest.fixture
def df_with_nulls() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1.0, None, 3.0, float("nan"), float("inf")],
            "b": ["x", "y", None, "x", "y"],
        }
    )


@pytest.fixture
def df_empty() -> pl.DataFrame:
    return pl.DataFrame(schema={"x": pl.Float64, "y": pl.Utf8})
