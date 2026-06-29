import polars as pl
import pytest
from langchain_core.messages import AIMessage, BaseMessage


class FakeChatModel:
    """不依赖网络的假 LLM：记录 bind_tools / invoke 的入参，返回预置回复。

    用于 react_node / summarize_conversation 测试——它们只调用
    ``llm.bind_tools(tools)`` 与 ``(.bind_tools 后的对象).invoke(messages)``，
    故无需是真正的 langchain Runnable，能记录入参以便断言即可。
    """

    def __init__(self, response: BaseMessage | None = None) -> None:
        self.response = response if response is not None else AIMessage(content="假回复")
        self.bound_tools: list | None = None
        self.invoked_with: list[BaseMessage] | None = None

    def bind_tools(self, tools: list) -> "FakeChatModel":
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        self.invoked_with = list(messages)
        return self.response


@pytest.fixture
def fake_llm() -> FakeChatModel:
    return FakeChatModel()


@pytest.fixture
def make_fake_llm():
    """工厂 fixture：按指定回复构造 FakeChatModel。"""

    def _make(response: BaseMessage | None = None) -> FakeChatModel:
        return FakeChatModel(response)

    return _make


@pytest.fixture
def csv_file(tmp_path) -> str:
    """写一个小 CSV 供 DataProvider.load_initial 真实读取（polars 内存读取，无网络）。"""
    path = tmp_path / "sample.csv"
    pl.DataFrame(
        {"id": [1, 2, 3], "name": ["a", "b", "c"], "score": [1.5, 2.5, 3.5]}
    ).write_csv(path)
    return str(path)


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
