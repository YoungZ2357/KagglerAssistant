import polars as pl
from langchain_core.tools import BaseTool

from kaggler.modes.registry import REGISTRY, ModeSpec
from kaggler.shared.types import Mode
from kaggler.workspace.data_provider import DataProvider


class TestRegistry:
    def test_contains_eda_mode(self):
        assert Mode.EDA in REGISTRY
        assert isinstance(REGISTRY[Mode.EDA], ModeSpec)

    def test_prompt_has_schema_placeholder(self):
        assert "{schema}" in REGISTRY[Mode.EDA].prompt

    def test_tool_factory_produces_tools(self):
        data = DataProvider()
        data._frames[0] = pl.DataFrame({"a": [1, 2]})
        tools = REGISTRY[Mode.EDA].tool_factory(data)
        assert isinstance(tools, list)
        assert tools
        assert all(isinstance(t, BaseTool) for t in tools)
