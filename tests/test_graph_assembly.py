from unittest.mock import MagicMock

import polars as pl
import pytest
from langgraph.checkpoint.memory import MemorySaver

from kaggler.graph.assembly import build_graph
from kaggler.graph.types import Node
from kaggler.shared.config import GraphConfig
from kaggler.persistence.data_provider import DataProvider


@pytest.fixture
def loaded_data() -> DataProvider:
    data = DataProvider()
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    data.add_source(lambda: df, description="test")
    return data


@pytest.fixture(autouse=True)
def _no_real_llm(mocker):
    """拦截 LLM 工厂，绝不实例化真实 ChatDeepSeek（无网络）。"""
    # make_llm_raw 在 assembly 模块命名空间内引用，故在该处 patch
    return mocker.patch(
        "kaggler.graph.assembly.make_llm_raw", return_value=MagicMock()
    )


class TestBuildGraph:
    def test_returns_compiled_graph_with_expected_nodes(self, loaded_data):
        graph = build_graph(loaded_data)
        node_values = {
            n.value if isinstance(n, Node) else n
            for n in graph.get_graph().nodes.keys()
        }
        assert {"react", "tools", "summarize", "finish"} <= node_values

    def test_uses_two_llm_tiers(self, loaded_data, _no_real_llm):
        build_graph(loaded_data)
        # react 用 PRO、summary 用 FLASH —— 工厂被调用两次
        assert _no_real_llm.call_count == 2

    def test_custom_checkpointer_used(self, loaded_data):
        saver = MemorySaver()
        graph = build_graph(loaded_data, checkpointer=saver)
        assert graph.checkpointer is saver

    def test_accepts_custom_graph_config(self, loaded_data):
        # 自定义配置不应导致组装失败
        graph = build_graph(loaded_data, graph_config=GraphConfig(summary_trigger_count=99))
        assert graph is not None

    def test_common_tools_include_data_version_tools(self, loaded_data):
        graph = build_graph(loaded_data)
        tool_node = graph.get_graph().nodes[Node.TOOLS.value].data
        names = set(tool_node.tools_by_name.keys())
        assert {"switch_mode", "switch_data_version", "list_data_versions"} <= names
