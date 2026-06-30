from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from kaggler.graph.edges import entry_condition, route_after_agent
from kaggler.graph.nodes import react_node, summarize_conversation, finish_turn
from kaggler.graph.state import CommonState
from kaggler.graph.types import Node
from kaggler.modes.common.tools import make_tools as make_common_tools
from kaggler.modes.registry import REGISTRY
from kaggler.shared.config import GraphConfig, make_llm_raw, DeepSeekModel
from kaggler.shared.types import Mode
from kaggler.persistence.data_provider import DataProvider


def build_graph(
        data: DataProvider,
        *,
        graph_config: GraphConfig | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """据当前 DataProvider 组装并 compile ReAct 图，作为入口层的唯一公共接口。

    tools/prompts 在运行时据 REGISTRY + DataProvider 实例化（见 registry 文档）；
    所有依赖经 ``functools.partial`` 绑进节点/边，与 react_node 的注入风格一致。
    路由全部使用 ``Node`` 枚举 + list 形式 path_map（省掉自映射 dict）。
    """
    cfg = graph_config or GraphConfig()

    # 从仓库根目录的 .env 注入密钥（DEEPSEEK_API_KEY 等），使安装后用 `kaggler`
    # 命令从任意工作目录启动也能读到。不覆盖已存在的环境变量（dotenv 默认行为）。
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    # 运行时实例化：每个 mode 的工具与提示词；common 工具全 mode 可见
    tools_by_mode: dict[Mode, list[BaseTool]] = {
        mode: spec.tool_factory(data) for mode, spec in REGISTRY.items()
    }
    prompt_templates: dict[Mode, str] = {
        mode: spec.prompt for mode, spec in REGISTRY.items()
    }
    common_tools = make_common_tools()

    # 两档模型：react 用 PRO 主模型；压缩用更便宜的 FLASH，且不绑工具
    agent_llm = make_llm_raw(DeepSeekModel.PRO)
    summary_llm = make_llm_raw(DeepSeekModel.FLASH)

    # ToolNode 须囊括所有可能被调用的工具（common + 全 mode 并集）
    all_tools = [*common_tools, *(t for tools in tools_by_mode.values() for t in tools)]

    # PyCharm 对「继承自 typing_extensions.TypedDict 的子类」识别有限，
    # 此处 StateGraph(CommonState) 的告警为 IDE 假阳性，运行时 CommonState 是合法 TypedDict。
    builder = StateGraph(CommonState)

    builder.add_node(Node.REACT, partial(
        react_node,
        llm=agent_llm,
        tools_by_mode=tools_by_mode,
        prompt_templates=prompt_templates,
        common_tools=common_tools,
    ))
    builder.add_node(Node.TOOLS, ToolNode(all_tools, handle_tool_errors=True))
    builder.add_node(Node.SUMMARIZE, partial(
        summarize_conversation,
        llm=summary_llm,
        graph_config=cfg,
    ))
    builder.add_node(Node.FINISH, finish_turn)

    # 入口：消息数达阈值先压缩，否则直接 react。list 形式 path_map = 省掉 {"x": "x"} 自映射
    builder.add_conditional_edges(
        START, partial(entry_condition, graph_config=cfg), [Node.SUMMARIZE, Node.REACT],
    )
    builder.add_edge(Node.SUMMARIZE, Node.REACT)
    # react 后：带 tool_calls 去 tools，否则进入收尾节点
    builder.add_conditional_edges(Node.REACT, route_after_agent, [Node.TOOLS, Node.FINISH])
    builder.add_edge(Node.TOOLS, Node.REACT)
    builder.add_edge(Node.FINISH, END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
