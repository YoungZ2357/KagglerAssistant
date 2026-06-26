from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from kaggler.graph.state import CommonState
from kaggler.shared.types import Mode

def react_node(
        state: CommonState,
        *,
        llm: BaseChatModel,
        tools_by_mode: dict[Mode, list[BaseTool]],
        prompt_templates: dict[Mode, str],
        common_tools: list[BaseTool] | None = None,
) -> dict:
    mode = state["current_mode"]


    tools = [*(common_tools or []), *tools_by_mode[mode]]
    bound = llm.bind_tools(tools)

    # 系统提示词：每 turn 用当前 state 现填，str.replace 对 JSON 花括号免疫
    schema = state.get("explored_schema", "")
    system_text = prompt_templates[mode].replace("{schema}", schema)
    system = SystemMessage(content=system_text)

    # system 仅临时置于最前，用于本次 invoke；不写回 state、不进 messages 历史
    response = bound.invoke([system, *state["messages"]])

    # 只把 LLM 回复累积进 state（messages 的 reducer 负责 append）
    return {"messages": [response]}



