from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

from kaggler.graph.state import CommonState
from kaggler.shared.types import Mode

def react_node(
        state: CommonState,
        *,
        llm: BaseChatModel,
        tools_by_mode: dict[Mode, list[BaseTool]],
        prompt_templates: dict[Mode, str],
) -> dict:
    mode = state["current_mode"]

    # tools_by_mode已经包含了所有的通用工具，不要在这里注入
    tools = [*tools_by_mode[mode]]
    bound = llm.bind_tools(tools)

    # 系统提示词：每 turn 用当前 state 现填，str.replace 对 JSON 花括号免疫
    schema = state.get("explored_schema", "")
    system_text = prompt_templates[mode].replace("{schema}", schema)
    system = SystemMessage(content=system_text)

    # system随调用注入，为维护状态值整洁不返回历史的system prompt对应记录
    response = bound.invoke([system, *state["messages"]])

    # 只把 LLM 回复累积进 state（messages 的 reducer 负责 append）
    return {"messages": [response]}



