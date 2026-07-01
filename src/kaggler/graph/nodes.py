from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langchain_core.tools import BaseTool

from kaggler.graph.state import CommonState
from kaggler.shared.types import Mode
from kaggler.shared.config import GraphConfig

SUMMARY_TEMPLATE = (
    "[对话摘要]\n"
    "用户目标：<用户核心分析意图>\n"
    "已完成操作：<已调用的工具及结果要点，按时间顺序>\n"
    "关键发现：<分析中的重要结论>\n"
    "待处理：<未完成的任务或用户最新问题>"
)
# 首次生成摘要的指令。占位符 {template}。
SUMMARY_PROMPT_INITIAL = (
    "请严格按照以下固定模板格式（不得更改字段名称、不得添加额外标题或 Markdown 装饰），"
    "将以上对话总结为摘要，总长度不超过500字：\n{template}"
)

# 已有摘要时，将新增消息合并进摘要的指令。占位符 {summary}、{template}。
SUMMARY_PROMPT_MERGE = (
    "已有摘要：\n{summary}\n\n"
    "请严格按照以下固定模板格式（不得更改字段名称、不得添加额外标题或 Markdown 装饰），"
    "将新增消息合并进摘要，总长度不超过500字：\n{template}"
)

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

    # (a) 注入对话摘要：summarize 节点删除的旧历史经此回注 react，否则压缩=信息净丢失。
    summary = state.get("summary", "")
    if summary:
        system_text += f"\n\n[对话历史摘要]\n{summary}"

    # (b) 显式声明当前可调用工具集：切模式后绑定集已变，但历史/摘要里仍残留旧模式的
    # tool_calls 与结果，模型可能据此误调未绑定的工具。每 turn 现填当前工具名以正视听。
    tool_names = ", ".join(t.name for t in tools)
    system_text += (
        f"\n\n[当前可调用工具] 仅以下工具可用，切勿调用未列出的工具：{tool_names}"
    )

    system = SystemMessage(content=system_text)

    # system 仅临时置于最前，用于本次 invoke；不写回 state、不进 messages 历史
    response = bound.invoke([system, *state["messages"]])

    # 只把 LLM 回复累积进 state（messages 的 reducer 负责 append）
    return {"messages": [response]}



def summary_cutoff(messages: list, *, keep: int, trigger: int) -> int:
    """选定删除截断点（落在 HumanMessage 边界），返回索引 i，删除 ``messages[:i]``。

    保留的尾部 ``messages[i:]`` 同时满足两条约束（取更靠后的截断点 = 删得更多）：
    - **回合预算**：至多保留最近 ``keep`` 个 HumanMessage；
    - **消息数上限**：当消息数达到 ``trigger`` 时，进一步保证保留数 < ``trigger``，
      使总结后不会立刻再次触发（消除工具密集回合下的「每轮总结」）。

    截断点始终落在 HumanMessage 边界，不割裂 AIMessage(tool_calls) 与其 ToolMessage。
    返回 0 表示无需删除（含单个进行中的巨型回合——无法在不割裂回合的前提下压缩）。
    """
    human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    n = len(messages)

    # 回合预算：保留最近 keep 个 Human，其余可删
    turn_cutoff = human_indices[-keep] if len(human_indices) > keep else 0

    # 消息数上限：仅在达阈值时收紧，取「保留 < trigger」的最靠前 Human 边界（保留最多上下文）
    cap_cutoff = 0
    if n >= trigger and human_indices:
        cap_cutoff = human_indices[-1]  # 兜底：连最后一回合都 >= trigger 时只留它
        for i in human_indices:
            if n - i < trigger:
                cap_cutoff = i
                break

    return max(turn_cutoff, cap_cutoff)


def summarize_conversation(
        state: CommonState,
        *,
        llm: BaseChatModel,
        graph_config: GraphConfig,
) -> dict:
    """将历史对话压缩进 summary，并删除已被摘要覆盖的旧消息。

    依赖均为仅关键字注入，与 react_node 一致，由 assembly 经 partial 绑定：
    - llm：不绑工具的裸模型（总结不该触发工具调用）。
    - graph_config：避免使用保留名 ``config``，否则会被 LangGraph 误注入 RunnableConfig。

    删除区间由 ``summary_cutoff`` 决定：兼顾回合预算与消息数上限，截断点落在
    HumanMessage 边界。``entry_condition`` 已保证进入本节点时 cutoff > 0，不空转。
    """
    # 与 react_node 一致用 str.replace 填充占位符：对摘要/模板里的 JSON 花括号免疫。
    # 先填可信的 {template}，再填可能含杂质的 {summary}，避免后者内容被二次替换。
    summary = state.get("summary", "")
    if summary:
        prompt = (SUMMARY_PROMPT_MERGE
                  .replace("{template}", SUMMARY_TEMPLATE)
                  .replace("{summary}", summary))
    else:
        prompt = SUMMARY_PROMPT_INITIAL.replace("{template}", SUMMARY_TEMPLATE)

    # 完整历史 + 一条临时指令喂给模型；不写回 state
    response = llm.invoke([*state["messages"], HumanMessage(content=prompt)])

    cutoff = summary_cutoff(
        state["messages"],
        keep=graph_config.summary_keep_recent,
        trigger=graph_config.summary_trigger_count,
    )
    # m.id 由 checkpointer 赋值；缺 id 的消息无法 RemoveMessage，跳过以防构造非法删除
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:cutoff] if m.id]

    return {"summary": response.content, "messages": delete_messages}


def finish_turn(state: CommonState) -> dict:
    """每个 turn 的唯一确定性终点：累加轮数。

    ``turn`` 由 state 上的 ``_add_turns`` reducer 负责累加，故此处只需返回增量 1。
    作为单一收尾点，未来 turn 级逻辑（如统计、配额、跑题计数）都可挂在这里，
    无需散落到多条出边。
    """
    return {"turn": 1}


__all__ = ["summarize_conversation", "react_node", "finish_turn", "summary_cutoff"]