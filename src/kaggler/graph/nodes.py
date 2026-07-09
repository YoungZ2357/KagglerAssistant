import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langchain_core.tools import BaseTool

from kaggler.graph.memory import (
    AgentMemory,
    merge_memory,
    parse_memory,
    render_memory,
)
from kaggler.graph.state import CommonState
from kaggler.shared.types import Mode
from kaggler.shared.config import GraphConfig
from kaggler.shared.token_estimate import build_breakdown, next_calibration_factor

# 要求 FLASH 输出结构化 JSON 记忆的模式说明（三段：用户目标 / 关键发现 / 进展）。
# 用字符串拼接而非 .format——正文含字面量花括号，不能被再次解析。
SUMMARY_SCHEMA_HINT = (
    "请仅输出一个 JSON 对象（不要任何解释文字、不要 Markdown 代码围栏），"
    "包含且仅包含以下三个键：\n"
    "{\n"
    '  "用户目标": "<用户核心分析意图，一句话；除非用户目标明确改变，请原样保留既有目标>",\n'
    '  "关键发现": ["<分析中得到的重要结论，每条一句话>"],\n'
    '  "进展": "<已完成的关键操作与其结果要点，按时间顺序的简短叙事，不超过300字>"\n'
    "}"
)
# 首次生成记忆的指令。
SUMMARY_PROMPT_INITIAL = "请将以上对话压缩为结构化记忆。\n" + SUMMARY_SCHEMA_HINT

# 已有记忆时，将新增消息合并进记忆的指令。占位符 {prev}（既有记忆 JSON）。
SUMMARY_PROMPT_MERGE = (
    "以下是已有的结构化记忆（JSON）：\n{prev}\n\n"
    "请把新增对话合并进来，输出更新后的结构化记忆。"
    "「关键发现」应在既有基础上补充新增结论（不要删除仍然成立的旧结论）；"
    "「用户目标」除非明确改变否则原样保留。\n" + SUMMARY_SCHEMA_HINT
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
    base_text = prompt_templates[mode].replace("{schema}", schema)

    # (a) 注入结构化记忆：summarize 节点删除的旧历史经此回注 react，否则压缩=信息净丢失。
    # 记忆仅对 Agent 可见——Agent 应在回复中自然体现对上下文的了解（如引用之前的分析结论），
    # 但除非被明确问到，不应向用户逐字复述记忆内容。
    mem = AgentMemory.from_dict(state.get("memory") or {})
    memory_block = ""
    if not mem.is_empty():
        memory_block = (
            f"\n\n[Agent对之前对话的已知信息]\n"
            f"以下是你与用户之前对话的压缩记忆。你应当基于此理解上下文、"
            f"延续之前的分析，但不要向用户逐字复述这些内容，"
            f"除非用户明确要求你回顾之前的操作：\n{render_memory(mem)}"
        )

    # (b) 显式声明当前可调用工具集：切模式后绑定集已变，但历史/记忆里仍残留旧模式的
    # tool_calls 与结果，模型可能据此误调未绑定的工具。每 turn 现填当前工具名以正视听。
    tool_names = ", ".join(t.name for t in tools)
    tools_block = (
        f"\n\n[当前可调用工具] 仅以下工具可用，切勿调用未列出的工具：{tool_names}"
    )

    # (c) 注入待办：每 turn 逐字回注、永不进摘要压缩，确保 Agent 不遗忘自己挂起的建议。
    todos_block = _render_todos_block(state.get("todos") or [])

    # (d) 注入方案：与待办同源——每 turn 逐字回注、永不压缩，承载尚未定型但重要的规划性思考。
    plans_block = _render_plans_block(state.get("plans") or [])

    system_text = base_text + memory_block + tools_block + todos_block + plans_block
    system = SystemMessage(content=system_text)

    # 上下文占用估算（离线，未乘校准系数）：系统提示词 = 模板 + 工具名块 + 待办块 + 方案块，记忆块单列。
    prev_factor = (state.get("context_usage") or {}).get("calibration_factor", 1.0)
    breakdown = build_breakdown(
        system_prompt_text=base_text + tools_block + todos_block + plans_block,
        summary_text=memory_block,
        tools=tools,
        messages=state["messages"],
    )

    # system 仅临时置于最前，用于本次 invoke；不写回 state、不进 messages 历史
    response = bound.invoke([system, *state["messages"]])

    # 用真实 prompt token 数（若模型回传）自适应校准离线估算。
    usage = getattr(response, "usage_metadata", None)
    actual = usage.get("input_tokens") if usage else None
    breakdown.calibration_factor = next_calibration_factor(
        prev_factor, breakdown.estimated_total_raw, actual
    )
    breakdown.actual_total = actual

    # 只把 LLM 回复累积进 state（messages 的 reducer 负责 append），并携出上下文占用拆分。
    return {"messages": [response], "context_usage": breakdown.to_dict()}



def _render_todos_block(todos: list[dict]) -> str:
    """渲染待办注入块：始终附「记录/完成」的轻量指引，并列出尚未完成的挂起项。

    指引恒在（即使当前无待办），使 Agent 知道可用 add_todo 挂起后续建议；仅展示
    status != "done" 的项，附 id 供 complete_todo 引用。
    """
    guidance = (
        "\n\n[待办管理] 待办用于「可立即执行的原子步骤」（做完即勾掉）。若你向用户提出了"
        "尚未执行的后续步骤，请调用 add_todo 登记以免遗忘；完成后调用 complete_todo 标记完成。"
        "（尚未定型、需反复修订的整体思路/方案请改用 add_plan，见下方[方案管理]。）"
    )
    open_todos = [t for t in todos if t.get("status") != "done"]
    if not open_todos:
        return guidance
    lines = "\n".join(f"- [#{t['id']}] {t['content']}" for t in open_todos)
    return f"{guidance}\n当前未完成的挂起项：\n{lines}"


def _render_plans_block(plans: list[dict]) -> str:
    """渲染方案注入块：始终附「记录/修订」引导，并逐字列出未归档方案的标题+完整正文。

    指引恒在（即使当前无方案），使 Agent 知道可用 add_plan 存放尚未定型的规划性思考；
    仅展示 status != "archived" 的项，附 id 供 update_plan 引用。方案正文完整注入、
    永不进摘要压缩，因此规划性思考不会随长程压缩丢失。
    """
    guidance = (
        "\n\n[方案管理] 方案用于「尚未定型但重要、需反复权衡修订的规划性内容」——"
        "整体思路、初步计划、设计取舍等。用 add_plan 存下正文；随想法演进用 update_plan "
        "修订（只传要改的字段）；确认采纳置 status=active，作废或被取代置 status=archived。"
    )
    live_plans = [p for p in plans if p.get("status") != "archived"]
    if not live_plans:
        return guidance
    blocks = "\n".join(
        f"── [#{p['id']}] ({p.get('status', 'draft')}) {p.get('title', '')}\n{p.get('content', '')}"
        for p in live_plans
    )
    return f"{guidance}\n当前方案：\n{blocks}"


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
    """将历史对话压缩进结构化记忆（memory），并删除已被覆盖的旧消息。

    依赖均为仅关键字注入，与 react_node 一致，由 assembly 经 partial 绑定：
    - llm：不绑工具的裸模型（总结不该触发工具调用）。
    - graph_config：避免使用保留名 ``config``，否则会被 LangGraph 误注入 RunnableConfig。

    记忆分三段各按生命周期合并（见 memory.merge_memory）：用户目标粘性锚定、关键发现
    累积去重、进展滚动压缩。FLASH 产出 JSON，解析失败时回退——保留旧记忆、把原始输出
    并入进展，避免信息净丢失。

    删除区间由 ``summary_cutoff`` 决定：兼顾回合预算与消息数上限，截断点落在
    HumanMessage 边界。``entry_condition`` 已保证进入本节点时 cutoff > 0，不空转。
    """
    prev = AgentMemory.from_dict(state.get("memory") or {})
    if not prev.is_empty():
        # 仅替换字面量 {prev}；插入的 JSON 含花括号但不再做二次解析，安全。
        prompt = SUMMARY_PROMPT_MERGE.replace(
            "{prev}", json.dumps(prev.to_dict(), ensure_ascii=False)
        )
    else:
        prompt = SUMMARY_PROMPT_INITIAL

    # 完整历史 + 一条临时指令喂给模型；不写回 state
    response = llm.invoke([*state["messages"], HumanMessage(content=prompt)])

    parsed = parse_memory(str(response.content))
    if parsed is None:
        # 回退：保留旧记忆，把原始输出并入进展叙事，避免本轮信息净丢失。
        parsed = AgentMemory(progress=str(response.content))
    merged = merge_memory(prev, parsed, key_findings_cap=graph_config.key_findings_cap)

    cutoff = summary_cutoff(
        state["messages"],
        keep=graph_config.summary_keep_recent,
        trigger=graph_config.summary_trigger_count,
    )
    # m.id 由 checkpointer 赋值；缺 id 的消息无法 RemoveMessage，跳过以防构造非法删除
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:cutoff] if m.id]

    return {"memory": merged.to_dict(), "messages": delete_messages}


def finish_turn(state: CommonState) -> dict:
    """每个 turn 的唯一确定性终点：累加轮数。

    ``turn`` 由 state 上的 ``_add_turns`` reducer 负责累加，故此处只需返回增量 1。
    作为单一收尾点，未来 turn 级逻辑（如统计、配额、跑题计数）都可挂在这里，
    无需散落到多条出边。
    """
    return {"turn": 1}


__all__ = ["summarize_conversation", "react_node", "finish_turn", "summary_cutoff"]