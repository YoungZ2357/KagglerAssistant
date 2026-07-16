# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: KagglerAssistant
# FileName: cli.py
# -------------------------------------------------------------------------
"""最小可运行入口：加载数据集 → 组装图 → 多轮问答 REPL。

用法：
    python -m kaggler.app.cli <数据集.csv>

会话记忆由编译时的 checkpointer 按 ``thread_id`` 持久化，故仅首轮需注入
种子 state（current_mode / file_path / data_version），后续只传新问题。
"""
import sys
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from kaggler.graph.assembly import build_graph
from kaggler.shared.types import Mode
from kaggler.persistence.data_provider import DataProvider


def _ask_approval(payload: dict) -> dict:
    """HITL：在终端打印待批的高风险操作并读取用户裁决，返回 resume 决策。

    读到 EOF/中断按拒绝处理（安全默认，不执行副作用）。
    """
    pending = (payload or {}).get("pending", [])
    print("\n⚠ 高风险操作需要确认：")
    for c in pending:
        print(f"  - {c.get('name')}  副作用={c.get('effects')}  参数={c.get('args')}")
    try:
        ans = input("批准执行? [y]是 / [n]否 / [a]本会话始终允许: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"
    if ans in ("a", "always"):
        return {"action": "always"}
    if ans in ("y", "yes"):
        return {"action": "approve"}
    return {"action": "reject"}


def _run_to_completion(graph, inputs, config: dict) -> dict:
    """驱动一次运行至结束；遇到 HITL 断点则就地询问并 ``Command(resume=...)`` 续跑。"""
    state = graph.invoke(inputs, config=config)
    while "__interrupt__" in state:
        intr = state["__interrupt__"]
        decision = _ask_approval(intr[0].value if intr else {})
        state = graph.invoke(Command(resume=decision), config=config)
    return state


def _last_ai_text(state: dict) -> str:
    """取最后一条非空 AIMessage 的文本（图内部 message 结构不外泄给调用方）。"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python -m kaggler.app.cli <数据集.csv>")
        raise SystemExit(2)
    csv_path = sys.argv[1]

    data = DataProvider()
    data.load_initial(csv_path)

    graph = build_graph(data)
    thread_id = uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    seeded = False
    print(f"已加载数据集：{csv_path}（thread={thread_id[:8]}）。输入问题，:q 退出。")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question in (":q", ":quit", "exit"):
            break
        if not question:
            continue

        payload: dict = {"messages": [HumanMessage(content=question)]}
        if not seeded:
            # 种子 state 仅首轮注入；其余 channel 由 checkpointer 按 thread_id 续写
            payload |= {
                "current_mode": Mode.EDA,
                "file_path": csv_path,
                "data_version": 0,
            }
            seeded = True

        state = _run_to_completion(graph, payload, config)
        print(_last_ai_text(state))


if __name__ == "__main__":
    main()
