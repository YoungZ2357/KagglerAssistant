import os
import re

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from kaggler.graph.state import CommonState


def _quote_if_needed(node_id: str) -> str:
    if re.search(r'[^a-zA-Z0-9_]', node_id):
        return f'"{node_id}"'
    return node_id


def _fix_mermaid_syntax(mermaid_code: str) -> str:
    lines = mermaid_code.split('\n')
    result = []

    for line in lines:
        if not line.strip():
            result.append(line)
            continue

        stripped = line.strip()
        if stripped.startswith('graph ') or stripped.startswith('flowchart '):
            result.append(line)
            continue

        line = re.sub(
            r'(\s*)([a-zA-Z_][\w.]*?)(\s*)(\[|\(\()',
            lambda m: m.group(1) + _quote_if_needed(m.group(2)) + m.group(3) + m.group(4),
            line,
        )

        line = re.sub(
            r'(?<=\s)([a-zA-Z_][\w.]*?)(?=\s*-->)',
            lambda m: _quote_if_needed(m.group(1)),
            line,
        )

        line = re.sub(
            r'(?<=-->)(\s*)([a-zA-Z_][\w.]*?)(?=\s|$)',
            lambda m: m.group(1) + _quote_if_needed(m.group(2)),
            line,
        )

        result.append(line)

    return '\n'.join(result)


def draw_graph(graph: CompiledStateGraph, save_directory: str = r"D:\files\coding\portfolio\KagglerAssistant\docs", file_name: str = "graph.mmd") -> None:
    mermaid_code = graph.get_graph().draw_mermaid()
    print(mermaid_code)
    mermaid_code = _fix_mermaid_syntax(mermaid_code)

    file_path = os.path.join(save_directory, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(mermaid_code)


def main() -> None:
    stub_nodes = {
        "react": lambda state: {"messages": []},
        "tools": lambda state: {"messages": []},
        "summarize": lambda state: {"messages": [], "summary": ""},
        "finish": lambda state: {"turn": 1},
    }

    builder = StateGraph(CommonState)
    for name, fn in stub_nodes.items():
        builder.add_node(name, fn)

    builder.add_conditional_edges(
        START,
        lambda state: "summarize" if state.get("messages") and len(state["messages"]) > 5 else "react",
        ["summarize", "react"],
    )
    builder.add_edge("summarize", "react")
    builder.add_conditional_edges(
        "react",
        lambda state: "tools" if state["messages"] and getattr(state["messages"][-1], "tool_calls", None) else "finish",
        ["tools", "finish"],
    )
    builder.add_edge("tools", "react")
    builder.add_edge("finish", END)

    draw_graph(builder.compile())


if __name__ == "__main__":
    main()
