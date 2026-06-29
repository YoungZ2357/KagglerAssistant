import os

from langgraph.graph.state import CompiledStateGraph



def draw_graph(graph: CompiledStateGraph, save_directory: str = "", file_name: str = "graph.png") -> None:

    png = graph.get_graph().draw_mermaid_png()

    file_path = os.path.join(save_directory, file_name)
    with open(file_path, "wb") as f:
        f.write(png)

