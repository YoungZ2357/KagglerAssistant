import json

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from kaggler.persistence.data_provider import DataProvider


def dumps_cn(obj) -> str:
    """统一 JSON 序列化：关闭 ASCII 转义，保留中文原文。"""
    return json.dumps(obj, ensure_ascii=False)


def commit_mutation(
    data: DataProvider,
    result: dict,
    tool_call_id: str,
    *,
    parent_version: int,
    tool_name: str,
    description: str,
) -> Command:
    """写工具的统一收尾。

    compute 层约定：失败返回含 ``"error"`` 键的 dict；成功返回含
    ``op`` 及 ``rows_before/rows_after/preview/summary`` 的 dict。
    - 失败：原样回一条 ToolMessage，不改数据版本，不记录谱系。
    - 成功：登记新版本 + 谱系（parent_version/tool_name/description 由调用方
      即各 FE 工具显式提供），回摘要并推进 ``data_version``。
    """
    if "error" in result:
        return Command(update={
            "messages": [
                ToolMessage(dumps_cn(result), tool_call_id=tool_call_id),
            ],
        })

    new_version = data.add_version(
        result["op"],
        parent=parent_version,
        tool=tool_name,
        description=description,
    )
    payload = {k: result[k] for k in ("rows_before", "rows_after", "preview", "summary")}
    return Command(update={
        "data_version": new_version,
        "messages": [
            ToolMessage(
                dumps_cn({"new_data_version": new_version, **payload}),
                tool_call_id=tool_call_id,
            ),
        ],
    })
