"""工具回传体积上限——集中管理，避免任一工具返回随数据规模无上界增长而撑爆模型上下文。

约定：所有截断都必须显式（返回 truncated/计数元信息），不做静默截断。
本模块为纯模块，不依赖 LangChain / polars，可被任意 compute 层安全导入。
"""

from __future__ import annotations

# 单个字符串值最大长度（sample_values / preview 单元格 / 分类取值等）
MAX_STR_VALUE_LEN = 200
# explore_schema 最多回传的列数
MAX_SCHEMA_COLUMNS = 200
# FE preview 每行最多回传的列数
MAX_PREVIEW_COLUMNS = 60
# 每种相关方法最多回传的 pair 数
MAX_CORRELATION_PAIRS = 100
# label 编码映射最多回传的条目数（完整映射仍用于实际编码，不受影响）
MAX_MAPPING_ENTRIES = 50
# one-hot new_columns 最多列出的数量
MAX_ONEHOT_COLUMNS = 50
# 通用列名清单上限（remaining_columns / original_features 等）
MAX_COLUMN_LIST = 100
# 目录文件列举上限
MAX_WORKSPACE_ENTRIES = 200
# descriptive_analysis 最多处理的列数
MAX_DESCRIBE_COLUMNS = 100


def truncate_str(s: str, limit: int = MAX_STR_VALUE_LEN) -> str:
    """超长字符串截断到 limit 字符，并显式标注省略的字符数。"""
    if len(s) <= limit:
        return s
    return s[:limit] + f"…[+{len(s) - limit} chars]"


def cap_list(items: list, limit: int) -> tuple[list, dict | None]:
    """按 limit 截断列表。

    Returns:
        (截断后的列表, 截断元信息)。未超限时元信息为 None；
        超限时为 {"truncated": True, "shown": limit, "total": 原长度}。
    """
    if len(items) <= limit:
        return items, None
    return items[:limit], {"truncated": True, "shown": limit, "total": len(items)}


__all__ = [
    "MAX_STR_VALUE_LEN",
    "MAX_SCHEMA_COLUMNS",
    "MAX_PREVIEW_COLUMNS",
    "MAX_CORRELATION_PAIRS",
    "MAX_MAPPING_ENTRIES",
    "MAX_ONEHOT_COLUMNS",
    "MAX_COLUMN_LIST",
    "MAX_WORKSPACE_ENTRIES",
    "MAX_DESCRIBE_COLUMNS",
    "truncate_str",
    "cap_list",
]
