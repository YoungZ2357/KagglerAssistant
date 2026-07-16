# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: KagglerAssistant
# FileName: types.py
# Date: 2026/6/29
# -------------------------------------------------------------------------
from enum import Enum


class Node(str, Enum):
    """图节点名的唯一真相源（与 Mode、DeepSeekModel 同为 str-Enum 风格）。

    继承 ``str`` → 枚举成员本身即字符串，可直接喂给 ``add_node`` /
    ``add_edge`` / ``add_conditional_edges`` 的 path_map，无需 ``.value``。

    边函数返回该枚举成员（而非裸字符串），配合 ``Literal[...]`` 返回标注：
    - 可追溯 / 可重命名：对 ``Node.X`` 做 Find Usages / Rename 即可全局生效；
    - 编译期抓拼写错误：``Node.SUMMRIZE`` 是 AttributeError，而非静默失效。
    """
    REACT = "react"
    APPROVAL = "approval"
    TOOLS = "tools"
    SUMMARIZE = "summarize"
    FINISH = "finish"
