# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: KagglerAssistant
# FileName: widgets.py
# -------------------------------------------------------------------------
"""对话窗与追溯栏的可寻址/可点击部件。

- :class:`ChatMessage`：一条对话消息。是一个**容器**（头部标签 + 正文两个子部件），
  可点击（点击发 ``Clicked`` 消息，供上层挂钩行为回溯等未来能力），携带
  ``role`` / ``turn_id`` / 原始文本。
- :class:`TraceLine`：右栏 Agent 行为追溯的一行，携带 ``turn_id``，供按轮高亮联动。

**为何用容器 + Textual ``Markdown`` 部件（而非 rich ``Markdown`` renderable）**：
Textual 的原生文本选择（鼠标拖选 + Ctrl+C 复制）只能从渲染成 ``Text``/``str`` 的
部件里取到文字；``rich.console.Group`` 与 ``rich.markdown.Markdown`` 这类**复合
renderable 取不到可选文本**（拖选后复制为空）。而 Textual 的 ``Markdown`` **部件**
把 markdown 渲染成一棵子部件树，每块都可被原生选择——于是「渲染 markdown」与
「可选中复制」两个需求得以兼得。

因此：正文在**流式过程中**是纯文本 ``Static``（显示未解析的原始 token，快且可选）；
**本轮结束**再把正文换成 ``Markdown`` 部件（格式化 + 可选）。用户/系统/错误消息正文
始终是纯文本 ``Static``。头部标签与所有正文都可被拖选复制。
"""
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Markdown, Static

# 角色 → 头部标签文本与样式（复用旧配色）。
_ROLE_HEADER: dict[str, tuple[str, str]] = {
    "user": ("你", "bold blue"),
    "assistant": ("助手", "bold green"),
    "system": ("系统", "bold yellow"),
    "error": ("错误", "bold red"),
}
# 正文用 Markdown 部件渲染的角色（仅 LLM 回答；用户输入按纯文本处理）。
_MARKDOWN_ROLES: frozenset[str] = frozenset({"assistant"})

class _Body(Static):
    """纯文本正文：渲染为 Rich ``Text``，可被原生选择/复制。"""


class ChatMessage(Vertical):
    """一条可点击的对话消息（头部标签 + 正文）。"""

    can_focus = True  # 为未来键盘导航铺垫

    class Clicked(Message):
        """消息被点击。``source`` 为被点的 :class:`ChatMessage`（未来行为回溯挂钩点）。"""

        def __init__(self, source: "ChatMessage") -> None:
            self.source = source
            super().__init__()

    def __init__(
        self,
        role: str,
        raw: str,
        *,
        turn_id: int | None = None,
        markdown: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.role = role
        self.turn_id = turn_id
        self._raw = raw
        self._markdown = markdown

    def compose(self) -> ComposeResult:
        label, style = _ROLE_HEADER.get(self.role, (self.role, "bold"))
        yield Static(Text(label, style=style), classes="msg-header")
        yield self._make_body()

    def on_click(self) -> None:
        self.post_message(self.Clicked(self))

    # ── 正文构造/更新 ────────────────────────────────────────────────────
    def _make_body(self):
        """据 role/markdown 生成正文部件：Markdown 部件（格式化+可选）或纯文本。"""
        if self._markdown and self.role in _MARKDOWN_ROLES:
            return Markdown(self._raw, classes="msg-body")
        return _Body(Text(self._raw), classes="msg-body")

    def append_cursor(self, raw: str) -> None:
        """流式进行中：纯文本正文 + 闪烁光标（不解析 markdown，快且不错乱）。"""
        self._raw = raw
        body = self.query_one(".msg-body", Static)
        text = Text(raw)
        text.append(" ▌", style="blink")
        body.update(text)

    def set_content(self, raw: str, *, markdown: bool | None = None) -> None:
        """本轮定稿：重建正文（助手切到 Markdown 部件渲染）。"""
        self._raw = raw
        if markdown is not None:
            self._markdown = markdown
        self.query_one(".msg-body").remove()
        self.mount(self._make_body())


class TraceLine(Static):
    """右栏一条 Agent 行为追溯行，携带 ``turn_id`` 供按轮高亮。"""

    def __init__(
        self, renderable: Any, *, turn_id: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__(renderable, markup=False, **kwargs)
        self.turn_id = turn_id


class TraceTable(TraceLine):
    """react 决策的一批 tool_calls，以表格渲染（工具名 | 参数）。

    继承 :class:`TraceLine` 而非直接继承 ``Static``：既自动携带 ``turn_id``，又能被
    ``on_chat_message_clicked`` 里的 ``query(TraceLine)`` 命中（Textual 类型选择器匹配
    基类，故 ``TraceLine.linked`` 高亮规则也对它生效），点击联动无需改动。构造时传入一个
    ``rich.table.Table`` 作为 renderable（``markup=False`` 已在基类设好）。
    """
