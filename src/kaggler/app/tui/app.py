# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: KagglerAssistant
# FileName: app.py
# -------------------------------------------------------------------------
"""KagglerAssistant 的 Textual TUI（左栏对话 + 右栏 Agent 行为追溯）。

设计要点（沿用旧项目原型的核心经验）：
- **单一对话窗**：对话历史与进行中的回答都活在左栏同一个可滚动容器
  （``VerticalScroll#chat-log``）里。每条消息是一个 ``Static``；本轮回答的
  ``Static`` 在流式过程中**原地更新**，结束即定稿，无两段式搬运。
- **Agent 行为追溯**：右栏 ``RichLog#agent-trace`` 以 append-only 方式逐行记录
  节点流转（▶ 进入 / ✓ 完成）与 react 节点决策的 tool_calls。仅呈现「Agent 做了
  什么」，不渲染 tool_result 数据表（那是另一类需求，不在追溯范围内）。
- **绝不把动态文本插值进 Rich markup 字符串**：对话/trace 写入一律用 Rich
  ``Text`` 对象，``Static``/``RichLog`` 设 ``markup=False``，因此 LLM 输出里出现
  ``[`` 等字符也不会触发 MarkupError。
- **流式 token 节流渲染**：worker 线程只往 buffer 累积 token 并标脏，由一个
  ``set_interval`` 定时器整体刷新进行中那条消息，避免每 token 全量重渲染卡死 UI。

用法：
    kaggler                       # 安装后
    python -m kaggler.app.tui.app # 未安装时（需 src 在路径上）
启动后弹窗输入数据集 CSV 路径，再进入对话。
"""
import random
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Header, Input, Label, RichLog, Static

from kaggler.app.tui.screens import FilePickerScreen
from kaggler.shared.wrapper import AgentSession

_GREETINGS = ["请讲！", "快点把问题端上来罢", "冲刺！冲刺！冲！冲！"]

# 进行中回答的刷新间隔（秒）。节流的核心参数：足够流畅，又不至于每 token 重渲染。
_FLUSH_INTERVAL: float = 0.1

# ── Agent 行为追溯：节点名 → 显示用标签 / 默认描述 ──────────────────────────
# 键用图节点名的字符串值（Node 为 str-Enum，故字符串键可被枚举成员命中）。
_NODE_LABELS: dict[str, str] = {
    "react": "agent",
    "tools": "tools",
    "summarize": "summarize",
    "finish": "finish",
}
# 无 tool_calls 时各节点 ✓ 行的默认描述
_NODE_DESC: dict[str, str] = {
    "react": "LLM 决策",
    "tools": "工具执行",
    "summarize": "压缩对话历史",
}
# finish 是内部计数节点（finish_turn），不进 trace
_SILENT_NODES: frozenset[str] = frozenset({"finish"})
# trace 行的节点名列宽（对齐用）
_LABEL_WIDTH: int = 10


class StreamEvent(Message):
    """Worker 线程向 Textual 事件循环推送的通用事件载体。"""

    def __init__(self, event: dict[str, Any]) -> None:
        self.event = event
        super().__init__()


class KagglerTUI(App[None]):
    CSS_PATH = "app.tcss"
    TITLE = "KagglerAssistant"
    BINDINGS = [("ctrl+q", "quit", "退出")]

    def __init__(self) -> None:
        super().__init__()
        # 注意：不能命名为 `_thread_id` —— 那会覆盖 Textual `App._thread_id`
        # （它存事件循环所在线程的 id，run_worker 据此判断是否需跨线程 marshal）。
        self._session: AgentSession | None = None
        self._streaming_buf: str = ""
        self._streaming_dirty: bool = False
        # 本轮回答对应的消息 widget；token 原地更新它，结束置空。
        self._stream_widget: Static | None = None

    # ── 布局 ───────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            # 左栏：单一对话窗
            yield VerticalScroll(id="chat-log")
            # 右栏：Agent 行为追溯
            with Vertical(id="trace-col"):
                yield Label("Agent 行为", classes="panel-title")
                yield RichLog(id="agent-trace", markup=False, highlight=False, wrap=True)
        yield Input(placeholder="> ", id="user-input", disabled=True)

    # ── 生命周期 ───────────────────────────────────────────────────────────
    def on_mount(self) -> None:
        # 节流定时器：进行中回答整体刷新，避免每 token 重渲染。
        self.set_interval(_FLUSH_INTERVAL, self._flush_streaming)
        self.push_screen(FilePickerScreen(), self._on_file_picked)

    def _on_file_picked(self, path: str | None) -> None:
        if not path:
            # 取消选择 → 无可用数据集，直接退出。
            self.exit()
            return
        if not Path(path).is_file():
            self.notify(f"文件不存在：{path}", severity="error")
            self.push_screen(FilePickerScreen(), self._on_file_picked)
            return
        self._append(
            Text.assemble(("系统: ", "bold yellow"), (f"正在加载数据集 {path} …", ""))
        )
        self.run_worker(lambda: self._init_worker(path), thread=True, name="init")

    # ── 对话窗写入 ─────────────────────────────────────────────────────────
    def _append(self, text: Text) -> Static:
        """向左栏对话窗追加一条消息，返回该 widget（供流式原地更新）。"""
        widget = Static(text, markup=False)
        log = self.query_one("#chat-log", VerticalScroll)
        log.mount(widget)
        log.scroll_end(animate=False)
        return widget

    # ── Worker 函数（后台线程）─────────────────────────────────────────────
    def _init_worker(self, path: str) -> None:
        try:
            session = AgentSession(path)
            self._session = session
            self.post_message(StreamEvent({"type": "init_done"}))
        except Exception as exc:  # noqa: BLE001 — 后台线程异常需回送到 UI 显示
            self.post_message(StreamEvent({"type": "error", "message": str(exc)}))

    def _stream_worker(self, question: str) -> None:
        try:
            for ev in self._session.stream_events(question):
                self.post_message(StreamEvent(ev))
        except Exception as exc:  # noqa: BLE001
            self.post_message(StreamEvent({"type": "error", "message": str(exc)}))
        finally:
            self.post_message(StreamEvent({"type": "turn_done"}))

    # ── 事件处理（保持轻量，重活交给节流定时器）──────────────────────────
    def on_stream_event(self, message: StreamEvent) -> None:
        e = message.event
        t = e["type"]

        if t == "init_done":
            self._append(
                Text.assemble(("助手: ", "bold green"), (random.choice(_GREETINGS), ""))
            )
            self._enable_input()

        elif t == "token":
            # 只累积 + 标脏，不在这里更新 widget（由 _flush_streaming 节流刷新）。
            self._streaming_buf += e["content"]
            self._streaming_dirty = True

        elif t == "node_active":
            self._handle_node_active(e["node"])

        elif t == "node_done":
            self._handle_node_done(e["node"], e.get("tool_calls", []))

        elif t == "turn_done":
            self._finalize_streaming()
            self._enable_input()

        elif t == "error":
            self._finalize_streaming()
            self._append(
                Text.assemble(("错误: ", "bold red"), (e.get("message", "未知错误"), ""))
            )
            self._enable_input()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value or not self._session:
            return
        event.input.value = ""
        event.input.disabled = True
        self._streaming_buf = ""
        self._streaming_dirty = False
        self._append(Text.assemble(("你: ", "bold blue"), (value, "")))
        # 为本轮回答预挂一个 widget，后续 token 原地更新它（同一窗、同一条消息）。
        self._stream_widget = self._append(Text.assemble(("助手: ", "bold green"), ("", "")))
        self.run_worker(
            lambda: self._stream_worker(value), thread=True, name="stream"
        )

    # ── 流式回答（原地更新当前消息）──────────────────────────────────────
    def _flush_streaming(self) -> None:
        """节流刷新：仅在有新 token 时整体重绘进行中那条消息。"""
        if not self._streaming_dirty or self._stream_widget is None:
            return
        self._streaming_dirty = False
        body = Text.assemble(("助手: ", "bold green"), (self._streaming_buf, ""))
        body.append(" ▌", style="blink")
        self._stream_widget.update(body)
        self.query_one("#chat-log", VerticalScroll).scroll_end(animate=False)

    def _finalize_streaming(self) -> None:
        """本轮结束：去掉光标定稿当前消息；若无内容则移除占位 widget。"""
        widget = self._stream_widget
        self._stream_widget = None
        if widget is not None:
            if self._streaming_buf:
                widget.update(
                    Text.assemble(("助手: ", "bold green"), (self._streaming_buf, ""))
                )
            else:
                widget.remove()
            self.query_one("#chat-log", VerticalScroll).scroll_end(animate=False)
        self._streaming_buf = ""
        self._streaming_dirty = False

    def _enable_input(self) -> None:
        inp = self.query_one("#user-input", Input)
        inp.disabled = False
        inp.focus()

    # ── Agent 行为追溯（右栏 append-only RichLog）────────────────────────────
    def _handle_node_active(self, node: str) -> None:
        if node in _SILENT_NODES:
            return
        label = _NODE_LABELS.get(node, node)
        self._write_trace("▶", "yellow", label, "生成中…")

    def _handle_node_done(self, node: str, tool_calls: list[dict]) -> None:
        if node in _SILENT_NODES:
            return
        label = _NODE_LABELS.get(node, node)
        # 一个 react 节点可能并行发起多个工具调用，逐个成行，避免只追溯到第一个。
        if tool_calls:
            for tc in tool_calls:
                self._write_trace("✓", "green", label, self._fmt_tool_call(tc))
        else:
            self._write_trace("✓", "green", label, _NODE_DESC.get(node, "完成"))

    def _write_trace(self, icon: str, icon_color: str, label: str, desc: str) -> None:
        line = Text.assemble(
            (f"{icon} ", icon_color),
            (f"{str(label):<{_LABEL_WIDTH}}", "cyan"),
            (desc, ""),
        )
        self.query_one("#agent-trace", RichLog).write(line)

    def _fmt_tool_call(self, tc: dict) -> str:
        name = tc.get("name", "")
        args = tc.get("args", {})
        arg_str = ", ".join(f"{k}='{v}'" for k, v in list(args.items())[:2])
        return f"{name}({arg_str})" if arg_str else f"{name}()"


def main() -> None:
    KagglerTUI().run()


if __name__ == "__main__":
    main()
