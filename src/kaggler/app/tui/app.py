# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: KagglerAssistant
# FileName: app.py
# -------------------------------------------------------------------------
"""KagglerAssistant 的 Textual TUI（单栏循环对话）。

设计要点（沿用旧项目原型的核心经验，剔除所有追溯 / 数据呈现）：
- **绝不把动态文本插值进 Rich markup 字符串**：对话写入一律用 Rich ``Text``
  对象（角色标签带 style，正文为纯文本），RichLog/Static 设 ``markup=False``，
  因此 LLM 输出里出现 ``[`` 等字符也不会触发 MarkupError。
- **流式 token 节流渲染**：worker 线程只往 buffer 累积 token 并标脏，由一个
  ``set_interval`` 定时器整体刷新进行中的回答，避免「每 token 全量重渲染」导致
  的 UI 线程打满 / 卡死。

用法：
    python -m kaggler.app.tui.app
启动后弹窗输入数据集 CSV 路径，再进入对话。
"""
import random
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.message import Message
from textual.widgets import Header, Input, RichLog, Static

from kaggler.app.tui.screens import FilePickerScreen
from kaggler.shared.wrapper import AgentSession

_GREETINGS = ["请讲！", "快点把问题端上来罢", "冲刺！冲刺！冲！冲！"]

# 进行中回答的刷新间隔（秒）。节流的核心参数：足够流畅，又不至于每 token 重渲染。
_FLUSH_INTERVAL: float = 0.1


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

    # ── 布局 ───────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="chat-log", markup=False, highlight=False, wrap=True)
        yield Static("", id="streaming-msg", markup=False)
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
        self.query_one("#chat-log", RichLog).write(
            Text.assemble(("系统: ", "bold yellow"), (f"正在加载数据集 {path} …", ""))
        )
        self.run_worker(lambda: self._init_worker(path), thread=True, name="init")

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
            for tok in self._session.stream(question):
                self.post_message(StreamEvent({"type": "token", "content": tok}))
        except Exception as exc:  # noqa: BLE001
            self.post_message(StreamEvent({"type": "error", "message": str(exc)}))
        finally:
            self.post_message(StreamEvent({"type": "turn_done"}))

    # ── 事件处理（保持轻量，重活交给节流定时器）──────────────────────────
    def on_stream_event(self, message: StreamEvent) -> None:
        e = message.event
        t = e["type"]

        if t == "init_done":
            self.query_one("#chat-log", RichLog).write(
                Text.assemble(("助手: ", "bold green"), (random.choice(_GREETINGS), ""))
            )
            self._enable_input()

        elif t == "token":
            # 只累积 + 标脏，不在这里更新 widget（由 _flush_streaming 节流刷新）。
            self._streaming_buf += e["content"]
            self._streaming_dirty = True

        elif t == "turn_done":
            self._commit_streaming()
            self._enable_input()

        elif t == "error":
            self.query_one("#chat-log", RichLog).write(
                Text.assemble(("错误: ", "bold red"), (e.get("message", "未知错误"), ""))
            )
            self._commit_streaming()
            self._enable_input()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value or not self._session:
            return
        event.input.value = ""
        event.input.disabled = True
        self._streaming_buf = ""
        self._streaming_dirty = False
        self.query_one("#chat-log", RichLog).write(
            Text.assemble(("你: ", "bold blue"), (value, ""))
        )
        self.run_worker(
            lambda: self._stream_worker(value), thread=True, name="stream"
        )

    # ── 流式回答 ───────────────────────────────────────────────────────────
    def _flush_streaming(self) -> None:
        """节流刷新：仅在有新 token 时整体重绘进行中的回答。"""
        if not self._streaming_dirty:
            return
        self._streaming_dirty = False
        body = Text.assemble(("助手: ", "bold green"), (self._streaming_buf, ""))
        body.append(" ▌", style="blink")
        self.query_one("#streaming-msg", Static).update(body)

    def _commit_streaming(self) -> None:
        """把进行中的回答定稿写入 chat-log，并清空进行中显示。"""
        if self._streaming_buf:
            self.query_one("#chat-log", RichLog).write(
                Text.assemble(("助手: ", "bold green"), (self._streaming_buf, ""))
            )
        self._streaming_buf = ""
        self._streaming_dirty = False
        self.query_one("#streaming-msg", Static).update(Text(""))

    def _enable_input(self) -> None:
        inp = self.query_one("#user-input", Input)
        inp.disabled = False
        inp.focus()


def main() -> None:
    KagglerTUI().run()


if __name__ == "__main__":
    main()
