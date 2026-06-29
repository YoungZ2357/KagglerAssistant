# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: KagglerAssistant
# FileName: screens.py
# -------------------------------------------------------------------------
"""TUI 模态屏：启动时选择数据集 CSV。

仅做路径文本输入（不实现文件树浏览，保持最小范围；DirectoryTree 可作后续增强）。
"""
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label


class FilePickerScreen(ModalScreen[str | None]):
    """输入数据集 CSV 路径。确认返回路径字符串，取消返回 None。"""

    BINDINGS = [("escape", "cancel", "取消")]

    def compose(self) -> ComposeResult:
        with Vertical(id="file-picker-dialog"):
            yield Label("选择数据集 CSV", id="fp-title")
            yield Input(placeholder="数据集.csv 路径", id="fp-input")
            with Horizontal(id="fp-buttons"):
                yield Button("确认", id="fp-confirm", variant="primary")
                yield Button("取消", id="fp-cancel")

    def on_mount(self) -> None:
        self.query_one("#fp-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "fp-confirm":
            self.dismiss(self.query_one("#fp-input", Input).value.strip() or None)
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)
