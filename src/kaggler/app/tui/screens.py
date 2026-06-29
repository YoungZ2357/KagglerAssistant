# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: KagglerAssistant
# FileName: screens.py
# -------------------------------------------------------------------------
"""TUI 模态屏：启动时选择数据集 CSV。

提供两种选取方式，二者可混用：
- **手动输入路径**：在输入框敲路径后回车 / 点「确认」。
- **文件浏览器**：点「浏览…」弹出 :class:`DirectoryBrowserScreen`（基于 Textual
  ``DirectoryTree``），树内只展示文件夹与 ``.csv``，点选某个 csv 即回填路径并直接
  确认。顶部可改根目录跳转到别处。纯终端实现、无图形依赖，SSH 下亦可用。
"""
from pathlib import Path
from typing import Iterable

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Label


class _CsvDirectoryTree(DirectoryTree):
    """只展示子目录与 ``.csv`` 文件的目录树（隐藏点开头的隐藏项）。"""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [
            p
            for p in paths
            if not p.name.startswith(".")
            and (p.is_dir() or p.suffix.lower() == ".csv")
        ]


class DirectoryBrowserScreen(ModalScreen[str | None]):
    """目录树浏览：点选某个 .csv 返回其路径，取消返回 None。"""

    BINDINGS = [("escape", "cancel", "取消")]

    def __init__(self, start_dir: str | None = None) -> None:
        super().__init__()
        # 起始根目录：给定值优先，否则当前工作目录。
        start = Path(start_dir).expanduser() if start_dir else Path.cwd()
        if not start.is_dir():
            start = Path.cwd()
        self._root = start.resolve()

    def compose(self) -> ComposeResult:
        with Vertical(id="dir-browser-dialog"):
            yield Label("浏览并选择 CSV（点选 .csv 即确认）", id="db-title")
            with Horizontal(id="db-rootbar"):
                yield Input(value=str(self._root), id="db-root", placeholder="根目录")
                yield Button("前往", id="db-go")
            yield _CsvDirectoryTree(str(self._root), id="db-tree")
            with Horizontal(id="db-buttons"):
                yield Button("取消", id="db-cancel")

    def on_mount(self) -> None:
        self.query_one("#db-tree", _CsvDirectoryTree).focus()

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        # DirectoryTree 已通过 filter_paths 限定为 .csv，这里直接回传。
        self.dismiss(str(event.path))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "db-go":
            self._goto_root()
        else:  # db-cancel
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "db-root":
            self._goto_root()

    def _goto_root(self) -> None:
        raw = self.query_one("#db-root", Input).value.strip()
        new_root = Path(raw).expanduser() if raw else Path.cwd()
        if not new_root.is_dir():
            self.notify(f"目录不存在：{raw}", severity="error")
            return
        tree = self.query_one("#db-tree", _CsvDirectoryTree)
        tree.path = str(new_root.resolve())  # 设置 reactive path 会自动 reload
        tree.focus()

    def action_cancel(self) -> None:
        self.dismiss(None)


class FilePickerScreen(ModalScreen[str | None]):
    """选择数据集 CSV：可手动输入路径或点「浏览…」用文件浏览器挑选。

    确认返回路径字符串，取消返回 None。
    """

    BINDINGS = [("escape", "cancel", "取消")]

    def compose(self) -> ComposeResult:
        with Vertical(id="file-picker-dialog"):
            yield Label("选择数据集 CSV", id="fp-title")
            yield Input(placeholder="数据集.csv 路径", id="fp-input")
            with Horizontal(id="fp-buttons"):
                yield Button("浏览…", id="fp-browse")
                yield Button("确认", id="fp-confirm", variant="primary")
                yield Button("取消", id="fp-cancel")

    def on_mount(self) -> None:
        self.query_one("#fp-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "fp-browse":
            self._open_browser()
        elif event.button.id == "fp-confirm":
            self.dismiss(self.query_one("#fp-input", Input).value.strip() or None)
        else:  # fp-cancel
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def _open_browser(self) -> None:
        # 以当前输入框路径所在目录作为浏览起点（若有），否则用默认 cwd。
        current = self.query_one("#fp-input", Input).value.strip()
        start_dir = None
        if current:
            p = Path(current).expanduser()
            start_dir = str(p.parent if not p.is_dir() else p)
        self.app.push_screen(DirectoryBrowserScreen(start_dir), self._on_browsed)

    def _on_browsed(self, path: str | None) -> None:
        if path:
            # 浏览器选中即直接确认，省去再点一次「确认」。
            self.dismiss(path)

    def action_cancel(self) -> None:
        self.dismiss(None)
