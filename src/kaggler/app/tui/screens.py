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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Label, ListItem, ListView

from kaggler.persistence.conversation_store import ConversationRecord


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
    """目录树浏览。

    ``select_dir_mode=False``（默认）：点选 .csv 文件即确认。
    ``select_dir_mode=True``：确认按钮返回当前浏览的目录路径，用于选择工作区。
    """

    BINDINGS = [("escape", "cancel", "取消")]

    def __init__(self, start_dir: str | None = None, *, select_dir_mode: bool = False) -> None:
        super().__init__()
        start = Path(start_dir).expanduser() if start_dir else Path.cwd()
        if not start.is_dir():
            start = Path.cwd()
        self._root = start.resolve()
        self._select_dir_mode = select_dir_mode

    def compose(self) -> ComposeResult:
        with Vertical(id="dir-browser-dialog"):
            if self._select_dir_mode:
                yield Label("浏览并选择工作区目录", id="db-title")
            else:
                yield Label("浏览并选择 CSV（点选 .csv 即确认）", id="db-title")
            with Horizontal(id="db-rootbar"):
                yield Input(value=str(self._root), id="db-root", placeholder="根目录")
                yield Button("上一级", id="db-up")
                yield Button("前往", id="db-go")
            yield _CsvDirectoryTree(str(self._root), id="db-tree")
            with Horizontal(id="db-buttons"):
                if self._select_dir_mode:
                    yield Button("选择此目录", id="db-select-dir", variant="primary")
                yield Button("取消", id="db-cancel")

    def on_mount(self) -> None:
        self.query_one("#db-tree", _CsvDirectoryTree).focus()

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        if self._select_dir_mode:
            return
        self.dismiss(str(event.path))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "db-go":
            self._goto_root()
        elif event.button.id == "db-up":
            self._go_parent()
        elif event.button.id == "db-select-dir":
            tree = self.query_one("#db-tree", _CsvDirectoryTree)
            cursor = tree.cursor_node
            if cursor is not None and cursor.data is not None:
                path = cursor.data.path
            else:
                path = Path(tree.path)
            self.dismiss(str(path.resolve()))
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

    def _go_parent(self) -> None:
        """跳到当前根目录的上一级；已在文件系统最顶层则提示。"""
        tree = self.query_one("#db-tree", _CsvDirectoryTree)
        current = Path(tree.path)
        parent = current.parent
        if parent == current:  # 到盘符/根，无更上层
            self.notify("已在最顶层目录", severity="information")
            return
        parent = parent.resolve()
        tree.path = str(parent)  # 设置 reactive path 会自动 reload
        self.query_one("#db-root", Input).value = str(parent)  # 同步根目录输入框
        tree.focus()

    def action_cancel(self) -> None:
        self.dismiss(None)


class FilePickerScreen(ModalScreen[str | None]):
    """选择数据集 CSV：可手动输入路径或点「浏览…」用文件浏览器挑选。

    确认返回路径字符串，取消返回 None。
    ``start_dir`` 指定文件浏览器的起始目录（默认当前工作目录）。
    """

    BINDINGS = [("escape", "cancel", "取消")]

    def __init__(self, start_dir: str | None = None) -> None:
        super().__init__()
        self._start_dir = start_dir

    def compose(self) -> ComposeResult:
        with Vertical(id="file-picker-dialog"):
            yield Label("选择数据集 CSV", id="fp-title")
            yield Input(placeholder="数据集.csv 路径", id="fp-input", value=self._start_dir or "")
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


@dataclass(frozen=True)
class ConversationAction:
    """``ConversationListScreen`` 的返回值：对某个对话执行的动作。"""

    action: str          # "resume" | "rename" | "delete"
    thread_id: str
    new_name: str | None = None   # 仅 rename 用


class ConversationListScreen(ModalScreen["ConversationAction | None"]):
    """列出当前工作区的对话，支持恢复 / 重命名 / 删除。

    纯视图：构造时传入已取好的 ``list[ConversationRecord]``，不自行查库。
    - Enter 或「恢复」→ 返回 resume 动作
    - 「重命名」→ 显出输入框，提交后返回 rename 动作
    - 「删除」→ 就地二次确认后返回 delete 动作
    - 取消 / Escape → 返回 None
    """

    BINDINGS = [("escape", "cancel", "取消")]

    def __init__(self, records: list[ConversationRecord]) -> None:
        super().__init__()
        self._records = records
        self._delete_armed = False

    def compose(self) -> ComposeResult:
        with Vertical(id="conv-dialog"):
            yield Label("对话（当前工作区）", id="conv-title")
            if not self._records:
                yield Label("当前工作区暂无对话。", id="conv-empty")
                with Horizontal(id="conv-buttons"):
                    yield Button("取消", id="conv-cancel")
                return
            items = [
                ListItem(Label(self._format_row(r)))
                for r in self._records
            ]
            yield ListView(*items, id="conv-list")
            yield Input(placeholder="输入新名称后回车", id="conv-rename", classes="hidden")
            with Horizontal(id="conv-buttons"):
                yield Button("恢复", id="conv-resume", variant="primary")
                yield Button("重命名", id="conv-rename-btn")
                yield Button("删除", id="conv-delete")
                yield Button("取消", id="conv-cancel")

    @staticmethod
    def _format_row(r: ConversationRecord) -> str:
        when = r.updated_at[:16].replace("T", " ")
        return f"[b]{r.name}[/b]  ·  {when}  ·  {Path(r.csv_path).name}"

    def on_mount(self) -> None:
        if self._records:
            self.query_one("#conv-list", ListView).focus()

    def _selected_record(self) -> ConversationRecord | None:
        idx = self.query_one("#conv-list", ListView).index
        if idx is None or not (0 <= idx < len(self._records)):
            return None
        return self._records[idx]

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # 在列表上按 Enter：等同「恢复」。
        record = self._selected_record()
        if record is not None:
            self.dismiss(ConversationAction("resume", record.thread_id))

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        # 切换选中项时撤销「删除」的武装态，避免误删刚切过去的另一条对话。
        self._disarm_delete()

    def _disarm_delete(self) -> None:
        if self._delete_armed:
            self._delete_armed = False
            try:
                self.query_one("#conv-delete", Button).label = "删除"
            except Exception:  # noqa: BLE001 — 空列表时无该按钮
                pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "conv-rename":
            return
        new_name = event.value.strip()
        record = self._selected_record()
        if record is None or not new_name:
            self.dismiss(None)
            return
        self.dismiss(ConversationAction("rename", record.thread_id, new_name=new_name))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "conv-cancel":
            self.dismiss(None)
            return
        record = self._selected_record()
        if record is None:
            self.notify("请先选中一个对话", severity="warning")
            return
        if bid == "conv-resume":
            self.dismiss(ConversationAction("resume", record.thread_id))
        elif bid == "conv-rename-btn":
            inp = self.query_one("#conv-rename", Input)
            inp.value = record.name
            inp.remove_class("hidden")
            inp.focus()
        elif bid == "conv-delete":
            # 两步确认：首按武装并改文案，再按才真正删除。
            if not self._delete_armed:
                self._delete_armed = True
                event.button.label = "确认删除?"
                return
            self.dismiss(ConversationAction("delete", record.thread_id))

    def action_cancel(self) -> None:
        self.dismiss(None)
