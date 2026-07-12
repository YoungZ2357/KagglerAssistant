# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: KagglerAssistant
# FileName: app.py
# -------------------------------------------------------------------------
"""KagglerAssistant 的 Textual TUI（左栏对话 + 右栏 Agent 行为追溯）。

设计要点（沿用旧项目原型的核心经验）：
- **单一对话窗**：对话历史与进行中的回答都活在左栏同一个可滚动容器
  （``VerticalScroll#chat-log``）里。每条消息是一个可点击的 ``ChatMessage``；本轮
  回答的 ``ChatMessage`` 在流式过程中**原地更新**，结束即定稿，无两段式搬运。
- **Agent 行为追溯**：右栏 ``VerticalScroll#agent-trace`` 以 append-only 方式逐行
  记录节点流转（▶ 进入 / ✓ 完成）与 react 节点决策的 tool_calls，每行是一个带
  ``turn_id`` 的 ``TraceLine``（可按轮高亮）。仅呈现「Agent 做了什么」，不渲染
  tool_result 数据表（那是另一类需求，不在追溯范围内）。
- **点击联动**：每条消息与其所在轮次的追溯行共享 ``turn_id``；点击消息即高亮该轮
  的追溯行（见 ``on_chat_message_clicked``），为未来行为回溯等能力铺垫。
- **Markdown 与 markup 安全**：用户/LLM 消息本轮结束后用 ``rich.markdown.Markdown``
  渲染（流式中先显示纯文本）；它与 Rich console markup（``[red]``）无关，故 LLM 输出
  里的 ``[`` 不会触发 MarkupError。系统/错误消息保持纯 ``Text``。详见 widgets.py。
- **流式 token 节流渲染**：worker 线程只往 buffer 累积 token 并标脏，由一个
  ``set_interval`` 定时器整体刷新进行中那条消息，避免每 token 全量重渲染卡死 UI。
  定时器仅在本轮流式期间运行（提交时 resume、定稿时 pause），空闲不空转。
- **跟底滚动**：两个滚动容器用 Textual 内建 ``anchor()`` 跟随底部——新内容自动
  钉底，用户向上滚动即自动解锚（流式刷新不再拽人），滚回底部自动恢复跟随。

用法：
    kaggler                       # 安装后
    python -m kaggler.app.tui.app # 未安装时（需 src 在路径上）
启动后弹窗输入数据集 CSV 路径，再进入对话。
"""
import random
from pathlib import Path
from typing import Any

from rich import box
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Header, Input, Label, Static

from kaggler.app.tui.commands import COMMANDS, SlashSuggester, hint, parse
from kaggler.app.tui.screens import (
    ConversationAction,
    ConversationListScreen,
    DirectoryBrowserScreen,
    FilePickerScreen,
)
from kaggler.app.tui.widgets import ChatMessage, ContextMeter, TraceLine, TraceTable
from kaggler.modes.common.compute import list_files
from kaggler.persistence.data_export import EXPORT_SUBDIR
from kaggler.shared.session_manager import SessionManager
from kaggler.shared.types import Mode
from kaggler.shared.wrapper import AgentSession
from kaggler.workspace.manager import (
    Workspace,
    get_active_workspace,
    load_last_workspace,
    set_active_workspace,
)

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

_MODE_LABELS: dict[str, str] = {
    "eda": "EDA 探索分析",
    "feature_engineering": "特征工程",
}


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
        self._workspace: Workspace | None = None
        self._session_manager: SessionManager | None = None
        self._streaming_buf: str = ""
        self._streaming_dirty: bool = False
        # 本轮回答对应的消息 widget；token 原地更新它，结束置空。
        self._stream_widget: ChatMessage | None = None
        # 节流定时器：仅在流式期间运行（on_mount 创建为暂停态）。
        self._flush_timer: Timer | None = None
        # 轮次计数：每轮普通问答 +1，用于打通「消息 ↔ 该轮追溯行」。
        self._turn_id: int = 0
        # 当前 node_active 建的活动行，等它的 node_done 到来时原地收尾（▶→✓）。
        self._active_trace: TraceLine | None = None
        # 活动行对应的节点名，防止串行错配到别的节点的 node_done。
        self._active_trace_node: str | None = None

    # ── 布局 ───────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header()
        # 顶部通栏：当前数据集信息（名/类型/绝对路径），为未来工作区加载预留。
        yield Static("", id="file-bar")
        with Horizontal(id="main"):
            # 左栏：单一对话窗
            yield VerticalScroll(id="chat-log")
            # 右栏：上下文占用面板 + Agent 行为追溯
            with Vertical(id="trace-col"):
                yield ContextMeter(id="context-meter")
                yield Label("Agent 行为", classes="panel-title")
                yield VerticalScroll(id="agent-trace")
        with Vertical(id="bottom-bar"):
            yield Static("", id="status-bar")
            # 指令提示行：输入 slash 指令时列出全部候选（命令 / 参数），否则收起。
            yield Static("", id="cmd-hint", markup=False)
            yield Input(
                placeholder="> ", id="user-input", disabled=True,
                suggester=SlashSuggester(),
            )

    # ── 生命周期 ───────────────────────────────────────────────────────────
    def on_mount(self) -> None:
        # 节流定时器：进行中回答整体刷新，避免每 token 重渲染。暂停态创建，
        # 提交问题时 resume、本轮定稿时 pause，空闲不空转。
        self._flush_timer = self.set_interval(
            _FLUSH_INTERVAL, self._flush_streaming, pause=True
        )
        # 锚定两个滚动容器：新内容自动跟随到底部；用户向上滚动即自动解锚、
        # 滚回底部自动恢复（Textual 内建机制，替代逐次手动 scroll_end）。
        self.query_one("#chat-log", VerticalScroll).anchor()
        self.query_one("#agent-trace", VerticalScroll).anchor()
        self.query_one("#status-bar", Static).update(
            Text("输入 /load-file 加载数据集后开始对话 · 拖选文本后 Ctrl+C 复制", "dim")
        )
        # 自动恢复上次工作区：免去每次启动重新 /select-workspace。
        last = load_last_workspace()
        if last is not None:
            self._activate_workspace(last)
            self._system_msg(
                f"已恢复上次工作区 {self._workspace.path}，持久化已启用。"
                "输入 /conversations 查看历史对话，或 /load-file 开始新对话。"
            )
        else:
            self._update_file_bar(None)
            self._system_msg(
                "未设置工作区，持久化功能不可用。输入 /select-workspace 选择工作区目录以启用持久化，"
                "或直接 /load-file 加载数据集（仅内存模式）。"
            )
        self._enable_input()

    def _activate_workspace(self, path: Path | str) -> None:
        """设为当前工作区并建好 SessionManager，刷新信息栏。供启动恢复与 /select-workspace 共用。"""
        self._workspace = set_active_workspace(path)
        self._session_manager = SessionManager(self._workspace.path)
        self._update_file_bar(self._session._csv_path if self._session else None)

    def _cmd_load_file(self) -> None:
        if self._workspace:
            self.push_screen(
                DirectoryBrowserScreen(str(self._workspace.path)),
                self._on_file_browsed,
            )
        else:
            self.push_screen(FilePickerScreen(), self._on_file_picked)

    def _on_file_browsed(self, path: str | None) -> None:
        if not path:
            return
        self._on_file_picked(path)

    def _on_file_picked(self, path: str | None) -> None:
        if not path:
            # 取消选择 → 保持当前状态（不退出程序）。
            return
        if not Path(path).is_file():
            self.notify(f"文件不存在：{path}", severity="error")
            return
        self.query_one("#user-input", Input).disabled = True
        self._add_message("system", f"正在加载数据集 {path} …")
        self.run_worker(lambda: self._init_worker(path), thread=True, name="init")

    # ── 对话窗写入 ─────────────────────────────────────────────────────────
    def _add_message(
        self, role: str, raw: str, *, turn_id: int | None = None, markdown: bool = False
    ) -> ChatMessage:
        """向左栏对话窗追加一条可点击消息，返回该 widget（供流式原地更新）。"""
        widget = ChatMessage(role, raw, turn_id=turn_id, markdown=markdown)
        log = self.query_one("#chat-log", VerticalScroll)
        log.mount(widget)
        log.scroll_end(animate=False)
        return widget

    # ── Worker 函数（后台线程）─────────────────────────────────────────────
    def _init_worker(self, path: str) -> None:
        try:
            if self._session_manager is not None:
                session = self._session_manager.create_conversation(path)
            else:
                session = AgentSession(path)
            self._session = session
            self.post_message(StreamEvent({"type": "init_done", "path": path}))
        except Exception as exc:  # noqa: BLE001 — 后台线程异常需回送到 UI 显示
            self.post_message(StreamEvent({"type": "error", "message": str(exc)}))

    def _resume_worker(self, thread_id: str) -> None:
        try:
            session = self._session_manager.resume_conversation(thread_id)
            self._session = session
            self.post_message(
                StreamEvent(
                    {
                        "type": "init_done",
                        "path": session._csv_path,
                        "resumed": True,
                        "history": session.history(),
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001
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
            # 重建会话：清空左栏与右栏追溯、归零轮次（新数据集 = 新对话）。
            self._clear_conversation()
            self._update_file_bar(e.get("path"))
            if e.get("resumed"):
                self._replay_history(e.get("history") or [])
                self._add_message(
                    "system",
                    "已恢复对话。历史消息已在上方重放；助手仍保留之前的上下文记忆。"
                    "（注：右栏 Agent 行为追溯不重建；更早被压缩的历史仅存于摘要、不在此显示。）",
                )
            else:
                self._add_message("assistant", random.choice(_GREETINGS))
            self._update_status_bar("eda")
            self._enable_input()

        elif t == "token":
            # 只累积 + 标脏，不在这里更新 widget（由 _flush_streaming 节流刷新）。
            self._streaming_buf += e["content"]
            self._streaming_dirty = True

        elif t == "node_active":
            self._handle_node_active(e["node"])

        elif t == "node_done":
            self._handle_node_done(e["node"], e.get("tool_calls", []))

        elif t == "mode_change":
            self._update_status_bar(e["mode"])

        elif t == "context":
            self._update_context_meter(e["usage"])

        elif t == "turn_done":
            self._finalize_streaming()
            self._enable_input()

        elif t == "error":
            self._finalize_streaming()
            self._add_message("error", e.get("message", "未知错误"))
            self._enable_input()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return
        event.input.value = ""
        # slash 指令走确定性同步分支：不 disable 输入、不预挂助手 widget、不起 worker。
        # 放在 session 守卫之前——/load-file 等指令在未加载数据集时也须可用。
        if value.startswith("/"):
            self._handle_slash_command(value)
            return
        if self._session is None:
            self._system_msg("尚未加载数据集，请先用 /load-file 选择 CSV 数据集。")
            return
        event.input.disabled = True
        self._streaming_buf = ""
        self._streaming_dirty = False
        # 推进轮次：用户消息、本轮回答、本轮追溯行共享同一 turn_id，供点击联动。
        self._turn_id += 1
        self._add_message("user", value, turn_id=self._turn_id)
        # 为本轮回答预挂一个 widget，后续 token 原地更新它（同一窗、同一条消息）。
        self._stream_widget = self._add_message(
            "assistant", "", turn_id=self._turn_id
        )
        self._flush_timer.resume()
        self.run_worker(
            lambda: self._stream_worker(value), thread=True, name="stream"
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        # 随输入实时刷新指令提示行：列出全部可用候选（命令 / 参数）。
        self.query_one("#cmd-hint", Static).update(hint(event.value))

    def on_chat_message_clicked(self, event: ChatMessage.Clicked) -> None:
        """演示联动：点击消息 → 高亮该条 + 右栏同轮的追溯行并滚动可见。"""
        msg = event.source
        # 清除上一次的高亮。
        for m in self.query(ChatMessage):
            m.remove_class("selected")
        for tl in self.query(TraceLine):
            tl.remove_class("linked")
        msg.add_class("selected")
        if msg.turn_id is None:
            return
        linked = [tl for tl in self.query(TraceLine) if tl.turn_id == msg.turn_id]
        for tl in linked:
            tl.add_class("linked")
        if linked:
            linked[0].scroll_visible()

    # ── Slash 指令（确定性、同步、不经 LLM）────────────────────────────────
    def _system_msg(self, msg: str) -> None:
        self._add_message("system", msg)

    def _handle_slash_command(self, raw: str) -> None:
        # 回显指令入对话历史，与普通消息一致（指令不推进轮次、不渲染 markdown）。
        self._add_message("user", raw)
        name, args = parse(raw)
        if name == "switch":
            self._cmd_switch(args)
        elif name == "load-file":
            self._cmd_load_file()
        elif name == "select-workspace":
            self._cmd_select_workspace()
        elif name == "ls":
            self._cmd_ls(args)
        elif name == "export":
            self._cmd_export(args)
        elif name == "conversations":
            self._cmd_conversations()
        elif name == "exit":
            self.exit()
        else:
            avail = "、".join(f"/{c}" for c in COMMANDS)
            self._system_msg(f"未知指令 /{name}，可用：{avail}")

    def _cmd_switch(self, args: list[str]) -> None:
        if self._session is None:
            self._system_msg("尚未加载数据集，请先用 /load-file 加载后再切换模式。")
            return
        valid = " / ".join(m.value for m in Mode)
        if not args:
            self._system_msg(f"用法：/switch <mode>，可用模式：{valid}")
            return
        try:
            mode = Mode(args[0])
        except ValueError:
            self._system_msg(f"未知模式：{args[0]}，可用模式：{valid}")
            return
        self._session.set_mode(mode)
        self._update_status_bar(mode.value)
        self._system_msg(f"已切换到 {_MODE_LABELS.get(mode.value, mode.value)}")

    def _cmd_select_workspace(self) -> None:
        self.push_screen(
            DirectoryBrowserScreen(select_dir_mode=True), self._on_workspace_picked
        )

    def _on_workspace_picked(self, path: str | None) -> None:
        if not path:
            return
        p = Path(path)
        if not p.is_dir():
            self.notify("不是有效目录", severity="error")
            return
        self._activate_workspace(p)
        self._system_msg(f"工作区已设置为 {self._workspace.path}，持久化已启用。")

    def _cmd_ls(self, args: list[str]) -> None:
        ws = get_active_workspace()
        if ws is None:
            self._system_msg("未设置工作区，请先用 /select-workspace 选择工作区目录。")
            return
        subdir = args[0] if args else ""
        target = ws.resolve_within(subdir)
        if target is None:
            self._system_msg("不允许访问工作区之外的路径。")
            return
        try:
            output = list_files(target)
        except Exception as exc:
            output = f"列出文件失败：{exc}"
        self._add_message("system", output, markdown=True)

    def _cmd_export(self, args: list[str]) -> None:
        if self._session is None:
            self._system_msg("尚未加载数据集，请先用 /load-file 加载后再导出。")
            return
        ws = get_active_workspace()
        if ws is None:
            self._system_msg("未设置工作区，请先用 /select-workspace 选择工作区目录。")
            return
        # 解析 /export [版本号] [路径]：首个纯数字 arg 视作版本号，其余视作路径。
        version: int | None = None
        path_arg: str | None = None
        for a in args:
            if version is None and path_arg is None and a.isdigit():
                version = int(a)
            else:
                path_arg = a
        v = version if version is not None else self._session.current_data_version
        # 路径规则(受控目录默认 + 可选外部)：缺省 → 受控目录下 version_{v}.csv；
        # 绝对路径 → 原样落盘(允许工作区外)；相对路径 → 受控目录内。
        if path_arg is None:
            target = ws.resolve_within(f"{EXPORT_SUBDIR}/version_{v}.csv")
        elif Path(path_arg).is_absolute():
            target = Path(path_arg)
        else:
            target = ws.resolve_within(f"{EXPORT_SUBDIR}/{path_arg}")
        if target is None:
            self._system_msg("不允许导出到工作区之外的相对路径（如需外部导出请给绝对路径）。")
            return
        db_path = (
            self._session_manager.workspace.data_version_db
            if self._session_manager is not None
            else None
        )
        try:
            result = self._session.export_data_version(version, target, db_path=db_path)
        except (RuntimeError, ValueError, OSError) as exc:
            self._system_msg(f"导出失败：{exc}")
            return
        self._system_msg(
            f"已导出版本 {result.version} 到 {result.path}"
            f"（{result.rows} 行 × {result.cols} 列，{result.format}）"
        )

    def _cmd_conversations(self) -> None:
        if self._session_manager is None:
            self._system_msg(
                "未设置工作区，无法管理对话。请先用 /select-workspace 选择工作区目录。"
            )
            return
        records = self._session_manager.list_conversations()
        self.push_screen(
            ConversationListScreen(records), self._on_conversation_action
        )

    def _on_conversation_action(self, result: ConversationAction | None) -> None:
        if result is None or self._session_manager is None:
            return
        if result.action == "resume":
            self.query_one("#user-input", Input).disabled = True
            self._add_message("system", "正在恢复对话 …")
            self.run_worker(
                lambda: self._resume_worker(result.thread_id),
                thread=True,
                name="init",
            )
        elif result.action == "rename":
            self._session_manager.rename_conversation(result.thread_id, result.new_name)
            self._system_msg(f"已重命名为 {result.new_name}")
        elif result.action == "delete":
            self._session_manager.delete_conversation(result.thread_id)
            self._system_msg("已删除对话")

    # ── 流式回答（原地更新当前消息）──────────────────────────────────────
    def _flush_streaming(self) -> None:
        """节流刷新：仅在有新 token 时整体重绘进行中那条消息（纯文本 + 光标）。"""
        if not self._streaming_dirty or self._stream_widget is None:
            return
        self._streaming_dirty = False
        self._stream_widget.append_cursor(self._streaming_buf)

    def _finalize_streaming(self) -> None:
        """本轮结束：去掉光标并渲染为 markdown 定稿；若无内容则移除占位 widget。"""
        widget = self._stream_widget
        self._stream_widget = None
        if widget is not None:
            if self._streaming_buf:
                widget.set_content(self._streaming_buf, markdown=True)
            else:
                widget.remove()
        self._streaming_buf = ""
        self._streaming_dirty = False
        # 本轮收尾：清空活动行引用，避免下一轮的 node_done 误改这条已定稿的行。
        self._active_trace = None
        self._active_trace_node = None
        if self._flush_timer is not None:
            self._flush_timer.pause()

    def _update_status_bar(self, mode: str) -> None:
        label = _MODE_LABELS.get(mode, mode)
        self.query_one("#status-bar", Static).update(
            Text.assemble(
                ("模式：", "dim"), (label, "bold cyan"),
                ("    点击消息可高亮该轮 Agent 行为 · 拖选文本后 Ctrl+C 复制", "dim"),
            )
        )

    def _update_context_meter(self, usage: dict) -> None:
        """刷新右栏「上下文占用」面板（context 事件驱动，每轮数次、非每 token）。"""
        self.query_one("#context-meter", ContextMeter).update_usage(usage)

    def _enable_input(self) -> None:
        inp = self.query_one("#user-input", Input)
        inp.disabled = False
        inp.focus()

    def _clear_conversation(self) -> None:
        """重建会话：清空左栏对话与右栏追溯，归零轮次与活动行引用。

        保留 workspace / session_manager 引用（工作区设置不应随数据集切换丢失）。
        """
        self._turn_id = 0
        self._active_trace = None
        self._active_trace_node = None
        self.query_one("#chat-log", VerticalScroll).remove_children()
        self.query_one("#agent-trace", VerticalScroll).remove_children()
        self.query_one("#context-meter", ContextMeter).clear()

    def _replay_history(self, history: list[dict[str, str]]) -> None:
        """恢复对话时把 checkpoint 中的历史消息重绘到左栏（用户/助手两类）。"""
        for msg in history:
            role = msg.get("role", "system")
            self._add_message(
                role, msg.get("content", ""), markdown=(role == "assistant")
            )

    def _update_file_bar(self, path: str | None) -> None:
        """刷新顶部信息栏：工作区路径 + 数据集文件名 · 类型 · 绝对路径。"""
        bar = self.query_one("#file-bar", Static)
        segments: list[tuple[str, str]] = []

        if self._workspace:
            segments.append(("工作区: ", "dim"))
            segments.append((str(self._workspace.path), "bold magenta"))

        if path:
            if segments:
                segments.append(("  ·  ", "dim"))
            p = Path(path).resolve()
            ftype = p.suffix[1:].upper() if p.suffix else "文件"
            segments.append(("", ""))
            segments.append((p.name, "bold cyan"))
            segments.append(("  ·  ", "dim"))
            segments.append((ftype, "green"))
            segments.append(("  ·  ", "dim"))
            segments.append((str(p), "dim"))
        elif not segments:
            segments.append(("未加载数据集 · 输入 /load-file 选择 CSV", "dim"))

        bar.update(Text.assemble(*segments))

    # ── Agent 行为追溯（每个节点访问一条，原地 ▶→✓ 收尾）──────────────────────
    def _handle_node_active(self, node: str) -> None:
        # node_active：建一条「▶ 生成中…」行并记为活动行，等 node_done 原地收尾。
        if node in _SILENT_NODES:
            return
        label = _NODE_LABELS.get(node, node)
        tl = TraceLine(
            self._trace_text("▶", "yellow", label, "生成中…"), turn_id=self._turn_id
        )
        # 先存引用再 mount：mount() 返回 AwaitMount 而非部件本身。
        self._active_trace = tl
        self._active_trace_node = node
        self.query_one("#agent-trace", VerticalScroll).mount(tl)

    def _handle_node_done(self, node: str, tool_calls: list[dict]) -> None:
        # node_done：把本节点的活动行原地改成「✓ 完成」；react 决策的工具批次内联成表。
        if node in _SILENT_NODES:
            return
        label = _NODE_LABELS.get(node, node)
        done = self._trace_text("✓", "green", label, _NODE_DESC.get(node, "完成"))
        log = self.query_one("#agent-trace", VerticalScroll)
        if self._active_trace is not None and self._active_trace_node == node:
            self._active_trace.update(done)  # 原地收尾：▶→✓、黄→绿、生成中→完成
        else:
            # 兜底：没有匹配的活动行（不该发生），退化为新挂一条完成行。
            log.mount(TraceLine(done, turn_id=self._turn_id))
        if tool_calls:
            log.mount(
                TraceTable(self._build_tool_table(tool_calls), turn_id=self._turn_id)
            )
        self._active_trace = None
        self._active_trace_node = None

    def _trace_text(self, icon: str, icon_color: str, label: str, desc: str) -> Text:
        return Text.assemble(
            (f"{icon} ", icon_color),
            (f"{str(label):<{_LABEL_WIDTH}}", "cyan"),
            (desc, ""),
        )

    def _build_tool_table(self, tool_calls: list[dict]) -> Table:
        # 窄栏（3fr）：轻量线框 + 撑满栏宽 + 长文换行不裁断。
        table = Table(box=box.SIMPLE, expand=True, pad_edge=False, show_edge=True)
        table.add_column("工具", style="cyan", overflow="fold")
        table.add_column("参数", overflow="fold")
        for tc in tool_calls:
            args = tc.get("args", {})
            arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items()) or "—"
            table.add_row(tc.get("name", ""), arg_str)
        return table


def main() -> None:
    KagglerTUI().run()


if __name__ == "__main__":
    main()
