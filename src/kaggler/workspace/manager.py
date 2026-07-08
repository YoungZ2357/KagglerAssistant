"""工作区管理：代表一个工作目录，其下的 .kaggler/ 存放该工作区的所有持久化数据。

设计原则：
- 本模块提供 Workspace 类用于在 composition root 处获取路径，把 Path 值作为
  普通参数传给下游消费者（SqliteSaver、ConversationStore 等），不传递 Workspace 对象。
- 全局 active workspace 状态仅在模块内维护，由 SessionManager 调用。
- 「上次工作区」记忆落在用户级目录（~/.kaggler），与具体工作区无关，供启动时自动恢复。
"""

from __future__ import annotations

from pathlib import Path

# 工作区内持久化布局的固定名称（单一事实源，勿散落成魔法串）。
_KAGGLER_DIR = ".kaggler"
_CHECKPOINT_DB = "checkpoints.sqlite"
_CONVERSATION_DB = "conversations.sqlite"

# 用户级状态（跨工作区、跨会话），记录最近一次使用的工作区路径。
_USER_STATE_DIR = Path.home() / ".kaggler"
_LAST_WORKSPACE_FILE = _USER_STATE_DIR / "last_workspace"

_active: Workspace | None = None


class Workspace:
    """代表一个工作目录，管理该目录下的 .kaggler 持久化布局。"""

    def __init__(self, path: Path) -> None:
        self._path = Path(path).resolve()
        self._kaggler = self._path / _KAGGLER_DIR

    @property
    def path(self) -> Path:
        return self._path

    @property
    def checkpoint_db(self) -> Path:
        return self._kaggler / _CHECKPOINT_DB

    @property
    def conversation_db(self) -> Path:
        return self._kaggler / _CONVERSATION_DB

    def ensure_layout(self) -> Path:
        self._kaggler.mkdir(parents=True, exist_ok=True)
        return self._kaggler

    def resolve_within(self, subpath: str) -> Path | None:
        """把相对子路径安全解析为工作区内的绝对路径；越界（含 ``..`` 逃逸）返回 None。

        用 ``Path.is_relative_to`` 判断包含关系，而非字符串前缀匹配——后者会把
        ``/ws-evil`` 误判为在 ``/ws`` 之内。
        """
        target = (self._path / subpath).resolve()
        if target.is_relative_to(self._path):
            return target
        return None

    def __repr__(self) -> str:
        return f"Workspace({self._path})"


def get_active_workspace() -> Workspace | None:
    return _active


def set_active_workspace(path: Path | str) -> Workspace:
    global _active
    _active = Workspace(Path(path))
    _active.ensure_layout()
    save_last_workspace(_active.path)
    return _active


def save_last_workspace(path: Path | str) -> None:
    """记录最近使用的工作区路径到用户级状态文件；写失败静默忽略（非关键路径）。"""
    try:
        _USER_STATE_DIR.mkdir(parents=True, exist_ok=True)
        _LAST_WORKSPACE_FILE.write_text(str(Path(path).resolve()), encoding="utf-8")
    except OSError:
        pass


def load_last_workspace() -> Path | None:
    """读取上次工作区路径；文件缺失/内容失效/已非目录则返回 None。"""
    try:
        raw = _LAST_WORKSPACE_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_dir() else None
