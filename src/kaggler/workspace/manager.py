"""工作区管理：代表一个工作目录，其下的 .kaggler/ 存放该工作区的所有持久化数据。

设计原则（与 shared/paths.py 一致）：
- 本模块提供 Workspace 类用于在 composition root 处获取路径，把 Path 值作为
  普通参数传给下游消费者（SqliteSaver、ConversationStore 等），不传递 Workspace 对象。
- 全局 active workspace 状态仅在模块内维护，由 SessionManager 调用。
"""

from __future__ import annotations

from pathlib import Path

_KAGGLER_DIR = ".kaggler"

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
        return self._kaggler / "checkpoints.sqlite"

    @property
    def conversation_db(self) -> Path:
        return self._kaggler / "conversations.sqlite"

    def ensure_layout(self) -> Path:
        self._kaggler.mkdir(parents=True, exist_ok=True)
        return self._kaggler

    def __repr__(self) -> str:
        return f"Workspace({self._path})"


def get_active_workspace() -> Workspace | None:
    return _active


def set_active_workspace(path: Path | str) -> Workspace:
    global _active
    _active = Workspace(Path(path))
    _active.ensure_layout()
    return _active
