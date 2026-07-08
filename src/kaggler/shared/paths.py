"""KagglerAssistant 的路径单一事实源 (path SSOT)。

所有落盘位置都从一个 root 派生。在 composition root 处调用这里的函数解析出
Path 值，把 **Path 值本身** 作为普通参数传给持久化等消费者——不要把本模块
或某个 "workspace 对象" 传来传去（那会重新引入依赖注入耦合）。

设计约束（刻意为之，勿改成 module-level 常量）：
- 全部做成函数（lazy），不是 import 期解析的常量。原因见下方 root()。
- import 本模块不产生任何 IO，也不做任何决定；建目录由 ensure_layout() 显式触发。
- 只声明 **已有消费者** 的路径。新增路径 = 新增一个 3 行函数，等消费者出现再加。
"""

from __future__ import annotations

import os
from pathlib import Path

# root 覆盖入口。未设置 => 项目相对默认值。
# 首要用途：测试把 root 指向临时目录；次要用途：将来的用户自托管。
_ROOT_ENV = "KAGGLER_HOME"

# 项目根：从本文件位置派生，**不用 CWD**（CWD 随启动方式漂移，不稳定）。
# 本文件位于 <project>/src/kaggler/shared/paths.py，故 parents[3] 为 <project>：
# parents[0]=shared, [1]=kaggler, [2]=src, [3]=<project root>。
# 若你把它挪到别的层级，改这里的下标即可（这是全模块唯一的位置耦合点）。
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ROOT = _PROJECT_ROOT / ".kaggler"


def root() -> Path:
    """返回 KagglerAssistant 的落盘 root。

    每次调用重新读取环境变量，**不缓存**：这样测试可以在每个用例里把
    KAGGLER_HOME 指向各自的临时目录，且覆盖立即生效。缓存(lru_cache)会
    让首次解析后的 env 改动失效，破坏可测性——故意不加。
    """
    override = os.environ.get(_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_ROOT


def checkpoint_db() -> Path:
    """SqliteSaver 的 checkpoint 数据库文件路径。

    做成函数（而非 module-level 常量）：常量会在 import 期把 root 冻结，
    使 KAGGLER_HOME 覆盖来不及生效。函数保证它始终跟随当前 root()。
    """
    return root() / "checkpoints.sqlite"


def conversation_db() -> Path:
    """ConversationStore 的对话元数据数据库文件路径。

    存储对话名称、thread_id、工作区路径等应用层元数据，与 checkpoint_db
    分离——checkpoint 的 schema 由 SqliteSaver 管理，不可混入应用层表。
    """
    return root() / "conversations.sqlite"


def ensure_layout() -> Path:
    """在启动时显式建好 root 目录，返回 root。

    只在应用启动/测试 setup 处调用一次，不要放进 import 或库代码里。
    """
    r = root()
    r.mkdir(parents=True, exist_ok=True)
    return r


# --- 如何生长（示例，勿提前启用）------------------------------------------
# 当 BaseStore 真正成为消费者时，加一个函数即可，别现在就建：
#
#     def store_db() -> Path:
#         return root() / "store.sqlite"
#
# 当 thread_id 注册表需要一个落点时（它在 checkpoint 之外，属应用层）：
#
#     def thread_registry() -> Path:
#         return root() / "threads.json"
