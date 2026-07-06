"""DataProvider —— 带活跃版本管理的版本存储层。

设计要点(与项目不变量对齐):
- 版本身份(存在性/版本号)与物化状态彻底解耦:_lineage 是唯一真相,_materialized 是其子集。
- 物化/淘汰不改变 data_version token —— 逻辑版本身份对上层透明,是纯存储层。
- 派生 op 是纯内存闭包(写请求隔离机制已废弃,无需可序列化 spec)。
- source 与派生版本统一建模:source 持 loader(无 parent),派生版本持 op(有 parent),
  为持久化/工作区重载留出干净接缝——重载的持久化版本即一个新 source。
- 可重算性由「root/source 恒可通过 loader 重建」保证,因此淘汰任何非 pin、非 HEAD 的
  物化版本永远安全,不需要深度守卫。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import polars as pl

logger = logging.getLogger(__name__)

# 派生 op:把父版本的 LazyFrame 变换为本版本的 LazyFrame。
# 拟合类操作(标准化/编码/中位数填充)须把拟合常量固化进闭包,或写成在父帧上确定性重算,
# 保证 op 严格闭合于「不可变父版本」这一个输入。
Op = Callable[[pl.LazyFrame], pl.LazyFrame]

# source loader:无参、产出 eager DataFrame。CSV 源 / 持久化重载都走这里。
Loader = Callable[[], pl.DataFrame]


@dataclass
class VersionInfo:
    parent: int | None
    tool: str | None
    description: str
    # 语义:「给定不可变父版本能否确定性复现」。seeded 随机(split/sample 带固定 seed)= True。
    # 仅未固定 seed 的真随机 = False —— 这类会被强制物化并 pin。
    reproducible: bool = True


class DataProvider:
    def __init__(
        self,
        *,
        max_materialized: int = 3,
        cache_on_read: bool = False,
        pin_root: bool = True,
    ) -> None:
        """
        max_materialized: 同时常驻的物化版本数上限(诉求1:控制活跃版本数量)。
                         必须 >= 被 pin 的版本数 + 1,否则无淘汰余量。
        cache_on_read:    读取一个 lazy 版本后是否将其物化并纳入 LRU(诉求2:同步时机)。
                          默认 False —— HEAD 与 rollback 目标本就常驻,余下多为一次性的
                          版本对比读取,即算即弃更省内存。若某老版本被高频重读再开 True。
        pin_root:         root 是否常驻(诉求2:优先保留什么)。默认 True,避免每次重算重读 CSV。
                          置 False 时 root 可被淘汰,重算时经 loader 重读文件——正确但慢。
        """
        # ---- 版本身份 / 血缘:覆盖全部版本,存在性的唯一真相 ----
        self._lineage: dict[int, VersionInfo] = {}
        self._ops: dict[int, Op] = {}          # 派生版本 -> op(source 不在此)
        self._loaders: dict[int, Loader] = {}  # source -> loader(派生版本不在此)

        # ---- 物化缓存:全部版本的子集 ----
        self._materialized: dict[int, pl.DataFrame] = {}
        self._last_access: dict[int, float] = {}
        self._pinned: set[int] = set()  # 永不淘汰:root(可选)/ 不可复现版本 / 用户显式 pin

        self._next_version = 0  # 单调计数器,与物化集解耦(修复版本号碰撞)
        self._root: int | None = None
        self._head: int | None = None

        self._max_materialized = max_materialized
        self._cache_on_read = cache_on_read
        self._pin_root = pin_root

    # ================= 构造 / source =================

    def add_source(
        self,
        loader: Loader,
        *,
        description: str,
        tool: str | None = None,
        pin: bool | None = None,
    ) -> int:
        """注册一个 source(无父版本,由 loader 产出)。

        供 load_initial 与持久化重载共用。持久化重载时传入 read_parquet loader,
        使被保存的版本以「新 source」身份回归——数据保住,上游 op 血缘不再可重放(符合预期)。
        """
        v = self._alloc()
        self._lineage[v] = VersionInfo(parent=None, tool=tool, description=description)
        self._loaders[v] = loader
        self._materialize(v)  # source 注册即加载
        if pin is None:
            pin = self._pin_root
        if pin:
            self._pinned.add(v)
        if self._root is None:
            self._root = v
        self._head = v
        self._evict_if_needed()
        self._check_budget()
        return v

    def load_initial(self, path: str) -> int:
        # loader 闭包捕获 path;持久化重载对应版本时改用 lambda: pl.read_parquet(file)。
        return self.add_source(lambda: pl.read_csv(path), description="原始数据集")

    # ================= 写:派生新版本 =================

    def add_version(
        self,
        op: Op,
        *,
        parent: int,
        tool: str | None = None,
        description: str = "",
        reproducible: bool = True,
        pin: bool = False,
    ) -> int:
        """在 parent 之上应用 op 派生新版本,并使其成为新 HEAD。

        reproducible=False:不可复现的真随机 —— 立即物化并强制 pin(重算会与原值分叉,
                            会污染以 data_version 为 key 的下游缓存;持久化时也必须落盘)。
        pin=True:          可复现但昂贵的派生(编码/大 groupby/join)—— 保护其结果不被
                            反复重放穿越。取代自动深度守卫:显式、针对真实成本。
        """
        if parent not in self._lineage:
            raise RuntimeError(f"父版本 `{parent}` 不存在")

        v = self._alloc()
        self._lineage[v] = VersionInfo(parent, tool, description, reproducible)
        self._ops[v] = op

        # 新 HEAD 立即物化:当前所有分析都打在 HEAD 上,懒化收益来自旧版本降级而非 HEAD 本身。
        self._head = v
        self._materialize(v)

        if not reproducible:
            self._pinned.add(v)
        elif pin:
            self._pinned.add(v)

        self._evict_if_needed()
        self._check_budget()
        return v

    # ================= 读 =================

    def get(self, data_version: int) -> pl.DataFrame:
        """返回指定版本的 eager DataFrame(交给工具层的始终是 collected 帧)。"""
        if data_version not in self._lineage:  # 查血缘,不查物化缓存
            raise RuntimeError(f"数据版本 `{data_version}` 不存在")
        if data_version in self._materialized:
            self._touch(data_version)
            return self._materialized[data_version]

        df = self._compute(data_version)
        if self._cache_on_read:
            self._materialized[data_version] = df
            self._touch(data_version)
            self._evict_if_needed()
        return df

    # ================= 版本切换 / 回滚 =================

    def set_head(self, version: int) -> None:
        """将 HEAD 切到目标版本(回滚 / 切分支)。目标被物化;被离开的旧 HEAD 因不再等于
        _head 自动降级为可淘汰——无需手动 unpin,结构上不留僵尸 pin。
        """
        if version not in self._lineage:
            raise RuntimeError(f"数据版本 `{version}` 不存在")
        self._head = version
        self._materialize(version)
        self._evict_if_needed()

    # ================= 手动 pin 控制 =================

    def pin(self, version: int) -> None:
        if version not in self._lineage:
            raise RuntimeError(f"数据版本 `{version}` 不存在")
        self._pinned.add(version)

    def unpin(self, version: int) -> None:
        if not self._lineage[version].reproducible:
            raise RuntimeError(f"版本 `{version}` 不可复现,必须常驻,不可 unpin")
        self._pinned.discard(version)

    # ================= 谱系 / 内省 =================

    def get_version_info(self, version: int) -> VersionInfo:
        if version not in self._lineage:
            raise RuntimeError(f"数据版本 `{version}` 不存在")
        return self._lineage[version]

    def list_versions(self) -> list[dict]:
        """按版本号升序返回全部版本(含 lazy 版本)的谱系与运行时状态,供 TUI / 浏览工具使用。"""
        return [
            {
                "version": v,
                "is_head": v == self._head,
                "materialized": v in self._materialized,
                "pinned": v in self._pinned,
                **asdict(self._lineage[v]),
            }
            for v in sorted(self._lineage)
        ]

    @property
    def head(self) -> int | None:
        return self._head

    @property
    def root(self) -> int | None:
        return self._root

    # ================= 内部 =================

    def _alloc(self) -> int:
        v = self._next_version
        self._next_version += 1
        return v

    def _touch(self, version: int) -> None:
        self._last_access[version] = time.monotonic()

    def _materialize(self, version: int) -> None:
        if version not in self._materialized:
            self._materialized[version] = self._compute(version)
        self._touch(version)

    def _compute(self, version: int) -> pl.DataFrame:
        """回溯到最近的已物化祖先(或 source),正向 replay,单次融合 collect。"""
        chain: list[int] = []
        cur = version
        while cur not in self._materialized:
            node = self._lineage[cur]
            if node.parent is None:  # 到达 source
                break
            chain.append(cur)
            cur = node.parent

        if cur in self._materialized:
            base = self._materialized[cur]
        else:
            base = self._loaders[cur]()  # source 未物化 -> 经 loader 按需重建

        if not chain:
            return base

        lf = base.lazy()
        for vid in reversed(chain):  # 从最靠近 base 的一步开始向 version 方向 replay
            lf = self._ops[vid](lf)
        return lf.collect()  # 整条链融合为一次带 pushdown 的 collect

    def _evict_if_needed(self) -> None:
        while len(self._materialized) > self._max_materialized:
            candidates = [
                v
                for v in self._materialized
                if v not in self._pinned and v != self._head
            ]
            if not candidates:
                # 全被 pin / 均为 HEAD:宁可暂时超预算也不破坏正确性。
                logger.warning(
                    "物化预算 %d 已被 pin+HEAD 占满(当前 %d),暂时超预算",
                    self._max_materialized,
                    len(self._materialized),
                )
                break
            victim = min(candidates, key=lambda v: self._last_access[v])  # LRU
            del self._materialized[victim]  # op/loader + 血缘仍在,版本仍可重算
            self._last_access.pop(victim, None)

    def _check_budget(self) -> None:
        # HEAD 恒物化,故 pin 数需给 HEAD 留一个位置。
        reserved = len(self._pinned | {self._head} if self._head is not None else self._pinned)
        if reserved > self._max_materialized:
            logger.warning(
                "max_materialized=%d 已不足以容纳 %d 个受保护版本(pin+HEAD),"
                "淘汰将无余量。考虑调大 max_materialized 或减少 pin。",
                self._max_materialized,
                reserved,
            )

    # ================= 过渡期适配(LazyFrame 重构完成后可删) =================

    @staticmethod
    def eager_op(fn: Callable[[pl.DataFrame], pl.DataFrame]) -> Op:
        """把旧式 (DataFrame)->DataFrame 变换包装成 lazy Op,用于工具尚未 lazy 化的过渡期。
        注意:中途 collect 会打断链融合,仅作迁移桥梁,不应长期保留。
        """
        def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
            return fn(lf.collect()).lazy()

        return _op
