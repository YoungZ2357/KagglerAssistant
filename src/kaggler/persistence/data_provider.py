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
from typing import Protocol

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


class VersionSink(Protocol):
    """持久化端口:DataProvider 每登记一个版本就通知一次(由组合根注入具体实现)。

    保持 DataProvider 为纯内存存储层——它只调用这个协议,不认识 sqlite / thread_id。
    """

    def record_version(
        self,
        version: int,
        *,
        parent: int | None,
        kind: str,  # 'source' | 'derived'
        tool: str | None,
        description: str,
        reproducible: bool,
        code: str | None,
    ) -> None: ...


class DataProvider:
    def __init__(
        self,
        *,
        max_materialized: int = 3,
        cache_on_read: bool = False,
        pin_root: bool = True,
        sink: VersionSink | None = None,
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
        # 每个版本对应的 Polars 代码片段(source 存读取表达式,派生版本存操作 lf 的语句);
        # None 表示该步无法生成代码(如 eager_op 桥),导出管道脚本时据此响亮报错。
        self._pipeline_code: dict[int, str | None] = {}

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
        # 持久化端口(可选):add_source/add_version 登记后通知;None 则纯内存(如 CLI/单测)。
        self._sink = sink

    # ================= 构造 / source =================

    def add_source(
        self,
        loader: Loader,
        *,
        description: str,
        tool: str | None = None,
        pin: bool | None = None,
        code: str | None = None,
    ) -> int:
        """注册一个 source(无父版本,由 loader 产出)。

        供 load_initial 与持久化重载共用。持久化重载时传入 read_parquet loader,
        使被保存的版本以「新 source」身份回归——数据保住,上游 op 血缘不再可重放(符合预期)。

        code:该 source 的 eager 读取表达式源码(如 ``pl.read_csv('train.csv')``),
              供导出管道脚本时作为链首;None 则该版本不可作为脚本起点。
        """
        v = self._alloc()
        self._lineage[v] = VersionInfo(parent=None, tool=tool, description=description)
        self._loaders[v] = loader
        self._pipeline_code[v] = code
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
        if self._sink is not None:
            self._sink.record_version(
                v, parent=None, kind="source", tool=tool,
                description=description, reproducible=True, code=code,
            )
        return v

    def load_initial(self, path: str) -> int:
        # loader 闭包捕获 path;持久化重载对应版本时改用 lambda: pl.read_parquet(file)。
        return self.add_source(
            lambda: pl.read_csv(path),
            description="原始数据集",
            code=f"pl.read_csv({path!r})",
        )

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
        code: str | None = None,
    ) -> int:
        """在 parent 之上应用 op 派生新版本,并使其成为新 HEAD。

        reproducible=False:不可复现的真随机 —— 立即物化并强制 pin(重算会与原值分叉,
                            会污染以 data_version 为 key 的下游缓存;持久化时也必须落盘)。
        pin=True:          可复现但昂贵的派生(编码/大 groupby/join)—— 保护其结果不被
                            反复重放穿越。取代自动深度守卫:显式、针对真实成本。
        code:              与 op 等价的 Polars 代码片段(操作变量 ``lf`` 的语句);None 表示
                            该步无法生成代码,导出管道脚本时会响亮报错而非产出残缺脚本。
        """
        if parent not in self._lineage:
            raise RuntimeError(f"父版本 `{parent}` 不存在")

        v = self._alloc()
        self._lineage[v] = VersionInfo(parent, tool, description, reproducible)
        self._ops[v] = op
        self._pipeline_code[v] = code

        # 新 HEAD 立即物化:当前所有分析都打在 HEAD 上,懒化收益来自旧版本降级而非 HEAD 本身。
        self._head = v
        self._materialize(v)

        if not reproducible:
            self._pinned.add(v)
        elif pin:
            self._pinned.add(v)

        self._evict_if_needed()
        self._check_budget()
        if self._sink is not None:
            self._sink.record_version(
                v, parent=parent, kind="derived", tool=tool,
                description=description, reproducible=reproducible, code=code,
            )
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

    # ================= 恢复重建(从持久化账本重放) =================

    def restore_source(
        self,
        version: int,
        *,
        description: str,
        tool: str | None,
        code: str | None,
        loader: Loader,
    ) -> None:
        """用给定版本号登记一个 source,不 _alloc、不写 sink、不立即物化(惰性)。

        专供恢复对话时按账本重建版本树;版本号沿用持久化时的原值以对齐 checkpoint 的
        data_version。物化推迟到重建结束后由 set_head(恢复点)触发。
        """
        self._lineage[version] = VersionInfo(parent=None, tool=tool, description=description)
        self._loaders[version] = loader
        self._pipeline_code[version] = code
        if self._root is None:
            self._root = version
        if self._pin_root:
            self._pinned.add(version)
        self._head = version
        self._next_version = max(self._next_version, version + 1)

    def restore_derived(
        self,
        version: int,
        *,
        parent: int,
        tool: str | None,
        description: str,
        reproducible: bool,
        op: Op,
        code: str | None,
    ) -> None:
        """用给定版本号登记一个派生版本,不 _alloc、不写 sink、不立即物化(惰性)。"""
        self._lineage[version] = VersionInfo(parent, tool, description, reproducible)
        self._ops[version] = op
        self._pipeline_code[version] = code
        if not reproducible:
            self._pinned.add(version)
        self._head = version
        self._next_version = max(self._next_version, version + 1)

    def generate_pipeline_code(
        self,
        version: int,
        *,
        output_path: str | None = None,
        output_fmt: str = "csv",
    ) -> str:
        """生成复现 ``version`` 的自包含 Polars 管道脚本。

        沿 _lineage 从 version 回溯到 source(parent 为 None),拼接各版本存下的代码片段——
        与 _compute 的 replay 同构,只是「拼代码」而非「调闭包」,故产出严格等价于
        ``get(version)`` 的预处理链。拟合常量已在片段中写死,脚本无需重新拟合。

        output_path 给定时追加 ``df.write_csv/parquet(output_path)``;否则给出注释示例。

        Raises:
            RuntimeError: version 不存在。
            ValueError:   链中任一版本无代码片段(如 eager_op 桥 / 无种子随机),
                          脚本无法完整复现,响亮报错而非产出残缺脚本。
        """
        if version not in self._lineage:
            raise RuntimeError(f"数据版本 `{version}` 不存在")

        # 回溯 source..version 的有序链(source 在前)。
        chain: list[int] = []
        cur = version
        while True:
            chain.append(cur)
            parent = self._lineage[cur].parent
            if parent is None:
                break
            cur = parent
        chain.reverse()

        source = chain[0]
        src_code = self._pipeline_code.get(source)
        if src_code is None:
            raise ValueError(
                f"源版本 `{source}` 无读取代码(可能是持久化重载而未记录来源),无法生成管道脚本"
            )

        src_info = self._lineage[source]
        lines: list[str] = [
            "import polars as pl",
            "",
            f"# 源数据（version {source}）：{src_info.description}",
            f"lf = ({src_code}).lazy()",
        ]

        for step, vid in enumerate(chain[1:], start=1):
            info = self._lineage[vid]
            frag = self._pipeline_code.get(vid)
            if frag is None:
                raise ValueError(
                    f"版本 `{vid}`（工具 {info.tool}）无可生成的 Polars 代码，"
                    "该数据版本不可复现为脚本"
                )
            lines.append("")
            lines.append(f"# 步骤 {step}（version {vid}，{info.tool}）：{info.description}")
            lines.append(frag)

        lines.append("")
        lines.append("df = lf.collect()")
        if output_path is not None:
            writer = "write_parquet" if output_fmt == "parquet" else "write_csv"
            lines.append(f"df.{writer}({output_path!r})")
        else:
            lines.append("# 取消注释并按需修改输出路径：")
            lines.append('# df.write_csv("output.csv")')
        return "\n".join(lines) + "\n"

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
