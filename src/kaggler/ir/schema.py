"""数据版本 IR 的节点结构与序列化(SSOT 的载体)。

IR 是数据版本表示的唯一真相:运行时 Op 闭包(ir.interpret)与面向用户的
Polars 代码(ir.emit)都是它的派生投影。每个 IR 节点对应一个数据版本
(= 一次工具调用),记录 op 类型(kind)、拟合参数 payload(params)、
父版本引用列表(parents)与随机种子(seed,当前 dormant)。

序列化约定:
- 单一出入口 ``dumps_ir`` / ``loads_ir``,账本只存不解读的 TEXT。
- ``json.dumps(..., allow_nan=True)``:冻结统计量可能是 nan/inf,Python 的
  json 以非标准记号 ``NaN``/``Infinity`` 写出并能原样读回(round-trip 精确)。
  该 TEXT 只被本项目 Python 读写;若未来有外部消费者,升 ``schema_version``
  换编码即可。
- params 仅允许 JSON 原生类型(dict[str]/list/str/int/float/bool/None),
  按**精确类型**校验 —— numpy 标量(np.float64 等)一律在 dumps 时响亮报错,
  归一(``float()``)是拟合侧的责任。
- 一切「数据值 -> 值」映射(label mapping、分组统计量)必须用平行对列表
  ``[[key, value], ...]`` 而非 dict:JSON 对象键必为 str,int/float/bool 键
  经 dict 往返会漂移成字符串。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

IR_SCHEMA_VERSION = 1

# op 类型全集:source + 与 feature_engineering 各 exec_* 一一对应的 9 种派生 kind。
KINDS = frozenset({
    "source",
    "fill_missing",
    "encode",
    "standardize",
    "drop_columns",
    "filter_rows",
    "create_indicator",
    "dim_reduct",
    "transform_mono",
    "transform_combination",
})


@dataclass(frozen=True)
class IRNode:
    """一个数据版本的完整 IR 表示(可落库)。

    parents:source 为 [];派生版本为 [parent]。schema 层允许多父
    (为 join/concat 的 fan-in 预留),但执行层(interpret/emit)遇
    len != 1 抛 NotImplementedError —— 现有变换无多父,不投机实现。
    """

    version: int
    kind: str
    parents: list[int]
    params: dict
    seed: int | None = None
    schema_version: int = IR_SCHEMA_VERSION


@dataclass(frozen=True)
class IRSpec:
    """exec_* 的产出物:不含 version/parents 的「半成品」。

    exec_* 不知道版本号 —— 在 ``DataProvider.add_version`` 处与
    version / parents=[parent] 组装成完整 IRNode。
    """

    kind: str
    params: dict = field(default_factory=dict)
    seed: int | None = None


_SCALAR_TYPES = (str, int, float, bool, type(None))


def _validate_json_native(v, path: str) -> None:
    """递归校验 payload 仅含 JSON 原生类型(精确类型比对)。

    ``type(v) in ...`` 而非 ``isinstance``:np.float64 是 float 子类、
    IntEnum 是 int 子类,isinstance 会放行这些经 json 序列化后表示不可控
    (或语义已丢失)的类型 —— 一律拒绝,强制拟合侧先归一。
    """
    if type(v) in _SCALAR_TYPES:
        return
    if type(v) is dict:
        for k, item in v.items():
            if type(k) is not str:
                raise ValueError(
                    f"IR payload 的 dict 键必须是 str,{path} 处键 {k!r} 为 {type(k).__name__};"
                    "数据值键请改用平行对列表 [[key, value], ...]"
                )
            _validate_json_native(item, f"{path}.{k}")
        return
    if type(v) in (list, tuple):
        for i, item in enumerate(v):
            _validate_json_native(item, f"{path}[{i}]")
        return
    raise ValueError(
        f"IR payload 含非 JSON 原生类型:{path} 处为 {type(v).__name__}({v!r});"
        "numpy 标量等须在拟合期归一为 Python 原生类型"
    )


def _validate_node(node: IRNode) -> None:
    if node.kind not in KINDS:
        raise ValueError(f"未知的 IR kind: {node.kind!r}(合法值:{sorted(KINDS)})")
    if type(node.version) is not int:
        raise ValueError(f"IRNode.version 必须是 int,收到 {type(node.version).__name__}")
    if type(node.parents) is not list or any(type(p) is not int for p in node.parents):
        raise ValueError(f"IRNode.parents 必须是 list[int],收到 {node.parents!r}")
    if node.seed is not None and type(node.seed) is not int:
        raise ValueError(f"IRNode.seed 必须是 int 或 None,收到 {type(node.seed).__name__}")
    if type(node.params) is not dict:
        raise ValueError(f"IRNode.params 必须是 dict,收到 {type(node.params).__name__}")
    _validate_json_native(node.params, "params")


def dumps_ir(node: IRNode) -> str:
    """IRNode -> JSON 文本(校验 + 序列化的唯一出口)。"""
    _validate_node(node)
    return json.dumps(
        {
            "schema_version": node.schema_version,
            "version": node.version,
            "kind": node.kind,
            "parents": node.parents,
            "params": node.params,
            "seed": node.seed,
        },
        ensure_ascii=False,
        allow_nan=True,
    )


def loads_ir(s: str) -> IRNode:
    """JSON 文本 -> IRNode(反序列化的唯一入口,未知版本/kind 响亮报错)。"""
    raw = json.loads(s)
    sv = raw.get("schema_version")
    if type(sv) is not int or sv > IR_SCHEMA_VERSION:
        raise ValueError(
            f"IR schema_version {sv!r} 高于当前支持的 {IR_SCHEMA_VERSION},"
            "请升级应用后再恢复该会话"
        )
    node = IRNode(
        version=raw["version"],
        kind=raw["kind"],
        parents=list(raw["parents"]),
        params=raw["params"],
        seed=raw.get("seed"),
        schema_version=sv,
    )
    _validate_node(node)
    return node
