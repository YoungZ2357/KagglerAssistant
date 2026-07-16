"""IR -> Op 闭包的 interpreter(运行时与 restore 共用的唯一构建路径)。

每种派生 kind 一个模块级 handler(``(params) -> Op``),集中注册在
``INTERPRETERS``。语义与 feature_engineering/compute.py 中各 ``exec_*``
的 ``_op`` 分支严格对齐;拟合常量一律从 params 透传(payload 即真相),
任何 handler 都不得重算统计量 —— 这是反数据泄漏不变量的落点。

本模块 string-keyed、只依赖 polars:method/op 等字段直接用字符串查表
(FE 的 str Enum 值天然兼容),未知取值经 dict KeyError 响亮暴露。
"""

from __future__ import annotations

from collections.abc import Callable
from functools import reduce

import polars as pl

from kaggler.ir.schema import IRNode

# 与 persistence.data_provider 的 Op/Loader 结构相同;此处独立定义,
# 保持 ir 包只依赖 stdlib + polars(依赖方向:persistence -> ir)。
Op = Callable[[pl.LazyFrame], pl.LazyFrame]
Loader = Callable[[], pl.DataFrame]

# ---- string-keyed 语义表(与 compute.py 的 enum-keyed 表逐项对应) ----

_MONO_EXPR: dict[str, Callable] = {
    "cos": lambda c, s: c.cos(),
    "sin": lambda c, s: c.sin(),
    "tan": lambda c, s: c.tan(),
    "exp": lambda c, s: c.exp(),
    "log": (
        lambda c, s: c.log(s["base"]) if s.get("base") is not None else c.log()
    ),
    "sqrt": lambda c, s: c.sqrt(),
    "square": lambda c, s: c.pow(2),
    "power": lambda c, s: c.pow(s["exponent"]),
    "linear": lambda c, s: c * s["a"] + s["b"],
    "reciprocal": lambda c, s: 1.0 / c,
    "abs": lambda c, s: c.abs(),
}

_COMBO_EXPR: dict[str, Callable] = {
    "product": lambda exprs: reduce(lambda a, b: a * b, exprs),
    "sum": lambda exprs: reduce(lambda a, b: a + b, exprs),
    "mean": lambda exprs: reduce(lambda a, b: a + b, exprs) / len(exprs),
    "difference": lambda exprs: reduce(lambda a, b: a - b, exprs),
    "ratio": lambda exprs: reduce(lambda a, b: a / b, exprs),
}

_OP_FUNCS: dict[str, Callable] = {
    "gt": lambda e, v: e > v,
    "lt": lambda e, v: e < v,
    "ge": lambda e, v: e >= v,
    "le": lambda e, v: e <= v,
    "eq": lambda e, v: e == v,
    "ne": lambda e, v: e != v,
    "is_null": lambda e, v: e.is_null(),
    "is_not_null": lambda e, v: e.is_not_null(),
}


# ---- 共享片段 ----

def _frozen_group_fill(e: pl.Expr, spec: dict) -> pl.Expr:
    """把冻结的「组键 -> 统计量」映射接为 fill_null 表达式。

    组键与拟合期(_freeze_group_stats/_freeze_group_modes)严格一致:
    未分箱按原始取值,分箱按 ``cut(breaks).cast(pl.String)`` 的字符串标签。
    映射未命中(新组键/被 drop 的空组)得 None,由外层全局兜底接住。
    """
    gmap = spec.get("group_map") or []
    if spec.get("group_col") and gmap:
        keys = [k for k, _ in gmap]
        vals = [v for _, v in gmap]
        gk = (
            pl.col(spec["group_col"])
            if spec["group_breaks"] is None
            else pl.col(spec["group_col"]).cut(spec["group_breaks"]).cast(pl.String)
        )
        e = e.fill_null(gk.replace_strict(keys, vals, default=None))
    return e


def _stat_fill_expr(spec: dict) -> pl.Expr:
    """mean/median 填充:分组映射与全局统计量均为冻结常量。"""
    e = _frozen_group_fill(pl.col(spec["column"]), spec)
    if spec["global_stat"] is not None:
        e = e.fill_null(spec["global_stat"])
    return e


def _mode_fill_expr(spec: dict) -> pl.Expr:
    """mode 填充:冻结的分组众数映射 + 冻结的全局众数兜底。"""
    e = _frozen_group_fill(pl.col(spec["column"]), spec)
    return e.fill_null(spec["value"])


def _conditions_expr(groups: list[dict], group_logic: str) -> pl.Expr:
    """两层条件结构 -> 组合布尔表达式,末尾 fill_null(False)。"""
    def leaf(cond: dict) -> pl.Expr:
        return _OP_FUNCS[cond["op"]](pl.col(cond["column"]), cond.get("value"))

    group_exprs = []
    for g in groups:
        conds = g["conditions"]
        expr = leaf(conds[0])
        for cond in conds[1:]:
            expr = (expr & leaf(cond)) if g["logic"] == "and" else (expr | leaf(cond))
        group_exprs.append(expr)

    combined = group_exprs[0]
    for expr in group_exprs[1:]:
        combined = (combined & expr) if group_logic == "and" else (combined | expr)
    return combined.fill_null(False)


# ---- per-kind handlers ----

def _interp_fill_missing(params: dict) -> Op:
    indicators = params.get("indicators") or []
    fills = params.get("fills") or []
    delete_columns = params.get("delete_columns") or []

    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        exprs = []
        for src, name in indicators:
            exprs.append(pl.col(src).is_null().cast(pl.Int8).alias(name))
        for spec in fills:
            t = spec["type"]
            if t == "zero":
                exprs.append(pl.col(spec["column"]).fill_null(spec["value"]))
            elif t in ("mean", "median"):
                exprs.append(_stat_fill_expr(spec))
            elif t == "mode":
                exprs.append(_mode_fill_expr(spec))
            else:
                raise ValueError(f"未知的 fill type: {t!r}")
        if exprs:
            lf = lf.with_columns(exprs)
        if delete_columns:
            lf = lf.filter(
                pl.all_horizontal([pl.col(c).is_not_null() for c in delete_columns])
            )
        return lf

    return _op


def _interp_encode(params: dict) -> Op:
    specs = params.get("specs") or []
    drop_columns = params.get("drop_columns") or []

    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        exprs = []
        for spec in specs:
            c = spec["column"]
            if spec["type"] == "one_hot":
                for v in spec["values"]:
                    name = f"{c}_{v}"
                    if spec["has_nulls"]:
                        exprs.append(
                            pl.when(pl.col(c) == pl.lit(v))
                            .then(pl.lit(True))
                            .when(pl.col(c).is_not_null())
                            .then(pl.lit(False))
                            .otherwise(pl.lit(None))
                            .alias(name)
                        )
                    else:
                        exprs.append((pl.col(c) == pl.lit(v)).alias(name))
            elif spec["type"] == "label":
                keys = [k for k, _ in spec["mapping"]]
                vals = [v for _, v in spec["mapping"]]
                exprs.append(
                    pl.when(pl.col(c).is_null())
                    .then(None)
                    .otherwise(
                        pl.col(c).replace_strict(old=keys, new=vals, default=None)
                    )
                    .cast(pl.Int64)
                    .alias(c)
                )
            else:
                raise ValueError(f"未知的 encode type: {spec['type']!r}")
        if exprs:
            lf = lf.with_columns(exprs)
        if drop_columns:
            lf = lf.drop(drop_columns)
        return lf

    return _op


def _interp_standardize(params: dict) -> Op:
    stats = params["stats"]

    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        exprs = [
            ((pl.col(s["column"]) - s["mean"]) / s["std"]).alias(s["column"])
            for s in stats
        ]
        return lf.with_columns(exprs)

    return _op


def _interp_drop_columns(params: dict) -> Op:
    columns = params["columns"]

    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.drop(columns)

    return _op


def _interp_filter_rows(params: dict) -> Op:
    expr = _conditions_expr(params["groups"], params["group_logic"])
    keep = params["action"] == "keep"

    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.filter(expr) if keep else lf.filter(~expr)

    return _op


def _interp_create_indicator(params: dict) -> Op:
    expr = (
        _conditions_expr(params["groups"], params["group_logic"])
        .cast(pl.Int8)
        .alias(params["output_name"])
    )

    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns(expr)

    return _op


def _interp_dim_reduct(params: dict) -> Op:
    components = params["components"]
    numeric_cols = params["numeric_cols"]
    out_cols = params["out_cols"]
    final_cols = params["final_cols"]

    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        exprs = []
        for i, comp in enumerate(components):
            expr = pl.lit(comp["bias"])
            for j, c in enumerate(numeric_cols):
                w = comp["weights"][j]
                if w != 0.0:
                    expr = expr + pl.col(c) * w
            exprs.append(expr.alias(out_cols[i]))
        return lf.with_columns(exprs).select(final_cols)

    return _op


def _interp_transform_mono(params: dict) -> Op:
    specs = params["specs"]

    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        exprs = [
            _MONO_EXPR[s["method"]](pl.col(s["column"]), s).alias(s["output_name"])
            for s in specs
        ]
        return lf.with_columns(exprs)

    return _op


def _interp_transform_combination(params: dict) -> Op:
    columns = params["columns"]
    method = params["method"]
    output_name = params["output_name"]

    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        exprs = [pl.col(c) for c in columns]
        return lf.with_columns(_COMBO_EXPR[method](exprs).alias(output_name))

    return _op


INTERPRETERS: dict[str, Callable[[dict], Op]] = {
    "fill_missing": _interp_fill_missing,
    "encode": _interp_encode,
    "standardize": _interp_standardize,
    "drop_columns": _interp_drop_columns,
    "filter_rows": _interp_filter_rows,
    "create_indicator": _interp_create_indicator,
    "dim_reduct": _interp_dim_reduct,
    "transform_mono": _interp_transform_mono,
    "transform_combination": _interp_transform_combination,
}


# ---- 入口 ----

def op_from(kind: str, params: dict) -> Op:
    """(kind, params) -> Op。供运行时(compute 层)与 build_op 共用。"""
    try:
        handler = INTERPRETERS[kind]
    except KeyError:
        raise ValueError(
            f"kind {kind!r} 没有对应的 interpreter(source 请用 build_loader)"
        ) from None
    return handler(params)


def build_op(node: IRNode) -> Op:
    """IRNode -> Op(restore 与运行时共用的唯一构建路径)。"""
    if node.kind == "source":
        raise ValueError("source 节点没有派生 op,请用 build_loader")
    if len(node.parents) != 1:
        raise NotImplementedError(
            f"多父 IR 节点的执行尚未实现(version={node.version}, parents={node.parents})"
        )
    return op_from(node.kind, node.params)


_READERS: dict[str, Callable[[str], pl.DataFrame]] = {
    "csv": pl.read_csv,
    "parquet": pl.read_parquet,
}


def build_loader(node: IRNode) -> Loader:
    """source 节点 -> 无参 loader。"""
    if node.kind != "source":
        raise ValueError(f"仅 source 节点可构建 loader,收到 kind={node.kind!r}")
    fmt = node.params["format"]
    path = node.params["path"]
    try:
        reader = _READERS[fmt]
    except KeyError:
        raise ValueError(f"未知的 source format: {fmt!r}") from None
    return lambda: reader(path)
