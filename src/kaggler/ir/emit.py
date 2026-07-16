"""IR -> 可复现 Polars 代码片段的 code generator(IR 的只读投影)。

约定:每个派生节点 emit 出若干条对变量 ``lf`` 重新赋值的语句,只依赖
``import polars as pl``,拟合常量(payload 里的统计量/权重/映射)原样写死,
从而能脱离 app 独立重放该数据版本 —— 用于备份与 Kaggle 脚本提交。
Code 不参与 restore(restore 走 ir.interpret 的同一路径)。

每种 kind 一个模块级 emitter(``(params) -> str``),集中注册在 ``EMITTERS``,
与 ir.interpret 的 handler 一一成对;两个投影的等价性由三路差分测试保证。

格式化原语(fmt_scalar/fmt_list/col/with_columns_block/over_code)源自
已删除的 feature_engineering/codegen.py,本模块是其唯一后继。
"""

from __future__ import annotations

import math
from collections.abc import Callable

from kaggler.ir.schema import IRNode


# ---- 格式化原语(自 codegen.py 迁入,语义不变) ----

def fmt_scalar(v) -> str:
    """把标量格式化为 Python 字面量源码。

    float 用 ``repr`` 保证最短且可精确 round-trip 的表示;str/bool/int/None 亦然。
    非有限 float(nan/inf)的 ``repr`` 是裸标识符 ``nan``/``inf``,脱离 app 重放时
    ``exec`` 会抛 ``NameError`` —— 冻结出的统计量常量可能是 nan/inf(如含 NaN 的列),
    故改写为可求值的 ``float('nan')`` / ``float('inf')`` / ``float('-inf')``。
    """
    if isinstance(v, float) and not math.isfinite(v):
        if math.isnan(v):
            return "float('nan')"
        return "float('inf')" if v > 0 else "float('-inf')"
    return repr(v)


def fmt_list(xs) -> str:
    """把标量序列格式化为 Python 列表字面量源码,逐元素走 fmt_scalar。

    比 ``repr(list)`` 更稳:能正确写出内含 nan/inf 的列表(用于分组统计量的
    old/new 映射),而不会产出无法 exec 的裸 ``nan``/``inf``。
    """
    return "[" + ", ".join(fmt_scalar(x) for x in xs) + "]"


def col(name: str) -> str:
    """列引用源码:``pl.col('name')``。"""
    return f"pl.col({name!r})"


def with_columns_block(exprs: list[str]) -> str:
    """把一串表达式源码拼成多行 ``lf = lf.with_columns([...])`` 语句。"""
    inner = ",\n".join(f"    {e}" for e in exprs)
    return f"lf = lf.with_columns([\n{inner}\n])"


def over_code(inner: str, group_col: str, breaks: list | None) -> str:
    """把统计量表达式源码 ``inner`` 包成分组窗口 ``inner.over(key)``。

    ``breaks`` 为 None 时按原始取值分组(``pl.col(g)``);否则用等宽分箱内部切点
    ``pl.col(g).cut([...])`` 作分组键。切点原样写死,保证脱离 app 重放确定性。
    """
    key = col(group_col) if breaks is None else f"{col(group_col)}.cut({breaks!r})"
    return f"{inner}.over({key})"


# 一元变换 method -> (列引用源码, spec) -> 表达式源码。与 interpret._MONO_EXPR 一一对应。
_MONO_CODE: dict[str, Callable] = {
    "cos": lambda c, s: f"{c}.cos()",
    "sin": lambda c, s: f"{c}.sin()",
    "tan": lambda c, s: f"{c}.tan()",
    "exp": lambda c, s: f"{c}.exp()",
    "log": (
        lambda c, s: f"{c}.log({fmt_scalar(s['base'])})"
        if s.get("base") is not None
        else f"{c}.log()"
    ),
    "sqrt": lambda c, s: f"{c}.sqrt()",
    "square": lambda c, s: f"{c}.pow(2)",
    "power": lambda c, s: f"{c}.pow({fmt_scalar(s['exponent'])})",
    "linear": lambda c, s: f"{c} * {fmt_scalar(s['a'])} + {fmt_scalar(s['b'])}",
    "reciprocal": lambda c, s: f"1.0 / {c}",
    "abs": lambda c, s: f"{c}.abs()",
}


def mono_expr_code(method: str, column: str, spec: dict) -> str:
    """一元变换单列表达式源码(不含 .alias)。"""
    return _MONO_CODE[method](col(column), spec)


# 组合特征 method -> 各列引用源码列表 -> 组合表达式源码。与 interpret._COMBO_EXPR
# 一一对应(reduce 均为左结合,join 拼出的字符串同样左结合)。
_COMBO_CODE: dict[str, Callable] = {
    "product": lambda parts: " * ".join(parts),
    "sum": lambda parts: " + ".join(parts),
    "mean": lambda parts: f"({' + '.join(parts)}) / {len(parts)}",
    "difference": lambda parts: " - ".join(parts),
    "ratio": lambda parts: " / ".join(parts),
}


def combine_expr_code(method: str, columns: list[str]) -> str:
    """组合特征表达式源码(不含 .alias)。"""
    parts = [col(c) for c in columns]
    return _COMBO_CODE[method](parts)


_OP_SYMBOLS: dict[str, str] = {
    "gt": ">",
    "lt": "<",
    "ge": ">=",
    "le": "<=",
    "eq": "==",
    "ne": "!=",
}

_NULL_OP_METHOD: dict[str, str] = {
    "is_null": "is_null",
    "is_not_null": "is_not_null",
}


# ---- 共享片段(与 interpret 的同名片段成对) ----

def _frozen_group_fill_code(e_code: str, spec: dict) -> str:
    """镜像 interpret._frozen_group_fill:冻结分组映射的 fill_null 源码。"""
    gmap = spec.get("group_map") or []
    if spec.get("group_col") and gmap:
        keys = [k for k, _ in gmap]
        vals = [v for _, v in gmap]
        gk = (
            col(spec["group_col"])
            if spec["group_breaks"] is None
            else f"{col(spec['group_col'])}.cut({spec['group_breaks']!r}).cast(pl.String)"
        )
        e_code += (
            f".fill_null({gk}.replace_strict("
            f"{fmt_list(keys)}, {fmt_list(vals)}, default=None))"
        )
    return e_code


def _stat_fill_code(spec: dict) -> str:
    """镜像 interpret._stat_fill_expr 的源码片段(不含 .alias)。"""
    e = _frozen_group_fill_code(col(spec["column"]), spec)
    if spec["global_stat"] is not None:
        e += f".fill_null({fmt_scalar(spec['global_stat'])})"
    return e


def _mode_fill_code(spec: dict) -> str:
    """镜像 interpret._mode_fill_expr 的源码片段(不含 .alias)。"""
    e = _frozen_group_fill_code(col(spec["column"]), spec)
    return e + f".fill_null({fmt_scalar(spec['value'])})"


def _conditions_code(groups: list[dict], group_logic: str) -> str:
    """镜像 interpret._conditions_expr:组合条件源码,末尾 .fill_null(False)。"""
    def leaf_code(cond: dict) -> str:
        op = cond["op"]
        if op in _NULL_OP_METHOD:
            return f"{col(cond['column'])}.{_NULL_OP_METHOD[op]}()"
        return f"({col(cond['column'])} {_OP_SYMBOLS[op]} {fmt_scalar(cond.get('value'))})"

    group_codes = []
    for g in groups:
        conds = g["conditions"]
        joiner = "&" if g["logic"] == "and" else "|"
        code_c = leaf_code(conds[0])
        for cond in conds[1:]:
            code_c = f"({code_c} {joiner} {leaf_code(cond)})"
        group_codes.append(code_c)

    combined = group_codes[0]
    gj = "&" if group_logic == "and" else "|"
    for code_c in group_codes[1:]:
        combined = f"({combined} {gj} {code_c})"
    return f"({combined}).fill_null(False)"


# ---- per-kind emitters ----

def _emit_fill_missing(params: dict) -> str:
    indicators = params.get("indicators") or []
    fills = params.get("fills") or []
    delete_columns = params.get("delete_columns") or []

    exprs: list[str] = []
    for src, name in indicators:
        exprs.append(f"{col(src)}.is_null().cast(pl.Int8).alias({name!r})")
    for spec in fills:
        t = spec["type"]
        c = spec["column"]
        if t == "zero":
            exprs.append(f"{col(c)}.fill_null({fmt_scalar(spec['value'])})")
        elif t in ("mean", "median"):
            exprs.append(_stat_fill_code(spec))
        elif t == "mode":
            exprs.append(_mode_fill_code(spec))
        else:
            raise ValueError(f"未知的 fill type: {t!r}")

    lines: list[str] = []
    if exprs:
        lines.append(with_columns_block(exprs))
    if delete_columns:
        notnull = ", ".join(f"{col(c)}.is_not_null()" for c in delete_columns)
        lines.append(f"lf = lf.filter(pl.all_horizontal([{notnull}]))")
    return "\n".join(lines) or "# (空值处理:无实际变换)"


def _emit_encode(params: dict) -> str:
    specs = params.get("specs") or []
    drop_columns = params.get("drop_columns") or []

    exprs: list[str] = []
    for spec in specs:
        c = spec["column"]
        if spec["type"] == "one_hot":
            for v in spec["values"]:
                name = f"{c}_{v}"
                if spec["has_nulls"]:
                    exprs.append(
                        f"pl.when({col(c)} == pl.lit({fmt_scalar(v)}))"
                        ".then(pl.lit(True))"
                        f".when({col(c)}.is_not_null()).then(pl.lit(False))"
                        f".otherwise(pl.lit(None)).alias({name!r})"
                    )
                else:
                    exprs.append(
                        f"({col(c)} == pl.lit({fmt_scalar(v)})).alias({name!r})"
                    )
        elif spec["type"] == "label":
            keys = [k for k, _ in spec["mapping"]]
            vals = [v for _, v in spec["mapping"]]
            exprs.append(
                f"pl.when({col(c)}.is_null()).then(None)"
                f".otherwise({col(c)}.replace_strict("
                f"old={fmt_list(keys)}, new={fmt_list(vals)}, default=None))"
                f".cast(pl.Int64).alias({c!r})"
            )
        else:
            raise ValueError(f"未知的 encode type: {spec['type']!r}")

    lines: list[str] = []
    if exprs:
        lines.append(with_columns_block(exprs))
    if drop_columns:
        lines.append(f"lf = lf.drop({drop_columns!r})")
    return "\n".join(lines) or "# (编码:无实际变换)"


def _emit_standardize(params: dict) -> str:
    exprs = [
        f"(({col(s['column'])} - {fmt_scalar(s['mean'])})"
        f" / {fmt_scalar(s['std'])}).alias({s['column']!r})"
        for s in params["stats"]
    ]
    return with_columns_block(exprs)


def _emit_drop_columns(params: dict) -> str:
    return f"lf = lf.drop({params['columns']!r})"


def _emit_filter_rows(params: dict) -> str:
    cc = _conditions_code(params["groups"], params["group_logic"])
    if params["action"] == "keep":
        return f"lf = lf.filter({cc})"
    return f"lf = lf.filter(~{cc})"


def _emit_create_indicator(params: dict) -> str:
    cc = _conditions_code(params["groups"], params["group_logic"])
    return (
        f"lf = lf.with_columns({cc}.cast(pl.Int8)"
        f".alias({params['output_name']!r}))"
    )


def _emit_dim_reduct(params: dict) -> str:
    numeric_cols = params["numeric_cols"]
    out_cols = params["out_cols"]
    exprs: list[str] = []
    for i, comp in enumerate(params["components"]):
        parts = [f"pl.lit({fmt_scalar(comp['bias'])})"]
        for j, c in enumerate(numeric_cols):
            w = comp["weights"][j]
            if w != 0.0:
                parts.append(f"{col(c)} * {fmt_scalar(w)}")
        exprs.append(f"({' + '.join(parts)}).alias({out_cols[i]!r})")
    return (
        with_columns_block(exprs)
        + f"\nlf = lf.select({params['final_cols']!r})"
    )


def _emit_transform_mono(params: dict) -> str:
    exprs = [
        f"({mono_expr_code(s['method'], s['column'], s)}).alias({s['output_name']!r})"
        for s in params["specs"]
    ]
    return with_columns_block(exprs)


def _emit_transform_combination(params: dict) -> str:
    return (
        f"lf = lf.with_columns(({combine_expr_code(params['method'], params['columns'])})"
        f".alias({params['output_name']!r}))"
    )


EMITTERS: dict[str, Callable[[dict], str]] = {
    "fill_missing": _emit_fill_missing,
    "encode": _emit_encode,
    "standardize": _emit_standardize,
    "drop_columns": _emit_drop_columns,
    "filter_rows": _emit_filter_rows,
    "create_indicator": _emit_create_indicator,
    "dim_reduct": _emit_dim_reduct,
    "transform_mono": _emit_transform_mono,
    "transform_combination": _emit_transform_combination,
}


# ---- 入口 ----

def code_from(kind: str, params: dict) -> str:
    """(kind, params) -> 代码片段。与 interpret.op_from 成对。"""
    try:
        emitter = EMITTERS[kind]
    except KeyError:
        raise ValueError(
            f"kind {kind!r} 没有对应的 emitter(source 请用 emit_source_expr)"
        ) from None
    return emitter(params)


def emit_code(node: IRNode) -> str:
    """派生 IRNode -> 操作 ``lf`` 的代码片段。"""
    if node.kind == "source":
        raise ValueError("source 节点没有派生代码,请用 emit_source_expr")
    if len(node.parents) != 1:
        raise NotImplementedError(
            f"多父 IR 节点的代码生成尚未实现(version={node.version}, parents={node.parents})"
        )
    return code_from(node.kind, node.params)


_READER_CODE: dict[str, str] = {
    "csv": "pl.read_csv",
    "parquet": "pl.read_parquet",
}


def emit_source_expr(node: IRNode) -> str:
    """source 节点 -> eager 读取表达式源码(如 ``pl.read_csv('train.csv')``)。"""
    if node.kind != "source":
        raise ValueError(f"仅 source 节点可生成读取表达式,收到 kind={node.kind!r}")
    fmt = node.params["format"]
    try:
        reader = _READER_CODE[fmt]
    except KeyError:
        raise ValueError(f"未知的 source format: {fmt!r}") from None
    return f"{reader}({node.params['path']!r})"
