"""FE 步骤 → 可复现 Polars 代码片段的格式化工具。

约定:每个 ``exec_*`` 产出的 ``code`` 片段是若干条对变量 ``lf`` 重新赋值的语句,
只依赖 ``import polars as pl``,把拟合常量(mean/std、PCA/LDA 权重、编码映射、mode 值)
原样写死,从而能脱离 app 独立重放出该数据版本——用于备份与 Kaggle 脚本提交。

设计要点:片段在「构建 op 闭包处」同步产出(单一真相,杜绝反解闭包带来的漂移),
本模块只提供把 spec/常量转成源码字符串的纯格式化原语。
"""

from __future__ import annotations

from kaggler.modes.feature_engineering.types import (
    CombineMethod,
    MonoTransform,
)


def fmt_scalar(v) -> str:
    """把标量格式化为 Python 字面量源码。

    float 用 ``repr`` 保证最短且可精确 round-trip 的表示;str/bool/int/None 亦然。
    """
    return repr(v)


def col(name: str) -> str:
    """列引用源码:``pl.col('name')``。"""
    return f"pl.col({name!r})"


def with_columns_block(exprs: list[str]) -> str:
    """把一串表达式源码拼成多行 ``lf = lf.with_columns([...])`` 语句。"""
    inner = ",\n".join(f"    {e}" for e in exprs)
    return f"lf = lf.with_columns([\n{inner}\n])"


# 一元变换 method -> (列引用源码, spec) -> 表达式源码。与 compute._MONO_EXPR 一一对应。
_MONO_CODE = {
    MonoTransform.COS: lambda c, s: f"{c}.cos()",
    MonoTransform.SIN: lambda c, s: f"{c}.sin()",
    MonoTransform.TAN: lambda c, s: f"{c}.tan()",
    MonoTransform.EXP: lambda c, s: f"{c}.exp()",
    MonoTransform.LOG: (
        lambda c, s: f"{c}.log({fmt_scalar(s['base'])})"
        if s.get("base") is not None
        else f"{c}.log()"
    ),
    MonoTransform.SQRT: lambda c, s: f"{c}.sqrt()",
    MonoTransform.SQUARE: lambda c, s: f"{c}.pow(2)",
    MonoTransform.POWER: lambda c, s: f"{c}.pow({fmt_scalar(s['exponent'])})",
    MonoTransform.LINEAR: lambda c, s: f"{c} * {fmt_scalar(s['a'])} + {fmt_scalar(s['b'])}",
    MonoTransform.RECIPROCAL: lambda c, s: f"1.0 / {c}",
    MonoTransform.ABS: lambda c, s: f"{c}.abs()",
}


def mono_expr_code(method: MonoTransform, column: str, spec: dict) -> str:
    """一元变换单列表达式源码(不含 .alias)。"""
    return _MONO_CODE[method](col(column), spec)


# 组合特征 method -> 各列引用源码列表 -> 组合表达式源码。与 compute._COMBO_EXPR 一一对应
# (reduce 均为左结合,join 拼出的字符串同样左结合)。
_COMBO_CODE = {
    CombineMethod.PRODUCT: lambda parts: " * ".join(parts),
    CombineMethod.SUM: lambda parts: " + ".join(parts),
    CombineMethod.MEAN: lambda parts: f"({' + '.join(parts)}) / {len(parts)}",
    CombineMethod.DIFFERENCE: lambda parts: " - ".join(parts),
    CombineMethod.RATIO: lambda parts: " / ".join(parts),
}


def combine_expr_code(method: CombineMethod, columns: list[str]) -> str:
    """组合特征表达式源码(不含 .alias)。"""
    parts = [col(c) for c in columns]
    return _COMBO_CODE[method](parts)
