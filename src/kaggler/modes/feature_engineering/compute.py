from functools import reduce

import numpy as np
import polars as pl
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from kaggler.shared.limits import (
    MAX_COLUMN_LIST,
    MAX_MAPPING_ENTRIES,
    MAX_ONEHOT_COLUMNS,
    MAX_PREVIEW_COLUMNS,
    cap_list,
)
from kaggler.shared.serialization import safe_val
from kaggler.modes.feature_engineering.types import (
    CombineMethod,
    ComparisonOp,
    DimReductMethod,
    EncodeMethod,
    FillMethod,
    MonoTransform,
    RowAction,
    RowLogic,
)


def _build_preview(result_df: pl.DataFrame) -> tuple[list[dict], dict | None]:
    """构造回传给模型的预览：前 3 行、逐格 safe_val，并对过宽的行截列。

    Returns:
        (preview 行列表, 截断提示 note)。列数未超限时 note 为 None；
        超限时 note 形如 {"note": "预览仅显示前 k/w 列，完整结构见 explore_schema"}，
        由调用方 append 进 summary，避免静默截断。
    """
    keep_cols, info = cap_list(result_df.columns, MAX_PREVIEW_COLUMNS)
    preview_rows = result_df.select(keep_cols).head(3).to_dicts()
    preview = [{k: safe_val(v) for k, v in row.items()} for row in preview_rows]
    note = None
    if info:
        note = {
            "note": (
                f"预览仅显示前 {info['shown']}/{info['total']} 列，"
                "完整结构请使用 explore_schema"
            )
        }
    return preview, note


def exec_empty(
    df: pl.DataFrame,
    pairs: list[dict],
) -> dict:
    """
    [HUMAN]空值处理运算逻辑
    Args:
        df: 数据集
        pairs: 列-方法对，指定每一列的处理方法

    Returns:

    """
    schema = df.schema

    columns_set = set(schema.names())
    unknown = []
    dtype_errors = []
    for pair in pairs:
        col = pair["column"]
        action_raw = pair["action"]
        if col not in columns_set:
            unknown.append(col)
            continue
        try:
            action = FillMethod(action_raw)
        except ValueError:
            dtype_errors.append({"column": col, "error": f"未知的填充方法: {action_raw}"})
            continue
        dtype = schema[col]
        if action in (FillMethod.AVG, FillMethod.MEDIAN) and not dtype.is_numeric():
            dtype_errors.append({
                "column": col,
                "error": f"填充方法 '{action.value}' 仅适用于数值列，当前类型: {dtype}",
            })

    if unknown:
        return {
            "error": f"以下列名不存在：{unknown}",
            "hint": "请先调用 explore_schema 确认列名。",
        }
    if dtype_errors:
        return {
            "error": "部分列与填充方法不兼容",
            "details": dtype_errors,
        }

    nulls_before: dict[str, int] = {}
    for pair in pairs:
        col = pair["column"]
        if col not in nulls_before:
            nulls_before[col] = df[col].null_count()

    fill_specs: list[dict] = []
    delete_columns: list[str] = []
    skip_summary: list[dict] = []

    for pair in pairs:
        col = pair["column"]
        action = FillMethod(pair["action"])
        dtype = schema[col]

        if action == FillMethod.DELETE:
            delete_columns.append(col)
            continue

        if action == FillMethod.ZERO:
            if dtype.is_numeric():
                fill_specs.append({"column": col, "type": "zero", "value": 0})
            elif dtype == pl.String:
                fill_specs.append({"column": col, "type": "zero", "value": "0"})
            elif dtype == pl.Boolean:
                fill_specs.append({"column": col, "type": "zero", "value": False})
            else:
                skip_summary.append({
                    "column": col,
                    "method": action.value,
                    "nulls_before": nulls_before[col],
                    "nulls_filled": 0,
                    "warnings": [f"列类型 {dtype} 不支持零值填充，已跳过"],
                })
            continue

        if action == FillMethod.AVG:
            fill_specs.append({"column": col, "type": "mean"})
            continue

        if action == FillMethod.MEDIAN:
            fill_specs.append({"column": col, "type": "median"})
            continue

        if action == FillMethod.MODE:
            mode_val = df[col].drop_nulls().mode()
            if mode_val is not None and mode_val.len() > 0:
                fill_specs.append({"column": col, "type": "mode", "value": mode_val[0]})
            else:
                skip_summary.append({
                    "column": col,
                    "method": action.value,
                    "nulls_before": nulls_before[col],
                    "nulls_filled": 0,
                    "warnings": ["列全部为空值，无法确定众数，已跳过"],
                })
            continue

    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        exprs = []
        for spec in fill_specs:
            col = spec["column"]
            if spec["type"] == "zero":
                exprs.append(pl.col(col).fill_null(spec["value"]))
            elif spec["type"] == "mean":
                exprs.append(pl.col(col).fill_null(pl.col(col).mean()))
            elif spec["type"] == "median":
                exprs.append(pl.col(col).fill_null(pl.col(col).median()))
            elif spec["type"] == "mode":
                exprs.append(pl.col(col).fill_null(spec["value"]))
        if exprs:
            lf = lf.with_columns(exprs)
        if delete_columns:
            lf = lf.filter(
                pl.all_horizontal([pl.col(c).is_not_null() for c in delete_columns])
            )
        return lf

    result_df = _op(df.lazy()).collect()

    summary: list[dict] = list(skip_summary)
    _method_names = {"zero": "zero", "mean": "avg", "median": "median", "mode": "mode"}
    for spec in fill_specs:
        col = spec["column"]
        if any(s.get("column") == col for s in summary):
            continue
        nulls_after = result_df[col].null_count()
        summary.append({
            "column": col,
            "method": _method_names[spec["type"]],
            "nulls_before": nulls_before[col],
            "nulls_filled": nulls_before[col] - nulls_after,
            "warnings": [],
        })

    if delete_columns:
        rows_before = df.height
        rows_after = result_df.height
        for col in delete_columns:
            summary.append({
                "column": col,
                "method": "delete",
                "nulls_before": nulls_before[col],
                "rows_deleted": rows_before - rows_after,
                "warnings": [],
            })

    preview, _preview_note = _build_preview(result_df)

    if _preview_note:
        summary.append(_preview_note)
    return {
        "op": _op,
        "preview": preview,
        "summary": summary,
        "rows_before": df.height,
        "rows_after": result_df.height,
    }


ONE_HOT_CARDINALITY_WARN = 10


def exec_encode(
    df: pl.DataFrame,
    pairs: list[dict],
) -> dict:
    """
    [HUMAN]对指定列执行编码操作。

    独热编码逻辑：对目标列的每个唯一值生成一列布尔指示变量，强制丢弃第一类
    （drop_first=True），因此 n 个唯一值产生 n-1 列。如果该列唯一值数量超过
    ONE_HOT_CARDINALITY_WARN，会附带 warning 但不阻止执行。

    标签编码逻辑：将列的唯一值按排序后顺序映射为整数 0, 1, 2, ...。
    排序规则：字符串按字典序，数值按自然序，布尔值 False < True。
    空值在编码后保持为空，不参与映射。
    这种排序保证同一个数据集在不同时间编码结果一致。

    Args:
        df: 数据集
        pairs: 列-方法对，每项 {"column": <列名>, "action": "one_hot"|"label"}

    Returns:
        dict: 包含 op, preview, summary, rows_before, rows_after
    """
    schema = df.schema
    columns_set = set(schema.names())

    unknown = []
    action_errors = []
    for pair in pairs:
        col = pair["column"]
        action_raw = pair["action"]
        if col not in columns_set:
            unknown.append(col)
            continue
        try:
            EncodeMethod(action_raw)
        except ValueError:
            action_errors.append({"column": col, "error": f"未知的编码方法: {action_raw}"})

    if unknown:
        return {
            "error": f"以下列名不存在：{unknown}",
            "hint": "请先调用 explore_schema 确认列名。",
        }
    if action_errors:
        return {
            "error": "部分编码方法无效",
            "details": action_errors,
        }

    columns_to_drop = []
    summary = []
    encode_specs: list[dict] = []

    for pair in pairs:
        col = pair["column"]
        action = EncodeMethod(pair["action"])

        if action == EncodeMethod.ONE_HOT:
            null_count = df[col].null_count()
            non_null_vals = df[col].drop_nulls().unique()
            unique_count = len(non_null_vals)

            warnings = []
            if unique_count == 0:
                warnings.append("列全部为空值，无法生成独热编码列")
                summary.append({
                    "column": col,
                    "method": action.value,
                    "new_columns": [],
                    "warnings": warnings,
                })
                columns_to_drop.append(col)
                continue

            if unique_count > ONE_HOT_CARDINALITY_WARN:
                warnings.append(f"唯一值数量为 {unique_count}，生成 {unique_count - 1} 个新列，可能使数据集过于稀疏")

            if unique_count <= 1:
                warnings.append("列仅有一个唯一值，drop_first 后无新列生成")

            # 用 to_dummies 提前确定 drop_first 后的列名（兼容旧行为）
            dummies = df.select(pl.col(col)).to_dummies(col, drop_first=True)
            dummy_cols = [c for c in dummies.columns if not c.endswith("_null")]
            kept_values = [dc[len(col) + 1:] for dc in dummy_cols]

            encode_specs.append({
                "column": col,
                "type": "one_hot",
                "values": kept_values,
                "has_nulls": null_count > 0,
            })
            columns_to_drop.append(col)

            # 高基数列生成的新列可能极多，回传摘要时截断列名清单（不影响实际编码）。
            shown_cols, cols_info = cap_list(dummy_cols, MAX_ONEHOT_COLUMNS)
            one_hot_summary = {
                "column": col,
                "method": action.value,
                "new_columns": shown_cols,
                "n_new_columns": len(dummy_cols),
                "unique_values": unique_count,
                "warnings": warnings,
            }
            if cols_info:
                one_hot_summary["new_columns_truncated"] = cols_info
            summary.append(one_hot_summary)

        elif action == EncodeMethod.LABEL:
            vals = df[col].drop_nulls().unique().to_list()
            unique_sorted = sorted(vals)
            if not unique_sorted:
                summary.append({
                    "column": col,
                    "method": action.value,
                    "mapping": {},
                    "warnings": ["列全部为空值，无法进行标签编码"],
                })
                continue

            mapping = {v: i for i, v in enumerate(unique_sorted)}

            encode_specs.append({
                "column": col,
                "type": "label",
                "mapping": mapping,
            })

            # 高基数列的完整映射可能有上万条；回传摘要时截断（完整映射已进 encode_specs，
            # 实际编码结果不受影响）。
            mapping_items = list(mapping.items())
            shown_items, map_info = cap_list(mapping_items, MAX_MAPPING_ENTRIES)
            label_summary = {
                "column": col,
                "method": action.value,
                "n_categories": len(mapping),
                "mapping": {str(k): v for k, v in shown_items},
                "warnings": [],
            }
            if map_info:
                label_summary["mapping_truncated"] = map_info
            summary.append(label_summary)

    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        exprs = []
        for spec in encode_specs:
            col = spec["column"]
            if spec["type"] == "one_hot":
                for v in spec["values"]:
                    name = f"{col}_{v}"
                    if spec["has_nulls"]:
                        exprs.append(
                            pl.when(pl.col(col) == pl.lit(v))
                            .then(pl.lit(True))
                            .when(pl.col(col).is_not_null())
                            .then(pl.lit(False))
                            .otherwise(pl.lit(None))
                            .alias(name)
                        )
                    else:
                        exprs.append(
                            (pl.col(col) == pl.lit(v)).alias(name)
                        )
            elif spec["type"] == "label":
                mapping = spec["mapping"]
                exprs.append(
                    pl.when(pl.col(col).is_null())
                    .then(None)
                    .otherwise(
                        pl.col(col).replace_strict(
                            old=list(mapping.keys()),
                            new=list(mapping.values()),
                            default=None,
                        )
                    )
                    .cast(pl.Int64)
                    .alias(col)
                )
        if exprs:
            lf = lf.with_columns(exprs)
        if columns_to_drop:
            lf = lf.drop(columns_to_drop)
        return lf

    result_df = _op(df.lazy()).collect()

    preview, _preview_note = _build_preview(result_df)

    if _preview_note:
        summary.append(_preview_note)
    return {
        "op": _op,
        "preview": preview,
        "summary": summary,
        "rows_before": df.height,
        "rows_after": result_df.height,
    }


def exec_standardize(
    df: pl.DataFrame,
    columns: list[str],
) -> dict:
    schema = df.schema
    columns_set = set(schema.names())

    unknown = [c for c in columns if c not in columns_set]
    if unknown:
        return {
            "error": f"以下列名不存在：{unknown}",
            "hint": "请先调用 explore_schema 确认列名。",
        }

    non_numeric = [c for c in columns if not schema[c].is_numeric()]
    if non_numeric:
        return {
            "error": f"以下列不是数值类型，无法标准化：{non_numeric}",
        }

    null_columns = [c for c in columns if df[c].null_count() > 0]
    if null_columns:
        return {
            "error": f"以下列存在空值，请先使用 execute_empty_value 处理：{null_columns}",
        }

    # 拟合阶段：计算每列 mean / std
    stats: dict[str, tuple[float, float]] = {}
    for c in columns:
        col_mean = df[c].mean()
        col_std = df[c].std()
        stats[c] = (col_mean, col_std)

    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        exprs = []
        for c in columns:
            mean, std = stats[c]
            exprs.append(((pl.col(c) - mean) / std).alias(c))
        return lf.with_columns(exprs)

    result_df = _op(df.lazy()).collect()

    preview, _preview_note = _build_preview(result_df)

    summary = [{
        "columns": columns,
        "description": "z-score标准化（均值=0，标准差=1）",
    }]

    if _preview_note:
        summary.append(_preview_note)
    return {
        "op": _op,
        "preview": preview,
        "summary": summary,
        "rows_before": df.height,
        "rows_after": result_df.height,
    }


def exec_drop_columns(
    df: pl.DataFrame,
    columns: list[str],
) -> dict:
    if not columns:
        return {"error": "columns 不能为空"}

    columns = list(dict.fromkeys(columns))

    schema = df.schema
    columns_set = set(schema.names())
    unknown = [c for c in columns if c not in columns_set]
    if unknown:
        return {
            "error": f"以下列名不存在：{unknown}",
            "hint": "请先调用 explore_schema 确认列名。",
        }

    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.drop(columns)

    result_df = _op(df.lazy()).collect()

    warnings = []
    if result_df.width == 0:
        warnings.append("已删除全部列，数据集为空")

    preview, _preview_note = _build_preview(result_df)

    remaining_shown, remaining_info = cap_list(result_df.columns, MAX_COLUMN_LIST)
    drop_summary = {
        "dropped_columns": columns,
        "remaining_columns": remaining_shown,
        "n_remaining_columns": result_df.width,
        "warnings": warnings,
    }
    if remaining_info:
        drop_summary["remaining_columns_truncated"] = remaining_info
    summary = [drop_summary]

    if _preview_note:
        summary.append(_preview_note)
    return {
        "op": _op,
        "preview": preview,
        "summary": summary,
        "rows_before": df.height,
        "rows_after": result_df.height,
    }


_OP_FUNCS = {
    ComparisonOp.GT: lambda e, v: e > v,
    ComparisonOp.LT: lambda e, v: e < v,
    ComparisonOp.GE: lambda e, v: e >= v,
    ComparisonOp.LE: lambda e, v: e <= v,
    ComparisonOp.EQ: lambda e, v: e == v,
    ComparisonOp.NE: lambda e, v: e != v,
}

_OP_SYMBOLS = {
    ComparisonOp.GT: ">",
    ComparisonOp.LT: "<",
    ComparisonOp.GE: ">=",
    ComparisonOp.LE: "<=",
    ComparisonOp.EQ: "==",
    ComparisonOp.NE: "!=",
}

_LOGIC_SYMBOLS = {
    RowLogic.AND: "且",
    RowLogic.OR: "或",
}


def _format_condition_value(value) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def exec_filter_rows(
    df: pl.DataFrame,
    groups: list[dict],
    group_logic: str,
    action: str,
) -> dict:
    try:
        action_enum = RowAction(action)
    except ValueError:
        return {"error": f"未知的 action: {action}", "hint": "支持的值: keep, delete"}

    try:
        group_logic_enum = RowLogic(group_logic)
    except ValueError:
        return {"error": f"未知的 group_logic: {group_logic}", "hint": "支持的值: and, or"}

    if not groups:
        return {"error": "groups 不能为空"}

    empty_group_indices = [i for i, g in enumerate(groups) if not g.get("conditions")]
    if empty_group_indices:
        return {"error": f"以下分组下标的 conditions 不能为空：{empty_group_indices}"}

    schema = df.schema
    columns_set = set(schema.names())
    all_conditions = [cond for g in groups for cond in g["conditions"]]

    unknown = sorted({
        cond["column"] for cond in all_conditions if cond["column"] not in columns_set
    })
    if unknown:
        return {
            "error": f"以下列名不存在：{unknown}",
            "hint": "请先调用 explore_schema 确认列名。",
        }

    errors = []
    for i, g in enumerate(groups):
        try:
            RowLogic(g["logic"])
        except ValueError:
            errors.append({"group": i, "error": f"未知的 logic: {g['logic']}"})

    for cond in all_conditions:
        col = cond["column"]
        value = cond["value"]
        dtype = schema[col]

        try:
            ComparisonOp(cond["op"])
        except ValueError:
            errors.append({"column": col, "error": f"未知的比较运算符: {cond['op']}"})
            continue

        if dtype.is_numeric():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append({
                    "column": col,
                    "error": f"列 '{col}' 是数值类型（{dtype}），比较值必须是数字，当前值: {value!r}",
                })
        elif dtype == pl.String:
            if not isinstance(value, str):
                errors.append({
                    "column": col,
                    "error": f"列 '{col}' 是字符串类型，比较值必须是字符串，当前值: {value!r}",
                })
        elif dtype == pl.Boolean:
            if not isinstance(value, bool):
                errors.append({
                    "column": col,
                    "error": f"列 '{col}' 是布尔类型，比较值必须是布尔值，当前值: {value!r}",
                })
        else:
            errors.append({
                "column": col,
                "error": f"列 '{col}' 的类型 {dtype} 暂不支持行筛选",
            })

    if errors:
        return {"error": "部分条件不合法", "details": errors}

    def leaf_expr(cond: dict) -> pl.Expr:
        op = ComparisonOp(cond["op"])
        return _OP_FUNCS[op](pl.col(cond["column"]), cond["value"])

    def leaf_desc(cond: dict) -> str:
        op = ComparisonOp(cond["op"])
        return f"{cond['column']} {_OP_SYMBOLS[op]} {_format_condition_value(cond['value'])}"

    group_exprs = []
    group_descs = []
    for g in groups:
        logic = RowLogic(g["logic"])
        conds = g["conditions"]

        expr = leaf_expr(conds[0])
        desc = leaf_desc(conds[0])
        for cond in conds[1:]:
            expr = (expr & leaf_expr(cond)) if logic == RowLogic.AND else (expr | leaf_expr(cond))
            desc = f"{desc} {_LOGIC_SYMBOLS[logic]} {leaf_desc(cond)}"

        group_exprs.append(expr)
        group_descs.append(f"({desc})" if len(conds) > 1 else desc)

    combined_expr = group_exprs[0]
    combined_desc = group_descs[0]
    for expr, desc in zip(group_exprs[1:], group_descs[1:]):
        combined_expr = (
            (combined_expr & expr) if group_logic_enum == RowLogic.AND else (combined_expr | expr)
        )
        combined_desc = f"{combined_desc} {_LOGIC_SYMBOLS[group_logic_enum]} {desc}"

    combined_expr = combined_expr.fill_null(False)

    keep = action_enum == RowAction.KEEP

    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        if keep:
            return lf.filter(combined_expr)
        else:
            return lf.filter(~combined_expr)

    result_df = _op(df.lazy()).collect()

    preview, _preview_note = _build_preview(result_df)

    summary = [{
        "action": action_enum.value,
        "condition_description": combined_desc,
        "group_logic": group_logic_enum.value,
        "rows_kept": result_df.height,
        "rows_removed": df.height - result_df.height,
    }]

    if _preview_note:
        summary.append(_preview_note)
    return {
        "op": _op,
        "preview": preview,
        "summary": summary,
        "rows_before": df.height,
        "rows_after": result_df.height,
    }


def exec_dim_reduct(
    df: pl.DataFrame,
    method: str,
    n_components: int,
    target: str | None = None,
    standardize: bool = True,
) -> dict:
    try:
        method_enum = DimReductMethod(method)
    except ValueError:
        return {
            "error": f"未知的降维方法: {method}",
            "hint": "支持的方法: pca, lda",
        }

    if n_components < 1:
        return {
            "error": f"n_components 必须为正整数，当前值: {n_components}",
        }

    schema = df.schema

    if method_enum == DimReductMethod.PCA:
        numeric_cols = [c for c, d in schema.items() if d.is_numeric()]
        if not numeric_cols:
            return {"error": "数据集中没有数值列，无法进行PCA降维"}
        if n_components > len(numeric_cols):
            return {
                "error": (
                    f"n_components ({n_components}) 超过数值列数"
                    f" ({len(numeric_cols)})"
                ),
            }

        # 拟合阶段：Polars 标准化参数（非 sklearn StandardScaler）
        std_stats: dict[str, tuple[float, float]] = {}
        if standardize:
            for c in numeric_cols:
                std_stats[c] = (df[c].mean(), df[c].std())

        if standardize:
            working_df = df.clone()
            std_exprs = []
            for c in numeric_cols:
                m, s = std_stats[c]
                std_exprs.append(((pl.col(c) - m) / s).alias(c))
            working_df = working_df.with_columns(std_exprs)
        else:
            working_df = df.clone()

        arr = working_df.select(numeric_cols).to_numpy()
        if np.any(np.isnan(arr)):
            return {
                "error": "数值列中存在 NaN 值，请先使用 execute_empty_value 处理空值",
            }

        pca = PCA(n_components=n_components)
        pca.fit(arr)

        pc_cols = [f"PC{i + 1}" for i in range(n_components)]
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
        final_cols = non_numeric_cols + pc_cols

        weights: list[tuple[float, list[float]]] = []
        for i in range(n_components):
            bias = 0.0
            ws: list[float] = []
            for j, c in enumerate(numeric_cols):
                w = float(pca.components_[i, j])
                if standardize:
                    m, s = std_stats[c]
                    w = w / s
                    bias -= (m / s + float(pca.mean_[j])) * float(pca.components_[i, j])
                else:
                    bias -= float(pca.mean_[j]) * float(pca.components_[i, j])
                ws.append(w)
            weights.append((bias, ws))

        explained_var = [round(float(v), 6) for v in pca.explained_variance_ratio_]

        def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
            exprs = []
            for i, (bias_val, ws) in enumerate(weights):
                expr = pl.lit(bias_val)
                for j, c in enumerate(numeric_cols):
                    if ws[j] != 0.0:
                        expr = expr + pl.col(c) * ws[j]
                exprs.append(expr.alias(pc_cols[i]))
            return lf.with_columns(exprs).select(final_cols)

        result_df = _op(df.lazy()).collect()

        preview, _preview_note = _build_preview(result_df)

        feats_shown, feats_info = cap_list(numeric_cols, MAX_COLUMN_LIST)
        pca_summary = {
            "method": "pca",
            "n_components": n_components,
            "original_features": feats_shown,
            "n_original_features": len(numeric_cols),
            "new_columns": pc_cols,
            "explained_variance_ratio": explained_var,
            "standardized": standardize,
        }
        if feats_info:
            pca_summary["original_features_truncated"] = feats_info
        summary = [pca_summary]

        if _preview_note:
            summary.append(_preview_note)
        return {
            "op": _op,
            "preview": preview,
            "summary": summary,
            "rows_before": df.height,
            "rows_after": result_df.height,
        }

    if method_enum == DimReductMethod.LDA:
        if target is None:
            return {"error": "LDA 降维需要指定 target 目标列名"}
        if target not in schema.names():
            return {"error": f"目标列 '{target}' 不存在"}

        numeric_cols = [
            c for c, d in schema.items() if d.is_numeric() and c != target
        ]
        if not numeric_cols:
            return {
                "error": "数据集中除目标列外没有数值特征列，无法进行LDA降维",
            }

        target_null_count = df[target].null_count()
        df_clean = (
            df.filter(pl.col(target).is_not_null())
            if target_null_count > 0
            else df.clone()
        )
        rows_dropped = df.height - df_clean.height

        target_vals = df_clean[target].drop_nulls().unique()
        n_classes = target_vals.len()
        if n_classes < 2:
            return {
                "error": (
                    f"目标列 '{target}' 只有 {n_classes} 个类别，LDA 需要至少 2 个类别"
                ),
            }

        max_components = min(n_classes - 1, len(numeric_cols))
        if n_components > max_components:
            return {
                "error": (
                    f"n_components ({n_components}) 超过最大允许值"
                    f" ({max_components})，LDA 最大分量数 = min(类别数-1, 特征数)"
                ),
            }

        std_stats: dict[str, tuple[float, float]] = {}
        if standardize:
            for c in numeric_cols:
                std_stats[c] = (df_clean[c].mean(), df_clean[c].std())

        if standardize:
            working_df = df_clean.clone()
            std_exprs = []
            for c in numeric_cols:
                m, s = std_stats[c]
                std_exprs.append(((pl.col(c) - m) / s).alias(c))
            working_df = working_df.with_columns(std_exprs)
        else:
            working_df = df_clean.clone()

        X = working_df.select(numeric_cols).to_numpy()
        if np.any(np.isnan(X)):
            return {
                "error": "数值特征列中存在 NaN 值，请先使用 execute_empty_value 处理空值",
            }

        y = df_clean[target].to_numpy()
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))

        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(X, y_encoded)
        transformed = lda.transform(X)
        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)
        if transformed.shape[1] == 0:
            return {
                "error": (
                    "LDA 未能提取有效分量，可能是因为数据量过少或类内散布矩阵奇异。"
                    "建议增加样本量或减少 n_components。"
                ),
            }

        n_components = transformed.shape[1]

        ld_cols = [f"LD{i + 1}" for i in range(n_components)]
        keep_cols = [c for c in df.columns if c not in numeric_cols]
        final_cols = keep_cols + ld_cols

        weights: list[tuple[float, list[float]]] = []
        for i in range(n_components):
            bias = 0.0
            ws: list[float] = []
            for j, c in enumerate(numeric_cols):
                w = float(lda.scalings_[j, i])
                if standardize:
                    m, s = std_stats[c]
                    w = w / s
                    bias -= (m / s + float(lda.xbar_[j])) * float(lda.scalings_[j, i])
                else:
                    bias -= float(lda.xbar_[j]) * float(lda.scalings_[j, i])
                ws.append(w)
            weights.append((bias, ws))

        def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
            exprs = []
            for i, (bias_val, ws) in enumerate(weights):
                expr = pl.lit(bias_val)
                for j, c in enumerate(numeric_cols):
                    if ws[j] != 0.0:
                        expr = expr + pl.col(c) * ws[j]
                exprs.append(expr.alias(ld_cols[i]))
            return lf.with_columns(exprs).select(final_cols)

        result_df = _op(df_clean.lazy()).collect()

        preview, _preview_note = _build_preview(result_df)

        feats_shown, feats_info = cap_list(numeric_cols, MAX_COLUMN_LIST)
        lda_summary = {
            "method": "lda",
            "n_components": n_components,
            "target": target,
            "original_features": feats_shown,
            "n_original_features": len(numeric_cols),
            "new_columns": ld_cols,
            "n_classes": n_classes,
            "standardized": standardize,
            "rows_dropped": rows_dropped,
        }
        if feats_info:
            lda_summary["original_features_truncated"] = feats_info
        summary = [lda_summary]

        if _preview_note:
            summary.append(_preview_note)
        return {
            "op": _op,
            "preview": preview,
            "summary": summary,
            "rows_before": df.height,
            "rows_after": result_df.height,
        }

    return {
        "error": f"未知的降维方法: {method}",
        "hint": "支持的方法: pca, lda",
    }


_MONO_EXPR = {
    MonoTransform.COS: lambda c, s: c.cos(),
    MonoTransform.SIN: lambda c, s: c.sin(),
    MonoTransform.TAN: lambda c, s: c.tan(),
    MonoTransform.EXP: lambda c, s: c.exp(),
    MonoTransform.LOG: (
        lambda c, s: c.log(s["base"]) if s.get("base") is not None else c.log()
    ),
    MonoTransform.SQRT: lambda c, s: c.sqrt(),
    MonoTransform.SQUARE: lambda c, s: c.pow(2),
    MonoTransform.POWER: lambda c, s: c.pow(s["exponent"]),
    MonoTransform.LINEAR: lambda c, s: c * s["a"] + s["b"],
    MonoTransform.RECIPROCAL: lambda c, s: 1.0 / c,
    MonoTransform.ABS: lambda c, s: c.abs(),
}

_MONO_NAME_PREFIX = {
    MonoTransform.COS: "cos",
    MonoTransform.SIN: "sin",
    MonoTransform.TAN: "tan",
    MonoTransform.EXP: "exp",
    MonoTransform.LOG: "log",
    MonoTransform.SQRT: "sqrt",
    MonoTransform.SQUARE: "square",
    MonoTransform.POWER: "power",
    MonoTransform.LINEAR: "linear",
    MonoTransform.RECIPROCAL: "recip",
    MonoTransform.ABS: "abs",
}

_COMBO_EXPR = {
    CombineMethod.PRODUCT: lambda exprs: reduce(lambda a, b: a * b, exprs),
    CombineMethod.SUM: lambda exprs: reduce(lambda a, b: a + b, exprs),
    CombineMethod.MEAN: lambda exprs: reduce(lambda a, b: a + b, exprs) / len(exprs),
    CombineMethod.DIFFERENCE: lambda exprs: reduce(lambda a, b: a - b, exprs),
    CombineMethod.RATIO: lambda exprs: reduce(lambda a, b: a / b, exprs),
}


def _default_mono_name(spec: dict) -> str:
    """为一元变换生成默认新列名，如 cos_x、linear_x、power2_x。"""
    method = MonoTransform(spec["method"])
    prefix = _MONO_NAME_PREFIX[method]
    if method == MonoTransform.POWER:
        exp = float(spec.get("exponent", 2.0))
        exp_str = str(int(exp)) if exp.is_integer() else str(exp).replace(".", "_")
        prefix = f"power{exp_str}"
    return f"{prefix}_{spec['column']}"


def _count_bad(series: pl.Series) -> tuple[int, int]:
    """统计浮点列中 NaN 与 Inf 的数量；非浮点列返回 (0, 0)。"""
    if series.dtype not in (pl.Float32, pl.Float64):
        return 0, 0
    nan_count = int(series.is_nan().fill_null(False).sum())
    inf_count = int(series.is_infinite().fill_null(False).sum())
    return nan_count, inf_count


def exec_transform_mono(
    df: pl.DataFrame,
    specs: list[dict],
) -> dict:
    """对单个数值列应用一元变换，产出新列附加到数据中（保留原列）。"""
    if not specs:
        return {"error": "specs 不能为空"}

    schema = df.schema
    columns_set = set(schema.names())

    unknown = [s["column"] for s in specs if s["column"] not in columns_set]
    if unknown:
        return {
            "error": f"以下列名不存在：{unknown}",
            "hint": "请先调用 explore_schema 确认列名。",
        }

    non_numeric = [
        s["column"] for s in specs if not schema[s["column"]].is_numeric()
    ]
    if non_numeric:
        return {"error": f"以下列不是数值类型，无法进行一元变换：{non_numeric}"}

    resolved = [s.get("output_name") or _default_mono_name(s) for s in specs]

    collisions = []
    seen_new: list[str] = []
    for out in resolved:
        if out in columns_set or out in seen_new:
            collisions.append(out)
        seen_new.append(out)
    if collisions:
        return {
            "error": f"新列名与已有列或本批其它新列冲突：{collisions}",
            "hint": "请为 output_name 指定不冲突的名称。",
        }

    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        exprs = []
        for spec, out in zip(specs, resolved):
            method = MonoTransform(spec["method"])
            expr = _MONO_EXPR[method](pl.col(spec["column"]), spec)
            exprs.append(expr.alias(out))
        return lf.with_columns(exprs)

    result_df = _op(df.lazy()).collect()
    summary = []
    for spec, out in zip(specs, resolved):
        method = MonoTransform(spec["method"])
        warnings = []
        nan_count, inf_count = _count_bad(result_df[out])
        if nan_count:
            warnings.append(
                f"变换后产生 {nan_count} 个 NaN（可能超出定义域，如对负数取对数或平方根）"
            )
        if inf_count:
            warnings.append(
                f"变换后产生 {inf_count} 个无穷值（可能存在除零，如对 0 取倒数）"
            )

        summary.append({
            "source_column": spec["column"],
            "method": method.value,
            "output_column": out,
            "warnings": warnings,
        })

    preview, _preview_note = _build_preview(result_df)

    if _preview_note:
        summary.append(_preview_note)
    return {
        "op": _op,
        "preview": preview,
        "summary": summary,
        "rows_before": df.height,
        "rows_after": result_df.height,
    }


def exec_transform_combination(
    df: pl.DataFrame,
    columns: list[str],
    method: str,
    output_name: str,
) -> dict:
    """对多个数值列做算术组合（交叉特征），产出一个新列附加到数据中（保留原列）。"""
    try:
        method_enum = CombineMethod(method)
    except ValueError:
        return {
            "error": f"未知的组合方法: {method}",
            "hint": "支持的方法: product, sum, mean, difference, ratio",
        }

    if not output_name:
        return {"error": "output_name 不能为空"}

    columns = list(dict.fromkeys(columns))
    if len(columns) < 2:
        return {"error": f"组合特征至少需要 2 个不同的列，当前: {columns}"}

    schema = df.schema
    columns_set = set(schema.names())
    unknown = [c for c in columns if c not in columns_set]
    if unknown:
        return {
            "error": f"以下列名不存在：{unknown}",
            "hint": "请先调用 explore_schema 确认列名。",
        }

    non_numeric = [c for c in columns if not schema[c].is_numeric()]
    if non_numeric:
        return {"error": f"以下列不是数值类型，无法进行组合：{non_numeric}"}

    if output_name in columns_set:
        return {
            "error": f"新列名与已有列冲突：{output_name}",
            "hint": "请为 output_name 指定不冲突的名称。",
        }

    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        exprs = [pl.col(c) for c in columns]
        combined = _COMBO_EXPR[method_enum](exprs)
        return lf.with_columns(combined.alias(output_name))

    result_df = _op(df.lazy()).collect()

    warnings = []
    nan_count, inf_count = _count_bad(result_df[output_name])
    if nan_count:
        warnings.append(f"组合后产生 {nan_count} 个 NaN")
    if inf_count:
        warnings.append(f"组合后产生 {inf_count} 个无穷值（可能存在除零）")

    summary = [{
        "source_columns": columns,
        "method": method_enum.value,
        "output_column": output_name,
        "warnings": warnings,
    }]

    preview, _preview_note = _build_preview(result_df)

    if _preview_note:
        summary.append(_preview_note)
    return {
        "op": _op,
        "preview": preview,
        "summary": summary,
        "rows_before": df.height,
        "rows_after": result_df.height,
    }
