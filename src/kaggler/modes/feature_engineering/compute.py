import polars as pl

from kaggler.shared.serialization import safe_val
from kaggler.modes.feature_engineering.types import FillMethod


def exec_empty(
    df: pl.DataFrame,
    pairs: list[dict],
) -> dict:
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

    result_df = df.clone()
    nulls_before = {col: result_df[col].null_count() for pair in pairs for col in [pair["column"]]}

    fill_exprs = []
    delete_columns = []
    summary = []

    for pair in pairs:
        col = pair["column"]
        action = FillMethod(pair["action"])
        dtype = schema[col]

        if action == FillMethod.DELETE:
            delete_columns.append(col)
            continue

        if action == FillMethod.ZERO:
            if dtype.is_numeric():
                fill_exprs.append(pl.col(col).fill_null(0))
            elif dtype == pl.String:
                fill_exprs.append(pl.col(col).fill_null("0"))
            elif dtype == pl.Boolean:
                fill_exprs.append(pl.col(col).fill_null(False))
            else:
                summary.append({
                    "column": col,
                    "method": action.value,
                    "nulls_before": nulls_before[col],
                    "nulls_filled": 0,
                    "warnings": [f"列类型 {dtype} 不支持零值填充，已跳过"],
                })
                continue
        elif action == FillMethod.AVG:
            fill_exprs.append(pl.col(col).fill_null(pl.col(col).mean()))
        elif action == FillMethod.MEDIAN:
            fill_exprs.append(pl.col(col).fill_null(pl.col(col).median()))
        elif action == FillMethod.MODE:
            mode_val = result_df[col].drop_nulls().mode()
            if mode_val is not None and mode_val.len() > 0:
                fill_val = mode_val[0]
                fill_exprs.append(pl.col(col).fill_null(fill_val))
            else:
                summary.append({
                    "column": col,
                    "method": action.value,
                    "nulls_before": nulls_before[col],
                    "nulls_filled": 0,
                    "warnings": ["列全部为空值，无法确定众数，已跳过"],
                })
                continue

    if fill_exprs:
        result_df = result_df.with_columns(fill_exprs)

    for pair in pairs:
        col = pair["column"]
        action = FillMethod(pair["action"])
        if action == FillMethod.DELETE:
            continue
        col_summary = {
            "column": col,
            "method": action.value,
            "nulls_before": nulls_before[col],
        }
        if not any(s["column"] == col for s in summary):
            nulls_after = result_df[col].null_count()
            col_summary["nulls_filled"] = nulls_before[col] - nulls_after
            col_summary["warnings"] = []
            summary.append(col_summary)

    if delete_columns:
        rows_before = result_df.height
        result_df = result_df.drop_nulls(subset=delete_columns)
        rows_after = result_df.height
        for col in delete_columns:
            summary.append({
                "column": col,
                "method": "delete",
                "nulls_before": nulls_before[col],
                "rows_deleted": rows_before - rows_after,
                "warnings": [],
            })

    preview_rows = result_df.head(3).to_dicts()
    preview = []
    for row in preview_rows:
        preview.append({k: safe_val(v) for k, v in row.items()})

    return {
        "processed_df": result_df,
        "preview": preview,
        "summary": summary,
        "rows_before": df.height,
        "rows_after": result_df.height,
    }
