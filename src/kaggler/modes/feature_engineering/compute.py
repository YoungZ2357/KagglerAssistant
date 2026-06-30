import numpy as np
import polars as pl
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

from kaggler.shared.serialization import safe_val
from kaggler.modes.feature_engineering.types import (
    DimReductMethod,
    EncodeMethod,
    FillMethod,
)


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
        dict: 包含 processed_df, preview, summary, rows_before, rows_after
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

    result_df = df.clone()
    columns_to_drop = []
    summary = []

    for pair in pairs:
        col = pair["column"]
        action = EncodeMethod(pair["action"])

        if action == EncodeMethod.ONE_HOT:
            null_count = result_df[col].null_count()
            non_null_vals = result_df[col].drop_nulls().unique()
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

            null_mask = result_df[col].is_null()

            dummies = result_df.select(pl.col(col)).to_dummies(col, drop_first=True)
            dummy_cols = [c for c in dummies.columns if not c.endswith("_null")]
            dummies = dummies.select(dummy_cols)

            if null_count > 0:
                for dc in dummy_cols:
                    dummies = dummies.with_columns(
                        pl.when(null_mask).then(None).otherwise(pl.col(dc)).alias(dc)
                    )

            result_df = result_df.with_columns(dummies)
            columns_to_drop.append(col)

            summary.append({
                "column": col,
                "method": action.value,
                "new_columns": dummy_cols,
                "unique_values": unique_count,
                "warnings": warnings,
            })

        elif action == EncodeMethod.LABEL:
            vals = result_df[col].drop_nulls().unique().to_list()
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

            result_df = result_df.with_columns(
                pl.when(pl.col(col).is_null())
                .then(None)
                .otherwise(pl.col(col).replace_strict(
                    old=list(mapping.keys()),
                    new=list(mapping.values()),
                    default=None,
                ))
                .cast(pl.Int64)
                .alias(col)
            )

            summary.append({
                "column": col,
                "method": action.value,
                "mapping": {str(k): v for k, v in mapping.items()},
                "warnings": [],
            })

    if columns_to_drop:
        result_df = result_df.drop(columns_to_drop)

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


def standardize_numeric(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    arr = df.select(columns).to_numpy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(arr)
    result = df.clone()
    for i, col in enumerate(columns):
        result = result.with_columns(pl.Series(col, scaled[:, i]))
    return result


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

    result_df = standardize_numeric(df, columns)

    preview_rows = result_df.head(3).to_dicts()
    preview = [{k: safe_val(v) for k, v in row.items()} for row in preview_rows]

    summary = [{
        "columns": columns,
        "description": "z-score标准化（均值=0，标准差=1）",
    }]

    return {
        "processed_df": result_df,
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

        working_df = df.clone()
        if standardize:
            working_df = standardize_numeric(working_df, numeric_cols)

        arr = working_df.select(numeric_cols).to_numpy()
        if np.any(np.isnan(arr)):
            return {
                "error": "数值列中存在 NaN 值，请先使用 execute_empty_value 处理空值",
            }

        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(arr)
        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)

        pc_cols = [f"PC{i + 1}" for i in range(n_components)]
        final_cols = [c for c in df.columns if c not in numeric_cols] + pc_cols

        drop_cols = [c for c, d in schema.items() if c in numeric_cols]
        result_df = df.clone().drop(drop_cols)
        pc_df = pl.from_numpy(transformed, schema=pc_cols)
        dupes = [c for c in pc_cols if c in result_df.columns]
        if dupes:
            result_df = result_df.drop(dupes)
        result_df = result_df.hstack(pc_df)
        result_df = result_df.select(final_cols)

        preview_rows = result_df.head(3).to_dicts()
        preview = [
            {k: safe_val(v) for k, v in row.items()} for row in preview_rows
        ]

        summary = [{
            "method": "pca",
            "n_components": n_components,
            "original_features": numeric_cols,
            "new_columns": pc_cols,
            "explained_variance_ratio": [
                round(float(v), 6) for v in pca.explained_variance_ratio_
            ],
            "standardized": standardize,
        }]

        return {
            "processed_df": result_df,
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

        working_df = df_clean.clone()
        if standardize:
            working_df = standardize_numeric(working_df, numeric_cols)

        X = working_df.select(numeric_cols).to_numpy()
        if np.any(np.isnan(X)):
            return {
                "error": "数值特征列中存在 NaN 值，请先使用 execute_empty_value 处理空值",
            }

        y = df_clean[target].to_numpy()
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))

        lda = LinearDiscriminantAnalysis(n_components=n_components)
        transformed = lda.fit_transform(X, y_encoded)
        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)
        if transformed.shape[1] == 0:
            return {
                "error": (
                    "LDA 未能提取有效分量，可能是因为数据量过少或类内散布矩阵奇异。"
                    "建议增加样本量或减少 n_components。"
                ),
            }

        ld_cols = [f"LD{i + 1}" for i in range(n_components)]
        keep_cols = [c for c in df.columns if c not in numeric_cols]
        final_cols = keep_cols + ld_cols

        result_df = df_clean.clone().drop(numeric_cols)
        ld_df = pl.from_numpy(transformed, schema=ld_cols)
        dupes = [c for c in ld_cols if c in result_df.columns]
        if dupes:
            result_df = result_df.drop(dupes)
        result_df = result_df.hstack(ld_df)
        result_df = result_df.select(final_cols)

        preview_rows = result_df.head(3).to_dicts()
        preview = [
            {k: safe_val(v) for k, v in row.items()} for row in preview_rows
        ]

        summary = [{
            "method": "lda",
            "n_components": n_components,
            "target": target,
            "original_features": numeric_cols,
            "new_columns": ld_cols,
            "n_classes": n_classes,
            "standardized": standardize,
            "rows_dropped": rows_dropped,
        }]

        return {
            "processed_df": result_df,
            "preview": preview,
            "summary": summary,
            "rows_before": df.height,
            "rows_after": result_df.height,
        }

    return {
        "error": f"未知的降维方法: {method}",
        "hint": "支持的方法: pca, lda",
    }