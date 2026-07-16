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
from kaggler.ir import IRSpec, op_from
from kaggler.ir.emit import _OP_SYMBOLS
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


_STAT_ACTIONS = (FillMethod.AVG, FillMethod.MEDIAN, FillMethod.MODE)


def _resolve_group(df: pl.DataFrame, pair: dict) -> tuple | None:
    """把 pair 的分组意图解析为 (group_col, group_breaks, warnings)，不分组则返回 None。

    - group_bins 为 None：按 group_by 原始取值直接分组（group_breaks=None）。
    - group_bins 有值：等宽分箱，内部切点 = edges[1:-1]（镜像 eda._box_data_raw 的
      edges = [min + i*(max-min)/bins] 公式）；常数列/全空列无法分箱，退化为直接分组并警告。
    切点提前算好写死，保证惰性重放确定性。
    """
    g = pair.get("group_by")
    if g is None:
        return None
    gbins = pair.get("group_bins")
    if gbins is None:
        return (g, None, [])
    clean = df[g].drop_nulls()
    if clean.len() == 0:
        return (g, None, [f"分组列 '{g}' 全为空，无法分箱，已退化为按取值分组"])
    gmin, gmax = float(clean.min()), float(clean.max())
    if gmax == gmin:
        return (g, None, [f"分组列 '{g}' 为常数，无法分箱，已退化为按取值分组"])
    edges = [gmin + i * (gmax - gmin) / gbins for i in range(gbins + 1)]
    return (g, edges[1:-1], [])


def _stat_expr(col_name: str, kind: str) -> pl.Expr:
    """统计量表达式:kind='mean' -> .mean();否则 .median()。"""
    c = pl.col(col_name)
    return c.mean() if kind == "mean" else c.median()


def _to_float(v):
    """把 polars 标量统计量归一成 Python float(或 None)。

    保证 repr 干净(避免 numpy 标量的 ``np.float64(..)`` 之类表示),供写死为常量。
    """
    return None if v is None else float(v)


def _freeze_group_stats(
    df: pl.DataFrame,
    col: str,
    kind: str,
    group_col: str,
    group_breaks: list | None,
) -> list[tuple]:
    """在训练集上 eager 算出「组键 -> 统计量」映射,供分组填充写死为常量。

    - 分组键与运行/重放时的表达式严格一致:未分箱按原始取值(``pl.col(g)``),
      分箱按 ``pl.col(g).cut(breaks).cast(pl.String)``(cut 产出 Enum,统一转字符串
      标签,replace_strict 匹配才稳定)。
    - drop 掉 null 组键与统计量为空的组:这两类在填充时都应回落到全局兜底,
      从映射里剔除后天然由外层 ``.fill_null(global)`` 兜住(与旧 ``.over()`` 行为等价)。
    - 按组键排序:group_by 结果行序不确定,排序保证导出的代码片段可复现。
    """
    key_expr = (
        pl.col(group_col)
        if group_breaks is None
        else pl.col(group_col).cut(group_breaks).cast(pl.String)
    )
    grp = (
        df.lazy()
        .group_by(key_expr.alias("_g"))
        .agg(_stat_expr(col, kind).alias("_s"))
        .collect()
    )
    pairs = [
        (k, _to_float(s))
        for k, s in zip(grp["_g"].to_list(), grp["_s"].to_list())
        if k is not None and s is not None
    ]
    pairs.sort(key=lambda kv: kv[0])
    return pairs


def _freeze_group_modes(
    df: pl.DataFrame,
    col: str,
    group_col: str,
    group_breaks: list | None,
) -> list[tuple]:
    """在训练集上 eager 算出「组键 -> 组内众数」映射,供分组 mode 填充冻结进 IR。

    与 _freeze_group_stats 同构:同一分组键表达式(分箱经 cut→cast(String))、
    drop 空键组与全空值组(回落全局兜底)、按组键排序保证可复现。
    组内众数平票时按取值排序取首,冻结结果确定性化(运行时 ``.over()`` 老路的
    平票顺序无保证 —— 差分测试须用平票无关数据)。
    """
    key_expr = (
        pl.col(group_col)
        if group_breaks is None
        else pl.col(group_col).cut(group_breaks).cast(pl.String)
    )
    grp = (
        df.lazy()
        .group_by(key_expr.alias("_g"))
        .agg(pl.col(col).drop_nulls().mode().sort().first().alias("_s"))
        .collect()
    )
    pairs = [
        (k, s)
        for k, s in zip(grp["_g"].to_list(), grp["_s"].to_list())
        if k is not None and s is not None
    ]
    pairs.sort(key=lambda kv: kv[0])
    return pairs


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

        # 分组填充校验：仅对统计量 action(avg/median/mode)生效；zero/delete 带 group_by
        # 不报错，后续忽略分组并在 summary 警告。
        group_by = pair.get("group_by")
        if group_by is not None and action in _STAT_ACTIONS:
            if group_by not in columns_set:
                unknown.append(group_by)
            elif group_by == col:
                dtype_errors.append({
                    "column": col,
                    "error": "group_by 不能与被填充列相同",
                })
            else:
                group_bins = pair.get("group_bins")
                if group_bins is not None:
                    if not schema[group_by].is_numeric():
                        dtype_errors.append({
                            "column": col,
                            "error": f"group_bins 仅适用于数值分组列，"
                            f"但 '{group_by}' 类型为 {schema[group_by]}",
                        })
                    elif group_bins < 2:
                        dtype_errors.append({
                            "column": col,
                            "error": f"group_bins 必须 >= 2，收到 {group_bins}",
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

    # 缺失标识列：在填充前生成 <col>_is_missing，保留“缺失本身即信息”。
    # 与 fill 放同一次 with_columns（下方 _op）——Polars 单次 with_columns 内所有表达式
    # 均读原始列，故标识列天然反映填充前的缺失状态，顺序不会出错。
    indicator_columns: list[tuple[str, str]] = []  # (源列, 标识列名)
    indicator_skips: list[dict] = []
    indicator_conflicts: list[dict] = []
    _existing_names = set(columns_set)
    _new_names: set[str] = set()
    for pair in pairs:
        if not pair.get("add_indicator"):
            continue
        col = pair["column"]
        action = FillMethod(pair["action"])
        indicator_name = f"{col}_is_missing"
        if action == FillMethod.DELETE:
            indicator_skips.append({
                "column": col,
                "action": "add_indicator",
                "warnings": ["action 为 delete，缺失行将被删除，标识列无意义，已跳过"],
            })
            continue
        if nulls_before[col] == 0:
            indicator_skips.append({
                "column": col,
                "action": "add_indicator",
                "warnings": ["该列无缺失值，标识列将全为 0 无信息，已跳过"],
            })
            continue
        if indicator_name in _existing_names or indicator_name in _new_names:
            indicator_conflicts.append({
                "column": col,
                "indicator_column": indicator_name,
                "error": f"标识列名 '{indicator_name}' 与已有列或本批其它标识列冲突",
            })
            continue
        _new_names.add(indicator_name)
        indicator_columns.append((col, indicator_name))

    if indicator_conflicts:
        return {
            "error": "部分缺失标识列名冲突",
            "details": indicator_conflicts,
            "hint": "请先用 drop_columns 移除同名列，或改用其它列名后重试。",
        }

    fill_specs: list[dict] = []
    delete_columns: list[str] = []
    delete_group_ignored: set[str] = set()  # delete 列中带了 group_by(无意义，已忽略)
    skip_summary: list[dict] = list(indicator_skips)

    for pair in pairs:
        col = pair["column"]
        action = FillMethod(pair["action"])
        dtype = schema[col]
        # zero/delete 带 group_by 无意义：忽略分组并在 summary 警告。
        group_ignored = pair.get("group_by") is not None and action not in _STAT_ACTIONS

        if action == FillMethod.DELETE:
            delete_columns.append(col)
            if group_ignored:
                delete_group_ignored.add(col)
            continue

        if action == FillMethod.ZERO:
            _zero_warn = (
                ["group_by 对 zero 填充无意义，已忽略分组"] if group_ignored else []
            )
            if dtype.is_numeric():
                fill_specs.append({"column": col, "type": "zero", "value": 0,
                                   "warnings": _zero_warn})
            elif dtype == pl.String:
                fill_specs.append({"column": col, "type": "zero", "value": "0",
                                   "warnings": _zero_warn})
            elif dtype == pl.Boolean:
                fill_specs.append({"column": col, "type": "zero", "value": False,
                                   "warnings": _zero_warn})
            else:
                skip_summary.append({
                    "column": col,
                    "method": action.value,
                    "nulls_before": nulls_before[col],
                    "nulls_filled": 0,
                    "warnings": [f"列类型 {dtype} 不支持零值填充，已跳过"],
                })
            continue

        # 统计量填充(avg/median/mode)：解析分组意图。
        group = _resolve_group(df, pair)
        group_col = group[0] if group else None
        group_breaks = group[1] if group else None
        group_warns = group[2] if group else []

        if action in (FillMethod.AVG, FillMethod.MEDIAN):
            # 拟合阶段:在训练集上 eager 算出全局统计量(与分组映射),写死为常量。
            # 训练集“构造”整个模型 —— 导出的 pipeline 在验证/测试集上不得重算统计量。
            kind = "mean" if action == FillMethod.AVG else "median"
            global_stat = _to_float(
                df[col].mean() if kind == "mean" else df[col].median()
            )
            group_map = (
                _freeze_group_stats(df, col, kind, group_col, group_breaks)
                if group_col else []
            )
            fill_specs.append({"column": col, "type": kind,
                               "group_col": group_col, "group_breaks": group_breaks,
                               "global_stat": global_stat, "group_map": group_map,
                               "warnings": group_warns})
            continue

        if action == FillMethod.MODE:
            mode_val = df[col].drop_nulls().mode()
            if mode_val is not None and mode_val.len() > 0:
                fill_specs.append({"column": col, "type": "mode", "value": mode_val[0],
                                   "group_col": group_col, "group_breaks": group_breaks,
                                   "warnings": group_warns})
            else:
                skip_summary.append({
                    "column": col,
                    "method": action.value,
                    "nulls_before": nulls_before[col],
                    "nulls_filled": 0,
                    "warnings": ["列全部为空值，无法确定众数，已跳过"],
                })
            continue

    # IR payload:本步的唯一真相 —— 拟合常量(全局统计量/分组映射/众数)全部冻结。
    # 分组 mode 同样在拟合期冻结为「组键->众数」映射(消除旧 .over() 重放期重算)。
    _ir_fills: list[dict] = []
    for spec in fill_specs:
        c = spec["column"]
        if spec["type"] == "zero":
            _ir_fills.append({"column": c, "type": "zero", "value": spec["value"]})
        elif spec["type"] in ("mean", "median"):
            _ir_fills.append({
                "column": c, "type": spec["type"],
                "group_col": spec.get("group_col"),
                "group_breaks": spec.get("group_breaks"),
                "global_stat": spec["global_stat"],
                "group_map": [list(kv) for kv in spec["group_map"]],
            })
        elif spec["type"] == "mode":
            _gmap = (
                _freeze_group_modes(df, c, spec["group_col"], spec["group_breaks"])
                if spec.get("group_col") else []
            )
            _ir_fills.append({
                "column": c, "type": "mode", "value": spec["value"],
                "group_col": spec.get("group_col"),
                "group_breaks": spec.get("group_breaks"),
                "group_map": [list(kv) for kv in _gmap],
            })
    ir = IRSpec("fill_missing", {
        "indicators": [list(t) for t in indicator_columns],
        "fills": _ir_fills,
        "delete_columns": delete_columns,
    })

    _op = op_from(ir.kind, ir.params)
    result_df = _op(df.lazy()).collect()

    summary: list[dict] = list(skip_summary)
    _method_names = {"zero": "zero", "mean": "avg", "median": "median", "mode": "mode"}
    for spec in fill_specs:
        col = spec["column"]
        if any(s.get("column") == col for s in summary):
            continue
        nulls_after = result_df[col].null_count()
        method = _method_names[spec["type"]]
        if spec.get("group_col"):
            if spec["group_breaks"] is not None:
                method += (
                    f" (grouped by {spec['group_col']}[{len(spec['group_breaks']) + 1} bins])"
                )
            else:
                method += f" (grouped by {spec['group_col']})"
        entry = {
            "column": col,
            "method": method,
            "nulls_before": nulls_before[col],
            "nulls_filled": nulls_before[col] - nulls_after,
            "warnings": list(spec.get("warnings", [])),
        }
        # 分组+全局兜底后仍有残余空值(如整列全空)时如实上报。
        if nulls_after > 0:
            entry["nulls_remaining"] = nulls_after
        summary.append(entry)

    if delete_columns:
        rows_before = df.height
        rows_after = result_df.height
        for col in delete_columns:
            summary.append({
                "column": col,
                "method": "delete",
                "nulls_before": nulls_before[col],
                "rows_deleted": rows_before - rows_after,
                "warnings": (
                    ["group_by 对 delete 无意义，已忽略分组"]
                    if col in delete_group_ignored else []
                ),
            })

    for src, indicator_name in indicator_columns:
        summary.append({
            "column": src,
            "indicator_column": indicator_name,
            "action": "add_indicator",
            "nulls_flagged": nulls_before[src],
            "warnings": [],
        })

    preview, _preview_note = _build_preview(result_df)

    if _preview_note:
        summary.append(_preview_note)
    return {
        "op": _op,
        "ir": ir,
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

    # IR payload:one_hot 的取值集合、label 的完整映射(平行对列表)均冻结。
    _ir_specs: list[dict] = []
    for spec in encode_specs:
        if spec["type"] == "one_hot":
            _ir_specs.append({
                "column": spec["column"], "type": "one_hot",
                "values": spec["values"], "has_nulls": spec["has_nulls"],
            })
        else:
            _ir_specs.append({
                "column": spec["column"], "type": "label",
                "mapping": [[k, v] for k, v in spec["mapping"].items()],
            })
    ir = IRSpec("encode", {"specs": _ir_specs, "drop_columns": columns_to_drop})

    _op = op_from(ir.kind, ir.params)
    result_df = _op(df.lazy()).collect()

    preview, _preview_note = _build_preview(result_df)

    if _preview_note:
        summary.append(_preview_note)
    return {
        "op": _op,
        "ir": ir,
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

    ir = IRSpec("standardize", {"stats": [
        {"column": c, "mean": stats[c][0], "std": stats[c][1]} for c in columns
    ]})

    _op = op_from(ir.kind, ir.params)
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
        "ir": ir,
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

    ir = IRSpec("drop_columns", {"columns": columns})

    _op = op_from(ir.kind, ir.params)
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
        "ir": ir,
        "preview": preview,
        "summary": summary,
        "rows_before": df.height,
        "rows_after": result_df.height,
    }


# 一元运算符（无比较值）的 description 文案。运算符号表(_OP_SYMBOLS)与执行/代码
# 语义同源,直接复用 ir.emit 的注册表(ComparisonOp 是 str Enum,可直接查 str 键表)。
_NULL_OP_DESC = {
    ComparisonOp.IS_NULL: "为空",
    ComparisonOp.IS_NOT_NULL: "非空",
}

_LOGIC_SYMBOLS = {
    RowLogic.AND: "且",
    RowLogic.OR: "或",
}


def _format_condition_value(value) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _validate_conditions(
    groups: list[dict],
    schema,
    *,
    require_value: bool = True,
) -> dict | None:
    """校验两层条件结构（groups → conditions）。

    通过返回 ``None``；否则返回可直接回传给模型的 error dict。被 exec_filter_rows
    与 exec_create_indicator 共用，是条件校验的单一真相。一元运算符
    (is_null/is_not_null) 无需比较值，跳过值-类型兼容校验。
    """
    if not groups:
        return {"error": "groups 不能为空"}

    empty_group_indices = [i for i, g in enumerate(groups) if not g.get("conditions")]
    if empty_group_indices:
        return {"error": f"以下分组下标的 conditions 不能为空：{empty_group_indices}"}

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
        value = cond.get("value")
        dtype = schema[col]

        try:
            op = ComparisonOp(cond["op"])
        except ValueError:
            errors.append({"column": col, "error": f"未知的比较运算符: {cond['op']}"})
            continue

        # 一元运算符：无需比较值，跳过类型兼容校验。
        if op in _NULL_OP_DESC:
            continue

        if require_value and value is None:
            errors.append({
                "column": col,
                "error": f"运算符 {op.value} 必须提供比较值 value",
            })
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
                "error": f"列 '{col}' 的类型 {dtype} 暂不支持条件比较",
            })

    if errors:
        return {"error": "部分条件不合法", "details": errors}
    return None


def _describe_conditions(
    groups: list[dict],
    group_logic_enum: RowLogic,
) -> str:
    """把两层条件结构编译成人类可读的中文描述(仅供 summary 展示)。

    被 exec_filter_rows 与 exec_create_indicator 共用。条件的**执行**语义在
    ir.interpret._conditions_expr,**代码**投影在 ir.emit._conditions_code ——
    此处只负责文案,不再参与任何计算。
    """
    def leaf_desc(cond: dict) -> str:
        op = ComparisonOp(cond["op"])
        if op in _NULL_OP_DESC:
            return f"{cond['column']} {_NULL_OP_DESC[op]}"
        return f"{cond['column']} {_OP_SYMBOLS[op]} {_format_condition_value(cond.get('value'))}"

    group_descs = []
    for g in groups:
        logic = RowLogic(g["logic"])
        conds = g["conditions"]
        desc = leaf_desc(conds[0])
        for cond in conds[1:]:
            desc = f"{desc} {_LOGIC_SYMBOLS[logic]} {leaf_desc(cond)}"
        group_descs.append(f"({desc})" if len(conds) > 1 else desc)

    combined_desc = group_descs[0]
    for desc in group_descs[1:]:
        combined_desc = f"{combined_desc} {_LOGIC_SYMBOLS[group_logic_enum]} {desc}"
    return combined_desc


def _ir_condition_groups(groups: list[dict]) -> list[dict]:
    """把(已通过校验的)两层条件结构归一成 JSON 原生形态,供 IR payload 使用。

    工具层经 ``model_dump(mode="json")`` 传入的已是 str,但直接调用方可能传
    enum 实例 —— 统一经 enum 取 ``.value`` 归一,一元运算符的 value 归一为 None。
    """
    out = []
    for g in groups:
        out.append({
            "logic": RowLogic(g["logic"]).value,
            "conditions": [
                {
                    "column": cond["column"],
                    "op": ComparisonOp(cond["op"]).value,
                    "value": cond.get("value"),
                }
                for cond in g["conditions"]
            ],
        })
    return out


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

    validation_error = _validate_conditions(groups, df.schema, require_value=True)
    if validation_error is not None:
        return validation_error

    combined_desc = _describe_conditions(groups, group_logic_enum)

    ir = IRSpec("filter_rows", {
        "groups": _ir_condition_groups(groups),
        "group_logic": group_logic_enum.value,
        "action": action_enum.value,
    })

    _op = op_from(ir.kind, ir.params)
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
        "ir": ir,
        "preview": preview,
        "summary": summary,
        "rows_before": df.height,
        "rows_after": result_df.height,
    }


def exec_create_indicator(
    df: pl.DataFrame,
    groups: list[dict],
    group_logic: str,
    output_name: str,
) -> dict:
    """根据逻辑条件新建一个 0/1 指示符列（Int8）。

    满足组合条件的行取 1，否则取 0；条件涉及列为空、无法判断真假的行按 0 处理
    （combined_expr 末尾 fill_null(False)）。行数不变，仅追加一列。
    """
    try:
        group_logic_enum = RowLogic(group_logic)
    except ValueError:
        return {"error": f"未知的 group_logic: {group_logic}", "hint": "支持的值: and, or"}

    if not output_name:
        return {"error": "output_name 不能为空"}

    if output_name in set(df.schema.names()):
        return {
            "error": f"新列名与已有列冲突：{output_name}",
            "hint": "请为 output_name 指定不冲突的名称。",
        }

    validation_error = _validate_conditions(groups, df.schema, require_value=True)
    if validation_error is not None:
        return validation_error

    combined_desc = _describe_conditions(groups, group_logic_enum)

    ir = IRSpec("create_indicator", {
        "groups": _ir_condition_groups(groups),
        "group_logic": group_logic_enum.value,
        "output_name": output_name,
    })

    _op = op_from(ir.kind, ir.params)
    result_df = _op(df.lazy()).collect()

    flagged = int(result_df[output_name].sum())
    preview, _preview_note = _build_preview(result_df)

    summary = [{
        "output_column": output_name,
        "condition_description": combined_desc,
        "group_logic": group_logic_enum.value,
        "rows_flagged": flagged,
        "rows_total": result_df.height,
    }]

    if _preview_note:
        summary.append(_preview_note)
    return {
        "op": _op,
        "ir": ir,
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

        ir = IRSpec("dim_reduct", {
            "method": "pca",
            "components": [{"bias": b, "weights": ws} for b, ws in weights],
            "numeric_cols": numeric_cols,
            "out_cols": pc_cols,
            "final_cols": final_cols,
        })

        _op = op_from(ir.kind, ir.params)
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
            "ir": ir,
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

        # 注意:LDA 的 n_components 可能在拟合后收缩(上方 transformed.shape 处),
        # payload 用收缩后的 weights/ld_cols/final_cols 构造(此处已是最终值)。
        ir = IRSpec("dim_reduct", {
            "method": "lda",
            "components": [{"bias": b, "weights": ws} for b, ws in weights],
            "numeric_cols": numeric_cols,
            "out_cols": ld_cols,
            "final_cols": final_cols,
        })

        _op = op_from(ir.kind, ir.params)
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
            "ir": ir,
            "preview": preview,
            "summary": summary,
            "rows_before": df.height,
            "rows_after": result_df.height,
        }

    return {
        "error": f"未知的降维方法: {method}",
        "hint": "支持的方法: pca, lda",
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

    _ir_specs = []
    for spec, out in zip(specs, resolved):
        method = MonoTransform(spec["method"])
        s: dict = {"column": spec["column"], "method": method.value, "output_name": out}
        if method == MonoTransform.LOG:
            s["base"] = spec.get("base")
        elif method == MonoTransform.POWER:
            s["exponent"] = spec.get("exponent", 2.0)
        elif method == MonoTransform.LINEAR:
            s["a"] = spec.get("a", 1.0)
            s["b"] = spec.get("b", 0.0)
        _ir_specs.append(s)
    ir = IRSpec("transform_mono", {"specs": _ir_specs})

    _op = op_from(ir.kind, ir.params)
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
        "ir": ir,
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

    ir = IRSpec("transform_combination", {
        "columns": columns,
        "method": method_enum.value,
        "output_name": output_name,
    })

    _op = op_from(ir.kind, ir.params)
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
        "ir": ir,
        "preview": preview,
        "summary": summary,
        "rows_before": df.height,
        "rows_after": result_df.height,
    }
