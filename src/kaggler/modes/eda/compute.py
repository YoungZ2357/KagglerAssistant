import itertools
import math
from dataclasses import dataclass
from math import isnan, isinf

import numpy as np
import polars as pl
from scipy import stats


# 帮手函数 #################################################################
def _safe_val(v):
    """
    [HUMAN]将Polars值转化为安全的Python原生类型，以便于序列化
    Args:
        v: Polars内的数据值，可能为多种类型

    Returns:

    """
    if v is None:
        return None
    if isinstance(v, float):
        if isnan(v) or isinf(v):
            return None
        return round(v, 6)
    if hasattr(v, "__int__") and not isinstance(v, (bool, str)):
        return int(v)
    return str(v)


# def _dumps(obj: dict | list) -> str:
#     return json.dumps(obj, ensure_ascii=False)

def _pearson_batch(
    df: pl.DataFrame, pairs: list[tuple[str, str]]
) -> list[float | None]:
    """
    [HUMAN]批量计算皮尔逊相关系数(仅允许 数值 v.s. 数值)
    Args:
        df: Polars DataFrame格式数据
        pairs: 待运算的对列表

    Returns:

    """
    if not pairs:
        return []
    values = df.select([
        pl.corr(a, b, method="pearson").alias(str(i))
        for i, (a, b) in enumerate(pairs)
    ]).row(0)
    return [_safe_val(v) for v in values]

def _cramers_v(df: pl.DataFrame, col_a: str, col_b: str) -> float | None:
    """
    [HUMAN]Cramers V 相关系数运算，针对离散列
    Args:
        df: Polars DataFrame 格式数据
        col_a: 运算列1
        col_b: 运算列2

    Returns:

    """
    pair_df = df.select([col_a, col_b]).drop_nulls()
    n = pair_df.height
    if n == 0:
        return None

    ct = (
        pair_df
        .group_by([col_a, col_b])
        .agg(pl.len().alias("observed"))
        .with_columns([
            pl.col("observed").sum().over(col_a).alias("row_total"),
            pl.col("observed").sum().over(col_b).alias("col_total"),
        ])
    )

    # chi2、r、c 三个量合并进单次 select，减少 Python ↔ Polars 往返。
    # expected 内联为表达式，避免额外 with_columns。
    expected = (
        pl.col("row_total").cast(pl.Float64)
        * pl.col("col_total").cast(pl.Float64)
        / n
    )
    chi2, r, c = ct.select([
        ((pl.col("observed") - expected).pow(2) / expected).sum().alias("chi2"),
        pl.col(col_a).n_unique().alias("r"),
        pl.col(col_b).n_unique().alias("c"),
    ]).row(0)

    min_dim = min(r, c) - 1
    if min_dim == 0:
        return None

    return math.sqrt(chi2 / (n * min_dim))


def _eta_squared(df: pl.DataFrame, cat_col: str, num_col: str) -> float | None:
    """
    [HUMAN]计算分类变量和连续变量之间的相关比(组间相关比)
    Args:
        df: Polars DataFrame 格式数据
        cat_col: 离散列
        num_col: 连续列

    Returns:

    """
    pair_df = df.select([cat_col, num_col]).drop_nulls()
    if pair_df.height == 0:
        return None

    # overall_mean 和 ss_total 合并为单次 select，原来是两次单独 .item()。
    overall_mean, ss_total = pair_df.select([
        pl.col(num_col).mean().alias("overall_mean"),
        ((pl.col(num_col) - pl.col(num_col).mean()).pow(2)).sum().alias("ss_total"),
    ]).row(0)

    if overall_mean is None or ss_total == 0:
        return None

    # 原实现：group_by().agg() 取组均值和组大小，再单独 .item() 计算 ss_between。
    # 改为：over() 将每行映射到其组均值，.sum() 即 Σ_rows (ȳ_g - ȳ)²
    #       = Σ_g n_g*(ȳ_g - ȳ)² = SS_between，无需 group_by 和单独的组大小列。
    ss_between = pair_df.select(
        ((pl.col(num_col).mean().over(cat_col) - overall_mean).pow(2)).sum()
    ).item()

    return ss_between / ss_total


@dataclass(frozen=True)
class BinnedColumn:
    """
    [HUMAN] 数值列等宽分箱后的机器友好结构，供下游计算工具（如分布拟合 / 卡方拟合优度检验）直接消费。

    约定：
    - counts 为定长向量，长度 == len(edges) - 1，包含计数为 0 的空箱；
    - edges 为等宽切分边界，长度 == bins + 1，保留完整精度（不做四舍五入）；
    - sum(counts) == total - null_count（空值不计入任何箱）。
    """
    column: str
    total: int               # 总行数，含空值
    null_count: int          # 空值数量
    min_val: float | None     # 非空最小值；全空时为 None
    max_val: float | None     # 非空最大值；全空时为 None
    edges: list[float]        # 分箱边界，全空时为 []
    counts: list[int]         # 各箱观测频数，全空时为 []


def _box_data_raw(df: pl.DataFrame, column: str, bins: int = 10) -> BinnedColumn:
    """
    [HUMAN] 数值列等宽分箱核心逻辑（机器侧）。仅支持数值列，空值在分箱前被剔除。

    与面向 LLM 的 get_boxed_data 区别：本函数返回定长、对齐的计数/边界，
    适合作为分布拟合优度检验等下游工具的输入；不做任何展示用的字符串化或取整。

    Args:
        df: Polars DataFrame 格式数据
        column: 待分箱的数值列
        bins: 等宽分箱数量（>= 1）

    Returns:
        BinnedColumn

    Raises:
        ValueError: bins < 1，或列为非数值类型。
    """
    if bins < 1:
        raise ValueError(f"bins 必须 >= 1，收到 {bins}。")

    dtype = df.schema[column]
    if not dtype.is_numeric():
        raise ValueError(
            f"列 '{column}' 类型为 {dtype}，_box_data_raw 仅支持数值列。"
            "分类列的分布请使用 get_boxed_data 的频率表。"
        )

    series = df.get_column(column)
    total = series.len()
    null_count = series.null_count()

    clean = series.drop_nulls()
    if clean.len() == 0:
        return BinnedColumn(column, total, null_count, None, None, [], [])

    min_val = float(clean.min())
    max_val = float(clean.max())

    # 退化情形：常数列无法等宽切分，全部非空值计入单一箱。
    if max_val == min_val:
        return BinnedColumn(
            column, total, null_count, min_val, max_val,
            [min_val, max_val], [clean.len()],
        )

    # 显式提供边界，保证返回定长计数（含空箱），且首箱左闭、其余左开右闭，
    # sum(counts) == clean.len()。相较 value_counts() 不会丢失计数为 0 的箱。
    edges = [min_val + i * (max_val - min_val) / bins for i in range(bins + 1)]
    counts = (
        clean
        .hist(bins=edges, include_breakpoint=False, include_category=False)
        .to_series()
        .to_list()
    )
    return BinnedColumn(
        column, total, null_count, min_val, max_val,
        edges, [int(c) for c in counts],
    )


def _render_numeric_observation(raw: BinnedColumn) -> dict:
    """
    [HUMAN] 将机器侧的 BinnedColumn 渲染为面向 LLM 的观察字典：
    箱标注字符串化、数值取整。get_boxed_data 与 distribution_evaluation 共用，
    避免"观察"出现两套不一致的表示。
    """
    bin_list = []
    for i, count in enumerate(raw.counts):
        lo, hi = raw.edges[i], raw.edges[i + 1]
        # 首箱左闭右闭，其余左开右闭，与 _box_data_raw 的切分语义一致。
        left = "[" if i == 0 else "("
        label = f"{left}{_safe_val(lo)}, {_safe_val(hi)}]"
        bin_list.append({"bin": label, "count": count})

    return {
        "column": raw.column,
        "dtype": "numeric",
        "total": raw.total,
        "null_count": raw.null_count,
        "min_val": _safe_val(raw.min_val),
        "max_val": _safe_val(raw.max_val),
        "bins": bin_list,
    }

# 工具函数后端 ###############################################################

def get_correlation(df: pl.DataFrame, columns: list[str]) -> dict:
    """
    [HUMAN]相关性分析工具运算函数。使用工具函数额外封装并与LLM绑定。LLM/Agent 只需要决定列名称，列类型将由函数内部完成处理。
    Args:
        df: Polars DataFrame 格式数据
        columns: 需要计算相关系数的列

    Returns: 序列化后的，可供LLM/Agent使用的相关系数结果，带有数值结果和对应的解释

    """

    if len(columns) < 2:
        return {"error": "至少需要 2 列才能计算相关性。"}

    schema = df.schema
    unknown = [c for c in columns if c not in schema.names()]
    if unknown:
        return {
            "error": f"以下列名不存在：{unknown}",
            "hint": "请先调用 explore_schema 确认列名。",
        }

    numeric_cols = [c for c in columns if schema[c].is_numeric()]
    categorical_cols = [c for c in columns if not schema[c].is_numeric()]

    df = df.select(columns)

    results: dict[str, list] = {}

    # 连续 vs 连续 → Pearson
    # 原实现：Python 层逐对调用 _pearson()，每次独立触发 Polars 计算。
    # 改为：所有列对打包进单次 df.select()，由 Polars 引擎统一调度（可并行）。
    if len(numeric_cols) >= 2:
        pairs = list(itertools.combinations(numeric_cols, 2))
        corr_values = _pearson_batch(df, pairs)
        results["pearson"] = sorted(
            [
                {"column_a": a, "column_b": b, "value": v}
                for (a, b), v in zip(pairs, corr_values)
            ],
            key=lambda p: abs(p["value"] or 0),
            reverse=True,
        )

    # 分类 vs 分类 → Cramér's V
    # 每对的列联表结构不同，无法跨对批量化，仍逐对调用；
    # 但单次调用本身已由上方 _cramers_v 优化。
    if len(categorical_cols) >= 2:
        results["cramers_v"] = sorted(
            [
                {"column_a": a, "column_b": b, "value": _safe_val(_cramers_v(df, a, b))}
                for a, b in itertools.combinations(categorical_cols, 2)
            ],
            key=lambda p: abs(p["value"] or 0),
            reverse=True,
        )

    # 分类 vs 连续 → Eta²（同上，每对独立）
    if categorical_cols and numeric_cols:
        results["eta_squared"] = sorted(
            [
                {"column_a": cat, "column_b": num, "value": _safe_val(_eta_squared(df, cat, num))}
                for cat in categorical_cols
                for num in numeric_cols
            ],
            key=lambda p: abs(p["value"] or 0),
            reverse=True,
        )

    if not results:
        return {"error": "给定列的组合无法计算任何相关性。"}

    return {
        "column_types": {
            c: "numeric" if c in numeric_cols else "categorical"
            for c in columns
        },
        "method_descriptions": {
            "pearson": "Pearson 相关系数，范围 -1 到 1，衡量线性相关强度",
            "cramers_v": "Cramér's V，范围 0 到 1，衡量分类变量间的关联强度",
            "eta_squared": "Eta²，范围 0 到 1，衡量分类变量对连续变量的解释力",
        },
        "results": results,
    }


def get_schema_report(df: pl.DataFrame) -> dict:
    """
    [HUMAN]获取数据集的结构信息，用于让LLM/Agent对数据集有初步感知，避免与数据集直接接触
    Returns: 数据库schema结果
    Args:
        df: Polars DataFrame 格式数据

    Returns:

    """
    # 获取结构
    schema = df.schema
    col_names = schema.names()
    dtypes = [schema[name] for name in col_names]

    total_rows = df.height

    # 初步统计信息(空值、独特值)
    stats_df = df.select([
        pl.all().null_count().name.prefix("null_"),
        pl.all().n_unique().name.prefix("unique_")
    ])

    # 获取样本数据
    head_df = df.head(3)

    columns = []
    for name, dtype in zip(col_names, dtypes):
        null_cnt = int(stats_df[0, f"null_{name}"])
        unique_cnt = int(stats_df[0, f"unique_{name}"])
        samples = [_safe_val(v) for v in head_df[name].to_list()]
        columns.append({
            "name": name,
            "dtype": str(dtype),
            "null_count": null_cnt,
            "null_rate": round(null_cnt / total_rows, 4) if total_rows else 0,
            "n_unique": unique_cnt,
            "sample_values": samples,
        })
    return {
        "total_rows": total_rows,
        "total_columns": len(col_names),
        "columns": columns,
    }


def get_descriptive_statistics(df: pl.DataFrame, columns: list[str]) -> dict:
    """
    [HUMAN] 对指定列进行描述性数据分析. 包含如下内容：行数、空值数、均值、中位数、标准差、最小值、最大值、四分位数
    Args:
        df: Polars DataFrame 结构数据
        columns: 待分析列

    Returns:

    """
    if not columns:
        return {"error": "未指定任何列。若列名未知，请先调用 explore_schema 获取数据结构。"}

    schema = df.schema
    numeric_cols = [c for c in columns if schema[c].is_numeric()]
    non_numeric = [c for c in columns if c not in numeric_cols]

    if not numeric_cols:
        return {
            "error": "指定的列均为非数值类型，无法计算描述性统计。",
            "non_numeric_columns": non_numeric,
        }

    agg_exprs = []
    for col in numeric_cols:
        agg_exprs.extend([
            pl.col(col).count().alias(f"{col}__count"),
            pl.col(col).null_count().alias(f"{col}__null_count"),
            pl.col(col).mean().alias(f"{col}__mean"),
            pl.col(col).median().alias(f"{col}__median"),
            pl.col(col).std().alias(f"{col}__std"),
            pl.col(col).min().alias(f"{col}__min"),
            pl.col(col).quantile(0.25).alias(f"{col}__q1"),
            pl.col(col).quantile(0.75).alias(f"{col}__q3"),
            pl.col(col).max().alias(f"{col}__max"),
        ])

    stats_row = df.select(agg_exprs)

    results = []
    for col in numeric_cols:
        results.append({
            "column": col,
            "count": _safe_val(stats_row[0, f"{col}__count"]),
            "null_count": _safe_val(stats_row[0, f"{col}__null_count"]),
            "mean": _safe_val(stats_row[0, f"{col}__mean"]),
            "median": _safe_val(stats_row[0, f"{col}__median"]),
            "std": _safe_val(stats_row[0, f"{col}__std"]),
            "min": _safe_val(stats_row[0, f"{col}__min"]),
            "q1": _safe_val(stats_row[0, f"{col}__q1"]),
            "q3": _safe_val(stats_row[0, f"{col}__q3"]),
            "max": _safe_val(stats_row[0, f"{col}__max"]),
        })

    output = {"stats": results}
    if non_numeric:
        output["skipped_non_numeric"] = non_numeric

    return output

def get_boxed_data(df: pl.DataFrame, column: str, bins: int = 10) -> dict:
    """
    [HUMAN]获取分箱后的数据，可以用于分析数据分布。该数据既可以直接让LLM进行观察，也可以作为"分箱后数据"要求LLM用于调用分布拟合函数
    Args:
        df:
        column: 待分箱列
        bins: 分箱数量

    Returns:

    """
    schema = df.schema
    dtype = schema[column]

    if dtype.is_numeric():
        # 复用机器侧分箱逻辑，再交由共享渲染器转成 LLM 易读的观察字典。
        return _render_numeric_observation(_box_data_raw(df, column, bins))
    else:
        col_series = df.select(column).to_series()
        total = len(col_series)
        null_count = col_series.null_count()
        full_unique = col_series.n_unique()

        top_n = 20
        freq_df = (
            col_series
            .value_counts()
            .sort(by="count", descending=True)
            .head(top_n)
        )
        freq_list = []
        for row in freq_df.iter_rows(named=True):
            cnt = int(row["count"])
            freq_list.append({
                "value": _safe_val(row[column]),
                "count": cnt,
                "proportion": round(cnt / total, 4) if total else 0,
            })
        return {
            "column": column,
            "dtype": "categorical",
            "total": total,
            "null_count": null_count,
            "n_unique": full_unique,
            "truncated": full_unique > top_n,
            "frequencies": freq_list,
        }

# 拟合优度配置（后端常量）#####################################################
# 两套方法：
# - chi2（默认）：基于分箱的 Pearson 卡方拟合优度。每个分布仅做 1 次 MLE + 解析 p 值，
#   耗时 ~毫秒，与样本量几乎无关，是「数据是否服从某分布」的快速判定首选。
# - monte_carlo（保留）：scipy.stats.goodness_of_fit（KS + 参数化自举），更严格但每个
#   分布要重复 MLE 数百次，gamma 等数值优化分布在大列上可达数十秒，故下采样 + 降重抽样。
_FIT_METHODS = ("chi2", "monte_carlo")
_DEFAULT_FIT_METHOD = "chi2"
# 容错别名 → 规范方法名：模型/调用方可能给出大小写或近义写法（如 "mc"、"Monte Carlo"），
# 归一后再匹配，避免"用户要求蒙特卡洛却被静默回退到 chi2"。
_FIT_METHOD_ALIASES = {
    "chi2": "chi2", "chisquare": "chi2", "chi_square": "chi2",
    "chi-square": "chi2", "chisq": "chi2",
    "monte_carlo": "monte_carlo", "montecarlo": "monte_carlo",
    "monte-carlo": "monte_carlo", "mc": "monte_carlo", "ks": "monte_carlo",
}


def _normalize_fit_method(method: str) -> str:
    """把调用方给的 method 归一到规范方法名；无法识别时回退默认（chi2）。"""
    key = str(method).strip().lower().replace(" ", "_")
    return _FIT_METHOD_ALIASES.get(key, _DEFAULT_FIT_METHOD)

_KS_MC_SAMPLES = 199      # 蒙特卡洛重抽样次数（原 999；gamma MLE 是瓶颈，降到 199 约 5x 提速）
_MC_MAX_SAMPLES = 5000    # 蒙特卡洛拟合前对超大列下采样的上限，避免单列耗时随行数爆炸
_KS_RANDOM_SEED = 0       # 固定种子，保证工具多次调用结果可复现（含下采样）
_MIN_FIT_SAMPLES = 8      # 有效样本量低于此值则跳过拟合优度检验
_CHI2_MIN_EXPECTED = 5.0  # 卡方检验每箱期望频数下限，低于此与相邻箱合并以满足卡方近似前提

# 候选分布为后端常量：LLM 只需指定列，不参与分布族的选择。
# 第三项 requires_positive 标记仅在数据严格为正时才有统计意义的分布（含位移参数也强制跳过，
# 避免在含 0/负值的数据上给出误导性拟合）。
_CANDIDATE_DISTRIBUTIONS = [
    ("normal", stats.norm, False),
    ("uniform", stats.uniform, False),
    ("exponential", stats.expon, False),
    ("lognormal", stats.lognorm, True),
    ("gamma", stats.gamma, True),
]


def _param_names(dist) -> list[str]:
    """分布的有序参数名：形状参数（若有）+ loc + scale，与 dist.fit 返回元组一一对应。"""
    shapes = [s.strip() for s in dist.shapes.split(",")] if dist.shapes else []
    return [*shapes, "loc", "scale"]


def _merge_low_expected(
    observed: np.ndarray, expected: np.ndarray, min_expected: float
) -> tuple[np.ndarray, np.ndarray]:
    """从左到右累积合并期望频数过低的相邻箱，使每个合并箱的期望 >= min_expected。

    卡方近似要求各箱期望频数不过小（经验阈值 5）；尾部残余不足一组时并入最后一组。
    """
    merged_o: list[float] = []
    merged_e: list[float] = []
    acc_o = acc_e = 0.0
    for o, e in zip(observed, expected):
        acc_o += float(o)
        acc_e += float(e)
        if acc_e >= min_expected:
            merged_o.append(acc_o)
            merged_e.append(acc_e)
            acc_o = acc_e = 0.0
    if acc_e > 0:
        if merged_e:
            merged_o[-1] += acc_o
            merged_e[-1] += acc_e
        else:
            merged_o.append(acc_o)
            merged_e.append(acc_e)
    return np.asarray(merged_o), np.asarray(merged_e)


def _chi_square_fit(name: str, dist, values: np.ndarray, raw: BinnedColumn) -> dict:
    """单分布卡方拟合优度检验（解析法）：1 次 MLE 估参 + 按分箱期望频数算卡方统计量。

    期望频数由拟合分布的 CDF 在分箱边界上的差分给出；首/末箱分别向 ∓∞ 延伸，
    使期望概率和为 1（覆盖落在 [min, max] 之外的概率质量）。自由度按"估计参数个数"
    修正（df = 合并后箱数 - 1 - 估参数），与蒙特卡洛对"参数由数据估计"的校正同源。
    """
    params = dist.fit(values)                 # 一次 MLE；蒙特卡洛要重复数百次
    frozen = dist(*params)

    edges = np.asarray(raw.edges, dtype=float)
    cdf = frozen.cdf(edges)
    probs = np.diff(cdf)
    probs[0] = cdf[1]              # 首箱左端延伸到 -∞：P(X <= edges[1])
    probs[-1] = 1.0 - cdf[-2]     # 末箱右端延伸到 +∞：P(X > edges[-2])

    observed = np.asarray(raw.counts, dtype=float)
    expected = observed.sum() * probs
    obs_m, exp_m = _merge_low_expected(observed, expected, _CHI2_MIN_EXPECTED)

    n_params = len(params)
    dof = len(obs_m) - 1 - n_params
    if dof < 1:
        raise ValueError(
            f"有效箱数 {len(obs_m)} 不足以在估计 {n_params} 个参数后保留自由度（df={dof}）。"
        )

    statistic = float(np.sum((obs_m - exp_m) ** 2 / exp_m))
    p_value = float(stats.chi2.sf(statistic, dof))
    return {
        "distribution": name,
        "statistic": _safe_val(statistic),
        "dof": dof,
        "p_value": _safe_val(p_value),
        "params": {
            k: _safe_val(float(v)) for k, v in zip(_param_names(dist), params)
        },
    }


def _monte_carlo_fit(name: str, dist, values: np.ndarray) -> dict:
    """单分布 KS 拟合优度（scipy.goodness_of_fit，参数化自举 p 值）。

    超过 _MC_MAX_SAMPLES 的列先无放回下采样（固定种子），把"每个 MC 样本重复 MLE"
    的成本压到可控范围；这只影响 p 值精度，不改变方法本身。
    """
    sample = values
    if sample.size > _MC_MAX_SAMPLES:
        rng = np.random.default_rng(_KS_RANDOM_SEED)
        sample = rng.choice(sample, _MC_MAX_SAMPLES, replace=False)
    res = stats.goodness_of_fit(
        dist, sample, statistic="ks",
        n_mc_samples=_KS_MC_SAMPLES,
        random_state=_KS_RANDOM_SEED,
    )
    return {
        "distribution": name,
        "statistic": _safe_val(float(res.statistic)),
        "p_value": _safe_val(float(res.pvalue)),
        "params": {
            k: _safe_val(float(v))
            for k, v in res.fit_result.params._asdict().items()
        },
    }


def distribution_evaluation(
    df: pl.DataFrame, column: str, bins: int = 10, method: str = _DEFAULT_FIT_METHOD,
) -> dict:
    """
    [HUMAN] 数值列分布评估工具后端：同时给出 (a) 分箱观测数据供 LLM 直接观察，
    (b) 针对后端候选分布的拟合优度检验结果。LLM 只需指定列。

    两种检验方法（method）：
    - "chi2"（默认）：基于分箱的 Pearson 卡方拟合优度。每个分布仅做 1 次 MLE，解析得到
      p 值，耗时与样本量几乎无关（毫秒级），适合作为"是否服从某分布"的默认快速判定。
    - "monte_carlo"：scipy.stats.goodness_of_fit（KS + 参数化自举）。统计上更严格，但每个
      分布要重复数百次 MLE，大列耗时显著（已做下采样 + 降重抽样优化）。

    两种方法的 candidate 结构一致（distribution / statistic / p_value / params），仅 chi2
    额外带 dof；p 值含义相同：越大越无法拒绝"数据来自该分布"的原假设。

    Args:
        df: Polars DataFrame 格式数据
        column: 待评估的数值列
        bins: 观测/卡方分箱数量（影响 observation 粒度；chi2 直接以此分箱算期望频数）
        method: "chi2"（默认）或 "monte_carlo"；非法取值回退为默认。

    Returns:
        {column, observation, fit} 的字典；列不存在 / 非数值时返回 {"error": ...}。
    """
    schema = df.schema
    if column not in schema.names():
        return {
            "error": f"列 '{column}' 不存在。",
            "hint": "请先调用 explore_schema 确认列名。",
        }
    if not schema[column].is_numeric():
        return {
            "error": f"列 '{column}' 为非数值类型，无法进行分布拟合。仅数值列支持拟合优度检验。",
            "hint": "分类列的取值分布请使用 distribution_analysis_raw 查看频率表。",
        }

    method = _normalize_fit_method(method)

    raw = _box_data_raw(df, column, bins)
    observation = _render_numeric_observation(raw)

    values = df.get_column(column).drop_nulls().to_numpy()
    n = int(values.size)

    if method == "chi2":
        fit: dict = {
            "method": "Pearson 卡方拟合优度（MLE 估参 + 解析 p 值，自由度按估参数修正）",
            "statistic_name": "chi_square",
            "sample_size": n,
            "caveat": (
                "基于分箱的卡方近似（每箱期望 < 5 时与相邻箱合并）；自由度已扣除估计参数个数。"
                "p 值越大越无法拒绝“数据来自该分布”的原假设——只能说明不矛盾，不能据此证明分布成立。"
            ),
        }
    else:
        fit = {
            "method": "Kolmogorov–Smirnov 拟合优度（MLE 估参 + 蒙特卡洛 p 值）",
            "statistic_name": "ks",
            "sample_size": n,
            "caveat": (
                f"p 值由蒙特卡洛模拟估计（{_KS_MC_SAMPLES} 次重抽样，样本超 {_MC_MAX_SAMPLES} 时下采样），"
                "已校正参数由数据估计带来的偏差；"
                "p 值越大越无法拒绝“数据来自该分布”的原假设——只能说明不矛盾，不能据此证明分布成立。"
            ),
        }

    if n < _MIN_FIT_SAMPLES:
        fit["error"] = f"有效样本量 {n} 不足（需 >= {_MIN_FIT_SAMPLES}），跳过拟合优度检验。"
        return {"column": column, "observation": observation, "fit": fit}

    min_val = float(values.min())
    candidates: list[dict] = []
    skipped: list[dict] = []
    for name, dist, requires_positive in _CANDIDATE_DISTRIBUTIONS:
        if requires_positive and min_val <= 0:
            skipped.append({
                "distribution": name,
                "reason": "该分布要求严格正值，但数据最小值 <= 0。",
            })
            continue
        try:
            if method == "chi2":
                candidates.append(_chi_square_fit(name, dist, values, raw))
            else:
                candidates.append(_monte_carlo_fit(name, dist, values))
        except Exception as e:
            skipped.append({
                "distribution": name,
                "reason": f"拟合失败：{type(e).__name__}",
            })

    # p 值降序（越大越可能服从）；p 值并列时按检验统计量升序（拟合越好）。
    candidates.sort(key=lambda c: (-(c["p_value"] or 0), c["statistic"] or 1))
    fit["candidates"] = candidates
    fit["best_fit"] = candidates[0]["distribution"] if candidates else None
    if skipped:
        fit["skipped"] = skipped

    return {"column": column, "observation": observation, "fit": fit}


__all__ = [
    "get_schema_report",
    "get_correlation",
    "get_descriptive_statistics",
    "get_boxed_data",
    "distribution_evaluation",
]