"""三路差分测试(阶段 2 gate):双写与 IR 两投影的等价性护网。

同一输入帧上,断言以下三路产出**逐帧精确相等**(check_exact,不比字符串):
1. 现有直接构建的 Op 闭包(``result["op"]``,双写老路);
2. ``IR -> interpreter -> Op``(``build_op``);
3. ``IR -> emit -> Code -> exec``(``emit_code``)。

路径 2/3 的节点先过 ``dumps_ir -> loads_ir`` JSON 往返,把序列化保真度
(浮点 repr、映射键类型)纳入同一断言。三路相等同时实证反数据泄漏不变量:
拟合常量同源透传、零 refit。

覆盖矩阵对照测试基线缺口逐项补齐:one-hot(含/无 null)、label 各键型、
LDA、mono 全部 method、combination 全部 method、复杂布尔树、
ungrouped/grouped/binned 统计填充、冻结分组 mode(平票无关数据)。
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from kaggler.ir import IRNode, build_op, dumps_ir, emit_code, loads_ir
from kaggler.modes.feature_engineering.compute import (
    exec_create_indicator,
    exec_dim_reduct,
    exec_drop_columns,
    exec_empty,
    exec_encode,
    exec_filter_rows,
    exec_standardize,
    exec_transform_combination,
    exec_transform_mono,
)


def three_way(df: pl.DataFrame, result: dict) -> pl.DataFrame:
    """三路差分核心断言;返回 legacy 帧供额外值断言。"""
    assert "error" not in result, f"exec 返回错误: {result}"
    spec = result["ir"]
    node = loads_ir(dumps_ir(IRNode(
        version=1, kind=spec.kind, parents=[0], params=spec.params,
    )))

    f_legacy = result["op"](df.lazy()).collect()
    f_interp = build_op(node)(df.lazy()).collect()
    ns = {"pl": pl, "lf": df.lazy()}
    exec(emit_code(node), ns)
    f_code = ns["lf"].collect()

    assert_frame_equal(f_interp, f_legacy, check_exact=True)
    assert_frame_equal(f_code, f_legacy, check_exact=True)
    return f_legacy


@pytest.fixture
def dfx():
    """通用宽表:数值(含空)/无空数值/整数/类别(含空)/字符串(含空)/布尔(含空)/无空字符串。"""
    return pl.DataFrame({
        "num_a": [1.0, 2.0, None, 4.0, 5.0, None],
        "num_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "int_c": [1, 2, 2, 3, 3, 3],
        "cat": ["x", "y", "x", None, "y", "x"],
        "strv": ["p", None, "q", "p", None, "p"],
        "boolv": [True, False, None, True, False, True],
        "city": ["sh", "bj", "sh", "gz", "bj", "sh"],
    })


@pytest.fixture
def df_clean():
    """无空数值帧(dim_reduct / standardize 用)。"""
    return pl.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "f2": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
        "f3": [0.5, 1.5, 1.0, 2.5, 2.0, 3.0],
        "label": ["u", "v", "u", "v", "u", "v"],
    })


class TestFillMissing:
    @pytest.mark.parametrize("column", ["num_a", "strv", "boolv"])
    def test_zero_fill_by_dtype(self, dfx, column):
        three_way(dfx, exec_empty(dfx, [{"column": column, "action": "zero"}]))

    @pytest.mark.parametrize("action", ["avg", "median"])
    def test_stat_global(self, dfx, action):
        three_way(dfx, exec_empty(dfx, [{"column": "num_a", "action": action}]))

    @pytest.mark.parametrize("action", ["avg", "median"])
    def test_stat_grouped_categorical(self, dfx, action):
        three_way(dfx, exec_empty(dfx, [
            {"column": "num_a", "action": action, "group_by": "cat"},
        ]))

    @pytest.mark.parametrize("action", ["avg", "median"])
    def test_stat_grouped_binned(self, dfx, action):
        three_way(dfx, exec_empty(dfx, [
            {"column": "num_a", "action": action, "group_by": "num_b", "group_bins": 2},
        ]))

    def test_mode_global(self, dfx):
        out = three_way(dfx, exec_empty(dfx, [{"column": "strv", "action": "mode"}]))
        assert out["strv"].null_count() == 0

    def test_mode_grouped_categorical(self):
        # 平票无关数据:组 x 众数 a;组 z 全空、组键 null 行的值也全空 ——
        # 两类都回落全局众数 a(冻结映射与 .over() 老路在此等价)。
        df = pl.DataFrame({
            "v": ["a", None, "a", "b", None, None],
            "g": ["x", "x", "x", "y", "z", None],
        })
        out = three_way(df, exec_empty(df, [
            {"column": "v", "action": "mode", "group_by": "g"},
        ]))
        assert out["v"].to_list() == ["a", "a", "a", "b", "a", "a"]

    def test_mode_grouped_binned(self):
        df = pl.DataFrame({
            "v": ["a", None, "a", "b", None, None],
            "gn": [1.0, 1.0, 2.0, 9.0, 9.0, 8.0],
        })
        out = three_way(df, exec_empty(df, [
            {"column": "v", "action": "mode", "group_by": "gn", "group_bins": 2},
        ]))
        # 箱 1(gn<=5)众数 a;箱 2 众数 b
        assert out["v"].to_list() == ["a", "a", "a", "b", "b", "b"]

    def test_delete_rows(self, dfx):
        out = three_way(dfx, exec_empty(dfx, [{"column": "num_a", "action": "delete"}]))
        assert out.height == 4

    def test_add_indicator(self, dfx):
        out = three_way(dfx, exec_empty(dfx, [
            {"column": "num_a", "action": "avg", "add_indicator": True},
        ]))
        assert out["num_a_is_missing"].to_list() == [0, 0, 1, 0, 0, 1]

    def test_mixed_multi_pair(self, dfx):
        three_way(dfx, exec_empty(dfx, [
            {"column": "strv", "action": "zero"},
            {"column": "num_a", "action": "avg", "group_by": "cat", "add_indicator": True},
            {"column": "boolv", "action": "mode"},
        ]))

    def test_all_null_column_stat(self):
        # 整列全空:global_stat=None 分支,表达式退化为原样列。
        df = pl.DataFrame(
            {"x": [1.0, 2.0], "an": [None, None]},
            schema_overrides={"an": pl.Float64},
        )
        out = three_way(df, exec_empty(df, [{"column": "an", "action": "avg"}]))
        assert out["an"].null_count() == 2


class TestEncode:
    def test_one_hot_with_nulls(self, dfx):
        out = three_way(dfx, exec_encode(dfx, [{"column": "cat", "action": "one_hot"}]))
        assert "cat" not in out.columns

    def test_one_hot_without_nulls(self, dfx):
        three_way(dfx, exec_encode(dfx, [{"column": "city", "action": "one_hot"}]))

    def test_one_hot_single_unique(self):
        df = pl.DataFrame({"only": ["k", "k", "k"], "x": [1, 2, 3]})
        out = three_way(df, exec_encode(df, [{"column": "only", "action": "one_hot"}]))
        assert out.columns == ["x"]  # drop_first 后无新列,原列被删

    def test_label_str_keys(self, dfx):
        out = three_way(dfx, exec_encode(dfx, [{"column": "city", "action": "label"}]))
        assert out["city"].dtype == pl.Int64

    def test_label_int_keys(self, dfx):
        three_way(dfx, exec_encode(dfx, [{"column": "int_c", "action": "label"}]))

    def test_label_float_keys(self, dfx):
        three_way(dfx, exec_encode(dfx, [{"column": "num_b", "action": "label"}]))

    def test_label_bool_keys(self, dfx):
        three_way(dfx, exec_encode(dfx, [{"column": "boolv", "action": "label"}]))

    def test_mixed_one_hot_and_label(self, dfx):
        three_way(dfx, exec_encode(dfx, [
            {"column": "cat", "action": "one_hot"},
            {"column": "int_c", "action": "label"},
        ]))


class TestStandardize:
    def test_multi_column(self, dfx):
        three_way(dfx, exec_standardize(dfx, ["num_b", "int_c"]))


class TestDropColumns:
    def test_basic(self, dfx):
        out = three_way(dfx, exec_drop_columns(dfx, ["num_a", "cat"]))
        assert set(out.columns) == {"num_b", "int_c", "strv", "boolv", "city"}


class TestFilterRows:
    def test_single_group_gt_keep(self, dfx):
        out = three_way(dfx, exec_filter_rows(
            dfx,
            groups=[{"logic": "and", "conditions": [
                {"column": "num_b", "op": "gt", "value": 25.0},
            ]}],
            group_logic="and",
            action="keep",
        ))
        assert out.height == 4

    def test_complex_tree_delete(self, dfx):
        three_way(dfx, exec_filter_rows(
            dfx,
            groups=[
                {"logic": "and", "conditions": [
                    {"column": "num_a", "op": "is_null"},
                    {"column": "num_b", "op": "ge", "value": 30.0},
                ]},
                {"logic": "or", "conditions": [
                    {"column": "cat", "op": "eq", "value": "y"},
                    {"column": "boolv", "op": "eq", "value": True},
                ]},
            ],
            group_logic="or",
            action="delete",
        ))

    def test_null_ops_mixed(self, dfx):
        three_way(dfx, exec_filter_rows(
            dfx,
            groups=[{"logic": "or", "conditions": [
                {"column": "strv", "op": "is_null"},
                {"column": "cat", "op": "is_not_null"},
            ]}],
            group_logic="and",
            action="keep",
        ))

    def test_str_eq_ne(self, dfx):
        three_way(dfx, exec_filter_rows(
            dfx,
            groups=[{"logic": "and", "conditions": [
                {"column": "city", "op": "ne", "value": "sh"},
                {"column": "strv", "op": "eq", "value": "p"},
            ]}],
            group_logic="and",
            action="keep",
        ))

    def test_int_boundaries(self, dfx):
        three_way(dfx, exec_filter_rows(
            dfx,
            groups=[{"logic": "and", "conditions": [
                {"column": "int_c", "op": "le", "value": 3},
                {"column": "int_c", "op": "lt", "value": 3},
            ]}],
            group_logic="and",
            action="keep",
        ))


class TestCreateIndicator:
    def test_complex_tree(self, dfx):
        out = three_way(dfx, exec_create_indicator(
            dfx,
            groups=[
                {"logic": "and", "conditions": [
                    {"column": "num_a", "op": "is_null"},
                ]},
                {"logic": "and", "conditions": [
                    {"column": "num_b", "op": "gt", "value": 45.0},
                ]},
            ],
            group_logic="or",
            output_name="flag",
        ))
        assert out["flag"].to_list() == [0, 0, 1, 0, 1, 1]

    def test_str_eq_tree(self, dfx):
        three_way(dfx, exec_create_indicator(
            dfx,
            groups=[{"logic": "and", "conditions": [
                {"column": "city", "op": "eq", "value": "sh"},
                {"column": "boolv", "op": "eq", "value": True},
            ]}],
            group_logic="and",
            output_name="sh_true",
        ))


class TestDimReduct:
    def test_pca_standardized(self, df_clean):
        out = three_way(df_clean, exec_dim_reduct(df_clean, "pca", 2))
        assert out.columns == ["label", "PC1", "PC2"]

    def test_pca_unstandardized(self, df_clean):
        three_way(df_clean, exec_dim_reduct(df_clean, "pca", 2, standardize=False))

    def test_lda(self, df_clean):
        out = three_way(df_clean, exec_dim_reduct(df_clean, "lda", 1, target="label"))
        assert out.columns == ["label", "LD1"]

    def test_lda_with_null_target(self):
        df = pl.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            "label": ["u", "v", "u", "v", "u", None],
        })
        three_way(df, exec_dim_reduct(df, "lda", 1, target="label"))


class TestTransformMono:
    @pytest.mark.parametrize("spec", [
        {"column": "num_b", "method": "cos"},
        {"column": "num_b", "method": "sin"},
        {"column": "num_b", "method": "tan"},
        {"column": "num_b", "method": "exp"},
        {"column": "num_b", "method": "log"},
        {"column": "num_b", "method": "log", "base": 2.0},
        {"column": "num_b", "method": "sqrt"},
        {"column": "num_b", "method": "square"},
        {"column": "num_b", "method": "power", "exponent": 3.0},
        {"column": "num_b", "method": "power", "exponent": 2.5},
        {"column": "num_b", "method": "linear", "a": 2.0, "b": 3.0},
        {"column": "num_b", "method": "reciprocal"},
        {"column": "num_b", "method": "abs"},
    ], ids=lambda s: f"{s['method']}{'_base' if 'base' in s else ''}"
       f"{'_' + str(s.get('exponent')) if 'exponent' in s else ''}")
    def test_all_methods(self, dfx, spec):
        three_way(dfx, exec_transform_mono(dfx, [dict(spec)]))

    def test_multi_spec_with_nulls(self, dfx):
        three_way(dfx, exec_transform_mono(dfx, [
            {"column": "num_a", "method": "sqrt"},
            {"column": "int_c", "method": "linear", "a": -1.0, "b": 0.5,
             "output_name": "neg_c"},
        ]))


class TestTransformCombination:
    @pytest.mark.parametrize(
        "method", ["product", "sum", "mean", "difference", "ratio"]
    )
    def test_all_methods(self, dfx, method):
        three_way(dfx, exec_transform_combination(
            dfx, ["num_b", "int_c"], method, f"combo_{method}",
        ))

    def test_three_columns_left_assoc(self, dfx):
        # 三列验证左结合语义(含空值传播)
        three_way(dfx, exec_transform_combination(
            dfx, ["num_b", "int_c", "num_a"], "difference", "d3",
        ))
