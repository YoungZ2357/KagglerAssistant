"""IR interpreter 与 code generator 的孤立单测(阶段 1,手工构造节点)。

覆盖 brief §6 阶段 1 要求:纯标量参数 op(standardize)、矩阵参数 op
(dim_reduct,nested array 往返)、无参数 op(drop/filter)、source;
多父骨架见 test_ir_schema。每个样例断言 interpreter 建出的 Op 可在样本
数据上运行,且 emit 的 Code exec 后与 Op 结果逐帧精确相等。

全量 9 kind × 各参数形态的三路差分(对照现有 exec_*)属阶段 2 的
test_ir_diff.py,此处不重复。
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from kaggler.ir import (
    IRNode,
    build_loader,
    build_op,
    dumps_ir,
    emit_code,
    emit_source_expr,
    loads_ir,
)


def _node(kind, params, parents=None):
    """构造节点并强制过一次 JSON 往返——序列化保真度纳入每个断言。"""
    return loads_ir(dumps_ir(IRNode(
        version=1,
        kind=kind,
        parents=[0] if parents is None else parents,
        params=params,
    )))


def _run_code(code: str, df: pl.DataFrame) -> pl.DataFrame:
    ns = {"pl": pl, "lf": df.lazy()}
    exec(code, ns)
    return ns["lf"].collect()


def _assert_op_code_equal(node, df: pl.DataFrame) -> pl.DataFrame:
    """Op 结果与 Code exec 结果逐帧精确相等;返回 Op 结果供值断言。"""
    op_result = build_op(node)(df.lazy()).collect()
    code_result = _run_code(emit_code(node), df)
    assert_frame_equal(code_result, op_result, check_exact=True)
    return op_result


@pytest.fixture
def df():
    return pl.DataFrame({
        "a": [1.0, 2.0, None, 4.0],
        "b": [10.0, 20.0, 30.0, 40.0],
        "g": ["x", "y", "x", None],
        "s": ["p", None, "q", None],
        "n": [1, 2, 2, 3],
    })


class TestStandardize:
    """纯标量参数 op。"""

    def test_op_runs_and_code_matches(self, df):
        node = _node("standardize", {"stats": [
            {"column": "b", "mean": 25.0, "std": 12.909944487358056},
        ]})
        out = _assert_op_code_equal(node, df)
        assert out["b"][0] == (10.0 - 25.0) / 12.909944487358056
        assert out.columns == df.columns


class TestDimReduct:
    """矩阵参数 op(nested array 经 JSON 往返)。"""

    def test_op_runs_and_code_matches(self, df):
        node = _node("dim_reduct", {
            "method": "pca",
            "components": [
                {"bias": -1.5, "weights": [0.25, 2.0]},
                {"bias": 0.0, "weights": [0.0, -0.5]},  # 零权重跳项路径
            ],
            "numeric_cols": ["b", "n"],
            "out_cols": ["PC1", "PC2"],
            "final_cols": ["g", "s", "PC1", "PC2"],
        })
        out = _assert_op_code_equal(node, df)
        assert out.columns == ["g", "s", "PC1", "PC2"]
        assert out["PC1"][0] == -1.5 + 10.0 * 0.25 + 1 * 2.0
        assert out["PC2"][0] == 0.0 + 1 * -0.5


class TestNoParamOps:
    """无拟合参数 op。"""

    def test_drop_columns(self, df):
        node = _node("drop_columns", {"columns": ["a", "s"]})
        out = _assert_op_code_equal(node, df)
        assert out.columns == ["b", "g", "n"]

    def test_filter_rows_complex_tree(self, df):
        node = _node("filter_rows", {
            "groups": [
                {"logic": "or", "conditions": [
                    {"column": "a", "op": "is_null", "value": None},
                    {"column": "b", "op": "ge", "value": 30.0},
                ]},
                {"logic": "and", "conditions": [
                    {"column": "g", "op": "is_not_null", "value": None},
                ]},
            ],
            "group_logic": "and",
            "action": "keep",
        })
        out = _assert_op_code_equal(node, df)
        # 满足 (a为空 or b>=30) and g非空 的行:仅第 3 行(a=None,b=30,g="x")
        assert out.height == 1
        assert out["b"][0] == 30.0

    def test_filter_rows_delete_action(self, df):
        node = _node("filter_rows", {
            "groups": [{"logic": "and", "conditions": [
                {"column": "n", "op": "eq", "value": 2},
            ]}],
            "group_logic": "and",
            "action": "delete",
        })
        out = _assert_op_code_equal(node, df)
        assert out["n"].to_list() == [1, 3]

    def test_create_indicator(self, df):
        node = _node("create_indicator", {
            "groups": [{"logic": "and", "conditions": [
                {"column": "b", "op": "gt", "value": 15.0},
            ]}],
            "group_logic": "and",
            "output_name": "big_b",
        })
        out = _assert_op_code_equal(node, df)
        assert out["big_b"].to_list() == [0, 1, 1, 1]

    def test_transform_mono_log_base_and_linear(self, df):
        node = _node("transform_mono", {"specs": [
            {"column": "b", "method": "log", "output_name": "log_b",
             "a": 1.0, "b": 0.0, "exponent": 2.0, "base": 10.0},
            {"column": "n", "method": "linear", "output_name": "lin_n",
             "a": 2.0, "b": -1.0, "exponent": 2.0, "base": None},
        ]})
        out = _assert_op_code_equal(node, df)
        assert out["log_b"][0] == 1.0  # log10(10)
        assert out["lin_n"].to_list() == [1.0, 3.0, 3.0, 5.0]

    def test_transform_combination_ratio(self, df):
        node = _node("transform_combination", {
            "columns": ["b", "n"], "method": "ratio", "output_name": "b_per_n",
        })
        out = _assert_op_code_equal(node, df)
        assert out["b_per_n"].to_list() == [10.0, 10.0, 15.0, 40.0 / 3]


class TestFillMissing:
    def test_frozen_group_mean_with_global_fallback(self, df):
        node = _node("fill_missing", {
            "indicators": [["a", "a_is_missing"]],
            "fills": [{
                "column": "a", "type": "mean", "group_col": "g",
                "group_breaks": None, "global_stat": 2.3333333333333335,
                "group_map": [["x", 1.0], ["y", 2.0]],
            }],
            "delete_columns": [],
        })
        out = _assert_op_code_equal(node, df)
        assert out["a_is_missing"].to_list() == [0, 0, 1, 0]
        assert out["a"][2] == 1.0  # g="x" 命中冻结组映射

    def test_frozen_group_mode(self, df):
        node = _node("fill_missing", {
            "indicators": [],
            "fills": [{
                "column": "s", "type": "mode", "value": "p",
                "group_col": "g", "group_breaks": None,
                "group_map": [["y", "q"]],
            }],
            "delete_columns": [],
        })
        out = _assert_op_code_equal(node, df)
        # 第 2 行 g="y" 命中冻结组众数 "q";第 4 行 g=None 回落全局 "p"
        assert out["s"].to_list() == ["p", "q", "q", "p"]

    def test_binned_group_map_and_zero_fill(self, df):
        node = _node("fill_missing", {
            "indicators": [],
            "fills": [
                {"column": "a", "type": "median", "group_col": "b",
                 "group_breaks": [20.0, 30.0], "global_stat": 9.0,
                 "group_map": [["(-inf, 20]", 5.0], ["(20, 30]", 7.0]]},
                {"column": "s", "type": "zero", "value": "0"},
            ],
            "delete_columns": [],
        })
        out = _assert_op_code_equal(node, df)
        assert out["a"][2] == 7.0  # b=30 落 (20,30] 箱
        assert out["s"].to_list() == ["p", "0", "q", "0"]

    def test_delete_rows_and_empty_op(self, df):
        node = _node("fill_missing", {
            "indicators": [], "fills": [], "delete_columns": ["a", "s"],
        })
        out = _assert_op_code_equal(node, df)
        assert out.height == 1  # 仅第 1 行 a 与 s 同时非空

        empty = _node("fill_missing", {
            "indicators": [], "fills": [], "delete_columns": [],
        })
        assert emit_code(empty).startswith("#")  # 注释-only 片段
        assert_frame_equal(_assert_op_code_equal(empty, df), df, check_exact=True)


class TestEncode:
    def test_one_hot_with_nulls(self, df):
        node = _node("encode", {
            "specs": [{"column": "g", "type": "one_hot",
                       "values": ["y"], "has_nulls": True}],
            "drop_columns": ["g"],
        })
        out = _assert_op_code_equal(node, df)
        assert "g" not in out.columns
        assert out["g_y"].to_list() == [False, True, False, None]

    def test_label_with_int_keys(self, df):
        node = _node("encode", {
            "specs": [{"column": "n", "type": "label",
                       "mapping": [[1, 0], [2, 1], [3, 2]]}],
            "drop_columns": [],
        })
        out = _assert_op_code_equal(node, df)
        assert out["n"].to_list() == [0, 1, 1, 2]


class TestSource:
    def test_loader_and_expr_match(self, tmp_path):
        p = tmp_path / "mini.csv"
        p.write_text("x,y\n1,a\n2,b\n", encoding="utf-8")
        node = loads_ir(dumps_ir(IRNode(
            version=0, kind="source", parents=[],
            params={"format": "csv", "path": str(p)},
        )))
        via_loader = build_loader(node)()
        via_expr = eval(emit_source_expr(node), {"pl": pl})
        assert_frame_equal(via_loader, pl.read_csv(str(p)), check_exact=True)
        assert_frame_equal(via_expr, via_loader, check_exact=True)

    def test_source_has_no_op_or_code(self):
        node = IRNode(version=0, kind="source", parents=[],
                      params={"format": "csv", "path": "x.csv"})
        with pytest.raises(ValueError, match="build_loader"):
            build_op(node)
        with pytest.raises(ValueError, match="emit_source_expr"):
            emit_code(node)

    def test_loader_rejects_derived(self):
        node = IRNode(version=1, kind="drop_columns", parents=[0],
                      params={"columns": ["a"]})
        with pytest.raises(ValueError, match="source"):
            build_loader(node)
