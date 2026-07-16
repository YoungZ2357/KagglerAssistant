"""IR schema 序列化的孤立单测(阶段 1,不接线)。

重点:round-trip 保真(含 nan/inf 经真实 sqlite TEXT 列往返,§8 经验主义)、
payload 类型校验响亮报错、多父骨架(schema 允许、执行层显式未实现)。
"""

import json
import math
import sqlite3

import numpy as np
import pytest

from kaggler.ir import (
    IR_SCHEMA_VERSION,
    IRNode,
    build_op,
    dumps_ir,
    emit_code,
    loads_ir,
)


def _mk(kind="standardize", params=None, parents=None, version=1, seed=None):
    return IRNode(
        version=version,
        kind=kind,
        parents=[0] if parents is None else parents,
        params={"stats": []} if params is None else params,
        seed=seed,
    )


class TestRoundTrip:
    def test_basic_round_trip(self):
        node = _mk(params={"stats": [{"column": "a", "mean": 1.5, "std": 2.0}]})
        assert loads_ir(dumps_ir(node)) == node

    def test_nested_array_round_trip(self):
        params = {
            "method": "pca",
            "components": [
                {"bias": -0.25, "weights": [1.0, 0.0, -3.5]},
                {"bias": 0.0, "weights": [0.5, 2.25, 0.125]},
            ],
            "numeric_cols": ["a", "b", "c"],
            "out_cols": ["PC1", "PC2"],
            "final_cols": ["PC1", "PC2"],
        }
        node = _mk(kind="dim_reduct", params=params)
        assert loads_ir(dumps_ir(node)).params == params

    def test_nan_inf_round_trip_through_sqlite_text(self, tmp_path):
        """nan/inf 经真实 sqlite TEXT 列往返——以实测钉死 allow_nan 策略。"""
        params = {
            "indicators": [],
            "fills": [{
                "column": "x", "type": "mean", "group_col": "g",
                "group_breaks": None,
                "global_stat": float("nan"),
                "group_map": [["k1", float("inf")], ["k2", float("-inf")]],
            }],
            "delete_columns": [],
        }
        node = _mk(kind="fill_missing", params=params)
        db = tmp_path / "t.sqlite"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE t (ir TEXT)")
        conn.execute("INSERT INTO t VALUES (?)", (dumps_ir(node),))
        conn.commit()
        conn.close()

        conn = sqlite3.connect(str(db))
        raw = conn.execute("SELECT ir FROM t").fetchone()[0]
        conn.close()
        spec = loads_ir(raw).params["fills"][0]
        assert math.isnan(spec["global_stat"])
        assert spec["group_map"][0][1] == float("inf")
        assert spec["group_map"][1][1] == float("-inf")

    def test_negative_zero_round_trip(self):
        node = _mk(params={"stats": [{"column": "a", "mean": -0.0, "std": 1.0}]})
        got = loads_ir(dumps_ir(node)).params["stats"][0]["mean"]
        assert got == 0.0
        assert math.copysign(1.0, got) == -1.0

    def test_pair_list_keys_preserve_types(self):
        """数据键走平行对列表,int/float/bool/str 键类型经 JSON 不得漂移。"""
        mapping = [[1, 0], [2.5, 1], [True, 2], ["x", 3]]
        node = _mk(
            kind="encode",
            params={
                "specs": [{"column": "c", "type": "label", "mapping": mapping}],
                "drop_columns": [],
            },
        )
        got = loads_ir(dumps_ir(node)).params["specs"][0]["mapping"]
        assert [type(k) for k, _ in got] == [int, float, bool, str]
        assert got == mapping

    def test_seed_round_trip(self):
        node = _mk(seed=42)
        assert loads_ir(dumps_ir(node)).seed == 42


class TestValidation:
    def test_rejects_numpy_scalar(self):
        node = _mk(params={"stats": [{"column": "a", "mean": np.float64(1.5), "std": 1.0}]})
        with pytest.raises(ValueError, match="非 JSON 原生类型"):
            dumps_ir(node)

    def test_rejects_non_str_dict_key(self):
        node = _mk(params={"bad": {1: "x"}})
        with pytest.raises(ValueError, match="dict 键"):
            dumps_ir(node)

    def test_unknown_kind_raises_on_dumps(self):
        node = _mk(kind="join")
        with pytest.raises(ValueError, match="未知的 IR kind"):
            dumps_ir(node)

    def test_unknown_kind_raises_on_loads(self):
        s = json.dumps({
            "schema_version": IR_SCHEMA_VERSION, "version": 1, "kind": "join",
            "parents": [0], "params": {}, "seed": None,
        })
        with pytest.raises(ValueError, match="未知的 IR kind"):
            loads_ir(s)

    def test_future_schema_version_raises(self):
        s = json.dumps({
            "schema_version": IR_SCHEMA_VERSION + 1, "version": 1,
            "kind": "standardize", "parents": [0], "params": {}, "seed": None,
        })
        with pytest.raises(ValueError, match="schema_version"):
            loads_ir(s)

    def test_rejects_non_int_parents(self):
        node = _mk(parents=["0"])
        with pytest.raises(ValueError, match="parents"):
            dumps_ir(node)


class TestMultiParentSkeleton:
    """多父(join/concat)预留:schema 层可序列化往返,执行层显式未实现。"""

    def test_multi_parent_serializes(self):
        node = _mk(parents=[1, 2])
        assert loads_ir(dumps_ir(node)).parents == [1, 2]

    def test_build_op_rejects_multi_parent(self):
        with pytest.raises(NotImplementedError):
            build_op(_mk(parents=[1, 2]))

    def test_emit_code_rejects_multi_parent(self):
        with pytest.raises(NotImplementedError):
            emit_code(_mk(parents=[1, 2]))
