# -*- coding: utf-8 -*-
"""跨进程验证的子进程侧脚本(由 test_ir_persistence.py 经 subprocess 调用,非测试模块)。

子命令:
- ``tree <csv> <ledger_db> <tid> <outdir>``:构建含 fork 的版本树,经 sink 落账本
  (含 IR),并把每个版本的期望帧写 parquet;末行输出版本号 JSON。
  父进程 rebuild_into 重建后逐帧比对——这是「运行时闭包(子进程) vs
  IR→interpreter(父进程)」的跨进程差分。
- ``state <ckpt_db> <tid>``:用 file-backed SqliteSaver 跑一个最小 CommonState 图,
  写满全部通道后退出;父进程新开连接读回逐通道断言。
"""
import json
import sys
from pathlib import Path


def _tree(csv_path: str, ledger_db: str, tid: str, outdir: str) -> None:
    from kaggler.ir import dumps_ir
    from kaggler.modes.feature_engineering import compute
    from kaggler.persistence.data_provider import DataProvider
    from kaggler.persistence.version_ledger_store import VersionLedgerStore

    class _Sink:
        def record_version(self, version, *, ir=None, **kw):
            s = VersionLedgerStore(Path(ledger_db))
            try:
                s.record(thread_id=tid, version=version,
                         ir=dumps_ir(ir) if ir is not None else None, **kw)
            finally:
                s.close()

    dp = DataProvider(sink=_Sink())
    root = dp.load_initial(csv_path)

    def step(fn, parent, tool, *args):
        r = fn(dp.get(parent), *args)
        assert "error" not in r, (tool, r)
        return dp.add_version(
            r["op"], parent=parent, tool=tool, description=tool, ir=r["ir"],
        )

    # 覆盖:标量(standardize)/分组冻结(avg+mode)/映射(label)/矩阵(pca)/无参(drop/filter)/fork
    v1 = step(compute.exec_standardize, root, "standardize", ["income"])
    v2 = step(compute.exec_empty, v1, "fill", [
        {"column": "bonus", "action": "avg", "group_by": "city"},
        {"column": "note", "action": "mode", "group_by": "city"},
    ])
    v3 = step(compute.exec_encode, v2, "encode", [{"column": "city", "action": "label"}])
    v4 = step(compute.exec_dim_reduct, v3, "pca", "pca", 2)
    dp.set_head(v2)  # fork:回到 v2 派生另一分支
    v5 = step(compute.exec_drop_columns, v2, "drop", ["target"])
    v6 = step(
        compute.exec_filter_rows, v5, "filter",
        [{"logic": "and", "conditions": [{"column": "age", "op": "gt", "value": 25}]}],
        "and", "keep",
    )

    versions = [root, v1, v2, v3, v4, v5, v6]
    out = Path(outdir)
    for v in versions:
        dp.get(v).write_parquet(out / f"v{v}.parquet")
    print(json.dumps({"versions": versions}))


def _state(ckpt_db: str, tid: str) -> None:
    from langchain_core.messages import HumanMessage
    from langgraph.graph import END, START, StateGraph

    from kaggler.graph.assembly import make_sqlite_saver
    from kaggler.graph.state import CommonState
    from kaggler.shared.types import Mode

    saver = make_sqlite_saver(Path(ckpt_db))
    g = StateGraph(CommonState)
    g.add_node("noop", lambda state: {})
    g.add_edge(START, "noop")
    g.add_edge("noop", END)
    graph = g.compile(checkpointer=saver)

    graph.invoke(
        {
            "messages": [HumanMessage(content="跨进程持久化验证")],
            "current_mode": Mode.FEAT_ENG,
            "file_path": "train.csv",
            "explored_schema": "age:Int64",
            "turn": 3,
            "memory": {"goal": "验证", "findings": ["a", "b"]},
            "data_version": 5,
            "todos": [{"content": "todo-1", "status": "open"}],
            "plans": [{"title": "p", "content": "c", "status": "draft"}],
            "context_usage": {"total": 123},
        },
        {"configurable": {"thread_id": tid}},
    )
    saver.conn.close()
    print("ok")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tree":
        _tree(*sys.argv[2:6])
    elif cmd == "state":
        _state(*sys.argv[2:4])
    else:
        raise SystemExit(f"未知子命令: {cmd}")
