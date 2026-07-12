"""从持久化账本重建 DataProvider 版本树（codegen 的逆过程）。

恢复对话时，把 VersionLedgerStore 里每个版本的 Polars 代码片段编译回 op / loader，
用原版本号 restore 进一个新的 DataProvider，从而复原整棵版本树（含 fork）。

安全说明：这里 ``exec`` / ``eval`` 的字符串来自本机 ``.kaggler`` 内、由本项目 codegen
（feature_engineering/codegen.py + DataProvider.load_initial）自产的代码，**非用户输入**；
执行命名空间被限定为仅含 ``pl``（与 ``lf``）。单用户本地应用的信任模型下可接受。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from kaggler.persistence.data_provider import DataProvider, Loader, Op

if TYPE_CHECKING:
    from kaggler.persistence.version_ledger_store import VersionRecord


def compile_op(code: str) -> Op:
    """把「重新赋值 ``lf`` 的语句」片段编译成 (LazyFrame)->LazyFrame。

    注释-only 片段（如「# (空值处理:无实际变换)」）exec 后 ``lf`` 不变，原样返回。
    """
    def _op(lf: pl.LazyFrame) -> pl.LazyFrame:
        ns: dict = {"pl": pl, "lf": lf}
        exec(code, ns)
        return ns["lf"]

    return _op


def compile_loader(code: str) -> Loader:
    """把 source 读取表达式（如 ``pl.read_csv('train.csv')``）编译成无参 loader。"""
    def _loader() -> pl.DataFrame:
        return eval(code, {"pl": pl})

    return _loader


def rebuild_into(
    data: DataProvider,
    records: list[VersionRecord],
    *,
    csv_path: str,
) -> None:
    """按 version 升序把账本记录重放进 data，复原版本树。

    records 需已按 version 升序（VersionLedgerStore.list_by_thread 已保证）。
    csv_path 仅作 source 无 code 时的兜底读取路径（正常情况下 source 恒有 read 表达式）。

    Raises:
        ValueError: 派生版本的 code 为 None（无法重建，如未来的 eager_op 桥 / 无种子随机）。
    """
    for rec in records:
        if rec.kind == "source":
            loader = compile_loader(rec.code) if rec.code else (lambda: pl.read_csv(csv_path))
            data.restore_source(
                rec.version,
                description=rec.description,
                tool=rec.tool,
                code=rec.code,
                loader=loader,
            )
        else:
            if rec.code is None:
                raise ValueError(
                    f"版本 `{rec.version}`（工具 {rec.tool}）无代码片段，无法从账本重建。"
                )
            data.restore_derived(
                rec.version,
                parent=rec.parent,
                tool=rec.tool,
                description=rec.description,
                reproducible=rec.reproducible,
                op=compile_op(rec.code),
                code=rec.code,
            )
