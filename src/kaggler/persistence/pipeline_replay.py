"""从持久化账本重建 DataProvider 版本树。

恢复对话时，把 VersionLedgerStore 里每个版本的 IR(JSON)反序列化,经
**与运行时同一个 interpreter**(``kaggler.ir.build_op`` / ``build_loader``)
重建 op / loader,用原版本号 restore 进一个新的 DataProvider,从而复原
整棵版本树(含 fork)——单路径不变量:不存在独立于运行时的 restore 构建逻辑。

旧账本(IR 重构前创建、只有 code 字符串的记录)不再支持恢复,遇到时
响亮报错提示新建对话。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaggler.ir import build_loader, build_op, loads_ir
from kaggler.persistence.data_provider import DataProvider

if TYPE_CHECKING:
    from kaggler.persistence.version_ledger_store import VersionRecord


def _load_node(rec: VersionRecord):
    """把账本行的 IR JSON 解析为 IRNode,并与账本行交叉校验。

    Raises:
        ValueError: 记录无 IR(旧版本应用创建的账本,已不支持恢复),或
                    IR 内容与账本行不一致(version/parent 漂移)。
    """
    if rec.ir is None:
        raise ValueError(
            f"版本 `{rec.version}`(工具 {rec.tool})无 IR 记录:该对话由旧版本应用"
            "创建,无法恢复。请新建对话重新加载数据。"
        )
    node = loads_ir(rec.ir)
    if node.version != rec.version:
        raise ValueError(
            f"账本记录与 IR 不一致:账本 version={rec.version},IR version={node.version}"
        )
    expected_parents = [] if rec.parent is None else [rec.parent]
    if node.parents != expected_parents:
        raise ValueError(
            f"账本记录与 IR 不一致(version={rec.version}):"
            f"账本 parent={rec.parent},IR parents={node.parents}"
        )
    return node


def rebuild_into(
    data: DataProvider,
    records: list[VersionRecord],
) -> None:
    """按 version 升序把账本记录的 IR 重放进 data，复原版本树。

    records 需已按 version 升序（VersionLedgerStore.list_by_thread 已保证）。
    op/loader 经 ``build_op``/``build_loader`` 重建——与运行时同一 interpreter。

    Raises:
        ValueError: 任一版本无 IR（旧账本 / eager_op 桥），或 IR 与账本行不一致。
    """
    for rec in records:
        node = _load_node(rec)
        if rec.kind == "source":
            data.restore_source(
                rec.version,
                description=rec.description,
                tool=rec.tool,
                loader=build_loader(node),
                ir=node,
            )
        else:
            data.restore_derived(
                rec.version,
                parent=rec.parent,
                tool=rec.tool,
                description=rec.description,
                reproducible=rec.reproducible,
                op=build_op(node),
                ir=node,
            )
