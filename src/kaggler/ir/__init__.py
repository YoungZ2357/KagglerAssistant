"""数据版本的单一中间表示(IR):schema + interpreter + code generator。

IR 是数据版本「这一步做什么」的唯一真相(SSOT):
- 运行时/恢复:``build_op``(或 ``op_from``)-> Op 闭包 —— 同一条路径;
- 持久化:``dumps_ir`` / ``loads_ir`` 落账本 TEXT 列;
- 代码导出:``emit_code`` / ``emit_source_expr`` -> 面向用户的 Polars 脚本片段。
"""

from kaggler.ir.schema import (
    IR_SCHEMA_VERSION,
    KINDS,
    IRNode,
    IRSpec,
    dumps_ir,
    loads_ir,
)
from kaggler.ir.interpret import Loader, Op, build_loader, build_op, op_from
from kaggler.ir.emit import code_from, emit_code, emit_source_expr

__all__ = [
    "IR_SCHEMA_VERSION",
    "KINDS",
    "IRNode",
    "IRSpec",
    "dumps_ir",
    "loads_ir",
    "Op",
    "Loader",
    "build_op",
    "op_from",
    "build_loader",
    "emit_code",
    "code_from",
    "emit_source_expr",
]
