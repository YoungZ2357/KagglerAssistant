"""数据版本导出服务：把某个版本的 eager DataFrame 落盘为文件（CSV / parquet / …）。

设计要点:
- 格式扩展只需往 WRITERS 加一条——这是唯一的扩展点。CSV / parquet 均即刻可用
  （Polars 原生写 parquet,无需 pyarrow）。
- export_version 为纯写:只依赖 DataProvider.get + 文件系统,不碰任何存储层,便于单测。
- export_and_record 是双通道(工具 / /export 指令)共用的编排入口:落盘 + 登记导出目录。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from kaggler.persistence.data_provider import DataProvider
from kaggler.persistence.data_version_store import DataVersionStore

# 格式 -> 写出函数。加新格式(如 "json")只需在此加一行。
WRITERS: dict[str, Callable[[pl.DataFrame, Path], None]] = {
    "csv": lambda df, p: df.write_csv(p),
    "parquet": lambda df, p: df.write_parquet(p),
}

# 文件后缀 -> 格式名(用于按扩展名推断)。
_SUFFIX_TO_FORMAT: dict[str, str] = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
}

_DEFAULT_FORMAT = "csv"

# 工作区内导出产物的受控子目录名。工具通道强制落此目录;/export 指令默认落此、亦允许外部绝对路径。
EXPORT_SUBDIR = "exports"


@dataclass
class ExportResult:
    version: int
    path: str
    format: str
    rows: int
    cols: int


def resolve_format(path: Path, fmt: str | None) -> str:
    """确定导出格式:显式 fmt 优先 → 否则按后缀推断 → 否则默认 csv。

    结果不在 WRITERS 中(含显式传入的未知格式、未知后缀)则抛 ValueError。
    """
    if fmt is not None:
        chosen = fmt.lower()
    else:
        suffix = path.suffix.lower()
        chosen = _SUFFIX_TO_FORMAT.get(suffix, _DEFAULT_FORMAT if suffix == "" else suffix.lstrip("."))
    if chosen not in WRITERS:
        supported = "、".join(WRITERS)
        raise ValueError(f"不支持的导出格式 `{chosen}`,当前支持:{supported}")
    return chosen


def export_version(
    data: DataProvider,
    version: int,
    path: Path,
    fmt: str | None = None,
) -> ExportResult:
    """把指定版本的 DataFrame 写到 path。纯写:不登记导出目录。

    版本不存在时 data.get 抛 RuntimeError,直接冒泡;格式非法抛 ValueError。
    """
    df = data.get(version)  # 版本不存在 -> RuntimeError
    chosen = resolve_format(path, fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    WRITERS[chosen](df, path)
    return ExportResult(
        version=version,
        path=str(path),
        format=chosen,
        rows=df.height,
        cols=df.width,
    )


def export_and_record(
    data: DataProvider,
    version: int,
    target: Path,
    fmt: str | None = None,
    *,
    db_path: Path | None = None,
    thread_id: str | None = None,
    description: str = "",
) -> ExportResult:
    """双通道共用编排:落盘 + (可选)登记导出目录。

    db_path 为 None(如未设工作区的裸会话)时跳过登记,仍完成落盘。
    """
    result = export_version(data, version, target, fmt)
    if db_path is not None:
        store = DataVersionStore(db_path)
        try:
            store.record(
                version=version,
                file_path=result.path,
                format=result.format,
                description=description,
                rows=result.rows,
                cols=result.cols,
                thread_id=thread_id,
            )
        finally:
            store.close()
    return result
