from dataclasses import asdict, dataclass

import polars as pl


@dataclass
class VersionInfo:
    parent: int | None
    tool: str | None
    description: str


class DataProvider:
    def __init__(self) -> None:
        self._frames: dict[int, pl.DataFrame] = {}
        self._version_info: dict[int, VersionInfo] = {}

    def load_initial(self, path: str) -> None:
        self._frames[0] = pl.read_csv(path)
        self._version_info[0] = VersionInfo(parent=None, tool=None, description="原始数据集")

    def get(self, data_version: int) -> pl.DataFrame:
        if data_version not in self._frames:
            raise RuntimeError(f"数据版本 `{data_version}` 不存在")
        return self._frames[data_version]

    def add_version(
        self,
        df: pl.DataFrame,
        *,
        parent: int | None = None,
        tool: str | None = None,
        description: str = "",
    ) -> int:
        new_version = max(self._frames.keys(), default=-1) + 1
        self._frames[new_version] = df
        self._version_info[new_version] = VersionInfo(parent=parent, tool=tool, description=description)
        return new_version

    def get_version_info(self, version: int) -> VersionInfo:
        if version not in self._version_info:
            raise RuntimeError(f"数据版本 `{version}` 不存在")
        return self._version_info[version]

    def list_versions(self) -> list[dict]:
        """按版本号升序返回全部版本的谱系信息，供浏览类工具使用。"""
        return [
            {"version": v, **asdict(self._version_info[v])}
            for v in sorted(self._frames.keys())
        ]

    # def persist_version(self, version: int, name: str) -> None:
    # 需要工作区系统才能实现
    #     pass
