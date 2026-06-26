# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: KagglerAssistant
# FileName: data_provider.py
# Date: 2026/6/26 11:06
# -------------------------------------------------------------------------
import polars as pl

class DataProvider:
    def __init__(self) -> None:
        self._frames = dict[int: pl.DataFrame]

    def load_initial(self, path: str) -> None:
        self._frames[0] = pl.read_csv(path)

    def get(self, data_version: int) -> pl.DataFrame:
        if data_version not in self._frames:
            raise RuntimeError(f"数据版本 `{data_version}` 不存在")
        return self._frames[data_version]

