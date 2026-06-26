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

