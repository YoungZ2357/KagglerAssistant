import polars as pl
import pytest

from kaggler.workspace.data_provider import DataProvider


class TestDataProvider:
    def test_load_initial_and_get(self, csv_file):
        data = DataProvider()
        data.load_initial(csv_file)
        df = data.get(0)
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (3, 3)
        assert df.columns == ["id", "name", "score"]

    def test_get_unknown_version_raises(self):
        data = DataProvider()
        with pytest.raises(RuntimeError, match="不存在"):
            data.get(0)

    def test_get_unknown_version_after_load_raises(self, csv_file):
        data = DataProvider()
        data.load_initial(csv_file)
        with pytest.raises(RuntimeError):
            data.get(99)

    def test_empty_provider_has_no_frames(self):
        data = DataProvider()
        assert data._frames == {}

    def test_add_version_after_load_returns_1(self, csv_file):
        data = DataProvider()
        data.load_initial(csv_file)
        df_modified = pl.DataFrame({"id": [10], "name": ["x"], "score": [9.9]})
        new_ver = data.add_version(df_modified)
        assert new_ver == 1

    def test_add_version_stores_and_retrieves(self, csv_file):
        data = DataProvider()
        data.load_initial(csv_file)
        df_modified = pl.DataFrame({"a": [1]})
        new_ver = data.add_version(df_modified)
        assert data.get(new_ver).equals(df_modified)
        assert data.get(0).shape == (3, 3)

    def test_add_version_empty_provider(self):
        data = DataProvider()
        df = pl.DataFrame({"x": [1]})
        new_ver = data.add_version(df)
        assert new_ver == 0
        assert data.get(0).equals(df)

    def test_add_version_increments_sequentially(self, csv_file):
        data = DataProvider()
        data.load_initial(csv_file)
        v1 = data.add_version(pl.DataFrame({"x": [1]}))
        v2 = data.add_version(pl.DataFrame({"x": [2]}))
        v3 = data.add_version(pl.DataFrame({"x": [3]}))
        assert v1 == 1
        assert v2 == 2
        assert v3 == 3
