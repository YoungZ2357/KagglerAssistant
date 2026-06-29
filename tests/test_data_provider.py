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
