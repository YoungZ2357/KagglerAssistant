import polars as pl
import pytest

from kaggler.persistence.data_provider import DataProvider


class TestDataProvider:
    def test_load_initial_and_get(self, csv_file):
        data = DataProvider()
        v0 = data.load_initial(csv_file)
        assert v0 == 0
        df = data.get(0)
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (3, 3)
        assert df.columns == ["id", "name", "score"]

    def test_head_and_root_after_load(self, csv_file):
        data = DataProvider()
        data.load_initial(csv_file)
        assert data.head == 0
        assert data.root == 0

    def test_get_unknown_version_raises(self):
        data = DataProvider()
        with pytest.raises(RuntimeError, match="不存在"):
            data.get(0)

    def test_get_unknown_version_after_load_raises(self, csv_file):
        data = DataProvider()
        data.load_initial(csv_file)
        with pytest.raises(RuntimeError):
            data.get(99)

    def test_empty_provider_list_versions_empty(self):
        data = DataProvider()
        assert data.list_versions() == []

    def test_add_source_on_empty_creates_v0(self):
        data = DataProvider()
        df = pl.DataFrame({"x": [1]})
        v0 = data.add_source(lambda: df, description="test")
        assert v0 == 0
        assert data.get(0).equals(df)

    def test_add_version_after_load_returns_1(self, csv_file):
        data = DataProvider()
        data.load_initial(csv_file)
        new_ver = data.add_version(
            DataProvider.eager_op(lambda _: pl.DataFrame({"id": [10], "name": ["x"], "score": [9.9]})),
            parent=0,
        )
        assert new_ver == 1

    def test_add_version_stores_and_retrieves(self, csv_file):
        data = DataProvider()
        data.load_initial(csv_file)
        new_df = pl.DataFrame({"a": [1]})
        new_ver = data.add_version(
            DataProvider.eager_op(lambda _: new_df),
            parent=0,
        )
        assert data.get(new_ver).equals(new_df)
        assert data.get(0).shape == (3, 3)

    def test_add_version_increments_sequentially(self, csv_file):
        data = DataProvider()
        data.load_initial(csv_file)
        v1 = data.add_version(
            DataProvider.eager_op(lambda _: pl.DataFrame({"x": [1]})),
            parent=0,
        )
        v2 = data.add_version(
            DataProvider.eager_op(lambda _: pl.DataFrame({"x": [2]})),
            parent=1,
        )
        v3 = data.add_version(
            DataProvider.eager_op(lambda _: pl.DataFrame({"x": [3]})),
            parent=2,
        )
        assert v1 == 1
        assert v2 == 2
        assert v3 == 3

    def test_set_head_and_get(self, csv_file):
        data = DataProvider()
        data.load_initial(csv_file)
        _ = data.add_version(
            DataProvider.eager_op(lambda _: pl.DataFrame({"x": [1]})),
            parent=0,
        )
        data.set_head(0)
        assert data.head == 0
        assert data.get(0).shape == (3, 3)

    def test_pin_source_by_default(self, csv_file):
        data = DataProvider(pin_root=True)
        data.load_initial(csv_file)
        versions = data.list_versions()
        root = [v for v in versions if v["version"] == 0][0]
        assert root["pinned"] is True

    def test_list_versions_marks_head(self, csv_file):
        data = DataProvider()
        data.load_initial(csv_file)
        data.add_version(
            DataProvider.eager_op(lambda _: pl.DataFrame({"x": [1]})),
            parent=0,
        )
        versions = data.list_versions()
        assert versions[0]["is_head"] is False
        assert versions[1]["is_head"] is True


class TestVersionLineage:
    def test_load_initial_root_version_info(self, csv_file):
        data = DataProvider()
        data.load_initial(csv_file)
        info = data.get_version_info(0)
        assert info.parent is None
        assert info.tool is None
        assert info.description == "原始数据集"

    def test_add_version_stores_lineage(self, csv_file):
        data = DataProvider()
        data.load_initial(csv_file)
        new_ver = data.add_version(
            DataProvider.eager_op(lambda _: pl.DataFrame({"x": [1]})),
            parent=0,
            tool="drop_columns",
            description="删除列: ['y']",
        )
        info = data.get_version_info(new_ver)
        assert info.parent == 0
        assert info.tool == "drop_columns"
        assert info.description == "删除列: ['y']"

    def test_get_version_info_unknown_version_raises(self):
        data = DataProvider()
        with pytest.raises(RuntimeError, match="不存在"):
            data.get_version_info(0)

    def test_list_versions_returns_all_in_order(self, csv_file):
        data = DataProvider()
        data.load_initial(csv_file)
        data.add_version(
            DataProvider.eager_op(lambda _: pl.DataFrame({"x": [1]})),
            parent=0, tool="drop_columns", description="删除列: ['y']",
        )
        data.add_version(
            DataProvider.eager_op(lambda _: pl.DataFrame({"x": [2]})),
            parent=1, tool="standardize_columns", description="标准化列: ['x']",
        )

        versions = data.list_versions()
        assert [v["version"] for v in versions] == [0, 1, 2]
        assert versions[0]["parent"] is None
        assert versions[0]["tool"] is None
        assert versions[0]["description"] == "原始数据集"
        assert versions[1]["parent"] == 0
        assert versions[1]["tool"] == "drop_columns"
        assert versions[2]["parent"] == 1
        assert versions[2]["tool"] == "standardize_columns"

    def test_lineage_and_list_versions_agree(self, csv_file):
        data = DataProvider()
        data.load_initial(csv_file)
        data.add_version(
            DataProvider.eager_op(lambda _: pl.DataFrame({"x": [1]})),
            parent=0,
        )
        data.add_version(
            DataProvider.eager_op(lambda _: pl.DataFrame({"x": [2]})),
            parent=1, tool="drop_columns", description="d",
        )
        assert {v["version"] for v in data.list_versions()} == {0, 1, 2}
