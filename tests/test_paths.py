"""路径 SSOT 模块 (kaggler.shared.paths) 的行为契约测试。

只测「集成 + 契约」而非实现细节：覆盖入口即时生效、默认 root 位置、
checkpoint_db 跟随 root、~ 展开、ensure_layout 真建目录、import 零副作用。
对应任务简报的验收清单 §9（路径 SSOT 块）。
"""

from pathlib import Path

from kaggler.shared import paths

# root 覆盖用的环境变量名（模块公开契约）。
_ENV = "KAGGLER_HOME"


class TestRoot:
    def test_override_takes_effect_immediately_and_is_not_cached(
        self, monkeypatch, tmp_path
    ) -> None:
        # 首次覆盖立即生效
        first = tmp_path / "a"
        monkeypatch.setenv(_ENV, str(first))
        assert paths.root() == first.resolve()

        # 同一进程内再次改 env，必须即时反映（守住「不缓存」不变式）
        second = tmp_path / "b"
        monkeypatch.setenv(_ENV, str(second))
        assert paths.root() == second.resolve()

    def test_default_is_project_relative_kaggler(self, monkeypatch) -> None:
        monkeypatch.delenv(_ENV, raising=False)
        expected = Path(paths.__file__).resolve().parents[3] / ".kaggler"
        # 显式钉住 parents[3]：下标错了此断言立即失败
        assert paths.root() == expected
        assert paths.root().name == ".kaggler"

    def test_expands_tilde(self, monkeypatch) -> None:
        monkeypatch.setenv(_ENV, "~/foo")
        resolved = paths.root()
        assert resolved.is_absolute()
        assert "~" not in str(resolved)


class TestCheckpointDb:
    def test_follows_root(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv(_ENV, str(tmp_path))
        assert paths.checkpoint_db() == tmp_path.resolve() / "checkpoints.sqlite"


class TestEnsureLayout:
    def test_creates_root_dir_and_returns_it(self, monkeypatch, tmp_path) -> None:
        target = tmp_path / "sub"
        monkeypatch.setenv(_ENV, str(target))
        assert not target.exists()

        returned = paths.ensure_layout()

        assert returned == target.resolve()
        assert target.is_dir()


class TestNoImportSideEffects:
    def test_root_query_alone_creates_nothing(self, monkeypatch, tmp_path) -> None:
        # 仅查询 root（不调用 ensure_layout）不得产生任何落盘
        target = tmp_path / "fresh"
        monkeypatch.setenv(_ENV, str(target))
        _ = paths.root()
        _ = paths.checkpoint_db()
        assert not target.exists()
