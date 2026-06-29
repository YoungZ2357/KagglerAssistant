import pytest

from kaggler.shared import config
from kaggler.shared.config import DeepSeekModel, GraphConfig, make_llm_raw


class TestDeepSeekModel:
    def test_values(self):
        assert DeepSeekModel.FLASH == "deepseek-v4-flash"
        assert DeepSeekModel.PRO == "deepseek-v4-pro"


class TestGraphConfig:
    def test_defaults(self):
        cfg = GraphConfig()
        assert cfg.summary_trigger_count == 20
        assert cfg.summary_keep_recent == 4

    def test_override(self):
        cfg = GraphConfig(summary_trigger_count=5, summary_keep_recent=1)
        assert cfg.summary_trigger_count == 5
        assert cfg.summary_keep_recent == 1


class TestMakeLLMRaw:
    def test_temperature_below_range_asserts(self):
        with pytest.raises(AssertionError):
            make_llm_raw(temperature=-0.1)

    def test_temperature_above_range_asserts(self):
        with pytest.raises(AssertionError):
            make_llm_raw(temperature=1.5)

    def test_passes_model_and_temperature(self, mocker):
        fake = mocker.patch.object(config, "ChatDeepSeek")
        make_llm_raw(DeepSeekModel.PRO, temperature=0.3)
        _, kwargs = fake.call_args
        assert kwargs["model"] == "deepseek-v4-pro"
        assert kwargs["temperature"] == 0.3

    def test_thinking_disabled_by_default(self, mocker):
        fake = mocker.patch.object(config, "ChatDeepSeek")
        make_llm_raw()
        _, kwargs = fake.call_args
        assert kwargs["extra_body"]["thinking"]["type"] == "disabled"

    def test_thinking_enabled_flag(self, mocker):
        fake = mocker.patch.object(config, "ChatDeepSeek")
        make_llm_raw(enable_thinking=True)
        _, kwargs = fake.call_args
        assert kwargs["extra_body"]["thinking"]["type"] == "enabled"

    def test_boundary_temperatures_allowed(self, mocker):
        mocker.patch.object(config, "ChatDeepSeek")
        make_llm_raw(temperature=0.0)
        make_llm_raw(temperature=1.0)  # 不应抛出
