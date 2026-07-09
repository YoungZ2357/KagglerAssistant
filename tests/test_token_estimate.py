"""token_estimate 纯函数与校准逻辑测试。"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from kaggler.modes.eda.tools import make_tools as make_eda_tools
from kaggler.persistence.data_provider import DataProvider
from kaggler.shared.token_estimate import (
    CJK_TOKEN_WEIGHT,
    CONTEXT_LIMIT,
    CONTEXT_RECOMMENDED,
    FACTOR_MAX,
    MIN_CALIB_SAMPLE,
    OTHER_TOKEN_WEIGHT,
    ContextBreakdown,
    bar_fill,
    build_breakdown,
    estimate_messages,
    estimate_text,
    estimate_tool_tokens,
    next_calibration_factor,
    utilization,
)


class TestEstimateText:
    def test_empty_is_zero(self):
        assert estimate_text("") == 0

    def test_pure_english(self):
        s = "hello world"
        assert estimate_text(s) == round(len(s) * OTHER_TOKEN_WEIGHT)

    def test_pure_chinese(self):
        s = "你好世界"
        assert estimate_text(s) == round(len(s) * CJK_TOKEN_WEIGHT)

    def test_chinese_weighs_more_than_ascii(self):
        assert estimate_text("字" * 10) > estimate_text("a" * 10)

    def test_mixed_between_bounds(self):
        n = 20
        mixed = "字a" * (n // 2)  # 一半中文一半英文
        assert estimate_text("a" * n) <= estimate_text(mixed) <= estimate_text("字" * n)

    def test_monotonic(self):
        assert estimate_text("abcabc") >= estimate_text("abc")


class TestEstimateToolTokens:
    def test_real_tools_positive(self, df_mixed):
        dp = DataProvider()
        dp.add_source(lambda: df_mixed, description="t")
        tools = make_eda_tools(dp)
        assert estimate_tool_tokens(tools) > 0

    def test_increases_with_more_tools(self, df_mixed):
        dp = DataProvider()
        dp.add_source(lambda: df_mixed, description="t")
        tools = make_eda_tools(dp)
        one = estimate_tool_tokens(tools[:1])
        several = estimate_tool_tokens(tools)
        assert several > one

    def test_empty_tools_zero(self):
        assert estimate_tool_tokens([]) == 0


class TestEstimateMessages:
    def test_buckets_by_type(self):
        msgs = [
            HumanMessage(content="用户问题"),
            AIMessage(content="助手回复"),
            ToolMessage(content='{"result": 1}', tool_call_id="c1"),
        ]
        buckets = estimate_messages(msgs)
        assert buckets["user"] > 0
        assert buckets["assistant"] > 0
        assert buckets["tool_results"] > 0

    def test_tool_call_args_counted_in_assistant(self):
        plain = AIMessage(content="x")
        with_calls = AIMessage(
            content="x",
            tool_calls=[{"name": "foo", "args": {"col": "很长的中文参数值" * 5}, "id": "c1"}],
        )
        b_plain = estimate_messages([plain])["assistant"]
        b_calls = estimate_messages([with_calls])["assistant"]
        assert b_calls > b_plain


class TestBuildBreakdown:
    def _tools(self, df_mixed):
        dp = DataProvider()
        dp.add_source(lambda: df_mixed, description="t")
        return make_eda_tools(dp)

    def test_categories_sum_to_estimated_total(self, df_mixed):
        bd = build_breakdown(
            system_prompt_text="系统提示词内容",
            summary_text="摘要内容",
            tools=self._tools(df_mixed),
            messages=[HumanMessage(content="hi"), AIMessage(content="yo")],
        )
        d = bd.to_dict()
        assert sum(d["categories"].values()) == d["estimated_total"]

    def test_actual_total_takes_precedence(self, df_mixed):
        bd = build_breakdown("sys", "", self._tools(df_mixed), [HumanMessage(content="hi")])
        bd.actual_total = 99999
        d = bd.to_dict()
        assert d["total"] == 99999
        assert d["actual_total"] == 99999

    def test_calibration_factor_scales_categories(self, df_mixed):
        bd = build_breakdown("sys 系统", "摘要", self._tools(df_mixed), [HumanMessage(content="hi")])
        raw_total = bd.estimated_total_raw
        bd.calibration_factor = 2.0
        assert bd.estimated_total == round(raw_total * 2.0)

    def test_budget_constants_in_payload(self, df_mixed):
        d = build_breakdown("s", "", self._tools(df_mixed), []).to_dict()
        assert d["recommended"] == CONTEXT_RECOMMENDED
        assert d["limit"] == CONTEXT_LIMIT


class TestNextCalibrationFactor:
    def test_no_actual_keeps_prev(self):
        assert next_calibration_factor(1.3, 1000, None) == 1.3

    def test_small_sample_keeps_prev(self):
        assert next_calibration_factor(1.0, MIN_CALIB_SAMPLE - 1, 500) == 1.0

    def test_ema_update(self):
        # prev=1.0, ratio=2.0 → 0.3*2 + 0.7*1 = 1.3
        assert next_calibration_factor(1.0, 1000, 2000) == 1.3

    def test_clamped_to_max(self):
        assert next_calibration_factor(1.0, 100, 100_000) == FACTOR_MAX


class TestDisplayHelpers:
    def test_utilization(self):
        assert utilization(0) == 0.0
        assert utilization(CONTEXT_RECOMMENDED) == 1.0
        assert utilization(CONTEXT_RECOMMENDED * 2) == 2.0

    def test_bar_fill_bounds(self):
        assert bar_fill(0.0, 16) == 0
        assert bar_fill(0.5, 16) == 8
        assert bar_fill(1.0, 16) == 16
        assert bar_fill(2.0, 16) == 16  # 超预算夹取到满
        assert bar_fill(0.5, 0) == 0


class TestContextBreakdownShape:
    def test_to_dict_keys(self):
        bd = ContextBreakdown(system=1, summary=2, tools=3, user=4, assistant=5, tool_results=6)
        d = bd.to_dict()
        assert set(d["categories"]) == {
            "system", "summary", "tools", "user", "assistant", "tool_results",
        }
        assert d["estimated_total"] == 21
        assert d["total"] == 21  # 无实测
        assert d["calibration_factor"] == 1.0
