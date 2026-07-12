from kaggler.graph.memory import (
    AgentMemory,
    merge_memory,
    parse_memory,
    render_memory,
)


class TestAgentMemoryRoundTrip:
    def test_to_from_dict_roundtrip(self):
        mem = AgentMemory(user_goal="g", key_findings=["a", "b"], progress="p")
        assert AgentMemory.from_dict(mem.to_dict()) == mem

    def test_from_dict_none_is_empty(self):
        assert AgentMemory.from_dict(None).is_empty()

    def test_from_dict_tolerates_string_findings(self):
        mem = AgentMemory.from_dict({"key_findings": "单条"})
        assert mem.key_findings == ["单条"]

    def test_from_dict_drops_blank_findings(self):
        mem = AgentMemory.from_dict({"key_findings": ["a", "", "  "]})
        assert mem.key_findings == ["a"]

    def test_is_empty_true_for_defaults(self):
        assert AgentMemory().is_empty()

    def test_is_empty_false_with_any_field(self):
        assert not AgentMemory(progress="x").is_empty()


class TestParseMemory:
    def test_parses_chinese_keys(self):
        mem = parse_memory('{"用户目标": "预测", "关键发现": ["f"], "进展": "p"}')
        assert mem == AgentMemory(user_goal="预测", key_findings=["f"], progress="p")

    def test_parses_english_keys(self):
        mem = parse_memory('{"user_goal": "predict", "key_findings": [], "progress": "p"}')
        assert mem.user_goal == "predict"

    def test_strips_code_fence(self):
        text = '```json\n{"用户目标": "g", "关键发现": [], "进展": ""}\n```'
        assert parse_memory(text).user_goal == "g"

    def test_returns_none_on_garbage(self):
        assert parse_memory("这不是JSON") is None

    def test_returns_none_on_empty(self):
        assert parse_memory("") is None

    def test_returns_none_on_non_object_json(self):
        assert parse_memory("[1, 2, 3]") is None


class TestMergeMemory:
    def test_new_goal_overrides(self):
        prev = AgentMemory(user_goal="old")
        upd = AgentMemory(user_goal="new")
        assert merge_memory(prev, upd).user_goal == "new"

    def test_empty_new_goal_keeps_old(self):
        prev = AgentMemory(user_goal="old")
        upd = AgentMemory(user_goal="")
        assert merge_memory(prev, upd).user_goal == "old"

    def test_findings_accumulate_and_dedup(self):
        prev = AgentMemory(key_findings=["a", "b"])
        upd = AgentMemory(key_findings=["b", "c"])
        assert merge_memory(prev, upd).key_findings == ["a", "b", "c"]

    def test_findings_capped_to_latest(self):
        prev = AgentMemory(key_findings=["a", "b", "c"])
        upd = AgentMemory(key_findings=["d"])
        merged = merge_memory(prev, upd, key_findings_cap=2)
        assert merged.key_findings == ["c", "d"]  # 截尾保留最新

    def test_progress_replaced_by_new(self):
        prev = AgentMemory(progress="old")
        upd = AgentMemory(progress="new")
        assert merge_memory(prev, upd).progress == "new"

    def test_empty_new_progress_keeps_old(self):
        prev = AgentMemory(progress="old")
        upd = AgentMemory(progress="")
        assert merge_memory(prev, upd).progress == "old"


class TestRenderMemory:
    def test_renders_all_sections(self):
        mem = AgentMemory(user_goal="g", key_findings=["a", "b"], progress="p")
        out = render_memory(mem)
        assert "用户目标：g" in out
        assert "- a" in out and "- b" in out
        assert "进展：p" in out

    def test_skips_empty_sections(self):
        out = render_memory(AgentMemory(user_goal="only"))
        assert "用户目标：only" in out
        assert "关键发现" not in out
        assert "进展" not in out
