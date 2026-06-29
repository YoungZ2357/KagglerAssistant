from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from kaggler.app.cli import _last_ai_text


class TestLastAiText:
    def test_returns_last_ai_content(self):
        state = {
            "messages": [
                HumanMessage(content="q"),
                AIMessage(content="第一条回复"),
                HumanMessage(content="q2"),
                AIMessage(content="最后回复"),
            ]
        }
        assert _last_ai_text(state) == "最后回复"

    def test_no_ai_message_returns_empty(self):
        state = {"messages": [HumanMessage(content="q"), ToolMessage(content="r", tool_call_id="t")]}
        assert _last_ai_text(state) == ""

    def test_empty_messages_returns_empty(self):
        assert _last_ai_text({"messages": []}) == ""

    def test_skips_empty_content_ai(self):
        # 末条 AIMessage 内容为空（仅 tool_calls 场景），应回退到上一条非空 AIMessage
        state = {
            "messages": [
                AIMessage(content="有内容"),
                AIMessage(content="", tool_calls=[{"name": "f", "args": {}, "id": "tc"}]),
            ]
        }
        assert _last_ai_text(state) == "有内容"

    def test_non_str_content_stringified(self):
        state = {"messages": [AIMessage(content=[{"type": "text", "text": "块"}])]}
        result = _last_ai_text(state)
        assert isinstance(result, str)
        assert "块" in result
