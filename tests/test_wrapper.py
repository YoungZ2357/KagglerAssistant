# -*- coding: utf-8 -*-
"""AgentSession（shared/wrapper.py）单测。

不触网、不跑真实图：用 csv_file fixture 让 DataProvider 真实读取小 CSV，
并把 ``build_graph`` patch 成返回受控的假图，从而单独验证：
- 种子 payload 仅首轮注入；
- ``.graph.stream(...)`` 的 (mode, data) 序列被正确翻译为 UI 事件；
- ``stream`` 便捷封装仅透出 token 文本。
"""
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from polars.testing import assert_frame_equal

from kaggler.modes.feature_engineering import compute
from kaggler.persistence.data_provider import DataProvider
from kaggler.persistence.version_ledger_store import VersionLedgerStore
from kaggler.shared import wrapper as wrapper_mod
from kaggler.shared.types import Mode
from kaggler.shared.wrapper import AgentSession


class FakeGraph:
    """假编译图：``stream`` 回放预置的 (mode, data) 序列，并记录入参。"""

    def __init__(self, script: list[tuple[str, Any]] | None = None) -> None:
        self.script = script or []
        self.stream_calls: list[dict] = []

    def stream(self, payload, config, stream_mode):
        self.stream_calls.append(
            {"payload": payload, "config": config, "stream_mode": stream_mode}
        )
        yield from self.script


@pytest.fixture
def make_session(mocker, csv_file):
    """构造 AgentSession，并把 build_graph patch 成返回给定 FakeGraph。"""

    def _make(script: list[tuple[str, Any]] | None = None) -> AgentSession:
        graph = FakeGraph(script)
        mocker.patch.object(wrapper_mod, "build_graph", return_value=graph)
        return AgentSession(csv_file)

    return _make


class TestInit:
    def test_builds_graph_and_config(self, mocker, csv_file):
        graph = FakeGraph()
        spy = mocker.patch.object(wrapper_mod, "build_graph", return_value=graph)
        session = AgentSession(csv_file)

        spy.assert_called_once()
        assert session._graph is graph
        assert session._csv_path == csv_file
        assert session._seeded is False
        # thread_id 应为非空的 uuid hex 串
        thread_id = session._config["configurable"]["thread_id"]
        assert isinstance(thread_id, str) and thread_id


class TestSeedPayload:
    def test_first_turn_injects_seed(self, make_session):
        session = make_session()
        payload = session._seed_payload("你好")

        assert payload["current_mode"] == Mode.EDA
        assert payload["file_path"] == session._csv_path
        assert payload["data_version"] == 0
        assert isinstance(payload["messages"][0], HumanMessage)
        assert payload["messages"][0].content == "你好"
        assert session._seeded is True

    def test_subsequent_turns_omit_seed(self, make_session):
        session = make_session()
        session._seed_payload("第一轮")
        payload = session._seed_payload("第二轮")

        assert "current_mode" not in payload
        assert "file_path" not in payload
        assert "data_version" not in payload
        assert payload["messages"][0].content == "第二轮"


def _seed_ledger(db, tid, csv):
    """用真实 DataProvider+sink 落一个 source + 一个 standardize 版本，返回 (v1, 期望帧)。"""
    from kaggler.ir import dumps_ir

    class _Sink:
        def record_version(self, version, *, ir=None, **kw):
            s = VersionLedgerStore(db)
            try:
                s.record(thread_id=tid, version=version,
                         ir=dumps_ir(ir) if ir is not None else None, **kw)
            finally:
                s.close()

    dp = DataProvider(sink=_Sink())
    root = dp.load_initial(str(csv))
    r = compute.exec_standardize(dp.get(root), ["score"])
    v1 = dp.add_version(r["op"], parent=root, tool="standardize", description="std",
                        ir=r["ir"])
    return v1, dp.get(v1)


class TestResumeRebuild:
    def test_resume_rebuilds_tree_and_marks_seeded(self, mocker, tmp_path, csv_file):
        db = tmp_path / "version_ledger.sqlite"
        tid = "tid-resume"
        v1, expected = _seed_ledger(db, tid, csv_file)

        # 假图：get_state 汇报当前 data_version = v1（恢复点）。
        class _FakeState:
            values = {"data_version": v1}

        graph = FakeGraph()
        graph.get_state = lambda config: _FakeState()
        mocker.patch.object(wrapper_mod, "build_graph", return_value=graph)

        session = AgentSession(csv_file, thread_id=tid, version_ledger_db=db)

        # 有账本 → 判定为恢复：seeded=True，重建出派生版本可读且逐值一致。
        assert session._seeded is True
        assert_frame_equal(session._data.get(v1), expected, check_dtypes=False)
        # 关键回归：恢复后首轮 payload 不得把 data_version/current_mode 打回 0。
        payload = session._seed_payload("继续")
        assert "data_version" not in payload
        assert "current_mode" not in payload

    def test_fresh_session_persists_v0_and_not_seeded(self, mocker, tmp_path, csv_file):
        db = tmp_path / "version_ledger.sqlite"
        graph = FakeGraph()
        mocker.patch.object(wrapper_mod, "build_graph", return_value=graph)

        session = AgentSession(csv_file, thread_id="fresh", version_ledger_db=db)

        assert session._seeded is False
        rows = _ledger_rows(db, "fresh")
        assert len(rows) == 1 and rows[0].kind == "source"

    def test_no_ledger_db_stays_pure_memory(self, mocker, csv_file):
        graph = FakeGraph()
        mocker.patch.object(wrapper_mod, "build_graph", return_value=graph)
        session = AgentSession(csv_file)  # 不传 version_ledger_db
        assert session._seeded is False
        assert session._data.get(0).height == 3  # v0 仍可用（load_initial）


def _ledger_rows(db, tid):
    s = VersionLedgerStore(db)
    try:
        return s.list_by_thread(tid)
    finally:
        s.close()


class TestStreamEvents:
    def test_new_node_emits_node_active_once(self, make_session):
        # 同一节点连续两个 messages 批次，只应报一次 node_active
        chunk = AIMessageChunk(content="")
        script = [
            ("messages", (chunk, {"langgraph_node": "react"})),
            ("messages", (chunk, {"langgraph_node": "react"})),
        ]
        session = make_session(script)
        events = list(session.stream_events("q"))

        active = [e for e in events if e["type"] == "node_active"]
        assert active == [{"type": "node_active", "node": "react"}]

    def test_react_chunk_with_content_emits_token(self, make_session):
        script = [
            ("messages", (AIMessageChunk(content="Hello"), {"langgraph_node": "react"})),
        ]
        session = make_session(script)
        events = list(session.stream_events("q"))

        assert {"type": "token", "content": "Hello"} in events

    def test_non_react_chunk_emits_no_token(self, make_session):
        script = [
            ("messages", (AIMessageChunk(content="x"), {"langgraph_node": "tools"})),
        ]
        session = make_session(script)
        events = list(session.stream_events("q"))

        assert not [e for e in events if e["type"] == "token"]

    def test_updates_with_tool_calls_emits_node_done(self, make_session):
        ai = AIMessage(
            content="",
            tool_calls=[
                {"name": "describe", "args": {"col": "age"}, "id": "1", "type": "tool_call"}
            ],
        )
        script = [("updates", {"react": {"messages": [ai]}})]
        session = make_session(script)
        events = list(session.stream_events("q"))

        assert events == [
            {
                "type": "node_done",
                "node": "react",
                "tool_calls": [{"name": "describe", "args": {"col": "age"}}],
            }
        ]

    def test_updates_without_tool_calls_emits_empty_list(self, make_session):
        script = [("updates", {"summarize": {"messages": [AIMessage(content="done")]}})]
        session = make_session(script)
        events = list(session.stream_events("q"))

        assert events == [
            {"type": "node_done", "node": "summarize", "tool_calls": []}
        ]

    def test_updates_with_none_state_handled(self, make_session):
        # state_update 可能为 None（节点无 messages 更新）——不应崩溃
        script = [("updates", {"finish": None})]
        session = make_session(script)
        events = list(session.stream_events("q"))

        assert events == [{"type": "node_done", "node": "finish", "tool_calls": []}]

    def test_updates_list_shape_does_not_crash(self, make_session):
        # ToolNode 一次执行多个工具且含返回 Command 的工具时，updates payload 是
        # list[dict] 而非 dict——历史上会崩 "'list' object has no attribute 'get'"。
        # 现应正常归一化处理，并从 list 中抽出 current_mode 发 mode_change。
        state_update = [
            {"current_mode": Mode.FEAT_ENG},
            {"messages": [ToolMessage("switched", tool_call_id="1")]},
            {"messages": [ToolMessage("plain", tool_call_id="2")]},
        ]
        script = [("updates", {"tools": state_update})]
        session = make_session(script)
        events = list(session.stream_events("q"))

        assert events == [
            {"type": "mode_change", "mode": str(Mode.FEAT_ENG)},
            {"type": "node_done", "node": "tools", "tool_calls": []},
        ]

    def test_updates_list_takes_last_current_mode(self, make_session):
        # list 内多次写 current_mode 时，mode_change 取最后一个（与 reducer 语义一致）
        state_update = [
            {"current_mode": Mode.FEAT_ENG},
            {"current_mode": Mode.EDA},
        ]
        script = [("updates", {"tools": state_update})]
        session = make_session(script)
        events = list(session.stream_events("q"))

        assert {"type": "mode_change", "mode": str(Mode.EDA)} in events
        assert {"type": "mode_change", "mode": str(Mode.FEAT_ENG)} not in events

    def test_updates_list_with_non_dict_entries_skipped(self, make_session):
        # list 中混入非 dict 元素也不应崩溃
        script = [("updates", {"tools": [None, {"current_mode": Mode.EDA}]})]
        session = make_session(script)
        events = list(session.stream_events("q"))

        assert events == [
            {"type": "mode_change", "mode": str(Mode.EDA)},
            {"type": "node_done", "node": "tools", "tool_calls": []},
        ]

    def test_updates_resets_current_node(self, make_session):
        # updates 批次后，再遇到同名节点的 messages 应重新报 node_active
        chunk = AIMessageChunk(content="")
        script = [
            ("messages", (chunk, {"langgraph_node": "react"})),
            ("updates", {"react": {"messages": []}}),
            ("messages", (chunk, {"langgraph_node": "react"})),
        ]
        session = make_session(script)
        events = list(session.stream_events("q"))

        active = [e for e in events if e["type"] == "node_active"]
        assert len(active) == 2

    def test_stream_forwards_payload_to_graph(self, make_session):
        session = make_session([])
        list(session.stream_events("提问"))

        call = session._graph.stream_calls[0]
        assert call["payload"]["messages"][0].content == "提问"
        assert call["config"] is session._config
        assert call["stream_mode"] == ["updates", "messages"]


class TestStream:
    def test_yields_only_token_text(self, make_session):
        script = [
            ("messages", (AIMessageChunk(content="a"), {"langgraph_node": "react"})),
            ("updates", {"react": {"messages": [AIMessage(content="x")]}}),
            ("messages", (AIMessageChunk(content="b"), {"langgraph_node": "react"})),
        ]
        session = make_session(script)

        assert list(session.stream("q")) == ["a", "b"]
