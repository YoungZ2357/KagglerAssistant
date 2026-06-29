from kaggler.graph.state import CommonState, _add_turns
from kaggler.shared.types import Mode


class TestAddTurns:
    def test_default_increment(self):
        assert _add_turns(0) == 1
        assert _add_turns(4) == 5

    def test_explicit_increment(self):
        assert _add_turns(2, 3) == 5

    def test_accumulates(self):
        turn = 0
        for _ in range(3):
            turn = _add_turns(turn)
        assert turn == 3


class TestCommonState:
    def test_dict_literal_construction(self):
        # CommonState 本质是 TypedDict，运行时即普通 dict
        state: CommonState = {
            "messages": [],
            "current_mode": Mode.EDA,
            "file_path": "data.csv",
            "explored_schema": "",
            "turn": 0,
            "summary": "",
            "data_version": 0,
        }
        assert state["current_mode"] == Mode.EDA
        assert state["data_version"] == 0
