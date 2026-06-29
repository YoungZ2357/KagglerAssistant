from kaggler.graph.types import Node
from kaggler.shared.types import Mode


class TestMode:
    def test_eda_value(self):
        assert Mode.EDA == "eda"
        assert Mode.EDA.value == "eda"

    def test_is_str_enum(self):
        # 继承 str → 成员本身即字符串，可直接拼接/比较
        assert isinstance(Mode.EDA, str)


class TestNode:
    def test_member_values(self):
        assert Node.REACT == "react"
        assert Node.TOOLS == "tools"
        assert Node.SUMMARIZE == "summarize"
        assert Node.FINISH == "finish"

    def test_all_members(self):
        assert {n.value for n in Node} == {"react", "tools", "summarize", "finish"}

    def test_is_str_enum(self):
        assert isinstance(Node.REACT, str)
