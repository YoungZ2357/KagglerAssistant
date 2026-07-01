# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: KagglerAssistant
# FileName: commands.py
# -------------------------------------------------------------------------
"""TUI slash-command 框架：指令注册表 + 解析 + 内联补全器（纯逻辑，不碰 UI）。

设计要点：
- **可扩展**：新增指令只需往 ``COMMANDS`` 加一条 ``CommandSpec``；``/switch`` 的
  候选值直接由 ``Mode`` 枚举派生，加新模式无需改本文件。
- **补全走 Textual 内置 Suggester**：``SlashSuggester.get_suggestion`` 返回「以当前
  输入为前缀的完整字符串」，Textual 才会把尾部作为灰色幽灵文本渲染（→/End 接受）。
- 指令的**执行**在 app.py（要碰 session 与 UI），本文件只负责配置、解析与补全。
"""
from dataclasses import dataclass

from textual.suggester import Suggester

from kaggler.shared.types import Mode


@dataclass(frozen=True)
class CommandSpec:
    """一条 slash 指令的元数据。``arg_candidates`` 为补全用的参数候选值。"""

    name: str
    description: str
    arg_candidates: tuple[str, ...] = ()


# 指令注册表。/switch 的候选模式由 Mode 枚举动态生成。
COMMANDS: dict[str, CommandSpec] = {
    "switch": CommandSpec(
        name="switch",
        description="切换工作模式",
        arg_candidates=tuple(m.value for m in Mode),
    ),
    "exit": CommandSpec(
        name="exit",
        description="退出程序",
    ),
}


def parse(raw: str) -> tuple[str, list[str]]:
    """把 ``/switch feature_engineering`` 解析为 ``("switch", ["feature_engineering"])``。

    去掉前导 ``/`` 后按空白切分；空指令返回 ``("", [])``。
    """
    parts = raw.lstrip("/").split()
    if not parts:
        return "", []
    return parts[0], parts[1:]


def hint(value: str) -> str:
    """据当前输入生成「所有可用候选」的提示文本，供输入框上方的提示行显示。

    与 ``SlashSuggester`` 互补：Suggester 只在输入框内联补全**首个**匹配（灰色幽灵
    文本），而本函数列出**全部**匹配项——命令名阶段列所有指令，参数阶段列所有候选
    参数（如所有可用模式）。非 slash 输入返回空串（提示行随之收起）。
    """
    if not value.startswith("/"):
        return ""
    body = value[1:]
    # 已到参数段（含空格）→ 列出该指令的全部候选参数。
    if " " in body:
        name, _, partial_arg = body.partition(" ")
        spec = COMMANDS.get(name)
        if spec is None or not spec.arg_candidates:
            return ""
        cands = [c for c in spec.arg_candidates if c.startswith(partial_arg)]
        if not cands:
            return ""
        return "可选参数： " + "  ·  ".join(cands)
    # 命令名段 → 列出所有前缀匹配的指令及其说明。
    cmds = [c for c in COMMANDS.values() if c.name.startswith(body)]
    if not cmds:
        return ""
    return "可用指令： " + "  ·  ".join(f"/{c.name} {c.description}" for c in cmds)


class SlashSuggester(Suggester):
    """内联幽灵文本补全：命令名 + 参数两级补全。"""

    def __init__(self) -> None:
        # 大小写敏感（指令与模式名均为小写）；不缓存（候选集小、逻辑纯）。
        super().__init__(use_cache=False, case_sensitive=True)

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        body = value[1:]
        # 已输入到参数段（含至少一个空格）→ 补全参数；否则补全命令名。
        if " " in body:
            name, _, partial_arg = body.partition(" ")
            spec = COMMANDS.get(name)
            if spec is None:
                return None
            for cand in spec.arg_candidates:
                if cand.startswith(partial_arg) and cand != partial_arg:
                    return f"/{name} {cand}"
            return None
        for cmd_name in COMMANDS:
            if cmd_name.startswith(body) and cmd_name != body:
                return f"/{cmd_name}"
        return None
