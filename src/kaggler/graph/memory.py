"""结构化即时记忆（Agent 对之前对话的压缩记忆）。

把原先单条自由文本 ``summary`` 升级为三段式结构化对象，各段有不同的生命周期：

- ``user_goal``    —— 粘性业务目标：只在意图明确改变时更新，不随每次摘要合并被重压缩，
                      故 Agent 不会随轮次推进丢失最初的业务框架。
- ``key_findings`` —— 累积去重的关键结论：合并时是「旧 + 新」并集（保序去重、截尾封顶），
                      而非整体重写，避免早期结论被后来的摘要挤掉。
- ``progress``     —— 已完成操作的滚动叙事：这是真正需要压缩的部分，每次合并整体重写。

本模块只依赖标准库（纯数据 + 纯函数），不引入 LangGraph，便于单测与在 state 中以 dict 存取。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

DEFAULT_KEY_FINDINGS_CAP = 12


@dataclass
class AgentMemory:
    user_goal: str = ""
    key_findings: list[str] = field(default_factory=list)
    progress: str = ""

    def is_empty(self) -> bool:
        return not (self.user_goal or self.key_findings or self.progress)

    def to_dict(self) -> dict:
        return {
            "user_goal": self.user_goal,
            "key_findings": list(self.key_findings),
            "progress": self.progress,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "AgentMemory":
        d = d or {}
        findings = d.get("key_findings") or []
        # 容错：非 list（如模型偶发返回字符串）时降级为单元素列表或空列表
        if isinstance(findings, str):
            findings = [findings] if findings.strip() else []
        elif not isinstance(findings, list):
            findings = []
        return cls(
            user_goal=str(d.get("user_goal") or ""),
            key_findings=[str(x) for x in findings if str(x).strip()],
            progress=str(d.get("progress") or ""),
        )


def _strip_code_fence(text: str) -> str:
    """剥离 ```json ... ``` / ``` ... ``` 围栏，返回其中的正文。"""
    s = text.strip()
    if not s.startswith("```"):
        return s
    # 去掉首行围栏（``` 或 ```json），再去掉末尾围栏
    lines = s.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_memory(text: str) -> AgentMemory | None:
    """容错解析 FLASH 产出的 JSON 记忆。

    支持中文键（用户目标 / 关键发现 / 进展）与英文键（user_goal / key_findings /
    progress）两种写法；剥离可能存在的 Markdown 代码围栏。解析失败返回 None，
    由调用方决定回退策略。
    """
    if not text or not text.strip():
        return None
    try:
        raw = json.loads(_strip_code_fence(text))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    normalized = {
        "user_goal": raw.get("user_goal", raw.get("用户目标", "")),
        "key_findings": raw.get("key_findings", raw.get("关键发现", [])),
        "progress": raw.get("progress", raw.get("进展", "")),
    }
    return AgentMemory.from_dict(normalized)


def merge_memory(
    prev: AgentMemory,
    update: AgentMemory,
    *,
    key_findings_cap: int = DEFAULT_KEY_FINDINGS_CAP,
) -> AgentMemory:
    """按各段生命周期合并旧记忆与新记忆。

    - user_goal：优先新值，缺省沿用旧值（模型被指示保留既有目标 → 锚定不漂移）。
    - key_findings：旧 + 新 保序去重，超出上限时截尾保留最新。
    - progress：直接取新值（对被删片段的滚动压缩）。
    """
    goal = update.user_goal or prev.user_goal

    findings: list[str] = []
    seen: set[str] = set()
    for f in [*prev.key_findings, *update.key_findings]:
        if f not in seen:
            seen.add(f)
            findings.append(f)
    if key_findings_cap > 0 and len(findings) > key_findings_cap:
        findings = findings[-key_findings_cap:]

    progress = update.progress or prev.progress

    return AgentMemory(user_goal=goal, key_findings=findings, progress=progress)


def render_memory(mem: AgentMemory) -> str:
    """把结构化记忆渲染为注入系统提示词的文本块（空字段跳过）。"""
    parts: list[str] = []
    if mem.user_goal:
        parts.append(f"用户目标：{mem.user_goal}")
    if mem.key_findings:
        findings = "\n".join(f"- {f}" for f in mem.key_findings)
        parts.append(f"关键发现：\n{findings}")
    if mem.progress:
        parts.append(f"进展：{mem.progress}")
    return "\n".join(parts)


__all__ = [
    "AgentMemory",
    "DEFAULT_KEY_FINDINGS_CAP",
    "parse_memory",
    "merge_memory",
    "render_memory",
]
