"""DeepSeek V4 上下文 token 估算 —— 按类别拆分 + 实测自适应校准。

全仓无精确分词器；本模块用「按字符类别加权」的启发式离线估算每次送入模型的
prompt 规模，并用 DeepSeek 返回的 `usage_metadata.input_tokens`（真实 prompt
token 数）做滑动平均校准，使离线估算随使用自我收敛。

校准约定（DeepSeek V4，用户提供）：
- 1 个中文字符 ≈ 0.8 token
- 1 个英文/ASCII 字符 ≈ 0.3 token
- 推荐有效上下文 256k，实际极限 1M

纯模块：仅依赖标准库与 langchain_core，可被 graph / TUI / 测试安全导入。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

# ── 权重与预算常量 ──────────────────────────────────────────────────────────
CJK_TOKEN_WEIGHT = 0.8       # 每个 CJK 字符的 token 估值
OTHER_TOKEN_WEIGHT = 0.3     # 其余字符（ASCII/数字/空白/标点）的 token 估值
MESSAGE_OVERHEAD_TOKENS = 4  # 每条消息的结构性开销（role 包裹、分隔符等），可由校准吸收

CONTEXT_RECOMMENDED = 256_000   # 推荐有效上下文
CONTEXT_LIMIT = 1_000_000       # 实际极限上下文

# ── 校准参数 ────────────────────────────────────────────────────────────────
EMA_ALPHA = 0.3              # 校准系数的滑动平均权重（越大越跟随最新实测）
FACTOR_MIN = 0.2             # 系数夹取下界
FACTOR_MAX = 5.0             # 系数夹取上界
MIN_CALIB_SAMPLE = 50        # 估算原始总量低于此值时不参与校准（避免小样本噪声）


def _is_cjk(ch: str) -> bool:
    """判定字符是否属于 CJK（含常用 CJK 标点、全角字符）。"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF      # CJK 统一表意文字
        or 0x3400 <= code <= 0x4DBF   # CJK 扩展 A
        or 0x3000 <= code <= 0x303F   # CJK 符号与标点
        or 0xFF00 <= code <= 0xFFEF   # 全角 ASCII / 半角片假名
    )


def estimate_text(text: str) -> int:
    """按字符类别加权估算文本 token 数（CJK 0.8/字，其余 0.3/字）。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return round(cjk * CJK_TOKEN_WEIGHT + other * OTHER_TOKEN_WEIGHT)


def _content_str(content) -> str:
    """把消息 content 归一为字符串（兼容多模态的 list content）。"""
    if isinstance(content, str):
        return content
    return str(content) if content is not None else ""


def estimate_tool_tokens(tools: Iterable[BaseTool]) -> int:
    """估算工具定义的 token：还原为实际下发的 OpenAI function JSON schema 后计数。

    复现 `bind_tools` 真正送给模型的字节（含中文 docstring 描述与参数 schema）。
    """
    total = 0
    for tool in tools:
        try:
            schema = convert_to_openai_tool(tool)
            total += estimate_text(json.dumps(schema, ensure_ascii=False))
        except Exception:  # noqa: BLE001 — 单个工具转换失败不应拖垮整体估算
            total += estimate_text(getattr(tool, "description", "") or "")
    return total


def estimate_messages(messages: Iterable[BaseMessage]) -> dict[str, int]:
    """按消息类型把会话历史归并为 user / assistant / tool_results 三桶的 token 估算。

    - HumanMessage → user
    - AIMessage → assistant（正文 + 序列化后的 tool_calls 参数）
    - ToolMessage → tool_results（工具返回内容，常为大 JSON）
    其余类型（如意外混入的 SystemMessage）忽略。每条计入一份结构性开销。
    """
    buckets = {"user": 0, "assistant": 0, "tool_results": 0}
    for msg in messages:
        if isinstance(msg, HumanMessage):
            buckets["user"] += estimate_text(_content_str(msg.content)) + MESSAGE_OVERHEAD_TOKENS
        elif isinstance(msg, AIMessage):
            tokens = estimate_text(_content_str(msg.content))
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                tokens += estimate_text(json.dumps(tool_calls, ensure_ascii=False, default=str))
            buckets["assistant"] += tokens + MESSAGE_OVERHEAD_TOKENS
        elif isinstance(msg, ToolMessage):
            buckets["tool_results"] += (
                estimate_text(_content_str(msg.content)) + MESSAGE_OVERHEAD_TOKENS
            )
    return buckets


@dataclass
class ContextBreakdown:
    """一次 invoke 的上下文 token 分类拆分（各字段为**原始**估算，未乘校准系数）。

    展示时统一经 `calibration_factor` 缩放；`to_dict()` 输出已缩放的自洽载荷，
    供 stream/state 传递与 TUI 渲染。
    """

    system: int
    summary: int
    tools: int
    user: int
    assistant: int
    tool_results: int
    calibration_factor: float = 1.0
    actual_total: int | None = None

    _CATEGORY_KEYS = ("system", "summary", "tools", "user", "assistant", "tool_results")

    @property
    def estimated_total_raw(self) -> int:
        """未经校准的原始估算总量（用于校准比值计算）。"""
        return sum(getattr(self, k) for k in self._CATEGORY_KEYS)

    def _scaled(self, value: int) -> int:
        return round(value * self.calibration_factor)

    @property
    def estimated_total(self) -> int:
        """经校准系数缩放后的估算总量。"""
        return self._scaled(self.estimated_total_raw)

    @property
    def total(self) -> int:
        """展示总量：有实测取实测，否则取校准后估算。"""
        return self.actual_total if self.actual_total is not None else self.estimated_total

    def to_dict(self) -> dict:
        """输出自洽的可视化载荷：已缩放的分类值 + 总量 + 校准信息 + 预算常量。"""
        return {
            "categories": {k: self._scaled(getattr(self, k)) for k in self._CATEGORY_KEYS},
            "estimated_total": self.estimated_total,
            "actual_total": self.actual_total,
            "total": self.total,
            "calibration_factor": round(self.calibration_factor, 4),
            "recommended": CONTEXT_RECOMMENDED,
            "limit": CONTEXT_LIMIT,
        }


def build_breakdown(
    system_prompt_text: str,
    summary_text: str,
    tools: Iterable[BaseTool],
    messages: Iterable[BaseMessage],
) -> ContextBreakdown:
    """从组装点的三类原料构造**原始**分类拆分（factor=1、无实测）。

    调用方随后设置 `calibration_factor` / `actual_total` 再 `to_dict()`。
    """
    msg_buckets = estimate_messages(messages)
    return ContextBreakdown(
        system=estimate_text(system_prompt_text),
        summary=estimate_text(summary_text) if summary_text else 0,
        tools=estimate_tool_tokens(tools),
        user=msg_buckets["user"],
        assistant=msg_buckets["assistant"],
        tool_results=msg_buckets["tool_results"],
    )


def next_calibration_factor(prev: float, estimated_raw: int, actual: int | None) -> float:
    """按实测/原始估算比值做 EMA 更新并夹取；缺实测或样本过小则原样返回。"""
    if actual is None or estimated_raw < MIN_CALIB_SAMPLE:
        return prev
    ratio = actual / estimated_raw
    updated = EMA_ALPHA * ratio + (1 - EMA_ALPHA) * prev
    clamped = max(FACTOR_MIN, min(FACTOR_MAX, updated))
    return round(clamped, 4)


# ── 展示辅助（供 widget 与测试共用；不含颜色/Rich 依赖）─────────────────────
def utilization(total: int, budget: int = CONTEXT_RECOMMENDED) -> float:
    """占用率 = total / 预算（可 > 1 表示已超预算）。"""
    if budget <= 0:
        return 0.0
    return total / budget


def bar_fill(fraction: float, width: int) -> int:
    """把占比映射为进度条填充格数，夹取到 [0, width]。"""
    if width <= 0:
        return 0
    return max(0, min(width, round(fraction * width)))


__all__ = [
    "CJK_TOKEN_WEIGHT",
    "OTHER_TOKEN_WEIGHT",
    "MESSAGE_OVERHEAD_TOKENS",
    "CONTEXT_RECOMMENDED",
    "CONTEXT_LIMIT",
    "EMA_ALPHA",
    "FACTOR_MIN",
    "FACTOR_MAX",
    "MIN_CALIB_SAMPLE",
    "ContextBreakdown",
    "estimate_text",
    "estimate_tool_tokens",
    "estimate_messages",
    "build_breakdown",
    "next_calibration_factor",
    "utilization",
    "bar_fill",
]
