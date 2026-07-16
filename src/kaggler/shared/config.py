import os
from enum import Enum

from langchain_deepseek import ChatDeepSeek
from pydantic_settings import BaseSettings

class DeepSeekModel(str, Enum):
    FLASH = "deepseek-v4-flash"
    PRO = "deepseek-v4-pro"


class GraphConfig(BaseSettings):
    summary_trigger_count: int = 20
    summary_keep_recent: int = 4
    key_findings_cap: int = 12




def make_llm_raw(
        model: DeepSeekModel = DeepSeekModel.PRO,
        temperature: float = 0.0,
        enable_thinking: bool = False  # 在#37065合并前不要更改为True
) -> ChatDeepSeek:
    assert 0.0 <= temperature <= 1., "模型温度应当位于[0, 1], 检查任何调用工厂函数的代码"
    return ChatDeepSeek(
        model=model.value,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=temperature,
        extra_body={"thinking": {"type": "enabled" if enable_thinking else "disabled"}},
    )


