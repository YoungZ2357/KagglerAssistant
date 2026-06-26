# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# ProjectName: KagglerAssistant
# FileName: registry.py
# Date: 2026/6/26 10:27
# -------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Callable

from langchain_core.tools import BaseTool

from kaggler.modes import eda
from kaggler.shared.types import Mode
from kaggler.workspace.data_provider import DataProvider



@dataclass
class ModeSpec:
    tool_factory: Callable[[DataProvider], list[BaseTool]]
    prompt: str

"""
使用方法：在运行时实时实例化
# app/cli.py

data = DataProvider()
data.load_initial(csv_path)

# 在这里初始化通用工具
common_tools = make_common_tools()


# 在这里注入通用工具
tools_by_mode = {
    mode: [spec.tool_factory(data), *common_tools]
    for mode, spec in REGISTRY.items()
}

prompt_templates: dict[Mode, str] = {
    mode: spec.prompt_templates
    for mode, spec in REGISTRY.items()
}

"""

REGISTRY: dict[Mode, ModeSpec] = {
    Mode.EDA: ModeSpec(
        tool_factory=eda.make_tools, prompt=eda.EDA_SYSTEM_PROMPT_TEMPLATE
    )
}


