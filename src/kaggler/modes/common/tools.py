from dataclasses import asdict
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langgraph.types import Command

from kaggler.persistence.data_provider import DataProvider
from kaggler.shared.tool_helpers import dumps_cn
from kaggler.shared.types import Mode


def make_tools(data: DataProvider) -> list[BaseTool]:

    @tool
    def switch_mode(
            new_mode: Mode,
            tool_call_id: Annotated[str, InjectedToolCallId]
            # state: Annotated[dict, InjectedState]
    ) -> Command:
        """切换当前工作模式（mode）。

        当用户的需求超出当前模式能力范围、需要进入另一能力切片时调用，
        例如从 EDA 探索切换到建模。new_mode 必须是受支持的模式枚举值。
        """
        return Command(update={
            "current_mode": new_mode,
            "messages": [
                ToolMessage(f"代理已经切换至新模式: {new_mode}", tool_call_id=tool_call_id),
            ]
        })

    @tool
    def switch_data_version(
            version: int,
            tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """切换当前所使用的数据版本（撤销/重做数据处理操作的手段）。

        每次特征工程工具成功执行后都会生成一个新版本；调用此工具可以把当前工作
        指针切换到任意已存在的历史版本，不会丢弃其他版本（所有版本一直保留）。
        切换成功后会返回该版本的谱系信息：源自哪个版本（parent）、由哪个工具
        产生（tool）、以及简要描述（description）。

        使用情景：
        - 用户要求撤销/回退上一步（或多步）数据处理操作
        - 用户希望在不同处理版本之间来回比较效果
        - 某次处理结果不理想，需要基于更早的版本重新尝试
        - 如果不确定有哪些版本可切，建议先调用 list_data_versions 查看

        version 必须是已存在的版本号，可通过 list_data_versions 获取合法范围。
        不要凭空猜测版本号；此工具只切换指针，不会创建新版本或修改数据内容。
        """
        try:
            data.get(version)
        except RuntimeError as e:
            return Command(update={
                "messages": [
                    ToolMessage(dumps_cn({"error": str(e)}), tool_call_id=tool_call_id),
                ],
            })

        info = data.get_version_info(version)
        return Command(update={
            "data_version": version,
            "messages": [
                ToolMessage(
                    dumps_cn({"current_data_version": version, **asdict(info)}),
                    tool_call_id=tool_call_id,
                ),
            ],
        })

    @tool
    def list_data_versions() -> str:
        """列出当前所有可用的数据版本及其谱系信息（父版本、产生它的工具、简要描述）。

        用于在切换数据版本（switch_data_version）之前，浏览有哪些历史版本可选、
        每个版本源自哪个版本、被哪个工具处理产生，从而判断应该切换到哪一个版本，
        或者重建完整的处理链条（沿 parent 指针可以追溯到根版本）。

        使用情景：
        - 用户询问有哪些历史版本、之前做过哪些处理
        - 调用 switch_data_version 前需要确认合法的版本号
        - 需要回顾/重建数据处理的完整操作链条

        不修改任何状态，仅读取已存在的数据版本信息。
        """
        return dumps_cn(data.list_versions())

    return [switch_mode, switch_data_version, list_data_versions]
