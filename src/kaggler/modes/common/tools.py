from dataclasses import asdict
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langgraph.types import Command

from kaggler.modes.common.compute import list_files
from kaggler.persistence.data_provider import DataProvider
from kaggler.shared.tool_helpers import dumps_cn
from kaggler.shared.types import Mode
from kaggler.workspace.manager import get_active_workspace


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

        data.set_head(version)
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

    @tool
    def list_workspace_files(
            directory: Annotated[str, "相对于工作区根目录的子路径，默认为 '.'"] = ".",
    ) -> str:
        """列出当前工作区指定目录下的所有文件和子目录。

        当需要了解工作区中有哪些数据文件、脚本、输出等资源时调用。
        目录参数为相对于工作区根目录的路径，默认为工作区根目录 ('')。
        返回按「目录优先 → 名称升序」排列的格式化列表，含文件大小。
        仅在已设置工作区时可用；若未设置工作区则返回错误提示。

        使用情景：
        - 用户询问有哪些数据文件可用
        - 用户需要查看工作区中的输出或中间结果
        - Agent 在决策前需要确认某文件是否存在
        """
        ws = get_active_workspace()
        if ws is None:
            return "错误：当前未设置工作区。请先使用 /select-workspace 指令选择工作区目录。"
        target = ws.resolve_within(directory)
        if target is None:
            return "错误：不允许访问工作区之外的路径。"
        return list_files(target)

    return [switch_mode, switch_data_version, list_data_versions, list_workspace_files]
