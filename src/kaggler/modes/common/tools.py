from dataclasses import asdict
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from kaggler.modes.common.compute import list_files
from kaggler.persistence.data_export import EXPORT_SUBDIR, export_and_record
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

    @tool
    def export_data_version(
            filename: Annotated[str, "导出文件名或工作区内子路径，如 'submission.csv'"],
            tool_call_id: Annotated[str, InjectedToolCallId],
            state: Annotated[dict, InjectedState],
            config: RunnableConfig,
            version: Annotated[int | None, "要导出的版本号；省略则导出当前版本"] = None,
            fmt: Annotated[str | None, "导出格式 csv / parquet；省略则按文件名后缀推断，默认 csv"] = None,
    ) -> str:
        """把某个数据版本导出（持久化）为文件，供在应用外使用（如提交、用 Excel 打开）。

        文件写入当前工作区的受控子目录 `exports/` 下（不允许写到工作区之外）。
        version 省略时导出当前正在使用的数据版本；可先用 list_data_versions 查看有哪些版本。
        格式默认按文件名后缀推断（.csv/.parquet），也可用 fmt 显式指定；当前支持 csv 与 parquet。

        使用情景：
        - 用户要求把处理后的数据 / 某个历史版本保存 / 导出为文件
        - 需要产出可提交（submission）或可在外部工具打开的数据文件

        本工具不创建新数据版本、不改变当前版本指针，仅落盘并登记一条导出记录。
        """
        ws = get_active_workspace()
        if ws is None:
            return "错误：当前未设置工作区，无法导出。请先使用 /select-workspace 指令选择工作区目录。"
        v = version if version is not None else state["data_version"]
        target = ws.resolve_within(f"{EXPORT_SUBDIR}/{filename}")
        if target is None:
            return "错误：不允许导出到工作区之外的路径。"
        try:
            description = data.get_version_info(v).description
            result = export_and_record(
                data, v, target, fmt,
                db_path=ws.data_version_db,
                thread_id=config.get("configurable", {}).get("thread_id"),
                description=description,
            )
        except (RuntimeError, ValueError) as e:
            return dumps_cn({"error": str(e)})
        return dumps_cn({
            "exported_version": v,
            "path": result.path,
            "format": result.format,
            "rows": result.rows,
            "cols": result.cols,
        })

    @tool
    def add_todo(
            content: Annotated[str, "待办内容，一句话描述要做的事"],
            tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """登记一条待办事项（挂起项），防止你遗忘自己提出但尚未执行的建议或后续步骤。

        当你向用户提出了「稍后再做 / 下一步可以 / 建议后续」之类尚未落实的动作时，
        应调用本工具把它记下来。待办会在之后每一轮都完整出现在你的上下文里、且永远
        不会被对话摘要压缩掉，因此这是确保长程规划不丢失的可靠手段。

        使用情景：
        - 你给出了一个多步方案，但本轮只执行了其中一部分，其余步骤需要挂起
        - 你建议了某项后续分析/处理，但当前不适合立即执行
        - 用户暂时转向别的问题，而某个未完成任务需要留待稍后

        新待办的编号（id）由系统自动分配；返回结果会告知登记成功。
        本工具不影响数据版本，仅向待办列表追加一项。
        """
        return Command(update={
            "todos": [{"content": content, "status": "open"}],
            "messages": [
                ToolMessage(
                    dumps_cn({"added_todo": content, "status": "open"}),
                    tool_call_id=tool_call_id,
                ),
            ],
        })

    @tool
    def complete_todo(
            todo_id: Annotated[int, "要标记完成的待办编号（见上下文中的待办列表 [#id]）"],
            tool_call_id: Annotated[str, InjectedToolCallId],
            state: Annotated[dict, InjectedState],
    ) -> Command:
        """把某条待办事项标记为已完成，使其从未完成挂起列表中移除。

        todo_id 必须是当前存在的待办编号——它会显示在你上下文的「未完成的挂起项」
        清单里，形如 [#3]。完成后该项不再出现在后续轮次的挂起列表中。

        使用情景：
        - 你已经执行完之前挂起的某个后续步骤
        - 某条待办因需求变化不再需要（也可标记完成以清理列表）

        若给定编号不存在，会返回错误提示且不做任何改动。
        """
        todos = state.get("todos") or []
        target = next((t for t in todos if t.get("id") == todo_id), None)
        if target is None:
            return Command(update={
                "messages": [
                    ToolMessage(
                        dumps_cn({"error": f"待办编号 {todo_id} 不存在"}),
                        tool_call_id=tool_call_id,
                    ),
                ],
            })
        return Command(update={
            "todos": [{"id": todo_id, "status": "done"}],
            "messages": [
                ToolMessage(
                    dumps_cn({"completed_todo": todo_id, "content": target.get("content")}),
                    tool_call_id=tool_call_id,
                ),
            ],
        })

    return [
        switch_mode,
        switch_data_version,
        list_data_versions,
        list_workspace_files,
        export_data_version,
        add_todo,
        complete_todo,
    ]
