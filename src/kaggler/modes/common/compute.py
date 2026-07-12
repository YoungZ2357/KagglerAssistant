"""通用工具的计算逻辑——不依赖 LangChain / graph，可被 TUI 与 Agent 工具共用。"""

from __future__ import annotations

from pathlib import Path

from kaggler.shared.limits import MAX_WORKSPACE_ENTRIES


def _format_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def list_files(directory: Path) -> str:
    """列出目录下的文件和子目录，返回格式化字符串。

    按「目录优先 → 名称升序」排序，目录名尾部加 ``/``，文件显示大小。
    """
    target = directory.resolve()
    if not target.exists():
        return f"目录不存在：{target}"
    if not target.is_dir():
        return f"不是目录：{target}"

    try:
        items = sorted(
            target.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
    except PermissionError:
        return f"无权限访问目录：{target}"

    if not items:
        return "(空目录)"

    total = len(items)
    truncated = total > MAX_WORKSPACE_ENTRIES
    shown_items = items[:MAX_WORKSPACE_ENTRIES] if truncated else items

    lines: list[str] = []
    for item in shown_items:
        if item.is_dir():
            lines.append(f"📁 {item.name}/")
        else:
            size = _format_size(item.stat().st_size)
            lines.append(f"📄 {item.name} ({size})")

    if truncated:
        lines.append(f"… 还有 {total - MAX_WORKSPACE_ENTRIES} 个条目未显示（共 {total} 个）")

    return "\n".join(lines)
