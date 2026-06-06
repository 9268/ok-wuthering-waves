"""Eval_Harness 统一异常类型。

所有子系统在遇到可预期的错误（目录缺失、参数集非法、缓存缺失、数据文件损坏等）
时统一抛出 :class:`HarnessError`，并在消息中携带相关的路径或名称，便于
Harness_CLI 顶层捕获后以非零退出码与清晰文案呈现。
"""

from __future__ import annotations

from typing import Optional


class HarnessError(Exception):
    """Eval_Harness 的统一异常。

    携带一条可读的错误消息，并可选地附带与该错误相关的路径或名称
    （例如缺失的目录、损坏的数据文件、非法的 Param_Set_Name 等），
    以便定位问题来源。

    Args:
        message: 人类可读的错误描述。
        path: 与该错误相关的文件 / 目录路径（可选）。
        name: 与该错误相关的标识名（如 Param_Set_Name，可选）。
    """

    def __init__(
        self,
        message: str,
        *,
        path: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        self.message = message
        self.path = path
        self.name = name
        super().__init__(self._format())

    def _format(self) -> str:
        parts = [self.message]
        if self.path is not None:
            parts.append(f"path={self.path!r}")
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        return " | ".join(parts)


__all__ = ["HarnessError"]
