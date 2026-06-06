"""Case_Loader：测试用例发现与加载（Requirement 1）。

负责从 ``screenshots`` 目录发现并加载全部 ``.png`` 单帧测试用例。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import HarnessError


@dataclass
class TestCase:
    """单个独立单帧测试用例（Requirement 1.2）。

    Attributes:
        case_id: 用例 ID，取该截图去扩展名的文件名（全局唯一、不含子目录）。
        image_path: 截图文件的路径，形如 ``screenshots/{case_id}.png``。
    """

    case_id: str
    image_path: str


class CaseLoader:
    """从 ``screenshots`` 目录发现并加载全部 ``.png`` 测试用例。

    Args:
        screenshots_dir: 存放游戏内截图的目录路径。
    """

    def __init__(self, screenshots_dir: str) -> None:
        self.screenshots_dir = screenshots_dir
        self._cases: list[TestCase] | None = None

    def load(self) -> list[TestCase]:
        """读取目录下全部 ``.png`` 文件作为 Test_Case 集合（Requirement 1.1, 1.2）。

        Returns:
            按 ``case_id`` 排序的 :class:`TestCase` 列表。

        Raises:
            HarnessError: 目录不存在或不包含任何 ``.png`` 文件（Requirement 1.3），
                错误消息中携带目录的绝对路径。
        """
        abs_dir = os.path.abspath(self.screenshots_dir)

        if not os.path.isdir(abs_dir):
            raise HarnessError(
                "screenshots 目录不存在",
                path=abs_dir,
            )

        cases: list[TestCase] = []
        for entry in os.listdir(abs_dir):
            full_path = os.path.join(abs_dir, entry)
            if not os.path.isfile(full_path):
                continue
            base = os.path.basename(entry)
            stem, ext = os.path.splitext(base)
            if ext.lower() != ".png":
                continue
            cases.append(TestCase(case_id=stem, image_path=full_path))

        if not cases:
            raise HarnessError(
                "screenshots 目录中未发现任何 .png 测试用例",
                path=abs_dir,
            )

        cases.sort(key=lambda c: c.case_id)
        self._cases = cases
        return cases

    def count(self) -> int:
        """报告成功加载的 Test_Case 数量（Requirement 1.4）。

        首次调用时若尚未加载则触发一次 :meth:`load`，供 CLI 报告加载数量。

        Returns:
            已加载的测试用例数量。
        """
        if self._cases is None:
            self.load()
        assert self._cases is not None
        return len(self._cases)


__all__ = ["TestCase", "CaseLoader"]
