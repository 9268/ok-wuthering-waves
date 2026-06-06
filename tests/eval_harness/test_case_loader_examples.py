"""Case_Loader 加载错误与计数示例 / 边界测试（Requirement 1.3, 1.4）。

覆盖：

- 目录缺失：``CaseLoader.load()`` 抛出 :class:`HarnessError`，错误信息携带目录绝对路径；
- 目录存在但无 ``.png``：``load()`` 抛出 :class:`HarnessError`，携带目录绝对路径；
- 计数：含 N 个 ``.png`` 的目录，``count()`` 与 ``len(load())`` 均等于 N。

Validates: Requirements 1.3, 1.4
"""

from __future__ import annotations

import os

import pytest

from src.eval_harness.case_loader import CaseLoader
from src.eval_harness.errors import HarnessError


def test_missing_directory_raises_with_path() -> None:
    """目录不存在时 load() 抛出 HarnessError，且消息含该目录的绝对路径（Req 1.3）。"""
    missing = os.path.join(os.sep, "nonexistent", "case_loader_path_xyz")
    abs_missing = os.path.abspath(missing)
    loader = CaseLoader(missing)

    with pytest.raises(HarnessError) as exc_info:
        loader.load()

    err = exc_info.value
    # 结构化属性精确携带绝对路径；消息以 repr 形式内嵌该路径。
    assert err.path == abs_missing
    assert repr(abs_missing) in str(err)


def test_empty_directory_raises_with_path(tmp_path) -> None:
    """目录存在但无 .png 时 load() 抛出 HarnessError，且消息含该目录绝对路径（Req 1.3）。"""
    # 放入一个非 png 文件，确认仍被视为“无用例”。
    (tmp_path / "readme.txt").write_text("not an image", encoding="utf-8")
    abs_dir = os.path.abspath(str(tmp_path))
    loader = CaseLoader(str(tmp_path))

    with pytest.raises(HarnessError) as exc_info:
        loader.load()

    err = exc_info.value
    # 结构化属性精确携带绝对路径；消息以 repr 形式内嵌该路径。
    assert err.path == abs_dir
    assert repr(abs_dir) in str(err)


@pytest.mark.parametrize("n", [1, 3, 5])
def test_count_matches_number_of_png_files(tmp_path, n: int) -> None:
    """含 N 个 .png 的目录，count() == N 且 len(load()) == N（Req 1.4）。"""
    for i in range(n):
        (tmp_path / f"case_{i}.png").write_bytes(b"\x89PNG\r\n")
    # 干扰文件不应计入。
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    loader = CaseLoader(str(tmp_path))

    assert loader.count() == n
    assert len(loader.load()) == n
