"""Property 1 的属性测试：case_id 映射是去扩展名的双射（Requirement 1.2）。

仅实现设计文档中编号的 Property 1，不引入额外属性。
"""

from __future__ import annotations

import os
import shutil
import tempfile

from hypothesis import given, settings
from hypothesis import strategies as st

from src.eval_harness.case_loader import CaseLoader

# 文件名主名（stem）字符集：仅小写字母、数字、下划线、连字符，
# 避免路径分隔符、点号（会改变去扩展名结果）与大小写歧义，
# 同时规避 Windows 保留名（如 CON/PRN/AUX/NUL 等，均为大写，故此处天然不会出现）。
_stem = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-",
    min_size=1,
    max_size=24,
)

# 生成一组互异的 stem 集合（至少 1 个，至多 20 个）。
_distinct_stems = st.lists(_stem, min_size=1, max_size=20, unique=True)


# Feature: map-match-eval-harness, Property 1: case_id 映射是去扩展名的双射
# Validates: Requirements 1.2
@settings(max_examples=200)
@given(stems=_distinct_stems)
def test_case_id_is_bijection_of_stripped_extension(stems: list[str]) -> None:
    """互异 ``.png`` 文件名一一对应互异的去扩展名 case_id（双射）。"""
    tmp_dir = tempfile.mkdtemp(prefix="case_loader_prop_")
    try:
        for stem in stems:
            path = os.path.join(tmp_dir, stem + ".png")
            with open(path, "wb"):
                pass

        cases = CaseLoader(tmp_dir).load()

        # 用例数量等于文件数量（不丢失、不重复）。
        assert len(cases) == len(stems)

        case_ids = [c.case_id for c in cases]

        # case_id 两两互异（单射）。
        assert len(set(case_ids)) == len(case_ids)

        # case_id 集合恰好等于 stem 集合（满射 + 单射 = 双射）。
        assert set(case_ids) == set(stems)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
