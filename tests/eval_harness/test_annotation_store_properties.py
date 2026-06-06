"""Property 3：Annotation_Store 往返一致（Requirement 3.3, 3.5, 3.6）。

仅实现设计文档中编号的 Property 3：写入合法 Ground_Truth（``reference_scale > 0``）
保存后，用新的 :class:`AnnotationStore` 读回时各字段在浮点容差内相等。
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from typing import Dict

from hypothesis import given, settings
from hypothesis import strategies as st

from src.eval_harness.annotation import AnnotationStore, GroundTruth

# 有界坐标：避免 nan/inf，幅度上限约 1e7。
_coord = st.floats(
    min_value=-1e7, max_value=1e7, allow_nan=False, allow_infinity=False
)
_xy = st.tuples(_coord, _coord)
# 严格正的参考缩放比（reference_scale > 0）。
_positive_scale = st.floats(
    min_value=1e-6, max_value=1e7, allow_nan=False, allow_infinity=False
)
# map_id：非空可打印字符串。
_map_id = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=20,
)
# case_id：限定为安全字符集，保证作为字典键互异且可读。
_case_id = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_",
    min_size=1,
    max_size=12,
)


@st.composite
def _ground_truths(draw: st.DrawFn) -> Dict[str, GroundTruth]:
    """生成 ``case_id -> GroundTruth`` 映射（case_id 互异，reference_scale > 0）。"""
    case_ids = draw(st.lists(_case_id, min_size=1, max_size=8, unique=True))
    result: Dict[str, GroundTruth] = {}
    for case_id in case_ids:
        result[case_id] = GroundTruth(
            case_id=case_id,
            map_id=draw(_map_id),
            pixel_xy=draw(_xy),
            game_xy=draw(_xy),
            reference_scale=draw(_positive_scale),
        )
    return result


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


# Feature: map-match-eval-harness, Property 3: Annotation_Store 往返一致
@settings(max_examples=100)
@given(gts=_ground_truths())
def test_annotation_store_roundtrip(gts: Dict[str, GroundTruth]) -> None:
    """写入合法 GT 保存后读回，各字段在容差内相等。

    **Validates: Requirements 3.3, 3.5, 3.6**
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp_dir, "annotations.json")

        store = AnnotationStore(path)
        for gt in gts.values():
            store.upsert(gt)
        store.save()

        loaded = AnnotationStore(path).load()

        # 相同的 case_id 集合。
        assert set(loaded) == set(gts)
        for case_id, expected in gts.items():
            got = loaded[case_id]
            assert got.map_id == expected.map_id
            assert _close(got.pixel_xy[0], expected.pixel_xy[0])
            assert _close(got.pixel_xy[1], expected.pixel_xy[1])
            assert _close(got.game_xy[0], expected.game_xy[0])
            assert _close(got.game_xy[1], expected.game_xy[1])
            assert _close(got.reference_scale, expected.reference_scale)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
