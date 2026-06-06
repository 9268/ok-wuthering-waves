"""Property 10: Frame_Sequence 构造不变量（sequence.build_sequence）。

覆盖 Requirement 9.1：

- 每个 :class:`FrameSequence` 内所有帧的 ``map_id`` 一致，且等于 ``seq.map_id``——
  不同地图的帧绝不进入同一序列。
- 序列内**任意两帧**的投影中心（地图像素 ``pixel_xy``）欧氏距离均严格 ``< threshold``
  （默认 1000）——全配对约束（design.md Property 10）。
- 构造是输入帧集合的一个划分：每个输入帧恰好出现在一个输出序列中（不丢失、不重复、
  不插值）。

属性测试使用 hypothesis，每条属性至少运行 100 个生成样例。
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from src.eval_harness.sequence import (
    DEFAULT_REGION_SIZE,
    Frame,
    build_sequence,
)

# map_id 取自小集合，使同图帧大概率被分到同组、触发实际的序列构造逻辑。
_map_id = st.sampled_from(["a", "b", "c"])
# 坐标范围 [-3000, 3000]：部分点对距离 < 1000、部分 > 1000，覆盖收束与另起新序列。
_coord = st.floats(
    min_value=-3000.0, max_value=3000.0, allow_nan=False, allow_infinity=False
)
_point = st.tuples(_coord, _coord)


def _frame(case_id, map_id, pixel_xy):
    return Frame(
        case_id=case_id,
        map_id=map_id,
        pixel_xy=pixel_xy,
        game_xy=pixel_xy,
        reference_scale=1.0,
    )


# 生成带唯一 case_id 的帧列表，便于校验划分（每帧恰好出现一次）。
@st.composite
def _frames(draw):
    specs = draw(
        st.lists(st.tuples(_map_id, _point), min_size=0, max_size=12)
    )
    return [
        _frame(f"case-{i}", map_id, pixel_xy)
        for i, (map_id, pixel_xy) in enumerate(specs)
    ]


# Feature: map-match-eval-harness, Property 10: Frame_Sequence 构造不变量
@settings(max_examples=200)
@given(frames=_frames())
def test_build_sequence_construction_invariants(frames):
    """同序列同 map_id；序列内任意两帧投影中心距离 < 阈值；构造为输入帧的划分。

    **Validates: Requirements 9.1**
    """
    threshold = DEFAULT_REGION_SIZE
    sequences = build_sequence(frames, threshold=threshold)

    seen = []
    for seq in sequences:
        # (1) 同序列同 map_id。
        for fr in seq.frames:
            assert fr.map_id == seq.map_id

        # (2) 任意两帧投影中心欧氏距离 < threshold（全配对约束）。
        for i in range(len(seq.frames)):
            for j in range(i + 1, len(seq.frames)):
                a = seq.frames[i].pixel_xy
                b = seq.frames[j].pixel_xy
                dist = math.hypot(a[0] - b[0], a[1] - b[1])
                assert dist < threshold

        seen.extend(seq.frames)

    # (3) 划分不变量：输入帧总数守恒，且每个输入帧恰好出现一次（按 case_id 标识）。
    assert len(seen) == len(frames)
    seen_ids = sorted(fr.case_id for fr in seen)
    input_ids = sorted(fr.case_id for fr in frames)
    assert seen_ids == input_ids
