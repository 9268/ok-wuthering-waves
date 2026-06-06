"""Property 5: 游戏坐标距离换算一致（metrics.error_distance / metrics.MAP_SCALE）。

覆盖 Requirements 6.2、9.6：

- 两个大地图像素点经 ``game = pixel × MAP_SCALE + offset`` 换算后的欧氏距离，
  等于其像素欧氏距离乘以 ``MAP_SCALE``（6.2）。
- 公共的 ``offset`` 在求差时抵消，因此结果与 ``offset`` 无关（9.6）。

属性测试使用 hypothesis，每条属性至少运行 100 个生成样例。
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from src.eval_harness.metrics import MAP_SCALE, error_distance

# 有界坐标 / 偏移：限制量级避免浮点溢出与精度退化（NaN / inf 已排除）。
_coord = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
_point = st.tuples(_coord, _coord)
_offset = st.tuples(_coord, _coord)


# Feature: map-match-eval-harness, Property 5: 游戏坐标距离换算一致
@settings(max_examples=200)
@given(p1=_point, p2=_point, offset=_offset)
def test_error_distance_equals_pixel_distance_times_scale(p1, p2, offset):
    """换算后欧氏距离 = 像素欧氏距离 × MAP_SCALE，且与 offset 无关。

    **Validates: Requirements 6.2, 9.6**
    """
    pixel_dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    expected = pixel_dist * MAP_SCALE

    actual = error_distance(p1, p2, offset)

    # 相对 + 绝对容差：覆盖大量级（相对）与近零（绝对）两种情形。
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6)


# Feature: map-match-eval-harness, Property 5: 游戏坐标距离换算一致
@settings(max_examples=200)
@given(p1=_point, p2=_point, offset_a=_offset, offset_b=_offset)
def test_error_distance_independent_of_offset(p1, p2, offset_a, offset_b):
    """任意两个 offset 下 Error_Distance 相同（offset 在求差时抵消）。

    **Validates: Requirements 9.6**
    """
    dist_a = error_distance(p1, p2, offset_a)
    dist_b = error_distance(p1, p2, offset_b)

    assert math.isclose(dist_a, dist_b, rel_tol=1e-9, abs_tol=1e-6)
