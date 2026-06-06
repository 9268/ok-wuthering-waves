"""Property 12: 平滑为窗口内凸组合（sequence.smooth）。

覆盖 Requirement 9.3：

:func:`src.eval_harness.sequence.smooth` 对窗口内最近的 ``game_center`` 序列做加权移动
平均。由于权重被归一化为非负且和为 1，输出是所用点的**凸组合**，因此每个坐标分量都落在
**实际参与平滑的点**对应分量的最小值与最大值之间，不会越界。

注意对齐规则（见 ``smooth`` docstring）：窗口长度与权重个数不一致时尾部对齐——

- 若 ``len(weights) >= len(points)``：使用全部点；
- 若 ``len(points) > len(weights)``：仅使用最近的 ``len(weights)`` 个点。

凸组合不变量只在**实际使用的点子集**上成立，故测试按同样的对齐规则计算 min/max 边界。

属性测试使用 hypothesis，每条属性至少运行 100 个生成样例。
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from src.eval_harness.sequence import smooth

# 有界坐标，避免 NaN/Inf 与极端量级带来的浮点噪声。
_coord = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
_point = st.tuples(_coord, _coord)
# 正权重（min_value > 0），保证总权重为正、归一化后是真正的凸组合。
_weight = st.floats(
    min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False
)


def _used_points(points, weights):
    """复现 smooth 的尾部对齐规则，返回实际参与平滑的点子集。"""
    pts = list(points)
    w = list(weights)
    if len(w) >= len(pts):
        return pts
    return pts[len(pts) - len(w):]


def _assert_convex(points, weights, out):
    used = _used_points(points, weights)
    xs = [p[0] for p in used]
    ys = [p[1] for p in used]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # 边界用绝对 + 相对容差吸收浮点误差。
    tol_x = 1e-6 + 1e-9 * max(abs(min_x), abs(max_x))
    tol_y = 1e-6 + 1e-9 * max(abs(min_y), abs(max_y))

    assert out[0] > min_x or math.isclose(out[0], min_x, abs_tol=tol_x)
    assert out[0] < max_x or math.isclose(out[0], max_x, abs_tol=tol_x)
    assert out[1] > min_y or math.isclose(out[1], min_y, abs_tol=tol_y)
    assert out[1] < max_y or math.isclose(out[1], max_y, abs_tol=tol_y)


# Feature: map-match-eval-harness, Property 12: 平滑为窗口内凸组合
@settings(max_examples=200)
@given(data=st.data())
def test_smooth_is_convex_combination_equal_lengths(data):
    """点与权重等长：所有点都被使用，输出各分量落在其 min/max 之间。

    **Validates: Requirements 9.3**
    """
    points = data.draw(st.lists(_point, min_size=1, max_size=12))
    weights = data.draw(
        st.lists(_weight, min_size=len(points), max_size=len(points))
    )
    out = smooth(points, weights)
    _assert_convex(points, weights, out)


# Feature: map-match-eval-harness, Property 12: 平滑为窗口内凸组合
@settings(max_examples=200)
@given(data=st.data())
def test_smooth_is_convex_combination_misaligned_lengths(data):
    """点与权重不等长：按尾部对齐规则在实际使用的点子集上仍是凸组合。

    **Validates: Requirements 9.3**
    """
    points = data.draw(st.lists(_point, min_size=1, max_size=12))
    weights = data.draw(st.lists(_weight, min_size=1, max_size=12))
    out = smooth(points, weights)
    _assert_convex(points, weights, out)
