"""Property 11: Simulated_OCR 噪声边界（src.eval_harness.sequence.simulate_ocr）。

覆盖 Requirement 9.2：

- 每个游戏坐标分量取 ``g × (1 + u)``，``u ~ U(−noise, +noise)``，
  因此加噪结果满足 ``|simulated − g| ≤ |g| × noise``（模浮点舍入）。
- profile 可为 :class:`LocalizationProfile`（取 ``ocr_noise``），也可为裸 ``noise`` 数值
  （``_extract_noise`` 两者皆支持）。

属性测试使用 hypothesis，至少运行 100 个生成样例。
"""

from __future__ import annotations

import random

from hypothesis import given, settings
from hypothesis import strategies as st

from src.eval_harness.profile import LocalizationProfile
from src.eval_harness.sequence import simulate_ocr

# 有界游戏坐标：限制量级避免浮点溢出 / 精度退化（NaN / inf 已排除）。
_coord = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
# 噪声幅度 noise ∈ [0, 1)。
_noise = st.floats(
    min_value=0.0, max_value=0.999, allow_nan=False, allow_infinity=False
)
_seed = st.integers(min_value=0, max_value=2**31 - 1)


def _assert_within_bound(simulated: float, g: float, noise: float) -> None:
    """断言 ``|simulated − g| ≤ |g| × noise``（含浮点容差）。"""
    abs_tol = 1e-9 * max(1.0, abs(g))
    assert abs(simulated - g) <= abs(g) * noise + abs_tol


# Feature: map-match-eval-harness, Property 11: Simulated_OCR 噪声边界
@settings(max_examples=200)
@given(gx=_coord, gy=_coord, noise=_noise, seed=_seed)
def test_simulated_ocr_within_noise_bound_profile(gx, gy, noise, seed):
    """以 LocalizationProfile 提供噪声时，每个分量满足噪声边界。

    **Validates: Requirements 9.2**
    """
    profile = LocalizationProfile(name="p", ocr_noise=noise)
    sx, sy = simulate_ocr((gx, gy), profile, random.Random(seed))

    _assert_within_bound(sx, gx, noise)
    _assert_within_bound(sy, gy, noise)


# Feature: map-match-eval-harness, Property 11: Simulated_OCR 噪声边界
@settings(max_examples=200)
@given(gx=_coord, gy=_coord, noise=_noise, seed=_seed)
def test_simulated_ocr_within_noise_bound_bare_noise(gx, gy, noise, seed):
    """以裸 noise 数值提供噪声时，每个分量满足噪声边界。

    **Validates: Requirements 9.2**
    """
    sx, sy = simulate_ocr((gx, gy), noise, random.Random(seed))

    _assert_within_bound(sx, gx, noise)
    _assert_within_bound(sy, gy, noise)
