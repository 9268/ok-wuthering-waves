"""Property 13: 地图锁定判定（sequence.should_lock）。

覆盖 Requirement 9.4：

:func:`src.eval_harness.sequence.should_lock` 当且仅当满足以下三条时返回 ``True``：

1. 当前未锁定（``not currently_locked``）；
2. 置信度达标（``confidence >= profile.lock_confidence``）；
3. 匹配点数达标（``match_count >= profile.lock_match_count``）。

属性测试使用 hypothesis，每条属性至少运行 100 个生成样例。生成的 confidence 跨越
``lock_confidence`` 阈值、match_count 跨越 ``lock_match_count`` 阈值，并随机变化
锁定状态与 profile 阈值，从而覆盖判定边界。
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.eval_harness.profile import LocalizationProfile
from src.eval_harness.sequence import should_lock

# confidence 跨越默认 lock_confidence(0.9) 的取值区间 [0, 1]。
_confidence = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
# match_count 跨越默认 lock_match_count(10) 的整数区间 [0, 30]。
_match_count = st.integers(min_value=0, max_value=30)
# 可变的 profile 阈值，用于覆盖不同判定边界。
_lock_confidence = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
_lock_match_count = st.integers(min_value=0, max_value=30)


# Feature: map-match-eval-harness, Property 13: 地图锁定判定
@settings(max_examples=200)
@given(
    confidence=_confidence,
    match_count=_match_count,
    currently_locked=st.booleans(),
    lock_confidence=_lock_confidence,
    lock_match_count=_lock_match_count,
)
def test_should_lock_iff_thresholds_met_and_unlocked(
    confidence, match_count, currently_locked, lock_confidence, lock_match_count
):
    """当且仅当未锁定且置信度与匹配点数双双达标时锁定。

    **Validates: Requirements 9.4**
    """
    profile = LocalizationProfile(
        name="prop13",
        lock_confidence=lock_confidence,
        lock_match_count=lock_match_count,
    )

    expected = (
        not currently_locked
        and confidence >= profile.lock_confidence
        and match_count >= profile.lock_match_count
    )

    assert should_lock(confidence, match_count, currently_locked, profile) == expected


# Feature: map-match-eval-harness, Property 13: 地图锁定判定
@settings(max_examples=200)
@given(
    confidence=_confidence,
    match_count=_match_count,
    lock_confidence=_lock_confidence,
    lock_match_count=_lock_match_count,
)
def test_should_lock_never_when_already_locked(
    confidence, match_count, lock_confidence, lock_match_count
):
    """已锁定时无论阈值是否达标都不会再次锁定。

    **Validates: Requirements 9.4**
    """
    profile = LocalizationProfile(
        name="prop13-locked",
        lock_confidence=lock_confidence,
        lock_match_count=lock_match_count,
    )

    assert should_lock(confidence, match_count, True, profile) is False
