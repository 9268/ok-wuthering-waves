"""Property 6: 单帧通过判定（metrics.is_pass / metrics.scale_error）。

覆盖 Requirements 6.3、6.5：

- 当且仅当 ``Error_Distance ≤ 3320`` 且 ``Scale_Error ≤ 0.10`` 时通过（6.3）。
- ``measured ≤ 0`` 或 ``reference ≤ 0`` → Scale_Error 未定义（``None``），
  从而判定失败（6.5）。

属性测试使用 hypothesis，每条属性至少运行 100 个生成样例。
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.eval_harness.metrics import (
    PASS_THRESHOLD,
    SCALE_TOLERANCE,
    is_pass,
    scale_error,
)

# error_dist / scale_err 生成器：横跨阈值两侧的有界浮点，并混入 None（未定义）。
_error_dist_values = st.one_of(
    st.none(),
    st.floats(
        min_value=0.0,
        max_value=2.0 * PASS_THRESHOLD,
        allow_nan=False,
        allow_infinity=False,
    ),
)
_scale_err_values = st.one_of(
    st.none(),
    st.floats(
        min_value=0.0,
        max_value=4.0 * SCALE_TOLERANCE,
        allow_nan=False,
        allow_infinity=False,
    ),
)


# Feature: map-match-eval-harness, Property 6: 单帧通过判定
@settings(max_examples=200)
@given(error_dist=_error_dist_values, scale_err=_scale_err_values)
def test_is_pass_iff_within_thresholds(error_dist, scale_err):
    """is_pass 当且仅当两项均已定义且各自落在阈值内。

    **Validates: Requirements 6.3**
    """
    expected = (
        error_dist is not None
        and scale_err is not None
        and error_dist <= PASS_THRESHOLD
        and scale_err <= SCALE_TOLERANCE
    )
    assert is_pass(error_dist, scale_err) is expected


# Feature: map-match-eval-harness, Property 6: 单帧通过判定
@settings(max_examples=200)
@given(
    measured=st.floats(
        min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False
    ),
    reference=st.floats(
        min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False
    ),
    error_dist=st.floats(
        min_value=0.0,
        max_value=PASS_THRESHOLD,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_scale_error_undefined_forces_fail(measured, reference, error_dist):
    """measured ≤ 0 或 reference ≤ 0 → Scale_Error 未定义且判定失败。

    **Validates: Requirements 6.5**
    """
    scale_err = scale_error(measured, reference)

    if measured <= 0 or reference <= 0:
        # scale 未定义：返回 None，且即使 error_dist 在阈值内也判失败。
        assert scale_err is None
        assert is_pass(error_dist, scale_err) is False
    else:
        # 两者皆正：scale_err 已定义且非负。
        assert scale_err is not None
        assert scale_err >= 0.0
