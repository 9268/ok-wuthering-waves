"""Property 15: 完成率与通过率不变量（report.completion_rate / pass_rate /
aggregate_frame_metrics）。

覆盖 Requirement 10.2：以具有 Ground_Truth 的用例数为分母计算完成率与
通过率，且因"通过蕴含成功"，恒有

    0 <= pass_rate <= completion_rate <= 1，

当分母（GT 用例数）为 0 时两率均定义为 0.0。

本文件提供两种互补的验证路径：

- 直接路径（(a)）：直接对 ``completion_rate`` / ``pass_rate`` 传入满足
  ``0 <= pass_count <= success_count <= gt_count`` 的整数计数。
- 指标路径（(b)）：构造一组 :class:`FrameMetric`，保证"通过蕴含成功"
  （凡通过的指标其 ``success`` 必为真），再经 ``aggregate_frame_metrics``
  聚合后验证同一不变量。

属性测试使用 hypothesis，每条属性至少运行 100 个生成样例。
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.eval_harness.metrics import PASS_THRESHOLD, SCALE_TOLERANCE, is_pass
from src.eval_harness.report import (
    aggregate_frame_metrics,
    completion_rate,
    pass_rate,
)
from src.eval_harness.result_store import FrameMetric


# --- (a) 直接路径：counts 满足 pass <= success <= gt -----------------------


@st.composite
def _ordered_counts(draw):
    """生成满足 ``0 <= pass_count <= success_count <= gt_count`` 的整数计数。"""
    gt_count = draw(st.integers(min_value=0, max_value=500))
    success_count = draw(st.integers(min_value=0, max_value=gt_count))
    pass_count = draw(st.integers(min_value=0, max_value=success_count))
    return gt_count, success_count, pass_count


# Feature: map-match-eval-harness, Property 15: 完成率与通过率不变量
@settings(max_examples=200)
@given(counts=_ordered_counts())
def test_rate_invariant_direct(counts):
    """0 <= pass_rate <= completion_rate <= 1；gt_count==0 时两率为 0.0。

    **Validates: Requirements 10.2**
    """
    gt_count, success_count, pass_count = counts
    cr = completion_rate(success_count, gt_count)
    pr = pass_rate(pass_count, gt_count)

    if gt_count == 0:
        # 无可评估用例：分母为 0，两率均定义为 0.0（避免除零）。
        assert cr == 0.0
        assert pr == 0.0
    else:
        assert 0.0 <= pr <= cr <= 1.0


# --- (b) 指标路径：构造 FrameMetric 列表后聚合 -----------------------------

# 安全落在"通过"区间内的取值（error_distance / scale_error 均在阈值内）。
_passing_error = st.floats(
    min_value=0.0, max_value=PASS_THRESHOLD, allow_nan=False, allow_infinity=False
)
_passing_scale = st.floats(
    min_value=0.0, max_value=SCALE_TOLERANCE, allow_nan=False, allow_infinity=False
)
# 落在阈值之外的取值（用于构造"不通过"的指标）。
_failing_error = st.floats(
    min_value=PASS_THRESHOLD + 1.0,
    max_value=4.0 * PASS_THRESHOLD,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def _frame_metric(draw, index: int):
    """生成单个 :class:`FrameMetric`，保证"通过蕴含成功"。

    若该指标落在通过区间（error_distance / scale_error 均在阈值内），
    则强制 ``success=True``，从而维持 pass ⇒ success 的不变量。
    """
    will_pass = draw(st.booleans())
    if will_pass:
        error_distance = draw(_passing_error)
        scale_error = draw(_passing_scale)
        success = True  # 通过蕴含成功
    else:
        # 不通过：误差超阈值，或指标未定义（None）；success 自由取值。
        defined = draw(st.booleans())
        if defined:
            error_distance = draw(_failing_error)
            scale_error = draw(_passing_scale)
        else:
            error_distance = None
            scale_error = None
        success = draw(st.booleans())

    return FrameMetric(
        case_id=f"case-{index}",
        param_set_name="ps",
        map_id=draw(st.one_of(st.none(), st.sampled_from(["m1", "m2"]))),
        success=success,
        error_distance=error_distance,
        scale_error=scale_error,
        confidence=draw(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)),
        match_count=draw(st.integers(0, 100)),
        inlier_count=draw(st.integers(0, 100)),
        elapsed_ms=draw(st.floats(0.0, 1000.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def _frame_metrics(draw):
    """生成一组（含空列表）满足"通过蕴含成功"的 FrameMetric。"""
    size = draw(st.integers(min_value=0, max_value=30))
    return [draw(_frame_metric(i)) for i in range(size)]


# Feature: map-match-eval-harness, Property 15: 完成率与通过率不变量
@settings(max_examples=200)
@given(metrics=_frame_metrics())
def test_rate_invariant_metric_based(metrics):
    """聚合后 0 <= pass_rate <= completion_rate <= 1 且 pass_count <= success_count。

    **Validates: Requirements 10.2**
    """
    summary = aggregate_frame_metrics(metrics)

    # 分母即 GT 用例数（列表长度）。
    assert summary.gt_count == len(metrics)
    # 通过蕴含成功：通过数不超过成功数，成功数不超过 GT 用例数。
    assert 0 <= summary.pass_count <= summary.success_count <= summary.gt_count

    # 交叉校验 pass_count 与 is_pass 判定一致。
    expected_pass = sum(
        1 for m in metrics if is_pass(m.error_distance, m.scale_error)
    )
    assert summary.pass_count == expected_pass

    if summary.gt_count == 0:
        assert summary.completion_rate == 0.0
        assert summary.pass_rate == 0.0
    else:
        assert 0.0 <= summary.pass_rate <= summary.completion_rate <= 1.0
