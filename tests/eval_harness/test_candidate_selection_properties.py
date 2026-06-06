"""Property 2: 标注候选选取取最高且高于阈值（annotation.select_candidate）。

覆盖 Requirements 2.2、2.6：

- 当存在 ``confidence`` 严格大于阈值（默认 0.9）的候选时，
  ``select_candidate`` 返回这些候选中 ``confidence`` 最高的一项，
  且返回值必为输入候选之一（2.2）。
- 当不存在任何 ``confidence > 阈值`` 的候选时，``select_candidate`` 返回
  ``None``（无候选），调用方据此记为待人工处理（2.6）。

属性测试使用 hypothesis，每条属性至少运行 100 个生成样例。
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.eval_harness.annotation import (
    CONFIDENCE_THRESHOLD,
    Candidate,
    select_candidate,
)

# confidence 取 [0, 1]，允许恰好等于 0.9 以验证“严格大于”边界排除该值。
_confidence = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
_map_scale = st.floats(
    min_value=-1.0, max_value=5.0, allow_nan=False, allow_infinity=False
)
_candidate = st.builds(
    Candidate,
    map_id=st.text(min_size=1, max_size=8),
    confidence=_confidence,
    map_scale=_map_scale,
)
_candidates = st.lists(_candidate, max_size=10)


# Feature: map-match-eval-harness, Property 2: 标注候选选取取最高且高于阈值
@settings(max_examples=200)
@given(candidates=_candidates)
def test_select_candidate_picks_highest_above_threshold(candidates):
    """选取 confidence 最高且 > 阈值的候选；无满足者则返回 None。

    **Validates: Requirements 2.2, 2.6**
    """
    above = [c for c in candidates if c.confidence > CONFIDENCE_THRESHOLD]
    result = select_candidate(candidates)

    if above:
        assert result is not None
        # 返回值必为输入候选之一。
        assert result in candidates
        # 其 confidence 等于所有 > 阈值候选中的最大值（处理并列：仅校验 confidence）。
        assert result.confidence == max(c.confidence for c in above)
        assert result.confidence > CONFIDENCE_THRESHOLD
    else:
        assert result is None


# Feature: map-match-eval-harness, Property 2: 标注候选选取取最高且高于阈值
@settings(max_examples=200)
@given(
    candidates=st.lists(
        st.builds(
            Candidate,
            map_id=st.text(min_size=1, max_size=8),
            # 强制全部 <= 0.9，覆盖“无候选”分支。
            confidence=st.floats(
                min_value=0.0, max_value=0.9, allow_nan=False, allow_infinity=False
            ),
            map_scale=_map_scale,
        ),
        max_size=10,
    )
)
def test_select_candidate_returns_none_when_all_below_threshold(candidates):
    """所有候选 confidence <= 0.9 时返回 None（含恰好 0.9 的边界）。

    **Validates: Requirements 2.2, 2.6**
    """
    assert select_candidate(candidates) is None
