"""Property 14: 聚合统计不变量（report.aggregate）。

覆盖 Requirements 9.7、10.3：

- 当至少存在一个已定义（非 ``None``）值时，``minimum <= mean <= maximum``，
  且 ``count`` 等于非 ``None`` 值的个数，``minimum`` / ``maximum`` 分别等于
  已定义值的最小 / 最大值（10.3）。
- 当不存在任何已定义值（空输入或全 ``None``）时，``count == 0`` 且
  ``mean`` / ``maximum`` / ``minimum`` 均为 ``None``（9.7 优雅降级）。

属性测试使用 hypothesis，每条属性至少运行 100 个生成样例。
"""

from __future__ import annotations

import math
from typing import List, Optional

from hypothesis import given, settings
from hypothesis import strategies as st

from src.eval_harness.report import aggregate

# 有界浮点：限制量级避免浮点溢出与精度退化（NaN / inf 已排除）。
_value = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
# 可选值：混合 None 与有界浮点，None 表示未定义项。
_optional = st.one_of(st.none(), _value)
# 值列表：允许为空，并有机会生成全 None / 全空。
_values = st.lists(_optional, min_size=0, max_size=50)


# Feature: map-match-eval-harness, Property 14: 聚合统计不变量
@settings(max_examples=200)
@given(values=_values)
def test_aggregate_invariants(values: List[Optional[float]]):
    """聚合统计满足 mean ∈ [min, max] 等不变量。

    **Validates: Requirements 9.7, 10.3**
    """
    stats = aggregate(values)
    defined = [v for v in values if v is not None]

    if defined:
        # count 等于非 None 值的个数。
        assert stats.count == len(defined)
        # minimum / maximum 与已定义值集合一致。
        assert stats.minimum == min(defined)
        assert stats.maximum == max(defined)
        # 平均值落在 [minimum, maximum]（含小浮点容差）。
        assert stats.mean is not None
        assert (
            stats.minimum <= stats.mean
            or math.isclose(stats.mean, stats.minimum, rel_tol=1e-9, abs_tol=1e-9)
        )
        assert (
            stats.mean <= stats.maximum
            or math.isclose(stats.mean, stats.maximum, rel_tol=1e-9, abs_tol=1e-9)
        )
    else:
        # 无任何已定义值：count 为 0，统计量均为 None。
        assert stats.count == 0
        assert stats.mean is None
        assert stats.maximum is None
        assert stats.minimum is None
