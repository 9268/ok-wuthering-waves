"""Property 8：Param_Set 序列化往返一致性（Requirement 8.1, 8.2, 8.4）。

仅实现设计文档中编号的 Property 8：
``ParamSet.from_dict(ps.to_dict()) == ps`` 且 ``from_dict(...).name == ps.name``。
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.eval_harness.params import ParamSet, SiftParams, SurfParams

# 正整数：约束到合理上界以保持示例可读且生成高效。
_positive_int = st.integers(min_value=1, max_value=10_000)
# Lowe 比率：0 < ratio <= 1。
_ratio = st.floats(
    min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False
)
# 正浮点：用于 max_dist / contrast_threshold / sigma。
_positive_float = st.floats(
    min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False
)


@st.composite
def _surf_param_sets(draw: st.DrawFn) -> ParamSet:
    params = SurfParams(
        hessian=draw(_positive_int),
        octaves=draw(_positive_int),
        layers=draw(_positive_int),
        extended=draw(st.booleans()),
        upright=draw(st.booleans()),
        grid=draw(_positive_int),
        max_per_cell=draw(_positive_int),
        ratio=draw(_ratio),
        max_dist=draw(_positive_float),
    )
    return ParamSet(algo="surf", params=params)


@st.composite
def _sift_param_sets(draw: st.DrawFn) -> ParamSet:
    params = SiftParams(
        contrast_threshold=draw(_positive_float),
        edge_threshold=draw(_positive_int),
        n_octave_layers=draw(_positive_int),
        sigma=draw(_positive_float),
        grid=draw(_positive_int),
        max_per_cell=draw(_positive_int),
        ratio=draw(_ratio),
    )
    return ParamSet(algo="sift", params=params)


# 生成 SURF 与 SIFT 两种有效参数集。
_param_sets = st.one_of(_surf_param_sets(), _sift_param_sets())


# Feature: map-match-eval-harness, Property 8: Param_Set 序列化往返一致
@settings(max_examples=200)
@given(ps=_param_sets)
def test_param_set_serialization_roundtrip(ps: ParamSet) -> None:
    """to_dict → from_dict 等价于原参数集，且 name 不变。

    **Validates: Requirements 8.1, 8.2, 8.4**
    """
    restored = ParamSet.from_dict(ps.to_dict())
    assert restored == ps
    assert restored.name == ps.name
