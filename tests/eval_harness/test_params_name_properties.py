"""Property 9 的属性测试：Param_Set_Name 字符集（Requirement 8.3）。

仅实现设计文档中编号的 Property 9，不引入额外属性。
"""

from __future__ import annotations

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from src.eval_harness.params import (
    NAME_PATTERN,
    ParamSet,
    SiftParams,
    SurfParams,
)

# 合法取值生成策略，对齐 ParamSet.validate() 规则：
# 正整数、ratio ∈ (0, 1]、正浮点。
_positive_int = st.integers(min_value=1, max_value=10_000)
_ratio = st.floats(
    min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False
)
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
    ps = ParamSet(algo="surf", params=params)
    ps.validate()  # 仅保留满足校验规则的合法参数集
    return ps


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
    ps = ParamSet(algo="sift", params=params)
    ps.validate()  # 仅保留满足校验规则的合法参数集
    return ps


_valid_param_sets = st.one_of(_surf_param_sets(), _sift_param_sets())


# Feature: map-match-eval-harness, Property 9: Param_Set_Name 字符集
# Validates: Requirements 8.3
@settings(max_examples=200)
@given(ps=_valid_param_sets)
def test_param_set_name_charset(ps: ParamSet) -> None:
    """任意合法 Param_Set 的名称仅含小写字母、数字与下划线（^[a-z0-9_]+$）。"""
    name = ps.name
    assert re.match(r"^[a-z0-9_]+$", name) is not None, name
    # 同时校验模块导出的 NAME_PATTERN 与该约束一致。
    assert NAME_PATTERN.match(name) is not None, name
