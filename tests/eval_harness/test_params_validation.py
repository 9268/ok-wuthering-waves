"""Param_Set 校验边界示例测试（Requirement 8.5）。

EDGE_CASE 示例测试：缺字段 / 非法值 / 未知算法时，应抛出携带
Param_Set_Name（或退回算法标识）的 :class:`HarnessError`，并指明具体问题。

Validates: Requirements 8.5
"""

from __future__ import annotations

import pytest

from src.eval_harness.errors import HarnessError
from src.eval_harness.params import ParamSet, SiftParams, SurfParams


def _valid_surf(**overrides) -> SurfParams:
    base = dict(
        hessian=300,
        octaves=4,
        layers=3,
        extended=False,
        upright=False,
        grid=8,
        max_per_cell=5,
        ratio=0.7,
        max_dist=0.3,
    )
    base.update(overrides)
    return SurfParams(**base)


def _valid_sift(**overrides) -> SiftParams:
    base = dict(
        contrast_threshold=0.04,
        edge_threshold=10,
        n_octave_layers=3,
        sigma=1.6,
        grid=8,
        max_per_cell=5,
        ratio=0.7,
    )
    base.update(overrides)
    return SiftParams(**base)


def _valid_surf_dict() -> dict:
    return ParamSet(algo="surf", params=_valid_surf()).to_dict()


# ---------------------------------------------------------------------------
# from_dict: 结构 / 缺字段 / 未知算法
# ---------------------------------------------------------------------------
def test_from_dict_missing_algo_raises():
    with pytest.raises(HarnessError) as exc:
        ParamSet.from_dict({"params": {}})
    assert "algo" in exc.value.message


def test_from_dict_missing_params_raises_with_algo_name():
    with pytest.raises(HarnessError) as exc:
        ParamSet.from_dict({"algo": "surf"})
    assert exc.value.name == "surf"
    assert "params" in exc.value.message


def test_from_dict_missing_required_field_raises_with_algo_name():
    d = _valid_surf_dict()
    del d["params"]["ratio"]
    with pytest.raises(HarnessError) as exc:
        ParamSet.from_dict(d)
    assert exc.value.name == "surf"
    assert "ratio" in exc.value.message


def test_from_dict_unknown_field_raises():
    d = _valid_surf_dict()
    d["params"]["bogus"] = 1
    with pytest.raises(HarnessError) as exc:
        ParamSet.from_dict(d)
    assert exc.value.name == "surf"
    assert "bogus" in exc.value.message


def test_from_dict_unknown_algo_raises():
    with pytest.raises(HarnessError) as exc:
        ParamSet.from_dict({"algo": "orb", "params": {}})
    assert exc.value.name == "orb"
    assert "orb" in exc.value.message


def test_from_dict_non_dict_raises():
    with pytest.raises(HarnessError) as exc:
        ParamSet.from_dict(["not", "a", "dict"])
    assert "字典" in exc.value.message


# ---------------------------------------------------------------------------
# validate: 非法值，错误应携带 Param_Set_Name
# ---------------------------------------------------------------------------
def test_validate_ratio_zero_raises_with_name():
    ps = ParamSet(algo="surf", params=_valid_surf(ratio=0))
    with pytest.raises(HarnessError) as exc:
        ps.validate()
    assert exc.value.name == ps.name
    assert "ratio" in exc.value.message


def test_validate_ratio_above_one_raises_with_name():
    ps = ParamSet(algo="surf", params=_valid_surf(ratio=1.5))
    with pytest.raises(HarnessError) as exc:
        ps.validate()
    assert exc.value.name == ps.name
    assert "ratio" in exc.value.message


def test_validate_negative_hessian_raises_with_name():
    ps = ParamSet(algo="surf", params=_valid_surf(hessian=-10))
    with pytest.raises(HarnessError) as exc:
        ps.validate()
    assert exc.value.name == ps.name
    assert "hessian" in exc.value.message


def test_validate_non_positive_sigma_raises_with_name():
    ps = ParamSet(algo="sift", params=_valid_sift(sigma=0))
    with pytest.raises(HarnessError) as exc:
        ps.validate()
    assert exc.value.name == ps.name
    assert "sigma" in exc.value.message


def test_validate_unknown_algo_raises_falls_back_to_algo_name():
    ps = ParamSet(algo="orb", params=_valid_surf())
    with pytest.raises(HarnessError) as exc:
        ps.validate()
    assert exc.value.name == "orb"
    assert "orb" in exc.value.message


def test_validate_algo_param_type_mismatch_raises():
    # algo 与 params 类型不匹配：错误退回算法标识。
    ps = ParamSet(algo="sift", params=_valid_surf())
    with pytest.raises(HarnessError) as exc:
        ps.validate()
    assert exc.value.name == "sift"
    assert "不匹配" in exc.value.message
