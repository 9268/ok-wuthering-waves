"""Param_Set 与 Param_Set_Name 的定义、命名、序列化与校验（Requirement 8）。

仅描述特征提取参数（SURF / SIFT）与匹配参数，不含任何连续定位后处理参数。
纯逻辑模块（无 I/O），适合属性测试。

命名约定（Req 8.2, 8.3，决策 8）：

- SURF::

    surf_h{hessian}_o{octaves}_l{layers}_g{grid}_mpc{max_per_cell}
        _r{round(ratio*100)}_md{round(max_dist*100)}

- SIFT::

    sift_ct{round(contrast_threshold*1000)}_et{edge_threshold}
        _ol{n_octave_layers}_s{round(sigma*10)}_g{grid}_mpc{max_per_cell}
        _r{round(ratio*100)}

Param_Set_Name 仅由小写字母、数字与下划线组成（匹配 ``^[a-z0-9_]+$``），
可直接用作目录名与数据库键。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from typing import Union

from .errors import HarnessError

# 合法 Param_Set_Name 字符集（Req 8.3）。
NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")

SURF = "surf"
SIFT = "sift"


@dataclass(frozen=True)
class SurfParams:
    """SURF 特征提取参数与匹配参数。

    Attributes:
        hessian: SURF Hessian 阈值（正整数）。
        octaves: 金字塔层组数（正整数）。
        layers: 每组层数（正整数）。
        extended: 是否使用扩展描述子（128 维）。
        upright: 是否使用 U-SURF（不计算主方向）。
        grid: 网格采样的网格边长（正整数）。
        max_per_cell: 每个网格单元保留的最大特征点数（正整数）。
        ratio: Lowe 比率测试阈值（0 < ratio <= 1）。
        max_dist: FLANN 匹配距离上限（正浮点）。
    """

    hessian: int
    octaves: int
    layers: int
    extended: bool
    upright: bool
    grid: int
    max_per_cell: int
    ratio: float
    max_dist: float


@dataclass(frozen=True)
class SiftParams:
    """SIFT 特征提取参数与匹配参数。

    Attributes:
        contrast_threshold: 对比度阈值（正浮点）。
        edge_threshold: 边缘阈值（正整数）。
        n_octave_layers: 每组层数（正整数）。
        sigma: 高斯尺度（正浮点）。
        grid: 网格采样的网格边长（正整数）。
        max_per_cell: 每个网格单元保留的最大特征点数（正整数）。
        ratio: Lowe 比率测试阈值（0 < ratio <= 1）。
    """

    contrast_threshold: float
    edge_threshold: int
    n_octave_layers: int
    sigma: float
    grid: int
    max_per_cell: int
    ratio: float


# 每种算法对应的参数 dataclass。
_PARAM_TYPES = {SURF: SurfParams, SIFT: SiftParams}


@dataclass(frozen=True)
class ParamSet:
    """一个具名参数集，仅含特征提取与匹配参数（Req 8.1）。

    Attributes:
        algo: 算法标识，``'surf'`` 或 ``'sift'``。
        params: 对应算法的参数对象（:class:`SurfParams` 或 :class:`SiftParams`）。
    """

    algo: str
    params: Union[SurfParams, SiftParams]

    @property
    def name(self) -> str:
        """计算 Param_Set_Name（Req 8.2）。

        Returns:
            该参数集的具名标识。

        Raises:
            HarnessError: 当 ``algo`` 非法或与 ``params`` 类型不匹配时。
        """
        p = self.params
        if self.algo == SURF and isinstance(p, SurfParams):
            return (
                f"surf_h{p.hessian}_o{p.octaves}_l{p.layers}"
                f"_g{p.grid}_mpc{p.max_per_cell}"
                f"_r{round(p.ratio * 100)}_md{round(p.max_dist * 100)}"
            )
        if self.algo == SIFT and isinstance(p, SiftParams):
            return (
                f"sift_ct{round(p.contrast_threshold * 1000)}"
                f"_et{p.edge_threshold}_ol{p.n_octave_layers}"
                f"_s{round(p.sigma * 10)}"
                f"_g{p.grid}_mpc{p.max_per_cell}"
                f"_r{round(p.ratio * 100)}"
            )
        raise HarnessError(
            f"算法标识与参数类型不匹配：algo={self.algo!r}, "
            f"params={type(p).__name__}",
            name=str(self.algo),
        )

    def to_dict(self) -> dict:
        """序列化为结构化字典（Req 8.1, 8.4）。

        Returns:
            形如 ``{"algo": <algo>, "params": {<字段...>}}`` 的字典。
        """
        return {"algo": self.algo, "params": asdict(self.params)}

    @staticmethod
    def from_dict(d: dict) -> "ParamSet":
        """从字典反序列化为 :class:`ParamSet`（Req 8.4）。

        Args:
            d: ``to_dict`` 产出的字典结构。

        Returns:
            重建的参数集。

        Raises:
            HarnessError: 当结构非法、算法未知或字段缺失 / 多余时。
        """
        if not isinstance(d, dict):
            raise HarnessError(f"Param_Set 定义必须为字典，得到 {type(d).__name__}")
        if "algo" not in d:
            raise HarnessError("Param_Set 定义缺少必需字段：algo")
        algo = d["algo"]
        param_type = _PARAM_TYPES.get(algo)
        if param_type is None:
            raise HarnessError(
                f"未知算法标识：{algo!r}（应为 'surf' 或 'sift'）",
                name=str(algo),
            )
        raw = d.get("params")
        if not isinstance(raw, dict):
            raise HarnessError(
                "Param_Set 定义缺少必需字段：params（应为字典）",
                name=str(algo),
            )
        expected = {f.name for f in fields(param_type)}
        missing = expected - raw.keys()
        if missing:
            raise HarnessError(
                f"Param_Set 缺少必需字段：{sorted(missing)}",
                name=str(algo),
            )
        extra = raw.keys() - expected
        if extra:
            raise HarnessError(
                f"Param_Set 含未知字段：{sorted(extra)}",
                name=str(algo),
            )
        params = param_type(**{k: raw[k] for k in expected})
        param_set = ParamSet(algo=algo, params=params)
        param_set.validate()
        return param_set

    def validate(self) -> None:
        """校验参数集的完整性与取值合法性（Req 8.5）。

        缺字段 / 非法值时抛出 :class:`HarnessError`，并在错误中携带
        Param_Set_Name（尽力计算）与具体问题。

        Raises:
            HarnessError: 当算法非法、类型不匹配或字段取值无效时。
        """
        p = self.params
        if self.algo not in _PARAM_TYPES:
            raise HarnessError(
                f"未知算法标识：{self.algo!r}（应为 'surf' 或 'sift'）",
                name=str(self.algo),
            )
        if not isinstance(p, _PARAM_TYPES[self.algo]):
            raise HarnessError(
                f"算法标识与参数类型不匹配：algo={self.algo!r}, "
                f"params={type(p).__name__}",
                name=str(self.algo),
            )

        name = self._safe_name()

        if self.algo == SURF:
            self._require_positive_int(p.hessian, "hessian", name)
            self._require_positive_int(p.octaves, "octaves", name)
            self._require_positive_int(p.layers, "layers", name)
            self._require_bool(p.extended, "extended", name)
            self._require_bool(p.upright, "upright", name)
            self._require_positive_int(p.grid, "grid", name)
            self._require_positive_int(p.max_per_cell, "max_per_cell", name)
            self._require_ratio(p.ratio, "ratio", name)
            self._require_positive_float(p.max_dist, "max_dist", name)
        else:  # SIFT
            self._require_positive_float(
                p.contrast_threshold, "contrast_threshold", name
            )
            self._require_positive_int(p.edge_threshold, "edge_threshold", name)
            self._require_positive_int(p.n_octave_layers, "n_octave_layers", name)
            self._require_positive_float(p.sigma, "sigma", name)
            self._require_positive_int(p.grid, "grid", name)
            self._require_positive_int(p.max_per_cell, "max_per_cell", name)
            self._require_ratio(p.ratio, "ratio", name)

    # ------------------------------------------------------------------
    # 内部校验辅助
    # ------------------------------------------------------------------
    def _safe_name(self) -> str:
        """尽力计算 Param_Set_Name，失败时退回到算法标识，供错误消息使用。"""
        try:
            return self.name
        except (HarnessError, TypeError, ValueError):
            return str(self.algo)

    @staticmethod
    def _require_positive_int(value: object, field: str, name: str) -> None:
        # bool 是 int 的子类，需显式排除以免把 True/False 当作合法整数。
        if isinstance(value, bool) or not isinstance(value, int):
            raise HarnessError(
                f"字段 {field} 必须为整数，得到 {value!r}", name=name
            )
        if value <= 0:
            raise HarnessError(
                f"字段 {field} 必须为正整数，得到 {value!r}", name=name
            )

    @staticmethod
    def _require_positive_float(value: object, field: str, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HarnessError(
                f"字段 {field} 必须为数值，得到 {value!r}", name=name
            )
        if not (value > 0):
            raise HarnessError(
                f"字段 {field} 必须为正数，得到 {value!r}", name=name
            )

    @staticmethod
    def _require_ratio(value: object, field: str, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HarnessError(
                f"字段 {field} 必须为数值，得到 {value!r}", name=name
            )
        if not (0 < value <= 1):
            raise HarnessError(
                f"字段 {field} 必须落在区间 (0, 1]，得到 {value!r}", name=name
            )

    @staticmethod
    def _require_bool(value: object, field: str, name: str) -> None:
        if not isinstance(value, bool):
            raise HarnessError(
                f"字段 {field} 必须为布尔值，得到 {value!r}", name=name
            )


__all__ = ["SurfParams", "SiftParams", "ParamSet", "NAME_PATTERN", "SURF", "SIFT"]
