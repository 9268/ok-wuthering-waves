"""Match_Runner：单帧匹配执行（Requirement 4, 6.1, 7.3）。

负责缓存加载、引擎调用与结果复用。测试机只读预置 Feature_Cache，缺失则跳过并
告警，绝不在测试机上触发全图特征提取。

执行 :meth:`MatchRunner.run_case` 的三段式逻辑（design.md > Components > Match_Runner）：

1. **结果复用（Req 7.3）**：若 Result_Store 已存在 ``(case_id, param_set_name)``
   的单帧结果且未要求 ``recompute``，直接由其重建一个 *MatchOutput 视图* 返回，
   不重新匹配。匹配为确定性计算，相同参数得到相同结果，故可安全复用。

   该视图为 **部分视图**：``FrameMetric`` 仅存 ``success`` / ``confidence`` /
   ``match_count`` / ``inlier_count`` / ``elapsed_ms``，并 **不** 持久化几何信息
   （``center`` / ``corners`` / ``game_center`` / ``H`` / ``map_scale``）。重建的
   ``MatchOutput`` 因此这些几何字段为空（``map_scale`` 为默认 ``0.0``）。该部分
   视图足以支撑评估聚合（成功率 / 置信度 / 匹配点数等）；若调用方需要完整几何，
   应以 ``recompute=True`` 触发重算。

2. **缓存定位（Req 4.3, 4.4）**：否则按 :func:`features.cache_path` 定位该
   ``(Param_Set_Name, map_id, algo)`` 的 Feature_Cache；缺失则返回 ``None`` 并把
   一条含缺失路径的告警追加到 :attr:`MatchRunner.warnings`，**绝不** 在测试机上
   触发全图特征提取。

3. **引擎构造（Req 4.1, 6.1）**：将 Param_Set 缓存目录
   （``{caches_dir}/{Param_Set_Name}``）作为 ``assets_dir`` 传入引擎，且仅当
   ``{map_id}_{algo}.npz`` 就位时才构造引擎，使引擎加载缓存而非从大地图原图提取。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

from .errors import HarnessError
from .features import cache_path
from .params import SIFT, SURF

if TYPE_CHECKING:  # 仅类型检查期导入，避免在运行时引入重依赖。
    from .case_loader import TestCase
    from .map_registry import MapEntry, MapRegistry
    from .params import ParamSet
    from .result_store import FrameMetric, ResultStore


class MatchRunner:
    """单帧匹配执行子系统：缓存加载 + 引擎调用 + 结果复用（Requirement 4, 6.1, 7.3）。

    Args:
        caches_dir: Feature_Cache 根目录（如 ``eval/caches``）。缓存按 Param_Set
            分目录：``{caches_dir}/{Param_Set_Name}/{map_id}_{algo}.npz``。
        map_registry: :class:`MapRegistry` 实例，用于发现大地图原图路径与坐标参考
            （引擎构造与 ``game_center`` 换算所需）。可为 ``None``，此时缺少地图信息
            的组合会被跳过并告警。
        result_store: :class:`ResultStore` 实例，用于复用已存的单帧结果（Req 7.3）。
            可为 ``None`` 以禁用复用（每次都执行匹配）。

    Attributes:
        warnings: 处理过程中产生的告警列表（缓存缺失、地图缺失、引擎构造失败等），
            其中缓存缺失告警包含缺失的缓存路径（Req 4.3）。
    """

    def __init__(
        self,
        caches_dir: str,
        map_registry: "Optional[MapRegistry]" = None,
        result_store: "Optional[ResultStore]" = None,
    ) -> None:
        self.caches_dir = caches_dir
        self.map_registry = map_registry
        self.result_store = result_store
        self.warnings: List[str] = []
        # 延迟发现的 map_id -> MapEntry 映射；首次需要时填充。
        self._maps: "Optional[dict[str, MapEntry]]" = None
        # 引擎按 (map_id, Param_Set_Name) 缓存复用，避免重复加载 .npz。
        self._engines: dict = {}

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def run_case(
        self,
        case: "TestCase",
        map_id: str,
        param_set: "ParamSet",
        region=None,
        recompute: bool = False,
    ) -> "Optional[object]":
        """对单个 Test_Case 在给定 ``(map_id, Param_Set)`` 下执行匹配。

        Args:
            case: 待匹配的测试用例（需有 ``case_id`` 与 ``image_path``）。
            map_id: 目标大地图标识。
            param_set: 参数集，其 ``name`` 用作缓存分组键与结果主键的一部分，
                其 ``algo`` / ``params`` 决定引擎构造。
            region: 可选的匹配搜索区间 ``(x, y, w, h)``（大地图像素），透传给引擎
                以收窄匹配范围；``None`` 表示全图匹配。
            recompute: 为真时忽略已存结果，强制重新匹配（Req 7.3 的显式重算）。

        Returns:
            ``MatchOutput``（或复用路径返回的部分 *MatchOutput 视图*）；当 Feature_Cache
            缺失、地图未在注册表中发现或引擎构造失败时返回 ``None``（并记录告警）。
        """
        param_set_name = param_set.name
        algo = param_set.algo

        # 1) 结果复用：已存结果且非显式重算 → 重建部分视图返回（Req 7.3）。
        if not recompute and self.result_store is not None:
            metric = self.result_store.get_frame(case.case_id, param_set_name)
            if metric is not None:
                return self._reconstruct_output(metric)

        # 2) 缓存定位：缺失即跳过并告警，绝不全图提取（Req 4.3, 4.4）。
        path = cache_path(self.caches_dir, param_set_name, map_id, algo)
        if not os.path.isfile(path):
            self.warnings.append(
                f"Feature_Cache 缺失，跳过 (map_id={map_id}, "
                f"param_set={param_set_name}, algo={algo})；缺失路径: {path}"
            )
            return None

        # 3) 构造引擎（assets_dir = Param_Set 缓存目录）并匹配（Req 4.1, 6.1）。
        engine = self._get_engine(map_id, param_set)
        if engine is None:
            return None
        return engine.match(case.image_path, region=region)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _reconstruct_output(self, metric: "FrameMetric") -> "object":
        """由已存的 :class:`FrameMetric` 重建一个部分 ``MatchOutput`` 视图（Req 7.3）。

        ``FrameMetric`` 不持久化几何信息，故视图的 ``center`` / ``corners`` /
        ``game_center`` / ``H`` 为 ``None``、``map_scale`` 为默认 ``0.0``。该视图
        足以支撑评估聚合；需要完整几何的调用方应以 ``recompute=True`` 触发重算。
        """
        # 延迟导入：MatchOutput 定义于引擎模块（依赖 cv2），仅在需要时加载。
        from src.match_engine.common import MatchOutput  # noqa: PLC0415

        return MatchOutput(
            success=metric.success,
            match_count=metric.match_count,
            inlier_count=metric.inlier_count,
            confidence=metric.confidence,
            elapsed_ms=metric.elapsed_ms,
        )

    def _get_engine(self, map_id: str, param_set: "ParamSet") -> "Optional[object]":
        """获取（并缓存）某 ``(map_id, Param_Set)`` 的引擎；不可用时告警并返回 ``None``。

        仅当 ``{caches_dir}/{Param_Set_Name}/{map_id}_{algo}.npz`` 就位时才构造引擎，
        从而保证引擎加载缓存而非从大地图原图提取（Req 4.1, 4.4, 6.1）。
        """
        key = (map_id, param_set.name)
        cached = self._engines.get(key)
        if cached is not None:
            return cached

        entry = self._get_map_entry(map_id)
        if entry is None:
            self.warnings.append(
                f"地图 {map_id} 未在注册表中发现，跳过 (param_set={param_set.name})"
            )
            return None

        assets_dir = os.path.join(self.caches_dir, param_set.name)
        npz_path = os.path.join(assets_dir, f"{map_id}_{param_set.algo}.npz")
        # 双重确认缓存就位，确保引擎只加载缓存、绝不回退到全图提取（Req 4.4）。
        if not os.path.isfile(npz_path):
            self.warnings.append(
                f"Feature_Cache 缺失，跳过 (map_id={map_id}, "
                f"param_set={param_set.name}, algo={param_set.algo})；"
                f"缺失路径: {npz_path}"
            )
            return None

        try:
            engine = self._build_engine(entry, param_set, assets_dir)
        except Exception as exc:  # noqa: BLE001 - 记录并跳过，不中断整体评估
            self.warnings.append(
                f"地图 {map_id} 引擎构造失败，跳过 "
                f"(param_set={param_set.name})：{exc}"
            )
            return None

        self._engines[key] = engine
        return engine

    @staticmethod
    def _build_engine(
        map_entry: "MapEntry", param_set: "ParamSet", assets_dir: str
    ) -> "object":
        """按 Param_Set 构造引擎，复用 ``default_engine_factory`` 的参数映射。

        将 Param_Set 缓存目录作为 ``assets_dir`` 传入引擎；由于
        ``{map_id}_{algo}.npz`` 已确认就位，引擎会加载缓存而非提取。构造后显式
        以注册表中的坐标参考覆盖 ``engine.coords``，确保 ``game_center`` 换算可用。
        """
        # 延迟导入引擎（依赖 cv2），保持模块导入轻量。
        from src.match_engine import SiftEngine, SurfEngine  # noqa: PLC0415

        p = param_set.params
        if param_set.algo == SURF:
            engine = SurfEngine(
                map_entry.map_id,
                map_entry.image_path,
                assets_dir,
                hessian=p.hessian,
                octaves=p.octaves,
                layers=p.layers,
                extended=p.extended,
                upright=p.upright,
                grid=p.grid,
                max_per_cell=p.max_per_cell,
                ratio=p.ratio,
                max_dist=p.max_dist,
                coords_path=None,
            )
        elif param_set.algo == SIFT:
            engine = SiftEngine(
                map_entry.map_id,
                map_entry.image_path,
                assets_dir,
                nfeatures=0,
                nOctaveLayers=p.n_octave_layers,
                contrastThreshold=p.contrast_threshold,
                edgeThreshold=p.edge_threshold,
                sigma=p.sigma,
                grid=p.grid,
                max_per_cell=p.max_per_cell,
                ratio=p.ratio,
                coords_path=None,
            )
        else:
            raise HarnessError(
                f"未知算法标识：{param_set.algo!r}（应为 'surf' 或 'sift'）",
                name=str(param_set.algo),
            )

        # 以注册表坐标参考覆盖，确保 game_center 换算稳定可用。
        if getattr(map_entry, "coords", None) is not None:
            engine.coords = map_entry.coords
        return engine

    def _get_map_entry(self, map_id: str) -> "Optional[MapEntry]":
        """惰性发现并返回某 map_id 的注册项；注册表为空或缺失时返回 ``None``。"""
        if self._maps is None:
            if self.map_registry is None:
                self._maps = {}
            else:
                self._maps = self.map_registry.discover()
        return self._maps.get(map_id)


__all__ = ["MatchRunner"]
