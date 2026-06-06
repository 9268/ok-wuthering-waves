"""Match_Evaluator：单帧精度评估（Requirement 6）。

将匹配结果与 Ground_Truth 比对，计算 Error_Distance 与 Scale_Error 并判定
通过 / 失败，记录指标到 Result_Store。

**FrameMetric 的复用（重要）**：本子系统的单帧指标 dataclass ``FrameMetric``
的"权威"定义位于 :mod:`src.eval_harness.result_store`（任务 10.1），其字段与
``frame_results`` 表 schema 完全对齐（``case_id`` / ``param_set_name`` /
``map_id`` / ``success`` / ``error_distance`` / ``scale_error`` / ``confidence`` /
``match_count`` / ``inlier_count`` / ``elapsed_ms``）。design.md 在
Match_Evaluator 一节也列出了同名 dataclass，但为避免重复定义与字段漂移，
evaluator **直接复用** ``result_store.FrameMetric`` 而非再定义一个，并通过
``__all__`` 重新导出，使 ``from src.eval_harness.evaluator import FrameMetric``
仍然可用。

评估语义（design.md > Components > Match_Evaluator，Requirement 6）：

- 通过 :class:`~src.eval_harness.runner.MatchRunner` 取匹配结果
  （``run_case`` 返回 ``MatchOutput | None``）。
- 匹配失败（``None`` 或 ``success`` 为假）→ ``success=False``，
  ``error_distance`` 与 ``scale_error`` 记为未定义（``None``）（Req 6.4）。
- 匹配成功 → 在 **游戏坐标** 下计算 Error_Distance（``match.game_center`` 与
  ``gt.game_xy`` 的欧氏距离），并计算 Scale_Error（Req 6.2）。
- ``map_scale ≤ 0`` 或参考缩放比 ≤ 0 → ``scale_error`` 记为未定义（Req 6.5）；
  此时匹配本身仍成功，故 ``success`` 仍反映 ``match.success``，通过 / 失败
  判定（``is_pass``）在报告聚合阶段由已存字段重算。
- 记录 ``success`` / Error_Distance / Scale_Error / ``confidence`` /
  ``match_count`` / ``inlier_count`` / ``elapsed_ms``（Req 6.6）。
- 若 runner 关联了 Result_Store，则以 ISO-8601 时间戳写入（Req 6.1 持久化路径）。
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from . import metrics
from .result_store import FrameMetric

if TYPE_CHECKING:  # 仅类型检查期导入，避免运行时引入重依赖。
    from .annotation import GroundTruth
    from .case_loader import TestCase
    from .params import ParamSet
    from .runner import MatchRunner


def _coerce_number(value: object, default: float = 0.0) -> float:
    """把可能为 ``None`` 的数值字段规整为 ``float``（缺失记为 ``default``）。"""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    """把可能为 ``None`` 的整数字段规整为 ``int``（缺失记为 ``default``）。"""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class MatchEvaluator:
    """单帧精度评估子系统（Requirement 6）。

    将 :class:`~src.eval_harness.runner.MatchRunner` 的匹配结果与 Ground_Truth
    比对，产出一条 :class:`~src.eval_harness.result_store.FrameMetric`，并在 runner
    关联了 Result_Store 时持久化该指标。
    """

    def evaluate(
        self,
        case: "TestCase",
        gt: "GroundTruth",
        param_set: "ParamSet",
        runner: "MatchRunner",
    ) -> FrameMetric:
        """评估一个具有 Ground_Truth 的 Test_Case 在某 Param_Set 下的单帧精度。

        Args:
            case: 待评估的测试用例（需有 ``case_id``）。
            gt: 该用例的 Ground_Truth（提供 ``map_id`` / ``game_xy`` /
                ``reference_scale``）。
            param_set: 参数集，其 ``name`` 用作结果主键的一部分。
            runner: :class:`MatchRunner`，用于执行匹配并（可选地）提供
                Result_Store 以持久化结果。

        Returns:
            一条 :class:`FrameMetric`（复用 ``result_store.FrameMetric``）。
        """
        param_set_name = param_set.name
        map_id = gt.map_id

        match = runner.run_case(case, map_id, param_set)

        metric = self._build_metric(case.case_id, param_set_name, map_id, gt, match)
        self._persist(runner, metric)
        return metric

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _build_metric(
        case_id: str,
        param_set_name: str,
        map_id: str,
        gt: "GroundTruth",
        match: "Optional[object]",
    ) -> FrameMetric:
        """按 Requirement 6 的判定语义构造 :class:`FrameMetric`。"""
        # 匹配失败：None 或 success 为假 → 误差未定义（Req 6.4）。
        if match is None or not getattr(match, "success", False):
            return FrameMetric(
                case_id=case_id,
                param_set_name=param_set_name,
                map_id=map_id,
                success=False,
                error_distance=None,
                scale_error=None,
                confidence=_coerce_number(getattr(match, "confidence", None)),
                match_count=_coerce_int(getattr(match, "match_count", None)),
                inlier_count=_coerce_int(getattr(match, "inlier_count", None)),
                elapsed_ms=_coerce_number(getattr(match, "elapsed_ms", None)),
            )

        # 匹配成功：在游戏坐标下计算 Error_Distance（Req 6.2）。
        game_center = getattr(match, "game_center", None)
        if game_center is not None:
            error_dist: Optional[float] = metrics.game_distance(
                (float(game_center[0]), float(game_center[1])),
                (float(gt.game_xy[0]), float(gt.game_xy[1])),
            )
        else:
            # 成功但缺少游戏坐标（如复用的部分视图）→ 误差无法度量，记为未定义。
            error_dist = None

        # Scale_Error；measured ≤ 0 或 reference ≤ 0 → 未定义（Req 6.5）。
        scale_err = metrics.scale_error(
            _coerce_number(getattr(match, "map_scale", None)),
            gt.reference_scale,
        )

        return FrameMetric(
            case_id=case_id,
            param_set_name=param_set_name,
            map_id=map_id,
            success=bool(match.success),
            error_distance=error_dist,
            scale_error=scale_err,
            confidence=_coerce_number(getattr(match, "confidence", None)),
            match_count=_coerce_int(getattr(match, "match_count", None)),
            inlier_count=_coerce_int(getattr(match, "inlier_count", None)),
            elapsed_ms=_coerce_number(getattr(match, "elapsed_ms", None)),
        )

    @staticmethod
    def _persist(runner: "MatchRunner", metric: FrameMetric) -> None:
        """若 runner 关联了 Result_Store，则以 ISO-8601 时间戳持久化指标（Req 6.1）。

        store 不存在时静默跳过，使评估在无持久化场景（如单元测试）下仍可用。
        """
        store = getattr(runner, "result_store", None)
        if store is None:
            return
        store.upsert_frame(metric, ts=datetime.now().isoformat())


__all__ = ["MatchEvaluator", "FrameMetric"]
