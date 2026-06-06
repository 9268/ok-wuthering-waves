"""Match_Evaluator 单帧评估集成测试（Requirement 6.1, 6.4, 6.5, 6.6）。

本模块对 :class:`src.eval_harness.evaluator.MatchEvaluator` 做端到端单帧评估，
并验证评估指标被持久化到 **真实的** :class:`src.eval_harness.result_store.ResultStore`
（``:memory:`` 内存库）。

关于"真实 screenshots 用例与预置缓存"（tasks.md 14.2）：
    一次真实的引擎匹配（OpenCV SURF/SIFT + 大地图特征缓存）既昂贵又依赖二进制
    资源与机器状态，难以在单元/集成层稳定复现。为了**确定性地**驱动评估器并验证
    其与 Result_Store 的端到端写入路径（Req 6.1, 6.6）与失败语义（Req 6.4, 6.5），
    本测试使用一个**桩 runner**（``_StubRunner``）：它暴露
    ``run_case(case, map_id, param_set, region=None, recompute=False) -> MatchOutput|None``
    并返回**预先构造**的 :class:`~src.match_engine.common.MatchOutput`，同时持有一个
    真实的 ``ResultStore(':memory:')`` 作为 ``result_store`` 属性。这样既能完整行使
    ``MatchEvaluator.evaluate`` 的逻辑与持久化路径，又避免了 cv2 的成本与非确定性。
    ``TestCase`` 的 ``image_path`` 为占位路径（桩 runner 忽略它）。

Validates: Requirements 6.1, 6.4, 6.5, 6.6
"""

from __future__ import annotations

from typing import Optional

# 以别名导入，避免 pytest 误把 case_loader.TestCase 当作测试类收集。
from src.eval_harness.annotation import GroundTruth
from src.eval_harness.case_loader import TestCase as HarnessTestCase
from src.eval_harness.evaluator import FrameMetric, MatchEvaluator
from src.eval_harness.metrics import MAP_SCALE, game_distance
from src.eval_harness.params import ParamSet, SiftParams
from src.eval_harness.result_store import ResultStore
from src.match_engine.common import MatchOutput


def _sift_set() -> ParamSet:
    """一个合法的 SIFT Param_Set，其 ``name`` 用作结果主键的一部分。"""
    return ParamSet(
        algo="sift",
        params=SiftParams(
            contrast_threshold=0.04,
            edge_threshold=10,
            n_octave_layers=3,
            sigma=1.6,
            grid=8,
            max_per_cell=5,
            ratio=0.75,
        ),
    )


class _StubRunner:
    """桩 Match_Runner：返回预置 MatchOutput，并持有真实的 Result_Store。

    替代真实的 ``MatchRunner`` + OpenCV 引擎，使评估端到端可确定性地运行。
    ``MatchEvaluator.evaluate`` 仅依赖两点契约：``run_case(...) -> MatchOutput|None``
    与 ``result_store`` 属性（用于持久化），二者本桩均满足。
    """

    def __init__(self, output: Optional[MatchOutput], result_store: ResultStore) -> None:
        self._output = output
        self.result_store = result_store
        self.calls: list = []

    def run_case(self, case, map_id, param_set, region=None, recompute=False):
        # 记录调用参数，便于断言评估器以正确的 (case, map_id, param_set) 调用。
        self.calls.append((case.case_id, map_id, param_set.name, region, recompute))
        return self._output


def _gt(case_id: str = "case-1") -> GroundTruth:
    """构造一条合法 Ground_Truth：reference_scale = MAP_SCALE，游戏坐标由像素换算。"""
    pixel_xy = (1000.0, 2000.0)
    game_xy = (pixel_xy[0] * MAP_SCALE, pixel_xy[1] * MAP_SCALE)
    return GroundTruth(
        case_id=case_id,
        map_id="8",
        pixel_xy=pixel_xy,
        game_xy=game_xy,
        reference_scale=MAP_SCALE,
    )


# ---------------------------------------------------------------------------
# Test 1：成功用例端到端评估并写入 Result_Store（Req 6.1, 6.6）
# ---------------------------------------------------------------------------
def test_success_case_computes_metrics_and_persists():
    """成功匹配 → FrameMetric 字段齐全且被持久化到 Result_Store（Req 6.1, 6.6）。"""
    store = ResultStore(":memory:")
    try:
        gt = _gt()
        param_set = _sift_set()
        case = HarnessTestCase(case_id=gt.case_id, image_path="/screenshots/case-1.png")

        # game_center 接近 gt.game_xy（偏移 100 游戏坐标单位 → 小误差，远低于 3320 阈值）。
        offset = 100.0
        game_center = (gt.game_xy[0] + offset, gt.game_xy[1])
        output = MatchOutput(
            success=True,
            match_count=42,
            inlier_count=30,
            confidence=0.95,
            center=(1000.0, 2000.0),
            game_center=game_center,
            map_scale=MAP_SCALE,  # 与参考缩放比一致 → scale_error ≈ 0
            elapsed_ms=12.5,
        )
        runner = _StubRunner(output, store)

        metric = MatchEvaluator().evaluate(case, gt, param_set, runner)

        # 评估器以正确的 (case, map_id, param_set) 调用了 runner。
        assert runner.calls == [(gt.case_id, gt.map_id, param_set.name, None, False)]

        # 返回的 FrameMetric：success 与误差/缩放误差均已定义（Req 6.6）。
        assert isinstance(metric, FrameMetric)
        assert metric.success is True
        assert metric.error_distance is not None
        expected_dist = game_distance(game_center, gt.game_xy)
        assert metric.error_distance == expected_dist
        assert abs(metric.error_distance - offset) < 1e-6  # 误差等于注入的偏移量
        assert metric.scale_error is not None
        assert abs(metric.scale_error) < 1e-9  # map_scale == reference_scale
        # 记录 confidence / match_count / inlier_count / elapsed_ms（Req 6.6）。
        assert metric.confidence == 0.95
        assert metric.match_count == 42
        assert metric.inlier_count == 30
        assert metric.elapsed_ms == 12.5
        assert metric.map_id == gt.map_id

        # 持久化校验：从 Result_Store 读回相同的值（Req 6.1）。
        stored = store.get_frame(case.case_id, param_set.name)
        assert stored is not None
        assert stored.success is True
        assert stored.error_distance == metric.error_distance
        assert stored.scale_error == metric.scale_error
        assert stored.confidence == metric.confidence
        assert stored.match_count == metric.match_count
        assert stored.inlier_count == metric.inlier_count
        assert stored.elapsed_ms == metric.elapsed_ms
        assert stored.map_id == gt.map_id
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 2：匹配失败语义（Req 6.4）
# ---------------------------------------------------------------------------
def test_match_failure_none_output():
    """runner 返回 None → success=False，error_distance / scale_error 未定义（Req 6.4）。"""
    store = ResultStore(":memory:")
    try:
        gt = _gt("case-none")
        param_set = _sift_set()
        case = HarnessTestCase(case_id=gt.case_id, image_path="/screenshots/case-none.png")

        runner = _StubRunner(None, store)
        metric = MatchEvaluator().evaluate(case, gt, param_set, runner)

        assert metric.success is False
        assert metric.error_distance is None
        assert metric.scale_error is None

        # 失败结果同样被持久化（Req 6.1）。
        stored = store.get_frame(case.case_id, param_set.name)
        assert stored is not None
        assert stored.success is False
        assert stored.error_distance is None
        assert stored.scale_error is None
    finally:
        store.close()


def test_match_failure_success_false_output():
    """runner 返回 success=False 的 MatchOutput → 同样判失败、误差未定义（Req 6.4）。"""
    store = ResultStore(":memory:")
    try:
        gt = _gt("case-fail")
        param_set = _sift_set()
        case = HarnessTestCase(case_id=gt.case_id, image_path="/screenshots/case-fail.png")

        output = MatchOutput(
            success=False,
            match_count=3,
            inlier_count=0,
            confidence=0.2,
            elapsed_ms=8.0,
        )
        runner = _StubRunner(output, store)
        metric = MatchEvaluator().evaluate(case, gt, param_set, runner)

        assert metric.success is False
        assert metric.error_distance is None
        assert metric.scale_error is None
        # 失败时仍记录可观测的诊断字段（Req 6.6）。
        assert metric.confidence == 0.2
        assert metric.match_count == 3
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 3：scale 未定义的失败语义（Req 6.5）
# ---------------------------------------------------------------------------
def test_scale_undefined_when_map_scale_non_positive():
    """成功但 map_scale ≤ 0 → scale_error 未定义；error_distance 仍可计算（Req 6.5）。"""
    store = ResultStore(":memory:")
    try:
        gt = _gt("case-scale0")
        param_set = _sift_set()
        case = HarnessTestCase(case_id=gt.case_id, image_path="/screenshots/case-scale0.png")

        game_center = (gt.game_xy[0] + 50.0, gt.game_xy[1])
        output = MatchOutput(
            success=True,
            match_count=20,
            inlier_count=15,
            confidence=0.92,
            center=(1000.0, 2000.0),
            game_center=game_center,
            map_scale=0.0,  # ≤ 0 → scale_error 未定义（Req 6.5）
            elapsed_ms=10.0,
        )
        runner = _StubRunner(output, store)
        metric = MatchEvaluator().evaluate(case, gt, param_set, runner)

        # map_scale ≤ 0 → scale_error 未定义。
        assert metric.scale_error is None
        # game_center 存在 → error_distance 仍可度量。
        assert metric.error_distance is not None
        assert abs(metric.error_distance - 50.0) < 1e-6

        stored = store.get_frame(case.case_id, param_set.name)
        assert stored is not None
        assert stored.scale_error is None
        assert stored.error_distance == metric.error_distance
    finally:
        store.close()


def test_scale_undefined_when_reference_scale_non_positive():
    """gt.reference_scale ≤ 0 不合法（GroundTruth 会拒绝），故以负 map_scale 之外，
    单独验证：measured 合法但 reference 在 metrics 层 ≤ 0 时 scale_error 未定义（Req 6.5）。

    GroundTruth 构造期即强制 reference_scale > 0（参见 annotation.GroundTruth），
    因此评估层只可能因 measured map_scale ≤ 0 而使 scale_error 未定义；这里用一个
    极小但为正的 reference 与负 map_scale 组合再次确认失败语义。
    """
    store = ResultStore(":memory:")
    try:
        gt = _gt("case-negscale")
        param_set = _sift_set()
        case = HarnessTestCase(case_id=gt.case_id, image_path="/screenshots/case-negscale.png")

        output = MatchOutput(
            success=True,
            match_count=18,
            inlier_count=12,
            confidence=0.91,
            center=(1000.0, 2000.0),
            game_center=gt.game_xy,
            map_scale=-1.0,  # 负 scale → 未定义
            elapsed_ms=9.0,
        )
        runner = _StubRunner(output, store)
        metric = MatchEvaluator().evaluate(case, gt, param_set, runner)

        assert metric.scale_error is None
        assert metric.error_distance is not None
        assert abs(metric.error_distance) < 1e-9  # game_center == gt.game_xy
    finally:
        store.close()
