"""Match_Runner 缓存缺失 / 复用边界与烟雾测试（Requirement 4.3, 4.4, 7.3）。

本模块覆盖 :class:`src.eval_harness.runner.MatchRunner` 的三段式逻辑中
与缓存缺失、结果复用相关的边界行为，并以烟雾测试验证"测试机只读缓存、
绝不触发全图特征提取"这一部署约束。

为隔离对真实 OpenCV 引擎的依赖，所有用例都被设计为 **引擎永不被构造**：

- 缓存缺失（test 1 / test 3）：``run_case`` 在 :func:`features.cache_path`
  的 ``os.path.isfile`` 检查处即短路返回 ``None`` 并告警，根本不进入
  ``_get_engine`` / ``_build_engine``（后者才会 import 并构造引擎）。
- 结果复用（test 2）：命中 Result_Store 已存结果后由 ``_reconstruct_output``
  重建一个部分 ``MatchOutput`` 视图返回，同样不构造引擎、不读缓存。

Validates: Requirements 4.3, 4.4, 7.3
"""

from __future__ import annotations

import os

# 以别名导入，避免 pytest 误把 case_loader.TestCase 当作测试类收集。
from src.eval_harness.case_loader import TestCase as HarnessTestCase
from src.eval_harness.features import cache_path
from src.eval_harness.map_registry import MapEntry
from src.eval_harness.params import ParamSet, SiftParams
from src.eval_harness.result_store import FrameMetric, ResultStore
from src.eval_harness.runner import MatchRunner


def _sift_set() -> ParamSet:
    """一个合法的 SIFT Param_Set，其 ``name`` 用作缓存分组键与结果主键的一部分。"""
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


class _StubRegistry:
    """最小 MapRegistry 替身：记录 ``discover`` 被调用的次数。

    用于烟雾测试断言"缓存缺失时 run_case 在进入引擎路径之前即短路"——
    若 ``discover`` 从未被调用，则 ``_get_engine`` / ``_build_engine``
    （唯一会构造真实引擎的代码路径）必然也未被触达。
    """

    def __init__(self, entries):
        self._entries = entries
        self.discover_calls = 0

    def discover(self):
        self.discover_calls += 1
        return self._entries


# ---------------------------------------------------------------------------
# Test 1：缓存缺失跳过 + 告警含缺失路径（Req 4.3）
# ---------------------------------------------------------------------------
def test_cache_miss_returns_none_and_warns_with_path(tmp_path):
    """缓存缺失时 run_case 返回 None，并在 warnings 中记录缺失缓存路径（Req 4.3）。"""
    caches_dir = str(tmp_path / "caches")  # 空目录：不存在任何 .npz
    os.makedirs(caches_dir, exist_ok=True)

    # 空 Result_Store（无任何已存结果），map_registry=None。
    store = ResultStore(":memory:")
    try:
        runner = MatchRunner(caches_dir, map_registry=None, result_store=store)
        case = HarnessTestCase(case_id="case-1", image_path="/screenshots/case-1.png")
        param_set = _sift_set()
        map_id = "8"

        result = runner.run_case(case, map_id, param_set)

        # 缓存缺失 → 跳过，返回 None。
        assert result is None

        # 告警有且仅有一条，且包含缺失缓存的精确路径。
        expected_path = cache_path(caches_dir, param_set.name, map_id, param_set.algo)
        assert len(runner.warnings) == 1
        assert expected_path in runner.warnings[0]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 2：已存结果复用，不触碰缓存 / 引擎（Req 7.3）
# ---------------------------------------------------------------------------
def test_reuse_existing_result_without_cache_or_engine(tmp_path):
    """命中 Result_Store 已存结果时复用并重建视图，不读缓存、不构造引擎（Req 7.3）。"""
    # caches_dir 故意为空：复用路径不应依赖任何缓存文件。
    caches_dir = str(tmp_path / "caches")
    os.makedirs(caches_dir, exist_ok=True)

    case = HarnessTestCase(case_id="case-A", image_path="/screenshots/case-A.png")
    param_set = _sift_set()
    map_id = "910"

    store = ResultStore(":memory:")
    try:
        stored = FrameMetric(
            case_id=case.case_id,
            param_set_name=param_set.name,
            map_id=map_id,
            success=True,
            error_distance=123.0,
            scale_error=0.05,
            confidence=0.875,
            match_count=42,
            inlier_count=30,
            elapsed_ms=7.5,
        )
        store.upsert_frame(stored, ts="2026-01-01T00:00:00")

        runner = MatchRunner(caches_dir, map_registry=None, result_store=store)
        result = runner.run_case(case, map_id, param_set, recompute=False)

        # 返回了由已存指标重建的 MatchOutput 视图。
        assert result is not None
        assert result.success is True
        assert result.confidence == 0.875
        assert result.match_count == 42
        assert result.inlier_count == 30
        assert result.elapsed_ms == 7.5

        # 复用路径不产生任何告警（未触碰缺失缓存逻辑）。
        assert runner.warnings == []
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 3（SMOKE）：缺失缓存绝不触发全图提取 / 引擎构造（Req 4.4）
# ---------------------------------------------------------------------------
def test_smoke_missing_cache_never_triggers_extraction(tmp_path):
    """烟雾：即便注册表可发现某地图，缺失 npz 仍只跳过+告警，绝不构造引擎（Req 4.4）。

    构造一个 *会* 发现 map_id="8" 的桩注册表，但 caches_dir 下没有任何
    ``.npz``。run_case 应在缓存检查处短路：返回 None、记一条含路径的告警、
    不抛异常；且桩注册表的 ``discover`` 从未被调用，证明从未进入会构造真实
    引擎的代码路径（测试机内存安全约束）。
    """
    caches_dir = str(tmp_path / "caches")  # 无任何缓存文件
    os.makedirs(caches_dir, exist_ok=True)

    # 注册表"能发现"一张大地图，但其原图路径并不存在——即便进入引擎路径也会
    # 因 npz 缺失而被二次拦截；而本用例验证根本不会走到那一步。
    entries = {"8": MapEntry(map_id="8", image_path="/nonexistent/8.png", coords=None)}
    registry = _StubRegistry(entries)

    runner = MatchRunner(caches_dir, map_registry=registry, result_store=None)
    case = HarnessTestCase(case_id="case-smoke", image_path="/screenshots/case-smoke.png")
    param_set = _sift_set()

    # 不应抛出任何异常（绝不触发全图提取）。
    result = runner.run_case(case, "8", param_set)

    assert result is None
    expected_path = cache_path(caches_dir, param_set.name, "8", param_set.algo)
    assert len(runner.warnings) == 1
    assert expected_path in runner.warnings[0]

    # 关键烟雾断言：从未进入引擎构造路径（discover 未被调用）。
    assert registry.discover_calls == 0
