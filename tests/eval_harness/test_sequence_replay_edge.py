"""序列重放边界（EDGE_CASE）示例测试：SequenceEvaluator.evaluate。

覆盖两个不易被属性测试稳定触发的控制流分支：

1. **连续失败回退全局、仍失败记定位失败（Req 9.5）**：
   已锁定后小区间匹配连续失败达 ``fallback_max_failures``（默认 2），解约束做全局
   匹配；全局仍失败 → 该帧记一次定位失败（``fail_frames`` 自增）。
2. **错误锁定记一次（Req 9.9，EDGE_CASE）**：
   某帧锁定到与 Ground_Truth ``map_id`` 不符的地图 → ``wrong_lock_frames`` 至少自增 1。

这些是确定性示例测试（非属性测试）：用脚本化的 STUB ``Match_Runner`` 按调用序号 /
``region`` 参数返回预设 :class:`MatchOutput`，从而精确驱动 :meth:`evaluate` 的回退 /
锁定 / 错误锁定分支，断言聚合指标符合预期。
"""

from __future__ import annotations

import random

from src.eval_harness.params import ParamSet, SiftParams
from src.eval_harness.profile import LocalizationProfile
from src.eval_harness.result_store import ResultStore
from src.eval_harness.sequence import Frame, FrameSequence, SequenceEvaluator
from src.match_engine.common import MatchOutput


# ---------------------------------------------------------------------------
# 测试夹具：脚本化 STUB Match_Runner
# ---------------------------------------------------------------------------


class StubRunner:
    """脚本化 STUB ``Match_Runner``。

    ``responder(call_index, region, case, map_id)`` 返回每次 ``run_case`` 的输出
    （:class:`MatchOutput` 或 ``None`` 表示失败）。``evaluate`` 在单帧内可能调用两次
    ``run_case``（小区间 + 全局回退），故 responder 同时拿到调用序号与 ``region``，
    可据此确定性地驱动各分支。
    """

    def __init__(self, responder, result_store=None):
        self._responder = responder
        self.result_store = result_store
        self.calls = []

    def run_case(self, case, map_id, param_set, region=None, recompute=False):
        idx = len(self.calls)
        self.calls.append((getattr(case, "case_id", None), map_id, region))
        return self._responder(idx, region, case, map_id)


def _param_set() -> ParamSet:
    """构造一个最小可用的 SIFT Param_Set（仅用于透传，stub 忽略其内容）。"""
    return ParamSet(
        algo="sift",
        params=SiftParams(
            contrast_threshold=0.04,
            edge_threshold=10,
            n_octave_layers=3,
            sigma=1.6,
            grid=4,
            max_per_cell=5,
            ratio=0.7,
        ),
    )


def _profile() -> LocalizationProfile:
    """默认阈值的 Localization_Profile：fallback_max_failures=2、lock_*=0.9/10。"""
    return LocalizationProfile(name="edge_profile")


def _lock_output() -> MatchOutput:
    """一次足以触发锁定的成功匹配（confidence>=0.9 且 match_count>=10）。"""
    return MatchOutput(
        success=True,
        match_count=20,
        inlier_count=20,
        confidence=0.95,
        game_center=(100.0, 200.0),
    )


# ---------------------------------------------------------------------------
# 场景 1：回退全局仍失败 → 记定位失败（Req 9.5）
# ---------------------------------------------------------------------------


def test_fallback_then_global_failure_counts_localization_failure():
    """连续小区间失败回退全局、全局仍失败 → fail_frames 记录定位失败。

    序列设计（map_id 全为 ``map_a``，fallback_max_failures=2）：

    - 帧 0：未锁定 → 全局匹配（region=None），返回成功且达锁定阈值 → 锁定 ``map_a``。
    - 帧 1：已锁定且未回退 → 小区间匹配（region≠None），失败（consecutive=1，软失败）。
    - 帧 2：已锁定且未回退 → 小区间匹配（region≠None），失败（consecutive=2，达阈值）
      → 当帧解约束做全局匹配（region=None），仍失败 → fail_frames += 1。

    **Validates: Requirements 9.5**
    """

    def responder(idx, region, case, map_id):
        # 仅首次全局匹配（帧 0 的锁定）成功；其余一律失败。
        if idx == 0 and region is None:
            return _lock_output()
        return None

    store = ResultStore(":memory:")
    runner = StubRunner(responder, result_store=store)
    param_set = _param_set()
    profile = _profile()

    frames = [
        Frame(case_id=f"c{i}", map_id="map_a", pixel_xy=(10.0 * i, 10.0 * i),
              game_xy=(100.0 + i, 200.0 + i), image_path=f"dummy_{i}.png")
        for i in range(3)
    ]
    sequence = FrameSequence(map_id="map_a", frames=frames)

    metric = SequenceEvaluator(profile).evaluate(
        sequence, param_set, profile, runner, rng=random.Random(0)
    )

    # 帧 2 触发回退全局且仍失败 → 至少一次定位失败。
    assert metric.fail_frames >= 1
    # 帧 0 已锁定到正确地图（map_a），不应记错误锁定。
    assert metric.lock_map_id == "map_a"
    assert metric.lock_frame_index == 0
    assert metric.wrong_lock_frames == 0
    # 单帧内发生了"小区间失败 + 全局回退"两次调用，证明确实走了回退路径。
    fallback_regions = [c[2] for c in runner.calls]
    assert None in fallback_regions  # 全局回退调用（region=None）
    assert any(r is not None for r in fallback_regions)  # 小区间调用（region≠None）

    # 指标已写入 sequence_results。
    row = store._conn.execute(
        "SELECT fail_frames FROM sequence_results WHERE sequence_id = ?",
        (sequence.sequence_id,),
    ).fetchone()
    assert row is not None
    assert row["fail_frames"] == metric.fail_frames
    store.close()


# ---------------------------------------------------------------------------
# 场景 2：错误锁定记一次（Req 9.9，EDGE_CASE）
# ---------------------------------------------------------------------------


def test_wrong_lock_counts_at_least_once():
    """锁定到与 Ground_Truth map_id 不符的地图 → wrong_lock_frames 至少记一次。

    帧 0 的 Ground_Truth ``map_id`` 为 ``map_a``，但全局匹配返回的 ``MatchOutput``
    带 ``map_id='map_b'``（错误地图）且达锁定阈值。``evaluate`` 据 ``out.map_id``
    取生效地图 ``map_b`` 并锁定，与 GT ``map_a`` 不符 → wrong_lock_frames += 1。

    **Validates: Requirements 9.9**
    """

    def responder(idx, region, case, map_id):
        out = _lock_output()
        # MatchOutput 默认无 map_id 属性；显式注入错误地图以驱动错误锁定分支。
        out.map_id = "map_b"
        return out

    store = ResultStore(":memory:")
    runner = StubRunner(responder, result_store=store)
    param_set = _param_set()
    profile = _profile()

    frame = Frame(
        case_id="c0", map_id="map_a", pixel_xy=(0.0, 0.0),
        game_xy=(100.0, 200.0), image_path="dummy.png",
    )
    sequence = FrameSequence(map_id="map_a", frames=[frame])

    metric = SequenceEvaluator(profile).evaluate(
        sequence, param_set, profile, runner, rng=random.Random(0)
    )

    assert metric.wrong_lock_frames >= 1
    assert metric.lock_map_id == "map_b"
    assert metric.lock_frame_index == 0

    # 指标已写入 sequence_results。
    row = store._conn.execute(
        "SELECT wrong_lock_frames, lock_map_id FROM sequence_results WHERE sequence_id = ?",
        (sequence.sequence_id,),
    ).fetchone()
    assert row is not None
    assert row["wrong_lock_frames"] == metric.wrong_lock_frames
    assert row["lock_map_id"] == "map_b"
    store.close()
