"""Sequence_Evaluator：连续定位评估（Requirement 9）。

包含 Frame_Sequence 构造、Simulated_OCR、平滑与锁定纯逻辑，以及逐帧重放评估。
连续序列由若干空间相邻的已标注单帧用例拼接 / 模拟而成。

本模块的纯逻辑部分（:func:`build_sequence`、:func:`simulate_ocr`、
:func:`smooth`、:func:`should_lock`）不做任何 I/O，给定注入的 ``rng`` 即为确定性，
便于属性测试（design.md Property 10–13，tasks.md 15.2–15.5）。

逐帧重放评估 :meth:`SequenceEvaluator.evaluate`（Task 15.6）在纯逻辑之上做副作用编排：
调用注入的 ``Match_Runner`` 执行匹配、聚合指标并写入 ``Result_Store``。给定同一序列
（或显式注入的 ``rng``）其聚合指标为确定性，便于回归对照。
"""

from __future__ import annotations

import datetime
import math
import random
import zlib
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from .metrics import game_distance

Point = Tuple[float, float]

#: Frame_Sequence 构造的默认投影中心（地图像素）距离阈值（Req 9.1）。
#: 与 :class:`src.eval_harness.profile.LocalizationProfile.region_size` 一致
#: （对应 ``MapOverlayTask.MAP_REGION_SIZE = 1000``）。
DEFAULT_REGION_SIZE: float = 1000.0


# ===========================================================================
# 数据结构
# ===========================================================================


@dataclass(frozen=True)
class Frame:
    """连续序列中的单帧（来自一条已标注 Ground_Truth）。

    Attributes:
        case_id: 来源用例 ID（去扩展名文件名）。
        map_id: 该帧所属大地图标识。
        pixel_xy: 投影中心的**地图像素**坐标 ``(x, y)``，用于序列构造的几何距离判定。
        game_xy: 对应的**游戏坐标** ``(x, y)``，作为重放时的 Ground_Truth 位置。
        reference_scale: 参考缩放比（标注阶段 ``MatchOutput.map_scale``）。
        image_path: 该帧来源截图的磁盘路径（重放时透传给 ``Match_Runner``）。
            默认 ``None`` 以保持向后兼容（纯逻辑测试无需图片）。
    """

    case_id: Optional[str]
    map_id: str
    pixel_xy: Point
    game_xy: Point
    reference_scale: float = 0.0
    image_path: Optional[str] = None


@dataclass
class FrameSequence:
    """一组有序的连续帧（同 map_id，两两投影中心距离 < 阈值）。

    Attributes:
        map_id: 该序列所属大地图标识（序列内所有帧一致）。
        frames: 按就近顺序排列的帧列表（不插值）。
    """

    map_id: str
    frames: List[Frame] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def sequence_id(self) -> str:
        """供 Result_Store 使用的确定性序列标识。

        由 map_id 与首尾帧 case_id 及帧数组合而成，使同一 map_id 下的不同序列可区分。
        """
        if not self.frames:
            return f"{self.map_id}__empty"
        first = self.frames[0].case_id or "?"
        last = self.frames[-1].case_id or "?"
        return f"{self.map_id}__{first}__{last}__n{len(self.frames)}"


@dataclass
class SequenceMetric:
    """一次序列评估的指标（design.md > Components > Sequence_Evaluator，Req 9.7–9.9）。

    Attributes:
        sequence_id: 序列标识（Result_Store 主键之一）。
        profile_name: 使用的 Localization_Profile 名称。
        param_set_name: 使用的 Param_Set 名称。
        map_id: 序列所属大地图标识。
        mean_error: 逐帧 Error_Distance 的平均值（游戏坐标单位）。
        max_error: 逐帧 Error_Distance 的最大值。
        fail_frames: 定位失败帧数。
        wrong_lock_frames: 错误锁定帧数（锁定到与 GT 不符的 map_id）。
        lock_map_id: 触发锁定的 map_id（未锁定为 ``None``）。
        lock_frame_index: 触发锁定的帧序号（未锁定为 ``None``）。
    """

    sequence_id: str
    profile_name: str
    param_set_name: str
    map_id: str
    mean_error: float
    max_error: float
    fail_frames: int
    wrong_lock_frames: int
    lock_map_id: Optional[str] = None
    lock_frame_index: Optional[int] = None


# ===========================================================================
# 归一化辅助
# ===========================================================================


def _coerce_point(value: object) -> Point:
    """把坐标规整为 ``(float, float)`` 二元组。"""
    try:
        x, y = value  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"坐标必须为二元 (x, y)，得到 {value!r}") from exc
    return (float(x), float(y))


def _as_frame(item: object) -> Frame:
    """把一条已标注记录归一化为 :class:`Frame`。

    接受：

    - :class:`Frame` 本身；
    - ``(case, gt)`` 二元组，其中 ``gt`` 带 ``map_id`` / ``pixel_xy`` 属性
      （如 :class:`src.eval_harness.annotation.GroundTruth`）；
    - 任意带 ``map_id`` 与 ``pixel_xy`` 属性的对象（``GroundTruth``）。

    Args:
        item: 待归一化的记录。

    Returns:
        归一化后的 :class:`Frame`。

    Raises:
        TypeError: 当 ``item`` 不含所需字段时。
    """
    if isinstance(item, Frame):
        return item

    # (case, gt) 二元组
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and hasattr(item[1], "map_id")
        and hasattr(item[1], "pixel_xy")
    ):
        case, gt = item
        case_id = getattr(case, "case_id", None) or getattr(gt, "case_id", None)
        image_path = getattr(case, "image_path", None) or getattr(
            gt, "image_path", None
        )
        return _frame_from_gt(gt, case_id, image_path)

    # GroundTruth-like 对象
    if hasattr(item, "map_id") and hasattr(item, "pixel_xy"):
        return _frame_from_gt(
            item, getattr(item, "case_id", None), getattr(item, "image_path", None)
        )

    raise TypeError(
        "序列构造输入必须为 Frame、GroundTruth（带 map_id/pixel_xy），"
        "或 (case, gt) 二元组"
    )


def _frame_from_gt(
    gt: object, case_id: Optional[str], image_path: Optional[str] = None
) -> Frame:
    """从 GroundTruth-like 对象构造 :class:`Frame`。"""
    pixel_xy = _coerce_point(getattr(gt, "pixel_xy"))
    game_attr = getattr(gt, "game_xy", None)
    game_xy = _coerce_point(game_attr) if game_attr is not None else pixel_xy
    return Frame(
        case_id=case_id,
        map_id=str(getattr(gt, "map_id")),
        pixel_xy=pixel_xy,
        game_xy=game_xy,
        reference_scale=float(getattr(gt, "reference_scale", 0.0) or 0.0),
        image_path=image_path,
    )


def _pixel_distance(a: Point, b: Point) -> float:
    """两个地图像素点的欧氏距离。"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ===========================================================================
# Property 10: Frame_Sequence 构造
# ===========================================================================


def build_sequence(
    cases_with_gt: Iterable[object],
    threshold: float = DEFAULT_REGION_SIZE,
) -> List[FrameSequence]:
    """构造 Frame_Sequence 列表（Req 9.1，design.md Property 10）。

    分组与排序规则：

    1. 先按 ``map_id`` 分组——不同地图的帧绝不进入同一序列。
    2. 在每个 map_id 组内，用"最近邻链 + 全配对约束"贪心构造序列：从一个种子帧
       出发，反复挑选距离**当前链尾**最近、且与序列内**所有**已有帧的投影中心
       （地图像素）距离均 ``< threshold`` 的帧加入序列；当没有这样的帧时收束本序列，
       再从剩余帧另起新序列。
    3. 不在帧之间做插值——序列只重排已有用例。

    这样构造出的每个序列满足不变量（Property 10）：序列内所有帧 ``map_id`` 相同，
    且**任意两帧**的投影中心几何距离均 ``< threshold``（默认 1000）。排序为就近顺序
    （最近邻链）。

    为保证确定性，组内帧先按 ``(case_id, pixel_xy)`` 稳定排序；最近邻挑选在距离相等时
    取先出现者。

    Args:
        cases_with_gt: 已标注记录的可迭代集合；每项可为 :class:`Frame`、
            ``GroundTruth``（带 ``map_id`` / ``pixel_xy``）或 ``(case, gt)`` 二元组。
        threshold: 投影中心（地图像素）距离阈值，默认 :data:`DEFAULT_REGION_SIZE`
            （= 1000）；判定为严格小于。

    Returns:
        :class:`FrameSequence` 列表，按 map_id 升序、组内按构造顺序排列。
    """
    frames = [_as_frame(c) for c in cases_with_gt]

    groups: dict[str, List[Frame]] = {}
    for f in frames:
        groups.setdefault(f.map_id, []).append(f)

    sequences: List[FrameSequence] = []
    for map_id in sorted(groups):
        remaining = sorted(
            groups[map_id], key=lambda fr: (str(fr.case_id), fr.pixel_xy)
        )
        while remaining:
            seed = remaining.pop(0)
            cluster: List[Frame] = [seed]
            while remaining:
                last = cluster[-1]
                best_idx: Optional[int] = None
                best_dist: Optional[float] = None
                for i, cand in enumerate(remaining):
                    # 必须与序列内所有已有帧均 < threshold（全配对约束）。
                    if not all(
                        _pixel_distance(cand.pixel_xy, m.pixel_xy) < threshold
                        for m in cluster
                    ):
                        continue
                    d = _pixel_distance(cand.pixel_xy, last.pixel_xy)
                    if best_dist is None or d < best_dist:
                        best_dist = d
                        best_idx = i
                if best_idx is None:
                    break
                cluster.append(remaining.pop(best_idx))
            sequences.append(FrameSequence(map_id=map_id, frames=cluster))

    return sequences


# ===========================================================================
# Property 11: Simulated_OCR 噪声
# ===========================================================================


def _extract_game_xy(gt: object) -> Point:
    """从 GroundTruth-like 对象或裸坐标取游戏坐标。"""
    game_attr = getattr(gt, "game_xy", None)
    if game_attr is not None:
        return _coerce_point(game_attr)
    return _coerce_point(gt)


def _extract_noise(profile: object) -> float:
    """从 Localization_Profile 或裸数值取 OCR 噪声幅度。"""
    noise = getattr(profile, "ocr_noise", None)
    if noise is None:
        noise = profile
    return float(noise)


def simulate_ocr(gt: object, profile: object, rng) -> Point:
    """为某帧生成 Simulated_OCR 游戏坐标（Req 9.2，design.md Property 11）。

    每个游戏坐标分量取 ``g × (1 + u)``，其中 ``u ~ U(−noise, +noise)``，
    ``noise`` 由 Localization_Profile 给出（默认 0.10）。由此保证
    ``|simulated − g| ≤ |g| × noise``。

    Simulated_OCR 仅用于确定匹配搜索区间中心，不作为定位输出。给定注入的 ``rng``
    本函数为确定性，便于属性测试。

    Args:
        gt: Ground_Truth-like 对象（带 ``game_xy``）或裸 ``(gx, gy)`` 游戏坐标。
        profile: :class:`LocalizationProfile`（取 ``ocr_noise``）或裸 ``noise`` 数值。
        rng: 随机源，需提供 ``uniform(a, b)``（如 ``random.Random`` 实例）。

    Returns:
        加噪后的游戏坐标 ``(x, y)``。
    """
    gx, gy = _extract_game_xy(gt)
    noise = _extract_noise(profile)
    ux = rng.uniform(-noise, noise)
    uy = rng.uniform(-noise, noise)
    return (gx * (1.0 + ux), gy * (1.0 + uy))


# ===========================================================================
# Property 12: 平滑为窗口内凸组合
# ===========================================================================


def smooth(window_points: Sequence[Point], weights: Sequence[float]) -> Point:
    """对窗口内的最近 ``game_center`` 序列做加权移动平均（Req 9.3，Property 12）。

    输出为窗口内各点的**凸组合**（权重归一化为非负且和为 1），因此每个坐标分量都落在
    所用窗口内对应分量的最小值与最大值之间，不会越界。

    窗口长度与权重个数不一致时的对齐策略（与 ``MapOverlayTask`` 一致：最近的帧权重最大，
    权重序列尾部对应最近的点）：

    - 若权重多于点（窗口较短）：取权重序列的**末尾 N 个**（高权重，对应较近的点）。
    - 若点多于权重（窗口较长）：仅取**最近的 len(weights) 个**点参与平滑。

    Args:
        window_points: 最近的 ``(x, y)`` 点序列（最旧在前、最新在后），非空。
        weights: 正权重序列（与点对齐，尾部对应最新）。

    Returns:
        加权移动平均后的 ``(x, y)``。

    Raises:
        ValueError: 当 ``window_points`` 为空时。
    """
    pts = list(window_points)
    if not pts:
        raise ValueError("smooth 需要非空的窗口点序列")

    w = [float(x) for x in weights]
    if not w:
        w = [1.0] * len(pts)

    # 对齐点与权重的数量（尾部对齐：最新的点 / 最大的权重）。
    if len(w) >= len(pts):
        w = w[len(w) - len(pts):]
    else:
        pts = pts[len(pts) - len(w):]

    total = math.fsum(w)
    if total <= 0:
        # 退化为均匀权重，仍是凸组合。
        n = len(pts)
        w = [1.0] * n
        total = float(n)

    sx = math.fsum(wi * p[0] for wi, p in zip(w, pts))
    sy = math.fsum(wi * p[1] for wi, p in zip(w, pts))
    return (sx / total, sy / total)


# ===========================================================================
# Property 13: 地图锁定判定
# ===========================================================================


def should_lock(
    confidence: float,
    match_count: int,
    currently_locked: bool,
    profile: object,
) -> bool:
    """判定是否发生新的地图锁定（Req 9.4，design.md Property 13）。

    当且仅当 ``confidence ≥ lock_confidence`` **且** ``match_count ≥ lock_match_count``
    **且** 当前未锁定时返回 ``True``。

    Args:
        confidence: 当前帧匹配置信度。
        match_count: 当前帧特征匹配点数。
        currently_locked: 当前是否已锁定某地图。
        profile: :class:`LocalizationProfile`，提供 ``lock_confidence`` 与
            ``lock_match_count`` 阈值。

    Returns:
        ``True`` 表示应在本帧锁定。
    """
    if currently_locked:
        return False
    return (
        confidence >= profile.lock_confidence
        and match_count >= profile.lock_match_count
    )


# ===========================================================================
# Sequence_Evaluator 逐帧重放（Task 15.6）
# ===========================================================================


#: 重放时透传给 ``Match_Runner.run_case`` 的轻量用例视图（仅需 case_id + image_path）。
_FrameCase = namedtuple("_FrameCase", ["case_id", "image_path"])


def _seed_from_sequence_id(sequence_id: str) -> int:
    """由 sequence_id 派生确定性随机种子。

    使用 :func:`zlib.crc32`（跨进程稳定，不受 ``PYTHONHASHSEED`` 影响），保证同一序列
    的重放结果可复现。
    """
    return zlib.crc32(sequence_id.encode("utf-8"))


def _is_success(out: object) -> bool:
    """判定一次 ``run_case`` 返回是否为成功匹配。"""
    return out is not None and bool(getattr(out, "success", False))


def _safe_run(runner, case, map_id, param_set, region):
    """健壮地调用 ``runner.run_case``；任何异常都按"失败"处理（返回 ``None``）。

    评估不应因单帧引擎异常而中断整条序列；异常帧记为匹配失败。
    """
    try:
        return runner.run_case(case, map_id, param_set, region=region)
    except Exception:  # noqa: BLE001 - 单帧异常降级为失败，不中断序列重放
        return None


def _predicted_pixel(frame: Frame, ocr_game: Point) -> Point:
    """由 Simulated_OCR 游戏坐标推得搜索区间中心的**地图像素**坐标。

    本工具的 Simulated_OCR 工作在游戏坐标系，而 ``Match_Runner`` 期望的 ``region``
    为地图像素。由于 ``game = pixel × scale + offset`` 为 per-axis 仿射变换，OCR 对
    游戏坐标施加的**相对扰动** ``g → g × (1 + u)`` 可等价地施加到像素坐标上
    （``offset`` 在标注阶段已确定，这里以"相对比例"近似还原像素中心，避免依赖未知
    的 ``offset`` 绝对值）。即 ``predicted_px = px × (ocr_g / g)``；当 ``g == 0`` 时
    取比例 1（不扰动该轴）。这是评估用的明确化简，不追求像素级精确。
    """
    gx, gy = frame.game_xy
    px, py = frame.pixel_xy
    ox, oy = ocr_game
    rx = (ox / gx) if gx != 0 else 1.0
    ry = (oy / gy) if gy != 0 else 1.0
    return (px * rx, py * ry)


def _narrow_region(frame: Frame, ocr_game: Point, profile: object) -> Tuple[float, float, float, float]:
    """构造以预测像素中心为中心、边长 ``region_size`` 的方形搜索区间（地图像素）。

    返回 ``(x, y, w, h)``，其中 ``(x, y)`` 为左上角。``region_size`` 取自
    Localization_Profile（对应 ``MapOverlayTask.MAP_REGION_SIZE``）。
    """
    cx, cy = _predicted_pixel(frame, ocr_game)
    size = float(getattr(profile, "region_size", DEFAULT_REGION_SIZE))
    half = size / 2.0
    return (cx - half, cy - half, size, size)


class SequenceEvaluator:
    """连续定位评估子系统（Requirement 9）。

    本类聚合序列构造、Simulated_OCR、平滑与锁定判定等纯逻辑（已实现），并为逐帧重放
    评估 :meth:`evaluate` 预留骨架（由 Task 15.6 填充）。

    Args:
        profile: :class:`LocalizationProfile`，提供噪声 / 区间 / 平滑 / 锁定等参数。
        runner: 单帧匹配执行器（``Match_Runner``），重放时调用；纯逻辑测试可不传。
        store: :class:`ResultStore`，写入 ``sequence_results``；纯逻辑测试可不传。
    """

    def __init__(self, profile, runner=None, store=None) -> None:
        self.profile = profile
        self.runner = runner
        self.store = store

    def build_sequence(self, cases_with_gt: Iterable[object]) -> List[FrameSequence]:
        """构造 Frame_Sequence 列表（委托给模块级 :func:`build_sequence`）。"""
        return build_sequence(cases_with_gt, threshold=self.profile.region_size)

    def simulate_ocr(self, gt: object, profile: object, rng) -> Point:
        """生成 Simulated_OCR（委托给模块级 :func:`simulate_ocr`）。"""
        return simulate_ocr(gt, profile, rng)

    def smooth(self, window_points: Sequence[Point]) -> Point:
        """按 profile 权重对窗口做加权移动平均（委托给 :func:`smooth`）。"""
        return smooth(window_points, self.profile.smooth_weights)

    def should_lock(
        self, confidence: float, match_count: int, currently_locked: bool
    ) -> bool:
        """按 profile 阈值判定锁定（委托给 :func:`should_lock`）。"""
        return should_lock(confidence, match_count, currently_locked, self.profile)

    def evaluate(self, sequence, param_set, profile, runner, rng=None) -> SequenceMetric:
        """逐帧重放序列并度量稳定性 / 正确性（Req 9.5–9.9）。

        逐帧流程（design.md > Components > Sequence_Evaluator）：

        1. 由 Ground_Truth 生成 Simulated_OCR 游戏坐标，作为匹配搜索区间中心
           （Req 9.2）。
        2. **已锁定**且未进入回退态时，在该地图由 OCR 收窄的小区间内匹配（Req 9.3）；
           **未锁定**时做全局匹配以尝试获取锁定。
        3. 某帧 ``confidence`` 与 ``match_count`` 同时达阈值且未锁定 → 锁定该 map_id，
           并记录触发锁定的帧序号（Req 9.4, 9.8）。
        4. 小区间匹配**连续失败**达 ``fallback_max_failures``（默认 2）→ 解约束做全局
           匹配；仍失败则记一次定位失败（Req 9.5）。
        5. 匹配成功 → 对 ``game_center`` 应用平滑（窗口内凸组合）得到该帧输出位置，并以
           游戏坐标记录其与 Ground_Truth 的 Error_Distance（Req 9.6）。
        6. 当前生效地图（锁定地图，未锁定时为本帧匹配地图）与 Ground_Truth ``map_id``
           不符 → 记一次错误锁定（Req 9.9）。

        序列处理完成后报告逐帧 Error_Distance 的均值 / 最大值与定位失败帧数（Req 9.7），
        并把指标写入 ``Result_Store.sequence_results``（若可用）。

        确定性：给定同一 ``sequence``（或显式注入的 ``rng``）重放结果可复现——默认随机源
        由 :func:`_seed_from_sequence_id` 从 ``sequence.sequence_id`` 派生。

        Args:
            sequence: 待重放的 :class:`FrameSequence`。
            param_set: 透传给 ``Match_Runner`` 的参数集。
            profile: :class:`LocalizationProfile`，提供噪声 / 区间 / 回退 / 平滑 / 锁定阈值。
            runner: ``Match_Runner``，提供 ``run_case`` 与（可选）``result_store``。
            rng: 可选随机源（需提供 ``uniform``）；缺省时从 ``sequence_id`` 确定性派生。

        Returns:
            本次序列评估的 :class:`SequenceMetric`。
        """
        if rng is None:
            rng = random.Random(_seed_from_sequence_id(sequence.sequence_id))

        max_failures = max(1, int(getattr(profile, "fallback_max_failures", 2)))
        smooth_window = max(1, int(getattr(profile, "smooth_window", 1)))
        smooth_weights = getattr(profile, "smooth_weights", (1,))

        locked_map_id: Optional[str] = None
        lock_frame_index: Optional[int] = None
        consecutive_failures = 0
        recent_centers: List[Point] = []
        errors: List[float] = []
        fail_frames = 0
        wrong_lock_frames = 0

        for i, frame in enumerate(sequence.frames):
            ocr_game = simulate_ocr(frame, profile, rng)
            case = _FrameCase(case_id=frame.case_id, image_path=frame.image_path)

            locked = locked_map_id is not None
            search_map = locked_map_id if locked else frame.map_id
            in_fallback = consecutive_failures >= max_failures

            # 区间选择：已锁定且未进入回退态用小区间，否则全局匹配（Req 9.3, 9.5）。
            if locked and not in_fallback:
                region = _narrow_region(frame, ocr_game, profile)
            else:
                region = None

            out = _safe_run(runner, case, search_map, param_set, region)
            success = _is_success(out)

            if not success:
                if region is not None:
                    # 小区间失败：累计连续失败，达阈值则当帧解约束做全局回退（Req 9.5）。
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        out = _safe_run(runner, case, search_map, param_set, None)
                        success = _is_success(out)
                        if not success:
                            fail_frames += 1
                            continue
                    else:
                        # 软失败，尚未达回退阈值：不记定位失败、不记误差。
                        continue
                else:
                    # 全局匹配失败（未锁定，或已处于回退态）→ 定位失败（Req 9.5）。
                    consecutive_failures += 1
                    fail_frames += 1
                    continue

            # ---- 成功分支 ----
            consecutive_failures = 0
            matched_map = getattr(out, "map_id", None) or search_map

            # 锁定判定（Req 9.4, 9.8）。
            confidence = float(getattr(out, "confidence", 0.0) or 0.0)
            match_count = int(getattr(out, "match_count", 0) or 0)
            if should_lock(confidence, match_count, locked_map_id is not None, profile):
                locked_map_id = matched_map
                lock_frame_index = i

            # 错误锁定：当前生效地图与 Ground_Truth 不符（Req 9.9）。
            effective_map = locked_map_id if locked_map_id is not None else matched_map
            if effective_map != frame.map_id:
                wrong_lock_frames += 1

            # 平滑输出 + Error_Distance（Req 9.6）；缺几何信息（如复用视图）则跳过误差。
            game_center = getattr(out, "game_center", None)
            if game_center is not None:
                recent_centers.append((float(game_center[0]), float(game_center[1])))
                window = recent_centers[-smooth_window:]
                output_pos = smooth(window, smooth_weights)
                errors.append(game_distance(output_pos, frame.game_xy))

        mean_error = math.fsum(errors) / len(errors) if errors else 0.0
        max_error = max(errors) if errors else 0.0

        metric = SequenceMetric(
            sequence_id=sequence.sequence_id,
            profile_name=getattr(profile, "name", ""),
            param_set_name=getattr(param_set, "name", ""),
            map_id=sequence.map_id,
            mean_error=mean_error,
            max_error=max_error,
            fail_frames=fail_frames,
            wrong_lock_frames=wrong_lock_frames,
            lock_map_id=locked_map_id,
            lock_frame_index=lock_frame_index,
        )

        self._persist(metric, runner)
        return metric

    def _persist(self, metric: SequenceMetric, runner) -> None:
        """把序列指标写入 ``Result_Store.sequence_results``（若可用）；否则安全跳过。

        优先使用注入 ``runner`` 上的 ``result_store``，其次回退到本评估器的 ``store``。
        任何持久化异常都被吞掉，保证评估返回不受存储故障影响（Req 7 写入序列结果）。
        """
        store = getattr(runner, "result_store", None) or self.store
        if store is None:
            return
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            store.upsert_sequence(metric, ts)
        except Exception:  # noqa: BLE001 - 存储故障不应中断评估
            pass


__all__ = [
    "DEFAULT_REGION_SIZE",
    "Frame",
    "FrameSequence",
    "SequenceMetric",
    "build_sequence",
    "simulate_ocr",
    "smooth",
    "should_lock",
    "SequenceEvaluator",
]
