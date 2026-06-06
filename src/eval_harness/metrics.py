"""metrics：坐标换算、Error_Distance / Scale_Error 与单帧通过判定（Requirement 6）。

纯函数模块，适合属性测试。所有位置误差以游戏坐标度量
（``game = pixel × 83.008 + offset``）。

设计参考（design.md > Data Models > 坐标与常量）：

- ``game = pixel × 83.008 + offset``（per-axis，scale 全图统一为 83.008）。
- ``Pass_Threshold = 3320``（= 40 px × 83.008），``Scale_Tolerance = 0.10``。
- ``Scale_Error = |measured − reference| / reference``；``reference ≤ 0``
  或 ``measured ≤ 0`` → 未定义（以 ``None`` 表示）→ 判定为失败。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

# --- 常量（design.md > Data Models > 坐标与常量） ---------------------------

#: 大地图像素到游戏坐标的统一缩放比（全图 83.008）。
MAP_SCALE: float = 83.008

#: 判定单帧匹配"通过"的 Error_Distance 上限（游戏坐标单位，= 40 px × 83.008）。
PASS_THRESHOLD: float = 3320.0

#: 判定单帧匹配"通过"的 Scale_Error 上限（10%）。
SCALE_TOLERANCE: float = 0.10


Point = Tuple[float, float]


def pixel_to_game(
    px: float,
    py: float,
    offset: Point = (0.0, 0.0),
    scale: float = MAP_SCALE,
) -> Point:
    """将地图像素坐标换算为游戏坐标（per-axis）。

    换算公式为 ``game = pixel × scale + offset``，与
    :meth:`src.match_engine.common.CoordsRef.pixel_to_game` 一致，但此处
    全图统一使用标量 ``scale``（默认 :data:`MAP_SCALE`）。

    Args:
        px: 地图像素 X 坐标。
        py: 地图像素 Y 坐标。
        offset: 游戏坐标偏移 ``(offset_x, offset_y)``，默认 ``(0, 0)``。
        scale: 缩放比，默认 :data:`MAP_SCALE`。

    Returns:
        换算后的游戏坐标 ``(gx, gy)``。
    """
    gx = px * scale + offset[0]
    gy = py * scale + offset[1]
    return gx, gy


def game_distance(a: Point, b: Point) -> float:
    """计算两个游戏坐标点之间的欧氏距离。

    Args:
        a: 第一个游戏坐标点 ``(gx, gy)``。
        b: 第二个游戏坐标点 ``(gx, gy)``。

    Returns:
        两点的欧氏距离（游戏坐标单位）。
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def error_distance(
    measured_pixel: Point,
    reference_pixel: Point,
    offset: Point = (0.0, 0.0),
    scale: float = MAP_SCALE,
) -> float:
    """计算 Error_Distance：两个地图像素点换算到游戏坐标后的欧氏距离。

    由于换算为 ``game = pixel × scale + offset``，公共的 ``offset`` 在求差时
    抵消，因此结果等价于"像素欧氏距离 × scale"，与 ``offset`` 无关
    （见 design.md Property 5）。

    Args:
        measured_pixel: 匹配得到的地图像素坐标 ``(px, py)``。
        reference_pixel: Ground_Truth 的地图像素坐标 ``(px, py)``。
        offset: 游戏坐标偏移，默认 ``(0, 0)``（不影响结果）。
        scale: 缩放比，默认 :data:`MAP_SCALE`。

    Returns:
        Error_Distance（游戏坐标单位）。
    """
    measured_game = pixel_to_game(measured_pixel[0], measured_pixel[1], offset, scale)
    reference_game = pixel_to_game(reference_pixel[0], reference_pixel[1], offset, scale)
    return game_distance(measured_game, reference_game)


def scale_error(measured: float, reference: float) -> Optional[float]:
    """计算 Scale_Error：``|measured − reference| / reference``。

    当 ``measured ≤ 0`` 或 ``reference ≤ 0`` 时，Scale_Error 未定义，
    返回 ``None``（参考缩放比与测量缩放比都必须为正）。

    Args:
        measured: 匹配返回的 ``MatchOutput.map_scale``。
        reference: Ground_Truth 的参考缩放比。

    Returns:
        相对偏差；当 ``measured`` 或 ``reference`` ≤ 0 时返回 ``None``。
    """
    if measured <= 0 or reference <= 0:
        return None
    return abs(measured - reference) / reference


def is_pass(
    error_dist: Optional[float],
    scale_err: Optional[float],
    *,
    pass_threshold: float = PASS_THRESHOLD,
    scale_tolerance: float = SCALE_TOLERANCE,
) -> bool:
    """单帧通过判定。

    当且仅当 Error_Distance 与 Scale_Error 均已定义，且
    ``error_dist ≤ pass_threshold`` **且** ``scale_err ≤ scale_tolerance``
    时判定为通过。任一项未定义（``None``，如匹配失败或 scale 未定义）
    均判定为失败。

    Args:
        error_dist: Error_Distance；未定义时为 ``None``。
        scale_err: Scale_Error；未定义时为 ``None``。
        pass_threshold: Error_Distance 上限，默认 :data:`PASS_THRESHOLD`。
        scale_tolerance: Scale_Error 上限，默认 :data:`SCALE_TOLERANCE`。

    Returns:
        ``True`` 表示通过，``False`` 表示失败。
    """
    if error_dist is None or scale_err is None:
        return False
    return error_dist <= pass_threshold and scale_err <= scale_tolerance


def evaluate_frame(
    success: bool,
    measured_pixel: Optional[Point],
    reference_pixel: Optional[Point],
    measured_scale: float,
    reference_scale: float,
    offset: Point = (0.0, 0.0),
    scale: float = MAP_SCALE,
) -> Tuple[bool, Optional[float], Optional[float]]:
    """综合单帧评估：计算 Error_Distance、Scale_Error 并判定通过 / 失败。

    实现 Requirement 6 的判定语义：

    - 匹配未成功（``success`` 为假，或缺少坐标）→ 判定失败，
      Error_Distance 与 Scale_Error 均记为未定义（``None``）（Req 6.4）。
    - ``measured_scale ≤ 0`` 或 ``reference_scale ≤ 0`` → 判定失败，
      Scale_Error 记为未定义（Req 6.5）。
    - 否则换算游戏坐标算 Error_Distance、算 Scale_Error，并按
      :func:`is_pass` 判定（Req 6.2, 6.3）。

    Args:
        success: 匹配是否成功。
        measured_pixel: 匹配得到的地图像素坐标；失败时可为 ``None``。
        reference_pixel: Ground_Truth 的地图像素坐标。
        measured_scale: 匹配返回的 ``map_scale``。
        reference_scale: Ground_Truth 的参考缩放比。
        offset: 游戏坐标偏移，默认 ``(0, 0)``。
        scale: 缩放比，默认 :data:`MAP_SCALE`。

    Returns:
        三元组 ``(passed, error_dist, scale_err)``，其中 ``error_dist`` 与
        ``scale_err`` 在未定义时为 ``None``。
    """
    if not success or measured_pixel is None or reference_pixel is None:
        return False, None, None

    error_dist = error_distance(measured_pixel, reference_pixel, offset, scale)
    scale_err = scale_error(measured_scale, reference_scale)
    return is_pass(error_dist, scale_err), error_dist, scale_err


__all__ = [
    "MAP_SCALE",
    "PASS_THRESHOLD",
    "SCALE_TOLERANCE",
    "pixel_to_game",
    "game_distance",
    "error_distance",
    "scale_error",
    "is_pass",
    "evaluate_frame",
]
