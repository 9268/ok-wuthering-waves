"""annotation：Annotation_Tool / Preview_Composite / Annotation_Store（Requirement 2, 3）。

负责一次性真值标注预览生成、人工复核后的真值提交，以及真值的结构化持久化
（``eval/annotations.json``）。

本模块当前实现 :class:`GroundTruth` 与 :class:`AnnotationStore`
（候选选取、Preview_Composite 与 AnnotationTool 见后续任务）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

from .errors import HarnessError

# Annotation_Store 文件格式版本（决策：结构化持久化）。
STORE_VERSION = 1


@dataclass
class GroundTruth:
    """某 Test_Case 经人工复核确认的真值（Req 3.3）。

    Attributes:
        case_id: 用例 ID（去扩展名文件名，全局唯一）。
        map_id: 匹配到的大地图标识。
        pixel_xy: 大地图像素坐标 ``(x, y)``。
        game_xy: 游戏坐标 ``(x, y)``。
        reference_scale: 标注阶段全图匹配返回的 ``MatchOutput.map_scale``，
            参考缩放比，必须 > 0（Req 3.4）。

    Raises:
        HarnessError: 当 ``reference_scale`` ≤ 0 或坐标不是二元组时。
    """

    case_id: str
    map_id: str
    pixel_xy: Tuple[float, float]
    game_xy: Tuple[float, float]
    reference_scale: float

    def __post_init__(self) -> None:
        # 规整坐标为浮点二元组，保证往返一致（JSON 列表 → 元组，Req 3.6）。
        object.__setattr__(self, "pixel_xy", self._coerce_xy(self.pixel_xy, "pixel_xy"))
        object.__setattr__(self, "game_xy", self._coerce_xy(self.game_xy, "game_xy"))
        if not isinstance(self.reference_scale, (int, float)) or isinstance(
            self.reference_scale, bool
        ):
            raise HarnessError(
                f"reference_scale 必须为数值，得到 {self.reference_scale!r}",
                name=str(self.case_id),
            )
        if not (self.reference_scale > 0):
            raise HarnessError(
                f"reference_scale 必须为正数（参考缩放比须 > 0），"
                f"得到 {self.reference_scale!r}",
                name=str(self.case_id),
            )

    @staticmethod
    def _coerce_xy(value: object, field: str) -> Tuple[float, float]:
        """把坐标规整为 ``(float, float)`` 二元组。"""
        try:
            x, y = value  # type: ignore[misc]
        except (TypeError, ValueError):
            raise HarnessError(
                f"字段 {field} 必须为二元坐标 (x, y)，得到 {value!r}"
            )
        return (float(x), float(y))

    def to_dict(self) -> dict:
        """序列化为 JSON 友好的字典（坐标以列表落盘）。"""
        return {
            "map_id": self.map_id,
            "pixel_xy": [self.pixel_xy[0], self.pixel_xy[1]],
            "game_xy": [self.game_xy[0], self.game_xy[1]],
            "reference_scale": self.reference_scale,
        }

    @staticmethod
    def from_dict(case_id: str, d: dict) -> "GroundTruth":
        """从字典反序列化为 :class:`GroundTruth`。

        Args:
            case_id: 该记录对应的用例 ID（来自 annotations 的键）。
            d: 单条真值的字段字典。

        Raises:
            HarnessError: 当结构非法或字段缺失时。
        """
        if not isinstance(d, dict):
            raise HarnessError(
                f"真值记录必须为字典，得到 {type(d).__name__}", name=str(case_id)
            )
        required = ("map_id", "pixel_xy", "game_xy", "reference_scale")
        missing = [k for k in required if k not in d]
        if missing:
            raise HarnessError(
                f"真值记录缺少必需字段：{missing}", name=str(case_id)
            )
        return GroundTruth(
            case_id=case_id,
            map_id=d["map_id"],
            pixel_xy=d["pixel_xy"],
            game_xy=d["game_xy"],
            reference_scale=d["reference_scale"],
        )


class AnnotationStore:
    """真值存储，持久化每个 Test_Case 的 Ground_Truth（Req 3.3, 3.5, 3.6, 3.7）。

    以结构化 JSON 文件（默认 ``eval/annotations.json``）持久化真值。采用
    "先解析后写"语义：:meth:`load` 在文件损坏时抛出含路径的错误且**不覆盖**
    原文件，避免破坏既有真值。

    Args:
        path: Annotation_Store 数据文件路径（``eval/annotations.json``）。
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._data: Dict[str, GroundTruth] = {}

    def load(self) -> Dict[str, GroundTruth]:
        """加载已有真值供各子系统读取（Req 3.5）。

        文件不存在视为空存储（返回空字典）。文件存在但无法解析时抛出含文件
        路径的 :class:`HarnessError` 且**不覆盖**原文件（Req 3.7）。

        Returns:
            ``case_id -> GroundTruth`` 的映射（内部数据的拷贝）。

        Raises:
            HarnessError: 当数据文件存在但无法解析时（消息含文件路径）。
        """
        if not os.path.exists(self.path):
            self._data = {}
            return dict(self._data)

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError(
                f"Annotation_Store 数据文件无法解析：{exc}", path=self.path
            )

        if not isinstance(raw, dict) or not isinstance(
            raw.get("annotations"), dict
        ):
            raise HarnessError(
                "Annotation_Store 数据文件结构非法："
                "缺少 'annotations' 字典字段",
                path=self.path,
            )

        data: Dict[str, GroundTruth] = {}
        for case_id, record in raw["annotations"].items():
            try:
                data[case_id] = GroundTruth.from_dict(case_id, record)
            except HarnessError as exc:
                # 单条记录损坏同样视为整体不可解析，且不覆盖原文件。
                raise HarnessError(
                    f"Annotation_Store 数据文件含非法真值记录：{exc.message}",
                    path=self.path,
                    name=str(case_id),
                )

        self._data = data
        return dict(self._data)

    def upsert(self, gt: GroundTruth) -> None:
        """插入或覆盖一条真值（以 ``case_id`` 为键）。

        Args:
            gt: 待写入的真值记录。
        """
        if not isinstance(gt, GroundTruth):
            raise HarnessError(
                f"upsert 需要 GroundTruth 实例，得到 {type(gt).__name__}"
            )
        self._data[gt.case_id] = gt

    def save(self) -> None:
        """将内存中的全部真值结构化写入数据文件（Req 3.3）。

        会按需创建父目录。写入采用约定的 ``{version, annotations}`` 结构，
        坐标以列表形式落盘，读回时恢复为元组（往返一致，Req 3.6）。
        """
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        payload = {
            "version": STORE_VERSION,
            "annotations": {
                case_id: gt.to_dict() for case_id, gt in self._data.items()
            },
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


# 模块公开符号。汇总 Task 11.1（GroundTruth / AnnotationStore）、
# Task 11.3（候选选取纯函数）与 Task 11.5（AnnotationTool / Preview_Composite /
# CommitReport）的全部公开 API。名字在 ``from ... import *`` 时按模块执行完毕后
# 解析，故此处可前置引用后文定义的符号。
__all__ = [
    # Task 11.1 —— 真值与存储
    "GroundTruth",
    "AnnotationStore",
    "STORE_VERSION",
    # Task 11.3 —— 候选选取纯函数
    "CONFIDENCE_THRESHOLD",
    "Candidate",
    "select_candidate",
    "needs_manual_handling",
    # Task 11.5 —— 标注工具 / 预览 / 提交
    "AnnotationTool",
    "CommitReport",
    "PreviewReport",
    "PREVIEW_EXPAND_PX",
    "CANDIDATES_FILENAME",
    "default_engine_factory",
    "build_preview_composite",
]

# ===========================================================================
# 候选选取逻辑（纯函数） — Task 11.3
#
# 说明：本段为标注阶段「候选选取」的纯逻辑实现，无任何 I/O，便于属性测试
# （见 design.md Property 2，tasks.md 11.4）。与本文件中
# GroundTruth / AnnotationStore（Task 11.1）相互独立，单独成节以避免编辑冲突。
# ===========================================================================

import dataclasses
from typing import Iterable, NamedTuple, Optional

#: 标注候选的 confidence 阈值，与 ``MapOverlayTask.CONFIDENCE_THRESHOLD`` 一致。
#: 注意比较为严格大于（``confidence > 0.9``），见 Requirement 2.1 / 2.2。
CONFIDENCE_THRESHOLD: float = 0.9


class Candidate(NamedTuple):
    """单个匹配候选（纯数据）。

    对应标注阶段对某一张大地图执行全图匹配得到的结果摘要。

    Attributes:
        map_id: 候选大地图标识。
        confidence: 匹配置信度（``MatchOutput.confidence``）。
        map_scale: 匹配返回的缩放比（``MatchOutput.map_scale``）。
    """

    map_id: str
    confidence: float
    map_scale: float


def _as_candidate(item: object) -> Candidate:
    """将一个候选项归一化为 :class:`Candidate`。

    接受 :class:`Candidate`、任意带 ``map_id`` / ``confidence`` / ``map_scale``
    属性的对象（如 dataclass），或 ``(map_id, confidence, map_scale)`` 形式的
    三元组 / 序列。

    Args:
        item: 待归一化的候选项。

    Returns:
        归一化后的 :class:`Candidate`。

    Raises:
        TypeError: 当 ``item`` 既无所需属性也不是长度为 3 的序列时。
    """
    if isinstance(item, Candidate):
        return item
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        return Candidate(
            getattr(item, "map_id"),
            float(getattr(item, "confidence")),
            float(getattr(item, "map_scale")),
        )
    if all(hasattr(item, attr) for attr in ("map_id", "confidence", "map_scale")):
        return Candidate(
            getattr(item, "map_id"),
            float(getattr(item, "confidence")),
            float(getattr(item, "map_scale")),
        )
    try:
        map_id, confidence, map_scale = item  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "候选项必须是 Candidate、带 map_id/confidence/map_scale 的对象，"
            "或 (map_id, confidence, map_scale) 三元组"
        ) from exc
    return Candidate(map_id, float(confidence), float(map_scale))


def select_candidate(
    candidates: Iterable[object],
    *,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Optional[Candidate]:
    """从匹配候选列表中选取标注候选（纯函数，无 I/O）。

    实现 Requirement 2.1 / 2.2 / 2.6 的候选选取语义（design.md Property 2）：

    - 仅保留 ``confidence`` **严格大于** ``confidence_threshold``（默认 0.9）
      的候选；
    - 当存在这样的候选时，返回其中 ``confidence`` **最高**的一项；
    - 当不存在任何 ``confidence > 阈值`` 的候选时，返回 ``None``（哨兵值），
      表示"无候选"——调用方应据此把该用例记为待人工处理（Req 2.6）。

    关于 ``map_scale``（Requirement 3.4）：候选选取本身只依据 confidence 规则，
    **不**因 ``map_scale ≤ 0`` 而排除候选。被选中的候选若 ``map_scale ≤ 0``，
    仍会被返回；提交真值时（``commit``）调用方必须检查 ``map_scale``，对
    ``map_scale ≤ 0`` 的候选拒绝写入 Ground_Truth 并记为待人工处理（参考缩放比
    必须为正）。可用 :func:`needs_manual_handling` 做此判断。

    Args:
        candidates: 候选项的可迭代集合；每项可为 :class:`Candidate`、带
            ``map_id`` / ``confidence`` / ``map_scale`` 属性的对象，或
            ``(map_id, confidence, map_scale)`` 三元组。
        confidence_threshold: confidence 阈值，默认 :data:`CONFIDENCE_THRESHOLD`
            （比较为严格大于）。

    Returns:
        confidence 最高且 ``> confidence_threshold`` 的 :class:`Candidate`；
        当无候选满足阈值时返回 ``None``。
    """
    best: Optional[Candidate] = None
    for raw in candidates:
        cand = _as_candidate(raw)
        if cand.confidence <= confidence_threshold:
            continue
        if best is None or cand.confidence > best.confidence:
            best = cand
    return best


def needs_manual_handling(candidate: Optional[Candidate]) -> bool:
    """判断某用例是否应记为"待人工处理"。

    两种情形需人工处理（Requirement 2.6 / 3.4）：

    - 没有任何满足 confidence 阈值的候选（``candidate is None``）；
    - 选中的候选 ``map_scale ≤ 0``（参考缩放比必须为正，否则拒绝写入
      Ground_Truth）。

    Args:
        candidate: :func:`select_candidate` 的返回值。

    Returns:
        ``True`` 表示该用例应记为待人工处理。
    """
    if candidate is None:
        return True
    return candidate.map_scale <= 0


# 注：本模块的 ``__all__`` 现集中维护于文件上方（Task 11.1 / 11.3 / 11.5 的
# 全部公开符号），候选选取相关符号 CONFIDENCE_THRESHOLD / Candidate /
# select_candidate / needs_manual_handling 已并入其中。


# ===========================================================================
# Preview_Composite + AnnotationTool + CommitReport — Task 11.5
#
# 本段实现一次性真值标注的「预览生成」与「提交」工作流（Requirement 2, 3）：
#
#   1. generate_previews：对尚无 Ground_Truth 的用例（除非 redo），逐一对**全部
#      已发现地图**执行全图匹配，仅保留 confidence > 0.9 的候选并取最高者
#      （复用 Task 11.3 的 select_candidate），生成一张 Preview_Composite 并以
#      ``{case_id}.png`` 命名保存到 Review_Folder（Req 2.1–2.5）。无候选者记为
#      待人工处理（Req 2.6）。同时把候选信息写入 Review_Folder 下的边车文件
#      ``_candidates.json``，供 commit 阶段无需重算即可写入真值。
#
#   2. commit：读取边车 ``_candidates.json``，对每个用例——
#        - 预览图**已被删除** → 人工确认通过 → 写入 Ground_Truth（Req 3.1）；
#        - 预览图**仍存在** → 拒绝，不写 GT（Req 3.2）；
#        - 候选 ``map_scale ≤ 0`` 且被确认通过 → 拒绝写入并记为待人工处理
#          （参考缩放比必须为正，Req 3.4）。
#      返回 :class:`CommitReport`（confirmed / rejected / needs_manual 计数）。
#
# 设计要点（design.md > Components > Annotation_Tool / 决策 4）：
#   - 重 cv2 / 引擎依赖均在方法内**延迟导入**，保持纯逻辑导入轻量。
#   - 引擎按 (map, Param_Set) 构造一次并在整批用例间复用；其 assets_dir 指向
#     ``{caches_dir}/{Param_Set_Name}`` 以复用 Feature_Cache 约定（features.py）。
#   - 边车文件把「候选 map_id / 像素坐标 / 游戏坐标 / map_scale」持久化，使
#     commit 与 generate 解耦——人工复核（删图）发生在两者之间。
# ===========================================================================

import math
from dataclasses import field
from typing import Callable, List, Optional as _Optional

from .map_registry import MapEntry
from .params import SIFT, SURF, ParamSet

#: Preview_Composite 中大地图局部相对匹配区域四向扩展的像素数（Req 2.2）。
PREVIEW_EXPAND_PX: int = 300

#: Review_Folder 下持久化候选信息的边车文件名，供 commit 读取（决策 4）。
CANDIDATES_FILENAME: str = "_candidates.json"

#: 边车文件结构版本。
CANDIDATES_VERSION: int = 1

#: 引擎工厂类型：``(map_entry, param_set, caches_dir) -> engine``，
#: engine 须暴露 ``match(test_path) -> MatchOutput`` 与 ``crop_size`` 属性。
EngineFactory = Callable[[MapEntry, ParamSet, _Optional[str]], object]


@dataclass
class PreviewReport:
    """:meth:`AnnotationTool.generate_previews` 的结果汇总。

    Attributes:
        generated: 成功生成 Preview_Composite 的用例 ID 列表。
        skipped: 因已存在 Ground_Truth 而跳过的用例 ID 列表（Req 2.5）。
        needs_manual: 无任何地图 ``confidence > 0.9`` 候选、记为待人工处理的
            用例 ID 列表（Req 2.6）。
        warnings: 处理过程中产生的告警（如某地图引擎构造失败、某用例匹配异常）。
    """

    generated: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    needs_manual: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def processed_count(self) -> int:
        """已处理（生成 + 待人工）的用例数，供 CLI 报告（Req 11.3）。"""
        return len(self.generated) + len(self.needs_manual)


@dataclass
class CommitReport:
    """:meth:`AnnotationTool.commit` 的结果汇总（Req 3.1, 3.2, 3.4）。

    Attributes:
        confirmed: 预览图已删除且候选合法、已写入 Ground_Truth 的用例 ID 列表。
        rejected: 预览图仍存在、被判定为拒绝（不写 GT）的用例 ID 列表。
        needs_manual: 虽被人工确认通过但候选 ``map_scale ≤ 0``、拒绝写入并记为
            待人工处理的用例 ID 列表（Req 3.4）。
    """

    confirmed: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    needs_manual: List[str] = field(default_factory=list)

    @property
    def confirmed_count(self) -> int:
        return len(self.confirmed)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def needs_manual_count(self) -> int:
        return len(self.needs_manual)


def default_engine_factory(
    map_entry: MapEntry, param_set: ParamSet, caches_dir: _Optional[str] = None
) -> object:
    """按 Param_Set 构造一个 :mod:`src.match_engine` 引擎（延迟导入）。

    把 Param_Set 的提取 / 匹配参数映射到引擎构造签名（见 ``surf.py`` /
    ``sift.py``）。引擎的 ``assets_dir`` 指向 ``{caches_dir}/{Param_Set_Name}``
    （与 :func:`features.cache_path` 的缓存布局一致）；若该处已存在
    ``{map_id}_{algo}.npz`` 则直接加载缓存，否则从大地图原图提取并落盘
    （标注为一次性流程，通常在性能机上执行）。当 ``caches_dir`` 为空时，
    退回到大地图原图所在目录作为 ``assets_dir``。

    Args:
        map_entry: 目标大地图注册项。
        param_set: 参数集（决定算法与构造参数）。
        caches_dir: Feature_Cache 根目录（可选）。

    Returns:
        构造好的 ``SurfEngine`` 或 ``SiftEngine`` 实例。

    Raises:
        HarnessError: 当算法非法或与参数类型不匹配时。
    """
    from src.match_engine import SiftEngine, SurfEngine  # noqa: PLC0415

    if caches_dir:
        assets_dir = os.path.join(caches_dir, param_set.name)
    else:
        assets_dir = os.path.dirname(os.path.abspath(map_entry.image_path))

    p = param_set.params
    if param_set.algo == SURF:
        return SurfEngine(
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
    if param_set.algo == SIFT:
        return SiftEngine(
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
    raise HarnessError(
        f"未知算法标识：{param_set.algo!r}（应为 'surf' 或 'sift'）",
        name=str(param_set.algo),
    )


def _ensure_bgr(img):
    """把灰度图升为 3 通道 BGR，便于叠加彩色标记。"""
    import cv2  # noqa: PLC0415

    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _test_crop_corners(tw: int, th: int, crop_size: int):
    """复现引擎 ``_prepare_test_image`` 的中心裁剪，返回裁剪框在原图中的四角。

    引擎在匹配前会把测试图按中心裁剪到 ``crop_size × crop_size``（仅当任一边
    大于 ``crop_size`` 时），单应性 ``H`` 与投影角点 ``corners`` 因此对应**裁剪后**
    的坐标系。本函数据此还原裁剪框四角（顺序与 ``project_corners`` 一致：
    左上、右上、右下、左下），用于在预览左图上绘制与右图投影角点对应的连线。
    """
    if crop_size > 0 and (tw > crop_size or th > crop_size):
        half = crop_size // 2
        cx, cy = tw // 2, th // 2
        left = cx - half
        top = cy - half
        right = left + crop_size
        bottom = top + crop_size
        return [(left, top), (right, top), (right, bottom), (left, bottom)]
    return [(0, 0), (tw, 0), (tw, th), (0, th)]


def build_preview_composite(test_img, map_img, match, crop_size: int = 350):
    """构造一张 Preview_Composite 图像（Req 2.2, 2.3）。

    左侧为测试图，右侧为大地图的匹配局部（在匹配区域基础上向上下左右各扩展
    :data:`PREVIEW_EXPAND_PX` 像素并裁剪），并叠加：

    - **特征点连线**：将测试图裁剪框四角与其在大地图上的投影角点
      （``match.corners``）相连，可视化单应性对应关系；
    - **投影中心十字**：在右图标出 ``match.center`` 对应位置；
    - **图例**：中心点地图像素坐标、中心点游戏坐标、缩放比、特征匹配数与耗时。

    本函数不对像素做精确断言，只保证产出一张有效的非空 BGR 图像。

    Args:
        test_img: 测试截图（灰度或 BGR 的 ndarray）。
        map_img: 大地图原图（灰度或 BGR 的 ndarray）。
        match: 该用例选中候选地图的 ``MatchOutput``（success=True）。
        crop_size: 匹配时使用的中心裁剪边长，用于还原测试图裁剪框，默认 350。

    Returns:
        合成后的 BGR 图像（``numpy.ndarray``）。
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    test_bgr = _ensure_bgr(test_img)
    map_bgr = _ensure_bgr(map_img)
    th, tw = test_bgr.shape[:2]
    mh, mw = map_bgr.shape[:2]

    corners = list(match.corners) if getattr(match, "corners", None) else []
    if corners:
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
    else:
        cx, cy = match.center if getattr(match, "center", None) else (mw / 2.0, mh / 2.0)
        minx = maxx = cx
        miny = maxy = cy

    # 匹配区域四向扩展 PREVIEW_EXPAND_PX 并裁剪到地图边界内。
    x0 = int(max(0, math.floor(minx - PREVIEW_EXPAND_PX)))
    y0 = int(max(0, math.floor(miny - PREVIEW_EXPAND_PX)))
    x1 = int(min(mw, math.ceil(maxx + PREVIEW_EXPAND_PX)))
    y1 = int(min(mh, math.ceil(maxy + PREVIEW_EXPAND_PX)))
    if x1 <= x0:
        x1 = min(mw, x0 + 1)
    if y1 <= y0:
        y1 = min(mh, y0 + 1)
    crop = map_bgr[y0:y1, x0:x1].copy()
    ch, cw = crop.shape[:2]

    # 将左右两幅缩放到统一高度后水平拼接。
    target_h = max(th, ch, 1)
    s_left = target_h / th if th else 1.0
    s_right = target_h / ch if ch else 1.0
    left = cv2.resize(test_bgr, (max(1, int(round(tw * s_left))), target_h))
    right = cv2.resize(crop, (max(1, int(round(cw * s_right))), target_h))
    left_w = left.shape[1]
    composite = cv2.hconcat([left, right])

    # 右图：投影角点四边形 + 中心十字。
    def to_right(mx, my):
        return (
            int(round((mx - x0) * s_right)) + left_w,
            int(round((my - y0) * s_right)),
        )

    if corners:
        quad = np.array([to_right(c[0], c[1]) for c in corners], dtype=np.int32)
        cv2.polylines(composite, [quad], isClosed=True, color=(255, 0, 0), thickness=2)

    if getattr(match, "center", None):
        rcx, rcy = to_right(match.center[0], match.center[1])
        cv2.drawMarker(
            composite, (rcx, rcy), color=(0, 0, 255),
            markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2,
        )

    # 特征点连线：测试图裁剪框四角 → 投影角点。
    if corners:
        tc = _test_crop_corners(tw, th, crop_size)
        for (txp, typ), (mxp, myp) in zip(tc, corners):
            lp = (int(round(txp * s_left)), int(round(typ * s_left)))
            rp = to_right(mxp, myp)
            cv2.line(composite, lp, rp, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(composite, lp, 4, (0, 255, 0), -1)
            cv2.circle(composite, rp, 4, (0, 255, 0), -1)

    # 图例文本块。
    center = getattr(match, "center", None)
    game = getattr(match, "game_center", None)
    px_txt = f"pixel=({center[0]:.1f}, {center[1]:.1f})" if center else "pixel=N/A"
    game_txt = f"game=({game[0]:.1f}, {game[1]:.1f})" if game else "game=N/A"
    legend = [
        px_txt,
        game_txt,
        f"map_scale={getattr(match, 'map_scale', 0.0):.4f}",
        f"matches={getattr(match, 'match_count', 0)} "
        f"(inliers={getattr(match, 'inlier_count', 0)})",
        f"elapsed_ms={getattr(match, 'elapsed_ms', 0.0):.1f}",
    ]
    line_h = 22
    box_h = line_h * len(legend) + 12
    box_w = 360
    cv2.rectangle(composite, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    for i, text in enumerate(legend):
        y = 22 + i * line_h
        cv2.putText(
            composite, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return composite


class AnnotationTool:
    """真值标注子系统：生成 Preview_Composite 并按人工复核结果提交 Ground_Truth。

    工作流分两步、以 Review_Folder 中预览图的「删除 / 保留」作为人工裁决信号：

    1. :meth:`generate_previews` —— 对尚无真值的用例匹配全部地图、取最高置信度
       候选、导出并排预览图，并把候选信息写入边车 ``_candidates.json``。
    2. 人工复核：删除匹配正确的预览图、保留错误的。
    3. :meth:`commit` —— 读取边车，预览图被删除者写入 GT、仍存在者拒绝。

    Args:
        store: :class:`AnnotationStore` 实例，用于跳过已标注用例与持久化真值。
        engine_factory: 可注入的引擎工厂 ``(map_entry, param_set, caches_dir)
            -> engine``；默认 :func:`default_engine_factory`（便于测试注入桩）。
        caches_dir: Feature_Cache 根目录（传给引擎工厂），可选。
    """

    def __init__(
        self,
        store: AnnotationStore,
        engine_factory: _Optional[EngineFactory] = None,
        caches_dir: _Optional[str] = None,
    ) -> None:
        self.store = store
        self.engine_factory = engine_factory or default_engine_factory
        self.caches_dir = caches_dir

    # ------------------------------------------------------------------
    # 预览生成（Req 2）
    # ------------------------------------------------------------------
    def generate_previews(
        self,
        cases,
        maps,
        param_set: ParamSet,
        review_dir: str,
        redo: bool = False,
    ) -> PreviewReport:
        """对尚无 GT 的用例生成 Preview_Composite（Req 2.1–2.6）。

        Args:
            cases: ``TestCase`` 可迭代集合（需有 ``case_id`` 与 ``image_path``）。
            maps: ``map_id -> MapEntry`` 映射或 ``MapEntry`` 可迭代集合。
            param_set: 用于全图匹配与引擎构造的参数集。
            review_dir: Review_Folder 路径；预览图与边车文件写入此处。
            redo: 为真时对已有 GT 的用例也重新标注（Req 2.5）。

        Returns:
            :class:`PreviewReport`，含生成 / 跳过 / 待人工处理列表与告警。
        """
        import cv2  # noqa: PLC0415

        report = PreviewReport()
        os.makedirs(review_dir, exist_ok=True)

        existing_gt = self.store.load()
        map_entries = self._normalize_maps(maps)

        # 为每张地图构造一次引擎，整批用例复用；构造失败记告警并跳过该地图。
        engines: dict = {}
        for entry in map_entries:
            try:
                engines[entry.map_id] = self.engine_factory(
                    entry, param_set, self.caches_dir
                )
            except Exception as exc:  # noqa: BLE001 - 记录并继续
                report.warnings.append(
                    f"地图 {entry.map_id} 引擎构造失败，跳过：{exc}"
                )

        entries_by_id = {e.map_id: e for e in map_entries}
        # 选中候选地图的原图缓存（按需加载，避免重复读盘）。
        map_img_cache: dict = {}
        # 持久化候选信息供 commit 使用。
        candidates: dict = {}

        for case in cases:
            case_id = getattr(case, "case_id")
            image_path = getattr(case, "image_path")

            if not redo and case_id in existing_gt:
                report.skipped.append(case_id)
                continue

            # 对全部地图执行匹配，收集成功结果与候选三元组。
            outputs: dict = {}
            cand_list: List[Candidate] = []
            for map_id, engine in engines.items():
                try:
                    out = engine.match(image_path)
                except Exception as exc:  # noqa: BLE001 - 单地图失败不终止
                    report.warnings.append(
                        f"用例 {case_id} 在地图 {map_id} 匹配异常：{exc}"
                    )
                    continue
                if out is not None and out.success:
                    outputs[map_id] = out
                    cand_list.append(
                        Candidate(map_id, out.confidence, out.map_scale)
                    )

            best = select_candidate(cand_list)
            if best is None:
                # 无任何 confidence > 0.9 的候选 → 待人工处理（Req 2.6）。
                report.needs_manual.append(case_id)
                continue

            match = outputs[best.map_id]
            entry = entries_by_id[best.map_id]

            if best.map_id not in map_img_cache:
                map_img_cache[best.map_id] = cv2.imread(entry.image_path)
            map_img = map_img_cache[best.map_id]
            test_img = cv2.imread(image_path)

            if map_img is None or test_img is None:
                report.warnings.append(
                    f"用例 {case_id} 预览跳过：无法读取测试图或地图原图"
                )
                report.needs_manual.append(case_id)
                continue

            crop_size = int(getattr(engines[best.map_id], "crop_size", 350))
            composite = build_preview_composite(
                test_img, map_img, match, crop_size
            )
            out_path = os.path.join(review_dir, f"{case_id}.png")
            cv2.imwrite(out_path, composite)

            # 记录候选供 commit 写入 GT（map_scale 可能 ≤ 0，由 commit 裁决）。
            center = match.center if match.center is not None else (0.0, 0.0)
            game = match.game_center if match.game_center is not None else (0.0, 0.0)
            candidates[case_id] = {
                "map_id": best.map_id,
                "pixel_xy": [float(center[0]), float(center[1])],
                "game_xy": [float(game[0]), float(game[1])],
                "map_scale": float(match.map_scale),
                "confidence": float(match.confidence),
                "match_count": int(match.match_count),
            }
            report.generated.append(case_id)

        self._write_candidates(review_dir, param_set, candidates)
        return report

    # ------------------------------------------------------------------
    # 真值提交（Req 3）
    # ------------------------------------------------------------------
    def commit(self, cases, review_dir: str) -> CommitReport:
        """根据人工复核结果提交 Ground_Truth（Req 3.1, 3.2, 3.4）。

        读取 Review_Folder 下的边车 ``_candidates.json``，对每个有候选的用例：

        - 预览图 ``{case_id}.png`` **已删除** → 人工确认通过：
          若候选 ``map_scale > 0`` 则写入 GT（confirmed，Req 3.1）；
          若 ``map_scale ≤ 0`` 则拒绝写入并记为待人工处理（Req 3.4）。
        - 预览图**仍存在** → 拒绝，不写 GT（rejected，Req 3.2）。

        仅处理同时出现在 ``cases`` 与边车候选中的用例。写入的真值会通过
        :meth:`AnnotationStore.save` 落盘。

        Args:
            cases: ``TestCase`` 可迭代集合（用于限定本次提交的用例范围）。
            review_dir: Review_Folder 路径。

        Returns:
            :class:`CommitReport`，含 confirmed / rejected / needs_manual 列表。

        Raises:
            HarnessError: 当边车 ``_candidates.json`` 缺失或无法解析时。
        """
        report = CommitReport()
        candidates = self._read_candidates(review_dir)

        # 确保最新的既有真值在内存中（避免覆盖其他用例的记录）。
        self.store.load()

        case_ids = [getattr(c, "case_id") for c in cases]
        wrote_any = False
        for case_id in case_ids:
            cand = candidates.get(case_id)
            if cand is None:
                # 本用例未生成候选（不在本批预览中），忽略。
                continue

            preview_path = os.path.join(review_dir, f"{case_id}.png")
            if os.path.exists(preview_path):
                # 预览图仍在 → 人工判定为错误 → 拒绝（Req 3.2）。
                report.rejected.append(case_id)
                continue

            # 预览图被删除 → 人工确认通过。
            map_scale = float(cand.get("map_scale", 0.0))
            if map_scale <= 0:
                # 参考缩放比必须为正，否则拒绝写入并记待人工处理（Req 3.4）。
                report.needs_manual.append(case_id)
                continue

            gt = GroundTruth(
                case_id=case_id,
                map_id=cand["map_id"],
                pixel_xy=tuple(cand["pixel_xy"]),
                game_xy=tuple(cand["game_xy"]),
                reference_scale=map_scale,
            )
            self.store.upsert(gt)
            wrote_any = True
            report.confirmed.append(case_id)

        if wrote_any:
            self.store.save()
        return report

    # ------------------------------------------------------------------
    # 边车文件 I/O 与辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_maps(maps) -> List[MapEntry]:
        """把 ``maps`` 规整为 ``MapEntry`` 列表（接受 dict 或可迭代）。"""
        if isinstance(maps, dict):
            return list(maps.values())
        return list(maps)

    @staticmethod
    def _write_candidates(review_dir: str, param_set: ParamSet, candidates: dict) -> None:
        """把候选信息写入 Review_Folder 下的边车 ``_candidates.json``。"""
        payload = {
            "version": CANDIDATES_VERSION,
            "param_set": param_set.name,
            "candidates": candidates,
        }
        path = os.path.join(review_dir, CANDIDATES_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _read_candidates(review_dir: str) -> dict:
        """读取边车 ``_candidates.json``，返回 ``case_id -> 候选信息`` 字典。

        Raises:
            HarnessError: 当文件缺失或无法解析时（消息含文件路径）。
        """
        path = os.path.join(review_dir, CANDIDATES_FILENAME)
        if not os.path.exists(path):
            raise HarnessError(
                "缺少候选边车文件，请先运行 generate_previews 生成预览",
                path=os.path.abspath(path),
            )
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError(
                f"候选边车文件无法解析：{exc}", path=os.path.abspath(path)
            )
        if not isinstance(raw, dict) or not isinstance(raw.get("candidates"), dict):
            raise HarnessError(
                "候选边车文件结构非法：缺少 'candidates' 字典字段",
                path=os.path.abspath(path),
            )
        return raw["candidates"]
