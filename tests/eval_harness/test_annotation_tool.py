"""Annotation 子系统集成 / 边界测试（Requirement 2, 3）。

覆盖 :class:`AnnotationTool` 的预览生成与提交工作流，以及
:class:`AnnotationStore` 的损坏保护语义。

为保持确定性并避免真实 OpenCV 特征提取 / 匹配的开销，本测试注入一个**桩引擎
工厂**（``stub engine_factory``）：桩引擎暴露 ``crop_size`` 属性与
``match(test_path) -> MatchOutput``，返回事先构造好的 :class:`MatchOutput`。
测试图与大地图均使用 numpy 合成的小图（经 ``cv2.imwrite`` 落盘），使
``build_preview_composite`` 能产出真实非空的预览图文件——既保证可靠性又无需
任何外部素材。

覆盖项：

- generate_previews：confidence>0.9 → 生成 ``{case_id}.png`` 且尺寸非零，
  写出边车 ``_candidates.json``（Req 2.1, 2.3, 2.4）。
- 已有 Ground_Truth 跳过；``redo=True`` 重新标注（Req 2.4, 2.5）。
- 无任何地图 confidence>0.9 → 记为待人工处理（Req 2.6）。
- commit：预览图删除 → 确认通过并写 GT；保留 → 拒绝不写（Req 3.1, 3.2）。
- commit：候选 ``map_scale ≤ 0`` 且预览删除 → 待人工处理、不写 GT（Req 3.4）。
- AnnotationStore 数据文件损坏 → 抛错且不覆盖原文件（Req 3.7）。

Validates: Requirements 2.1, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.4, 3.7
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from src.eval_harness.annotation import (
    CANDIDATES_FILENAME,
    AnnotationStore,
    AnnotationTool,
    GroundTruth,
    build_preview_composite,
)
from src.eval_harness.errors import HarnessError
from src.eval_harness.map_registry import MapEntry
from src.eval_harness.params import ParamSet, SurfParams
from src.match_engine.common import MatchOutput


# ---------------------------------------------------------------------------
# 测试替身与辅助
# ---------------------------------------------------------------------------
@dataclass
class _Case:
    """轻量 TestCase 替身，仅需 case_id 与 image_path（AnnotationTool 用 getattr）。"""

    case_id: str
    image_path: str


class _StubEngine:
    """确定性桩引擎：暴露 crop_size 并对任意 test_path 返回固定 MatchOutput。"""

    def __init__(self, output, crop_size: int = 128) -> None:
        self._output = output
        self.crop_size = crop_size

    def match(self, test_path):  # noqa: D401 - 桩实现，忽略输入路径
        return self._output


def _make_factory(outputs_by_map, crop_size: int = 128):
    """构造注入用的 engine_factory：按 map_id 返回对应桩引擎。"""

    def factory(map_entry, param_set, caches_dir):  # noqa: ANN001 - 签名对齐
        return _StubEngine(outputs_by_map.get(map_entry.map_id), crop_size)

    return factory


def _make_output(confidence: float = 0.95, map_scale: float = 1.0,
                 success: bool = True) -> MatchOutput:
    """构造一个匹配区域落在 256×256 地图内的 MatchOutput。"""
    return MatchOutput(
        success=success,
        match_count=20,
        inlier_count=int(round(20 * confidence)),
        confidence=confidence,
        center=(120.0, 120.0),
        corners=[(60.0, 60.0), (180.0, 60.0), (180.0, 180.0), (60.0, 180.0)],
        game_center=(1000.0, 2000.0),
        map_scale=map_scale,
        elapsed_ms=5.0,
        H=None,
    )


def _write_image(path: str, h: int, w: int) -> None:
    """用合成的随机彩色图写一张 PNG（确保 build_preview_composite 产出非空图）。"""
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(path, img)


def _surf_set() -> ParamSet:
    return ParamSet(
        algo="surf",
        params=SurfParams(
            hessian=300, octaves=4, layers=3,
            extended=False, upright=False,
            grid=8, max_per_cell=5, ratio=0.7, max_dist=0.3,
        ),
    )


def _map_entry(tmp_path, map_id: str) -> MapEntry:
    """写一张合成大地图并返回 MapEntry（coords=None，本流程不读取坐标）。"""
    image_path = os.path.join(str(tmp_path), f"map_{map_id}.png")
    _write_image(image_path, 256, 256)
    return MapEntry(map_id=map_id, image_path=image_path, coords=None)


def _make_case(tmp_path, case_id: str) -> _Case:
    image_path = os.path.join(str(tmp_path), f"{case_id}.png")
    _write_image(image_path, 128, 128)
    return _Case(case_id=case_id, image_path=image_path)


# ---------------------------------------------------------------------------
# 1. 预览生成：文件 + 边车（Req 2.1, 2.3, 2.4）
# ---------------------------------------------------------------------------
def test_generate_previews_creates_preview_and_sidecar(tmp_path):
    store = AnnotationStore(os.path.join(str(tmp_path), "annotations.json"))
    review_dir = os.path.join(str(tmp_path), "review")
    case = _make_case(tmp_path, "case1")
    entry = _map_entry(tmp_path, "m1")

    factory = _make_factory({"m1": _make_output(confidence=0.95)})
    tool = AnnotationTool(store, engine_factory=factory)

    report = tool.generate_previews([case], [entry], _surf_set(), review_dir)

    assert report.generated == ["case1"]
    assert not report.skipped and not report.needs_manual

    # 预览图生成、尺寸非零、可解码为非空图像（Req 2.1, 2.3）。
    preview_path = os.path.join(review_dir, "case1.png")
    assert os.path.isfile(preview_path)
    assert os.path.getsize(preview_path) > 0
    decoded = cv2.imread(preview_path)
    assert decoded is not None
    assert decoded.shape[0] > 0 and decoded.shape[1] > 0

    # 边车候选文件被写出（供 commit 复用）。
    assert os.path.isfile(os.path.join(review_dir, CANDIDATES_FILENAME))


def test_build_preview_composite_returns_nonempty_image():
    """直接验证 build_preview_composite 产出有效非空 BGR 图（Req 2.3）。"""
    test_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    map_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    composite = build_preview_composite(
        test_img, map_img, _make_output(), crop_size=128
    )
    assert composite is not None
    assert composite.ndim == 3 and composite.shape[2] == 3
    assert composite.shape[0] > 0 and composite.shape[1] > 0


# ---------------------------------------------------------------------------
# 2. 已有 GT 跳过；redo=True 重新标注（Req 2.4, 2.5）
# ---------------------------------------------------------------------------
def test_existing_gt_is_skipped_unless_redo(tmp_path):
    store_path = os.path.join(str(tmp_path), "annotations.json")
    store = AnnotationStore(store_path)
    # 预置一条已确认真值。
    store.upsert(GroundTruth(
        case_id="case1", map_id="m1",
        pixel_xy=(120.0, 120.0), game_xy=(1000.0, 2000.0),
        reference_scale=1.0,
    ))
    store.save()

    review_dir = os.path.join(str(tmp_path), "review")
    case = _make_case(tmp_path, "case1")
    entry = _map_entry(tmp_path, "m1")
    factory = _make_factory({"m1": _make_output(confidence=0.95)})
    tool = AnnotationTool(store, engine_factory=factory)

    # 默认：已有 GT 的用例被跳过，不生成预览（Req 2.5）。
    skipped_report = tool.generate_previews([case], [entry], _surf_set(), review_dir)
    assert skipped_report.skipped == ["case1"]
    assert skipped_report.generated == []
    assert not os.path.isfile(os.path.join(review_dir, "case1.png"))

    # redo=True：即便已有 GT 也重新标注（Req 2.5）。
    redo_report = tool.generate_previews(
        [case], [entry], _surf_set(), review_dir, redo=True
    )
    assert redo_report.generated == ["case1"]
    assert os.path.isfile(os.path.join(review_dir, "case1.png"))


# ---------------------------------------------------------------------------
# 3. 无任何地图 confidence>0.9 → 待人工处理（Req 2.6）
# ---------------------------------------------------------------------------
def test_no_high_confidence_map_records_needs_manual(tmp_path):
    store = AnnotationStore(os.path.join(str(tmp_path), "annotations.json"))
    review_dir = os.path.join(str(tmp_path), "review")
    case = _make_case(tmp_path, "case1")
    entry = _map_entry(tmp_path, "m1")

    # 匹配成功但 confidence ≤ 0.9 → 无候选 → 待人工处理。
    factory = _make_factory({"m1": _make_output(confidence=0.5)})
    tool = AnnotationTool(store, engine_factory=factory)

    report = tool.generate_previews([case], [entry], _surf_set(), review_dir)

    assert report.needs_manual == ["case1"]
    assert report.generated == []
    assert not os.path.isfile(os.path.join(review_dir, "case1.png"))


# ---------------------------------------------------------------------------
# 4. commit：删除→确认写 GT；保留→拒绝不写（Req 3.1, 3.2）
# ---------------------------------------------------------------------------
def test_commit_deleted_confirms_kept_rejects(tmp_path):
    store_path = os.path.join(str(tmp_path), "annotations.json")
    store = AnnotationStore(store_path)
    review_dir = os.path.join(str(tmp_path), "review")

    case_a = _make_case(tmp_path, "case_a")
    case_b = _make_case(tmp_path, "case_b")
    entry = _map_entry(tmp_path, "m1")
    factory = _make_factory({"m1": _make_output(confidence=0.95, map_scale=1.0)})
    tool = AnnotationTool(store, engine_factory=factory)

    gen = tool.generate_previews([case_a, case_b], [entry], _surf_set(), review_dir)
    assert set(gen.generated) == {"case_a", "case_b"}

    # 人工复核：删除 case_a 的预览（接受），保留 case_b（拒绝）。
    os.remove(os.path.join(review_dir, "case_a.png"))
    assert os.path.isfile(os.path.join(review_dir, "case_b.png"))

    report = tool.commit([case_a, case_b], review_dir)
    assert report.confirmed == ["case_a"]
    assert report.rejected == ["case_b"]
    assert report.needs_manual == []

    # 真值落盘：仅 case_a 写入 GT（Req 3.1），case_b 未写（Req 3.2）。
    reloaded = AnnotationStore(store_path).load()
    assert "case_a" in reloaded
    assert "case_b" not in reloaded
    gt = reloaded["case_a"]
    assert gt.map_id == "m1"
    assert gt.reference_scale == 1.0


# ---------------------------------------------------------------------------
# 5. commit：候选 map_scale ≤ 0 + 预览删除 → 待人工处理、不写 GT（Req 3.4）
# ---------------------------------------------------------------------------
def test_commit_nonpositive_scale_records_needs_manual(tmp_path):
    store_path = os.path.join(str(tmp_path), "annotations.json")
    store = AnnotationStore(store_path)
    review_dir = os.path.join(str(tmp_path), "review")

    case = _make_case(tmp_path, "case1")
    entry = _map_entry(tmp_path, "m1")
    # confidence>0.9 仍选为候选，但 map_scale ≤ 0。
    factory = _make_factory({"m1": _make_output(confidence=0.95, map_scale=0.0)})
    tool = AnnotationTool(store, engine_factory=factory)

    gen = tool.generate_previews([case], [entry], _surf_set(), review_dir)
    assert gen.generated == ["case1"]

    # 人工确认通过（删除预览图），但候选 scale ≤ 0 → 拒绝写入并记待人工处理。
    os.remove(os.path.join(review_dir, "case1.png"))
    report = tool.commit([case], review_dir)

    assert report.needs_manual == ["case1"]
    assert report.confirmed == []
    # 未写入任何 GT。
    assert AnnotationStore(store_path).load() == {}


# ---------------------------------------------------------------------------
# 6. AnnotationStore 损坏：抛错且不覆盖原文件（Req 3.7）
# ---------------------------------------------------------------------------
def test_corrupt_store_raises_and_preserves_file(tmp_path):
    store_path = os.path.join(str(tmp_path), "annotations.json")
    corrupt = "{ this is : not valid json ]]"
    with open(store_path, "w", encoding="utf-8") as f:
        f.write(corrupt)

    store = AnnotationStore(store_path)
    try:
        store.load()
        raised = False
    except HarnessError as exc:
        raised = True
        # 错误消息应指明文件路径（Req 3.7）。
        assert store_path in (exc.path or "") or store_path in str(exc)

    assert raised, "损坏的数据文件应触发 HarnessError"

    # 原文件内容未被覆盖（Req 3.7）。
    with open(store_path, "r", encoding="utf-8") as f:
        assert f.read() == corrupt
