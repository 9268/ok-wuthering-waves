"""Batch_Extractor 集成 / 烟雾测试（Requirement 5.1, 5.5）。

与 ``test_features_batch_examples.py`` 互补：后者用假 FeatureExtractor 覆盖
跳过 / force / 失败继续 / 汇总（Req 5.2–5.4）的编排语义；本文件聚焦真实路径——

- **集成（Req 5.1）**：用 **真实** :class:`FeatureExtractor` + cv2/SIFT，对一张
  合成的小地图原图提取，验证缓存确实落于 ``cache_path(...)`` 约定路径，且产物
  为可被 ``src.match_engine.common.load_npz`` 解析的合法 ``.npz``。
- **烟雾（Req 5.5）**：独立机器约束——``BatchExtractor`` 仅需 ``MapEntry``（源
  地图路径）+ ``ParamSet`` 列表 + ``caches_dir`` 即可构造并运行，不依赖
  Result_Store、截图目录或网络。

选用 SIFT 而非 SURF：SIFT 在标准 OpenCV 中可用，SURF 需 ``opencv-contrib``
的 ``xfeatures2d``，在部分环境不可用。若环境完全缺少 cv2，则集成测试优雅跳过
（``pytest.importorskip('cv2')``），但实现仍然保留。

Validates: Requirements 5.1, 5.5
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.eval_harness.features import (
    STATUS_OK,
    BatchExtractor,
    FeatureExtractor,
    cache_path,
)
from src.eval_harness.map_registry import MapEntry
from src.eval_harness.params import ParamSet, SiftParams


def _sift_set() -> ParamSet:
    """一个用于真实提取的 SIFT Param_Set（无需 opencv-contrib）。"""
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


def _write_textured_map(path: str, size: int = 256) -> None:
    """写出一张富纹理的合成大地图原图，确保 SIFT 能稳定提取到特征。

    用随机噪声块上采样得到大尺度纹理，再叠加若干随机形状（矩形 / 圆 / 线），
    使图像同时具备低频结构与高频角点，避免 SIFT 在过于平滑或纯高频噪声上
    提不到稳定关键点。
    """
    import cv2

    rng = np.random.default_rng(12345)
    # 低分辨率随机块上采样 → 大尺度斑块纹理。
    base = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
    img = cv2.resize(base, (size, size), interpolation=cv2.INTER_CUBIC)

    # 叠加随机形状制造角点 / 边缘。
    for _ in range(40):
        x1, y1 = rng.integers(0, size, 2)
        x2, y2 = rng.integers(0, size, 2)
        color = int(rng.integers(0, 256))
        kind = int(rng.integers(0, 3))
        if kind == 0:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
        elif kind == 1:
            radius = int(rng.integers(3, 20))
            cv2.circle(img, (int(x1), int(y1)), radius, color, -1)
        else:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # 轻度高频噪声进一步丰富细节。
    noise = rng.integers(0, 40, size=(size, size), dtype=np.uint8)
    img = cv2.add(img, noise)

    assert cv2.imwrite(path, img), f"无法写出合成地图 {path}"


def test_batch_extract_real_sift_writes_loadable_cache(tmp_path):
    """真实 FeatureExtractor + SIFT：缓存落于约定路径且为合法 .npz（Req 5.1）。"""
    pytest.importorskip("cv2")
    from src.match_engine.common import load_npz

    map_id = "synthetic_map"
    map_png = os.path.join(str(tmp_path), f"{map_id}.png")
    _write_textured_map(map_png)

    # coords 不被 FeatureExtractor / BatchExtractor 使用，置 None。
    entry = MapEntry(map_id=map_id, image_path=map_png, coords=None)
    ps = _sift_set()
    caches_dir = os.path.join(str(tmp_path), "caches")

    # 真实提取器（默认即真实 cv2 适配），无注入替身。
    report = BatchExtractor(extractor=FeatureExtractor()).run(
        [entry], [ps], caches_dir
    )

    # 单一组合成功落盘（Req 5.1）。
    assert len(report.entries) == 1
    only = report.entries[0]
    assert only.status == STATUS_OK, f"提取失败：{only.error}"

    # 缓存确实落于约定路径 cache_path(...)（Req 5.1）。
    expected = cache_path(caches_dir, ps.name, map_id, ps.algo)
    assert only.path == expected
    assert os.path.isfile(expected)

    # 产物为可被引擎 load_npz 解析的合法 .npz。
    kps, descs, map_h, map_w = load_npz(expected)
    assert (map_h, map_w) == (256, 256)
    assert len(kps) > 0, "合成地图未提取到任何特征"
    # 描述子数量与关键点数量一致，且 SIFT 描述子为 128 维。
    assert len(descs) == len(kps)
    assert len(descs[0]) == 128


def test_batch_extractor_independent_machine_inputs(tmp_path):
    """烟雾：仅凭 MapEntry + ParamSet + caches_dir 即可运行（Req 5.5）。

    独立机器约束——批量提取不依赖 Result_Store、截图目录或网络，构造与运行
    所需输入仅为「源地图条目 + 参数集列表 + 缓存根目录」。这里用一个不触碰
    cv2 的假提取器，专注断言「输入面」最小化这一部署约束本身。
    """
    calls = []

    class _RecordingExtractor:
        def extract(self, map_entry, param_set, out_path):
            calls.append((map_entry.map_id, param_set.name))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"npz-placeholder")

    # 仅源地图（带路径）+ 参数集 + 缓存目录，别无其它依赖。
    entry = MapEntry(map_id="m1", image_path="/src/m1.png", coords=None)
    ps = _sift_set()
    caches_dir = os.path.join(str(tmp_path), "caches")

    extractor = BatchExtractor(extractor=_RecordingExtractor())
    report = extractor.run([entry], [ps], caches_dir)

    # 运行成功，且确实只用到了所提供的三类输入。
    assert len(report.ok) == 1
    assert calls == [("m1", ps.name)]
    assert os.path.isfile(cache_path(caches_dir, ps.name, "m1", ps.algo))
