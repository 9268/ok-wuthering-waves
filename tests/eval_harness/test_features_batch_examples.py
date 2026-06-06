"""Batch_Extractor 批量提取示例测试（Requirement 5.1–5.5）。

用一个不依赖 cv2 / 引擎的假 FeatureExtractor 注入 :class:`BatchExtractor`，
覆盖：

- 每个 (map_id, Param_Set) 组合落盘到正确路径并记 ``ok``（Req 5.1）；
- 缓存已存在则跳过，``force=True`` 时重算（Req 5.2）；
- 单组合失败记录组合与原因并继续处理其余组合（Req 5.3）；
- 处理完后报告含每组合状态与路径汇总（Req 5.4）；
- 仅依赖 MapEntry + Param_Set 即可运行，无需测试机本地资源（Req 5.5）。

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
"""

from __future__ import annotations

import os

from src.eval_harness.features import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    BatchExtractor,
    cache_path,
)
from src.eval_harness.map_registry import MapEntry
from src.eval_harness.params import ParamSet, SiftParams, SurfParams


def _surf_set() -> ParamSet:
    return ParamSet(
        algo="surf",
        params=SurfParams(
            hessian=300,
            octaves=4,
            layers=3,
            extended=False,
            upright=False,
            grid=8,
            max_per_cell=5,
            ratio=0.7,
            max_dist=0.3,
        ),
    )


def _sift_set() -> ParamSet:
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


def _entry(map_id: str) -> MapEntry:
    # coords 不被 BatchExtractor 使用，置 None 即可（假提取器不读取）。
    return MapEntry(map_id=map_id, image_path=f"/src/{map_id}.png", coords=None)


class _FakeExtractor:
    """记录调用并向 out_path 写入占位文件的假提取器。"""

    def __init__(self, fail_for=()):
        self.calls = []
        self._fail_for = set(fail_for)

    def extract(self, map_entry, param_set, out_path):
        self.calls.append((map_entry.map_id, param_set.name, out_path))
        if map_entry.map_id in self._fail_for:
            raise MemoryError("内存不足，无法提取特征")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(b"npz-placeholder")


def test_run_extracts_every_combination_to_expected_path(tmp_path):
    """每个组合提取并落盘到 cache_path 指定位置，状态为 ok（Req 5.1, 5.4）。"""
    caches_dir = os.path.join(str(tmp_path), "caches")
    fake = _FakeExtractor()
    param_sets = [_surf_set(), _sift_set()]
    entries = [_entry("8"), _entry("910")]

    report = BatchExtractor(extractor=fake).run(entries, param_sets, caches_dir)

    assert len(report.entries) == 4
    assert len(report.ok) == 4
    assert not report.skipped and not report.failed
    for ps in param_sets:
        for me in entries:
            expected = cache_path(caches_dir, ps.name, me.map_id, ps.algo)
            assert os.path.isfile(expected)
            match = [e for e in report.ok if e.path == expected]
            assert len(match) == 1
            assert match[0].status == STATUS_OK
            assert match[0].map_id == me.map_id
            assert match[0].param_set_name == ps.name


def test_run_skips_existing_cache_unless_force(tmp_path):
    """缓存已存在则跳过；force=True 时重新提取（Req 5.2）。"""
    caches_dir = os.path.join(str(tmp_path), "caches")
    ps = _surf_set()
    me = _entry("8")

    # 预先放置缓存文件。
    existing = cache_path(caches_dir, ps.name, me.map_id, ps.algo)
    os.makedirs(os.path.dirname(existing), exist_ok=True)
    with open(existing, "wb") as f:
        f.write(b"old")

    fake = _FakeExtractor()
    report = BatchExtractor(extractor=fake).run([me], [ps], caches_dir)
    assert [e.status for e in report.entries] == [STATUS_SKIPPED]
    assert fake.calls == []  # 未触发提取

    fake_forced = _FakeExtractor()
    forced = BatchExtractor(extractor=fake_forced).run(
        [me], [ps], caches_dir, force=True
    )
    assert [e.status for e in forced.entries] == [STATUS_OK]
    assert len(fake_forced.calls) == 1  # 强制重算触发提取


def test_run_records_failure_and_continues(tmp_path):
    """单组合失败记录组合与原因并继续处理其余组合（Req 5.3, 5.4）。"""
    caches_dir = os.path.join(str(tmp_path), "caches")
    ps = _surf_set()
    entries = [_entry("ok1"), _entry("boom"), _entry("ok2")]
    fake = _FakeExtractor(fail_for={"boom"})

    report = BatchExtractor(extractor=fake).run(entries, [ps], caches_dir)

    assert len(report.entries) == 3
    assert {e.map_id for e in report.ok} == {"ok1", "ok2"}
    assert len(report.failed) == 1
    failed = report.failed[0]
    assert failed.map_id == "boom"
    assert failed.status == STATUS_FAILED
    assert "内存不足" in (failed.error or "")
    # 失败后续组合仍被处理（继续遍历）。
    assert fake.calls[-1][0] == "ok2"


def test_summary_counts(tmp_path):
    """summary 给出 ok/skipped/failed/total 计数（Req 5.4）。"""
    caches_dir = os.path.join(str(tmp_path), "caches")
    ps = _surf_set()
    entries = [_entry("a"), _entry("bad")]
    report = BatchExtractor(extractor=_FakeExtractor(fail_for={"bad"})).run(
        entries, [ps], caches_dir
    )
    summary = report.summary()
    assert "ok=1" in summary
    assert "failed=1" in summary
    assert "total=2" in summary
