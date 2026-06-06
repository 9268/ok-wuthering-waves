"""Report_Generator HTML 导出示例测试（Requirement 10.1, 10.4, 10.5）。

验证：

- 用预置的 Result_Store 数据生成 HTML 报告，文件被写入指定路径（Req 10.5）。
- 报告聚合所选 Param_Set 的指标，HTML 含 Param_Set 名称与关键指标表头
  （完成率 / 通过率 / 误差均值 等），并体现成功 / 通过的聚合（Req 10.1, 10.3）。
- ``by_map=True`` 时按 ``map_id`` 细分，HTML 出现各 map_id 的细分行（Req 10.4）。
"""

from __future__ import annotations

from src.eval_harness.report import _TABLE_HEADERS, ReportGenerator
from src.eval_harness.result_store import FrameMetric, ResultStore

# 通过阈值（与 metrics.PASS_THRESHOLD / SCALE_TOLERANCE 对齐）：
# error_distance <= 3320 且 scale_error <= 0.10 判定为通过。
_TS = "2026-01-01T00:00:00"

_PARAM_SET = "surf_default"


def _make_store() -> ResultStore:
    """构造内存库并预置若干 FrameMetric：跨 mapA/mapB，混合 成功/通过与失败。"""
    store = ResultStore(":memory:")

    rows = [
        # mapA：两条通过（成功），一条失败（误差过大）。
        FrameMetric(
            case_id="a1",
            param_set_name=_PARAM_SET,
            map_id="mapA",
            success=True,
            error_distance=100.0,
            scale_error=0.01,
            confidence=0.9,
            match_count=50,
            inlier_count=40,
            elapsed_ms=12.0,
        ),
        FrameMetric(
            case_id="a2",
            param_set_name=_PARAM_SET,
            map_id="mapA",
            success=True,
            error_distance=200.0,
            scale_error=0.02,
            confidence=0.8,
            match_count=45,
            inlier_count=30,
            elapsed_ms=15.0,
        ),
        FrameMetric(
            case_id="a3",
            param_set_name=_PARAM_SET,
            map_id="mapA",
            success=False,
            error_distance=9999.0,
            scale_error=0.5,
            confidence=0.3,
            match_count=10,
            inlier_count=2,
            elapsed_ms=20.0,
        ),
        # mapB：一条通过，一条失败（scale 误差超限，成功但不通过）。
        FrameMetric(
            case_id="b1",
            param_set_name=_PARAM_SET,
            map_id="mapB",
            success=True,
            error_distance=300.0,
            scale_error=0.03,
            confidence=0.85,
            match_count=60,
            inlier_count=48,
            elapsed_ms=18.0,
        ),
        FrameMetric(
            case_id="b2",
            param_set_name=_PARAM_SET,
            map_id="mapB",
            success=True,
            error_distance=400.0,
            scale_error=0.9,
            confidence=0.7,
            match_count=40,
            inlier_count=25,
            elapsed_ms=22.0,
        ),
    ]
    for m in rows:
        store.upsert_frame(m, _TS)
    return store


def test_generate_writes_html_with_param_set_and_metric_headers(tmp_path):
    """生成报告：文件存在（Req 10.5），含 Param_Set 名称与关键指标表头（Req 10.1）。"""
    store = _make_store()
    out_path = tmp_path / "report.html"

    html_text = ReportGenerator().generate(
        [_PARAM_SET], store, str(out_path), by_map=False
    )

    # Req 10.5：HTML 写入指定路径。
    assert out_path.exists()
    written = out_path.read_text(encoding="utf-8")
    assert written == html_text

    # Req 10.1：报告含所选 Param_Set 名称。
    assert _PARAM_SET in html_text

    # Req 10.3：关键指标表头出现（完成率 / 通过率 / 误差均值 等）。
    for header in _TABLE_HEADERS:
        assert header in html_text
    for key_label in ("完成率", "通过率", "误差均值", "Scale 误差均值", "elapsed_ms 均值"):
        assert key_label in html_text

    # 总体聚合：5 个 GT 用例、4 个成功、3 个通过（a1/a2/b1）。
    # 这些数值会作为表格单元格出现在 HTML 中。
    assert "<td>5</td>" in html_text  # GT 用例数
    assert "<td>4</td>" in html_text  # 成功数
    assert "<td>3</td>" in html_text  # 通过数

    store.close()


def test_generate_by_map_includes_per_map_rows(tmp_path):
    """by_map=True：HTML 出现各 map_id 的细分行（Req 10.4）。"""
    store = _make_store()
    out_path = tmp_path / "report_by_map.html"

    html_text = ReportGenerator().generate(
        [_PARAM_SET], store, str(out_path), by_map=True
    )

    assert out_path.exists()
    # Req 10.4：按 map_id 细分，两个地图标识都出现在报告中。
    assert "mapA" in html_text
    assert "mapB" in html_text
    # 细分行使用 by-map class。
    assert 'class="by-map"' in html_text

    store.close()


def test_generate_without_by_map_has_no_per_map_rows(tmp_path):
    """by_map=False：仅总体行，无 map_id 细分行（对照 Req 10.4）。"""
    store = _make_store()
    out_path = tmp_path / "report_overall.html"

    html_text = ReportGenerator().generate(
        [_PARAM_SET], store, str(out_path), by_map=False
    )

    assert 'class="by-map"' not in html_text
    # 总体行仍然存在。
    assert 'class="overall"' in html_text

    store.close()
