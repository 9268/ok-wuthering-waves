"""Report_Generator：跨用例与跨 Param_Set 聚合并导出 HTML 报告（Requirement 10）。

从 Result_Store 读取指标，聚合完成率 / 通过率 / 误差均值最大值 / Scale_Error
均值 / elapsed 均值，可按 map_id 细分，导出 HTML 到 ``eval/reports/``。

本模块的聚合统计为**纯函数**（无 I/O），便于属性测试（任务 16.2 / 16.3）：

- :func:`aggregate` —— 忽略未定义项（``None``）后计算 mean/max/min（Property 14）。
- :func:`completion_rate` / :func:`pass_rate` —— 以 GT 用例数为分母的比率（Property 15）。
- :func:`aggregate_frame_metrics` —— 在一组 :class:`FrameMetric` 上汇总成功数 /
  通过数与各项统计，支撑 Req 10.3。

HTML 导出（``ReportGenerator.generate``）属于另一任务（16.4），此处不实现。
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from src.eval_harness.metrics import (
    PASS_THRESHOLD,
    SCALE_TOLERANCE,
    is_pass,
)
from src.eval_harness.result_store import FrameMetric


@dataclass(frozen=True)
class AggregateStats:
    """一组数值（忽略未定义项后）的聚合统计。

    Attributes:
        count: 参与统计的已定义值个数（已忽略 ``None``）。
        mean: 平均值；当无任何已定义值时为 ``None``。
        maximum: 最大值；当无任何已定义值时为 ``None``。
        minimum: 最小值；当无任何已定义值时为 ``None``。
    """

    count: int
    mean: Optional[float]
    maximum: Optional[float]
    minimum: Optional[float]


def aggregate(values: Iterable[Optional[float]]) -> AggregateStats:
    """对一组可选数值计算 mean / max / min，忽略未定义项（``None``）。

    设计决策（空 / 全 ``None`` 输入）：当不存在任何已定义值时，``mean`` /
    ``maximum`` / ``minimum`` 均返回 ``None``（而非 ``NaN`` 或 ``0``），
    ``count`` 为 ``0``。这样调用方（如 HTML 报告）可显式渲染"无数据"，
    且避免引入 ``NaN`` 破坏后续比较。

    Property 14（design.md）：当至少存在一个已定义值时，
    ``minimum <= mean <= maximum``。

    Args:
        values: 可选浮点数序列；``None`` 表示未定义项，将被忽略。

    Returns:
        :class:`AggregateStats`，包含已定义值的数量与 mean/max/min。
    """
    defined: List[float] = [float(v) for v in values if v is not None]
    if not defined:
        return AggregateStats(count=0, mean=None, maximum=None, minimum=None)
    return AggregateStats(
        count=len(defined),
        mean=sum(defined) / len(defined),
        maximum=max(defined),
        minimum=min(defined),
    )


def completion_rate(success_count: int, gt_count: int) -> float:
    """完成率 = 成功用例数 / GT 用例数（Req 10.2，决策 15）。

    分母为具有 Ground_Truth 的用例数。设计决策：当 ``gt_count == 0`` 时
    没有可评估用例，比率定义为 ``0.0``（避免除零）。

    Args:
        success_count: 匹配成功的用例数。
        gt_count: 具有 Ground_Truth 的用例数（分母）。

    Returns:
        完成率，落在 ``[0.0, 1.0]``（当入参满足 ``0 <= success_count <= gt_count``）。
    """
    if gt_count <= 0:
        return 0.0
    return success_count / gt_count


def pass_rate(pass_count: int, gt_count: int) -> float:
    """通过率 = 通过用例数 / GT 用例数（Req 10.2，决策 15）。

    分母为具有 Ground_Truth 的用例数。设计决策：当 ``gt_count == 0`` 时
    比率定义为 ``0.0``（避免除零）。

    Property 15（design.md）：通过蕴含成功，故当
    ``pass_count <= success_count <= gt_count`` 时
    ``0 <= pass_rate <= completion_rate <= 1``。

    Args:
        pass_count: 通过判定的用例数（通过蕴含成功）。
        gt_count: 具有 Ground_Truth 的用例数（分母）。

    Returns:
        通过率，落在 ``[0.0, 1.0]``（当入参满足 ``0 <= pass_count <= gt_count``）。
    """
    if gt_count <= 0:
        return 0.0
    return pass_count / gt_count


@dataclass(frozen=True)
class MetricSummary:
    """一组 :class:`FrameMetric` 的汇总结果（支撑 Req 10.2 / 10.3）。

    Attributes:
        gt_count: 参与汇总的用例数（即具有 Ground_Truth 的用例数，分母）。
        success_count: ``success`` 为真的用例数。
        pass_count: 通过判定（:func:`metrics.is_pass`）的用例数。
        completion_rate: 完成率（:func:`completion_rate`）。
        pass_rate: 通过率（:func:`pass_rate`）。
        error_distance: Error_Distance 的聚合统计（忽略未定义项）。
        scale_error: Scale_Error 的聚合统计（忽略未定义项）。
        elapsed_ms: elapsed_ms 的聚合统计（忽略未定义项）。
    """

    gt_count: int
    success_count: int
    pass_count: int
    completion_rate: float
    pass_rate: float
    error_distance: AggregateStats
    scale_error: AggregateStats
    elapsed_ms: AggregateStats


def aggregate_frame_metrics(
    metrics: Sequence[FrameMetric],
    *,
    pass_threshold: float = PASS_THRESHOLD,
    scale_tolerance: float = SCALE_TOLERANCE,
) -> MetricSummary:
    """在一组单帧指标上汇总成功数 / 通过数与各项统计（Req 10.2, 10.3）。

    入参 ``metrics`` 视为某 Param_Set 在一组**具有 Ground_Truth 的用例**上的
    结果集合，因此其长度即为 GT 用例数（分母）。通过判定复用
    :func:`metrics.is_pass`（要求 Error_Distance 与 Scale_Error 均已定义且
    分别 ``<= pass_threshold`` / ``<= scale_tolerance``）。

    Args:
        metrics: 单帧指标集合。
        pass_threshold: Error_Distance 通过上限，默认 :data:`metrics.PASS_THRESHOLD`。
        scale_tolerance: Scale_Error 通过上限，默认 :data:`metrics.SCALE_TOLERANCE`。

    Returns:
        :class:`MetricSummary`。
    """
    gt_count = len(metrics)
    success_count = sum(1 for m in metrics if m.success)
    pass_count = sum(
        1
        for m in metrics
        if is_pass(
            m.error_distance,
            m.scale_error,
            pass_threshold=pass_threshold,
            scale_tolerance=scale_tolerance,
        )
    )
    return MetricSummary(
        gt_count=gt_count,
        success_count=success_count,
        pass_count=pass_count,
        completion_rate=completion_rate(success_count, gt_count),
        pass_rate=pass_rate(pass_count, gt_count),
        error_distance=aggregate(m.error_distance for m in metrics),
        scale_error=aggregate(m.scale_error for m in metrics),
        elapsed_ms=aggregate(m.elapsed_ms for m in metrics),
    )


__all__ = [
    "AggregateStats",
    "MetricSummary",
    "aggregate",
    "completion_rate",
    "pass_rate",
    "aggregate_frame_metrics",
    "ReportGenerator",
]


# 报告中未定义聚合值（None）的占位文本。
_NA = "N/A"

# 报告表头（与 Req 10.2 / 10.3 字段对应）。
_TABLE_HEADERS = (
    "分组",
    "GT 用例数",
    "成功数",
    "通过数",
    "完成率",
    "通过率",
    "误差均值",
    "误差最大值",
    "Scale 误差均值",
    "elapsed_ms 均值",
)


def _fmt_num(value: Optional[float], ndigits: int = 3) -> str:
    """格式化可选数值；未定义（``None``）渲染为 'N/A'（Req 10.3 优雅降级）。"""
    if value is None:
        return _NA
    return f"{value:.{ndigits}f}"


def _fmt_rate(value: Optional[float], ndigits: int = 2) -> str:
    """将比率格式化为百分比字符串；``None`` 渲染为 'N/A'。"""
    if value is None:
        return _NA
    return f"{value * 100:.{ndigits}f}%"


def _group_by_map(
    metrics: Sequence[FrameMetric],
) -> "Dict[Optional[str], List[FrameMetric]]":
    """按 ``map_id`` 分组（``None`` 作为独立分组，标记为"未标注地图"）。"""
    groups: Dict[Optional[str], List[FrameMetric]] = {}
    for m in metrics:
        groups.setdefault(m.map_id, []).append(m)
    return groups


class ReportGenerator:
    """跨用例与跨 Param_Set 聚合评估指标并导出 HTML 报告（Requirement 10）。

    仅依赖标准库 :mod:`html` 与字符串模板渲染，不引入任何外部框架。聚合
    逻辑完全复用本模块的纯函数（:func:`aggregate_frame_metrics` 等）。
    """

    def generate(
        self,
        param_set_names: Sequence[str],
        store,
        out_path: str,
        by_map: bool = False,
    ) -> str:
        """聚合所选 Param_Set 的指标并导出 HTML 报告。

        对 ``param_set_names`` 中的每个 Param_Set，从 ``store`` 读取其单帧指标
        （Req 10.1），计算完成率 / 通过率 / 误差均值最大值 / Scale_Error 均值 /
        elapsed_ms 均值（Req 10.2, 10.3）。当 ``by_map`` 为真时，额外按 ``map_id``
        细分给出上述指标（Req 10.4）。最终将 HTML 写入 ``out_path``（其父目录
        若不存在则创建；通常位于 ``eval/reports/``），并返回 HTML 字符串（Req 10.5）。

        Args:
            param_set_names: 需纳入报告的 Param_Set 名称序列。
            store: 提供 ``query_by_param_set(name) -> list[FrameMetric]`` 的结果存储。
            out_path: HTML 输出路径（父目录会被自动创建）。
            by_map: 为真时按 ``map_id`` 细分。

        Returns:
            写入文件的完整 HTML 字符串。
        """
        sections: List[str] = []
        for name in param_set_names:
            metrics = list(store.query_by_param_set(name))
            sections.append(self._render_param_set(name, metrics, by_map))

        document = self._render_document(sections)

        parent = os.path.dirname(os.path.abspath(out_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(document)
        return document

    # ----- 渲染辅助 -----------------------------------------------------

    def _render_param_set(
        self, name: str, metrics: Sequence[FrameMetric], by_map: bool
    ) -> str:
        """渲染单个 Param_Set 的小节（含总体行，可选按地图细分子行）。"""
        overall = aggregate_frame_metrics(metrics)
        rows: List[str] = [self._render_row("（全部）", overall, is_overall=True)]

        if by_map:
            groups = _group_by_map(metrics)
            for map_id in sorted(groups, key=lambda k: (k is None, k or "")):
                label = map_id if map_id is not None else "（未标注地图）"
                summary = aggregate_frame_metrics(groups[map_id])
                rows.append(self._render_row(label, summary, is_overall=False))

        header_cells = "".join(
            f"<th>{html.escape(h)}</th>" for h in _TABLE_HEADERS
        )
        return (
            f'<section class="param-set">\n'
            f"  <h2>Param_Set: {html.escape(name)}</h2>\n"
            f'  <table>\n'
            f"    <thead><tr>{header_cells}</tr></thead>\n"
            f"    <tbody>\n{''.join(rows)}    </tbody>\n"
            f"  </table>\n"
            f"</section>\n"
        )

    def _render_row(
        self, label: str, summary: MetricSummary, *, is_overall: bool
    ) -> str:
        """渲染一行指标（总体行或按地图细分子行）。"""
        cells = [
            html.escape(label),
            str(summary.gt_count),
            str(summary.success_count),
            str(summary.pass_count),
            _fmt_rate(summary.completion_rate),
            _fmt_rate(summary.pass_rate),
            _fmt_num(summary.error_distance.mean),
            _fmt_num(summary.error_distance.maximum),
            _fmt_num(summary.scale_error.mean),
            _fmt_num(summary.elapsed_ms.mean),
        ]
        cls = ' class="overall"' if is_overall else ' class="by-map"'
        tds = "".join(f"<td>{c}</td>" for c in cells)
        return f"      <tr{cls}>{tds}</tr>\n"

    def _render_document(self, sections: Sequence[str]) -> str:
        """将各 Param_Set 小节包裹为完整 HTML 文档。"""
        body = "".join(sections) if sections else "<p>无可用数据。</p>\n"
        style = (
            "body{font-family:system-ui,Arial,sans-serif;margin:2rem;}"
            "table{border-collapse:collapse;margin-bottom:1.5rem;}"
            "th,td{border:1px solid #ccc;padding:4px 8px;text-align:right;}"
            "th:first-child,td:first-child{text-align:left;}"
            "tr.overall{font-weight:bold;background:#f0f0f0;}"
            "tr.by-map td:first-child{padding-left:1.5rem;}"
        )
        return (
            "<!DOCTYPE html>\n"
            '<html lang="zh">\n'
            "<head>\n"
            '  <meta charset="utf-8">\n'
            "  <title>Map Match Eval Report</title>\n"
            f"  <style>{style}</style>\n"
            "</head>\n"
            "<body>\n"
            "  <h1>地图匹配评估报告</h1>\n"
            f"{body}"
            "</body>\n"
            "</html>\n"
        )
