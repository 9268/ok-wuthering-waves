"""Harness_CLI：纯命令行入口（Requirement 11）。

提供 ``annotate-preview``、``annotate-commit``、``batch-extract``、``eval-frame``、
``eval-sequence``、``report`` 等子命令，编排各子系统操作；顶层捕获 HarnessError。

设计要点：

- ``main(argv=None) -> int`` 为统一入口。argparse 解析阶段 **不** 触发任何重依赖
  （cv2 / match_engine 引擎）：所有依赖 cv2 的子系统模块（map_registry / features /
  annotation / runner）均在各子命令处理函数内部 **延迟导入**，因此
  ``cli.py --help`` 与子命令 ``--help`` 在缺少 cv2 的环境下也可用（决策：纯命令行）。
- 每个子命令先校验必需输入，缺失时抛出指明缺失项的 :class:`HarnessError`（Req 11.2）；
  完成后向标准输出报告已处理的用例 / 组合数量与失败数量（Req 11.3）。
- 顶层捕获 :class:`HarnessError`，以非零退出码与清晰文案写入 stderr（Req 11.1, 11.3）。

产物目录约定（design.md / Resolved Decisions 7）：

- ``eval/caches/``：特征缓存，按 Param_Set_Name 分目录。
- ``eval/results.db``：SQLite 结果数据库。
- ``eval/review/``：标注预览复核目录。
- ``eval/reports/``：HTML 报告输出目录。
- ``eval/annotations.json``：Annotation_Store 真值存储。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

# 仅纯逻辑依赖在模块顶层导入；errors 不依赖 cv2，供顶层异常处理使用。
from .errors import HarnessError

# ---------------------------------------------------------------------------
# 产物目录默认约定（均可由命令行参数覆盖）。
# ---------------------------------------------------------------------------
DEFAULT_EVAL_DIR = "eval"
DEFAULT_SCREENSHOTS_DIR = "screenshots"
DEFAULT_STITCHED_DIR = os.path.join("assets", "stitched")
DEFAULT_CACHES_DIR = os.path.join(DEFAULT_EVAL_DIR, "caches")
DEFAULT_RESULTS_DB = os.path.join(DEFAULT_EVAL_DIR, "results.db")
DEFAULT_REVIEW_DIR = os.path.join(DEFAULT_EVAL_DIR, "review")
DEFAULT_REPORTS_DIR = os.path.join(DEFAULT_EVAL_DIR, "reports")
DEFAULT_ANNOTATIONS = os.path.join(DEFAULT_EVAL_DIR, "annotations.json")
DEFAULT_REPORT_OUT = os.path.join(DEFAULT_REPORTS_DIR, "report.html")


# ===========================================================================
# 输入加载 / 校验辅助（纯逻辑 + 文件 I/O，不触发 cv2）
# ===========================================================================
def _read_json(path: str, what: str) -> object:
    """读取一个 JSON 文件，缺失 / 损坏时抛出指明路径的 HarnessError。"""
    if not path:
        raise HarnessError(f"缺少必需输入：{what}")
    if not os.path.isfile(path):
        raise HarnessError(f"{what} 文件不存在", path=os.path.abspath(path))
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(
            f"{what} 文件无法解析：{exc}", path=os.path.abspath(path)
        ) from exc


def _load_param_sets(path: Optional[str]):
    """从 JSON 文件加载 Param_Set 定义列表（Req 8.4）。

    文件结构可为：

    - 顶层为列表：``[{"algo": ..., "params": {...}}, ...]``；
    - 或顶层为对象且含 ``"param_sets"`` 列表字段。

    Args:
        path: ``--params`` 指向的 JSON 文件路径。

    Returns:
        :class:`~src.eval_harness.params.ParamSet` 列表（已逐项校验）。

    Raises:
        HarnessError: 当路径缺失、文件不存在 / 损坏、结构非法或为空时。
    """
    from .params import ParamSet  # 纯逻辑，但集中于此保持加载语义一致。

    if not path:
        raise HarnessError("缺少必需输入：--params（Param_Set 定义 JSON 文件）")
    raw = _read_json(path, "--params（Param_Set 定义）")

    if isinstance(raw, dict) and "param_sets" in raw:
        raw = raw["param_sets"]
    if not isinstance(raw, list):
        raise HarnessError(
            "Param_Set 定义文件结构非法：应为列表或含 'param_sets' 列表字段",
            path=os.path.abspath(path),
        )
    if not raw:
        raise HarnessError(
            "Param_Set 定义文件未包含任何 Param_Set",
            path=os.path.abspath(path),
        )

    param_sets = [ParamSet.from_dict(item) for item in raw]
    return param_sets


def _load_profile(path: Optional[str]):
    """加载 Localization_Profile；未提供时返回名为 ``default`` 的默认 Profile。"""
    from .profile import LocalizationProfile

    if not path:
        return LocalizationProfile(name="default")
    raw = _read_json(path, "--profile（Localization_Profile 定义）")
    if not isinstance(raw, dict):
        raise HarnessError(
            "Localization_Profile 定义文件结构非法：应为对象",
            path=os.path.abspath(path),
        )
    raw.setdefault("name", "default")
    return LocalizationProfile.from_dict(raw)


def _resolve_param_set_names(args, param_sets) -> List[str]:
    """确定报告所用的 Param_Set 名称集合。

    优先取 ``--param-set-names``；否则由 ``--params`` 加载的 Param_Set 推导名称。
    """
    if getattr(args, "param_set_names", None):
        return list(args.param_set_names)
    if param_sets is not None:
        return [ps.name for ps in param_sets]
    raise HarnessError(
        "缺少必需输入：--param-set-names 或 --params（用于确定报告的 Param_Set）"
    )


# ===========================================================================
# 子命令处理函数
# ===========================================================================
def _cmd_annotate_preview(args) -> int:
    """生成 Preview_Composite（Req 2）。需要 screenshots / stitched / params / review。"""
    from .annotation import AnnotationStore, AnnotationTool
    from .case_loader import CaseLoader
    from .map_registry import MapRegistry

    param_sets = _load_param_sets(args.params)
    param_set = param_sets[0]

    cases = CaseLoader(args.screenshots).load()
    maps = MapRegistry(args.stitched).discover()
    if not maps:
        raise HarnessError(
            "stitched 目录中未发现任何大地图（缺少成对的 *_coords.json/.png）",
            path=os.path.abspath(args.stitched),
        )

    store = AnnotationStore(args.annotations)
    tool = AnnotationTool(store, caches_dir=args.caches)
    report = tool.generate_previews(
        cases, maps, param_set, args.review, redo=args.redo
    )

    print(
        f"[annotate-preview] generated={len(report.generated)} "
        f"skipped={len(report.skipped)} "
        f"needs_manual={len(report.needs_manual)} "
        f"processed={report.processed_count} "
        f"failures={len(report.needs_manual)}"
    )
    for warning in report.warnings:
        print(f"  warning: {warning}", file=sys.stderr)
    return 0


def _cmd_annotate_commit(args) -> int:
    """提交 Ground_Truth（Req 3）。需要 screenshots / review / annotations。"""
    from .annotation import AnnotationStore, AnnotationTool
    from .case_loader import CaseLoader

    if not os.path.isdir(args.review):
        raise HarnessError(
            "缺少必需输入：review 目录不存在", path=os.path.abspath(args.review)
        )

    cases = CaseLoader(args.screenshots).load()
    store = AnnotationStore(args.annotations)
    tool = AnnotationTool(store)
    report = tool.commit(cases, args.review)

    print(
        f"[annotate-commit] confirmed={report.confirmed_count} "
        f"rejected={report.rejected_count} "
        f"needs_manual={report.needs_manual_count} "
        f"failures={report.rejected_count + report.needs_manual_count}"
    )
    return 0


def _cmd_batch_extract(args) -> int:
    """批量提取 Feature_Cache（Req 5）。需要 stitched / params / caches。"""
    from .features import BatchExtractor
    from .map_registry import MapRegistry

    param_sets = _load_param_sets(args.params)
    maps = MapRegistry(args.stitched).discover()
    if not maps:
        raise HarnessError(
            "stitched 目录中未发现任何大地图（缺少成对的 *_coords.json/.png）",
            path=os.path.abspath(args.stitched),
        )
    map_entries = list(maps.values())

    report = BatchExtractor().run(
        map_entries, param_sets, args.caches, force=args.force
    )

    print(
        f"[batch-extract] ok={len(report.ok)} skipped={len(report.skipped)} "
        f"failed={len(report.failed)} total={len(report.entries)} "
        f"failures={len(report.failed)}"
    )
    for entry in report.failed:
        print(
            f"  failed: map_id={entry.map_id} "
            f"param_set={entry.param_set_name} error={entry.error}",
            file=sys.stderr,
        )
    return 0


def _cmd_eval_frame(args) -> int:
    """单帧匹配评估（Req 6）。需要 screenshots / annotations / params / caches / db / stitched。"""
    from .case_loader import CaseLoader
    from .evaluator import MatchEvaluator
    from .map_registry import MapRegistry
    from .result_store import ResultStore
    from .runner import MatchRunner

    param_sets = _load_param_sets(args.params)
    cases = CaseLoader(args.screenshots).load()

    store = _load_annotations(args.annotations)
    if not store:
        raise HarnessError(
            "Annotation_Store 中没有任何 Ground_Truth，无法评估",
            path=os.path.abspath(args.annotations),
        )

    registry = MapRegistry(args.stitched)
    result_store = ResultStore(args.results_db)
    runner = MatchRunner(args.caches, registry, result_store)
    evaluator = MatchEvaluator()

    cases_by_id = {c.case_id: c for c in cases}

    evaluated = 0
    failures = 0
    try:
        for param_set in param_sets:
            for case_id, gt in store.items():
                case = cases_by_id.get(case_id)
                if case is None:
                    # 有真值但缺对应截图：记录并继续（记录-继续策略）。
                    print(
                        f"  warning: GT case '{case_id}' 缺少对应截图，跳过",
                        file=sys.stderr,
                    )
                    continue
                try:
                    metric = evaluator.evaluate(case, gt, param_set, runner)
                except HarnessError as exc:
                    failures += 1
                    print(f"  failed: case={case_id} error={exc}", file=sys.stderr)
                    continue
                evaluated += 1
                if not metric.success:
                    failures += 1
        for warning in runner.warnings:
            print(f"  warning: {warning}", file=sys.stderr)
    finally:
        result_store.close()

    print(f"[eval-frame] evaluated={evaluated} failures={failures}")
    return 0


def _cmd_eval_sequence(args) -> int:
    """连续定位评估（Req 9）。需要 annotations / profile / params / caches / db / stitched。"""
    from .map_registry import MapRegistry
    from .result_store import ResultStore
    from .runner import MatchRunner
    from .sequence import SequenceEvaluator, build_sequence

    param_sets = _load_param_sets(args.params)
    profile = _load_profile(args.profile)

    gts = _load_annotations(args.annotations)
    if not gts:
        raise HarnessError(
            "Annotation_Store 中没有任何 Ground_Truth，无法构造序列",
            path=os.path.abspath(args.annotations),
        )

    sequences = build_sequence(list(gts.values()), threshold=profile.region_size)

    registry = MapRegistry(args.stitched)
    result_store = ResultStore(args.results_db)
    runner = MatchRunner(args.caches, registry, result_store)

    processed = 0
    failures = 0
    try:
        for param_set in param_sets:
            evaluator = SequenceEvaluator(profile, runner, result_store)
            for sequence in sequences:
                try:
                    evaluator.evaluate(sequence, param_set, profile, runner)
                except Exception as exc:  # noqa: BLE001 - 记录并继续（决策：记录-继续）
                    failures += 1
                    print(
                        f"  failed: sequence={sequence.sequence_id} error={exc}",
                        file=sys.stderr,
                    )
                    continue
                processed += 1
        for warning in runner.warnings:
            print(f"  warning: {warning}", file=sys.stderr)
    finally:
        result_store.close()

    print(f"[eval-sequence] sequences={processed} failures={failures}")
    return 0


def _cmd_report(args) -> int:
    """生成 HTML 报告（Req 10）。需要 results db / param-set 名称 / out 路径。"""
    from .report import ReportGenerator
    from .result_store import ResultStore

    if not os.path.isfile(args.results_db):
        raise HarnessError(
            "缺少必需输入：Result_Store 数据库不存在",
            path=os.path.abspath(args.results_db),
        )

    # 名称优先取 --param-set-names；否则尝试从 --params 推导。
    param_sets = None
    if args.params:
        param_sets = _load_param_sets(args.params)
    names = _resolve_param_set_names(args, param_sets)

    result_store = ResultStore(args.results_db)
    try:
        ReportGenerator().generate(
            names, result_store, args.out, by_map=args.by_map
        )
    finally:
        result_store.close()

    print(f"[report] out={os.path.abspath(args.out)} param_sets={len(names)}")
    return 0


# ===========================================================================
# 共享小工具
# ===========================================================================
def _load_annotations(annotations_path: str):
    """加载 Annotation_Store 真值映射（``case_id -> GroundTruth``）。

    封装延迟导入与 ``load``，供 eval-frame / eval-sequence 复用。
    """
    from .annotation import AnnotationStore

    return AnnotationStore(annotations_path).load()


# ===========================================================================
# argparse 构造
# ===========================================================================
def _add_screenshots_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--screenshots",
        default=DEFAULT_SCREENSHOTS_DIR,
        help=f"截图（Test_Case）目录，默认 {DEFAULT_SCREENSHOTS_DIR}",
    )


def _add_stitched_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--stitched",
        default=DEFAULT_STITCHED_DIR,
        help=f"大地图（stitched）目录，默认 {DEFAULT_STITCHED_DIR}",
    )


def _add_caches_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--caches",
        default=DEFAULT_CACHES_DIR,
        help=f"Feature_Cache 根目录，默认 {DEFAULT_CACHES_DIR}",
    )


def _add_results_db_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--results-db",
        default=DEFAULT_RESULTS_DB,
        help=f"Result_Store SQLite 路径，默认 {DEFAULT_RESULTS_DB}",
    )


def _add_review_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--review",
        default=DEFAULT_REVIEW_DIR,
        help=f"Review_Folder 复核目录，默认 {DEFAULT_REVIEW_DIR}",
    )


def _add_annotations_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--annotations",
        default=DEFAULT_ANNOTATIONS,
        help=f"Annotation_Store 路径，默认 {DEFAULT_ANNOTATIONS}",
    )


def _add_params_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--params",
        default=None,
        help="Param_Set 定义 JSON 文件（列表或含 'param_sets' 字段）",
    )


def build_parser() -> argparse.ArgumentParser:
    """构造顶层 argparse 解析器与全部子命令（Req 11.1）。"""
    parser = argparse.ArgumentParser(
        prog="eval-harness",
        description="map-match-eval-harness 纯命令行评估工具（Requirement 11）。",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # annotate-preview
    p_prev = sub.add_parser(
        "annotate-preview", help="对尚无真值的用例生成 Preview_Composite（Req 2）"
    )
    _add_screenshots_arg(p_prev)
    _add_stitched_arg(p_prev)
    _add_params_arg(p_prev)
    _add_caches_arg(p_prev)
    _add_review_arg(p_prev)
    _add_annotations_arg(p_prev)
    p_prev.add_argument(
        "--redo", action="store_true", help="对已有真值的用例也重新生成预览（Req 2.5）"
    )
    p_prev.set_defaults(func=_cmd_annotate_preview)

    # annotate-commit
    p_commit = sub.add_parser(
        "annotate-commit", help="根据人工复核结果提交 Ground_Truth（Req 3）"
    )
    _add_screenshots_arg(p_commit)
    _add_review_arg(p_commit)
    _add_annotations_arg(p_commit)
    p_commit.set_defaults(func=_cmd_annotate_commit)

    # batch-extract
    p_batch = sub.add_parser(
        "batch-extract", help="为 (map_id, Param_Set) 组合批量提取 Feature_Cache（Req 5）"
    )
    _add_stitched_arg(p_batch)
    _add_params_arg(p_batch)
    _add_caches_arg(p_batch)
    p_batch.add_argument(
        "--force", action="store_true", help="即便缓存已存在也重新生成（Req 5.2）"
    )
    p_batch.set_defaults(func=_cmd_batch_extract)

    # eval-frame
    p_frame = sub.add_parser(
        "eval-frame", help="对有真值的用例执行单帧匹配评估（Req 6）"
    )
    _add_screenshots_arg(p_frame)
    _add_annotations_arg(p_frame)
    _add_params_arg(p_frame)
    _add_caches_arg(p_frame)
    _add_results_db_arg(p_frame)
    _add_stitched_arg(p_frame)
    p_frame.set_defaults(func=_cmd_eval_frame)

    # eval-sequence
    p_seq = sub.add_parser(
        "eval-sequence", help="构造并重放连续序列做定位评估（Req 9）"
    )
    _add_annotations_arg(p_seq)
    p_seq.add_argument(
        "--profile", default=None, help="Localization_Profile 定义 JSON 文件"
    )
    _add_params_arg(p_seq)
    _add_caches_arg(p_seq)
    _add_results_db_arg(p_seq)
    _add_stitched_arg(p_seq)
    p_seq.set_defaults(func=_cmd_eval_sequence)

    # report
    p_report = sub.add_parser(
        "report", help="聚合并导出 HTML 报告（Req 10）"
    )
    _add_results_db_arg(p_report)
    _add_params_arg(p_report)
    p_report.add_argument(
        "--param-set-names",
        nargs="+",
        default=None,
        help="纳入报告的 Param_Set 名称；缺省时由 --params 推导",
    )
    p_report.add_argument(
        "--out",
        default=DEFAULT_REPORT_OUT,
        help=f"HTML 报告输出路径，默认 {DEFAULT_REPORT_OUT}",
    )
    p_report.add_argument(
        "--by-map", action="store_true", help="按 map_id 细分报告（Req 10.4）"
    )
    p_report.set_defaults(func=_cmd_report)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Harness_CLI 入口（Req 11.1, 11.2, 11.3）。

    Args:
        argv: 命令行参数（不含程序名）；``None`` 时取 ``sys.argv[1:]``。

    Returns:
        进程退出码：成功为 ``0``；遇到 :class:`HarnessError` 为 ``1``。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except HarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - 进程入口
    sys.exit(main())
