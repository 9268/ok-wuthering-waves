"""Harness_CLI 示例 / 边界测试（Requirement 11.1, 11.2, 11.3）。

覆盖 ``src.eval_harness.cli`` 的 ``main(argv) -> int`` 与 ``build_parser()``：

- 边界 / 缺必需输入（Req 11.2，EDGE_CASE）：
  - ``batch-extract`` 缺 ``--params`` → 返回非零并在 stderr 指明缺失项。
  - ``report`` 在 Result_Store 数据库不存在时 → 返回 1 并给出清晰错误。
- 示例 / 正常路径（Req 11.1, 11.3）：
  - ``report`` 正常路径：预置 SQLite 结果库 + 若干 ``FrameMetric``，运行后
    返回 0、向 stdout 打印汇总行、HTML 产物文件存在。
  - 同时覆盖 ``--params`` JSON 加载（``_load_param_sets``）推导名称的路径。
- ``build_parser()`` 返回可用解析器，``report`` 子命令能解析其参数。
- 无子命令（``main([])``）：argparse 要求子命令 → 抛 ``SystemExit``。

子命令覆盖说明（受 cv2 / 内存约束）：
- ``report`` 由正常路径（happy-path）与边界（缺库）共同覆盖；其编排不触发 cv2。
- ``batch-extract`` 仅覆盖缺必需输入的边界路径；其正常路径需 ``FeatureExtractor``
  （cv2）做实际全图提取，内存敏感，按设计不在此处跑端到端正常路径。
"""

from __future__ import annotations

import json

from src.eval_harness.cli import build_parser, main
from src.eval_harness.params import ParamSet, SurfParams
from src.eval_harness.result_store import FrameMetric, ResultStore

_TS = "2026-01-01T00:00:00"


def _make_param_set() -> ParamSet:
    """构造一个合法的 SURF Param_Set（其 name 用作结果库主键与报告键）。"""
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
            ratio=0.75,
            max_dist=0.25,
        ),
    )


def _seed_results_db(db_path: str, param_set_name: str) -> None:
    """在 db_path 处建库并写入若干单帧结果（混合成功 / 通过 / 失败）。"""
    store = ResultStore(db_path)
    rows = [
        FrameMetric(
            case_id="a1",
            param_set_name=param_set_name,
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
            param_set_name=param_set_name,
            map_id="mapA",
            success=False,
            error_distance=9999.0,
            scale_error=0.5,
            confidence=0.3,
            match_count=10,
            inlier_count=2,
            elapsed_ms=20.0,
        ),
    ]
    for m in rows:
        store.upsert_frame(m, _TS)
    store.close()


# ===========================================================================
# 边界 / 缺必需输入（Req 11.2，EDGE_CASE）
# ===========================================================================
def test_batch_extract_missing_params_returns_error(tmp_path, capsys):
    """batch-extract 缺 --params：返回非零并在 stderr 指明缺失项（Req 11.2）。"""
    stitched = tmp_path / "nope_stitched"  # 不存在的 stitched 目录
    rc = main(["batch-extract", "--stitched", str(stitched)])

    assert rc == 1
    err = capsys.readouterr().err
    assert "error:" in err
    # 错误文案指明缺失项：--params（Param_Set 定义）。
    assert "--params" in err
    assert "缺少必需输入" in err


def test_report_missing_results_db_returns_error(tmp_path, capsys):
    """report 在结果库不存在时：返回 1 并给出清晰错误（Req 11.2）。"""
    missing_db = tmp_path / "does_not_exist.db"
    rc = main(
        [
            "report",
            "--results-db",
            str(missing_db),
            "--param-set-names",
            "surf_default",
            "--out",
            str(tmp_path / "report.html"),
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "不存在" in err
    # 错误应携带相关路径，便于定位。
    assert "does_not_exist.db" in err


# ===========================================================================
# 示例 / 正常路径（Req 11.1, 11.3）
# ===========================================================================
def test_report_happy_path_with_param_set_names(tmp_path, capsys):
    """report 正常路径：返回 0、打印汇总行、HTML 产物存在（Req 11.1, 11.3）。"""
    name = _make_param_set().name
    db_path = tmp_path / "results.db"
    _seed_results_db(str(db_path), name)

    out_html = tmp_path / "out" / "report.html"
    rc = main(
        [
            "report",
            "--results-db",
            str(db_path),
            "--param-set-names",
            name,
            "--out",
            str(out_html),
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    # Req 11.3：完成后向 stdout 报告（子命令标签 + Param_Set 数量）。
    assert "[report]" in out
    assert "param_sets=1" in out
    # Req 11.1 / 10.5：HTML 报告产物落盘。
    assert out_html.exists()
    assert name in out_html.read_text(encoding="utf-8")


def test_report_happy_path_with_params_json(tmp_path, capsys):
    """report 经由 --params JSON 推导名称的正常路径（Req 11.1, 11.3）。

    验证 _load_param_sets 加载 ParamSet.to_dict() 列表并推导报告键。
    """
    param_set = _make_param_set()
    name = param_set.name
    db_path = tmp_path / "results.db"
    _seed_results_db(str(db_path), name)

    params_json = tmp_path / "params.json"
    params_json.write_text(
        json.dumps([param_set.to_dict()], ensure_ascii=False),
        encoding="utf-8",
    )

    out_html = tmp_path / "report.html"
    rc = main(
        [
            "report",
            "--results-db",
            str(db_path),
            "--params",
            str(params_json),
            "--out",
            str(out_html),
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "[report]" in out
    assert "param_sets=1" in out
    assert out_html.exists()


# ===========================================================================
# build_parser / argparse 行为
# ===========================================================================
def test_build_parser_parses_report_subcommand():
    """build_parser 返回解析器，report 子命令可解析其参数（Req 11.1）。"""
    parser = build_parser()
    args = parser.parse_args(
        [
            "report",
            "--results-db",
            "some.db",
            "--param-set-names",
            "surf_a",
            "surf_b",
            "--out",
            "out.html",
            "--by-map",
        ]
    )
    assert args.command == "report"
    assert args.results_db == "some.db"
    assert args.param_set_names == ["surf_a", "surf_b"]
    assert args.out == "out.html"
    assert args.by_map is True
    # set_defaults 绑定了处理函数。
    assert callable(args.func)


def test_main_without_subcommand_raises_system_exit(capsys):
    """无子命令时 argparse 要求子命令，触发 SystemExit（非零）。"""
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        main([])
    # argparse 参数错误使用退出码 2。
    assert exc_info.value.code != 0
