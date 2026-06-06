"""Result_Store 建库示例测试（Requirement 7.5）。

验证：当目标数据库文件不存在时（含嵌套的不存在子目录），构造
``ResultStore`` 会创建出具备所需结构（``frame_results`` /
``sequence_results`` 两张表及预期列）的新库，并能在新库上完成
``upsert_frame`` + ``get_frame`` 往返。
"""

from __future__ import annotations

import math
import sqlite3

from src.eval_harness.result_store import FrameMetric, ResultStore

# 两张表的预期列集合（与 result_store.py 的建表 schema 对齐）。
_EXPECTED_FRAME_COLUMNS = {
    "case_id",
    "param_set_name",
    "map_id",
    "success",
    "error_distance",
    "scale_error",
    "confidence",
    "match_count",
    "inlier_count",
    "elapsed_ms",
    "updated_at",
}

_EXPECTED_SEQUENCE_COLUMNS = {
    "sequence_id",
    "param_set_name",
    "profile_name",
    "map_id",
    "mean_error",
    "max_error",
    "fail_frames",
    "wrong_lock_frames",
    "lock_map_id",
    "lock_frame_index",
    "updated_at",
}


def _table_columns(db_path: str, table: str) -> set[str]:
    """用全新连接读取某表的列名集合（PRAGMA table_info）。"""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {row[1] for row in rows}


def _table_names(db_path: str) -> set[str]:
    """用全新连接读取库内所有表名（sqlite_master）。"""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    finally:
        conn.close()
    return {row[0] for row in rows}


def test_creates_new_db_with_required_structure(tmp_path):
    """文件不存在（含嵌套不存在子目录）时创建具备所需结构的新库（Req 7.5）。"""
    db_path = tmp_path / "nested" / "sub" / "results.db"
    # 前置断言：路径与父目录此刻都不存在。
    assert not db_path.exists()
    assert not db_path.parent.exists()

    store = ResultStore(str(db_path))
    try:
        # 文件已被创建。
        assert db_path.exists()

        # 两张所需表都存在。
        tables = _table_names(str(db_path))
        assert {"frame_results", "sequence_results"} <= tables

        # 两张表的列集合与预期 schema 一致。
        assert _table_columns(str(db_path), "frame_results") == _EXPECTED_FRAME_COLUMNS
        assert (
            _table_columns(str(db_path), "sequence_results")
            == _EXPECTED_SEQUENCE_COLUMNS
        )
    finally:
        store.close()


def test_upsert_and_get_roundtrip_on_fresh_db(tmp_path):
    """新建库可立即完成 upsert_frame + get_frame 往返（Req 7.5）。"""
    db_path = tmp_path / "fresh" / "results.db"
    assert not db_path.exists()

    store = ResultStore(str(db_path))
    try:
        metric = FrameMetric(
            case_id="case-1",
            param_set_name="param-A",
            map_id="m1",
            success=True,
            error_distance=12.0,
            scale_error=3.0,
            confidence=0.9,
            match_count=42,
            inlier_count=30,
            elapsed_ms=5.0,
        )
        store.upsert_frame(metric, ts="2026-01-01T00:00:00")

        got = store.get_frame("case-1", "param-A")
        assert got is not None
        assert got.case_id == "case-1"
        assert got.param_set_name == "param-A"
        assert got.map_id == "m1"
        assert got.success is True
        assert got.error_distance == 12.0
        assert got.scale_error == 3.0
        assert math.isclose(got.confidence, 0.9, rel_tol=1e-9)
        assert got.match_count == 42
        assert got.inlier_count == 30
        assert got.elapsed_ms == 5.0
    finally:
        store.close()
