"""Result_Store：评估结果的 SQLite 持久化（Requirement 7）。

以 ``(case_id, param_set_name)`` 为主键持久化单帧指标，以
``(sequence_id, param_set_name, profile_name)`` 为主键持久化序列指标；
显式重算时原地覆盖（UPSERT）。

设计说明：``FrameMetric`` 与 ``SequenceMetric`` 的"权威"定义分别属于
``evaluator.py``（任务 14.1）与 ``sequence.py``（任务 15.6），尚未实现。
为避免对这两个模块产生硬性（且可能循环）的导入依赖，本模块在此处定义
**字段名与 SQLite schema 对齐**的轻量 dataclass。``upsert_*`` 接口按
属性名鸭子类型读取入参，因此后续可平滑替换为权威定义。
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Optional


@dataclass
class FrameMetric:
    """单帧评估指标，字段与 ``frame_results`` 表对齐。

    与 design.md 的 Match_Evaluator.FrameMetric 字段一致；当 evaluator.py
    实现其权威版本后可由此处统一复用。
    """

    case_id: str
    param_set_name: str
    map_id: Optional[str]
    success: bool
    error_distance: Optional[float]
    scale_error: Optional[float]
    confidence: float
    match_count: int
    inlier_count: int
    elapsed_ms: float


@dataclass
class SequenceMetric:
    """序列评估指标，字段与 ``sequence_results`` 表对齐。

    在 design.md 的 Sequence_Evaluator.SequenceMetric 基础上补充 ``sequence_id``
    （schema 主键的一部分），使其可独立用于持久化。
    """

    sequence_id: str
    param_set_name: str
    profile_name: str
    map_id: Optional[str]
    mean_error: float
    max_error: float
    fail_frames: int
    wrong_lock_frames: int
    lock_map_id: Optional[str]
    lock_frame_index: Optional[int]


_CREATE_FRAME_TABLE = """
CREATE TABLE IF NOT EXISTS frame_results (
    case_id         TEXT NOT NULL,
    param_set_name  TEXT NOT NULL,
    map_id          TEXT,
    success         INTEGER NOT NULL,
    error_distance  REAL,
    scale_error     REAL,
    confidence      REAL,
    match_count     INTEGER,
    inlier_count    INTEGER,
    elapsed_ms      REAL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (case_id, param_set_name)
);
"""

_CREATE_SEQUENCE_TABLE = """
CREATE TABLE IF NOT EXISTS sequence_results (
    sequence_id       TEXT NOT NULL,
    param_set_name    TEXT NOT NULL,
    profile_name      TEXT NOT NULL,
    map_id            TEXT,
    mean_error        REAL,
    max_error         REAL,
    fail_frames       INTEGER,
    wrong_lock_frames INTEGER,
    lock_map_id       TEXT,
    lock_frame_index  INTEGER,
    updated_at        TEXT NOT NULL,
    PRIMARY KEY (sequence_id, param_set_name, profile_name)
);
"""


class ResultStore:
    """评估结果的 SQLite 持久化存储。

    数据库文件不存在时自动创建并建表（Req 7.5）。``:memory:`` 亦受支持，
    用于属性测试与单元测试。

    Args:
        db_path: SQLite 数据库文件路径；``":memory:"`` 表示内存库。
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        # 文件型数据库：确保父目录存在（Req 7.5 —— 不存在则创建新库）。
        if db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        with self._conn:
            self._conn.execute(_CREATE_FRAME_TABLE)
            self._conn.execute(_CREATE_SEQUENCE_TABLE)

    # ----- 单帧结果 -----------------------------------------------------

    def upsert_frame(self, m: "FrameMetric", ts: str) -> None:
        """插入或原地覆盖一条单帧结果，并更新写入时间戳（Req 7.1, 7.2, 7.4）。

        以 ``(case_id, param_set_name)`` 为主键；主键冲突时覆盖既有记录的
        全部字段并更新 ``updated_at``。入参按属性名鸭子类型读取。
        """
        self._conn.execute(
            """
            INSERT INTO frame_results (
                case_id, param_set_name, map_id, success,
                error_distance, scale_error, confidence,
                match_count, inlier_count, elapsed_ms, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(case_id, param_set_name) DO UPDATE SET
                map_id         = excluded.map_id,
                success        = excluded.success,
                error_distance = excluded.error_distance,
                scale_error    = excluded.scale_error,
                confidence     = excluded.confidence,
                match_count    = excluded.match_count,
                inlier_count   = excluded.inlier_count,
                elapsed_ms     = excluded.elapsed_ms,
                updated_at     = excluded.updated_at
            """,
            (
                m.case_id,
                m.param_set_name,
                m.map_id,
                1 if m.success else 0,
                m.error_distance,
                m.scale_error,
                m.confidence,
                m.match_count,
                m.inlier_count,
                m.elapsed_ms,
                ts,
            ),
        )
        self._conn.commit()

    def get_frame(self, case_id: str, param_set_name: str) -> Optional[FrameMetric]:
        """读取某 ``(case_id, param_set_name)`` 的单帧结果，不存在返回 ``None``。"""
        row = self._conn.execute(
            """
            SELECT case_id, param_set_name, map_id, success,
                   error_distance, scale_error, confidence,
                   match_count, inlier_count, elapsed_ms
            FROM frame_results
            WHERE case_id = ? AND param_set_name = ?
            """,
            (case_id, param_set_name),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_frame(row)

    def query_by_param_set(self, param_set_name: str) -> list[FrameMetric]:
        """返回某 Param_Set 下的全部单帧结果（按 case_id 排序）。"""
        rows = self._conn.execute(
            """
            SELECT case_id, param_set_name, map_id, success,
                   error_distance, scale_error, confidence,
                   match_count, inlier_count, elapsed_ms
            FROM frame_results
            WHERE param_set_name = ?
            ORDER BY case_id
            """,
            (param_set_name,),
        ).fetchall()
        return [self._row_to_frame(r) for r in rows]

    @staticmethod
    def _row_to_frame(row: sqlite3.Row) -> FrameMetric:
        return FrameMetric(
            case_id=row["case_id"],
            param_set_name=row["param_set_name"],
            map_id=row["map_id"],
            success=bool(row["success"]),
            error_distance=row["error_distance"],
            scale_error=row["scale_error"],
            confidence=row["confidence"],
            match_count=row["match_count"],
            inlier_count=row["inlier_count"],
            elapsed_ms=row["elapsed_ms"],
        )

    # ----- 序列结果 -----------------------------------------------------

    def upsert_sequence(self, m: "SequenceMetric", ts: str) -> None:
        """插入或原地覆盖一条序列结果，并更新写入时间戳（Req 7.1, 7.4）。

        以 ``(sequence_id, param_set_name, profile_name)`` 为主键；主键冲突时
        覆盖既有记录的全部字段并更新 ``updated_at``。
        """
        self._conn.execute(
            """
            INSERT INTO sequence_results (
                sequence_id, param_set_name, profile_name, map_id,
                mean_error, max_error, fail_frames, wrong_lock_frames,
                lock_map_id, lock_frame_index, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sequence_id, param_set_name, profile_name) DO UPDATE SET
                map_id            = excluded.map_id,
                mean_error        = excluded.mean_error,
                max_error         = excluded.max_error,
                fail_frames       = excluded.fail_frames,
                wrong_lock_frames = excluded.wrong_lock_frames,
                lock_map_id       = excluded.lock_map_id,
                lock_frame_index  = excluded.lock_frame_index,
                updated_at        = excluded.updated_at
            """,
            (
                m.sequence_id,
                m.param_set_name,
                m.profile_name,
                m.map_id,
                m.mean_error,
                m.max_error,
                m.fail_frames,
                m.wrong_lock_frames,
                m.lock_map_id,
                m.lock_frame_index,
                ts,
            ),
        )
        self._conn.commit()

    # ----- 资源管理 -----------------------------------------------------

    def close(self) -> None:
        """关闭底层数据库连接。"""
        self._conn.close()

    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["ResultStore", "FrameMetric", "SequenceMetric"]
