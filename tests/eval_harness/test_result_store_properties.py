"""Property 7：Result_Store UPSERT 往返与覆盖幂等（Requirement 7.1, 7.2, 7.3, 7.4）。

仅实现设计文档中编号的 Property 7：对一串以小池主键（强制碰撞）施加的写操作，
``get_frame`` 返回每个主键**最后一次写入**的字段值，且总记录数恰等于**不同主键数**。
使用内存数据库 ``ResultStore(":memory:")``，每个示例构造一个全新的 store。
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from src.eval_harness.result_store import FrameMetric, ResultStore

# 小主键池：用少量 case_id / param_set_name 强制产生主键碰撞（覆盖路径）。
_case_ids = st.sampled_from(["c0", "c1", "c2"])
_param_set_names = st.sampled_from(["p0", "p1"])

# 浮点字段：限制为可被 SQLite REAL 精确往返的小整数值（如 0.0、12.0），
# 从而无需容差即可比较；None 单独混入以覆盖可空字段。
_round_trip_float = st.builds(float, st.integers(min_value=0, max_value=1000))
_optional_float = st.one_of(st.none(), _round_trip_float)
_optional_str = st.one_of(st.none(), st.sampled_from(["m1", "m2", "m3"]))


@st.composite
def _frame_metrics(draw: st.DrawFn) -> FrameMetric:
    return FrameMetric(
        case_id=draw(_case_ids),
        param_set_name=draw(_param_set_names),
        map_id=draw(_optional_str),
        success=draw(st.booleans()),
        error_distance=draw(_optional_float),
        scale_error=draw(_optional_float),
        confidence=draw(_round_trip_float),
        match_count=draw(st.integers(min_value=0, max_value=10_000)),
        inlier_count=draw(st.integers(min_value=0, max_value=10_000)),
        elapsed_ms=draw(_round_trip_float),
    )


def _assert_same_frame(actual: FrameMetric, expected: FrameMetric) -> None:
    assert actual.case_id == expected.case_id
    assert actual.param_set_name == expected.param_set_name
    assert actual.map_id == expected.map_id
    assert actual.success == expected.success
    assert actual.match_count == expected.match_count
    assert actual.inlier_count == expected.inlier_count
    for got, want in (
        (actual.error_distance, expected.error_distance),
        (actual.scale_error, expected.scale_error),
    ):
        if want is None:
            assert got is None
        else:
            assert got is not None and math.isclose(got, want, abs_tol=1e-9)
    assert math.isclose(actual.confidence, expected.confidence, abs_tol=1e-9)
    assert math.isclose(actual.elapsed_ms, expected.elapsed_ms, abs_tol=1e-9)


# Feature: map-match-eval-harness, Property 7: Result_Store 的 UPSERT 往返与覆盖幂等
@settings(max_examples=200)
@given(
    writes=st.lists(_frame_metrics(), min_size=0, max_size=40),
    timestamps=st.lists(
        st.integers(min_value=0, max_value=1_000_000), min_size=40, max_size=40
    ),
)
def test_upsert_roundtrip_and_overwrite_idempotent(
    writes: list[FrameMetric], timestamps: list[int]
) -> None:
    """按顺序施加写操作后，主键查询返回最后一次写入，记录数 = 不同主键数。

    **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
    """
    store = ResultStore(":memory:")
    try:
        last_write: dict[tuple[str, str], FrameMetric] = {}
        for i, m in enumerate(writes):
            store.upsert_frame(m, ts=f"ts-{timestamps[i]:07d}")
            last_write[(m.case_id, m.param_set_name)] = m

        # 1) 每个主键返回最后一次写入的字段值（往返 + 覆盖）。
        for (case_id, param_set_name), expected in last_write.items():
            got = store.get_frame(case_id, param_set_name)
            assert got is not None
            _assert_same_frame(got, expected)

        # 2) 总记录数 == 不同主键数（覆盖不增加记录）。
        param_set_names = {key[1] for key in last_write}
        total = sum(
            len(store.query_by_param_set(name)) for name in param_set_names
        )
        assert total == len(last_write)
    finally:
        store.close()
