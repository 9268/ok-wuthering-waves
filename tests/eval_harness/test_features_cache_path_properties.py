"""Property 4 的属性测试：缓存路径对三要素单射且可解析（Requirement 4.2）。

仅实现设计文档中编号的 Property 4，不引入额外属性。验证
:func:`src.eval_harness.features.cache_path` 与
:func:`src.eval_harness.features.parse_cache_path` 在生成域上：

1. 往返一致：``parse_cache_path(cache_path(c, p, m, a)) == (p, m, a)``。
2. 单射：两个不同的 ``(p, m, a)`` 三元组生成的路径互不相同。
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.eval_harness.features import (
    KNOWN_ALGOS,
    cache_path,
    parse_cache_path,
)

# Param_Set_Name 命名字符集：小写字母 / 数字 / 下划线，非空。
_param_set_names = st.from_regex(r"\A[a-z0-9_]+\Z", fullmatch=True)

# map_id：非空 [a-z0-9_]+，且不以下划线结尾。
#
# 解析规则在文件名主名上按 *最后一个* 下划线切分出 algo（algo 由 cache_path
# 追加在 map_id 之后），因此最后一个下划线恒为 algo 分隔符——只要 map_id 非空，
# 即便其自身含下划线或以 ``surf`` / ``sift`` 结尾也不会产生歧义。这里额外禁止
# map_id 以下划线结尾，避免与 algo 分隔符相邻而产生空段，使生成域更贴近真实
# map_id（如 ``8``、``910``）的命名习惯，同时保证往返在该域上良定义。
_map_ids = st.from_regex(r"\A[a-z0-9]([a-z0-9_]*[a-z0-9])?\Z", fullmatch=True)

# 算法标识取自固定已知集合。
_algos = st.sampled_from(KNOWN_ALGOS)

# 缓存根目录：允许多级相对目录，覆盖 ``eval/caches`` 这类约定值。
_caches_dirs = st.sampled_from(
    ["eval/caches", "caches", "a/b/c", "eval/caches/sub"]
)

# 三元组生成策略：(param_set_name, map_id, algo)。
_triples = st.tuples(_param_set_names, _map_ids, _algos)


# Feature: map-match-eval-harness, Property 4: 缓存路径对三要素单射且可解析
# Validates: Requirements 4.2
@settings(max_examples=200)
@given(caches_dir=_caches_dirs, triple=_triples)
def test_cache_path_round_trip(
    caches_dir: str, triple: tuple[str, str, str]
) -> None:
    """缓存路径可被无歧义解析回原始三要素 ``(param_set_name, map_id, algo)``。"""
    param_set_name, map_id, algo = triple
    path = cache_path(caches_dir, param_set_name, map_id, algo)
    assert parse_cache_path(path) == (param_set_name, map_id, algo)


# Feature: map-match-eval-harness, Property 4: 缓存路径对三要素单射且可解析
# Validates: Requirements 4.2
@settings(max_examples=200)
@given(caches_dir=_caches_dirs, t1=_triples, t2=_triples)
def test_cache_path_injective(
    caches_dir: str,
    t1: tuple[str, str, str],
    t2: tuple[str, str, str],
) -> None:
    """不同的三元组在同一 caches_dir 下生成互异的缓存路径（单射）。"""
    p1 = cache_path(caches_dir, *t1)
    p2 = cache_path(caches_dir, *t2)
    if t1 == t2:
        assert p1 == p2
    else:
        assert p1 != p2
