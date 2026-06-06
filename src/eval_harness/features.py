"""features：Feature_Extractor 与 Batch_Extractor（Requirement 4, 5）。

包含缓存路径纯函数（``cache_path`` 及其逆解析）与引擎适配的特征提取 / 批量提取。
Feature_Cache 按 Param_Set 分目录存放：``{caches_dir}/{Param_Set_Name}/{map_id}_{algo}.npz``。
"""

from __future__ import annotations

import os
import posixpath
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .errors import HarnessError
from .map_registry import MapEntry
from .params import SIFT, SURF, ParamSet, SiftParams, SurfParams

#: 已知的算法标识集合。算法是缓存文件名中最后一个下划线之后的段，
#: 取自固定集合，使得 ``{map_id}_{algo}`` 的解析可以在 map_id 自身包含
#: 下划线时仍无歧义（见 design.md Property 4）。
KNOWN_ALGOS: Tuple[str, ...] = ("surf", "sift")

#: Feature_Cache 文件扩展名。
_CACHE_EXT: str = ".npz"


def cache_path(caches_dir: str, param_set_name: str, map_id: str, algo: str) -> str:
    """构造某 ``(Param_Set_Name, map_id, algo)`` 三元组的 Feature_Cache 路径。

    路径形如 ``{caches_dir}/{param_set_name}/{map_id}_{algo}.npz``
    （design.md > Components > Feature_Extractor，Req 4.1, 4.2）。Param_Set
    作为独立的目录层级，使 map_id、算法与 Param_Set 三者可由路径唯一区分。

    Args:
        caches_dir: 特征缓存根目录（如 ``eval/caches``）。
        param_set_name: Param_Set_Name，作为分组目录名。
        map_id: 大地图标识。
        algo: 算法标识，须属于 :data:`KNOWN_ALGOS`。

    Returns:
        以 ``/`` 分隔的缓存文件路径。

    Raises:
        HarnessError: 当 ``algo`` 不在 :data:`KNOWN_ALGOS` 中时。
    """
    if algo not in KNOWN_ALGOS:
        raise HarnessError(
            f"未知算法 '{algo}'，可选值为 {KNOWN_ALGOS}"
        )
    filename = f"{map_id}_{algo}{_CACHE_EXT}"
    return posixpath.join(caches_dir, param_set_name, filename)


def parse_cache_path(path: str) -> Tuple[str, str, str]:
    """从 Feature_Cache 路径无歧义地解析回 ``(param_set_name, map_id, algo)``。

    这是 :func:`cache_path` 的逆函数（design.md Property 4：缓存路径对三要素
    单射且可解析）。解析规则：

    - ``param_set_name`` 取文件所在目录的最末一级目录名。
    - 去除 ``.npz`` 扩展名后，在文件名主名上按 **最后一个** 下划线切分，
      下划线之后为 ``algo``（须属于 :data:`KNOWN_ALGOS`），之前为 ``map_id``。
      因此 ``map_id`` 自身可以包含下划线与数字。

    同时接受 ``/`` 与 ``\\`` 作为路径分隔符。

    Args:
        path: 由 :func:`cache_path` 生成（或与之同构）的缓存路径。

    Returns:
        三元组 ``(param_set_name, map_id, algo)``。

    Raises:
        HarnessError: 当路径缺少扩展名、缺少 Param_Set 目录层级、
            缺少 ``map_id_algo`` 结构，或 ``algo`` 不在已知集合中时。
    """
    # 统一分隔符后拆分路径组件。
    normalized = path.replace("\\", "/")
    components = [c for c in normalized.split("/") if c not in ("", ".")]
    if len(components) < 2:
        raise HarnessError(
            f"缓存路径 '{path}' 缺少 Param_Set 目录层级，无法解析"
        )

    filename = components[-1]
    param_set_name = components[-2]

    if not filename.endswith(_CACHE_EXT):
        raise HarnessError(
            f"缓存路径 '{path}' 缺少 '{_CACHE_EXT}' 扩展名，无法解析"
        )
    stem = filename[: -len(_CACHE_EXT)]

    if "_" not in stem:
        raise HarnessError(
            f"缓存文件名 '{filename}' 缺少 'map_id_algo' 结构，无法解析"
        )
    map_id, algo = stem.rsplit("_", 1)
    if not map_id:
        raise HarnessError(
            f"缓存文件名 '{filename}' 的 map_id 为空，无法解析"
        )
    if algo not in KNOWN_ALGOS:
        raise HarnessError(
            f"缓存文件名 '{filename}' 的算法段 '{algo}' 不在 {KNOWN_ALGOS} 中"
        )
    return param_set_name, map_id, algo


class FeatureExtractor:
    """按 Param_Set 的提取参数从大地图原图提特征并落盘为 Feature_Cache。

    复用 :mod:`src.match_engine` 引擎的提取约定（见 ``surf.py`` / ``sift.py``
    的 ``_extract``）：构造一个临时的 ``cv2`` detector（按 ``algo`` + 提取参数）
    → ``detectAndCompute`` → :func:`grid_sample` → :func:`save_npz`。产出的
    ``.npz`` 文件格式与引擎 ``load_npz`` 所加载的完全一致（Req 4.1, 5.1）。

    本类不修改 ``src/match_engine`` 的对外行为，仅在 Harness 侧按 Param_Set
    参数构造等价的 detector 并复用引擎内部的 ``grid_sample`` / ``save_npz``。
    """

    def extract(
        self, map_entry: MapEntry, param_set: ParamSet, out_path: str
    ) -> None:
        """用 ``param_set`` 的提取参数从大地图原图提特征并 ``save_npz`` 到 ``out_path``。

        Args:
            map_entry: 目标大地图注册项，提供原图路径 ``image_path``。
            param_set: 参数集，其 ``algo`` 与 ``params`` 决定 detector 构造、
                网格采样的 ``grid`` 与 ``max_per_cell``。
            out_path: Feature_Cache 输出路径（``.npz``）。所在目录若不存在会被创建。

        Raises:
            HarnessError: 当 Param_Set 非法、依赖缺失（如 SURF 需 opencv-contrib）、
                原图无法读取或图中提不到特征时。
        """
        # 延迟导入：cv2 与引擎仅在真正提取时才需要，避免纯逻辑场景引入重依赖。
        try:
            import cv2  # noqa: PLC0415
            from src.match_engine.common import grid_sample, save_npz  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001 - 统一转换为 HarnessError
            raise HarnessError(
                f"无法加载特征提取依赖（cv2 / match_engine）：{exc}",
                name=self._safe_name(param_set),
            ) from exc

        # 校验参数集，确保字段齐全且取值合法（与命名一致）。
        param_set.validate()

        detector = self._build_detector(cv2, param_set)
        p = param_set.params
        grid = p.grid
        max_per_cell = p.max_per_cell

        map_path = map_entry.image_path
        img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HarnessError(
                "无法读取大地图原图",
                path=os.path.abspath(map_path),
                name=self._safe_name(param_set),
            )
        h, w = img.shape[:2]

        kps, desc = detector.detectAndCompute(img, None)
        if not kps or desc is None:
            raise HarnessError(
                "大地图原图未提取到任何特征",
                path=os.path.abspath(map_path),
                name=self._safe_name(param_set),
            )

        sel_kps, sel_descs = grid_sample(
            kps, desc, h, w, grid, grid, max_per_cell
        )

        # 确保输出目录存在后再落盘。
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        save_npz(out_path, sel_kps, sel_descs, h, w)

    @staticmethod
    def _build_detector(cv2_module, param_set: ParamSet):
        """按 ``algo`` + 提取参数构造与引擎一致的临时 ``cv2`` detector。

        SURF 复用 ``cv2.xfeatures2d.SURF_create(hessian, octaves, layers,
        extended, upright)``（需 opencv-contrib）；SIFT 复用
        ``cv2.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold,
        edgeThreshold, sigma)``，其中 ``nfeatures`` 取引擎默认值 ``0``。
        """
        p = param_set.params
        if param_set.algo == SURF and isinstance(p, SurfParams):
            xfeatures2d = getattr(cv2_module, "xfeatures2d", None)
            if xfeatures2d is None:
                raise HarnessError(
                    "SURF 特征提取需要 opencv-contrib（cv2.xfeatures2d）",
                    name=param_set.name,
                )
            return xfeatures2d.SURF_create(
                p.hessian, p.octaves, p.layers, p.extended, p.upright
            )
        if param_set.algo == SIFT and isinstance(p, SiftParams):
            return cv2_module.SIFT_create(
                0,
                p.n_octave_layers,
                p.contrast_threshold,
                p.edge_threshold,
                p.sigma,
            )
        raise HarnessError(
            f"算法标识与参数类型不匹配：algo={param_set.algo!r}, "
            f"params={type(p).__name__}",
            name=str(param_set.algo),
        )

    @staticmethod
    def _safe_name(param_set: ParamSet) -> str:
        """尽力计算 Param_Set_Name，失败时退回算法标识，供错误消息使用。"""
        try:
            return param_set.name
        except (HarnessError, TypeError, ValueError):
            return str(param_set.algo)


#: Batch_Extractor 单组合的可能状态（Req 5.2, 5.3, 5.4）。
STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class BatchEntry:
    """Batch_Extractor 中单个 ``(map_id, Param_Set)`` 组合的处理结果（Req 5.4）。

    Attributes:
        map_id: 大地图标识。
        param_set_name: Param_Set_Name（分组键）。
        algo: 算法标识（``'surf'`` / ``'sift'``）。
        status: 处理状态，取值为 :data:`STATUS_OK`、:data:`STATUS_SKIPPED`
            或 :data:`STATUS_FAILED`。
        path: 该组合对应的 Feature_Cache 路径。即使失败也给出预期路径，
            便于定位；无法计算路径（如 Param_Set 非法）时为 ``None``。
        error: 失败原因（仅 ``status == STATUS_FAILED`` 时非空）。
    """

    map_id: str
    param_set_name: Optional[str]
    algo: str
    status: str
    path: Optional[str]
    error: Optional[str] = None


@dataclass(frozen=True)
class BatchReport:
    """Batch_Extractor 处理完全部组合后的汇总报告（Req 5.4）。

    持有每个 ``(map_id, Param_Set)`` 组合的 :class:`BatchEntry` 记录，并提供
    按状态分组的便捷视图，供 CLI 打印每组合的成功 / 失败状态与缓存路径。

    Attributes:
        entries: 按处理顺序排列的逐组合结果记录。
    """

    entries: List[BatchEntry] = field(default_factory=list)

    @property
    def ok(self) -> List[BatchEntry]:
        """成功提取并落盘的组合。"""
        return [e for e in self.entries if e.status == STATUS_OK]

    @property
    def skipped(self) -> List[BatchEntry]:
        """因缓存已存在而跳过的组合（非 ``force``）。"""
        return [e for e in self.entries if e.status == STATUS_SKIPPED]

    @property
    def failed(self) -> List[BatchEntry]:
        """提取失败的组合（已记录原因并继续处理其余组合）。"""
        return [e for e in self.entries if e.status == STATUS_FAILED]

    def summary(self) -> str:
        """生成一行人类可读的计数汇总（``ok=.. skipped=.. failed=..``）。"""
        return (
            f"ok={len(self.ok)} skipped={len(self.skipped)} "
            f"failed={len(self.failed)} total={len(self.entries)}"
        )


class BatchExtractor:
    """为一批 ``(map_id, Param_Set)`` 组合批量提取 Feature_Cache 并落盘（Req 5）。

    仅依赖源大地图（:class:`MapEntry`）与 Param_Set 定义即可在独立的高性能机器上
    运行（Req 5.5），产出的缓存可按 Param_Set 目录拷回测试机使用。

    行为约定：

    - 对每个组合计算 :func:`cache_path` 并复用 :class:`FeatureExtractor` 落盘（Req 5.1）。
    - 缓存已存在且未要求 ``force`` 时跳过该组合（Req 5.2）。
    - 单组合失败（内存不足或其他错误）记录组合与原因并 **继续** 处理其余组合（Req 5.3）。
    - 处理完全部组合后返回汇总每组合状态与路径的 :class:`BatchReport`（Req 5.4）。
    """

    def __init__(self, extractor: Optional[FeatureExtractor] = None) -> None:
        """构造批量提取器。

        Args:
            extractor: 用于单组合提取的 :class:`FeatureExtractor`。默认新建一个，
                测试时可注入替身以避免真实的 cv2 / 引擎依赖。
        """
        self._extractor = extractor or FeatureExtractor()

    def run(
        self,
        map_entries: Sequence[MapEntry],
        param_sets: Sequence[ParamSet],
        caches_dir: str,
        force: bool = False,
    ) -> BatchReport:
        """对每个 ``(map_entry, param_set)`` 组合提取并落盘，返回汇总报告。

        采用 ``map_entries: Sequence[MapEntry]`` 而非裸 ``map_id``，使本类仅需源地图
        与 Param_Set 定义即可独立运行（Req 5.5），无需访问测试机本地资源或注册表。
        CLI 可先用 :meth:`MapRegistry.discover` 得到条目再按需筛选后传入。

        遍历顺序为 ``param_set`` 外层、``map_entry`` 内层，使同一 Param_Set 的缓存
        集中产出。任一组合失败都不会中断整体处理（Req 5.3）。

        Args:
            map_entries: 待处理的大地图注册项序列，提供 ``map_id`` 与原图路径。
            param_sets: 待处理的参数集序列。
            caches_dir: 特征缓存根目录（如 ``eval/caches``）。
            force: 为真时即便缓存已存在也重新生成；否则跳过已存在的组合（Req 5.2）。

        Returns:
            汇总每个组合状态（``ok`` / ``skipped`` / ``failed``）与缓存路径的
            :class:`BatchReport`（Req 5.4）。
        """
        entries: List[BatchEntry] = []
        for param_set in param_sets:
            for map_entry in map_entries:
                entries.append(
                    self._process_one(map_entry, param_set, caches_dir, force)
                )
        return BatchReport(entries=entries)

    def _process_one(
        self,
        map_entry: MapEntry,
        param_set: ParamSet,
        caches_dir: str,
        force: bool,
    ) -> BatchEntry:
        """处理单个组合，捕获所有异常并归一化为一条 :class:`BatchEntry`。"""
        map_id = map_entry.map_id
        algo = param_set.algo

        # 先计算缓存路径；Param_Set_Name 计算依赖合法的 algo/params，若此处即失败，
        # 记为 failed（无可用路径）并继续（Req 5.3）。
        try:
            param_set_name = param_set.name
            out_path = cache_path(caches_dir, param_set_name, map_id, algo)
        except Exception as exc:  # noqa: BLE001 - 统一归一化为失败记录
            return BatchEntry(
                map_id=map_id,
                param_set_name=None,
                algo=str(algo),
                status=STATUS_FAILED,
                path=None,
                error=str(exc),
            )

        # 缓存已存在且未要求强制重算 → 跳过（Req 5.2）。
        if not force and os.path.isfile(out_path):
            return BatchEntry(
                map_id=map_id,
                param_set_name=param_set_name,
                algo=algo,
                status=STATUS_SKIPPED,
                path=out_path,
            )

        # 实际提取；任何失败都记录原因并继续（Req 5.3）。
        try:
            self._extractor.extract(map_entry, param_set, out_path)
        except Exception as exc:  # noqa: BLE001 - 单组合失败不中断整体
            return BatchEntry(
                map_id=map_id,
                param_set_name=param_set_name,
                algo=algo,
                status=STATUS_FAILED,
                path=out_path,
                error=str(exc),
            )

        return BatchEntry(
            map_id=map_id,
            param_set_name=param_set_name,
            algo=algo,
            status=STATUS_OK,
            path=out_path,
        )


__all__ = [
    "KNOWN_ALGOS",
    "cache_path",
    "parse_cache_path",
    "FeatureExtractor",
    "STATUS_OK",
    "STATUS_SKIPPED",
    "STATUS_FAILED",
    "BatchEntry",
    "BatchReport",
    "BatchExtractor",
]
