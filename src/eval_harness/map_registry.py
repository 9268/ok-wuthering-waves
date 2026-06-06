"""Map_Registry：动态发现 map_id / 坐标 / 大地图路径（Requirement 1, 2, 4, 5）。

以 ``*_coords.json`` 后缀剥离得到 map_id，要求同名 ``.png`` 存在。

发现约定与 ``src/task/MapOverlayTask.py`` 的 ``_load_coords_dict`` 保持一致：
扫描 stitched 目录中以 ``_coords.json`` 结尾的文件，剥离该后缀得到 map_id。
在此基础上额外要求存在同名的 ``{map_id}.png`` 大地图原图，缺失则不计入。

坐标参考复用 :class:`src.match_engine.common.CoordsRef`。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.match_engine.common import CoordsRef

from .errors import HarnessError

_COORDS_SUFFIX = "_coords.json"


@dataclass
class MapEntry:
    """单个大地图的注册项。

    Attributes:
        map_id: 大地图标识，等于源图片 / 坐标文件名主名（如 ``"8"``、``"910"``）。
        image_path: 大地图原图路径，形如 ``{stitched_dir}/{map_id}.png``。
        coords: 该地图的坐标参考（像素 ↔ 游戏坐标换算），复用
            :class:`src.match_engine.common.CoordsRef`。
    """

    map_id: str
    image_path: str
    coords: CoordsRef


class MapRegistry:
    """从 stitched 目录动态发现大地图。

    Args:
        stitched_dir: 存放 ``{map_id}.png`` 与 ``{map_id}_coords.json`` 成对文件的目录。
    """

    def __init__(self, stitched_dir: str) -> None:
        self.stitched_dir = stitched_dir

    def discover(self) -> dict[str, MapEntry]:
        """扫描目录，返回 ``map_id -> MapEntry`` 映射。

        以 ``*_coords.json`` 后缀剥离得到 map_id，并要求同名 ``{map_id}.png``
        存在；缺失对应 ``.png`` 的坐标文件被跳过。坐标文件解析失败时抛出
        :class:`HarnessError` 并指明文件路径。

        Returns:
            按 map_id 索引的 :class:`MapEntry` 字典。

        Raises:
            HarnessError: 当 stitched 目录不存在，或某坐标文件无法解析时。
        """
        if not os.path.isdir(self.stitched_dir):
            raise HarnessError(
                "stitched directory not found",
                path=os.path.abspath(self.stitched_dir),
            )

        entries: dict[str, MapEntry] = {}
        for name in sorted(os.listdir(self.stitched_dir)):
            if not name.endswith(_COORDS_SUFFIX):
                continue
            map_id = name[: -len(_COORDS_SUFFIX)]
            image_path = os.path.join(self.stitched_dir, f"{map_id}.png")
            if not os.path.isfile(image_path):
                # 缺失同名大地图原图，跳过该坐标文件。
                continue

            coords_path = os.path.join(self.stitched_dir, name)
            try:
                coords = CoordsRef.load(coords_path)
            except Exception as exc:  # noqa: BLE001 - 统一转换为 HarnessError
                raise HarnessError(
                    f"failed to parse coords file: {exc}",
                    path=os.path.abspath(coords_path),
                ) from exc

            entries[map_id] = MapEntry(
                map_id=map_id,
                image_path=image_path,
                coords=coords,
            )

        return entries


__all__ = ["MapEntry", "MapRegistry"]
