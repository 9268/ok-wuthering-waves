"""Map_Registry 发现示例测试（Requirement 1.1）。

用临时 stitched 目录构造若干成对的 ``{map_id}.png`` / ``{map_id}_coords.json``
文件，断言：

- 同时具有 ``.png`` 与 ``_coords.json`` 的 map_id 被正确发现；
- 仅有 ``_coords.json`` 而缺失同名 ``.png`` 的条目不计入；
- 发现项的 ``map_id`` / ``image_path`` 正确，且坐标被成功加载。

Validates: Requirements 1.1
"""

from __future__ import annotations

import json
import os

from src.eval_harness.map_registry import MapEntry, MapRegistry


def _write_coords(path: str) -> None:
    """写入 CoordsRef.load 可解析的最小坐标 JSON。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "offset": [10.0, 20.0],
                "scale": [2.0, 3.0],
                "min": [0.0, 0.0],
                "max": [100.0, 200.0],
            },
            f,
        )


def _write_pair(stitched_dir: str, map_id: str) -> str:
    """写入成对的 {map_id}.png 与 {map_id}_coords.json，返回 png 路径。"""
    image_path = os.path.join(stitched_dir, f"{map_id}.png")
    # discover() 仅要求 .png 存在，不读取其字节，空文件即可。
    with open(image_path, "wb") as f:
        f.write(b"")
    _write_coords(os.path.join(stitched_dir, f"{map_id}_coords.json"))
    return image_path


def test_discover_returns_pairs_and_excludes_missing_png(tmp_path) -> None:
    stitched_dir = str(tmp_path)

    # 两个完整成对的大地图。
    img_8 = _write_pair(stitched_dir, "8")
    img_900 = _write_pair(stitched_dir, "900")

    # 一个仅有坐标文件、缺失同名 .png 的条目，应被跳过。
    _write_coords(os.path.join(stitched_dir, "404_coords.json"))

    # 一个孤立的 .png（无坐标文件），同样不应被发现。
    with open(os.path.join(stitched_dir, "777.png"), "wb") as f:
        f.write(b"")

    registry = MapRegistry(stitched_dir)
    entries = registry.discover()

    # 仅成对的 map_id 被发现，缺 .png 的 404 与缺坐标的 777 都不计入。
    assert set(entries.keys()) == {"8", "900"}

    entry_8 = entries["8"]
    assert isinstance(entry_8, MapEntry)
    assert entry_8.map_id == "8"
    assert entry_8.image_path == img_8
    # 坐标被成功加载，并保留传入的数值。
    assert entry_8.coords.offset == (10.0, 20.0)
    assert entry_8.coords.scale == (2.0, 3.0)

    entry_900 = entries["900"]
    assert entry_900.map_id == "900"
    assert entry_900.image_path == img_900
    assert entry_900.coords.max_xy == (100.0, 200.0)
