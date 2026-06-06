"""map-match-eval-harness 包。

一套面向 ``src/match_engine``（SurfEngine / SiftEngine）与
``src/task/MapOverlayTask.py`` 连续定位逻辑的纯命令行评估工具。

产物目录约定（均在仓库内 ``eval/`` 下）：

- ``eval/caches/``：特征缓存，按 ``Param_Set_Name`` 分目录。
- ``eval/results.db``：SQLite 结果数据库。
- ``eval/review/``：标注预览复核目录。
- ``eval/reports/``：HTML 报告输出目录。
- ``eval/annotations.json``：Annotation_Store 真值存储。
"""

from .errors import HarnessError

__all__ = ["HarnessError"]
