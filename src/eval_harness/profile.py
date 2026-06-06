"""Localization_Profile：连续定位后处理逻辑配置的序列化与校验（Requirement 9）。

描述 Simulated_OCR 噪声幅度、区间收窄尺寸、回退阈值、平滑窗口 / 权重、
锁定阈值等参数，独立于 Param_Set。不含瞬移检测与离群点过滤参数。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LocalizationProfile:
    """连续定位后处理逻辑的可序列化配置（Req 9.2–9.5）。

    将 ``MapOverlayTask`` 中的后处理"魔法常量"抽离为可注入、可对照的字段。
    **不包含**瞬移检测 / 离群点过滤参数——这些属 OCR 前处理，不在重放范围内。
    """

    name: str
    # Simulated_OCR 噪声幅度 U(-noise, +noise)（Req 9.2）
    ocr_noise: float = 0.10
    # 小区间收窄边长（像素），对应 MAP_REGION_SIZE
    region_size: int = 1000
    # 帧中心裁剪边长（像素），对应 FRAME_CROP_SIZE
    frame_crop_size: int = 400
    # 连续小区间失败回退全局阈值（默认 2，Req 9.5）
    fallback_max_failures: int = 2
    # 平滑窗口大小，对应 SMOOTH_WINDOW_SIZE（Req 9.3）
    smooth_window: int = 3
    # 加权移动平均权重，对应 SMOOTH_WEIGHTS（Req 9.3）
    smooth_weights: tuple = (1, 2, 3)
    # 锁定置信度阈值，对应 CONFIDENCE_THRESHOLD（Req 9.4）
    lock_confidence: float = 0.9
    # 锁定匹配点阈值（Req 9.4）
    lock_match_count: int = 10

    def to_dict(self) -> dict:
        """序列化为 JSON 友好的字典。

        ``smooth_weights`` 元组以列表形式落盘（JSON 无元组类型）。
        """
        d = asdict(self)
        d["smooth_weights"] = list(self.smooth_weights)
        return d

    @staticmethod
    def from_dict(d: dict) -> "LocalizationProfile":
        """从字典反序列化，缺失字段回退到默认值。

        ``smooth_weights`` 若为列表则恢复为元组，保持 frozen dataclass 可哈希。
        """
        fields = {k: v for k, v in d.items() if k in _FIELD_NAMES}
        if "smooth_weights" in fields and fields["smooth_weights"] is not None:
            fields["smooth_weights"] = tuple(fields["smooth_weights"])
        return LocalizationProfile(**fields)


_FIELD_NAMES = frozenset(LocalizationProfile.__dataclass_fields__.keys())
