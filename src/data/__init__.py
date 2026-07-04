"""Horizon-Incident 数据接入层（Builder-A 产出）。

公开 API：
- ``price_feed``  现货价格（CryptoCompare REST，5s 轮询）
- ``event_feed``  币安 eapi 事件合约
- ``proxy``       代理统一入口
- ``models``      Pydantic 模型（re-export 自 src.common.schemas）
- ``bus``         消息总线（re-export 自 src.common.bus）
- ``main``        可独立运行的入口
"""

__all__ = [
    "price_feed",
    "event_feed",
    "proxy",
    "models",
    "bus",
    "main",
]
