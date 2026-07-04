"""Horizon-Incident 统一异常类型。"""


class ProxyError(RuntimeError):
    """代理配置缺失或连接失败。"""


class FeedError(RuntimeError):
    """数据源（WebSocket / REST）异常。"""


class SignalError(RuntimeError):
    """因子计算或信号生成异常。"""


class BacktestError(RuntimeError):
    """回测异常（数据不足、配置错误等）。"""
