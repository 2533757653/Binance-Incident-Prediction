"""代理统一入口（Builder-A 产出）。

按 dataContract §0 + spec §0：

优先级：环境变量 ``HTTPS_PROXY`` / ``HTTP_PROXY`` / ``ALL_PROXY``（大小写不敏感）
     > ``config/proxy.yaml``（http / https / socks5）
     > 报错退出（exit code 10，ProxyError）

返回 ``httpx`` 可直接使用的 ``proxies=`` dict：``{"http://": "...", "https://": "..."}``。
HTTPS 端点用 ``https://`` 键，HTTP 端点用 ``http://`` 键。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

from src.common.errors import ProxyError

log = logging.getLogger(__name__)

# 根目录 = src/data/proxy.py 的父父父级（Horizon-Incident/）
_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_YAML = _ROOT / "config" / "proxy.yaml"

# 退出码
EXIT_PROXY_MISSING = 10
EXIT_PROXY_RETRY_FAIL = 11

_ENV_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY")


def _read_env() -> dict[str, str]:
    out: dict[str, str] = {}
    for k in _ENV_KEYS:
        v = os.environ.get(k) or os.environ.get(k.lower())
        if v and v.strip():
            out[k] = v.strip()
    return out


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        log.warning("[proxy] 读取 %s 失败: %s", path, e)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _materialize(cfg: dict) -> dict[str, str]:
    """把 yaml 字段 + 环境变量折成 httpx 期望的 {scheme: url} 形式。"""
    env = _read_env()

    http_url = env.get("HTTP_PROXY") or cfg.get("http") or ""
    https_url = env.get("HTTPS_PROXY") or env.get("ALL_PROXY") or cfg.get("https") or cfg.get("socks5") or ""
    if not https_url and http_url:
        https_url = http_url

    proxies: dict[str, str] = {}
    if http_url:
        proxies["http://"] = http_url
    if https_url:
        proxies["https://"] = https_url
    return proxies


def get_proxies(yaml_path: Optional[Path] = None) -> dict[str, str]:
    """返回 httpx 的 proxies 字典；空字典表示"不使用代理"。

    **业务约束**（spec §0）：币安现货 / eapi 在国内默认直连阻塞。
    若完全无代理配置 → 抛 ``ProxyError``，调用方决定是否降级（Builder-A
    对事件合约 fetch 失败时降级到 sample；对价格 fetch 失败时走 no-proxy
    重试，因为 CryptoCompare 是国内直连的，不依赖币安通道）。
    """
    yaml_path = yaml_path or _DEFAULT_YAML
    cfg = _read_yaml(yaml_path)
    return _materialize(cfg)


def has_proxy(proxies: dict[str, str]) -> bool:
    return any(v for v in proxies.values())


def assert_configured(proxies: dict[str, str]) -> None:
    """严格模式：要求至少有一条代理，否则 ``sys.exit(10)``。

    用于 eapi 路径。"""
    if not has_proxy(proxies):
        log.error(
            "[proxy] 未配置代理（env=%s, yaml=%s）。币安 eapi 国内直连阻塞，"
            "请设置 HTTPS_PROXY 或在 config/proxy.yaml 写入 https。",
            _read_env() or "{}",
            _DEFAULT_YAML,
        )
        sys.exit(EXIT_PROXY_MISSING)


__all__ = [
    "get_proxies",
    "has_proxy",
    "assert_configured",
    "EXIT_PROXY_MISSING",
    "EXIT_PROXY_RETRY_FAIL",
]
