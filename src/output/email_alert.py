# -*- coding: utf-8 -*-
"""邮件提醒（读 config/smtp.env，凭据不入代码/不入 git）。"""
from __future__ import annotations

import logging
import smtplib
import ssl
from email.header import Header
from email.mime.text import MIMEText
from pathlib import Path

log = logging.getLogger("horizon.email")

_CFG_PATH = Path(__file__).resolve().parents[2] / "config" / "smtp.env"


def _load_cfg() -> dict:
    cfg = {}
    if _CFG_PATH.exists():
        for line in _CFG_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


class EmailAlerter:
    def __init__(self):
        c = _load_cfg()
        self.email = c.get("SMTP_EMAIL", "")
        self.password = c.get("SMTP_PASSWORD", "")
        self.to = c.get("SMTP_TO") or self.email
        server = c.get("SMTP_SERVER", "smtp.qq.com:465")
        host, _, port = server.partition(":")
        self.host = host
        self.port = int(port or 465)
        self.use_ssl = c.get("SMTP_SSL", "true").lower() == "true" or self.port == 465
        self.enabled = bool(self.email and self.password)

    def send(self, subject: str, body: str) -> bool:
        if not self.enabled:
            return False
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = Header(subject, "utf-8")
        msg["From"] = self.email
        msg["To"] = self.to
        try:
            if self.use_ssl:
                ctx = ssl.create_default_context()
                with smtplib.SMTP_SSL(self.host, self.port, timeout=15, context=ctx) as s:
                    s.login(self.email, self.password)
                    s.sendmail(self.email, [self.to], msg.as_string())
            else:
                with smtplib.SMTP(self.host, self.port, timeout=15) as s:
                    s.starttls(context=ssl.create_default_context())
                    s.login(self.email, self.password)
                    s.sendmail(self.email, [self.to], msg.as_string())
            return True
        except Exception as e:
            log.warning("[email] 发送失败: %s", e)
            return False


if __name__ == "__main__":
    from datetime import datetime, timezone, timedelta
    a = EmailAlerter()
    print("enabled =", a.enabled, "| host =", a.host, "| port =", a.port,
          "| ssl =", a.use_ssl, "| to =", a.to)
    ts = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    ok = a.send("Horizon 邮件测试 ✅",
                f"这是一封测试邮件。\n如果你收到了，说明邮件提醒配置成功。\n时间：{ts}")
    print("send ok =", ok)
