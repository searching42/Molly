from __future__ import annotations

import hmac
import ipaddress
import secrets
from urllib.parse import urlparse

from flask import Flask, jsonify, request


_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def install_local_request_protection(app: Flask) -> None:
    """Protect browser-originated localhost writes without breaking trusted local CLI clients."""

    app.config.setdefault("MOLLY_LOCAL_SESSION_TOKEN", secrets.token_urlsafe(32))

    @app.before_request
    def protect_local_request():
        if not _loopback_hostname(request.remote_addr or ""):
            return jsonify({"ok": False, "error": "remote client is not loopback"}), 403
        if not _loopback_host(request.host):
            return jsonify({"ok": False, "error": "invalid Host header"}), 400
        if request.method in _SAFE_METHODS:
            return None
        origin = str(request.headers.get("Origin") or "").strip()
        fetch_site = str(request.headers.get("Sec-Fetch-Site") or "").strip().lower()
        if not origin and not fetch_site:
            return None
        if fetch_site not in {"same-origin", "same-site", "none"}:
            return jsonify({"ok": False, "error": "cross-site write request rejected"}), 403
        if origin and not _allowed_origin(origin):
            return jsonify({"ok": False, "error": "request Origin is not allowed"}), 403
        supplied = str(request.headers.get("X-Molly-Local-Token") or "")
        expected = str(app.config.get("MOLLY_LOCAL_SESSION_TOKEN") or "")
        if not supplied or not hmac.compare_digest(supplied, expected):
            return jsonify({"ok": False, "error": "local session token required"}), 403
        return None


def _allowed_origin(origin: str) -> bool:
    parsed = urlparse(origin)
    return parsed.scheme in {"http", "https"} and _loopback_hostname(parsed.hostname or "")


def _loopback_host(host: str) -> bool:
    parsed = urlparse(f"//{host}")
    return _loopback_hostname(parsed.hostname or "")


def _loopback_hostname(hostname: str) -> bool:
    clean = str(hostname or "").strip().lower()
    if clean == "localhost":
        return True
    try:
        return ipaddress.ip_address(clean).is_loopback
    except ValueError:
        return False
