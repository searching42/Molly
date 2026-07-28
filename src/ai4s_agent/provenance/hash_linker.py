from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_artifact_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hash_artifact_bytes(encoded)


def hash_artifact_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()
