from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_LOG_PATH = Path("debug-3d97b5.log")
_SESSION_ID = "3d97b5"


def debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
    run_id: str = "run1",
) -> None:
    payload = {
        "sessionId": _SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    try:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass
