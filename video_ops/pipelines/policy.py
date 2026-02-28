from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, time
from typing import Any, Dict, List, Optional, Tuple


def _parse_iso(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        # Python 3.11+ supports ISO with offset; 3.10 supports most ISO too.
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None


def _parse_hhmm(s: str) -> Optional[time]:
    try:
        s = s.strip()
        if not s:
            return None
        hh, mm = s.split(":")
        return time(hour=int(hh), minute=int(mm))
    except Exception:
        return None


def _in_window(t: time, start: time, end: time) -> bool:
    """Return True if t is within [start,end) allowing wrap across midnight."""
    if start == end:
        # treat as always active
        return True
    if start < end:
        return start <= t < end
    # wraps midnight
    return (t >= start) or (t < end)


def annotate_with_time_windows(
    events: List[Dict[str, Any]],
    video_start_iso: str,
    after_hours_start: str = "20:00",
    after_hours_end: str = "06:00",
    enabled: bool = True,
) -> List[Dict[str, Any]]:
    """Annotate events with absolute local datetime and after-hours tagging.

    - Adds `local_dt` (ISO string) when video_start_iso is provided.
    - Adds tag `after_hours` when enabled and the event occurs in the after-hours window.

    This does not duplicate events; it tags them.
    """
    if not events:
        return events

    start_dt = _parse_iso(video_start_iso)
    if start_dt is None:
        return events

    ah_s = _parse_hhmm(after_hours_start) or time(20, 0)
    ah_e = _parse_hhmm(after_hours_end) or time(6, 0)

    out: List[Dict[str, Any]] = []
    for e in events:
        ne = dict(e)
        t_sec = float(ne.get("t", 0.0))
        dt = start_dt + timedelta(seconds=t_sec)
        ne["local_dt"] = dt.isoformat()

        tags = list(ne.get("tags") or [])
        tt = dt.timetz().replace(tzinfo=None)
        if enabled and _in_window(tt, ah_s, ah_e):
            if "after_hours" not in tags:
                tags.append("after_hours")
            ne["policy_after_hours"] = True
            # suggested severity bump for certain event kinds
            if ne.get("kind") in ("zone_enter", "zone_motion", "zone_loiter", "line_cross"):
                ne.setdefault("severity_suggested", "high")
        ne["tags"] = tags
        out.append(ne)

    return out
