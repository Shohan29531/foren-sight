from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from video_ops.pipelines.events import _make_clip
from video_ops.storage.project_store import ProjectPaths


def _load_frames_manifest(project: ProjectPaths) -> Dict[str, Any]:
    mp = project.frames_dir / "frames_manifest.json"
    if not mp.exists():
        raise RuntimeError("frames_manifest.json not found. Run frame sampling first.")
    return json.loads(mp.read_text(encoding="utf-8"))


def _first_frame_size(project: ProjectPaths) -> Tuple[int, int]:
    mf = _load_frames_manifest(project)
    frames = mf.get("frames", [])
    if not frames:
        raise RuntimeError("No frames in manifest.")
    p = frames[0]["path"]
    img = cv2.imread(p)
    if img is None:
        raise RuntimeError("Could not read first frame.")
    h, w = img.shape[:2]
    return w, h


def _line_to_pixels(line: Dict[str, Any], w: int, h: int) -> Tuple[int, int, int, int]:
    if "line_norm" in line:
        x1n, y1n, x2n, y2n = line["line_norm"]
        x1 = int(round(float(x1n) * w))
        y1 = int(round(float(y1n) * h))
        x2 = int(round(float(x2n) * w))
        y2 = int(round(float(y2n) * h))
    else:
        x1, y1, x2, y2 = line.get("line_px", [0, 0, w - 1, h - 1])
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if abs(x2 - x1) + abs(y2 - y1) < 2:
        x2 = min(w - 1, x1 + 3)
        y2 = min(h - 1, y1 + 3)
    return x1, y1, x2, y2


def _side_of_line(x1: float, y1: float, x2: float, y2: float, px: float, py: float) -> float:
    """Cross-product sign of (x2-x1,y2-y1) x (px-x1,py-y1)."""
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def _motion_centroid(mask: np.ndarray, min_area: int = 150) -> Optional[Tuple[float, float, float]]:
    """Return (cx, cy, area_frac) for motion mask; None if not enough motion."""
    if mask.size == 0:
        return None
    # Clean mask
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.dilate(m, kernel, iterations=1)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    areas = [cv2.contourArea(c) for c in cnts]
    total_area = float(sum(areas))
    if total_area < float(min_area):
        return None

    # Weighted centroid over all contours
    cx_sum, cy_sum, a_sum = 0.0, 0.0, 0.0
    for c, a in zip(cnts, areas):
        if a <= 0:
            continue
        M = cv2.moments(c)
        if M.get("m00", 0) == 0:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        cx_sum += cx * a
        cy_sum += cy * a
        a_sum += a

    if a_sum <= 0:
        return None

    cx = cx_sum / a_sum
    cy = cy_sum / a_sum
    frac = float(np.mean(m > 0))
    return cx, cy, frac


def analyze_line_crossings(
    project: ProjectPaths,
    lines: List[Dict[str, Any]],
    motion_threshold: float = 0.01,
    min_motion_area: int = 150,
    min_gap_s: float = 3.0,
    clip_seconds: int = 8,
    max_events_per_line: int = 60,
) -> List[Dict[str, Any]]:
    """Generate line-crossing events from sampled frames.

    This is intentionally "no training": it uses background subtraction to find moving pixels and
    checks when the motion centroid crosses a user-defined line.

    Direction semantics:
      - any: accept any crossing
      - neg_to_pos: accept only sign change from negative to positive
      - pos_to_neg: accept only sign change from positive to negative
    """
    if not lines:
        return []

    mf = _load_frames_manifest(project)
    frames = mf.get("frames", [])
    if not frames:
        return []

    w, h = _first_frame_size(project)
    lines_px = []
    for ln in lines:
        x1, y1, x2, y2 = _line_to_pixels(ln, w, h)
        lines_px.append((ln, (x1, y1, x2, y2)))

    backsub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)

    # Track state per line
    prev_side: Dict[str, Optional[float]] = {str(ln.get("id")): None for ln, _ in lines_px}
    last_event_t: Dict[str, float] = {str(ln.get("id")): -1e9 for ln, _ in lines_px}

    video_path = project.norm_video_path if project.norm_video_path.exists() else project.raw_video_path

    events: List[Dict[str, Any]] = []

    for fr in frames:
        t = float(fr["t"])
        p = fr["path"]
        img = cv2.imread(p)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fg = backsub.apply(gray)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        mc = _motion_centroid(fg, min_area=min_motion_area)
        if mc is None:
            # still update prev_side? no, keep as-is
            continue
        cx, cy, frac = mc
        if frac < float(motion_threshold):
            continue

        for ln, (x1, y1, x2, y2) in lines_px:
            lid = str(ln.get("id"))
            direction = str(ln.get("direction", "any"))
            name = str(ln.get("name", lid))

            s = _side_of_line(x1, y1, x2, y2, cx, cy)
            ps = prev_side.get(lid)
            prev_side[lid] = s

            if ps is None:
                continue

            # Crossing is sign change
            if s == 0 or ps == 0:
                continue

            crossed = (s > 0 and ps < 0) or (s < 0 and ps > 0)
            if not crossed:
                continue

            # Direction filter
            if direction == "neg_to_pos" and not (ps < 0 and s > 0):
                continue
            if direction == "pos_to_neg" and not (ps > 0 and s < 0):
                continue

            # Debounce
            if t - float(last_event_t.get(lid, -1e9)) < float(min_gap_s):
                continue

            k = sum(1 for e in events if e.get("line_id") == lid)
            if k >= max_events_per_line:
                continue

            last_event_t[lid] = t
            ev_id = f"ln_{lid}_cross_{k:03d}"
            clip_path = project.clips_dir / f"{ev_id}_t{t:.2f}.mp4"
            _make_clip(video_path, clip_path, max(0.0, t - clip_seconds), t + clip_seconds)

            dir_label = "crossing"
            if ps < 0 and s > 0:
                dir_label = "neg→pos"
            elif ps > 0 and s < 0:
                dir_label = "pos→neg"

            events.append(
                {
                    "id": ev_id,
                    "kind": "line_cross",
                    "line_id": lid,
                    "line_name": name,
                    "direction": direction,
                    "t": float(t),
                    "start": float(t),
                    "end": float(t),
                    "score_z": None,
                    "frame_path": p,
                    "clip_path": str(clip_path),
                    "title": f"Line crossing: {name} ({dir_label})",
                    "description": f"Motion centroid crossed line '{name}' around {t:.1f}s (direction={dir_label}).",
                    "confidence": float(min(0.9, 0.45 + 25.0 * frac)),
                    "tags": ["roi", "line"],
                }
            )

    events.sort(key=lambda e: float(e.get("t", 0.0)))
    return events
