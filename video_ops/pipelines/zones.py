from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from video_ops.pipelines.events import _make_clip
from video_ops.storage.project_store import ProjectPaths


def _load_frames_manifest(project: ProjectPaths) -> Dict[str, Any]:
    mp = project.frames_dir / "frames_manifest.json"
    if not mp.exists():
        raise RuntimeError("frames_manifest.json not found. Run frame sampling first.")
    return json.loads(mp.read_text(encoding="utf-8"))


def _first_frame_size(project: ProjectPaths) -> Tuple[int, int, str]:
    mf = _load_frames_manifest(project)
    frames = mf.get("frames", [])
    if not frames:
        raise RuntimeError("No frames in manifest.")
    p = frames[0]["path"]
    img = cv2.imread(p)
    if img is None:
        raise RuntimeError("Could not read first frame.")
    h, w = img.shape[:2]
    return w, h, p


def _zone_to_pixels(zone: Dict[str, Any], w: int, h: int) -> Tuple[int, int, int, int]:
    """Convert a zone dict to pixel rectangle (x1,y1,x2,y2)."""
    if "rect_norm" in zone:
        x1n, y1n, x2n, y2n = zone["rect_norm"]
        x1 = int(round(float(x1n) * w))
        y1 = int(round(float(y1n) * h))
        x2 = int(round(float(x2n) * w))
        y2 = int(round(float(y2n) * h))
    else:
        # fallback: absolute pixels
        x1, y1, x2, y2 = zone.get("rect_px", [0, 0, w, h])
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 + 1:
        x2 = min(w, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(h, y1 + 2)
    return x1, y1, x2, y2


def _hog_people_detector() -> cv2.HOGDescriptor:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def _segments_from_bool(times: List[float], active: List[bool], min_duration_s: float = 1.0) -> List[Tuple[float, float, int, int]]:
    """Return list of (start_t, end_t, start_idx, end_idx_inclusive) for contiguous True runs."""
    segs: List[Tuple[float, float, int, int]] = []
    start = None
    s_idx = None
    for i, a in enumerate(active):
        if a and start is None:
            start = times[i]
            s_idx = i
        if (not a or i == len(active) - 1) and start is not None:
            # end at previous time if current is false, else at current
            e_idx = i if a and i == len(active) - 1 else i - 1
            end = times[e_idx]
            if end - start >= min_duration_s:
                segs.append((float(start), float(end), int(s_idx), int(e_idx)))
            start = None
            s_idx = None
    return segs


def analyze_zones(
    project: ProjectPaths,
    zones: List[Dict[str, Any]],
    motion_threshold: float = 0.02,
    motion_gate_for_person: float = 0.01,
    min_motion_duration_s: float = 1.0,
    enable_person: bool = True,
    loiter_seconds: float = 20.0,
    clip_seconds: int = 8,
    max_events_per_zone: int = 50,
) -> List[Dict[str, Any]]:
    """Create zone-rule events from sampled frames.

    No training. Uses:
      - Background subtraction (MOG2) to detect motion inside ROI
      - Optional OpenCV HOG person detector gated by motion
    """
    if not zones:
        return []

    mf = _load_frames_manifest(project)
    frames = mf.get("frames", [])
    if not frames:
        return []

    w, h, _ = _first_frame_size(project)
    zones_px = []
    for z in zones:
        x1, y1, x2, y2 = _zone_to_pixels(z, w, h)
        zones_px.append((z, (x1, y1, x2, y2)))

    backsub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)
    hog = _hog_people_detector() if enable_person else None

    # per-zone time series
    times: List[float] = []
    zone_motion_active: Dict[str, List[bool]] = {z["id"]: [] for z, _ in zones_px}
    zone_person_present: Dict[str, List[bool]] = {z["id"]: [] for z, _ in zones_px}
    zone_motion_frac: Dict[str, List[float]] = {z["id"]: [] for z, _ in zones_px}
    zone_person_count: Dict[str, List[int]] = {z["id"]: [] for z, _ in zones_px}
    frame_paths: List[str] = []

    for fr in frames:
        t = float(fr["t"])
        p = fr["path"]
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fg = backsub.apply(gray)
        # binarize
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        times.append(t)
        frame_paths.append(p)

        for z, (x1, y1, x2, y2) in zones_px:
            zid = z["id"]
            roi_fg = fg[y1:y2, x1:x2]
            frac = float(np.mean(roi_fg > 0)) if roi_fg.size else 0.0
            zone_motion_frac[zid].append(frac)
            zone_motion_active[zid].append(frac >= motion_threshold)

            # person detection (gated)
            cnt = 0
            present = False
            if hog is not None and frac >= motion_gate_for_person:
                roi = img[y1:y2, x1:x2]
                if roi.size:
                    # Downscale ROI for speed if large
                    rh, rw = roi.shape[:2]
                    scale = 1.0
                    if max(rh, rw) > 640:
                        scale = 640.0 / float(max(rh, rw))
                        roi = cv2.resize(roi, (int(rw * scale), int(rh * scale)))
                    rects, _ = hog.detectMultiScale(roi, winStride=(8, 8), padding=(8, 8), scale=1.05)
                    cnt = int(len(rects))
                    present = cnt >= 1

            zone_person_count[zid].append(cnt)
            zone_person_present[zid].append(present)

    if not times:
        return []

    # Build events
    events: List[Dict[str, Any]] = []
    video_path = project.norm_video_path if project.norm_video_path.exists() else project.raw_video_path

    def add_event(base: Dict[str, Any]) -> None:
        events.append(base)

    for z, (x1, y1, x2, y2) in zones_px:
        zid = str(z["id"])
        zname = str(z.get("name", zid))

        # Motion segments
        motion_segs = _segments_from_bool(times, zone_motion_active[zid], min_duration_s=min_motion_duration_s)
        for k, (st, en, si, ei) in enumerate(motion_segs[:max_events_per_zone]):
            t0 = st
            ev_id = f"zn_{zid}_motion_{k:03d}"
            clip_path = project.clips_dir / f"{ev_id}_t{t0:.2f}.mp4"
            _make_clip(video_path, clip_path, max(0.0, st - clip_seconds), en + clip_seconds)
            add_event(
                {
                    "id": ev_id,
                    "kind": "zone_motion",
                    "zone_id": zid,
                    "zone_name": zname,
                    "t": float(t0),
                    "start": float(st),
                    "end": float(en),
                    "score_z": None,
                    "frame_path": frame_paths[si],
                    "clip_path": str(clip_path),
                    "title": f"Motion in {zname}",
                    "description": f"Detected sustained motion inside ROI '{zname}' from {st:.1f}s to {en:.1f}s.",
                    "confidence": float(min(0.95, 0.4 + 10.0 * float(np.mean(zone_motion_frac[zid][si:ei+1])))),
                }
            )

        # Person entry/exit + loiter
        present = zone_person_present[zid]
        # transitions
        last = False
        enter_idxs: List[int] = []
        exit_idxs: List[int] = []
        for i, cur in enumerate(present):
            if cur and not last:
                enter_idxs.append(i)
            if (not cur) and last:
                exit_idxs.append(i)
            last = cur

        for k, i in enumerate(enter_idxs[:max_events_per_zone]):
            t0 = times[i]
            ev_id = f"zn_{zid}_enter_{k:03d}"
            clip_path = project.clips_dir / f"{ev_id}_t{t0:.2f}.mp4"
            _make_clip(video_path, clip_path, max(0.0, t0 - clip_seconds), t0 + clip_seconds)
            add_event(
                {
                    "id": ev_id,
                    "kind": "zone_enter",
                    "zone_id": zid,
                    "zone_name": zname,
                    "t": float(t0),
                    "start": float(t0),
                    "end": float(t0),
                    "score_z": None,
                    "frame_path": frame_paths[i],
                    "clip_path": str(clip_path),
                    "title": f"Person entered {zname}",
                    "description": f"A person-like figure appears to enter ROI '{zname}' around {t0:.1f}s.",
                    "confidence": 0.55,
                }
            )

        for k, i in enumerate(exit_idxs[:max_events_per_zone]):
            t0 = times[i]
            ev_id = f"zn_{zid}_exit_{k:03d}"
            clip_path = project.clips_dir / f"{ev_id}_t{t0:.2f}.mp4"
            _make_clip(video_path, clip_path, max(0.0, t0 - clip_seconds), t0 + clip_seconds)
            add_event(
                {
                    "id": ev_id,
                    "kind": "zone_exit",
                    "zone_id": zid,
                    "zone_name": zname,
                    "t": float(t0),
                    "start": float(t0),
                    "end": float(t0),
                    "score_z": None,
                    "frame_path": frame_paths[i],
                    "clip_path": str(clip_path),
                    "title": f"Person exited {zname}",
                    "description": f"A person-like figure appears to leave ROI '{zname}' around {t0:.1f}s.",
                    "confidence": 0.5,
                }
            )

        # Loitering segments (continuous present)
        loiter_segs = _segments_from_bool(times, present, min_duration_s=float(loiter_seconds))
        for k, (st, en, si, ei) in enumerate(loiter_segs[:max_events_per_zone]):
            t0 = st
            ev_id = f"zn_{zid}_loiter_{k:03d}"
            clip_path = project.clips_dir / f"{ev_id}_t{t0:.2f}.mp4"
            _make_clip(video_path, clip_path, max(0.0, st - clip_seconds), en + clip_seconds)
            add_event(
                {
                    "id": ev_id,
                    "kind": "zone_loiter",
                    "zone_id": zid,
                    "zone_name": zname,
                    "t": float(t0),
                    "start": float(st),
                    "end": float(en),
                    "score_z": None,
                    "frame_path": frame_paths[si],
                    "clip_path": str(clip_path),
                    "title": f"Loitering in {zname}",
                    "description": f"Person-like presence persists in ROI '{zname}' for ~{en - st:.0f}s (from {st:.1f}s to {en:.1f}s).",
                    "confidence": 0.6,
                }
            )

    events.sort(key=lambda e: float(e.get("t", 0.0)))
    return events
