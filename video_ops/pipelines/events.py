from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from video_ops.models.llm import LLM, image_file_to_b64_jpeg, load_llm_config
from video_ops.storage.project_store import ProjectPaths


def _load_meta(project: ProjectPaths) -> List[Dict[str, Any]]:
    meta_path = project.index_dir / "frames.meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_embeddings(project: ProjectPaths) -> np.ndarray:
    emb_path = project.index_dir / "frames.embeddings.npy"
    if not emb_path.exists():
        raise RuntimeError("Embeddings not found. Build the index first.")
    return np.load(str(emb_path)).astype(np.float32)


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = float(x.mean())
    sigma = float(x.std() + 1e-8)
    return (x - mu) / sigma


def _make_clip(video_path: Path, out_path: Path, start_s: float, end_s: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    start_s = max(0.0, start_s)
    dur = max(0.5, end_s - start_s)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{dur:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-an",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def propose_events(
    project: ProjectPaths,
    index_bundle: Any,
    z_thresh: float = 2.5,
    clip_seconds: int = 8,
    min_gap_seconds: float = 8.0,
    max_events: int = 40,
) -> List[Dict[str, Any]]:
    """Propose events via embedding change-points.

    Returns a list of candidate events with clip paths.
    """
    embs = _load_embeddings(project)
    meta = _load_meta(project)

    if len(meta) != embs.shape[0]:
        n = min(len(meta), embs.shape[0])
        meta = meta[:n]
        embs = embs[:n]

    # embs are L2-normalized; cosine similarity is dot product
    sim_prev = np.sum(embs[1:] * embs[:-1], axis=1)
    delta = 1.0 - sim_prev  # higher => bigger change
    z = _zscore(delta)

    cand = np.where(z > z_thresh)[0] + 1  # shift to frame index
    # Sort by strongest spikes
    cand = sorted(cand.tolist(), key=lambda i: float(z[i - 1]), reverse=True)

    events: List[Dict[str, Any]] = []
    selected_ts: List[float] = []

    for idx in cand:
        t = float(meta[idx]["t"])
        if any(abs(t - s) < min_gap_seconds for s in selected_ts):
            continue
        start = max(0.0, t - clip_seconds)
        end = t + clip_seconds
        ev_id = f"ev_{len(events):03d}"
        clip_path = project.clips_dir / f"{ev_id}_t{t:.2f}.mp4"
        _make_clip(project.norm_video_path if project.norm_video_path.exists() else project.raw_video_path, clip_path, start, end)

        events.append(
            {
                "id": ev_id,
                "kind": "changepoint",
                "zone_id": None,
                "zone_name": None,
                "t": t,
                "start": start,
                "end": end,
                "score_z": float(z[idx - 1]),
                "frame_path": meta[idx]["path"],
                "clip_path": str(clip_path),
                "title": "",
                "description": "",
                "confidence": None,
            }
        )
        selected_ts.append(t)
        if len(events) >= max_events:
            break

    # Sort chronologically for timeline
    events.sort(key=lambda e: e["t"])
    return events


def label_events(project: ProjectPaths, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Label events with either:
    - LLM (optional) grounded on a few evidence frames
    - heuristic fallback
    """
    cfg = load_llm_config()
    llm = LLM(cfg)

    if not events:
        return events

    if not llm.enabled():
        for e in events:
            e["title"] = "Activity spike / scene change"
            e["description"] = f"Detected a notable visual change around t={e['t']:.1f}s. Review the clip for details."
            e["confidence"] = float(min(0.95, max(0.2, (e.get('score_z', 0.0) - 1.0) / 4.0)))
        return events

    # LLM path: use 3 frames (prev, current, next) as evidence.
    out: List[Dict[str, Any]] = []
    for e in events:
        frame = Path(e["frame_path"])
        # try to get neighbors from filename index
        stem = frame.stem
        try:
            idx = int(stem.split("_")[1])
        except Exception:
            idx = None

        candidates = []
        if idx is not None:
            for j in [idx - 1, idx, idx + 1]:
                p = frame.parent / f"frame_{j:06d}_t"  # prefix match
                # find actual file
                matches = sorted(frame.parent.glob(f"frame_{j:06d}_t*.jpg"))
                if matches:
                    candidates.append(matches[0])
        if not candidates:
            candidates = [frame]

        images_b64 = [image_file_to_b64_jpeg(str(p)) for p in candidates[:3]]

        prompt = (
            "You are labeling candidate events in a security review timeline. "
            "Given a few frames around the timestamp, produce a short title and one-sentence description. "
            "Be conservative; if uncertain, say so. Output strict JSON with keys: title, description, confidence (0-1)."
        )
        user = f"Timestamp: {e['t']:.2f}s. Provide JSON." 
        msgs = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ]

        try:
            txt = llm.chat(msgs, images_b64_jpeg=images_b64)
            # best-effort parse
            import json as _json

            j = _json.loads(txt.strip())
            e["title"] = str(j.get("title", "Event"))[:120]
            e["description"] = str(j.get("description", ""))[:400]
            conf = j.get("confidence", None)
            e["confidence"] = float(conf) if conf is not None else None
        except Exception:
            e["title"] = "Candidate event"
            e["description"] = f"Possible notable activity around t={e['t']:.1f}s."
            e["confidence"] = None

        out.append(e)

    return out
