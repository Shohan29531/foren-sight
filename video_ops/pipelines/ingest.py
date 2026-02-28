import json
import subprocess
from pathlib import Path
from typing import Dict, List

import cv2
from tqdm import tqdm


def normalize_video(input_path: Path, output_path: Path) -> None:
    """Normalize to mp4/h264/aac for consistent decoding."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        "scale='min(1280,iw)':-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def sample_frames(video_path: Path, frames_dir: Path, fps: float = 1.0) -> Path:
    """Sample frames at fixed rate using OpenCV (records timestamps precisely).

    Writes:
      - frames as JPG
      - frames_manifest.json with timestamps

    Returns path to manifest.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = frames_dir / "frames_manifest.json"
    if manifest_path.exists():
        return manifest_path

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    duration_ms = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1e-6) * 1000.0
    step_ms = 1000.0 / max(fps, 1e-6)

    frames: List[Dict] = []
    t = 0.0
    i = 0

    pbar = tqdm(total=int(duration_ms // step_ms) + 1, desc="Sampling frames")
    while t <= duration_ms + 1:
        cap.set(cv2.CAP_PROP_POS_MSEC, t)
        ok, frame = cap.read()
        if not ok:
            break

        ts_sec = float(t) / 1000.0
        fname = f"frame_{i:06d}_t{ts_sec:010.2f}.jpg"
        fpath = frames_dir / fname
        cv2.imwrite(str(fpath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        frames.append({"i": i, "t": ts_sec, "path": str(fpath)})
        i += 1
        t += step_ms
        pbar.update(1)

    pbar.close()
    cap.release()

    manifest_path.write_text(json.dumps({"fps_sample": fps, "frames": frames}, indent=2), encoding="utf-8")
    return manifest_path
