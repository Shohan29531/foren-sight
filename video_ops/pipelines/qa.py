from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from video_ops.models.llm import LLM, image_file_to_b64_jpeg, load_llm_config
from video_ops.pipelines.indexing import search_text
from video_ops.pipelines.events import _make_clip
from video_ops.storage.project_store import ProjectPaths


def answer_question(project: ProjectPaths, index_bundle: Any, question: str, top_k: int = 8) -> Dict[str, Any]:
    """Answer with timestamp-grounded evidence.

    Strategy:
    1) Use CLIP text->frame retrieval on the question
    2) Return cited timestamps + clips
    3) Optionally ask an LLM to summarize, grounded on evidence frames
    """
    cfg = load_llm_config()
    llm = LLM(cfg)

    hits = search_text(project, index_bundle, question, top_k=top_k)

    # generate short clips for the top hits
    clip_seconds = int(os.getenv("CLIP_SECONDS_AROUND_EVENT", "6"))
    evidence = []
    for i, h in enumerate(hits):
        t = float(h["t"])
        start = max(0.0, t - clip_seconds)
        end = t + clip_seconds
        clip_path = project.clips_dir / f"qa_{i:02d}_t{t:.2f}.mp4"
        _make_clip(project.norm_video_path if project.norm_video_path.exists() else project.raw_video_path, clip_path, start, end)
        evidence.append(
            {
                "t": t,
                "score": float(h["score"]),
                "frame_path": h["frame_path"],
                "clip_path": str(clip_path),
            }
        )

    if not llm.enabled():
        return {
            "answer": "I can’t generate a narrative answer without an LLM configured. Here are the most relevant moments (with timestamps) to review:",
            "evidence": evidence,
            "mode": "evidence_only",
        }

    # LLM grounded answer with a small number of frames
    images_b64 = [image_file_to_b64_jpeg(e["frame_path"]) for e in evidence[:4] if Path(e["frame_path"]).exists()]

    system = (
        "You are a security video review assistant. Answer the user's question using ONLY the provided evidence frames. "
        "Cite timestamps explicitly (e.g., 'at 01:23'). If uncertain, say what is missing. "
        "Return a concise answer followed by a bullet list of timestamps you relied on."
    )
    ts_list = ", ".join([f"{e['t']:.2f}s" for e in evidence[:6]])
    user = f"Question: {question}\nCandidate relevant timestamps: {ts_list}\nAnswer now."

    txt = llm.chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], images_b64_jpeg=images_b64)

    return {
        "answer": txt.strip(),
        "evidence": evidence,
        "mode": "llm_grounded",
    }
