from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

from video_ops.models.clip_embedder import default_embedder
from video_ops.storage.project_store import ProjectPaths


@dataclass
class IndexBundle:
    faiss_index: Any
    meta: List[Dict[str, Any]]


def _load_frames_manifest(frames_dir: Path) -> List[Dict[str, Any]]:
    manifest = json.loads((frames_dir / "frames_manifest.json").read_text(encoding="utf-8"))
    return manifest["frames"]


def build_or_load_index(project: ProjectPaths) -> IndexBundle:
    index_dir = project.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "frames.index.faiss"
    meta_path = index_dir / "frames.meta.json"
    emb_path = index_dir / "frames.embeddings.npy"

    if index_path.exists() and meta_path.exists():
        idx = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return IndexBundle(faiss_index=idx, meta=meta)

    frames = _load_frames_manifest(project.frames_dir)
    frame_paths = [Path(f["path"]) for f in frames]

    embedder = default_embedder()
    embs = embedder.embed_images(frame_paths, batch_size=32)

    if embs.size == 0:
        raise RuntimeError("No embeddings computed. Are frames present?")

    d = embs.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(embs)

    meta = [{"t": f["t"], "path": f["path"]} for f in frames]

    faiss.write_index(idx, str(index_path))
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.save(str(emb_path), embs)

    return IndexBundle(faiss_index=idx, meta=meta)


def search_text(project: ProjectPaths, bundle: IndexBundle, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    embedder = default_embedder()
    q = embedder.embed_text([query])
    D, I = bundle.faiss_index.search(q, top_k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0:
            continue
        m = bundle.meta[idx]
        results.append({
            "t": float(m["t"]),
            "frame_path": m["path"],
            "score": float(score),
        })
    return results
