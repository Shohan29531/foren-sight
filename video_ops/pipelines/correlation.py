from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from video_ops.models.clip_embedder import default_embedder


def _parse_iso(dt_str: str) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None


def _abs_seconds(dt: datetime) -> float:
    if dt.tzinfo is not None:
        return float(dt.timestamp())
    # Avoid platform-local timezone ambiguity: treat naive as UTC.
    return float(dt.replace(tzinfo=timezone.utc).timestamp())


@dataclass
class EventRef:
    project_id: str
    event_id: str
    t: float
    abs_t: float
    kind: str
    title: str
    frame_path: str
    clip_path: str
    zone_name: str
    line_name: str
    tags: List[str]


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def build_event_refs(
    project_id: str,
    events: List[Dict[str, Any]],
    video_start_iso: str,
    include_kinds: Optional[List[str]] = None,
) -> List[EventRef]:
    include_kinds = include_kinds or [
        "zone_motion",
        "zone_enter",
        "zone_exit",
        "zone_loiter",
        "line_cross",
    ]

    start_dt = _parse_iso(video_start_iso)
    if start_dt is None:
        return []

    refs: List[EventRef] = []
    for e in events:
        kind = str(e.get("kind", ""))
        if kind not in include_kinds:
            continue
        t = float(e.get("t", 0.0))
        dt = start_dt + timedelta(seconds=t)
        abs_t = _abs_seconds(dt)
        refs.append(
            EventRef(
                project_id=project_id,
                event_id=str(e.get("id", "")),
                t=t,
                abs_t=abs_t,
                kind=kind,
                title=str(e.get("title", "")),
                frame_path=str(e.get("frame_path", "")),
                clip_path=str(e.get("clip_path", "")),
                zone_name=str(e.get("zone_name", "")) if e.get("zone_name") else "",
                line_name=str(e.get("line_name", "")) if e.get("line_name") else "",
                tags=list(e.get("tags") or []),
            )
        )
    return refs


def correlate_events(
    refs: List[EventRef],
    time_window_s: float = 30.0,
    sim_threshold: float = 0.26,
) -> Dict[str, Any]:
    """Correlate events across projects using time proximity + CLIP similarity.

    Returns dict with keys: groups (list), params

    groups: [{group_id, events:[...], max_sim, span_s}]
    """
    if len(refs) < 2:
        return {"groups": [], "params": {"time_window_s": time_window_s, "sim_threshold": sim_threshold}}

    # Embed all frames (best-effort). Missing frames -> zeros.
    embedder = default_embedder()
    img_paths = []
    ok_idx = []
    for i, r in enumerate(refs):
        p = r.frame_path
        if p and p.strip():
            img_paths.append(p)
            ok_idx.append(i)

    embs = np.zeros((len(refs), 512), dtype=np.float32)
    if img_paths:
        from pathlib import Path

        vecs = embedder.embed_images([Path(p) for p in img_paths], batch_size=16)
        # vecs already normalized
        for j, i in enumerate(ok_idx):
            if vecs.shape[0] > j:
                # Some CLIP variants might not be 512; adapt dynamically
                if embs.shape[1] != vecs.shape[1]:
                    embs = np.zeros((len(refs), vecs.shape[1]), dtype=np.float32)
                embs[i] = vecs[j]

    uf = _UF(len(refs))
    max_sim = {}

    # Pairwise compare (small-N expected)
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            if refs[i].project_id == refs[j].project_id:
                continue
            dt = abs(refs[i].abs_t - refs[j].abs_t)
            if dt > float(time_window_s):
                continue
            # cosine similarity = dot (embeddings normalized)
            sim = float(np.dot(embs[i], embs[j])) if embs.shape[1] else 0.0
            if sim >= float(sim_threshold):
                uf.union(i, j)
                key = (min(i, j), max(i, j))
                max_sim[key] = sim

    # Build groups
    buckets: Dict[int, List[int]] = {}
    for i in range(len(refs)):
        r = uf.find(i)
        buckets.setdefault(r, []).append(i)

    groups = []
    gid = 0
    for root, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        # must include at least 2 distinct projects
        projs = {refs[i].project_id for i in idxs}
        if len(projs) < 2:
            continue

        # compute max similarity among pairs in this group we recorded
        sims = []
        for a_i in range(len(idxs)):
            for b_i in range(a_i + 1, len(idxs)):
                a, b = idxs[a_i], idxs[b_i]
                key = (min(a, b), max(a, b))
                if key in max_sim:
                    sims.append(max_sim[key])
        group_max_sim = float(max(sims)) if sims else 0.0

        idxs_sorted = sorted(idxs, key=lambda k: refs[k].abs_t)
        span_s = float(refs[idxs_sorted[-1]].abs_t - refs[idxs_sorted[0]].abs_t)

        events_out = []
        for i in idxs_sorted:
            r = refs[i]
            events_out.append(
                {
                    "project_id": r.project_id,
                    "event_id": r.event_id,
                    "t": r.t,
                    "abs_t": r.abs_t,
                    "kind": r.kind,
                    "title": r.title,
                    "zone_name": r.zone_name,
                    "line_name": r.line_name,
                    "frame_path": r.frame_path,
                    "clip_path": r.clip_path,
                    "tags": r.tags,
                }
            )

        groups.append(
            {
                "group_id": f"g{gid:03d}",
                "projects": sorted(list(projs)),
                "max_sim": group_max_sim,
                "span_s": span_s,
                "events": events_out,
            }
        )
        gid += 1

    # Sort most confident first
    groups.sort(key=lambda g: (float(g.get("max_sim", 0.0)), -len(g.get("events", []))), reverse=True)

    return {"groups": groups, "params": {"time_window_s": time_window_s, "sim_threshold": sim_threshold}}
