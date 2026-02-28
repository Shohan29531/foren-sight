from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from video_ops.storage.project_store import ProjectPaths


def _fmt_time(seconds: float) -> str:
    seconds = float(seconds)
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m:02d}:{s:05.2f}"


def render_timeline(project: ProjectPaths, events: List[Dict[str, Any]]) -> None:
    # Filters
    kinds = sorted({str(e.get("kind", "changepoint")) for e in events})
    zones = sorted({str(e.get("zone_name") or "") for e in events if e.get("zone_name")})

    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        kind_sel = st.multiselect("Event types", kinds, default=kinds)
    with f2:
        zone_sel = st.multiselect("Zones", zones, default=zones)
    with f3:
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

    def _keep(e: Dict[str, Any]) -> bool:
        k = str(e.get("kind", "changepoint"))
        if kind_sel and k not in kind_sel:
            return False
        zn = str(e.get("zone_name") or "")
        if zone_sel and zones and zn and zn not in zone_sel:
            return False
        c = e.get("confidence", None)
        if c is not None and float(c) < float(min_conf):
            return False
        return True

    filtered = [e for e in events if _keep(e)]
    if not filtered:
        st.warning("No events match the current filters.")
        return

    df = pd.DataFrame([
        {
            "time": _fmt_time(e["t"]),
            "t_sec": float(e["t"]),
            "local_dt": str(e.get("local_dt") or ""),
            "title": e.get("title", "") or "(unlabeled)",
            "type": str(e.get("kind", "changepoint")),
            "zone": str(e.get("zone_name") or ""),
            "line": str(e.get("line_name") or ""),
            "tags": ", ".join(list(e.get("tags") or [])),
            "confidence": e.get("confidence", None),
            "z": float(e.get("score_z") or 0.0),
            "id": e.get("id", ""),
        }
        for e in filtered
    ])

    cols = ["time", "local_dt", "type", "zone", "line", "title", "tags", "confidence", "z"]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)

    ids = [e["id"] for e in filtered]
    pick = st.selectbox("Inspect event", ids)
    e = next(x for x in filtered if x["id"] == pick)

    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        st.markdown(f"**{e.get('title','Event')}**")
        st.caption(
            f"Time: {_fmt_time(e['t'])} (t={e['t']:.2f}s) | type={e.get('kind','')} | zone={e.get('zone_name','') or '-'} | z={float(e.get('score_z') or 0.0):.2f}"
        )
        if e.get("description"):
            st.write(e["description"])

        # Case/bookmark UX
        case_key = "case_items"
        if case_key not in st.session_state:
            st.session_state[case_key] = []
        if st.button("Add to Investigation Case", key=f"add_case_{project.project_id}_{e['id']}"):
            existing_ids = {x.get("id") for x in st.session_state[case_key]}
            if e.get("id") in existing_ids:
                st.info("Already added.")
            else:
                st.session_state[case_key].append({
                    "project_id": project.project_id,
                    "id": e.get("id"),
                    "kind": e.get("kind"),
                    "zone_id": e.get("zone_id"),
                    "zone_name": e.get("zone_name"),
                    "line_id": e.get("line_id"),
                    "line_name": e.get("line_name"),
                    "t": float(e.get("t", 0.0)),
                    "start": float(e.get("start", e.get("t", 0.0))),
                    "end": float(e.get("end", e.get("t", 0.0))),
                    "local_dt": e.get("local_dt"),
                    "title": e.get("title"),
                    "description": e.get("description"),
                    "clip_path": e.get("clip_path"),
                    "frame_path": e.get("frame_path"),
                    "confidence": e.get("confidence"),
                    "severity": "",
                    "notes": "",
                    "tags": list(e.get("tags") or []),
                })
                st.success("Added to case.")

        clip_path = e.get("clip_path")
        if clip_path and Path(clip_path).exists():
            st.video(str(clip_path))
        else:
            st.warning("Clip not found.")

    with c2:
        st.markdown("**Evidence frame**")
        fp = e.get("frame_path")
        if fp and Path(fp).exists():
            st.image(str(fp), use_container_width=True)
        else:
            st.warning("Frame not found.")


def render_search_results(project: ProjectPaths, query: str, results: List[Dict[str, Any]]) -> None:
    st.write(f"Top matches for: `{query}`")
    for r in results:
        cols = st.columns([1, 2], gap="large")
        with cols[0]:
            fp = r["frame_path"]
            if Path(fp).exists():
                st.image(fp, use_container_width=True)
        with cols[1]:
            st.markdown(f"**t={_fmt_time(r['t'])}** (score={r['score']:.3f})")
            st.caption(fp)


def render_qa(project: ProjectPaths, response: Dict[str, Any]) -> None:
    st.markdown("### Answer")
    st.write(response.get("answer", ""))

    st.markdown("### Evidence (timestamps)")
    for e in response.get("evidence", []):
        cols = st.columns([1, 2], gap="large")
        with cols[0]:
            fp = e.get("frame_path")
            if fp and Path(fp).exists():
                st.image(fp, use_container_width=True)
        with cols[1]:
            st.markdown(f"**t={_fmt_time(e['t'])}** (score={e['score']:.3f})")
            cp = e.get("clip_path")
            if cp and Path(cp).exists():
                st.video(cp)
