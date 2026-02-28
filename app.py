import os
import uuid
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st

from video_ops.pipelines.ingest import normalize_video, sample_frames
from video_ops.pipelines.indexing import build_or_load_index, search_text
from video_ops.pipelines.events import propose_events, label_events
from video_ops.pipelines.zones import analyze_zones
from video_ops.pipelines.lines import analyze_line_crossings
from video_ops.pipelines.policy import annotate_with_time_windows
from video_ops.pipelines.correlation import build_event_refs, correlate_events
from video_ops.pipelines.qa import answer_question
from video_ops.storage.project_store import ProjectStore
from video_ops.ui.render import render_timeline, render_search_results, render_qa
from video_ops.ui.zones import render_zones_editor
from video_ops.ui.lines import render_lines_editor
from video_ops.ui.investigation import render_investigation

load_dotenv()

st.set_page_config(page_title="Video Ops Security Timeline", layout="wide")

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
FRAME_SAMPLE_FPS = float(os.getenv("FRAME_SAMPLE_FPS", "1"))
CHANGEPOINT_Z = float(os.getenv("CHANGEPOINT_Z", "2.5"))
CLIP_SECONDS_AROUND_EVENT = int(os.getenv("CLIP_SECONDS_AROUND_EVENT", "8"))

# Zone/rule defaults
ZONE_MOTION_THRESH = float(os.getenv("ZONE_MOTION_THRESH", "0.02"))
ZONE_LOITER_SECONDS = float(os.getenv("ZONE_LOITER_SECONDS", "20"))
ZONE_ENABLE_PERSON = os.getenv("ZONE_ENABLE_PERSON", "1").strip() not in ("0", "false", "False")

# Line (tripwire) defaults
LINE_MOTION_THRESH = float(os.getenv("LINE_MOTION_THRESH", "0.01"))
LINE_MIN_GAP_S = float(os.getenv("LINE_MIN_GAP_S", "3"))

store = ProjectStore(DATA_DIR)


def _apply_policy_if_configured(pid: str, events: list) -> list:
    meta = store.load_meta(pid)
    start_iso = str(meta.get("video_start_iso") or "").strip()
    if not start_iso:
        return events
    return annotate_with_time_windows(
        events,
        video_start_iso=start_iso,
        after_hours_start=str(meta.get("after_hours_start", "20:00")),
        after_hours_end=str(meta.get("after_hours_end", "06:00")),
        enabled=bool(meta.get("after_hours_enabled", True)),
    )

st.title("Video Ops Security Timeline")
st.caption("Upload a security video → build a searchable index → generate candidate events → review clips → ask timestamp-grounded questions.")

with st.sidebar:
    st.header("Settings")
    st.write(f"Data dir: `{DATA_DIR}`")
    st.write(f"Frame sampling: `{FRAME_SAMPLE_FPS} fps`")
    st.write(f"Change-point z-threshold: `{CHANGEPOINT_Z}`")
    st.write(f"Clip context: `±{CLIP_SECONDS_AROUND_EVENT}s`")
    st.write(f"Zone motion threshold: `{ZONE_MOTION_THRESH}`")
    st.write(f"Zone loiter seconds: `{ZONE_LOITER_SECONDS}`")

    st.divider()
    st.subheader("Projects")
    projects = store.list_projects()
    selected_project = st.selectbox("Select", ["(new upload)"] + projects)


col_upload, col_main = st.columns([1, 2], gap="large")

with col_upload:
    st.subheader("1) Upload")
    uploaded = st.file_uploader("Video file", type=["mp4", "mov", "mkv", "avi"]) 

    if uploaded is not None:
        if st.button("Create project from upload", type="primary"):
            project_id = store.create_project_id(prefix="sec")
            project = store.init_project(project_id)
            raw_path = project.raw_video_path
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(uploaded.getbuffer())
            st.session_state["project_id"] = project_id
            st.success(f"Created project: {project_id}")

    st.divider()
    st.subheader("2) Process")

    project_id = st.session_state.get("project_id")
    if selected_project != "(new upload)":
        project_id = selected_project
        st.session_state["project_id"] = project_id

    if project_id:
        project = store.get_project(project_id)
        st.write(f"Project: `{project_id}`")
        if project.raw_video_path.exists():
            st.video(str(project.raw_video_path))
        else:
            st.warning("Raw video not found. Recreate the project.")

        if st.button("Run full pipeline", type="primary"):
            with st.status("Running pipeline...", expanded=True) as status:
                st.write("Normalizing video")
                normalize_video(project.raw_video_path, project.norm_video_path)

                st.write("Sampling frames")
                sample_frames(project.norm_video_path, project.frames_dir, fps=FRAME_SAMPLE_FPS)

                st.write("Building search index")
                index = build_or_load_index(project)

                st.write("Proposing events")
                events = propose_events(project, index, z_thresh=CHANGEPOINT_Z, clip_seconds=CLIP_SECONDS_AROUND_EVENT)

                st.write("Labeling events")
                labeled = label_events(project, events)

                # Optional: zone rules
                zones = store.load_zones(project_id)
                if zones:
                    st.write(f"Running zone rules ({len(zones)} zones)")
                    zone_events = analyze_zones(
                        project,
                        zones,
                        motion_threshold=ZONE_MOTION_THRESH,
                        loiter_seconds=ZONE_LOITER_SECONDS,
                        enable_person=ZONE_ENABLE_PERSON,
                        clip_seconds=CLIP_SECONDS_AROUND_EVENT,
                    )
                    labeled = sorted(labeled + zone_events, key=lambda e: float(e.get("t", 0.0)))

                # Optional: line rules
                lines = store.load_lines(project_id)
                if lines:
                    st.write(f"Running line-cross rules ({len(lines)} lines)")
                    line_events = analyze_line_crossings(
                        project,
                        lines,
                        motion_threshold=LINE_MOTION_THRESH,
                        min_gap_s=LINE_MIN_GAP_S,
                        clip_seconds=CLIP_SECONDS_AROUND_EVENT,
                    )
                    labeled = sorted(labeled + line_events, key=lambda e: float(e.get("t", 0.0)))

                # Policy tagging (after-hours)
                labeled = _apply_policy_if_configured(project_id, labeled)

                store.save_events(project_id, labeled)
                status.update(label="Pipeline complete", state="complete", expanded=False)

            st.success("Done. Go to the right panel to review timeline, search, and Q&A.")


with col_main:
    if not project_id:
        st.info("Upload a video and create a project to begin.")
        st.stop()

    project = store.get_project(project_id)

    tab_timeline, tab_zones, tab_lines, tab_policy, tab_multicam, tab_search, tab_qa, tab_case = st.tabs(
        ["Timeline", "Zones", "Lines", "Policy", "Multi-camera", "Search", "Q&A", "Investigation"]
    )

    with tab_timeline:
        st.subheader("Timeline")
        events = store.load_events(project_id)
        if not events:
            st.warning("No events yet. Run the pipeline first.")
        else:
            render_timeline(project, events)

    with tab_zones:
        st.subheader("ROI / Zone rules")
        zones = render_zones_editor(store, project_id)
        st.divider()
        st.markdown("#### Run zone analysis")
        st.caption("Generates rule-based events (motion, enter/exit, loiter) and merges them into the Timeline.")

        colz1, colz2, colz3 = st.columns([1, 1, 2])
        with colz1:
            enable_person = st.checkbox("Enable person detector (HOG)", value=ZONE_ENABLE_PERSON)
        with colz2:
            motion_thr = st.number_input("Motion threshold", min_value=0.0, max_value=1.0, value=float(ZONE_MOTION_THRESH), step=0.005)
        with colz3:
            loiter_s = st.number_input("Loiter seconds", min_value=5.0, max_value=600.0, value=float(ZONE_LOITER_SECONDS), step=5.0)

        if st.button("Run zone rules now", type="primary", disabled=not bool(zones)):
            with st.status("Analyzing zones...", expanded=True) as status:
                # Ensure we have normalized video + frames
                if not project.norm_video_path.exists() and project.raw_video_path.exists():
                    st.write("Normalizing video")
                    normalize_video(project.raw_video_path, project.norm_video_path)
                if not (project.frames_dir / "frames_manifest.json").exists():
                    st.write("Sampling frames")
                    sample_frames(project.norm_video_path if project.norm_video_path.exists() else project.raw_video_path, project.frames_dir, fps=FRAME_SAMPLE_FPS)

                st.write("Generating zone events")
                zone_events = analyze_zones(
                    project,
                    zones,
                    motion_threshold=float(motion_thr),
                    loiter_seconds=float(loiter_s),
                    enable_person=bool(enable_person),
                    clip_seconds=CLIP_SECONDS_AROUND_EVENT,
                )

                st.write("Merging into events.json")
                existing = store.load_events(project_id)
                # Remove old zone events (re-run should replace)
                existing = [e for e in existing if not str(e.get("kind", "")).startswith("zone_")]
                merged = sorted(existing + zone_events, key=lambda e: float(e.get("t", 0.0)))
                merged = _apply_policy_if_configured(project_id, merged)
                store.save_events(project_id, merged)
                status.update(label="Zone analysis complete", state="complete", expanded=False)
            st.success(f"Added {len(zone_events)} zone events to the timeline.")

    with tab_lines:
        st.subheader("Tripwires / Line crossing rules")
        lines = render_lines_editor(store, project_id)
        st.divider()
        st.markdown("#### Run line-cross analysis")
        st.caption("Generates rule-based events when motion crosses a defined line and merges them into the Timeline.")

        coll1, coll2, coll3 = st.columns([1, 1, 2])
        with coll1:
            motion_thr = st.number_input(
                "Motion threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(LINE_MOTION_THRESH),
                step=0.005,
                help="Minimum fraction of moving pixels to consider a frame as 'motion'.",
            )
        with coll2:
            min_gap = st.number_input(
                "Min gap between events (s)",
                min_value=0.0,
                max_value=60.0,
                value=float(LINE_MIN_GAP_S),
                step=0.5,
                help="Debounce to avoid multiple triggers for the same crossing.",
            )
        with coll3:
            st.caption("Tip: For busy scenes, raise motion threshold or increase min-gap.")

        if st.button("Run line rules now", type="primary", disabled=not bool(lines)):
            with st.status("Analyzing line crossings...", expanded=True) as status:
                if not project.norm_video_path.exists() and project.raw_video_path.exists():
                    st.write("Normalizing video")
                    normalize_video(project.raw_video_path, project.norm_video_path)
                if not (project.frames_dir / "frames_manifest.json").exists():
                    st.write("Sampling frames")
                    sample_frames(project.norm_video_path if project.norm_video_path.exists() else project.raw_video_path, project.frames_dir, fps=FRAME_SAMPLE_FPS)

                st.write("Generating line-crossing events")
                line_events = analyze_line_crossings(
                    project,
                    lines,
                    motion_threshold=float(motion_thr),
                    min_gap_s=float(min_gap),
                    clip_seconds=CLIP_SECONDS_AROUND_EVENT,
                )

                st.write("Merging into events.json")
                existing = store.load_events(project_id)
                existing = [e for e in existing if str(e.get("kind", "")) != "line_cross"]
                merged = sorted(existing + line_events, key=lambda e: float(e.get("t", 0.0)))
                merged = _apply_policy_if_configured(project_id, merged)
                store.save_events(project_id, merged)
                status.update(label="Line analysis complete", state="complete", expanded=False)
            st.success(f"Added {len(line_events)} line-crossing events to the timeline.")

    with tab_policy:
        st.subheader("Policy / Time-window rules")
        st.caption(
            "If you provide the video's real start time, the app can tag events that happen during a time window (e.g., after-hours)."
        )

        meta = store.load_meta(project_id)

        cam_name = st.text_input("Camera name (optional)", value=str(meta.get("camera_name", "")), key=f"cam_{project_id}")
        start_iso = st.text_input(
            "Video start datetime (ISO 8601)",
            value=str(meta.get("video_start_iso", "")),
            placeholder="Example: 2026-02-19T20:13:05-05:00",
            key=f"vstart_{project_id}",
            help="Include timezone offset if possible (e.g., -05:00).",
        )

        st.markdown("#### After-hours window")
        ah_enabled = st.checkbox("Enable after-hours tagging", value=bool(meta.get("after_hours_enabled", True)), key=f"ah_en_{project_id}")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            ah_start = st.text_input("After-hours start (HH:MM)", value=str(meta.get("after_hours_start", "20:00")), key=f"ah_s_{project_id}")
        with c2:
            ah_end = st.text_input("After-hours end (HH:MM)", value=str(meta.get("after_hours_end", "06:00")), key=f"ah_e_{project_id}")
        with c3:
            st.caption("Supports overnight windows (e.g., 20:00 → 06:00).")

        if st.button("Save policy settings", type="primary", key=f"save_pol_{project_id}"):
            meta2 = dict(meta)
            meta2["camera_name"] = cam_name.strip()
            meta2["video_start_iso"] = start_iso.strip()
            meta2["after_hours_enabled"] = bool(ah_enabled)
            meta2["after_hours_start"] = ah_start.strip() or "20:00"
            meta2["after_hours_end"] = ah_end.strip() or "06:00"
            store.save_meta(project_id, meta2)
            st.success("Saved project_meta.json")

        st.divider()
        st.markdown("#### Apply policy tags to existing timeline")
        st.caption("Rewrites events.json with `local_dt` and tags like `after_hours`.")
        if st.button("Apply now", key=f"apply_pol_{project_id}"):
            ev = store.load_events(project_id)
            ev2 = _apply_policy_if_configured(project_id, ev)
            store.save_events(project_id, ev2)
            st.success("Updated events.json with policy tags.")

    with tab_multicam:
        st.subheader("Multi-camera correlation")
        st.caption(
            "Correlate events across multiple camera projects by time proximity + CLIP similarity (no tracking / no training)."
        )

        all_projects = store.list_projects()
        default_sel = [project_id] if project_id in all_projects else []
        sel = st.multiselect("Select projects (cameras)", options=all_projects, default=default_sel)

        # quick meta status
        if sel:
            st.markdown("##### Metadata check")
            for pid in sel:
                m = store.load_meta(pid)
                ok = bool(str(m.get("video_start_iso") or "").strip())
                st.write(f"- `{pid}` start time set: {'✅' if ok else '❌'}")
            st.caption("If start times are missing, set them in the Policy tab for each project. Correlation needs real-world time alignment.")

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            tw = st.number_input("Time window (seconds)", min_value=1.0, max_value=600.0, value=30.0, step=5.0)
        with c2:
            st_sim = st.number_input("Similarity threshold", min_value=-1.0, max_value=1.0, value=0.26, step=0.02)
        with c3:
            st.caption("Higher threshold = stricter matching. Typical range: 0.22–0.35.")

        if st.button("Run correlation", type="primary", disabled=len(sel) < 2):
            refs_all = []
            for pid in sel:
                meta = store.load_meta(pid)
                start = str(meta.get("video_start_iso") or "").strip()
                if not start:
                    continue
                evs = store.load_events(pid)
                refs_all.extend(build_event_refs(pid, evs, start))

            result = correlate_events(refs_all, time_window_s=float(tw), sim_threshold=float(st_sim))
            st.session_state["multicam_result"] = result

        result = st.session_state.get("multicam_result")
        if result and result.get("groups"):
            st.markdown("##### Correlated groups")
            for g in result["groups"]:
                with st.container(border=True):
                    st.markdown(
                        f"**{g['group_id']}** &nbsp; projects={', '.join(g.get('projects', []))} &nbsp; max_sim={float(g.get('max_sim',0.0)):.2f} &nbsp; span={float(g.get('span_s',0.0)):.1f}s"
                    )

                    # add all button
                    if st.button(f"Add group {g['group_id']} to case", key=f"addgrp_{g['group_id']}"):
                        if "case_items" not in st.session_state:
                            st.session_state["case_items"] = []
                        existing_ids = {(x.get("project_id"), x.get("id")) for x in st.session_state["case_items"]}

                        # Look up full event payloads for richer fields
                        for evref in g.get("events", []):
                            pid = evref["project_id"]
                            eid = evref["event_id"]
                            full = None
                            for e in store.load_events(pid):
                                if str(e.get("id")) == str(eid):
                                    full = e
                                    break
                            if full is None:
                                full = evref
                            key = (pid, str(full.get("id")))
                            if key in existing_ids:
                                continue
                            st.session_state["case_items"].append(
                                {
                                    "project_id": pid,
                                    "id": full.get("id"),
                                    "kind": full.get("kind"),
                                    "zone_id": full.get("zone_id"),
                                    "zone_name": full.get("zone_name"),
                                    "line_id": full.get("line_id"),
                                    "line_name": full.get("line_name"),
                                    "t": float(full.get("t", 0.0)),
                                    "start": float(full.get("start", full.get("t", 0.0))),
                                    "end": float(full.get("end", full.get("t", 0.0))),
                                    "local_dt": full.get("local_dt"),
                                    "title": full.get("title"),
                                    "description": full.get("description"),
                                    "clip_path": full.get("clip_path"),
                                    "frame_path": full.get("frame_path"),
                                    "confidence": full.get("confidence"),
                                    "severity": "",
                                    "notes": "",
                                    "tags": list(full.get("tags") or []),
                                }
                            )
                        st.success("Added group events to the Investigation case.")

                    # List events
                    for evref in g.get("events", []):
                        pid = evref.get("project_id")
                        title = evref.get("title")
                        st.write(f"- `{pid}` t={float(evref.get('t',0.0)):.1f}s · {title}")
        elif result:
            st.info("No correlated groups found (try increasing time window or lowering similarity threshold).")

    with tab_search:
        st.subheader("Search moments")
        query = st.text_input("Describe what you want to find", placeholder="e.g., person enters doorway, someone approaches counter, a bag left unattended")
        if st.button("Search", disabled=not bool(query.strip())):
            idx = build_or_load_index(project)
            results = search_text(project, idx, query, top_k=12)
            render_search_results(project, query, results)

    with tab_qa:
        st.subheader("Ask a question (timestamp-grounded)")
        q = st.text_area(
            "Question",
            placeholder="e.g., When does a person enter the room? How many times did the door open? Show the clip where someone leaves a package.",
            height=90,
        )
        if st.button("Answer", disabled=not bool(q.strip())):
            idx = build_or_load_index(project)
            response = answer_question(project, idx, q)
            render_qa(project, response)

    with tab_case:
        st.subheader("Investigation Case Builder")
        render_investigation(store, project_id)
