from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from video_ops.pipelines.investigation import export_case_package
from video_ops.storage.project_store import ProjectPaths, ProjectStore


def render_investigation(store: ProjectStore, project_id: str) -> None:
    project = store.get_project(project_id)
    # Global case workspace (supports multi-camera)
    case_key = "case_items"
    if case_key not in st.session_state:
        st.session_state[case_key] = []
    items: List[Dict[str, Any]] = st.session_state[case_key]

    st.caption("Build a case: pick events from the Timeline, add notes/severity, then export a zip (clips + report).")

    meta_key = "case_meta"
    if meta_key not in st.session_state:
        st.session_state[meta_key] = {
            "case_title": "",
            "incident_id": "",
            "investigator": "",
            "case_notes": "",
        }
    meta = st.session_state[meta_key]

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        meta["case_title"] = st.text_input("Case title", value=meta.get("case_title", ""))
        meta["incident_id"] = st.text_input("Incident ID", value=meta.get("incident_id", ""))
        meta["investigator"] = st.text_input("Investigator", value=meta.get("investigator", ""))
    with c2:
        meta["case_notes"] = st.text_area("Case notes", value=meta.get("case_notes", ""), height=120)

    st.divider()
    st.subheader("Evidence items")
    if not items:
        st.info("No evidence yet. Go to Timeline → inspect an event → 'Add to Investigation Case'.")
        return

    # Render editable list
    for idx, it in enumerate(list(items)):
        with st.container(border=True):
            top = st.columns([3, 1])
            with top[0]:
                st.markdown(f"**{it.get('title','(untitled)')}**")
                st.caption(
                    f"proj={it.get('project_id','-')} | id={it.get('id')} | type={it.get('kind')} | zone={it.get('zone_name') or '-'} | line={it.get('line_name') or '-'} | t={float(it.get('t',0.0)):.2f}s"
                    + (f" | local={it.get('local_dt')}" if it.get("local_dt") else "")
                )
            with top[1]:
                if st.button("Remove", key=f"rm_{project.project_id}_{idx}"):
                    items.remove(it)
                    st.experimental_rerun()

            c = st.columns([1, 2])
            with c[0]:
                fp = it.get("frame_path")
                if fp and Path(fp).exists():
                    st.image(fp, use_container_width=True)
            with c[1]:
                cp = it.get("clip_path")
                if cp and Path(cp).exists():
                    st.video(cp)

                it["severity"] = st.selectbox(
                    "Severity",
                    options=["", "low", "medium", "high", "critical"],
                    index=["", "low", "medium", "high", "critical"].index(it.get("severity", "")) if it.get("severity", "") in ["", "low", "medium", "high", "critical"] else 0,
                    key=f"sev_{project.project_id}_{idx}",
                )
                it["notes"] = st.text_area("Item notes", value=it.get("notes", ""), height=80, key=f"note_{project.project_id}_{idx}")

    st.session_state[case_key] = items
    st.session_state[meta_key] = meta

    st.divider()
    st.subheader("Export")
    if st.button("Export case as zip (clips + report)", type="primary"):
        out = export_case_package(project, meta, items)
        # persist a case record
        store.add_case_record(project_id, {"case_id": out["case_id"], "zip_path": out["zip_path"], "report_html_path": out["report_html_path"]})
        zpath = Path(out["zip_path"])
        if zpath.exists():
            st.success(f"Exported: {out['case_id']}")
            st.download_button(
                "Download case zip",
                data=zpath.read_bytes(),
                file_name=zpath.name,
                mime="application/zip",
            )
        else:
            st.error("Failed to create zip.")
