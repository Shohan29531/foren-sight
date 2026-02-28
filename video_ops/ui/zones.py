from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from video_ops.storage.project_store import ProjectPaths, ProjectStore


def _load_reference_frame(project: ProjectPaths) -> Optional[Tuple[str, int, int]]:
    manifest = project.frames_dir / "frames_manifest.json"
    if not manifest.exists():
        return None
    try:
        j = json.loads(manifest.read_text(encoding="utf-8"))
        frames = j.get("frames", [])
        if not frames:
            return None
        p = frames[0]["path"]
        from PIL import Image

        img = Image.open(p)
        w, h = img.size
        return p, w, h
    except Exception:
        return None


def _rect_norm_from_px(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> List[float]:
    x1 = max(0.0, min(float(img_w), float(x)))
    y1 = max(0.0, min(float(img_h), float(y)))
    x2 = max(0.0, min(float(img_w), float(x) + float(w)))
    y2 = max(0.0, min(float(img_h), float(y) + float(h)))
    if x2 <= x1 + 1:
        x2 = min(float(img_w), x1 + 2.0)
    if y2 <= y1 + 1:
        y2 = min(float(img_h), y1 + 2.0)
    return [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]


def render_zones_editor(store: ProjectStore, project_id: str) -> List[Dict[str, Any]]:
    """Zone/ROI editor.

    Stores zones in `zones.json` as normalized rectangles.
    """
    project = store.get_project(project_id)
    ref = _load_reference_frame(project)
    if ref is None:
        st.warning("No frames found yet. Run frame sampling first (Run full pipeline or Sampling step).")
        return store.load_zones(project_id)

    frame_path, img_w, img_h = ref
    st.caption("Define rectangular ROIs (zones). These are used for rule-based events like motion, entry/exit, and loitering.")

    zones_key = f"zones_{project_id}"
    if zones_key not in st.session_state:
        st.session_state[zones_key] = store.load_zones(project_id)

    zones: List[Dict[str, Any]] = st.session_state[zones_key]

    # Try to provide a drawing UI; fallback to manual coordinates.
    canvas_ok = False
    try:
        # streamlit-drawable-canvas (0.9.3) calls `streamlit.elements.image.image_to_url`,
        # but Streamlit moved that helper in newer versions (e.g., 1.41+).
        # Monkey-patch the old attribute name for compatibility.
        import streamlit.elements.image as st_image  # type: ignore

        if not hasattr(st_image, "image_to_url"):
            try:
                # Newer Streamlit location
                from streamlit.elements.lib.image_utils import image_to_url  # type: ignore
            except Exception:
                # Fallback for potential future moves
                from streamlit.elements.image_utils import image_to_url  # type: ignore
            setattr(st_image, "image_to_url", image_to_url)

        from streamlit_drawable_canvas import st_canvas  # type: ignore

        canvas_ok = True
    except Exception:
        canvas_ok = False

    left, right = st.columns([2, 1], gap="large")
    with left:
        st.markdown("#### Draw a zone")
        if canvas_ok:
            st.caption("Draw a rectangle on the frame, then click 'Add zone from drawing'.")
            from PIL import Image

            bg = Image.open(frame_path)
            canvas_w = min(900, img_w)
            canvas_h = int(canvas_w * (img_h / img_w))
            canvas = st_canvas(
                fill_color="rgba(255, 0, 0, 0.15)",
                stroke_width=2,
                stroke_color="#ff0000",
                background_image=bg,
                background_color="#000000",
                drawing_mode="rect",
                key=f"canvas_{project_id}",
                width=canvas_w,
                height=canvas_h,
            )


            zname = st.text_input("Zone name", value="Doorway", key=f"zname_{project_id}")
            if st.button("Add zone from drawing", key=f"add_zone_{project_id}"):
                objs = (canvas.json_data or {}).get("objects", []) if canvas is not None else []
                rects = [o for o in objs if o.get("type") == "rect"]
                if not rects:
                    st.error("No rectangle found. Draw one first.")
                else:
                    r = rects[-1]
                    # Fabric.js coords: left/top/width/height
                    rect_norm = _rect_norm_from_px(
                        r.get("left", 0),
                        r.get("top", 0),
                        r.get("width", 0),
                        r.get("height", 0),
                        canvas_w,
                        canvas_h,
                    )
                    zid = f"z{len(zones) + 1}"
                    zones.append({"id": zid, "name": zname.strip() or zid, "rect_norm": rect_norm, "ref_size": [img_w, img_h]})
                    st.success(f"Added zone: {zname} ({zid})")
        else:
            st.image(frame_path, use_container_width=True)
            st.info("Optional: install streamlit-drawable-canvas for click-and-drag ROI drawing. Using manual coordinates instead.")

            zname = st.text_input("Zone name", value="Doorway", key=f"zname2_{project_id}")
            x1 = st.number_input("x1", min_value=0, max_value=img_w - 1, value=0, key=f"x1_{project_id}")
            y1 = st.number_input("y1", min_value=0, max_value=img_h - 1, value=0, key=f"y1_{project_id}")
            x2 = st.number_input("x2", min_value=1, max_value=img_w, value=min(img_w, 200), key=f"x2_{project_id}")
            y2 = st.number_input("y2", min_value=1, max_value=img_h, value=min(img_h, 200), key=f"y2_{project_id}")
            if st.button("Add zone (manual)", key=f"add_zone_manual_{project_id}"):
                rect_norm = [float(x1) / img_w, float(y1) / img_h, float(x2) / img_w, float(y2) / img_h]
                zid = f"z{len(zones) + 1}"
                zones.append({"id": zid, "name": zname.strip() or zid, "rect_norm": rect_norm, "ref_size": [img_w, img_h]})
                st.success(f"Added zone: {zname} ({zid})")

    with right:
        st.markdown("#### Current zones")
        if not zones:
            st.caption("No zones defined yet.")
        else:
            for i, z in enumerate(list(zones)):
                st.write(f"**{z.get('name', z.get('id'))}**  ")
                st.caption(f"id={z.get('id')} rect_norm={z.get('rect_norm')}")
                cols = st.columns([1, 1])
                if cols[0].button("Rename", key=f"ren_{project_id}_{i}"):
                    new_name = st.text_input("New name", value=z.get("name", ""), key=f"ren_in_{project_id}_{i}")
                    z["name"] = new_name.strip() or z.get("id")
                if cols[1].button("Delete", key=f"del_{project_id}_{i}"):
                    zones.remove(z)
                    st.experimental_rerun()

        st.divider()
        if st.button("Save zones", type="primary", key=f"save_zones_{project_id}"):
            store.save_zones(project_id, zones)
            st.success("Saved zones.json")

    st.session_state[zones_key] = zones
    return zones
