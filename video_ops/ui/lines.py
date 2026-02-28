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


def _line_norm_from_px(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> List[float]:
    x1 = max(0.0, min(float(img_w), float(x1)))
    y1 = max(0.0, min(float(img_h), float(y1)))
    x2 = max(0.0, min(float(img_w), float(x2)))
    y2 = max(0.0, min(float(img_h), float(y2)))
    # prevent degenerate
    if abs(x2 - x1) + abs(y2 - y1) < 2.0:
        x2 = min(float(img_w), x1 + 5.0)
        y2 = min(float(img_h), y1 + 5.0)
    return [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]


def render_lines_editor(store: ProjectStore, project_id: str) -> List[Dict[str, Any]]:
    """Line/Tripwire editor.

    Stores lines in `lines.json` as normalized endpoints.

    Direction semantics:
      - any: crossing in either direction
      - neg_to_pos: cross-product sign changes from negative to positive
      - pos_to_neg: sign changes from positive to negative
    """
    project = store.get_project(project_id)
    ref = _load_reference_frame(project)
    if ref is None:
        st.warning("No frames found yet. Run frame sampling first (Run full pipeline or Sampling step).")
        return store.load_lines(project_id)

    frame_path, img_w, img_h = ref

    lines_key = f"lines_{project_id}"
    if lines_key not in st.session_state:
        st.session_state[lines_key] = store.load_lines(project_id)

    lines: List[Dict[str, Any]] = st.session_state[lines_key]

    # Canvas support (may be pinned by the environment)
    canvas_ok = False
    try:
        import streamlit.elements.image as st_image  # type: ignore

        if not hasattr(st_image, "image_to_url"):
            try:
                from streamlit.elements.lib.image_utils import image_to_url  # type: ignore
            except Exception:
                from streamlit.elements.image_utils import image_to_url  # type: ignore
            setattr(st_image, "image_to_url", image_to_url)

        from streamlit_drawable_canvas import st_canvas  # type: ignore

        canvas_ok = True
    except Exception:
        canvas_ok = False

    left, right = st.columns([2, 1], gap="large")

    with left:
        st.caption("Draw a 'tripwire' line. The app can generate rule-based events when motion crosses this line.")
        if canvas_ok:
            from PIL import Image

            bg = Image.open(frame_path)
            canvas_w = min(900, img_w)
            canvas_h = int(canvas_w * (img_h / img_w))

            canvas = st_canvas(
                fill_color="rgba(0, 0, 255, 0.05)",
                stroke_width=3,
                stroke_color="#00a2ff",
                background_image=bg,
                background_color="#000000",
                drawing_mode="line",
                key=f"line_canvas_{project_id}",
                width=canvas_w,
                height=canvas_h,
            )

            lname = st.text_input("Line name", value="Tripwire", key=f"lname_{project_id}")
            direction = st.selectbox(
                "Direction",
                options=["any", "neg_to_pos", "pos_to_neg"],
                index=0,
                help="Direction is based on which side of the line motion centroid moves to, using cross-product sign.",
                key=f"ldir_{project_id}",
            )

            if st.button("Add line from drawing", key=f"add_line_{project_id}"):
                objs = (canvas.json_data or {}).get("objects", []) if canvas is not None else []
                lns = [o for o in objs if o.get("type") == "line"]
                if not lns:
                    st.error("No line found. Draw one first.")
                else:
                    o = lns[-1]
                    # Fabric.js line stores x1,y1,x2,y2 in local coords; also has left/top.
                    # We convert to canvas coords by adding left/top when present.
                    left0 = float(o.get("left", 0.0))
                    top0 = float(o.get("top", 0.0))
                    x1 = float(o.get("x1", 0.0)) + left0
                    y1 = float(o.get("y1", 0.0)) + top0
                    x2 = float(o.get("x2", 0.0)) + left0
                    y2 = float(o.get("y2", 0.0)) + top0

                    line_norm = _line_norm_from_px(x1, y1, x2, y2, canvas_w, canvas_h)
                    lid = f"l{len(lines) + 1}"
                    lines.append({"id": lid, "name": (lname.strip() or lid), "line_norm": line_norm, "direction": direction, "ref_size": [img_w, img_h]})
                    st.success(f"Added line: {lname} ({lid})")
        else:
            st.image(frame_path, use_container_width=True)
            st.info("Canvas not available; define line endpoints manually.")

            lname = st.text_input("Line name", value="Tripwire", key=f"lname2_{project_id}")
            direction = st.selectbox("Direction", options=["any", "neg_to_pos", "pos_to_neg"], index=0, key=f"ldir2_{project_id}")
            x1 = st.number_input("x1", min_value=0, max_value=img_w - 1, value=0, key=f"lx1_{project_id}")
            y1 = st.number_input("y1", min_value=0, max_value=img_h - 1, value=0, key=f"ly1_{project_id}")
            x2 = st.number_input("x2", min_value=1, max_value=img_w, value=min(img_w, 200), key=f"lx2_{project_id}")
            y2 = st.number_input("y2", min_value=1, max_value=img_h, value=min(img_h, 200), key=f"ly2_{project_id}")
            if st.button("Add line (manual)", key=f"add_line_manual_{project_id}"):
                line_norm = _line_norm_from_px(float(x1), float(y1), float(x2), float(y2), img_w, img_h)
                lid = f"l{len(lines) + 1}"
                lines.append({"id": lid, "name": (lname.strip() or lid), "line_norm": line_norm, "direction": direction, "ref_size": [img_w, img_h]})
                st.success(f"Added line: {lname} ({lid})")

    with right:
        st.markdown("#### Current lines")
        if not lines:
            st.caption("No lines defined yet.")
        else:
            for i, ln in enumerate(list(lines)):
                st.write(f"**{ln.get('name', ln.get('id'))}**")
                st.caption(f"id={ln.get('id')} dir={ln.get('direction','any')} line_norm={ln.get('line_norm')}")
                cols = st.columns([1, 1])
                if cols[0].button("Rename", key=f"lren_{project_id}_{i}"):
                    new_name = st.text_input("New name", value=ln.get("name", ""), key=f"lren_in_{project_id}_{i}")
                    ln["name"] = new_name.strip() or ln.get("id")
                if cols[1].button("Delete", key=f"ldel_{project_id}_{i}"):
                    lines.remove(ln)
                    st.experimental_rerun()

        st.divider()
        if st.button("Save lines", type="primary", key=f"save_lines_{project_id}"):
            store.save_lines(project_id, lines)
            st.success("Saved lines.json")

    st.session_state[lines_key] = lines
    return lines
