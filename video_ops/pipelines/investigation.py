from __future__ import annotations

import html
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from video_ops.storage.project_store import ProjectPaths


def _safe(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum() or ch in "-_ ").strip().replace(" ", "_")


def _fmt_time(seconds: float) -> str:
    seconds = float(seconds)
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m:02d}:{s:05.2f}"


def export_case_package(
    project: ProjectPaths,
    case_meta: Dict[str, Any],
    case_items: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Create an investigation 'case' folder with copied clips + an HTML report, then zip it.

    Returns dict with keys: case_id, case_dir, zip_path, report_html_path
    """
    project.cases_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    case_id = f"case_{ts}_{int(time.time() * 1000) % 100000}"
    case_dir = project.cases_dir / case_id
    clips_dir = case_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Copy evidence clips (best-effort)
    exported_items = []
    for i, it in enumerate(case_items):
        src = Path(it.get("clip_path", ""))
        dst_name = f"{i:03d}_{_safe(it.get('id','item'))}_{_safe(it.get('kind','event'))}_t{float(it.get('t',0.0)):.2f}.mp4"
        dst = clips_dir / dst_name
        if src.exists():
            shutil.copy2(src, dst)
        exported = dict(it)
        exported["exported_clip"] = str(dst.relative_to(case_dir))
        exported_items.append(exported)

    payload = {
        "case_id": case_id,
        "created_at": ts,
        "case_meta": case_meta,
        "items": exported_items,
    }
    (case_dir / "case.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Build HTML report
    title = html.escape(str(case_meta.get("case_title", "Investigation Case")))
    incident = html.escape(str(case_meta.get("incident_id", "")))
    investigator = html.escape(str(case_meta.get("investigator", "")))
    notes = html.escape(str(case_meta.get("case_notes", "")))

    rows = []
    for it in exported_items:
        t = float(it.get("t", 0.0))
        proj = html.escape(str(it.get("project_id", "")))
        local_dt = html.escape(str(it.get("local_dt", "")))
        kind = html.escape(str(it.get("kind", "")))
        zone = html.escape(str(it.get("zone_name", "") or ""))
        line = html.escape(str(it.get("line_name", "") or ""))
        ev_title = html.escape(str(it.get("title", "")))
        sev = html.escape(str(it.get("severity", "")))
        item_notes = html.escape(str(it.get("notes", "")))
        clip_rel = html.escape(str(it.get("exported_clip", "")))
        rows.append(
            f"<tr>"
            f"<td>{proj}</td>"
            f"<td>{_fmt_time(t)}</td>"
            f"<td>{local_dt}</td>"
            f"<td>{kind}</td>"
            f"<td>{zone}</td>"
            f"<td>{line}</td>"
            f"<td>{ev_title}</td>"
            f"<td>{sev}</td>"
            f"<td>{item_notes}</td>"
            f"<td><a href='{clip_rel}'>clip</a></td>"
            f"</tr>"
        )

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: #111; }}
    .meta {{ margin-bottom: 16px; }}
    .meta div {{ margin: 4px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f6f6f6; text-align: left; }}
    .small {{ color: #666; font-size: 12px; }}
    .notes {{ white-space: pre-wrap; background: #fafafa; border: 1px solid #eee; padding: 10px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class='meta'>
    <div><b>Incident ID:</b> {incident}</div>
    <div><b>Investigator:</b> {investigator}</div>
    <div class='small'><b>Case ID:</b> {html.escape(case_id)} &nbsp; <b>Created:</b> {html.escape(ts)}</div>
  </div>
  <h2>Case Notes</h2>
  <div class='notes'>{notes}</div>
  <h2>Evidence Timeline</h2>
  <table>
    <thead>
      <tr>
        <th>Project</th>
        <th>Time</th>
        <th>Local time</th>
        <th>Type</th>
        <th>Zone</th>
        <th>Line</th>
        <th>Title</th>
        <th>Severity</th>
        <th>Notes</th>
        <th>Clip</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <p class='small'>This report was generated locally by the Video Ops Security Timeline app.</p>
</body>
</html>
"""
    report_html_path = case_dir / "report.html"
    report_html_path.write_text(html_doc, encoding="utf-8")

    # Zip the case folder
    zip_path = project.cases_dir / f"{case_id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in case_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(case_dir)))

    return {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "zip_path": str(zip_path),
        "report_html_path": str(report_html_path),
    }
