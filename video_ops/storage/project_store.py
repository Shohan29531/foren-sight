import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ProjectPaths:
    project_id: str
    root: Path

    @property
    def raw_video_path(self) -> Path:
        return self.root / "raw" / "input.mp4"

    @property
    def norm_video_path(self) -> Path:
        return self.root / "video" / "normalized.mp4"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def index_dir(self) -> Path:
        return self.root / "index"

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    @property
    def events_path(self) -> Path:
        return self.root / "events.json"

    @property
    def zones_path(self) -> Path:
        return self.root / "zones.json"

    @property
    def lines_path(self) -> Path:
        return self.root / "lines.json"

    @property
    def meta_path(self) -> Path:
        return self.root / "project_meta.json"

    @property
    def cases_dir(self) -> Path:
        return self.root / "cases"

    @property
    def case_index_path(self) -> Path:
        return self.cases_dir / "cases.json"


class ProjectStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.projects_dir = self.data_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def create_project_id(self, prefix: str = "proj") -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{ts}_{int(time.time() * 1000) % 100000}"

    def init_project(self, project_id: str) -> ProjectPaths:
        root = self.projects_dir / project_id
        (root / "raw").mkdir(parents=True, exist_ok=True)
        (root / "video").mkdir(parents=True, exist_ok=True)
        (root / "frames").mkdir(parents=True, exist_ok=True)
        (root / "index").mkdir(parents=True, exist_ok=True)
        (root / "clips").mkdir(parents=True, exist_ok=True)
        (root / "cases").mkdir(parents=True, exist_ok=True)
        return ProjectPaths(project_id=project_id, root=root)

    def get_project(self, project_id: str) -> ProjectPaths:
        root = self.projects_dir / project_id
        return ProjectPaths(project_id=project_id, root=root)

    def list_projects(self) -> List[str]:
        if not self.projects_dir.exists():
            return []
        items = [p.name for p in self.projects_dir.iterdir() if p.is_dir()]
        # newest first based on timestamp in name when present
        return sorted(items, reverse=True)

    def save_events(self, project_id: str, events: List[Dict[str, Any]]) -> None:
        project = self.get_project(project_id)
        project.events_path.write_text(json.dumps(events, indent=2), encoding="utf-8")

    def load_events(self, project_id: str) -> List[Dict[str, Any]]:
        project = self.get_project(project_id)
        if not project.events_path.exists():
            return []
        try:
            return json.loads(project.events_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def save_zones(self, project_id: str, zones: List[Dict[str, Any]]) -> None:
        project = self.get_project(project_id)
        project.zones_path.write_text(json.dumps(zones, indent=2), encoding="utf-8")

    def load_zones(self, project_id: str) -> List[Dict[str, Any]]:
        project = self.get_project(project_id)
        if not project.zones_path.exists():
            return []

    def save_lines(self, project_id: str, lines: List[Dict[str, Any]]) -> None:
        project = self.get_project(project_id)
        project.lines_path.write_text(json.dumps(lines, indent=2), encoding="utf-8")

    def load_lines(self, project_id: str) -> List[Dict[str, Any]]:
        project = self.get_project(project_id)
        if not project.lines_path.exists():
            return []
        try:
            return json.loads(project.lines_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def save_meta(self, project_id: str, meta: Dict[str, Any]) -> None:
        project = self.get_project(project_id)
        project.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load_meta(self, project_id: str) -> Dict[str, Any]:
        project = self.get_project(project_id)
        if not project.meta_path.exists():
            return {}
        try:
            return json.loads(project.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        try:
            return json.loads(project.zones_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def list_cases(self, project_id: str) -> List[Dict[str, Any]]:
        project = self.get_project(project_id)
        if not project.case_index_path.exists():
            return []
        try:
            return json.loads(project.case_index_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def add_case_record(self, project_id: str, case_record: Dict[str, Any]) -> None:
        project = self.get_project(project_id)
        project.cases_dir.mkdir(parents=True, exist_ok=True)
        existing = self.list_cases(project_id)
        existing.append(case_record)
        project.case_index_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
