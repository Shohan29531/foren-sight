# ForenSight

ForenSight is a **local-first video investigation pipeline** using **CLIP embeddings + FAISS** for **grounded text-to-timestamp retrieval** and event-centric playback. It enables faster, repeatable investigations with **ROI/line rules**, **policy-based tagging**, **multi-camera correlation**, and **structured case export** (clips + notes + report).

This repo is intentionally **no-training**: it relies on pretrained models (CLIP) plus lightweight, explainable rules (change-points, ROI motion, tripwires).

---

## What you can do

- **Text → timestamp search** (CLIP text embedding → FAISS retrieval over frame embeddings)
- **Event timeline**
  - **Change-point spikes** (embedding deltas over time; good for “something changed here”)
  - **ROI / Zone rules** (motion, optional person enter/exit, loiter)
  - **Line crossing (tripwire)** rules (motion centroid crosses a user-drawn line; optional direction)
- **Policy-based tagging**
  - Convert video-relative time (t=123.4s) to a real datetime using a configured **video start datetime**
  - Tag events as **after-hours** (e.g., 20:00–06:00)
- **Multi-camera correlation (no tracking)**
  - Correlate events across cameras by **time proximity + CLIP similarity** of evidence frames
- **Investigation UX**
  - Review event clips, add severity + notes, and **export a case zip** containing:
    - evidence clips
    - `case.json` (structured log)
    - `report.html` (human-readable report)

---

## Quickstart (Conda)

> Notes:
> - The ROI/Line drawing UI uses `streamlit-drawable-canvas`, which requires **Streamlit < 1.41**.
>   The provided `environment.yml` already pins a compatible version.

```bash
conda env create -f environment.yml
conda activate video-ops-security
streamlit run app.py
```

Then open the Streamlit URL (usually `http://localhost:8501`).

---

## Quickstart (Docker)

```bash
docker build -t forensight .
docker run --rm -p 8501:8501 -v $(pwd)/data:/app/data forensight
```

---

## Configuration

ForenSight runs fully offline by default (LLM is optional). You can configure via environment variables or a `.env` file.

### Common environment variables

- `DATA_DIR` (default: `./data`) — where projects are stored
- `FRAME_FPS` (default: `1.0`) — frame sampling rate
- `CHANGEPOINT_Z` (default: `2.5`) — change-point sensitivity (lower = more events)
- `CLIP_SECONDS` (default: `10`) — clip radius around an event (± seconds)
- `CLIP_MODEL` (default: `ViT-B-32`)
- `CLIP_PRETRAINED` (default: `openai`)
- `CLIP_DEVICE` (default: `cpu`) — set to `cuda` if you install GPU deps

### LLM (optional)

- `LLM_PROVIDER=none` (default), or `openai`, or `ollama`
- If `openai`:
  - `OPENAI_API_KEY=...`
- If `ollama`:
  - `OLLAMA_BASE_URL=http://localhost:11434`
  - `OLLAMA_MODEL=...` (e.g., `llama3.1`)

Example `.env`:

```bash
DATA_DIR=./data
FRAME_FPS=1.0
CHANGEPOINT_Z=2.5
CLIP_SECONDS=10

LLM_PROVIDER=none
# LLM_PROVIDER=openai
# OPENAI_API_KEY=...

# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.1
```

---

## Recommended workflow in the app

### 1) Create a project
- Upload a video → **Create project**
- Each upload becomes its own project folder under `DATA_DIR/projects/<project_id>/...`

### 2) Run the pipeline
Click **Run full pipeline** to:
1. Normalize the video (ffmpeg)
2. Sample frames (OpenCV; stored with timestamps)
3. Embed frames with CLIP and build a FAISS index
4. Propose change-point events and generate clips
5. Merge any saved ROI/Line rules (if present)
6. Apply policy tags (if start time is configured)

### 3) Policy tagging (real-world time + after-hours)
Go to **Policy**:
- Set **Video start datetime (ISO 8601)** (ideally include timezone, e.g. `2026-02-19T20:13:05-05:00`)
- Enable **After-hours** and set a window (supports overnight windows like 20:00–06:00)
- Click **Apply now** to annotate timeline events with:
  - `local_dt` (real datetime)
  - `tags` including `after_hours` when applicable

### 4) Define Zones (ROI rules)
Go to **Zones**:
- Draw one or more rectangles and name them (e.g., `Doorway`, `Safe`, `FrontDesk`)
- Save zones (`zones.json`)
- Run zone rules to generate:
  - `zone_motion` (motion in ROI)
  - optional `zone_enter` / `zone_exit` (if person detector enabled)
  - optional `zone_loiter` (presence sustained ≥ N seconds)

### 5) Define Lines (Tripwires)
Go to **Lines**:
- Draw one or more lines and name them
- Choose direction: `any`, `neg_to_pos`, `pos_to_neg`
- Save lines (`lines.json`)
- Run line rules to generate:
  - `line_cross` events when motion centroid crosses the line (with debouncing)

### 6) Multi-camera correlation
Go to **Multi-camera**:
- Select multiple projects (cameras)
- Ensure each has a configured **video start datetime**
- Run correlation to create correlated incident groups using:
  - time-window gating (seconds)
  - CLIP similarity threshold on evidence frames
- Add a correlated group to an investigation case in one click

### 7) Build an Investigation Case
Go to **Timeline**:
- Inspect an event → **Add to case**
Go to **Investigation**:
- Set severity + notes per item
- Export a case zip containing clips + `case.json` + `report.html`

---

## How it works (under the hood)

### Text-to-timestamp retrieval
- Sampled frames are embedded with CLIP → vectors
- FAISS index (`IndexFlatIP`) stores normalized embeddings
- A text query is embedded with CLIP text encoder and searched against FAISS
- Results are returned as (timestamp, score, frame evidence)

### Change-point timeline
- Compute cosine similarity between consecutive frame embeddings
- Convert to change magnitude: `delta = 1 - cos_sim`
- Z-score normalize and detect spikes above `CHANGEPOINT_Z`
- Debounce to avoid clustered events
- Generate event clips with ffmpeg

### ROI motion / loiter (no training)
- Background subtraction (MOG2) → foreground mask
- Motion fraction inside ROI triggers `zone_motion`
- Optional OpenCV HOG pedestrian detector toggles person presence
- Loitering triggers when presence stays on ≥ threshold

### Tripwire (line crossing)
- Background subtraction → motion mask
- Motion centroid computed from contours
- Crossing triggers when centroid changes sides of a drawn line
- Optional direction constraint + minimum-gap debouncing

### Multi-camera correlation (no tracking)
- Convert each event to real-world time using its project start datetime
- Candidate pairs require |Δt| ≤ time window
- Evidence frame embeddings compared with cosine similarity
- Union-find clusters linked events into correlated groups

---

## Performance tips

- Lower `FRAME_FPS` (e.g., 0.5) to speed up indexing on CPU
- Increase `CHANGEPOINT_Z` to reduce noisy change-point events
- For busy scenes, increase line/ROI motion thresholds and increase minimum gaps
- The first run loads CLIP weights and can feel slow; subsequent runs are faster

---

## Troubleshooting

### ROI/Line drawing error: `image_to_url`
If you see an error like:
`AttributeError: module 'streamlit.elements.image' has no attribute 'image_to_url'`

Use Streamlit < 1.41. This repo already pins that in `environment.yml`:
- `streamlit>=1.31,<1.41`

If you created your env earlier with a newer Streamlit, reinstall/pin:
```bash
pip install "streamlit<1.41"
```

---

## License
MIT (see `LICENSE`).
