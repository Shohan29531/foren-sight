# Video Ops Security Timeline

Turn long security videos into:
- **Event timeline** (timestamped candidate events)
- **ROI / zone rules** (motion, enter/exit, loitering inside defined zones)
- **Searchable moments** (text query -> timestamps)
- **Timestamp-grounded Q&A** (answers cite exact times + supporting frames)
- **Clips** around events for quick review
- **Investigation case export** (build a case, add notes/severity, export a zip with clips + HTML report)

This repo is intentionally **no-training**: it uses pretrained models (CLIP embeddings) + lightweight change-point logic.

## What you get
- Streamlit UI: upload a video, process it, browse timeline, search, and ask questions.
- Local indexing with FAISS.
- Optional LLM summarization:
  - OpenAI API (if you provide `OPENAI_API_KEY`), or
  - Ollama (if you run a local model).
  - If neither is configured, the app still works and returns evidence + a structured, non-LLM response.

---

## 1) Quickstart (recommended)

### Option A: Docker

```bash
docker build -t video-ops-security .
docker run --rm -p 8501:8501 -v $(pwd)/data:/app/data video-ops-security
```
Open: http://localhost:8501

### Option B: Local (Conda recommended)

This repo works great with **conda** so you don’t have to manage a virtualenv manually.

```bash
conda env create -f environment.yml
conda activate video-ops-security

# Optional: configure runtime vars via a .env file (read automatically by the app)
cp .env.example .env

streamlit run app.py
```

**Tip (no `.env` file):** you can also set variables inside the conda environment:

```bash
conda env config vars set LLM_PROVIDER=none DATA_DIR=./data
conda deactivate && conda activate video-ops-security
```


---

## 2) How it works (pipeline)

1. **Ingest**
   - Normalize video
   - Sample frames (default 1 frame / 1 sec)
2. **Index**
   - Compute CLIP image embeddings for frames
   - Store in FAISS with timestamp metadata
3. **Event proposals**
   - Compute embedding deltas over time
   - Identify change points -> candidate events
   - (Optional) add ROI/zone rule events (motion + person-like activity)
4. **Timeline labels**
   - Heuristic labeler OR LLM labeler (grounded to frames)
5. **Q&A**
   - Retrieve top timestamps for the question
   - Answer with timestamp citations

---

## 3) Configuration

Create a `.env` (see `.env.example`). Key settings:

- `DATA_DIR`: where processed videos/frames/index live
- `FRAME_SAMPLE_FPS`: frame sampling rate (e.g., `1` means 1 frame per second)
- `CLIP_MODEL`: e.g., `ViT-B-32`
- `LLM_PROVIDER`: `none | openai | ollama`

### OpenAI (optional)
Set:
- `LLM_PROVIDER=openai`
- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=gpt-4o-mini` (example)

### Ollama (optional)
Set:
- `LLM_PROVIDER=ollama`
- `OLLAMA_HOST=http://localhost:11434`
- `OLLAMA_MODEL=qwen2.5:14b` (example)

---

## 4) Notes on security use-cases

This is a **review assistant**, not a real-time alarm system. For real deployments you would add:
- camera calibration / polygon ROIs
- stronger tracking
- privacy controls + redaction
- audit logging

---

## 5) ROI / Zone rules + Investigation workflow

1) Run the full pipeline once (so the app samples frames).
2) Go to **Zones**:
   - Draw one or more rectangular ROIs (e.g., Doorway, Counter, Safe).
   - Save.
   - Run **Zone rules** to generate events like:
     - Motion in zone
     - Person entered/exited zone (OpenCV HOG detector; no training)
     - Loitering in zone
3) Go to **Timeline** and add relevant events to the **Investigation Case**.
4) Go to **Investigation**:
   - Fill case metadata
   - Add severity + notes per evidence item
   - Export a **zip** containing copied clips + `case.json` + `report.html`

---

## 6) Repo layout

- `app.py` — Streamlit UI
- `video_ops/pipelines/` — ingest, indexing, event proposals
- `video_ops/models/` — CLIP embedder + optional detectors
- `video_ops/storage/` — metadata store + FAISS persistence
- `video_ops/ui/` — UI helpers

---

## 7) License
MIT
