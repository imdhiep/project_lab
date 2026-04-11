# Surveillance Multimodal Search Engine

This folder contains a full implementation path for Team 119:

- text-to-video moment retrieval
- image-to-video retrieval
- hybrid text + image search
- sparse + dense + CLIP fusion
- FastAPI backend
- Streamlit demo UI
- selective download from Hugging Face or public archives

The project can index either the original LAVA traffic dataset or a person-first tracking dataset such as PersonPath22. It treats the problem as a search engine, not as an object detection training task.

## Why LAVA and PersonPath22

LAVA is a strong fit for vehicle-centric traffic search because it already contains:

- traffic surveillance videos
- frame-level object annotations
- captions per object
- track ids across time

That is enough to build searchable moments directly from existing metadata, then upgrade to visual retrieval with CLIP once videos are available locally.

For people-centric surveillance search, PersonPath22 is the better fit because it contains:

- multi-person video tracking
- static-camera surveillance footage
- visible/amodal person boxes with track ids
- person-first annotations that work with `search_mode=person`

To make people queries verifiable instead of generic, the engine now also supports optional enrichment sidecars:

- person appearance evidence from attribute models trained with datasets such as `PA-100K` or `RAP/RAPv2`
- scene relation evidence such as `near road`, produced by segmentation pipelines trained with `BDD100K` or `Cityscapes`

## Project layout

```text
surveillance_search/
├── surveillance_search/
│   ├── api.py
│   ├── cli.py
│   ├── config.py
│   ├── dataset.py
│   ├── indexer.py
│   ├── models.py
│   ├── streamlit_app.py
│   ├── video_tools.py
│   └── vision.py
├── tests/
├── requirements.txt
└── .gitignore
```

## Install

```powershell
cd "D:\python ky 9\A20-App-119\surveillance_search"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

By default the CLI now keeps everything under `./data/`:

- `./data/lava/` for the LAVA dataset
- `./data/personpath22/` for PersonPath22
- `./data/artifacts/` for generated indexes, manifests, and visual assets

It also still supports the older layouts `./data/<location>/...` and `./data/Lava_Dataset/...` automatically.
The default dataset type is `lava`. To use PersonPath22, pass `--dataset-type personpath22`.

Recommended strongest setup:

- `BAAI/bge-m3` for dense multilingual text search
- `ViT-L-14` CLIP for visual retrieval
- `laion2b_s32b_b82k` pretrained weights

The search stack is tuned for precision-first retrieval:

- weighted reciprocal rank fusion instead of raw score averaging
- lexical precision gating when the query has reliable token overlap
- conservative default weights that favor sparse + dense agreement over CLIP text drift
- optional evidence-aware reranking for person queries, separating:
  - `person evidence`
  - `attribute evidence`
  - `scene evidence`

## Input / output

### Supported query modes

- text only
- image only
- text + image hybrid

### Search output

Each result returns:

- `location`
- `split`
- `video_path`
- `start_frame`, `end_frame`
- `start_second`, `end_second`
- `captions`
- `track_id`
- `score`
- score breakdown by retrieval branch
- representative visual asset if available
- optional `attribute_evidence` and `scene_evidence`
- `attribute_match` and `scene_match` when the query asks for person appearance or scene relations
- `match_quality` such as `fully_verified`, `partially_verified`, or `generic_person_only`

Optional:

- export a short mp4 clip around the matched moment

## Full pipeline

### 1. Download selected data from Hugging Face

Labels only:

```powershell
python -m surveillance_search download --locations amsterdam --splits test
```

Labels + videos:

```powershell
python -m surveillance_search download --locations amsterdam --splits test --include-videos
```

`download` is only implemented for `--dataset-type lava`. PersonPath22 should be downloaded manually and pointed to with `--dataset-root`.

### 1b. One-command bootstrap

For an end-to-end production-style setup, use:

```powershell
python -m surveillance_search bootstrap --locations amsterdam --splits test --include-videos --profile strongest
```

This command:

- downloads the selected split
- builds searchable moments
- extracts visual assets if videos exist
- creates the runtime manifest
- writes the final bundle under `data/artifacts/full/`

For PersonPath22, use the same command shape but pass the dataset type and local root:

```powershell
python -m surveillance_search bootstrap --dataset-type personpath22 --dataset-root ".\personpath22" --profile strongest
```

### 2. Prepare visual assets

This extracts a few representative full frames and object crops per moment.

```powershell
python -m surveillance_search prepare-assets --locations amsterdam --splits test
```

### 3. Build the strongest bundle

This builds:

- sparse TF-IDF branch
- dense sentence-transformer branch
- CLIP image branch
- hybrid-ready bundle metadata

```powershell
python -m surveillance_search build --profile strongest --locations amsterdam --splits test
```

PersonPath22 example:

```powershell
python -m surveillance_search build --dataset-type personpath22 --dataset-root ".\personpath22" --profile strongest
```

Fully end-to-end person-query build:

```powershell
python -m surveillance_search build --dataset-type personpath22 --profile strongest --auto-enrich
```

This one command will:

- collect tracked person moments from video annotations
- extract representative frame and crop assets from the source videos when needed
- infer appearance evidence such as top color and carried items from person crops
- infer scene evidence such as `near_road` from frame previews
- build the final bundle with that evidence already wired into ranking

## Person-query stack scaffold

If you want the project structure that matches the recommended combo:

- `PersonPath22` as the video base
- `PA-100K` as attribute supervision
- `BDD100K` as near-road / drivable-area supervision

run:

```powershell
python scripts/setup_person_query_stack_once.py
```

This will create:

```text
surveillance_search/
├── configs/person_query_stack/
│   ├── stack.json
│   ├── attribute_model.json
│   └── scene_model.json
└── data/
    ├── personpath22/
    ├── pa100k/
    ├── bdd100k/
    ├── prepared/pa100k/
    ├── prepared/bdd100k/
    └── artifacts/models/
```

To also download the public video base and build right away:

```powershell
python scripts/setup_person_query_stack_once.py --download-personpath22 --extract --build-now
```

To prepare the supervision datasets after you place them on disk:

```powershell
python scripts/prepare_pa100k_dataset.py
python scripts/prepare_bdd100k_dataset.py
python scripts/train_attribute_baseline.py
python scripts/train_scene_baseline.py
```

The bundle is written by default to:

```text
surveillance_search/data/artifacts/full/
```

### 3b. Add person-attribute and scene evidence

If you want queries like `person in red shirt near the road` to become evidence-backed, add sidecar JSON files under:

```text
surveillance_search/data/<dataset>/enrichment/
├── person_attributes.json
└── scene_context.json
```

The loader accepts records keyed by `moment_id` or by `video_name + track_id`.

Example `person_attributes.json`:

```json
[
  {
    "video_name": "uid_vid_00000.mp4",
    "track_id": "7",
    "attributes": {
      "top_color": "red",
      "top_type": "shirt",
      "backpack": true
    }
  }
]
```

Example `scene_context.json`:

```json
[
  {
    "video_name": "uid_vid_00000.mp4",
    "track_id": "7",
    "scene": {
      "near_road": true
    }
  }
]
```

After adding enrichment, rebuild:

```powershell
python -m surveillance_search rebuild --dataset-type personpath22 --profile strongest
```

Or let the project generate enrichment itself during build:

```powershell
python -m surveillance_search build --dataset-type personpath22 --profile strongest --auto-enrich
```

### 4. Search by text

```powershell
python -m surveillance_search search --query-text "red car turning left" --top-k 5
```

JSON output:

```powershell
python -m surveillance_search search --query-text "pedestrian crossing the road" --json
```

If you use `--search-mode person` on a dataset that does not contain person-labeled tracks, the search will now fail with a clear error instead of returning non-person matches.

### 5. Search by image

```powershell
python -m surveillance_search search --image-path ".\query.jpg" --top-k 5
```

### 6. Hybrid search

```powershell
python -m surveillance_search search --query-text "white truck near intersection" --image-path ".\query.jpg"
```

Custom branch weights:

```powershell
python -m surveillance_search search --query-text "red sedan" --weights "sparse=0.1,dense=0.4,clip_text=0.2,clip_image=0.3"
```

### 7. Export a demo clip

```powershell
python -m surveillance_search clip --query-text "red car turning left" --rank 1 --output ".\clips\result.mp4"
```

## Run as a service

### FastAPI

```powershell
python -m surveillance_search serve-api --host 127.0.0.1 --port 8000
```

Main endpoints:

- `GET /health`
- `GET /manifest`
- `POST /search`
- `POST /clip`

### Streamlit demo

```powershell
python -m surveillance_search demo --port 8501
```

## Profiles

- `strongest`: sparse + dense + CLIP
- `balanced`: same as strongest, for future tuning
- `text-only`: sparse + dense, no CLIP
- `lite`: sparse only

## Artifact layout

```text
data/artifacts/full/
├── bundle.json
├── moments.json
├── sparse_vectorizer.pkl
├── sparse_matrix.npz
├── dense_embeddings.npy
├── dense.index
├── clip_image_embeddings.npy
├── clip_moment_indices.npy
└── clip.index
```

Visual assets are stored under:

```text
data/artifacts/visual_assets/
```

## Notes

- If you want CLIP image retrieval, videos must be downloaded locally first.
- If your machine is weak, start with one location and one split.
- If you do not have `ffmpeg`, clip export will not work.
- If `faiss-cpu` is missing, the project still keeps `.npy` embeddings for scoring, but FAISS index files will not be written.

## Practical recommendation

For a strong but still realistic demo:

1. Download one location, one split, with videos.
2. Run `prepare-assets`.
3. Build with `--profile strongest`.
4. Demo text query first.
5. Demo image query second.
6. Export one short clip for presentation.

## Near real-time indexing

The project now includes polling-based reindexing for production-style operation.

Rebuild once:

```powershell
python -m surveillance_search rebuild --locations amsterdam --splits test --profile strongest
```

Watch the dataset directory and rebuild automatically:

```powershell
python -m surveillance_search watch-index --locations amsterdam --splits test --poll-seconds 10
```

This is near-real-time indexing, not a streaming pipeline. It is the practical version for this project scope.

## PowerShell shortcuts

Run the full pipeline and start the API:

```powershell
.\run_end_to_end.ps1 -Location amsterdam -Split test -Port 8000
```

Run the watcher in a second terminal:

```powershell
.\run_watch_index.ps1 -Location amsterdam -Split test -PollSeconds 10
```
