# Surveillance Search Project

This repository contains Team 119's AI project for surveillance retrieval. The main product is a multimodal search system that takes a text query, an image query, or both, then returns ranked video tracks from surveillance datasets such as `LAVA` and `PersonPath22`.

The repo is organized so a new contributor can do three things quickly:

- understand what the AI system does and does not do
- run the search app locally without guessing hidden steps
- follow the team's working rhythm for experiments, debugging, and documentation

## What This Project Solves

The core problem is retrieval, not detector training.

Given a query like:

- `white van near intersection`
- `person crossing the street`
- a reference crop image of a target object or person

the system builds a searchable index over labeled tracks, then ranks the most relevant moments using a fusion of sparse text search, dense embeddings, and optional CLIP-based visual retrieval.

The main application lives in [surveillance_search/README.md](/teamspace/studios/this_studio/project_lab/surveillance_search/README.md), while this root README is the onboarding guide for the whole repository.

## Problem Statement

The project aims to make surveillance footage easier to query by description instead of manual scrubbing through long videos.

Typical use cases:

- find traffic objects from a natural-language prompt
- retrieve person tracks from a people-centric dataset
- compare text-only and text-plus-image retrieval
- inspect ranking evidence and search limitations honestly

Non-goals:

- full end-to-end detector training inside this repo
- guaranteed attribute-level understanding for datasets that only provide generic labels

Example:

- `PersonPath22` can confirm there is a `person track`
- it cannot reliably confirm `red shirt` unless the project adds an attribute-captioning stage

## Input / Output

### Input

The system accepts one or both of the following:

- text query
- image query

Example text-only request:

```json
{
  "query_text": "person in red shirt near the road",
  "search_mode": "person",
  "top_k": 5
}
```

Example hybrid request:

```json
{
  "query_text": "white van near intersection",
  "image_path": "query.jpg",
  "search_mode": "default",
  "top_k": 5
}
```

### Output

Each result is a ranked track or moment with retrieval evidence.

Important fields:

- `score`: final fused ranking score
- `captions`: text labels available for that track
- `video_path`: source video
- `track_id`: track identifier
- `start_frame`, `end_frame`, `start_second`, `end_second`
- `frame_info`: preview frame, bbox, sample frames, split, location
- `score_breakdown`, `normalized_breakdown`, `fusion_breakdown`
- `person_match`: how strongly the result supports a person track
- `precision_match`: how much of the literal text description is supported
- `result_warning`: warning shown when the query asks for attributes the dataset does not label

Example result shape:

```json
{
  "caption_text": "person",
  "score": 0.000294,
  "person_match": 1.0,
  "precision_match": 0.0,
  "match_quality": "generic_person_only",
  "result_warning": "Generic person-only match: this dataset confirms a person track here, but it does not label clothing color or scene relations like 'near the road'.",
  "video_path": "data/personpath22/raw_data/uid_vid_00008.mp4"
}
```

## Data Description

The project currently supports two dataset families.

### 1. `LAVA`

Best for:

- vehicle-centric traffic search
- captioned objects such as vans, bicycles, boats, trucks

Typical layout:

```text
surveillance_search/data/lava/<location>/<split>/
├── label.json
└── <split>.mp4
```

Label format:

- frame-organized object list
- each object may contain `caption`, bbox coordinates, and `track_id`

### 2. `PersonPath22`

Best for:

- people-centric surveillance retrieval
- static-camera multi-person tracking

Typical layout:

```text
surveillance_search/data/personpath22/
├── annotations/
│   └── anno_visible_2022/*.mp4.json
└── raw_data/*.mp4
```

Important limitation:

- `PersonPath22` labels person tracks and boxes
- it does not natively label attributes like clothing color, pose description, or scene relations such as `near the road`

That limitation must be documented clearly because it directly affects retrieval quality and how results should be interpreted.

### Generated Artifacts

Built indexes and visual assets are stored under:

```text
surveillance_search/data/artifacts/
├── full/
└── visual_assets/
```

## Model Stack

The current retrieval pipeline uses three branches.

### Sparse lexical retrieval

- `TF-IDF`
- strongest when literal terms overlap with available captions or keywords

### Dense text retrieval

- `BAAI/bge-m3`
- multilingual sentence embedding model
- used to retrieve semantically similar tracks even when lexical overlap is weak

### Visual retrieval

- `ViT-L-14`
- pretrained weights: `laion2b_s32b_b82k`
- used for CLIP image search and CLIP text-image alignment when visual assets exist

### Fusion strategy

Ranking uses a precision-first hybrid setup:

- weighted reciprocal rank fusion
- text precision gating
- person-mode filtering and boosting
- explicit warnings for generic person-only matches on attribute-heavy queries

## Repository Structure

```text
project_lab/
├── README.md
├── AGENTS.md
├── JOURNAL.md
├── WORKLOG.md
├── scripts/
├── src/
└── surveillance_search/
    ├── README.md
    ├── requirements.txt
    ├── data/
    ├── scripts/
    ├── surveillance_search/
    └── tests/
```

What each top-level file is for:

- [surveillance_search/README.md](/teamspace/studios/this_studio/project_lab/surveillance_search/README.md): detailed app commands and dataset operations
- [AGENTS.md](/teamspace/studios/this_studio/project_lab/AGENTS.md): rules for AI coding agents used on the repo
- [JOURNAL.md](/teamspace/studios/this_studio/project_lab/JOURNAL.md): weekly product and learning log
- [WORKLOG.md](/teamspace/studios/this_studio/project_lab/WORKLOG.md): technical decisions and major changes

## Prerequisites

Minimum local setup:

- Python `3.10+`
- `pip`
- `ffmpeg` if you want clip export or visual asset extraction
- enough disk space for datasets and built artifacts

Helpful but optional:

- `aws` CLI for downloading `PersonPath22` via the provided script
- GPU for faster embedding and CLIP workloads

## Quick Start

This is the fastest practical path to a running local demo with `PersonPath22`.

### 1. Create environment

```bash
cd /teamspace/studios/this_studio/project_lab/surveillance_search
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download `PersonPath22`

```bash
python scripts/download_personpath22_once.py --mode full --extract
```

Expected output location:

```text
surveillance_search/data/personpath22/
```

### 3. Build the bundle

```bash
python -m surveillance_search build --dataset-type personpath22 --profile strongest
```

Expected output:

```text
surveillance_search/data/artifacts/full/bundle.json
```

### 4. Run the demo UI

```bash
python -m surveillance_search demo --port 8501
```

Open:

```text
http://localhost:8501
```

## Run Guide

### Build on `LAVA`

```bash
python -m surveillance_search build --dataset-type lava --locations amsterdam --splits test --profile strongest
```

### Build on `PersonPath22`

```bash
python -m surveillance_search build --dataset-type personpath22 --profile strongest
```

### Search from CLI

```bash
python -m surveillance_search search --query-text "person crossing the street" --search-mode person --top-k 5
```

### Export a clip

```bash
python -m surveillance_search clip --query-text "white van near intersection" --rank 1 --output clips/result.mp4
```

### Rebuild after dataset changes

```bash
python -m surveillance_search rebuild --dataset-type personpath22 --profile strongest
```

## Configuration

Main CLI controls:

- `--dataset-type`: `lava` or `personpath22`
- `--dataset-root`: override default data folder
- `--profile`: `strongest`, `balanced`, `text-only`, `lite`
- `--output-dir`: where bundle files are written
- `--assets-dir`: where visual assets are written
- `--search-mode`: `default` or `person`
- `--top-k`: number of ranked results returned

Operational defaults:

- default `LAVA` root: `surveillance_search/data/lava`
- default `PersonPath22` root: `surveillance_search/data/personpath22`
- default bundle root: `surveillance_search/data/artifacts/full`
- default visual assets root: `surveillance_search/data/artifacts/visual_assets`

Config quality note:

- `strongest` gives the best retrieval stack, but it only uses CLIP if visual assets exist
- if visual assets are missing, the bundle still works with sparse + dense text retrieval

## Workflow

This is the recommended end-to-end workflow for the team.

### Product / engineering workflow

1. Pull latest code from `main`
2. Create a branch such as `feature/<short-name>` or `fix/<short-name>`
3. Update or add data under `surveillance_search/data/...`
4. Build or rebuild the bundle
5. Verify behavior in CLI or Streamlit
6. Run tests before pushing
7. Update [WORKLOG.md](/teamspace/studios/this_studio/project_lab/WORKLOG.md) if a technical decision changed
8. Update [JOURNAL.md](/teamspace/studios/this_studio/project_lab/JOURNAL.md) for weekly progress

### AI pipeline workflow

1. Acquire dataset
2. Validate data layout
3. Build searchable moments
4. Extract visual assets if needed
5. Build retrieval bundle
6. Run search queries
7. Inspect evidence and known limitations
8. Tune search logic or UI only after verifying the data supports the intended query

## Evaluation Expectations

This project is a retrieval system, so the most important quality questions are:

- does the top result belong to the right semantic category
- does the result have evidence for the literal query terms
- is the UI honest when the dataset cannot support certain attributes

Current qualitative expectation:

- `LAVA` is better for object and vehicle queries
- `PersonPath22` is better for person-track queries
- attribute-heavy person queries such as `red shirt` should be treated cautiously unless an extra captioning or attribute model is added

## Troubleshooting

### `No moments found. Download labels first or check dataset path.`

Check:

- dataset is actually under `surveillance_search/data/personpath22` or `surveillance_search/data/lava`
- annotation files exist
- you used the right `--dataset-type`

### `CLIP branch disabled because no visual assets were found after asset preparation.`

This means:

- the bundle built successfully
- but visual assets were not available, so CLIP search is off

You can still use sparse + dense search. If you need CLIP, verify:

- the source `.mp4` files exist
- `ffmpeg` is installed
- visual asset extraction is working

### Query returns `person` but not `red shirt`

This is expected on `PersonPath22` right now.

Reason:

- the dataset confirms generic person tracks
- it does not label clothing color or scene relations

The app now marks these as `generic_person_only` matches instead of pretending the attribute was verified.

### Streamlit loads slowly on first search

Expected on the first query because embedding and CLIP models load once into memory. Later queries in the same process are faster.

## Environment and Repo Operations

Root environment template: [.env.example](/teamspace/studios/this_studio/project_lab/.env.example)

This root file is mostly for:

- AI tool provider keys
- default model selection for repo tooling
- logging hooks used in the course setup

If your workflow uses the root repo tooling, copy and fill it:

```bash
cd /teamspace/studios/this_studio/project_lab
cp .env.example .env
```

If required for the course setup, install hooks once:

```bash
bash scripts/setup_hooks.sh
```

## Documentation Standards

To keep this repository maintainable, documentation should satisfy these rules:

- commands must be copy-pasteable
- examples must reflect the real file layout
- dataset limitations must be stated explicitly
- README changes should stay consistent with actual code behavior

The difference between an `8/10` README and a `9.5/10` README in this repo is simple:

- `8/10`: reader understands the project
- `9.5/10`: reader can run it, debug it, and avoid over-claiming what the dataset supports

## Contacts

Maintainers:

- Team 119 contributors working in this repository

When changing product direction or infrastructure assumptions, record it in:

- [WORKLOG.md](/teamspace/studios/this_studio/project_lab/WORKLOG.md)
- [JOURNAL.md](/teamspace/studios/this_studio/project_lab/JOURNAL.md)
