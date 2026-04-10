from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .config import DEFAULT_FPS
from .models import Moment

TOKEN_RE = re.compile(r"[a-z0-9]+")


def download_from_huggingface(
    dataset_root: Path,
    repo_id: str,
    locations: list[str],
    splits: list[str],
    include_videos: bool = False,
    include_docs: bool = False,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    allow_patterns: list[str] = []
    for location in locations:
        for split in splits:
            prefix = f"{location}/{split}"
            allow_patterns.append(f"{prefix}/label.json")
            if include_videos:
                allow_patterns.append(f"{prefix}/*.mp4")

    if include_docs:
        allow_patterns.extend(["README*", "*.md"])

    dataset_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dataset_root),
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )
    return dataset_root


def iter_label_files(
    dataset_root: Path,
    locations: list[str] | None = None,
    splits: list[str] | None = None,
) -> Iterable[Path]:
    wanted_locations = set(locations or [])
    wanted_splits = set(splits or [])
    for label_path in sorted(dataset_root.rglob("label.json")):
        location = label_path.parent.parent.name
        split = label_path.parent.name
        if wanted_locations and location not in wanted_locations:
            continue
        if wanted_splits and split not in wanted_splits:
            continue
        yield label_path


def load_label(label_path: Path) -> list[list[dict]]:
    with label_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_captions(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _video_path_for_label(label_path: Path) -> Path:
    split_name = label_path.parent.name
    return label_path.with_name(f"{split_name}.mp4")


def _bbox_from_object(raw_object: dict) -> list[int]:
    return [
        int(raw_object.get("left", 0)),
        int(raw_object.get("top", 0)),
        int(raw_object.get("right", 0)),
        int(raw_object.get("bottom", 0)),
    ]


def _build_keywords(captions: list[str], location: str, split: str, track_id: str) -> list[str]:
    text = " ".join(captions + [location, split, track_id]).lower()
    return sorted(set(TOKEN_RE.findall(text)))


def _build_text(captions: list[str], location: str, split: str, track_id: str) -> str:
    caption_text = ", ".join(captions) if captions else "unlabeled object"
    return (
        f"{caption_text}. "
        f"traffic scene at {location}. "
        f"dataset split {split}. "
        f"tracked object {track_id}."
    )


def build_moments_from_label(
    label_path: Path,
    fps: float = DEFAULT_FPS,
    group_by_track: bool = True,
) -> list[Moment]:
    frames = load_label(label_path)
    location = label_path.parent.parent.name
    split = label_path.parent.name
    video_path = _video_path_for_label(label_path)

    grouped: dict[str, dict] = {}

    for frame_idx, objects in enumerate(frames):
        for object_idx, raw_object in enumerate(objects):
            track_id_value = raw_object.get("track_id")
            if group_by_track and track_id_value is not None:
                track_key = str(track_id_value)
                moment_id = f"{location}:{split}:track:{track_key}"
            else:
                track_key = f"frame-{frame_idx}-obj-{object_idx}"
                moment_id = f"{location}:{split}:{track_key}"

            bbox = _bbox_from_object(raw_object)
            captions = normalize_captions(raw_object.get("caption"))

            if moment_id not in grouped:
                grouped[moment_id] = {
                    "id": moment_id,
                    "location": location,
                    "split": split,
                    "video_path": str(video_path),
                    "label_path": str(label_path),
                    "track_id": track_key,
                    "start_frame": frame_idx,
                    "end_frame": frame_idx,
                    "captions": set(captions),
                    "representative_bbox": bbox,
                    "sample_frames": [frame_idx],
                }
                continue

            entry = grouped[moment_id]
            entry["end_frame"] = frame_idx
            entry["captions"].update(captions)
            if len(entry["sample_frames"]) < 8 and frame_idx not in entry["sample_frames"]:
                entry["sample_frames"].append(frame_idx)

    moments: list[Moment] = []
    for payload in grouped.values():
        captions = sorted(payload["captions"])
        keywords = _build_keywords(captions, payload["location"], payload["split"], payload["track_id"])
        start_second = payload["start_frame"] / fps
        end_second = (payload["end_frame"] + 1) / fps
        moments.append(
            Moment(
                id=payload["id"],
                location=payload["location"],
                split=payload["split"],
                video_path=payload["video_path"],
                label_path=payload["label_path"],
                track_id=payload["track_id"],
                fps=float(fps),
                start_frame=payload["start_frame"],
                end_frame=payload["end_frame"],
                start_second=round(start_second, 3),
                end_second=round(end_second, 3),
                captions=captions,
                representative_bbox=payload["representative_bbox"],
                sample_frames=payload["sample_frames"],
                text=_build_text(captions, payload["location"], payload["split"], payload["track_id"]),
                keywords=keywords,
            )
        )

    return sorted(moments, key=lambda item: (item.location, item.split, item.start_frame, item.id))


def collect_moments(
    dataset_root: Path,
    fps: float = DEFAULT_FPS,
    group_by_track: bool = True,
    locations: list[str] | None = None,
    splits: list[str] | None = None,
) -> list[Moment]:
    moments: list[Moment] = []
    for label_path in iter_label_files(dataset_root, locations=locations, splits=splits):
        moments.extend(
            build_moments_from_label(
                label_path,
                fps=fps,
                group_by_track=group_by_track,
            )
        )
    return moments
