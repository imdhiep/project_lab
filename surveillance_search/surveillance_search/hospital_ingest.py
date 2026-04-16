from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from .enrichment import summarize_attribute_evidence, summarize_scene_evidence
from .enrichment_inference import infer_enrichment_payloads
from .models import Moment
from .video_tools import extract_visual_assets, sanitize_name

CATALOG_FILENAME = "catalog.sqlite3"
METADATA_VERSION = 1
DEFAULT_QUEUE_SIZE = 32
DEFAULT_DETECTION_FPS = 1.0
DEFAULT_MIN_TRACK_FRAMES = 2
DEFAULT_MIN_PERSON_AREA = 1600
DEFAULT_TRACK_IOU = 0.35


@dataclass
class HospitalIngestConfig:
    dataset_root: Path
    assets_dir: Path
    ffmpeg_bin: str = "ffmpeg"
    preferred_encoder: str = "hevc_nvenc"
    queue_size: int = DEFAULT_QUEUE_SIZE
    detection_fps: float = DEFAULT_DETECTION_FPS
    min_track_frames: int = DEFAULT_MIN_TRACK_FRAMES
    min_person_area: int = DEFAULT_MIN_PERSON_AREA
    track_iou_threshold: float = DEFAULT_TRACK_IOU
    enable_enrichment: bool = True
    clip_model_name: str = "ViT-L-14"
    clip_pretrained: str = "laion2b_s32b_b82k"
    device: str | None = None
    batch_size: int = 16
    max_assets_per_moment: int = 3
    crop_padding: float = 0.08
    timeline_stride_seconds: float = 1.0
    max_timeline_segments: int = 16


def hospital_encoded_dir(dataset_root: Path) -> Path:
    return Path(dataset_root) / "encoded"


def hospital_metadata_dir(dataset_root: Path) -> Path:
    return Path(dataset_root) / "metadata"


def hospital_catalog_path(dataset_root: Path) -> Path:
    return Path(dataset_root) / CATALOG_FILENAME


def ensure_hospital_layout(dataset_root: Path) -> dict[str, Path]:
    root = Path(dataset_root)
    encoded_dir = hospital_encoded_dir(root)
    metadata_dir = hospital_metadata_dir(root)
    for path in (root, encoded_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)
    _ensure_catalog(hospital_catalog_path(root))
    return {
        "root": root,
        "encoded_dir": encoded_dir,
        "metadata_dir": metadata_dir,
        "catalog_path": hospital_catalog_path(root),
    }


def _connect_catalog(catalog_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(catalog_path))
    connection.row_factory = sqlite3.Row
    return connection


def _ensure_catalog(catalog_path: Path) -> None:
    with _connect_catalog(catalog_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                camera_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                encoded_path TEXT NOT NULL,
                metadata_path TEXT NOT NULL,
                recorded_start TEXT NOT NULL,
                recorded_end TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                codec TEXT,
                width INTEGER,
                height INTEGER,
                fps REAL,
                duration_seconds REAL,
                file_size_bytes INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                moment_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                track_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                start_second REAL NOT NULL,
                end_second REAL NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY(video_id) REFERENCES videos(video_id) ON DELETE CASCADE
            )
            """
        )
        connection.commit()


def parse_recorded_time(value: str | datetime | None, fallback: datetime) -> datetime:
    if value is None:
        return fallback.astimezone(timezone.utc)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return fallback.astimezone(timezone.utc)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _slug(value: str) -> str:
    return sanitize_name(str(value or "").strip()) or "unknown"


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_video_id(camera_id: str, recorded_start: datetime) -> str:
    return f"{_slug(camera_id)}_{recorded_start.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _default_camera_id(source_path: Path) -> str:
    stem = source_path.stem.strip()
    return _slug(stem.split("_")[0] if "_" in stem else stem)


def probe_video(video_path: Path) -> dict:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for hospital video probing.") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()

    duration_seconds = float(frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
    stat = video_path.stat()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
        "file_size_bytes": int(stat.st_size),
    }


def convert_video_to_h265(
    source_path: Path,
    output_path: Path,
    ffmpeg_bin: str = "ffmpeg",
    preferred_encoder: str = "hevc_nvenc",
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempted_encoders = [preferred_encoder]
    if preferred_encoder != "libx265":
        attempted_encoders.append("libx265")

    for encoder in attempted_encoders:
        command = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-an",
            "-c:v",
            encoder,
        ]
        if encoder == "hevc_nvenc":
            command.extend(["-preset", "p4", "-cq", "28"])
        else:
            command.extend(["-preset", "medium", "-crf", "28"])
        command.append(str(output_path))
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return {"encoder": encoder, "path": str(output_path)}

    stderr = completed.stderr.strip() if completed.stderr else "unknown ffmpeg failure"
    raise RuntimeError(f"Failed to convert {source_path} to H.265. Last error: {stderr}")


def _bbox_area(bbox: list[int]) -> int:
    return max(int(bbox[2]) - int(bbox[0]), 0) * max(int(bbox[3]) - int(bbox[1]), 0)


def _bbox_iou(left: list[int], right: list[int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    if intersection <= 0:
        return 0.0
    union = _bbox_area(left) + _bbox_area(right) - intersection
    return float(intersection / union) if union > 0 else 0.0


def _detect_people_hog(frame) -> list[list[int]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for hospital person detection.") from exc

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    if len(rects) == 0:
        return []

    boxes_xywh = []
    confidences = []
    for (x, y, w, h), weight in zip(rects, weights):
        if int(w) * int(h) <= 0:
            continue
        boxes_xywh.append([int(x), int(y), int(w), int(h)])
        confidences.append(float(weight))
    if not boxes_xywh:
        return []

    kept_indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences, score_threshold=0.0, nms_threshold=0.35)
    if len(kept_indices) == 0:
        return []

    results: list[list[int]] = []
    for raw_index in kept_indices:
        index = int(raw_index[0] if isinstance(raw_index, (list, tuple)) else raw_index)
        x, y, w, h = boxes_xywh[index]
        results.append([x, y, x + w, y + h])
    return results


def _sample_stride(source_fps: float, detection_fps: float) -> int:
    if source_fps <= 0 or detection_fps <= 0:
        return 1
    return max(int(round(source_fps / detection_fps)), 1)


def _bbox_center(bbox: list[int]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _movement_labels(previous_bbox: list[int] | None, current_bbox: list[int]) -> tuple[list[str], dict]:
    if previous_bbox is None:
        return ["appears"], {"direction": "unknown", "speed_px": 0.0, "depth": "steady"}

    prev_cx, prev_cy = _bbox_center(previous_bbox)
    cur_cx, cur_cy = _bbox_center(current_bbox)
    dx = cur_cx - prev_cx
    dy = cur_cy - prev_cy
    distance = (dx**2 + dy**2) ** 0.5
    prev_area = max(_bbox_area(previous_bbox), 1)
    cur_area = max(_bbox_area(current_bbox), 1)
    scale_ratio = cur_area / prev_area

    labels: list[str] = []
    if distance < 8:
        labels.append("standing")
    elif distance < 30:
        labels.append("walking")
    else:
        labels.append("moving_quickly")

    if abs(dx) >= 6:
        labels.append("moving_right" if dx > 0 else "moving_left")
    if abs(dy) >= 6:
        labels.append("moving_down" if dy > 0 else "moving_up")
    if scale_ratio > 1.12:
        labels.append("approaching_camera")
        depth = "approaching"
    elif scale_ratio < 0.9:
        labels.append("moving_away")
        depth = "moving_away"
    else:
        depth = "steady"

    return labels, {
        "direction": labels[-1] if labels else "unknown",
        "speed_px": round(distance, 3),
        "dx": round(dx, 3),
        "dy": round(dy, 3),
        "depth": depth,
        "scale_ratio": round(scale_ratio, 4),
    }


def _summarize_action_labels(labels: list[str]) -> str:
    ordered = []
    seen: set[str] = set()
    for label in labels:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    if not ordered:
        return "person visible in frame"
    return ", ".join(ordered)


def _normalize_segment_action_labels(labels: list[str], movement_stats: list[dict]) -> list[str]:
    if not labels:
        return []

    normalized: list[str] = []
    motion_priority = ["moving_quickly", "walking", "standing", "appears"]
    direction_priority = ["moving_left", "moving_right", "moving_up", "moving_down"]
    depth_priority = ["approaching_camera", "moving_away"]

    for group in (motion_priority, direction_priority, depth_priority):
        for label in group:
            if label in labels:
                normalized.append(label)
                break

    if not normalized and movement_stats:
        normalized.append("standing" if movement_stats[-1].get("speed_px", 0.0) < 8 else "walking")
    return normalized


def _segment_track_timeline(
    *,
    frames: list[int],
    bboxes: list[list[int]],
    fps: float,
    recorded_start: datetime,
    stride_seconds: float,
    max_segments: int,
) -> list[dict]:
    if not frames or not bboxes:
        return []

    stride_frames = max(int(round(max(stride_seconds, 0.1) * max(fps, 1.0))), 1)
    segments: list[dict] = []
    previous_bbox: list[int] | None = None

    for offset in range(0, min(len(frames), max_segments * stride_frames), stride_frames):
        frame_slice = frames[offset : offset + stride_frames]
        bbox_slice = bboxes[offset : offset + stride_frames]
        if not frame_slice or not bbox_slice:
            continue
        anchor_bbox = bbox_slice[len(bbox_slice) // 2]
        action_labels: list[str] = []
        movement_stats: list[dict] = []
        for bbox in bbox_slice:
            labels, movement = _movement_labels(previous_bbox, bbox)
            action_labels.extend(labels)
            movement_stats.append(movement)
            previous_bbox = bbox
        normalized_labels = _normalize_segment_action_labels(action_labels, movement_stats)
        start_second = frame_slice[0] / fps
        end_second = (frame_slice[-1] + 1) / fps
        segments.append(
            {
                "start_frame": int(frame_slice[0]),
                "end_frame": int(frame_slice[-1]),
                "start_second": round(start_second, 3),
                "end_second": round(end_second, 3),
                "actual_start": _isoformat(recorded_start + timedelta(seconds=start_second)),
                "actual_end": _isoformat(recorded_start + timedelta(seconds=end_second)),
                "bbox": [int(value) for value in anchor_bbox],
                "action_labels": normalized_labels,
                "action_summary": _summarize_action_labels(normalized_labels),
                "movement_stats": movement_stats[-1] if movement_stats else {},
            }
        )
    return segments


def analyze_hospital_video(
    video_path: Path,
    camera_id: str,
    recorded_start: datetime,
    probe: dict,
    split: str = "live",
    detection_fps: float = DEFAULT_DETECTION_FPS,
    min_track_frames: int = DEFAULT_MIN_TRACK_FRAMES,
    min_person_area: int = DEFAULT_MIN_PERSON_AREA,
    track_iou_threshold: float = DEFAULT_TRACK_IOU,
    timeline_stride_seconds: float = 1.0,
    max_timeline_segments: int = 16,
) -> list[Moment]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for hospital video analysis.") from exc

    fps = float(probe.get("fps") or 25.0)
    stride = _sample_stride(fps, detection_fps)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to decode encoded hospital video: {video_path}")

    next_track_id = 1
    active_tracks: dict[str, dict] = {}
    tracks: dict[str, dict] = {}
    frame_idx = -1
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % stride != 0:
                continue

            detections = [
                bbox
                for bbox in _detect_people_hog(frame)
                if _bbox_area(bbox) >= min_person_area
            ]
            matched_track_ids: set[str] = set()

            for bbox in sorted(detections, key=_bbox_area, reverse=True):
                best_track_id = None
                best_iou = 0.0
                for track_id, track in active_tracks.items():
                    if track_id in matched_track_ids:
                        continue
                    iou_value = _bbox_iou(bbox, track["last_bbox"])
                    if iou_value > best_iou:
                        best_iou = iou_value
                        best_track_id = track_id

                if best_track_id and best_iou >= track_iou_threshold:
                    track = active_tracks[best_track_id]
                else:
                    best_track_id = str(next_track_id)
                    next_track_id += 1
                    track = {
                        "track_id": best_track_id,
                        "frames": [],
                        "bboxes": [],
                        "largest_bbox": bbox,
                        "largest_area": _bbox_area(bbox),
                    }
                    active_tracks[best_track_id] = track
                    tracks[best_track_id] = track

                matched_track_ids.add(best_track_id)
                track["frames"].append(frame_idx)
                track["bboxes"].append(bbox)
                track["last_bbox"] = bbox
                if _bbox_area(bbox) >= int(track.get("largest_area", 0)):
                    track["largest_bbox"] = bbox
                    track["largest_area"] = _bbox_area(bbox)

            active_tracks = {track_id: value for track_id, value in active_tracks.items() if track_id in matched_track_ids}
    finally:
        capture.release()

    recorded_start = recorded_start.astimezone(timezone.utc)
    location = camera_id
    video_stem = Path(video_path).stem
    moments: list[Moment] = []

    for raw_track_id, track in sorted(tracks.items(), key=lambda item: int(item[0])):
        frames = track["frames"]
        if len(frames) < min_track_frames:
            continue
        start_frame = min(frames)
        end_frame = max(frames)
        start_second = round(start_frame / fps, 3)
        end_second = round((end_frame + 1) / fps, 3)
        actual_start = recorded_start + timedelta(seconds=start_second)
        actual_end = recorded_start + timedelta(seconds=end_second)
        content_frames = []
        for frame_number, bbox in zip(frames[:10], track["bboxes"][:10]):
            content_frames.append(
                {
                    "frame_idx": int(frame_number),
                    "second": round(frame_number / fps, 3),
                    "actual_time": _isoformat(recorded_start + timedelta(seconds=(frame_number / fps))),
                    "bbox": [int(value) for value in bbox],
                }
            )
        timeline = _segment_track_timeline(
            frames=frames,
            bboxes=track["bboxes"],
            fps=fps,
            recorded_start=recorded_start,
            stride_seconds=timeline_stride_seconds,
            max_segments=max_timeline_segments,
        )

        moment_id = f"{video_stem}:track:{raw_track_id}"
        metadata = {
            "schema": "hospital_person_track",
            "camera_id": camera_id,
            "content_frames": content_frames,
            "timeline": timeline,
            "recorded_start_time": _isoformat(recorded_start),
            "recorded_end_time": _isoformat(actual_end if actual_end > recorded_start else recorded_start),
            "person_start_time": _isoformat(actual_start),
            "person_end_time": _isoformat(actual_end),
            "segmentation": {
                "available": False,
                "masks": [],
            },
            "embedding_inputs": {
                "dense_text": f"person in hospital camera {camera_id}",
                "visual": "crop_and_frame",
            },
        }
        timeline_text = " ".join(segment["action_summary"] for segment in timeline if segment.get("action_summary"))
        moments.append(
            Moment(
                id=moment_id,
                location=location,
                split=split,
                video_path=str(video_path),
                label_path=str(video_path.with_suffix(".json")),
                track_id=str(raw_track_id),
                fps=fps,
                start_frame=start_frame,
                end_frame=end_frame,
                start_second=start_second,
                end_second=end_second,
                captions=["person"],
                representative_bbox=[int(value) for value in track["largest_bbox"]],
                sample_frames=[int(frame_number) for frame_number in frames[:8]],
                text=(
                    f"person in hospital surveillance video {video_stem}. "
                    f"camera {camera_id}. "
                    f"tracked person {raw_track_id}. "
                    f"Timeline: {timeline_text or 'person visible in frame'}."
                ),
                keywords=sorted({"person", "hospital", "surveillance", _slug(camera_id), f"track_{raw_track_id}"}),
                camera_id=camera_id,
                metadata=metadata,
            )
        )

    return moments


def _merge_inferred_enrichment(moments: list[Moment], attribute_records: list[dict], scene_records: list[dict]) -> list[Moment]:
    by_id = {moment.id: moment for moment in moments}

    for record in attribute_records:
        moment_id = str(record.get("moment_id") or "").strip()
        if not moment_id or moment_id not in by_id:
            continue
        moment = by_id[moment_id]
        attributes = record.get("attributes")
        if isinstance(attributes, dict):
            moment.attribute_evidence = attributes

    for record in scene_records:
        moment_id = str(record.get("moment_id") or "").strip()
        if not moment_id or moment_id not in by_id:
            continue
        moment = by_id[moment_id]
        scene = record.get("scene")
        if isinstance(scene, dict):
            moment.scene_evidence = scene

    for moment in moments:
        phrases = summarize_attribute_evidence(moment.attribute_evidence) + summarize_scene_evidence(moment.scene_evidence)
        if not phrases:
            continue
        moment.text = f"{moment.text} Evidence: {'. '.join(phrases)}."
        keywords = set(moment.keywords)
        for phrase in phrases:
            keywords.update(token for token in sanitize_name(phrase).replace("_", " ").split() if token)
        moment.keywords = sorted(keywords)
        moment.metadata.setdefault("description", ", ".join(phrases))
        moment.metadata.setdefault("embedding_inputs", {})
        moment.metadata["embedding_inputs"]["dense_text"] = moment.text

    return moments


@lru_cache(maxsize=2)
def _load_blip_captioning_components(model_id: str):
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(model_id, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(model_id, local_files_only=True).to(device)
    model.eval()
    return processor, model, device


def enrich_moments_with_blip_captions(
    moments: list[Moment],
    model_id: str = "Salesforce/blip-image-captioning-large",
    prompt: str = "A single hospital person is",
) -> list[Moment]:
    items: list[tuple[Moment, Path]] = []
    for moment in moments:
        candidate_path = None
        if moment.crop_paths:
            candidate_path = Path(moment.crop_paths[0])
        elif moment.frame_paths:
            candidate_path = Path(moment.frame_paths[0])
        if candidate_path and candidate_path.exists():
            items.append((moment, candidate_path))

    if not items:
        return moments

    try:
        import torch
        from PIL import Image
    except ImportError:
        return moments

    try:
        processor, model, device = _load_blip_captioning_components(model_id)
    except Exception:
        return moments

    images = [Image.open(path).convert("RGB") for _, path in items]
    prompts = [prompt] * len(images)
    inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=48)
    captions = processor.batch_decode(outputs, skip_special_tokens=True)

    for index, (moment, _) in enumerate(items):
        caption = " ".join(str(captions[index]).split()).strip()
        if not caption:
            continue
        moment.metadata["person_caption"] = caption
        moment.metadata.setdefault("keyframe_captions", [])
        moment.metadata["keyframe_captions"].append(
            {
                "frame_idx": int(moment.sample_frames[0]) if moment.sample_frames else int(moment.start_frame),
                "caption": caption,
            }
        )
        moment.captions = [caption, *[value for value in moment.captions if value != caption]]
        moment.text = f"{moment.text} Detailed person caption: {caption}."
        moment.metadata.setdefault("embedding_inputs", {})
        moment.metadata["embedding_inputs"]["dense_text"] = moment.text
        keywords = set(moment.keywords)
        keywords.update(token for token in sanitize_name(caption).replace("_", " ").split() if token)
        moment.keywords = sorted(keywords)

    return moments


def _metadata_payload_for_video(
    *,
    video_id: str,
    camera_id: str,
    source_path: Path,
    encoded_path: Path,
    metadata_path: Path,
    recorded_start: datetime,
    recorded_end: datetime,
    probe: dict,
    encoder: str,
    moments: list[Moment],
) -> dict:
    return {
        "schema_version": METADATA_VERSION,
        "video": {
            "video_id": video_id,
            "camera_id": camera_id,
            "source_path": str(source_path),
            "encoded_path": str(encoded_path),
            "metadata_path": str(metadata_path),
            "recorded_start": _isoformat(recorded_start),
            "recorded_end": _isoformat(recorded_end),
            "codec": "h265",
            "encoder": encoder,
            "fps": float(probe.get("fps") or 0.0),
            "frame_count": int(probe.get("frame_count") or 0),
            "duration_seconds": round(float(probe.get("duration_seconds") or 0.0), 3),
            "width": int(probe.get("width") or 0),
            "height": int(probe.get("height") or 0),
            "file_size_bytes": int(encoded_path.stat().st_size) if encoded_path.exists() else 0,
        },
        "people": [moment.to_dict() for moment in moments],
    }


def _write_video_metadata(metadata_path: Path, payload: dict) -> Path:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return metadata_path


def persist_hospital_video(
    *,
    source_path: Path,
    encoded_path: Path,
    metadata_path: Path,
    catalog_path: Path,
    camera_id: str,
    video_id: str,
    recorded_start: datetime,
    recorded_end: datetime,
    probe: dict,
    encoder: str,
    moments: list[Moment],
    assets_dir: Path,
    queue_size: int,
) -> dict:
    metadata_payload = _metadata_payload_for_video(
        video_id=video_id,
        camera_id=camera_id,
        source_path=source_path,
        encoded_path=encoded_path,
        metadata_path=metadata_path,
        recorded_start=recorded_start,
        recorded_end=recorded_end,
        probe=probe,
        encoder=encoder,
        moments=moments,
    )
    _write_video_metadata(metadata_path, metadata_payload)
    _upsert_video_record(
        catalog_path=catalog_path,
        video_payload=metadata_payload["video"],
        moments=moments,
    )
    evicted_video_ids = evict_hospital_queue(
        dataset_root=catalog_path.parent,
        assets_dir=assets_dir,
        queue_size=queue_size,
    )
    return {
        "video_id": video_id,
        "camera_id": camera_id,
        "source_path": str(source_path),
        "encoded_path": str(encoded_path),
        "metadata_path": str(metadata_path),
        "recorded_start": _isoformat(recorded_start),
        "recorded_end": _isoformat(recorded_end),
        "person_count": len(moments),
        "evicted_video_ids": evicted_video_ids,
        "queue_size": queue_size,
        "encoder": encoder,
    }


def _upsert_video_record(
    *,
    catalog_path: Path,
    video_payload: dict,
    moments: list[Moment],
) -> None:
    with _connect_catalog(catalog_path) as connection:
        connection.execute("DELETE FROM persons WHERE video_id = ?", (video_payload["video_id"],))
        connection.execute(
            """
            INSERT OR REPLACE INTO videos (
                video_id, camera_id, source_path, encoded_path, metadata_path, recorded_start,
                recorded_end, ingested_at, codec, width, height, fps, duration_seconds, file_size_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_payload["video_id"],
                video_payload["camera_id"],
                video_payload["source_path"],
                video_payload["encoded_path"],
                video_payload["metadata_path"],
                video_payload["recorded_start"],
                video_payload["recorded_end"],
                _isoformat(datetime.now(timezone.utc)),
                "h265",
                int(video_payload.get("width") or 0),
                int(video_payload.get("height") or 0),
                float(video_payload.get("fps") or 0.0),
                float(video_payload.get("duration_seconds") or 0.0),
                int(video_payload.get("file_size_bytes") or 0),
            ),
        )
        connection.executemany(
            """
            INSERT OR REPLACE INTO persons (
                moment_id, video_id, track_id, camera_id, start_second, end_second, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    moment.id,
                    video_payload["video_id"],
                    moment.track_id,
                    moment.camera_id or video_payload["camera_id"],
                    float(moment.start_second),
                    float(moment.end_second),
                    json.dumps(moment.to_dict(), ensure_ascii=False),
                )
                for moment in moments
            ],
        )
        connection.commit()


def _remove_moment_assets(assets_dir: Path, moment_id: str) -> None:
    shutil.rmtree(Path(assets_dir) / sanitize_name(moment_id), ignore_errors=True)


def evict_hospital_queue(dataset_root: Path, assets_dir: Path, queue_size: int) -> list[str]:
    catalog_path = hospital_catalog_path(dataset_root)
    evicted_video_ids: list[str] = []
    with _connect_catalog(catalog_path) as connection:
        rows = connection.execute(
            """
            SELECT video_id, encoded_path, metadata_path
            FROM videos
            ORDER BY recorded_start ASC, ingested_at ASC
            """
        ).fetchall()
        if len(rows) <= queue_size:
            return []

        excess = len(rows) - queue_size
        for row in rows[:excess]:
            video_id = str(row["video_id"])
            person_rows = connection.execute(
                "SELECT moment_id FROM persons WHERE video_id = ?",
                (video_id,),
            ).fetchall()
            for person_row in person_rows:
                _remove_moment_assets(assets_dir, str(person_row["moment_id"]))
            encoded_path = Path(str(row["encoded_path"]))
            metadata_path = Path(str(row["metadata_path"]))
            encoded_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            connection.execute("DELETE FROM persons WHERE video_id = ?", (video_id,))
            connection.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
            evicted_video_ids.append(video_id)
        connection.commit()
    return evicted_video_ids


def _default_nvidia_hospital_video_paths(nvidia_root: Path) -> list[Path]:
    videos_dir = Path(nvidia_root) / "videos"
    return sorted(path for path in videos_dir.glob("Camera_*.mp4") if path.stem != "Camera")


def _load_nvidia_ground_truth(ground_truth_path: Path) -> dict:
    with ground_truth_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected NVIDIA ground truth schema: {ground_truth_path}")
    return payload


def build_moments_from_nvidia_ground_truth(
    *,
    camera_id: str,
    video_path: Path,
    ground_truth: dict,
    recorded_start: datetime,
    fps: float,
    timeline_stride_seconds: float,
    max_timeline_segments: int,
) -> list[Moment]:
    grouped: dict[str, dict] = {}
    for raw_frame_idx, entries in ground_truth.items():
        try:
            frame_idx = int(raw_frame_idx)
        except Exception:
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            visible = entry.get("2d bounding box visible") or {}
            bbox = visible.get(camera_id)
            if not bbox:
                continue
            object_id = str(entry.get("object id"))
            record = grouped.setdefault(
                object_id,
                {
                    "frames": [],
                    "bboxes": [],
                    "world_locations": [],
                    "box_scales": [],
                    "box_rotations": [],
                },
            )
            record["frames"].append(frame_idx)
            record["bboxes"].append([int(value) for value in bbox])
            record["world_locations"].append(entry.get("3d location") or [])
            record["box_scales"].append(entry.get("3d bounding box scale") or [])
            record["box_rotations"].append(entry.get("3d bounding box rotation") or [])

    video_stem = video_path.stem
    moments: list[Moment] = []
    for object_id, payload in sorted(grouped.items(), key=lambda item: int(item[0])):
        frames = payload["frames"]
        bboxes = payload["bboxes"]
        if not frames:
            continue
        start_frame = min(frames)
        end_frame = max(frames)
        start_second = round(start_frame / fps, 3)
        end_second = round((end_frame + 1) / fps, 3)
        actual_start = recorded_start + timedelta(seconds=start_second)
        actual_end = recorded_start + timedelta(seconds=end_second)
        largest_bbox = max(bboxes, key=_bbox_area)
        timeline = _segment_track_timeline(
            frames=frames,
            bboxes=bboxes,
            fps=fps,
            recorded_start=recorded_start,
            stride_seconds=timeline_stride_seconds,
            max_segments=max_timeline_segments,
        )
        content_frames = []
        for index, frame_idx in enumerate(frames[: max_timeline_segments]):
            content_frames.append(
                {
                    "frame_idx": int(frame_idx),
                    "second": round(frame_idx / fps, 3),
                    "actual_time": _isoformat(recorded_start + timedelta(seconds=(frame_idx / fps))),
                    "bbox": [int(value) for value in bboxes[index]],
                    "world_location": payload["world_locations"][index],
                }
            )
        world_path = [value for value in payload["world_locations"][: max_timeline_segments] if value]
        metadata = {
            "schema": "nvidia_hospital_ground_truth_person_track",
            "camera_id": camera_id,
            "content_frames": content_frames,
            "timeline": timeline,
            "recorded_start_time": _isoformat(recorded_start),
            "recorded_end_time": _isoformat(recorded_start + timedelta(seconds=end_second)),
            "person_start_time": _isoformat(actual_start),
            "person_end_time": _isoformat(actual_end),
            "ground_truth": {
                "object_id": int(object_id),
                "world_locations": world_path,
                "box_scales": payload["box_scales"][: max_timeline_segments],
                "box_rotations": payload["box_rotations"][: max_timeline_segments],
            },
            "segmentation": {
                "available": False,
                "masks": [],
            },
            "embedding_inputs": {
                "dense_text": f"single hospital person in {camera_id}",
                "visual": "crop_and_frame",
            },
        }
        timeline_text = " ".join(segment["action_summary"] for segment in timeline if segment.get("action_summary"))
        moments.append(
            Moment(
                id=f"{video_stem}:track:{object_id}",
                location=camera_id,
                split="nvidia_gt",
                video_path=str(video_path),
                label_path=str(video_path.with_suffix(".json")),
                track_id=str(object_id),
                fps=fps,
                start_frame=start_frame,
                end_frame=end_frame,
                start_second=start_second,
                end_second=end_second,
                captions=["person"],
                representative_bbox=[int(value) for value in largest_bbox],
                sample_frames=[int(frame_number) for frame_number in frames[: max_timeline_segments]],
                text=(
                    f"single hospital person in camera {camera_id}. "
                    f"tracked person {object_id}. "
                    f"Timeline: {timeline_text or 'person visible in frame'}."
                ),
                keywords=sorted({"person", "hospital", "surveillance", _slug(camera_id), f"track_{object_id}"}),
                camera_id=camera_id,
                metadata=metadata,
            )
        )
    return moments


def bootstrap_nvidia_hospital_dataset(
    *,
    nvidia_root: Path,
    config: HospitalIngestConfig,
    recorded_start: str | datetime | None = None,
    limit: int | None = 31,
) -> dict:
    layout = ensure_hospital_layout(config.dataset_root)
    ground_truth_path = Path(nvidia_root) / "ground_truth.json"
    ground_truth = _load_nvidia_ground_truth(ground_truth_path)
    video_paths = _default_nvidia_hospital_video_paths(nvidia_root)
    if limit is not None:
        video_paths = video_paths[:limit]

    results = []
    base_start = parse_recorded_time(recorded_start, datetime.now(timezone.utc))
    for index, source_path in enumerate(video_paths):
        camera_id = source_path.stem
        source_probe = probe_video(source_path)
        recorded_start_dt = base_start + timedelta(seconds=index * 10)
        video_id = _build_video_id(camera_id, recorded_start_dt)
        encoded_path = layout["encoded_dir"] / f"{video_id}.h265"
        metadata_path = layout["metadata_dir"] / f"{video_id}.json"
        conversion = convert_video_to_h265(
            source_path=source_path,
            output_path=encoded_path,
            ffmpeg_bin=config.ffmpeg_bin,
            preferred_encoder=config.preferred_encoder,
        )
        encoded_probe = probe_video(encoded_path)
        moments = build_moments_from_nvidia_ground_truth(
            camera_id=camera_id,
            video_path=encoded_path,
            ground_truth=ground_truth,
            recorded_start=recorded_start_dt,
            fps=float(encoded_probe.get("fps") or source_probe.get("fps") or 25.0),
            timeline_stride_seconds=config.timeline_stride_seconds,
            max_timeline_segments=config.max_timeline_segments,
        )
        if moments:
            extract_visual_assets(
                moments=moments,
                output_dir=config.assets_dir,
                max_assets_per_moment=max(config.max_assets_per_moment, min(config.max_timeline_segments, 12)),
                crop_padding=config.crop_padding,
                overwrite=True,
                ffmpeg_bin=config.ffmpeg_bin,
            )
            moments = enrich_moments_with_blip_captions(moments)
        if config.enable_enrichment and moments:
            attribute_records, scene_records = infer_enrichment_payloads(
                moments=moments,
                clip_model_name=config.clip_model_name,
                clip_pretrained=config.clip_pretrained,
                device=config.device,
                batch_size=config.batch_size,
            )
            moments = _merge_inferred_enrichment(moments, attribute_records, scene_records)
        result = persist_hospital_video(
            source_path=source_path,
            encoded_path=encoded_path,
            metadata_path=metadata_path,
            catalog_path=layout["catalog_path"],
            camera_id=camera_id,
            video_id=video_id,
            recorded_start=recorded_start_dt,
            recorded_end=recorded_start_dt + timedelta(seconds=float(encoded_probe.get("duration_seconds") or 0.0)),
            probe=encoded_probe,
            encoder=conversion["encoder"],
            moments=moments,
            assets_dir=config.assets_dir,
            queue_size=config.queue_size,
        )
        results.append(result)

    return {
        "processed_videos": len(results),
        "results": results,
    }


def load_hospital_moments(dataset_root: Path) -> list[Moment]:
    metadata_dir = hospital_metadata_dir(dataset_root)
    if not metadata_dir.exists():
        return []

    moments: list[Moment] = []
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for raw_moment in payload.get("people", []):
            if not isinstance(raw_moment, dict):
                continue
            raw_moment.setdefault("label_path", str(metadata_path))
            moments.append(Moment.from_dict(raw_moment))
    return sorted(moments, key=lambda item: (item.camera_id or item.location, item.start_second, item.id))


def ingest_hospital_video(
    source_path: Path,
    config: HospitalIngestConfig,
    camera_id: str | None = None,
    recorded_start: str | datetime | None = None,
) -> dict:
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Hospital source video does not exist: {source_path}")

    layout = ensure_hospital_layout(config.dataset_root)
    source_probe = probe_video(source_path)
    fallback_start = datetime.fromtimestamp(source_path.stat().st_mtime, tz=timezone.utc)
    resolved_start = parse_recorded_time(recorded_start, fallback_start)
    resolved_camera_id = _slug(camera_id or _default_camera_id(source_path))
    video_id = _build_video_id(resolved_camera_id, resolved_start)
    encoded_path = layout["encoded_dir"] / f"{video_id}.h265"
    metadata_path = layout["metadata_dir"] / f"{video_id}.json"

    conversion = convert_video_to_h265(
        source_path=source_path,
        output_path=encoded_path,
        ffmpeg_bin=config.ffmpeg_bin,
        preferred_encoder=config.preferred_encoder,
    )
    encoded_probe = probe_video(encoded_path)
    recorded_end = resolved_start + timedelta(seconds=float(encoded_probe.get("duration_seconds") or 0.0))

    moments = analyze_hospital_video(
        video_path=encoded_path,
        camera_id=resolved_camera_id,
        recorded_start=resolved_start,
        probe=encoded_probe,
        detection_fps=config.detection_fps,
        min_track_frames=config.min_track_frames,
        min_person_area=config.min_person_area,
        track_iou_threshold=config.track_iou_threshold,
        timeline_stride_seconds=config.timeline_stride_seconds,
        max_timeline_segments=config.max_timeline_segments,
    )

    if moments:
        extract_visual_assets(
            moments=moments,
            output_dir=config.assets_dir,
            max_assets_per_moment=config.max_assets_per_moment,
            crop_padding=config.crop_padding,
            overwrite=True,
            ffmpeg_bin=config.ffmpeg_bin,
        )
        moments = enrich_moments_with_blip_captions(moments)

    if config.enable_enrichment and moments:
        attribute_records, scene_records = infer_enrichment_payloads(
            moments=moments,
            clip_model_name=config.clip_model_name,
            clip_pretrained=config.clip_pretrained,
            device=config.device,
            batch_size=config.batch_size,
        )
        moments = _merge_inferred_enrichment(moments, attribute_records, scene_records)

    return persist_hospital_video(
        source_path=source_path,
        encoded_path=encoded_path,
        metadata_path=metadata_path,
        catalog_path=layout["catalog_path"],
        camera_id=resolved_camera_id,
        video_id=video_id,
        recorded_start=resolved_start,
        recorded_end=recorded_end,
        probe=encoded_probe,
        encoder=conversion["encoder"],
        moments=moments,
        assets_dir=config.assets_dir,
        queue_size=config.queue_size,
    )
