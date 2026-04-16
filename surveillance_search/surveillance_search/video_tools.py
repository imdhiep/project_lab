from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from .models import Moment


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


def select_frame_path(moment: Moment) -> str | None:
    if moment.frame_paths:
        return moment.frame_paths[0]
    return None


def select_crop_path(moment: Moment) -> str | None:
    if moment.crop_paths:
        return moment.crop_paths[0]
    return None


def select_visual_path(moment: Moment, preference: str = "frame") -> str | None:
    if preference == "frame":
        return select_frame_path(moment) or select_crop_path(moment)
    if preference == "crop":
        return select_crop_path(moment) or select_frame_path(moment)
    raise ValueError(f"Unsupported visual preference: {preference}")


def select_preview_frame_idx(moment: Moment) -> int:
    if moment.sample_frames:
        return int(moment.sample_frames[0])
    return int(moment.start_frame)


def select_preview_second(moment: Moment) -> float:
    fps = moment.fps if moment.fps > 0 else 30.0
    return round(select_preview_frame_idx(moment) / fps, 3)


def summarize_captions(moment: Moment, limit: int = 2) -> str:
    if not moment.captions:
        return f"Tracked object {moment.track_id} in {moment.location} ({moment.split})"

    selected = moment.captions[:limit]
    summary = " | ".join(selected)
    if len(moment.captions) > limit:
        summary = f"{summary} | +{len(moment.captions) - limit} more"
    return summary


def build_frame_info(moment: Moment) -> dict:
    preview_frame_idx = select_preview_frame_idx(moment)
    return {
        "preview_frame_idx": preview_frame_idx,
        "preview_second": select_preview_second(moment),
        "start_frame": int(moment.start_frame),
        "end_frame": int(moment.end_frame),
        "start_second": round(float(moment.start_second), 3),
        "end_second": round(float(moment.end_second), 3),
        "fps": float(moment.fps),
        "bbox": list(moment.representative_bbox),
        "sample_frames": [int(frame_idx) for frame_idx in moment.sample_frames],
        "track_id": moment.track_id,
        "camera_id": moment.camera_id or moment.location,
        "location": moment.location,
        "split": moment.split,
        "content_frames": list(moment.metadata.get("content_frames", [])) if isinstance(moment.metadata, dict) else [],
    }


def attach_existing_visual_assets(moments: list[Moment], assets_dir: Path) -> list[Moment]:
    assets_dir = Path(assets_dir)
    if not assets_dir.exists():
        return moments

    for moment in moments:
        moment_dir = assets_dir / sanitize_name(moment.id)
        if not moment_dir.exists():
            continue

        frame_paths = sorted(str(path) for path in moment_dir.glob("frame_*.jpg"))
        crop_paths = sorted(str(path) for path in moment_dir.glob("crop_*.jpg"))
        if frame_paths:
            moment.frame_paths = frame_paths
        if crop_paths:
            moment.crop_paths = crop_paths

    return moments


def count_visual_assets(moments: list[Moment]) -> dict[str, int]:
    frame_count = sum(len(moment.frame_paths) for moment in moments)
    crop_count = sum(len(moment.crop_paths) for moment in moments)
    moments_with_visuals = sum(1 for moment in moments if moment.frame_paths or moment.crop_paths)
    return {
        "moments_with_visuals": moments_with_visuals,
        "frame_count": frame_count,
        "crop_count": crop_count,
    }


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python-headless is required for frame extraction."
        ) from exc
    return cv2


def _require_pil_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for ffmpeg fallback image loading.") from exc
    return Image


def _clip_bbox(bbox: list[int], width: int, height: int, padding_ratio: float) -> list[int]:
    left, top, right, bottom = bbox
    box_width = max(right - left, 1)
    box_height = max(bottom - top, 1)
    pad_x = int(box_width * padding_ratio)
    pad_y = int(box_height * padding_ratio)
    return [
        max(left - pad_x, 0),
        max(top - pad_y, 0),
        min(right + pad_x, width),
        min(bottom + pad_y, height),
    ]


def _frame_time_for(moment: Moment, frame_idx: int) -> float:
    fps = moment.fps if moment.fps > 0 else 30.0
    return frame_idx / fps


def _extract_frame_with_ffmpeg(video_path: Path, second: float, ffmpeg_bin: str) -> object | None:
    Image = _require_pil_image()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        command = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{second:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(temp_path),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0 or not temp_path.exists() or temp_path.stat().st_size == 0:
            return None
        image = Image.open(temp_path).convert("RGB")
        return image.copy()
    finally:
        temp_path.unlink(missing_ok=True)


def _write_frame(frame, frame_path: Path) -> None:
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(frame, "save"):
        frame.save(frame_path, format="JPEG")
        return

    cv2 = _require_cv2()
    cv2.imwrite(str(frame_path), frame)


def _frame_size(frame) -> tuple[int, int]:
    if hasattr(frame, "size") and not isinstance(frame.size, int):
        width, height = frame.size
        return int(width), int(height)
    height, width = frame.shape[:2]
    return int(width), int(height)


def _crop_frame(frame, bbox: list[int]):
    if hasattr(frame, "crop"):
        return frame.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    return frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]


def _crop_has_content(crop) -> bool:
    if hasattr(crop, "size") and not isinstance(crop.size, int):
        return crop.size[0] > 0 and crop.size[1] > 0
    return bool(getattr(crop, "size", 0))


def extract_visual_assets(
    moments: list[Moment],
    output_dir: Path,
    max_assets_per_moment: int = 3,
    crop_padding: float = 0.08,
    overwrite: bool = False,
    ffmpeg_bin: str = "ffmpeg",
) -> list[Moment]:
    try:
        cv2 = _require_cv2()
    except RuntimeError:
        cv2 = None
    output_dir.mkdir(parents=True, exist_ok=True)

    for moment in moments:
        moment.frame_paths = []
        moment.crop_paths = []

    moments_by_video: dict[str, list[Moment]] = {}
    for moment in moments:
        moments_by_video.setdefault(moment.video_path, []).append(moment)

    for video_path, video_moments in moments_by_video.items():
        video_file = Path(video_path)
        if not video_file.exists():
            continue

        capture = None
        if cv2 is not None:
            capture = cv2.VideoCapture(str(video_file))
            if not capture.isOpened():
                capture.release()
                capture = None

        try:
            for moment in video_moments:
                chosen_frames = moment.sample_frames[:max_assets_per_moment] or [moment.start_frame]
                moment_dir = output_dir / sanitize_name(moment.id)
                moment_dir.mkdir(parents=True, exist_ok=True)

                for frame_idx in chosen_frames:
                    frame = None
                    if capture is not None:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ok, frame = capture.read()
                        if not ok:
                            frame = None
                    if frame is None:
                        frame = _extract_frame_with_ffmpeg(
                            video_file,
                            second=_frame_time_for(moment, frame_idx),
                            ffmpeg_bin=ffmpeg_bin,
                        )
                    if frame is None:
                        continue

                    frame_path = moment_dir / f"frame_{frame_idx:06d}.jpg"
                    if overwrite or not frame_path.exists():
                        _write_frame(frame, frame_path)
                    moment.frame_paths.append(str(frame_path))

                    bbox = moment.representative_bbox
                    if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        width, height = _frame_size(frame)
                        clip_box = _clip_bbox(bbox, width, height, crop_padding)
                        crop = _crop_frame(frame, clip_box)
                        if _crop_has_content(crop):
                            crop_path = moment_dir / f"crop_{frame_idx:06d}.jpg"
                            if overwrite or not crop_path.exists():
                                _write_frame(crop, crop_path)
                            moment.crop_paths.append(str(crop_path))
        finally:
            if capture is not None:
                capture.release()

    return moments


def extract_clip(
    video_path: Path,
    output_path: Path,
    start_second: float,
    duration_seconds: float,
    ffmpeg_bin: str = "ffmpeg",
) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-ss",
        f"{start_second:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-c",
        "copy",
        "-y",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path
