from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .enrichment import summarize_attribute_evidence, summarize_scene_evidence
from .models import Moment
from .video_tools import select_crop_path, select_frame_path
from .vision import encode_images_clip, encode_texts_clip

COLOR_LABELS = (
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "white",
    "black",
    "gray",
    "brown",
    "pink",
    "purple",
)
TYPE_LABELS = ("shirt", "jacket", "hoodie", "coat", "sweater")
ACCESSORY_PROMPTS = {
    "backpack": (
        "a surveillance crop of a person wearing a backpack",
        "a surveillance crop of a person without a backpack",
    ),
    "hat": (
        "a surveillance crop of a person wearing a hat",
        "a surveillance crop of a person without a hat",
    ),
    "bag": (
        "a surveillance crop of a person carrying a bag",
        "a surveillance crop of a person without a bag",
    ),
}
SCENE_PROMPTS = {
    "near_road": (
        "a surveillance frame showing a person next to a road",
        "a surveillance frame showing a person far from any road",
    ),
}


def _record_prefix(moment: Moment) -> dict:
    return {
        "moment_id": moment.id,
        "video_name": Path(moment.video_path).name,
        "track_id": moment.track_id,
        "location": moment.location,
        "split": moment.split,
    }


def _load_rgb(path: str | Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _dominant_color_from_crop(image_path: str | Path) -> str | None:
    image = _load_rgb(image_path)
    width, height = image.size
    if width <= 0 or height <= 0:
        return None

    left = int(width * 0.2)
    right = max(int(width * 0.8), left + 1)
    top = int(height * 0.05)
    bottom = max(int(height * 0.45), top + 1)
    crop = image.crop((left, top, right, bottom))
    array = np.asarray(crop, dtype="float32")
    if array.size == 0:
        return None

    mean_rgb = array.mean(axis=(0, 1))
    r, g, b = [float(value) / 255.0 for value in mean_rgb]
    vmax = max(r, g, b)
    vmin = min(r, g, b)
    delta = vmax - vmin
    brightness = vmax

    if brightness < 0.16:
        return "black"
    if delta < 0.08:
        if brightness > 0.82:
            return "white"
        if brightness > 0.45:
            return "gray"
        return "black"

    if vmax == r:
        hue = ((g - b) / delta) % 6
    elif vmax == g:
        hue = ((b - r) / delta) + 2
    else:
        hue = ((r - g) / delta) + 4
    hue *= 60.0

    if 0 <= hue < 15 or 345 <= hue <= 360:
        return "red"
    if 15 <= hue < 40:
        return "orange"
    if 40 <= hue < 70:
        return "yellow"
    if 70 <= hue < 165:
        return "green"
    if 165 <= hue < 255:
        return "blue"
    if 255 <= hue < 290:
        return "purple"
    if 290 <= hue < 345:
        return "pink"
    return "brown" if brightness < 0.5 else "red"


def _multiclass_labels_with_scores(
    embeddings: np.ndarray,
    prompt_groups: dict[str, str],
    model_name: str,
    pretrained: str,
    device: str | None,
    batch_size: int,
) -> list[tuple[str, float]]:
    labels = list(prompt_groups.keys())
    prompts = list(prompt_groups.values())
    text_embeddings = encode_texts_clip(
        prompts,
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        batch_size=min(batch_size, max(len(prompts), 1)),
    )
    scores = embeddings @ text_embeddings.T
    results: list[tuple[str, float]] = []
    for row in scores:
        best_idx = int(np.argmax(row))
        results.append((labels[best_idx], float(row[best_idx])))
    return results


def _binary_labels_with_scores(
    embeddings: np.ndarray,
    prompt_pairs: dict[str, tuple[str, str]],
    model_name: str,
    pretrained: str,
    device: str | None,
    batch_size: int,
) -> dict[str, np.ndarray]:
    results: dict[str, np.ndarray] = {}
    for label, (positive_prompt, negative_prompt) in prompt_pairs.items():
        text_embeddings = encode_texts_clip(
            [positive_prompt, negative_prompt],
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            batch_size=2,
        )
        scores = embeddings @ text_embeddings.T
        results[label] = scores.astype("float32")
    return results


def infer_enrichment_payloads(
    moments: list[Moment],
    clip_model_name: str,
    clip_pretrained: str,
    device: str | None = None,
    batch_size: int = 16,
) -> tuple[list[dict], list[dict]]:
    crop_moments: list[Moment] = []
    crop_paths: list[str] = []
    frame_moments: list[Moment] = []
    frame_paths: list[str] = []

    for moment in moments:
        crop_path = select_crop_path(moment)
        if crop_path and Path(crop_path).exists():
            crop_moments.append(moment)
            crop_paths.append(crop_path)
        frame_path = select_frame_path(moment)
        if frame_path and Path(frame_path).exists():
            frame_moments.append(moment)
            frame_paths.append(frame_path)

    attribute_records: list[dict] = []
    scene_records: list[dict] = []

    if crop_paths:
        crop_embeddings = encode_images_clip(
            crop_paths,
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            device=device,
            batch_size=batch_size,
        )
        type_results = _multiclass_labels_with_scores(
            crop_embeddings,
            {
                label: f"a surveillance crop of a person wearing a {label}"
                for label in TYPE_LABELS
            },
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            device=device,
            batch_size=batch_size,
        )
        accessory_scores = _binary_labels_with_scores(
            crop_embeddings,
            ACCESSORY_PROMPTS,
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            device=device,
            batch_size=batch_size,
        )

        for index, moment in enumerate(crop_moments):
            crop_path = crop_paths[index]
            record = _record_prefix(moment)
            top_color = _dominant_color_from_crop(crop_path)
            top_type, top_type_score = type_results[index]
            attributes: dict[str, object] = {}
            if top_color in COLOR_LABELS:
                attributes["top_color"] = top_color
            if top_type_score > 0.18:
                attributes["top_type"] = top_type
                attributes["top_type_score"] = round(top_type_score, 4)

            for label, scores in accessory_scores.items():
                positive_score = float(scores[index][0])
                negative_score = float(scores[index][1])
                if positive_score > negative_score + 0.01:
                    attributes[label] = True
                    attributes[f"{label}_score"] = round(positive_score, 4)

            if attributes:
                record["attributes"] = attributes
                record["summary"] = summarize_attribute_evidence(attributes)
                attribute_records.append(record)

    if frame_paths:
        frame_embeddings = encode_images_clip(
            frame_paths,
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            device=device,
            batch_size=batch_size,
        )
        scene_scores = _binary_labels_with_scores(
            frame_embeddings,
            SCENE_PROMPTS,
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            device=device,
            batch_size=batch_size,
        )
        for index, moment in enumerate(frame_moments):
            record = _record_prefix(moment)
            scene: dict[str, object] = {}
            for label, scores in scene_scores.items():
                positive_score = float(scores[index][0])
                negative_score = float(scores[index][1])
                if positive_score > negative_score + 0.01:
                    scene[label] = True
                    scene[f"{label}_score"] = round(positive_score, 4)
            if scene:
                record["scene"] = scene
                record["summary"] = summarize_scene_evidence(scene)
                scene_records.append(record)

    return attribute_records, scene_records


def write_enrichment_payloads(
    dataset_root: Path,
    attribute_records: list[dict],
    scene_records: list[dict],
) -> dict[str, str | int]:
    enrichment_dir = dataset_root / "enrichment"
    enrichment_dir.mkdir(parents=True, exist_ok=True)

    attribute_path = enrichment_dir / "person_attributes.json"
    scene_path = enrichment_dir / "scene_context.json"
    with attribute_path.open("w", encoding="utf-8") as handle:
        json.dump(attribute_records, handle, ensure_ascii=False, indent=2)
    with scene_path.open("w", encoding="utf-8") as handle:
        json.dump(scene_records, handle, ensure_ascii=False, indent=2)

    return {
        "attribute_path": str(attribute_path),
        "scene_path": str(scene_path),
        "attribute_count": len(attribute_records),
        "scene_count": len(scene_records),
    }


def infer_and_write_enrichment(
    moments: list[Moment],
    dataset_root: Path,
    clip_model_name: str,
    clip_pretrained: str,
    device: str | None = None,
    batch_size: int = 16,
) -> dict[str, str | int]:
    attribute_records, scene_records = infer_enrichment_payloads(
        moments=moments,
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
        device=device,
        batch_size=batch_size,
    )
    return write_enrichment_payloads(
        dataset_root=dataset_root,
        attribute_records=attribute_records,
        scene_records=scene_records,
    )
