from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

from .vision import encode_images_clip, encode_texts_clip

TEMPORAL_TERMS = frozenset(
    {
        "before",
        "after",
        "while",
        "when",
        "during",
        "first",
        "last",
        "then",
        "just",
        "right",
        "immediately",
        "trước",
        "sau",
        "khi",
        "lúc",
        "ngay",
    }
)
SCENE_TERMS = frozenset({"road", "street", "curb", "sidewalk", "đường", "phố", "lề"})


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE) if token}


def build_query_profile(
    query_text: str | None,
    search_mode: str,
    structured_query: dict[str, object] | None = None,
    query_requirements: dict[str, object] | None = None,
) -> dict[str, object]:
    tokens = _tokenize(query_text or "")
    attribute_terms = set((query_requirements or {}).get("attribute_terms") or [])
    scene_terms = set((query_requirements or {}).get("scene_terms") or [])
    asks_attributes = bool(attribute_terms)
    asks_scene = bool(scene_terms)
    asks_temporal = bool(tokens & TEMPORAL_TERMS)

    if structured_query:
        asks_attributes = asks_attributes or bool(
            structured_query.get("top_color")
            or structured_query.get("top_type")
            or structured_query.get("accessories")
        )
        asks_scene = asks_scene or bool(structured_query.get("near_road"))

    if asks_scene and not asks_attributes:
        preferred_visual_focus = "frame"
        primary_intent = "scene_relation"
    elif asks_attributes and not asks_scene:
        preferred_visual_focus = "crop"
        primary_intent = "appearance_match"
    elif asks_attributes and asks_scene:
        preferred_visual_focus = "hybrid"
        primary_intent = "appearance_plus_scene"
    elif asks_temporal:
        preferred_visual_focus = "frame"
        primary_intent = "temporal_person_search"
    else:
        preferred_visual_focus = "crop" if search_mode == "person" else "frame"
        primary_intent = "generic_person_search" if search_mode == "person" else "generic_search"

    return {
        "primary_intent": primary_intent,
        "asks_attributes": asks_attributes,
        "asks_scene": asks_scene,
        "asks_temporal": asks_temporal,
        "preferred_visual_focus": preferred_visual_focus,
        "scene_terms": sorted(str(term) for term in scene_terms),
        "attribute_terms": sorted(str(term) for term in attribute_terms),
    }


def _asset_frame_idx(path: str | None) -> int | None:
    if not path:
        return None
    match = re.search(r"_(\d+)\.jpg$", Path(path).name)
    if not match:
        return None
    return int(match.group(1))


def _sibling_frame_path(result: dict, frame_idx: int | None, asset_type: str) -> str | None:
    if asset_type == "frame":
        return None
    if frame_idx is None:
        return None
    target_name = f"frame_{frame_idx:06d}.jpg"
    for path in result.get("frame_paths") or []:
        if Path(path).name == target_name:
            return path
    return None


def _candidate_assets(result: dict) -> list[dict[str, object]]:
    assets: list[dict[str, object]] = []
    for path in result.get("frame_paths") or []:
        if Path(path).exists():
            assets.append(
                {
                    "path": path,
                    "asset_type": "frame",
                    "frame_idx": _asset_frame_idx(path),
                }
            )
    for path in result.get("crop_paths") or []:
        if Path(path).exists():
            assets.append(
                {
                    "path": path,
                    "asset_type": "crop",
                    "frame_idx": _asset_frame_idx(path),
                }
            )
    return assets


@lru_cache(maxsize=4096)
def _asset_quality(path: str) -> tuple[float, float, float]:
    import cv2
    import numpy as np

    image = cv2.imread(path)
    if image is None:
        return 0.0, 0.0, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness = min(sharpness_raw / 300.0, 1.0)
    brightness = float(np.mean(gray) / 255.0)
    exposure = max(0.0, 1.0 - (abs(brightness - 0.52) / 0.52))
    quality = (0.68 * sharpness) + (0.32 * exposure)
    return quality, sharpness, brightness


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high, abs_tol=1e-9):
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _answer_window(result: dict, frame_idx: int | None, query_profile: dict[str, object]) -> dict[str, float]:
    fps = float((result.get("frame_info") or {}).get("fps") or 30.0)
    start_second = float((result.get("frame_info") or {}).get("start_second") or result.get("start_second") or 0.0)
    end_second = float((result.get("frame_info") or {}).get("end_second") or result.get("end_second") or start_second)
    center_second = (frame_idx / fps) if frame_idx is not None and fps > 0 else float(
        (result.get("frame_info") or {}).get("preview_second") or start_second
    )
    center_second = min(max(center_second, start_second), end_second)
    track_duration = max(end_second - start_second, 0.0)
    base_duration = 3.5 if query_profile.get("asks_temporal") else 2.4
    duration = min(max(base_duration, min(track_duration, 1.5)), max(track_duration, base_duration))
    half = duration / 2.0
    window_start = max(start_second, center_second - half)
    window_end = min(end_second, center_second + half)
    if window_end <= window_start:
        window_end = min(end_second, window_start + max(track_duration, 1.0))
    return {
        "center_second": round(center_second, 3),
        "start_second": round(window_start, 3),
        "end_second": round(window_end, 3),
        "duration_seconds": round(max(window_end - window_start, 0.0), 3),
    }


def annotate_results_with_answer_selection(
    results: list[dict],
    query_text: str | None,
    search_mode: str,
    structured_query: dict[str, object] | None,
    clip_model_name: str,
    clip_pretrained: str,
    device: str | None,
    batch_size: int,
) -> list[dict]:
    if not results:
        return results

    query_requirements = results[0].get("query_requirements") or {}
    query_profile = build_query_profile(
        query_text=query_text,
        search_mode=search_mode,
        structured_query=structured_query,
        query_requirements=query_requirements,
    )

    all_assets: list[dict[str, object]] = []
    per_result_assets: list[list[dict[str, object]]] = []
    for result in results:
        assets = _candidate_assets(result)
        per_result_assets.append(assets)
        for asset in assets:
            asset["result_ref"] = result
            all_assets.append(asset)

    relevance_by_path: dict[str, float] = {}
    if query_text and all_assets:
        image_paths = [str(asset["path"]) for asset in all_assets]
        image_embeddings = encode_images_clip(
            image_paths,
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            device=device,
            batch_size=max(1, min(batch_size, len(image_paths))),
        )
        query_embedding = encode_texts_clip(
            [query_text],
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            device=device,
            batch_size=1,
        )[0]
        raw_scores = (image_embeddings @ query_embedding).astype("float32").tolist()
        norm_scores = _normalize([float(value) for value in raw_scores])
        for asset, score in zip(all_assets, norm_scores):
            relevance_by_path[str(asset["path"])] = float(score)

    focus = query_profile["preferred_visual_focus"]
    for result, assets in zip(results, per_result_assets):
        result["query_profile"] = query_profile
        if not assets:
            continue

        best_asset = None
        best_score = -1.0
        for asset in assets:
            path = str(asset["path"])
            relevance = relevance_by_path.get(path, 0.0)
            quality, sharpness, brightness = _asset_quality(path)
            if focus == "crop":
                focus_bias = 0.08 if asset["asset_type"] == "crop" else -0.02
            elif focus == "frame":
                focus_bias = 0.08 if asset["asset_type"] == "frame" else -0.02
            else:
                focus_bias = 0.03 if asset["asset_type"] == "frame" else 0.03

            if relevance_by_path:
                answer_score = (0.72 * relevance) + (0.28 * quality) + focus_bias
            else:
                answer_score = quality + focus_bias

            candidate = {
                **asset,
                "relevance_score": round(relevance, 6),
                "quality_score": round(quality, 6),
                "sharpness_score": round(sharpness, 6),
                "brightness_score": round(brightness, 6),
                "answer_frame_score": round(answer_score, 6),
            }
            if best_asset is None or answer_score > best_score:
                best_asset = candidate
                best_score = answer_score

        if best_asset is None:
            continue

        frame_idx = best_asset.get("frame_idx")
        result["answer_visual_path"] = str(best_asset["path"])
        result["answer_asset_type"] = str(best_asset["asset_type"])
        result["answer_frame_idx"] = frame_idx
        if frame_idx is not None:
            fps = float((result.get("frame_info") or {}).get("fps") or 30.0)
            result["answer_second"] = round(frame_idx / fps, 3)
        result["answer_relevance_score"] = best_asset["relevance_score"]
        result["answer_quality_score"] = best_asset["quality_score"]
        result["answer_frame_score"] = best_asset["answer_frame_score"]
        result["answer_selection"] = {
            "asset_type": best_asset["asset_type"],
            "relevance_score": best_asset["relevance_score"],
            "quality_score": best_asset["quality_score"],
            "sharpness_score": best_asset["sharpness_score"],
            "brightness_score": best_asset["brightness_score"],
            "query_focus": focus,
        }
        sibling_frame = _sibling_frame_path(result, frame_idx, str(best_asset["asset_type"]))
        if sibling_frame:
            result["answer_context_frame_path"] = sibling_frame
        result["answer_window"] = _answer_window(result, frame_idx, query_profile)

    return results
