from __future__ import annotations

import json
import re
from pathlib import Path

from .models import Moment

ATTRIBUTE_FILENAMES = ("person_attributes.json", "attributes.json")
SCENE_FILENAMES = ("scene_context.json", "scene_evidence.json")
TOKEN_RE = re.compile(r"[a-z0-9]+")
TRUTHY_STRINGS = {"1", "true", "yes", "present", "available", "on"}
FALSEY_STRINGS = {"0", "false", "no", "absent", "off"}
TOP_COLOR_KEYS = ("top_color", "upper_color", "shirt_color", "upper_body_color", "clothing_color")
TOP_TYPE_KEYS = ("top_type", "upper_type", "shirt_type", "upper_body_type", "clothing_type")
ACCESSORY_KEYWORDS = {
    "backpack": ("backpack", "has_backpack", "wearing_backpack"),
    "bag": ("bag", "handbag", "shoulder_bag", "carrying_bag"),
    "hat": ("hat", "cap", "helmet", "headwear"),
    "umbrella": ("umbrella",),
}
SCENE_NEAR_ROAD_KEYS = ("near_road", "roadside", "road_proximity")


def _tokenize(value: str) -> set[str]:
    return set(TOKEN_RE.findall(str(value or "").lower()))


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUTHY_STRINGS:
            return True
        if lowered in FALSEY_STRINGS:
            return False
    return None


def _string_value(payload: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _is_near_road(scene_evidence: dict) -> bool:
    for key in SCENE_NEAR_ROAD_KEYS:
        if key not in scene_evidence:
            continue
        value = scene_evidence.get(key)
        coerced = _coerce_bool(value)
        if coerced is not None:
            return coerced
        try:
            return float(value) <= 2.5
        except Exception:
            continue

    for key in ("road_distance_m", "distance_to_road", "road_distance"):
        if key not in scene_evidence:
            continue
        try:
            return float(scene_evidence[key]) <= 2.5
        except Exception:
            continue
    return False


def summarize_attribute_evidence(attribute_evidence: dict | None) -> list[str]:
    if not isinstance(attribute_evidence, dict):
        return []

    phrases: list[str] = []
    top_color = _string_value(attribute_evidence, TOP_COLOR_KEYS)
    top_type = _string_value(attribute_evidence, TOP_TYPE_KEYS)
    if top_color and top_type:
        phrases.append(f"{top_color.strip().lower()} {top_type.strip().lower()}")
    elif top_color:
        phrases.append(top_color.strip().lower())

    for label, keys in ACCESSORY_KEYWORDS.items():
        for key in keys:
            if key not in attribute_evidence:
                continue
            bool_value = _coerce_bool(attribute_evidence[key])
            if bool_value:
                phrases.append(label)
                break
            if isinstance(attribute_evidence[key], str) and label in _tokenize(attribute_evidence[key]):
                phrases.append(label)
                break

    for key in ("lower_color", "pants_color", "shoe_color"):
        value = attribute_evidence.get(key)
        if not value:
            continue
        phrases.append(str(value).strip().lower())

    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        normalized = " ".join(phrase.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def summarize_scene_evidence(scene_evidence: dict | None) -> list[str]:
    if not isinstance(scene_evidence, dict):
        return []

    phrases: list[str] = []
    if _is_near_road(scene_evidence):
        phrases.append("near road")

    for key in ("near_sidewalk", "sidewalk", "near_crosswalk", "crosswalk", "drivable_area"):
        if key not in scene_evidence:
            continue
        bool_value = _coerce_bool(scene_evidence[key])
        if bool_value:
            phrases.append(str(key).replace("_", " "))

    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        normalized = " ".join(phrase.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def evidence_terms(attribute_evidence: dict | None, scene_evidence: dict | None) -> set[str]:
    terms: set[str] = set()
    for phrase in summarize_attribute_evidence(attribute_evidence):
        terms.update(_tokenize(phrase))
    for phrase in summarize_scene_evidence(scene_evidence):
        terms.update(_tokenize(phrase))
    return terms


def _candidate_record_keys(record: dict) -> set[str]:
    keys: set[str] = set()
    moment_id = str(record.get("moment_id") or "").strip()
    if moment_id:
        keys.add(f"moment::{moment_id}")

    track_id = str(record.get("track_id") or "").strip()
    if track_id:
        video_name = str(record.get("video_name") or record.get("video") or "").strip()
        if video_name:
            keys.add(f"video_track::{Path(video_name).stem}:{track_id}")
        location = str(record.get("location") or "").strip()
        split = str(record.get("split") or "").strip()
        if location and split:
            keys.add(f"location_split_track::{location}:{split}:{track_id}")
    return keys


def _candidate_moment_keys(moment: Moment) -> list[str]:
    video_stem = Path(moment.video_path).stem
    return [
        f"moment::{moment.id}",
        f"location_split_track::{moment.location}:{moment.split}:{moment.track_id}",
        f"video_track::{video_stem}:{moment.track_id}",
    ]


def _load_json_file(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_record_payloads(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if "records" in payload and isinstance(payload["records"], list):
            return [item for item in payload["records"] if isinstance(item, dict)]
        if all(isinstance(value, dict) for value in payload.values()):
            records: list[dict] = []
            for key, value in payload.items():
                record = dict(value)
                record.setdefault("moment_id", key)
                records.append(record)
            return records
    return []


def _merge_payload(target: dict, source: dict) -> dict:
    merged = dict(target)
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def load_enrichment_index(dataset_root: Path) -> dict[str, dict]:
    enrichment_dir = dataset_root / "enrichment"
    if not enrichment_dir.exists():
        return {}

    combined: dict[str, dict] = {}
    for filenames, field_name in (
        (ATTRIBUTE_FILENAMES, "attributes"),
        (SCENE_FILENAMES, "scene"),
    ):
        for filename in filenames:
            path = enrichment_dir / filename
            if not path.exists():
                continue
            payload = _load_json_file(path)
            for record in _iter_record_payloads(payload):
                record_keys = _candidate_record_keys(record)
                if not record_keys:
                    continue
                values = record.get(field_name)
                if not isinstance(values, dict):
                    values = {
                        key: value
                        for key, value in record.items()
                        if key not in {"moment_id", "video_name", "video", "track_id", "location", "split", "attributes", "scene"}
                    }
                normalized_payload = {field_name: values}
                for record_key in record_keys:
                    combined[record_key] = _merge_payload(combined.get(record_key, {}), normalized_payload)
    return combined


def apply_enrichment(moments: list[Moment], dataset_root: Path) -> list[Moment]:
    enrichment_index = load_enrichment_index(dataset_root)
    if not enrichment_index:
        return moments

    for moment in moments:
        merged: dict = {}
        for key in _candidate_moment_keys(moment):
            if key in enrichment_index:
                merged = _merge_payload(merged, enrichment_index[key])

        attribute_evidence = merged.get("attributes", {}) if isinstance(merged.get("attributes"), dict) else {}
        scene_evidence = merged.get("scene", {}) if isinstance(merged.get("scene"), dict) else {}
        if not attribute_evidence and not scene_evidence:
            continue

        moment.attribute_evidence = attribute_evidence
        moment.scene_evidence = scene_evidence

        phrases = summarize_attribute_evidence(attribute_evidence) + summarize_scene_evidence(scene_evidence)
        if phrases:
            evidence_text = ". ".join(phrases)
            if evidence_text:
                moment.text = f"{moment.text} Evidence: {evidence_text}."
                keywords = set(moment.keywords)
                for phrase in phrases:
                    keywords.update(_tokenize(phrase))
                moment.keywords = sorted(keywords)

    return moments
