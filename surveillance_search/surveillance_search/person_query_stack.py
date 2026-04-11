from __future__ import annotations

import json
from pathlib import Path

from .config import project_root


def default_stack_layout(root: Path | None = None) -> dict[str, Path]:
    base_root = root or project_root()
    data_root = base_root / "data"
    artifacts_root = data_root / "artifacts"
    configs_root = base_root / "configs" / "person_query_stack"
    return {
        "project_root": base_root,
        "data_root": data_root,
        "personpath22_root": data_root / "personpath22",
        "pa100k_root": data_root / "pa100k",
        "bdd100k_root": data_root / "bdd100k",
        "prepared_root": data_root / "prepared",
        "pa100k_prepared_root": data_root / "prepared" / "pa100k",
        "bdd100k_prepared_root": data_root / "prepared" / "bdd100k",
        "artifacts_root": artifacts_root,
        "bundle_root": artifacts_root / "full",
        "visual_root": artifacts_root / "visual_assets",
        "models_root": artifacts_root / "models",
        "configs_root": configs_root,
    }


def ensure_person_query_stack_layout(root: Path | None = None) -> dict[str, Path]:
    layout = default_stack_layout(root)
    for key, path in layout.items():
        if key.endswith("_root") and key != "project_root":
            path.mkdir(parents=True, exist_ok=True)
    return layout


def write_stack_configs(root: Path | None = None) -> dict[str, Path]:
    layout = ensure_person_query_stack_layout(root)
    configs_root = layout["configs_root"]

    attribute_config = configs_root / "attribute_model.json"
    scene_config = configs_root / "scene_model.json"
    stack_config = configs_root / "stack.json"

    attribute_payload = {
        "name": "person_attribute_baseline",
        "strategy": "clip_zero_shot_baseline",
        "supervision_dataset": "PA-100K",
        "prepared_dataset_root": str(layout["pa100k_prepared_root"]),
        "output_artifact": str(layout["models_root"] / "attribute_baseline.json"),
        "labels_of_interest": [
            "top_color",
            "top_type",
            "backpack",
            "bag",
            "hat",
        ],
        "notes": [
            "This is a baseline scaffold, not a heavy supervised trainer.",
            "Replace the strategy with a fine-tuned PA-100K or RAPv2 model when available.",
        ],
    }
    scene_payload = {
        "name": "scene_relation_baseline",
        "strategy": "clip_zero_shot_baseline",
        "supervision_dataset": "BDD100K",
        "prepared_dataset_root": str(layout["bdd100k_prepared_root"]),
        "output_artifact": str(layout["models_root"] / "scene_baseline.json"),
        "labels_of_interest": [
            "near_road",
            "drivable_area",
        ],
        "notes": [
            "This baseline keeps the pipeline end-to-end with free pretrained components.",
            "Replace with a fine-tuned drivable-area segmentation model if stronger scene grounding is needed.",
        ],
    }
    stack_payload = {
        "video_base_dataset": {
            "name": "PersonPath22",
            "root": str(layout["personpath22_root"]),
        },
        "attribute_supervision": {
            "name": "PA-100K",
            "raw_root": str(layout["pa100k_root"]),
            "prepared_root": str(layout["pa100k_prepared_root"]),
            "config": str(attribute_config),
        },
        "scene_supervision": {
            "name": "BDD100K",
            "raw_root": str(layout["bdd100k_root"]),
            "prepared_root": str(layout["bdd100k_prepared_root"]),
            "config": str(scene_config),
        },
        "bundle_output": str(layout["bundle_root"]),
        "visual_assets": str(layout["visual_root"]),
        "models_root": str(layout["models_root"]),
        "end_to_end_build_command": "python -m surveillance_search build --dataset-type personpath22 --profile strongest --auto-enrich",
    }

    for path, payload in (
        (attribute_config, attribute_payload),
        (scene_config, scene_payload),
        (stack_config, stack_payload),
    ):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    return {
        "attribute_config": attribute_config,
        "scene_config": scene_config,
        "stack_config": stack_config,
    }
