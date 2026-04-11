from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import DEFAULT_DATASET_TYPE
from .dataset import collect_moments, iter_dataset_source_files
from .enrichment import apply_enrichment
from .enrichment_inference import infer_and_write_enrichment
from .indexer import build_search_bundle
from .video_tools import attach_existing_visual_assets, extract_visual_assets


@dataclass
class RuntimeConfig:
    dataset_type: str
    dataset_root: Path
    output_dir: Path
    assets_dir: Path
    locations: list[str] | None
    splits: list[str] | None
    group_by_track: bool
    fps: float
    enable_sparse: bool
    enable_dense: bool
    enable_clip: bool
    sentence_model_name: str
    clip_model_name: str
    clip_pretrained: str
    device: str | None
    batch_size: int
    max_assets_per_moment: int
    crop_padding: float
    ffmpeg_bin: str
    enable_enrichment: bool = False


def runtime_manifest_path(output_dir: Path) -> Path:
    return output_dir / "runtime_manifest.json"


def bundle_path(output_dir: Path) -> Path:
    return output_dir / "bundle.json"


def snapshot_dataset_sources(
    dataset_root: Path,
    dataset_type: str = DEFAULT_DATASET_TYPE,
    locations: list[str] | None = None,
    splits: list[str] | None = None,
) -> list[dict]:
    entries: list[dict] = []
    for source_path in iter_dataset_source_files(
        dataset_root,
        dataset_type=dataset_type,
        locations=locations,
        splits=splits,
    ):
        if not source_path.exists():
            continue
        stat = source_path.stat()
        entries.append(
            {
                "path": source_path.relative_to(dataset_root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return sorted(entries, key=lambda item: item["path"])


def load_runtime_manifest(output_dir: Path) -> dict | None:
    manifest_path = runtime_manifest_path(output_dir)
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_runtime_manifest(output_dir: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runtime_manifest_path(output_dir)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return manifest_path


def dataset_has_changes(config: RuntimeConfig) -> bool:
    current_sources = snapshot_dataset_sources(
        config.dataset_root,
        dataset_type=config.dataset_type,
        locations=config.locations,
        splits=config.splits,
    )
    previous_manifest = load_runtime_manifest(config.output_dir)
    if previous_manifest is None:
        return True
    if not bundle_path(config.output_dir).exists():
        return True
    return previous_manifest.get("sources") != current_sources


def rebuild_runtime_bundle(config: RuntimeConfig) -> dict:
    moments = collect_moments(
        dataset_root=config.dataset_root,
        fps=config.fps,
        group_by_track=config.group_by_track,
        locations=config.locations,
        splits=config.splits,
        dataset_type=config.dataset_type,
    )

    videos_available = any(Path(moment.video_path).exists() for moment in moments)

    if (config.enable_clip or config.enable_enrichment) and videos_available:
        attach_existing_visual_assets(moments, config.assets_dir)
        extract_visual_assets(
            moments,
            output_dir=config.assets_dir,
            max_assets_per_moment=config.max_assets_per_moment,
            crop_padding=config.crop_padding,
            overwrite=False,
            ffmpeg_bin=config.ffmpeg_bin,
        )
        attach_existing_visual_assets(moments, config.assets_dir)

    if config.enable_enrichment:
        infer_and_write_enrichment(
            moments=moments,
            dataset_root=config.dataset_root,
            clip_model_name=config.clip_model_name,
            clip_pretrained=config.clip_pretrained,
            device=config.device,
            batch_size=config.batch_size,
        )
        moments = apply_enrichment(moments, config.dataset_root)

    bundle_file = build_search_bundle(
        moments=moments,
        output_dir=config.output_dir,
        enable_sparse=config.enable_sparse,
        enable_dense=config.enable_dense,
        enable_clip=config.enable_clip,
        sentence_model_name=config.sentence_model_name,
        clip_model_name=config.clip_model_name,
        clip_pretrained=config.clip_pretrained,
        device=config.device,
        batch_size=config.batch_size,
    )

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "bundle_path": str(bundle_file),
        "output_dir": str(config.output_dir),
        "dataset_root": str(config.dataset_root),
        "dataset_type": config.dataset_type,
        "assets_dir": str(config.assets_dir),
        "locations": config.locations,
        "splits": config.splits,
        "group_by_track": config.group_by_track,
        "fps": config.fps,
        "enable_sparse": config.enable_sparse,
        "enable_dense": config.enable_dense,
        "enable_clip": config.enable_clip,
        "moment_count": len(moments),
        "sources": snapshot_dataset_sources(
            config.dataset_root,
            dataset_type=config.dataset_type,
            locations=config.locations,
            splits=config.splits,
        ),
        "config": _serialize_runtime_config(config),
    }
    save_runtime_manifest(config.output_dir, manifest)
    return manifest


def watch_and_rebuild(config: RuntimeConfig, poll_seconds: float = 10.0, run_once: bool = False) -> None:
    while True:
        if dataset_has_changes(config):
            manifest = rebuild_runtime_bundle(config)
            print(
                f"Rebuilt bundle at {manifest['built_at']} with {manifest['moment_count']} moments "
                f"from {config.dataset_root}"
            )
        else:
            print("No dataset changes detected.")

        if run_once:
            return
        time.sleep(max(poll_seconds, 1.0))


def _serialize_runtime_config(config: RuntimeConfig) -> dict:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload
