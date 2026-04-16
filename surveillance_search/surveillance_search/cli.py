from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_PRETRAINED,
    DEFAULT_DATASET_TYPE,
    DEFAULT_FPS,
    DEFAULT_LOCATIONS,
    DEFAULT_MAX_ASSETS_PER_MOMENT,
    DEFAULT_PROFILE,
    DEFAULT_REPO_ID,
    DEFAULT_SENTENCE_MODEL,
    DEFAULT_SPLITS,
    DEFAULT_TOP_K,
    SUPPORTED_DATASET_TYPES,
    default_bundle_root,
    default_data_root,
    default_visual_root,
)
from .dataset import collect_moments, download_from_huggingface
from .enrichment import apply_enrichment
from .enrichment_inference import infer_and_write_enrichment
from .hospital_ingest import (
    HospitalIngestConfig,
    DEFAULT_QUEUE_SIZE,
    bootstrap_nvidia_hospital_dataset,
    ingest_hospital_video,
)
from .indexer import build_search_bundle, search_index
from .qwen_integration import local_qwen_embedding_model
from .runtime import RuntimeConfig, rebuild_runtime_bundle, watch_and_rebuild
from .video_tools import (
    attach_existing_visual_assets,
    count_visual_assets,
    extract_clip,
    extract_visual_assets,
)


def parse_csv(value: str | None, fallback: tuple[str, ...] | None = None) -> list[str] | None:
    if not value:
        return list(fallback) if fallback else None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_weights(value: str | None) -> dict[str, float] | None:
    if not value:
        return None
    weights: dict[str, float] = {}
    for item in value.split(","):
        chunk = item.strip()
        if not chunk:
            continue
        name, raw_value = chunk.split("=", 1)
        weights[name.strip()] = float(raw_value.strip())
    return weights


def resolve_profile(profile: str, args: argparse.Namespace) -> tuple[bool, bool, bool]:
    profiles = {
        "strongest": (True, True, True),
        "balanced": (True, True, True),
        "text-only": (True, True, False),
        "lite": (True, False, False),
    }
    enable_sparse, enable_dense, enable_clip = profiles[profile]
    if getattr(args, "no_sparse", False):
        enable_sparse = False
    if getattr(args, "no_dense", False):
        enable_dense = False
    if getattr(args, "no_clip", False):
        enable_clip = False
    return enable_sparse, enable_dense, enable_clip


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    if getattr(args, "dataset_root", None):
        return Path(args.dataset_root)
    return default_data_root(args.dataset_type)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "output_dir", None):
        return Path(args.output_dir)
    return default_bundle_root()


def resolve_assets_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "assets_dir", None):
        return Path(args.assets_dir)
    return default_visual_root()


def resolve_sentence_model_name(args: argparse.Namespace) -> str:
    if getattr(args, "use_qwen_embedding", False):
        model_path = local_qwen_embedding_model()
        if not model_path:
            raise FileNotFoundError(
                "Local Qwen embedding model was requested but not found under ./models/Qwen3-Embedding-4B."
            )
        return model_path
    return args.sentence_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full-featured surveillance multimodal search engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download labels, videos and docs from Hugging Face")
    download_parser.add_argument("--dataset-type", choices=SUPPORTED_DATASET_TYPES, default=DEFAULT_DATASET_TYPE)
    download_parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    download_parser.add_argument("--dataset-root", default=None)
    download_parser.add_argument("--locations", default=None, help="Comma-separated list")
    download_parser.add_argument("--splits", default=None, help="Comma-separated list")
    download_parser.add_argument("--include-videos", action="store_true")
    download_parser.add_argument("--include-docs", action="store_true")

    bootstrap_parser = subparsers.add_parser("bootstrap", help="Download selected data and build a production-ready bundle")
    bootstrap_parser.add_argument("--dataset-type", choices=SUPPORTED_DATASET_TYPES, default=DEFAULT_DATASET_TYPE)
    bootstrap_parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    bootstrap_parser.add_argument("--dataset-root", default=None)
    bootstrap_parser.add_argument("--output-dir", default=None)
    bootstrap_parser.add_argument("--assets-dir", default=None)
    bootstrap_parser.add_argument("--locations", default="amsterdam", help="Comma-separated list")
    bootstrap_parser.add_argument("--splits", default="test", help="Comma-separated list")
    bootstrap_parser.add_argument("--group-mode", choices=("track", "frame"), default="track")
    bootstrap_parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    bootstrap_parser.add_argument("--profile", choices=("strongest", "balanced", "text-only", "lite"), default=DEFAULT_PROFILE)
    bootstrap_parser.add_argument("--include-videos", action="store_true")
    bootstrap_parser.add_argument("--include-docs", action="store_true")
    bootstrap_parser.add_argument("--max-assets-per-moment", type=int, default=DEFAULT_MAX_ASSETS_PER_MOMENT)
    bootstrap_parser.add_argument("--crop-padding", type=float, default=0.08)
    bootstrap_parser.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
    bootstrap_parser.add_argument("--use-qwen-embedding", action="store_true", help="Use the local ./models/Qwen3-Embedding-4B directory as the dense embedding model.")
    bootstrap_parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    bootstrap_parser.add_argument("--clip-pretrained", default=DEFAULT_CLIP_PRETRAINED)
    bootstrap_parser.add_argument("--device", default=None)
    bootstrap_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    bootstrap_parser.add_argument("--no-sparse", action="store_true")
    bootstrap_parser.add_argument("--no-dense", action="store_true")
    bootstrap_parser.add_argument("--no-clip", action="store_true")
    bootstrap_parser.add_argument("--auto-enrich", action="store_true", help="Infer person attributes and scene evidence from visual assets before building the bundle.")

    prepare_parser = subparsers.add_parser("prepare-assets", help="Extract frames and crops for CLIP indexing")
    prepare_parser.add_argument("--dataset-type", choices=SUPPORTED_DATASET_TYPES, default=DEFAULT_DATASET_TYPE)
    prepare_parser.add_argument("--dataset-root", default=None)
    prepare_parser.add_argument("--assets-dir", default=None)
    prepare_parser.add_argument("--locations", default=None, help="Comma-separated list")
    prepare_parser.add_argument("--splits", default=None, help="Comma-separated list")
    prepare_parser.add_argument("--group-mode", choices=("track", "frame"), default="track")
    prepare_parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    prepare_parser.add_argument("--max-assets-per-moment", type=int, default=DEFAULT_MAX_ASSETS_PER_MOMENT)
    prepare_parser.add_argument("--crop-padding", type=float, default=0.08)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.add_argument("--ffmpeg-bin", default="ffmpeg")

    build_parser_cmd = subparsers.add_parser("build", help="Build a full multimodal search bundle")
    build_parser_cmd.add_argument("--dataset-type", choices=SUPPORTED_DATASET_TYPES, default=DEFAULT_DATASET_TYPE)
    build_parser_cmd.add_argument("--dataset-root", default=None)
    build_parser_cmd.add_argument("--output-dir", default=None)
    build_parser_cmd.add_argument("--assets-dir", default=None)
    build_parser_cmd.add_argument("--locations", default=None, help="Comma-separated list")
    build_parser_cmd.add_argument("--splits", default=None, help="Comma-separated list")
    build_parser_cmd.add_argument("--group-mode", choices=("track", "frame"), default="track")
    build_parser_cmd.add_argument("--fps", type=float, default=DEFAULT_FPS)
    build_parser_cmd.add_argument("--profile", choices=("strongest", "balanced", "text-only", "lite"), default=DEFAULT_PROFILE)
    build_parser_cmd.add_argument("--prepare-assets", action="store_true")
    build_parser_cmd.add_argument("--max-assets-per-moment", type=int, default=DEFAULT_MAX_ASSETS_PER_MOMENT)
    build_parser_cmd.add_argument("--crop-padding", type=float, default=0.08)
    build_parser_cmd.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
    build_parser_cmd.add_argument("--use-qwen-embedding", action="store_true", help="Use the local ./models/Qwen3-Embedding-4B directory as the dense embedding model.")
    build_parser_cmd.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    build_parser_cmd.add_argument("--clip-pretrained", default=DEFAULT_CLIP_PRETRAINED)
    build_parser_cmd.add_argument("--device", default=None)
    build_parser_cmd.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    build_parser_cmd.add_argument("--ffmpeg-bin", default="ffmpeg")
    build_parser_cmd.add_argument("--no-sparse", action="store_true")
    build_parser_cmd.add_argument("--no-dense", action="store_true")
    build_parser_cmd.add_argument("--no-clip", action="store_true")
    build_parser_cmd.add_argument("--auto-enrich", action="store_true", help="Infer person attributes and scene evidence from visual assets before building the bundle.")

    search_parser = subparsers.add_parser("search", help="Search by text, image, or both")
    search_parser.add_argument("--output-dir", default=str(default_bundle_root()))
    search_parser.add_argument("--query-text", default=None)
    search_parser.add_argument("--image-path", default=None)
    search_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    search_parser.add_argument("--search-mode", choices=("default", "person"), default="default")
    search_parser.add_argument("--weights", default=None, help="Example: sparse=0.1,dense=0.4,clip_text=0.2,clip_image=0.3")
    search_parser.add_argument("--json", action="store_true")
    search_parser.add_argument("--use-qwen-parser", action="store_true", help="Use local Qwen3 query parsing for better multilingual person-query understanding.")
    search_parser.add_argument("--use-qwen-reranker", action="store_true", help="Use local Qwen3 reranking on the top search candidates.")
    search_parser.add_argument("--qwen-rerank-limit", type=int, default=20)
    search_parser.add_argument("--qwen-device", default=None)

    clip_parser = subparsers.add_parser("clip", help="Extract a clip from a retrieved result")
    clip_parser.add_argument("--output-dir", default=str(default_bundle_root()))
    clip_parser.add_argument("--query-text", default=None)
    clip_parser.add_argument("--image-path", default=None)
    clip_parser.add_argument("--search-mode", choices=("default", "person"), default="default")
    clip_parser.add_argument("--weights", default=None)
    clip_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    clip_parser.add_argument("--rank", type=int, default=1, help="1-based result rank")
    clip_parser.add_argument("--duration", type=float, default=4.0)
    clip_parser.add_argument("--padding", type=float, default=0.5)
    clip_parser.add_argument("--output", required=True)
    clip_parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    clip_parser.add_argument("--use-qwen-parser", action="store_true", help="Use local Qwen3 query parsing for better multilingual person-query understanding.")
    clip_parser.add_argument("--use-qwen-reranker", action="store_true", help="Use local Qwen3 reranking on the top search candidates.")
    clip_parser.add_argument("--qwen-rerank-limit", type=int, default=20)
    clip_parser.add_argument("--qwen-device", default=None)

    api_parser = subparsers.add_parser("serve-api", help="Run FastAPI service")
    api_parser.add_argument("--output-dir", default=str(default_bundle_root()))
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8000)
    api_parser.add_argument("--reload", action="store_true")

    rebuild_parser = subparsers.add_parser("rebuild", help="Rebuild the bundle from current dataset files")
    rebuild_parser.add_argument("--dataset-type", choices=SUPPORTED_DATASET_TYPES, default=DEFAULT_DATASET_TYPE)
    rebuild_parser.add_argument("--dataset-root", default=None)
    rebuild_parser.add_argument("--output-dir", default=None)
    rebuild_parser.add_argument("--assets-dir", default=None)
    rebuild_parser.add_argument("--locations", default=None, help="Comma-separated list")
    rebuild_parser.add_argument("--splits", default=None, help="Comma-separated list")
    rebuild_parser.add_argument("--group-mode", choices=("track", "frame"), default="track")
    rebuild_parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    rebuild_parser.add_argument("--profile", choices=("strongest", "balanced", "text-only", "lite"), default=DEFAULT_PROFILE)
    rebuild_parser.add_argument("--max-assets-per-moment", type=int, default=DEFAULT_MAX_ASSETS_PER_MOMENT)
    rebuild_parser.add_argument("--crop-padding", type=float, default=0.08)
    rebuild_parser.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
    rebuild_parser.add_argument("--use-qwen-embedding", action="store_true", help="Use the local ./models/Qwen3-Embedding-4B directory as the dense embedding model.")
    rebuild_parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    rebuild_parser.add_argument("--clip-pretrained", default=DEFAULT_CLIP_PRETRAINED)
    rebuild_parser.add_argument("--device", default=None)
    rebuild_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    rebuild_parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    rebuild_parser.add_argument("--no-sparse", action="store_true")
    rebuild_parser.add_argument("--no-dense", action="store_true")
    rebuild_parser.add_argument("--no-clip", action="store_true")
    rebuild_parser.add_argument("--auto-enrich", action="store_true", help="Infer person attributes and scene evidence from visual assets before rebuilding the bundle.")

    watch_parser = subparsers.add_parser("watch-index", help="Poll the dataset directory and rebuild when files change")
    watch_parser.add_argument("--dataset-type", choices=SUPPORTED_DATASET_TYPES, default=DEFAULT_DATASET_TYPE)
    watch_parser.add_argument("--dataset-root", default=None)
    watch_parser.add_argument("--output-dir", default=None)
    watch_parser.add_argument("--assets-dir", default=None)
    watch_parser.add_argument("--locations", default=None, help="Comma-separated list")
    watch_parser.add_argument("--splits", default=None, help="Comma-separated list")
    watch_parser.add_argument("--group-mode", choices=("track", "frame"), default="track")
    watch_parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    watch_parser.add_argument("--profile", choices=("strongest", "balanced", "text-only", "lite"), default=DEFAULT_PROFILE)
    watch_parser.add_argument("--max-assets-per-moment", type=int, default=DEFAULT_MAX_ASSETS_PER_MOMENT)
    watch_parser.add_argument("--crop-padding", type=float, default=0.08)
    watch_parser.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
    watch_parser.add_argument("--use-qwen-embedding", action="store_true", help="Use the local ./models/Qwen3-Embedding-4B directory as the dense embedding model.")
    watch_parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    watch_parser.add_argument("--clip-pretrained", default=DEFAULT_CLIP_PRETRAINED)
    watch_parser.add_argument("--device", default=None)
    watch_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    watch_parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    watch_parser.add_argument("--poll-seconds", type=float, default=10.0)
    watch_parser.add_argument("--once", action="store_true")
    watch_parser.add_argument("--no-sparse", action="store_true")
    watch_parser.add_argument("--no-dense", action="store_true")
    watch_parser.add_argument("--no-clip", action="store_true")
    watch_parser.add_argument("--auto-enrich", action="store_true", help="Infer person attributes and scene evidence during each rebuild cycle.")

    enrich_parser = subparsers.add_parser("enrich", help="Infer person attributes and scene evidence from visual assets")
    enrich_parser.add_argument("--dataset-type", choices=SUPPORTED_DATASET_TYPES, default=DEFAULT_DATASET_TYPE)
    enrich_parser.add_argument("--dataset-root", default=None)
    enrich_parser.add_argument("--assets-dir", default=None)
    enrich_parser.add_argument("--locations", default=None, help="Comma-separated list")
    enrich_parser.add_argument("--splits", default=None, help="Comma-separated list")
    enrich_parser.add_argument("--group-mode", choices=("track", "frame"), default="track")
    enrich_parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    enrich_parser.add_argument("--prepare-assets", action="store_true")
    enrich_parser.add_argument("--max-assets-per-moment", type=int, default=DEFAULT_MAX_ASSETS_PER_MOMENT)
    enrich_parser.add_argument("--crop-padding", type=float, default=0.08)
    enrich_parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    enrich_parser.add_argument("--clip-pretrained", default=DEFAULT_CLIP_PRETRAINED)
    enrich_parser.add_argument("--device", default=None)
    enrich_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    enrich_parser.add_argument("--ffmpeg-bin", default="ffmpeg")

    demo_parser = subparsers.add_parser("demo", help="Launch Streamlit demo UI")
    demo_parser.add_argument("--port", type=int, default=8501)

    ingest_parser = subparsers.add_parser("ingest-video", help="Ingest a new hospital camera video incrementally")
    ingest_parser.add_argument("--source-path", required=True)
    ingest_parser.add_argument("--dataset-type", choices=("hospital",), default="hospital")
    ingest_parser.add_argument("--dataset-root", default=None)
    ingest_parser.add_argument("--output-dir", default=None)
    ingest_parser.add_argument("--assets-dir", default=None)
    ingest_parser.add_argument("--camera-id", default=None)
    ingest_parser.add_argument("--start-time", default=None, help="Recorded UTC start time in ISO 8601.")
    ingest_parser.add_argument("--queue-size", type=int, default=DEFAULT_QUEUE_SIZE)
    ingest_parser.add_argument("--detection-fps", type=float, default=1.0)
    ingest_parser.add_argument("--min-track-frames", type=int, default=2)
    ingest_parser.add_argument("--min-person-area", type=int, default=1600)
    ingest_parser.add_argument("--profile", choices=("strongest", "balanced", "text-only", "lite"), default=DEFAULT_PROFILE)
    ingest_parser.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
    ingest_parser.add_argument("--use-qwen-embedding", action="store_true", help="Use the local ./models/Qwen3-Embedding-4B directory as the dense embedding model.")
    ingest_parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    ingest_parser.add_argument("--clip-pretrained", default=DEFAULT_CLIP_PRETRAINED)
    ingest_parser.add_argument("--device", default=None)
    ingest_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ingest_parser.add_argument("--max-assets-per-moment", type=int, default=DEFAULT_MAX_ASSETS_PER_MOMENT)
    ingest_parser.add_argument("--crop-padding", type=float, default=0.08)
    ingest_parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    ingest_parser.add_argument("--encoder", default="hevc_nvenc")
    ingest_parser.add_argument("--no-enrichment", action="store_true")
    ingest_parser.add_argument("--no-sparse", action="store_true")
    ingest_parser.add_argument("--no-dense", action="store_true")
    ingest_parser.add_argument("--no-clip", action="store_true")

    bootstrap_nvidia_parser = subparsers.add_parser(
        "bootstrap-nvidia-hospital",
        help="Precompute the NVIDIA Hospital camera set into detailed per-person metadata",
    )
    bootstrap_nvidia_parser.add_argument("--dataset-type", choices=("hospital",), default="hospital")
    bootstrap_nvidia_parser.add_argument("--dataset-root", default=None)
    bootstrap_nvidia_parser.add_argument("--output-dir", default=None)
    bootstrap_nvidia_parser.add_argument("--assets-dir", default=None)
    bootstrap_nvidia_parser.add_argument(
        "--nvidia-root",
        default="/teamspace/studios/this_studio/Multi-Camera-Person-Tracking-and-Re-Identification/data/NVIDIA_SmartSpaces/MTMC_Tracking_2025/val/Hospital_000",
    )
    bootstrap_nvidia_parser.add_argument("--start-time", default=None, help="Recorded UTC base start time in ISO 8601.")
    bootstrap_nvidia_parser.add_argument("--limit", type=int, default=31)
    bootstrap_nvidia_parser.add_argument("--queue-size", type=int, default=DEFAULT_QUEUE_SIZE)
    bootstrap_nvidia_parser.add_argument("--group-mode", choices=("track", "frame"), default="track")
    bootstrap_nvidia_parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    bootstrap_nvidia_parser.add_argument("--locations", default=None)
    bootstrap_nvidia_parser.add_argument("--splits", default=None)
    bootstrap_nvidia_parser.add_argument("--profile", choices=("strongest", "balanced", "text-only", "lite"), default=DEFAULT_PROFILE)
    bootstrap_nvidia_parser.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
    bootstrap_nvidia_parser.add_argument("--use-qwen-embedding", action="store_true", help="Use the local ./models/Qwen3-Embedding-4B directory as the dense embedding model.")
    bootstrap_nvidia_parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    bootstrap_nvidia_parser.add_argument("--clip-pretrained", default=DEFAULT_CLIP_PRETRAINED)
    bootstrap_nvidia_parser.add_argument("--device", default=None)
    bootstrap_nvidia_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    bootstrap_nvidia_parser.add_argument("--max-assets-per-moment", type=int, default=8)
    bootstrap_nvidia_parser.add_argument("--crop-padding", type=float, default=0.08)
    bootstrap_nvidia_parser.add_argument("--timeline-stride-seconds", type=float, default=1.0)
    bootstrap_nvidia_parser.add_argument("--max-timeline-segments", type=int, default=16)
    bootstrap_nvidia_parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    bootstrap_nvidia_parser.add_argument("--encoder", default="hevc_nvenc")
    bootstrap_nvidia_parser.add_argument("--no-enrichment", action="store_true")
    bootstrap_nvidia_parser.add_argument("--no-sparse", action="store_true")
    bootstrap_nvidia_parser.add_argument("--no-dense", action="store_true")
    bootstrap_nvidia_parser.add_argument("--no-clip", action="store_true")

    return parser


def _collect_moments_for_args(args: argparse.Namespace):
    return collect_moments(
        dataset_type=args.dataset_type,
        dataset_root=resolve_dataset_root(args),
        fps=args.fps,
        group_by_track=args.group_mode == "track",
        locations=parse_csv(args.locations),
        splits=parse_csv(args.splits),
    )


def _runtime_config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    enable_sparse, enable_dense, enable_clip = resolve_profile(args.profile, args)
    return RuntimeConfig(
        dataset_type=args.dataset_type,
        dataset_root=resolve_dataset_root(args),
        output_dir=resolve_output_dir(args),
        assets_dir=resolve_assets_dir(args),
        locations=parse_csv(args.locations),
        splits=parse_csv(args.splits),
        group_by_track=args.group_mode == "track",
        fps=args.fps,
        enable_sparse=enable_sparse,
        enable_dense=enable_dense,
        enable_clip=enable_clip,
        sentence_model_name=resolve_sentence_model_name(args),
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=args.device,
        batch_size=args.batch_size,
        max_assets_per_moment=args.max_assets_per_moment,
        crop_padding=args.crop_padding,
        ffmpeg_bin=args.ffmpeg_bin,
        enable_enrichment=getattr(args, "auto_enrich", False),
    )


def _hospital_ingest_config_from_args(args: argparse.Namespace) -> HospitalIngestConfig:
    return HospitalIngestConfig(
        dataset_root=resolve_dataset_root(args),
        assets_dir=resolve_assets_dir(args),
        ffmpeg_bin=getattr(args, "ffmpeg_bin", "ffmpeg"),
        preferred_encoder=getattr(args, "encoder", "hevc_nvenc"),
        queue_size=getattr(args, "queue_size", DEFAULT_QUEUE_SIZE),
        detection_fps=getattr(args, "detection_fps", 1.0),
        min_track_frames=getattr(args, "min_track_frames", 2),
        min_person_area=getattr(args, "min_person_area", 1600),
        enable_enrichment=not getattr(args, "no_enrichment", False),
        clip_model_name=getattr(args, "clip_model", DEFAULT_CLIP_MODEL),
        clip_pretrained=getattr(args, "clip_pretrained", DEFAULT_CLIP_PRETRAINED),
        device=getattr(args, "device", None),
        batch_size=getattr(args, "batch_size", DEFAULT_BATCH_SIZE),
        max_assets_per_moment=getattr(args, "max_assets_per_moment", DEFAULT_MAX_ASSETS_PER_MOMENT),
        crop_padding=getattr(args, "crop_padding", 0.08),
        timeline_stride_seconds=getattr(args, "timeline_stride_seconds", 1.0),
        max_timeline_segments=getattr(args, "max_timeline_segments", 16),
    )


def command_download(args: argparse.Namespace) -> None:
    locations = parse_csv(args.locations, DEFAULT_LOCATIONS)
    splits = parse_csv(args.splits, DEFAULT_SPLITS)
    dataset_root = resolve_dataset_root(args)
    download_from_huggingface(
        dataset_type=args.dataset_type,
        dataset_root=dataset_root,
        repo_id=args.repo_id,
        locations=locations,
        splits=splits,
        include_videos=args.include_videos,
        include_docs=args.include_docs,
    )
    print(f"Downloaded selected files into {dataset_root}")


def command_bootstrap(args: argparse.Namespace) -> None:
    if args.dataset_type == "lava":
        locations = parse_csv(args.locations, DEFAULT_LOCATIONS)
        splits = parse_csv(args.splits, DEFAULT_SPLITS)
        dataset_root = resolve_dataset_root(args)
        download_from_huggingface(
            dataset_type=args.dataset_type,
            dataset_root=dataset_root,
            repo_id=args.repo_id,
            locations=locations,
            splits=splits,
            include_videos=args.include_videos,
            include_docs=args.include_docs,
        )
    elif args.dataset_type == "personpath22":
        print("Skipping download step for PersonPath22. Expecting local annotations and videos under --dataset-root.")
    else:
        print("Skipping download step for hospital video mode. Expecting incremental ingest via `ingest-video`.")
    manifest = rebuild_runtime_bundle(_runtime_config_from_args(args))
    print(
        f"Bootstrap completed. Built {manifest['moment_count']} moments into "
        f"{manifest['output_dir']}"
    )


def command_prepare_assets(args: argparse.Namespace) -> None:
    moments = _collect_moments_for_args(args)
    assets_dir = resolve_assets_dir(args)
    extract_visual_assets(
        moments,
        output_dir=assets_dir,
        max_assets_per_moment=args.max_assets_per_moment,
        crop_padding=args.crop_padding,
        overwrite=args.overwrite,
        ffmpeg_bin=args.ffmpeg_bin,
    )
    attach_existing_visual_assets(moments, assets_dir)
    summary = count_visual_assets(moments)
    if summary["moments_with_visuals"] == 0:
        raise RuntimeError(
            "No visual assets were created. Check that the .mp4 exists and ffmpeg/cv2 can decode it."
        )
    print(
        f"Prepared visual assets for {summary['moments_with_visuals']} moments "
        f"({summary['frame_count']} frames, {summary['crop_count']} crops) in {assets_dir}"
    )


def command_build(args: argparse.Namespace) -> None:
    enable_sparse, enable_dense, enable_clip = resolve_profile(args.profile, args)
    moments = _collect_moments_for_args(args)
    assets_dir = resolve_assets_dir(args)
    videos_available = any(Path(moment.video_path).exists() for moment in moments)

    if enable_clip or args.auto_enrich:
        attach_existing_visual_assets(moments, assets_dir)
        missing_visuals = not any(moment.frame_paths or moment.crop_paths for moment in moments)
        if videos_available and (args.prepare_assets or missing_visuals):
            extract_visual_assets(
                moments,
                output_dir=assets_dir,
                max_assets_per_moment=args.max_assets_per_moment,
                crop_padding=args.crop_padding,
                overwrite=False,
                ffmpeg_bin=args.ffmpeg_bin,
            )
            attach_existing_visual_assets(moments, assets_dir)
        visual_summary = count_visual_assets(moments)
        if enable_clip:
            if visual_summary["moments_with_visuals"] == 0:
                print(
                    "Warning: CLIP branch disabled because no visual assets were found after asset preparation."
                )
            elif args.prepare_assets and not videos_available:
                raise FileNotFoundError(
                    "CLIP assets requested but videos are missing. Re-run download with --include-videos."
                )

    if args.auto_enrich:
        attach_existing_visual_assets(moments, assets_dir)
        summary = infer_and_write_enrichment(
            moments=moments,
            dataset_root=resolve_dataset_root(args),
            clip_model_name=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            device=args.device,
            batch_size=args.batch_size,
        )
        moments = apply_enrichment(moments, resolve_dataset_root(args))
        print(
            f"Inferred enrichment: {summary['attribute_count']} attribute records and "
            f"{summary['scene_count']} scene records."
        )

    bundle_path = build_search_bundle(
        moments=moments,
        output_dir=resolve_output_dir(args),
        enable_sparse=enable_sparse,
        enable_dense=enable_dense,
        enable_clip=enable_clip,
        sentence_model_name=resolve_sentence_model_name(args),
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=args.device,
        batch_size=args.batch_size,
    )

    print(f"Indexed {len(moments)} moments")
    print(f"Bundle saved to {bundle_path}")


def command_search(args: argparse.Namespace) -> None:
    results = search_index(
        Path(args.output_dir),
        query_text=args.query_text,
        query_image_path=args.image_path,
        top_k=args.top_k,
        weights=parse_weights(args.weights),
        search_mode=args.search_mode,
        use_qwen_parser=args.use_qwen_parser,
        use_qwen_reranker=args.use_qwen_reranker,
        qwen_rerank_limit=args.qwen_rerank_limit,
        qwen_device=args.qwen_device,
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    for rank, item in enumerate(results, start=1):
        frame_info = item.get("frame_info", {})
        print(
            f"[{rank}] score={item['score']:.4f} "
            f"location={item['location']} split={item['split']} "
            f"preview_frame={frame_info.get('preview_frame_idx', item['start_frame'])} "
            f"seconds={item['start_second']:.2f}-{item['end_second']:.2f} "
            f"caption={item.get('caption_text', 'no caption')} "
            f"frame={item.get('frame_path')} "
            f"crop={item.get('crop_path')} "
            f"video={item['video_path']}"
        )


def command_clip(args: argparse.Namespace) -> None:
    results = search_index(
        Path(args.output_dir),
        query_text=args.query_text,
        query_image_path=args.image_path,
        top_k=max(args.rank, args.top_k),
        weights=parse_weights(args.weights),
        search_mode=args.search_mode,
        use_qwen_parser=args.use_qwen_parser,
        use_qwen_reranker=args.use_qwen_reranker,
        qwen_rerank_limit=args.qwen_rerank_limit,
        qwen_device=args.qwen_device,
    )
    if len(results) < args.rank:
        raise ValueError("Requested rank exceeds number of search results.")

    chosen = results[args.rank - 1]
    answer_window = chosen.get("answer_window") or {}
    if answer_window:
        start_second = max(float(answer_window.get("start_second", chosen["start_second"])) - args.padding, 0.0)
        duration = max(float(answer_window.get("duration_seconds", args.duration)) + args.padding, 0.1)
    else:
        start_second = max(chosen["start_second"] - args.padding, 0.0)
        duration = args.duration
    clip_path = extract_clip(
        video_path=Path(chosen["video_path"]),
        output_path=Path(args.output),
        start_second=start_second,
        duration_seconds=duration,
        ffmpeg_bin=args.ffmpeg_bin,
    )
    print(f"Clip exported to {clip_path}")


def command_serve_api(args: argparse.Namespace) -> None:
    from .api import create_app

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to run the API server.") from exc

    app = create_app(args.output_dir)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


def command_rebuild(args: argparse.Namespace) -> None:
    manifest = rebuild_runtime_bundle(_runtime_config_from_args(args))
    print(
        f"Rebuilt bundle with {manifest['moment_count']} moments at "
        f"{manifest['output_dir']}"
    )


def command_enrich(args: argparse.Namespace) -> None:
    moments = _collect_moments_for_args(args)
    assets_dir = resolve_assets_dir(args)
    attach_existing_visual_assets(moments, assets_dir)
    videos_available = any(Path(moment.video_path).exists() for moment in moments)
    missing_visuals = not any(moment.frame_paths or moment.crop_paths for moment in moments)
    if videos_available and (args.prepare_assets or missing_visuals):
        extract_visual_assets(
            moments,
            output_dir=assets_dir,
            max_assets_per_moment=args.max_assets_per_moment,
            crop_padding=args.crop_padding,
            overwrite=False,
            ffmpeg_bin=args.ffmpeg_bin,
        )
        attach_existing_visual_assets(moments, assets_dir)

    summary = infer_and_write_enrichment(
        moments=moments,
        dataset_root=resolve_dataset_root(args),
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(
        f"Wrote enrichment sidecars to {resolve_dataset_root(args) / 'enrichment'} "
        f"({summary['attribute_count']} attribute records, {summary['scene_count']} scene records)"
    )


def command_watch_index(args: argparse.Namespace) -> None:
    watch_and_rebuild(
        _runtime_config_from_args(args),
        poll_seconds=args.poll_seconds,
        run_once=args.once,
    )


def command_ingest_video(args: argparse.Namespace) -> None:
    ingest_summary = ingest_hospital_video(
        source_path=Path(args.source_path),
        config=_hospital_ingest_config_from_args(args),
        camera_id=args.camera_id,
        recorded_start=args.start_time,
    )
    manifest = rebuild_runtime_bundle(_runtime_config_from_args(args))
    print(
        f"Ingested {ingest_summary['video_id']} with {ingest_summary['person_count']} person tracks. "
        f"Bundle now contains {manifest['moment_count']} moments at {manifest['output_dir']}."
    )
    if ingest_summary["evicted_video_ids"]:
        print(f"Evicted oldest videos: {', '.join(ingest_summary['evicted_video_ids'])}")
    print(f"Encoded video: {ingest_summary['encoded_path']}")
    print(f"Metadata: {ingest_summary['metadata_path']}")


def command_bootstrap_nvidia_hospital(args: argparse.Namespace) -> None:
    summary = bootstrap_nvidia_hospital_dataset(
        nvidia_root=Path(args.nvidia_root),
        config=_hospital_ingest_config_from_args(args),
        recorded_start=args.start_time,
        limit=args.limit,
    )
    manifest = rebuild_runtime_bundle(_runtime_config_from_args(args))
    print(
        f"Prepared {summary['processed_videos']} NVIDIA hospital videos with detailed person metadata. "
        f"Bundle now contains {manifest['moment_count']} moments at {manifest['output_dir']}."
    )


def command_demo(args: argparse.Namespace) -> None:
    app_path = Path(__file__).with_name("streamlit_app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "download":
        command_download(args)
    elif args.command == "bootstrap":
        command_bootstrap(args)
    elif args.command == "prepare-assets":
        command_prepare_assets(args)
    elif args.command == "build":
        command_build(args)
    elif args.command == "search":
        command_search(args)
    elif args.command == "clip":
        command_clip(args)
    elif args.command == "serve-api":
        command_serve_api(args)
    elif args.command == "rebuild":
        command_rebuild(args)
    elif args.command == "enrich":
        command_enrich(args)
    elif args.command == "watch-index":
        command_watch_index(args)
    elif args.command == "ingest-video":
        command_ingest_video(args)
    elif args.command == "bootstrap-nvidia-hospital":
        command_bootstrap_nvidia_hospital(args)
    elif args.command == "demo":
        command_demo(args)
    else:
        parser.error(f"Unknown command: {args.command}")
