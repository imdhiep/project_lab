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
    DEFAULT_FPS,
    DEFAULT_LOCATIONS,
    DEFAULT_MAX_ASSETS_PER_MOMENT,
    DEFAULT_PROFILE,
    DEFAULT_REPO_ID,
    DEFAULT_SENTENCE_MODEL,
    DEFAULT_SPLITS,
    DEFAULT_TOP_K,
    default_bundle_root,
    default_data_root,
    default_visual_root,
)
from .dataset import collect_moments, download_from_huggingface
from .indexer import build_search_bundle, search_index
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full-featured LAVA multimodal search engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download labels, videos and docs from Hugging Face")
    download_parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    download_parser.add_argument("--dataset-root", default=str(default_data_root()))
    download_parser.add_argument("--locations", default=None, help="Comma-separated list")
    download_parser.add_argument("--splits", default=None, help="Comma-separated list")
    download_parser.add_argument("--include-videos", action="store_true")
    download_parser.add_argument("--include-docs", action="store_true")

    bootstrap_parser = subparsers.add_parser("bootstrap", help="Download selected data and build a production-ready bundle")
    bootstrap_parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    bootstrap_parser.add_argument("--dataset-root", default=str(default_data_root()))
    bootstrap_parser.add_argument("--output-dir", default=str(default_bundle_root()))
    bootstrap_parser.add_argument("--assets-dir", default=str(default_visual_root()))
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
    bootstrap_parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    bootstrap_parser.add_argument("--clip-pretrained", default=DEFAULT_CLIP_PRETRAINED)
    bootstrap_parser.add_argument("--device", default=None)
    bootstrap_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    bootstrap_parser.add_argument("--no-sparse", action="store_true")
    bootstrap_parser.add_argument("--no-dense", action="store_true")
    bootstrap_parser.add_argument("--no-clip", action="store_true")

    prepare_parser = subparsers.add_parser("prepare-assets", help="Extract frames and crops for CLIP indexing")
    prepare_parser.add_argument("--dataset-root", default=str(default_data_root()))
    prepare_parser.add_argument("--assets-dir", default=str(default_visual_root()))
    prepare_parser.add_argument("--locations", default=None, help="Comma-separated list")
    prepare_parser.add_argument("--splits", default=None, help="Comma-separated list")
    prepare_parser.add_argument("--group-mode", choices=("track", "frame"), default="track")
    prepare_parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    prepare_parser.add_argument("--max-assets-per-moment", type=int, default=DEFAULT_MAX_ASSETS_PER_MOMENT)
    prepare_parser.add_argument("--crop-padding", type=float, default=0.08)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.add_argument("--ffmpeg-bin", default="ffmpeg")

    build_parser_cmd = subparsers.add_parser("build", help="Build a full multimodal search bundle")
    build_parser_cmd.add_argument("--dataset-root", default=str(default_data_root()))
    build_parser_cmd.add_argument("--output-dir", default=str(default_bundle_root()))
    build_parser_cmd.add_argument("--assets-dir", default=str(default_visual_root()))
    build_parser_cmd.add_argument("--locations", default=None, help="Comma-separated list")
    build_parser_cmd.add_argument("--splits", default=None, help="Comma-separated list")
    build_parser_cmd.add_argument("--group-mode", choices=("track", "frame"), default="track")
    build_parser_cmd.add_argument("--fps", type=float, default=DEFAULT_FPS)
    build_parser_cmd.add_argument("--profile", choices=("strongest", "balanced", "text-only", "lite"), default=DEFAULT_PROFILE)
    build_parser_cmd.add_argument("--prepare-assets", action="store_true")
    build_parser_cmd.add_argument("--max-assets-per-moment", type=int, default=DEFAULT_MAX_ASSETS_PER_MOMENT)
    build_parser_cmd.add_argument("--crop-padding", type=float, default=0.08)
    build_parser_cmd.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
    build_parser_cmd.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    build_parser_cmd.add_argument("--clip-pretrained", default=DEFAULT_CLIP_PRETRAINED)
    build_parser_cmd.add_argument("--device", default=None)
    build_parser_cmd.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    build_parser_cmd.add_argument("--ffmpeg-bin", default="ffmpeg")
    build_parser_cmd.add_argument("--no-sparse", action="store_true")
    build_parser_cmd.add_argument("--no-dense", action="store_true")
    build_parser_cmd.add_argument("--no-clip", action="store_true")

    search_parser = subparsers.add_parser("search", help="Search by text, image, or both")
    search_parser.add_argument("--output-dir", default=str(default_bundle_root()))
    search_parser.add_argument("--query-text", default=None)
    search_parser.add_argument("--image-path", default=None)
    search_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    search_parser.add_argument("--search-mode", choices=("default", "person"), default="default")
    search_parser.add_argument("--weights", default=None, help="Example: sparse=0.1,dense=0.4,clip_text=0.2,clip_image=0.3")
    search_parser.add_argument("--json", action="store_true")

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

    api_parser = subparsers.add_parser("serve-api", help="Run FastAPI service")
    api_parser.add_argument("--output-dir", default=str(default_bundle_root()))
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8000)
    api_parser.add_argument("--reload", action="store_true")

    rebuild_parser = subparsers.add_parser("rebuild", help="Rebuild the bundle from current dataset files")
    rebuild_parser.add_argument("--dataset-root", default=str(default_data_root()))
    rebuild_parser.add_argument("--output-dir", default=str(default_bundle_root()))
    rebuild_parser.add_argument("--assets-dir", default=str(default_visual_root()))
    rebuild_parser.add_argument("--locations", default=None, help="Comma-separated list")
    rebuild_parser.add_argument("--splits", default=None, help="Comma-separated list")
    rebuild_parser.add_argument("--group-mode", choices=("track", "frame"), default="track")
    rebuild_parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    rebuild_parser.add_argument("--profile", choices=("strongest", "balanced", "text-only", "lite"), default=DEFAULT_PROFILE)
    rebuild_parser.add_argument("--max-assets-per-moment", type=int, default=DEFAULT_MAX_ASSETS_PER_MOMENT)
    rebuild_parser.add_argument("--crop-padding", type=float, default=0.08)
    rebuild_parser.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
    rebuild_parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    rebuild_parser.add_argument("--clip-pretrained", default=DEFAULT_CLIP_PRETRAINED)
    rebuild_parser.add_argument("--device", default=None)
    rebuild_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    rebuild_parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    rebuild_parser.add_argument("--no-sparse", action="store_true")
    rebuild_parser.add_argument("--no-dense", action="store_true")
    rebuild_parser.add_argument("--no-clip", action="store_true")

    watch_parser = subparsers.add_parser("watch-index", help="Poll the dataset directory and rebuild when files change")
    watch_parser.add_argument("--dataset-root", default=str(default_data_root()))
    watch_parser.add_argument("--output-dir", default=str(default_bundle_root()))
    watch_parser.add_argument("--assets-dir", default=str(default_visual_root()))
    watch_parser.add_argument("--locations", default=None, help="Comma-separated list")
    watch_parser.add_argument("--splits", default=None, help="Comma-separated list")
    watch_parser.add_argument("--group-mode", choices=("track", "frame"), default="track")
    watch_parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    watch_parser.add_argument("--profile", choices=("strongest", "balanced", "text-only", "lite"), default=DEFAULT_PROFILE)
    watch_parser.add_argument("--max-assets-per-moment", type=int, default=DEFAULT_MAX_ASSETS_PER_MOMENT)
    watch_parser.add_argument("--crop-padding", type=float, default=0.08)
    watch_parser.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
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

    demo_parser = subparsers.add_parser("demo", help="Launch Streamlit demo UI")
    demo_parser.add_argument("--port", type=int, default=8501)

    return parser


def _collect_moments_for_args(args: argparse.Namespace):
    return collect_moments(
        dataset_root=Path(args.dataset_root),
        fps=args.fps,
        group_by_track=args.group_mode == "track",
        locations=parse_csv(args.locations),
        splits=parse_csv(args.splits),
    )


def _runtime_config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    enable_sparse, enable_dense, enable_clip = resolve_profile(args.profile, args)
    return RuntimeConfig(
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        assets_dir=Path(args.assets_dir),
        locations=parse_csv(args.locations),
        splits=parse_csv(args.splits),
        group_by_track=args.group_mode == "track",
        fps=args.fps,
        enable_sparse=enable_sparse,
        enable_dense=enable_dense,
        enable_clip=enable_clip,
        sentence_model_name=args.sentence_model,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=args.device,
        batch_size=args.batch_size,
        max_assets_per_moment=args.max_assets_per_moment,
        crop_padding=args.crop_padding,
        ffmpeg_bin=args.ffmpeg_bin,
    )


def command_download(args: argparse.Namespace) -> None:
    locations = parse_csv(args.locations, DEFAULT_LOCATIONS)
    splits = parse_csv(args.splits, DEFAULT_SPLITS)
    dataset_root = Path(args.dataset_root)
    download_from_huggingface(
        dataset_root=dataset_root,
        repo_id=args.repo_id,
        locations=locations,
        splits=splits,
        include_videos=args.include_videos,
        include_docs=args.include_docs,
    )
    print(f"Downloaded selected files into {dataset_root}")


def command_bootstrap(args: argparse.Namespace) -> None:
    locations = parse_csv(args.locations, DEFAULT_LOCATIONS)
    splits = parse_csv(args.splits, DEFAULT_SPLITS)
    dataset_root = Path(args.dataset_root)
    download_from_huggingface(
        dataset_root=dataset_root,
        repo_id=args.repo_id,
        locations=locations,
        splits=splits,
        include_videos=args.include_videos,
        include_docs=args.include_docs,
    )
    manifest = rebuild_runtime_bundle(_runtime_config_from_args(args))
    print(
        f"Bootstrap completed. Built {manifest['moment_count']} moments into "
        f"{manifest['output_dir']}"
    )


def command_prepare_assets(args: argparse.Namespace) -> None:
    moments = _collect_moments_for_args(args)
    assets_dir = Path(args.assets_dir)
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
    assets_dir = Path(args.assets_dir)

    if enable_clip:
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
        visual_summary = count_visual_assets(moments)
        if visual_summary["moments_with_visuals"] == 0:
            print(
                "Warning: CLIP branch disabled because no visual assets were found after asset preparation."
            )
        elif args.prepare_assets:
            raise FileNotFoundError(
                "CLIP assets requested but videos are missing. Re-run download with --include-videos."
            )

    bundle_path = build_search_bundle(
        moments=moments,
        output_dir=Path(args.output_dir),
        enable_sparse=enable_sparse,
        enable_dense=enable_dense,
        enable_clip=enable_clip,
        sentence_model_name=args.sentence_model,
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
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    for rank, item in enumerate(results, start=1):
        captions = ", ".join(item["captions"]) if item["captions"] else "no caption"
        print(
            f"[{rank}] score={item['score']:.4f} "
            f"location={item['location']} split={item['split']} "
            f"frames={item['start_frame']}-{item['end_frame']} "
            f"seconds={item['start_second']:.2f}-{item['end_second']:.2f} "
            f"captions={captions} "
            f"visual={item.get('visual_path')} "
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
    )
    if len(results) < args.rank:
        raise ValueError("Requested rank exceeds number of search results.")

    chosen = results[args.rank - 1]
    start_second = max(chosen["start_second"] - args.padding, 0.0)
    clip_path = extract_clip(
        video_path=Path(chosen["video_path"]),
        output_path=Path(args.output),
        start_second=start_second,
        duration_seconds=args.duration,
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


def command_watch_index(args: argparse.Namespace) -> None:
    watch_and_rebuild(
        _runtime_config_from_args(args),
        poll_seconds=args.poll_seconds,
        run_once=args.once,
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
    elif args.command == "watch-index":
        command_watch_index(args)
    elif args.command == "demo":
        command_demo(args)
    else:
        parser.error(f"Unknown command: {args.command}")
