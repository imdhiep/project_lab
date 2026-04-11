from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

S3_ROOT = "s3://tracking-dataset-eccv-2022/dataset"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "personpath22"
ANNOTATION_OBJECTS = (
    ("annotation/anno_visible.zip", "annotations"),
    ("annotation/anno_amodal.zip", "annotations"),
    ("annotation/splits.json", "annotations"),
)
VIDEO_OBJECTS = (
    ("raw_data/videos.zip", "raw_data"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-shot downloader for the public PersonPath22 dataset."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where PersonPath22 files should be stored.",
    )
    parser.add_argument(
        "--mode",
        choices=("annotations", "full"),
        default="annotations",
        help="Download only annotations, or annotations plus video archive.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded zip files after download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist locally.",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep zip archives after extraction.",
    )
    return parser


def require_aws_cli() -> str:
    aws_bin = shutil.which("aws")
    if not aws_bin:
        raise RuntimeError(
            "AWS CLI is required. Install it first or run this script on a machine that has `aws` in PATH."
        )
    return aws_bin


def iter_objects(mode: str):
    for item in ANNOTATION_OBJECTS:
        yield item
    if mode == "full":
        for item in VIDEO_OBJECTS:
            yield item


def download_object(aws_bin: str, s3_key: str, destination_dir: Path, force: bool) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / Path(s3_key).name
    if destination_path.exists() and not force:
        print(f"Skipping existing file: {destination_path}")
        return destination_path

    command = [
        aws_bin,
        "s3",
        "cp",
        "--no-sign-request",
        f"{S3_ROOT}/{s3_key}",
        str(destination_path),
    ]
    print(f"Downloading {s3_key} -> {destination_path}")
    subprocess.run(command, check=True)
    return destination_path


def extract_zip(zip_path: Path, destination_dir: Path, keep_zip: bool) -> None:
    print(f"Extracting {zip_path} -> {destination_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as handle:
        handle.extractall(destination_dir)
    if not keep_zip:
        zip_path.unlink(missing_ok=True)


def main() -> int:
    args = build_parser().parse_args()
    aws_bin = require_aws_cli()
    output_dir = Path(args.output_dir).expanduser().resolve()

    downloaded_paths: list[tuple[Path, Path]] = []
    for s3_key, relative_dir in iter_objects(args.mode):
        destination_dir = output_dir / relative_dir
        path = download_object(aws_bin, s3_key, destination_dir, args.force)
        downloaded_paths.append((path, destination_dir))

    if args.extract:
        for path, destination_dir in downloaded_paths:
            if path.suffix.lower() != ".zip":
                continue
            extract_zip(path, destination_dir, args.keep_zips)

    print("\nDone.")
    print(f"Dataset root: {output_dir}")
    print("Recommended next steps:")
    print(
        "  1. If you downloaded annotations only, make sure your videos are available under "
        f"{output_dir / 'raw_data'} before building."
    )
    print(
        "  2. Build the bundle with:\n"
        "     python -m surveillance_search build --dataset-type personpath22 "
        f"--dataset-root {output_dir} --profile strongest"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
