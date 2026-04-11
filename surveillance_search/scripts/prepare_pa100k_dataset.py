from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize PA-100K into a lightweight JSONL-like manifest.")
    parser.add_argument(
        "--dataset-root",
        default=str(Path(__file__).resolve().parents[1] / "data" / "pa100k"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "prepared" / "pa100k"),
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Optional override for the directory containing PA-100K images.",
    )
    parser.add_argument(
        "--annotation-file",
        default=None,
        help="Optional override for the PA-100K annotation JSON/CSV file.",
    )
    return parser


def _guess_images_dir(dataset_root: Path) -> Path:
    for candidate in ("images", "release_data", "imgs"):
        path = dataset_root / candidate
        if path.exists():
            return path
    return dataset_root


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve() if args.images_dir else _guess_images_dir(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "PA-100K",
        "raw_root": str(dataset_root),
        "images_dir": str(images_dir),
        "annotation_file": str(Path(args.annotation_file).expanduser().resolve()) if args.annotation_file else None,
        "status": "prepared_scaffold",
        "notes": [
            "Place the official PA-100K images and annotations under the raw root.",
            "This scaffold records paths and gives the training/inference stack a stable prepared root.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"Wrote PA-100K prepared manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
