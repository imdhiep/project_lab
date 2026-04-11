from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize BDD100K drivable-area assets into a stable prepared manifest.")
    parser.add_argument(
        "--dataset-root",
        default=str(Path(__file__).resolve().parents[1] / "data" / "bdd100k"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "prepared" / "bdd100k"),
    )
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--labels-dir", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images_dir).expanduser().resolve() if args.images_dir else dataset_root / "images"
    labels_dir = Path(args.labels_dir).expanduser().resolve() if args.labels_dir else dataset_root / "labels"
    manifest = {
        "dataset": "BDD100K",
        "raw_root": str(dataset_root),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "status": "prepared_scaffold",
        "notes": [
            "Store BDD100K images and drivable/segmentation labels under the raw root.",
            "This scaffold gives the scene-evidence pipeline a stable prepared root.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"Wrote BDD100K prepared manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
