from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register the PA-100K attribute baseline used by the person-query stack.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "person_query_stack" / "attribute_model.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_artifact = Path(config["output_artifact"]).expanduser().resolve()
    output_artifact.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "name": config["name"],
        "strategy": config["strategy"],
        "supervision_dataset": config["supervision_dataset"],
        "prepared_dataset_root": config["prepared_dataset_root"],
        "labels_of_interest": config["labels_of_interest"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "status": "baseline_registered",
        "notes": config.get("notes", []),
    }
    with output_artifact.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
    print(f"Wrote attribute baseline artifact to {output_artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
