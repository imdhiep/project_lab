from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set up the full person-query stack scaffold for PersonPath22 + PA-100K + BDD100K."
    )
    parser.add_argument(
        "--download-personpath22",
        action="store_true",
        help="Download the public PersonPath22 dataset before building.",
    )
    parser.add_argument(
        "--personpath22-mode",
        choices=("annotations", "full"),
        default="full",
        help="Which PersonPath22 payload to download when --download-personpath22 is enabled.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded PersonPath22 archives.",
    )
    parser.add_argument(
        "--build-now",
        action="store_true",
        help="Run the end-to-end PersonPath22 build with auto enrichment after setup.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override passed through to the build command.",
    )
    return parser


def main() -> int:
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from surveillance_search.person_query_stack import ensure_person_query_stack_layout, write_stack_configs

    layout = ensure_person_query_stack_layout(project_root)
    config_paths = write_stack_configs(project_root)

    print("Person-query stack layout is ready.")
    print(f"  PersonPath22 root: {layout['personpath22_root']}")
    print(f"  PA-100K root:      {layout['pa100k_root']}")
    print(f"  BDD100K root:      {layout['bdd100k_root']}")
    print(f"  Config root:       {layout['configs_root']}")

    if args := build_parser().parse_args():
        if args.download_personpath22:
            download_script = project_root / "scripts" / "download_personpath22_once.py"
            command = [
                sys.executable,
                str(download_script),
                "--output-dir",
                str(layout["personpath22_root"]),
                "--mode",
                args.personpath22_mode,
            ]
            if args.extract:
                command.append("--extract")
            print("\nDownloading PersonPath22...")
            subprocess.run(command, check=True)

        if args.build_now:
            build_command = [
                sys.executable,
                "-m",
                "surveillance_search",
                "build",
                "--dataset-type",
                "personpath22",
                "--dataset-root",
                str(layout["personpath22_root"]),
                "--profile",
                "strongest",
                "--auto-enrich",
            ]
            if args.device:
                build_command.extend(["--device", args.device])
            print("\nBuilding end-to-end bundle...")
            subprocess.run(build_command, check=True, cwd=project_root)

    print("\nScaffold files written:")
    for name, path in config_paths.items():
        print(f"  {name}: {path}")

    print("\nRecommended next steps:")
    print("  1. Place PA-100K under data/pa100k and BDD100K under data/bdd100k.")
    print("  2. Normalize them with:")
    print("     python scripts/prepare_pa100k_dataset.py")
    print("     python scripts/prepare_bdd100k_dataset.py")
    print("  3. Register baseline metadata with:")
    print("     python scripts/train_attribute_baseline.py")
    print("     python scripts/train_scene_baseline.py")
    print("  4. Build or rebuild the searchable bundle with:")
    print("     python -m surveillance_search build --dataset-type personpath22 --profile strongest --auto-enrich")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
