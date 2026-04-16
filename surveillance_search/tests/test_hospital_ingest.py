import json
import shutil
import sqlite3
import unittest
from pathlib import Path
from unittest import mock

from surveillance_search.dataset import collect_moments
from surveillance_search.hospital_ingest import (
    HospitalIngestConfig,
    hospital_catalog_path,
    ingest_hospital_video,
    load_hospital_moments,
)
from surveillance_search.models import Moment


class HospitalIngestTests(unittest.TestCase):
    def test_ingest_video_keeps_only_latest_items_in_queue(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "hospital-ingest-queue"
        shutil.rmtree(case_root, ignore_errors=True)
        case_root.mkdir(parents=True, exist_ok=True)

        assets_dir = case_root / "assets"
        dataset_root = case_root / "dataset"
        config = HospitalIngestConfig(
            dataset_root=dataset_root,
            assets_dir=assets_dir,
            queue_size=2,
            enable_enrichment=False,
        )

        def fake_probe(video_path: Path) -> dict:
            return {
                "fps": 10.0,
                "frame_count": 100,
                "width": 320,
                "height": 240,
                "duration_seconds": 10.0,
                "file_size_bytes": video_path.stat().st_size if video_path.exists() else 1,
            }

        def fake_convert(source_path: Path, output_path: Path, ffmpeg_bin: str = "ffmpeg", preferred_encoder: str = "hevc_nvenc") -> dict:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(f"encoded::{source_path.name}".encode("utf-8"))
            return {"encoder": preferred_encoder, "path": str(output_path)}

        def fake_analyze(video_path: Path, camera_id: str, recorded_start, probe: dict, **_: object) -> list[Moment]:
            return [
                Moment(
                    id=f"{video_path.stem}:track:1",
                    location=camera_id,
                    split="live",
                    video_path=str(video_path),
                    label_path=str(video_path.with_suffix(".json")),
                    track_id="1",
                    fps=probe["fps"],
                    start_frame=0,
                    end_frame=9,
                    start_second=0.0,
                    end_second=1.0,
                    captions=["person"],
                    representative_bbox=[10, 20, 60, 120],
                    sample_frames=[0, 5, 9],
                    text=f"person in hospital camera {camera_id}",
                    keywords=["person", "hospital", camera_id],
                    camera_id=camera_id,
                    metadata={
                        "content_frames": [
                            {
                                "frame_idx": 0,
                                "second": 0.0,
                                "actual_time": "2026-01-01T00:00:00Z",
                                "bbox": [10, 20, 60, 120],
                            }
                        ]
                    },
                )
            ]

        source_paths = []
        for index in range(3):
            source_path = case_root / f"cam01_raw_{index}.mp4"
            source_path.write_bytes(f"source::{index}".encode("utf-8"))
            source_paths.append(source_path)

        with mock.patch("surveillance_search.hospital_ingest.probe_video", side_effect=fake_probe), mock.patch(
            "surveillance_search.hospital_ingest.convert_video_to_h265",
            side_effect=fake_convert,
        ), mock.patch(
            "surveillance_search.hospital_ingest.analyze_hospital_video",
            side_effect=fake_analyze,
        ), mock.patch(
            "surveillance_search.hospital_ingest.extract_visual_assets",
            return_value=[],
        ):
            ingest_hospital_video(source_paths[0], config=config, camera_id="cam01", recorded_start="2026-01-01T00:00:00Z")
            ingest_hospital_video(source_paths[1], config=config, camera_id="cam01", recorded_start="2026-01-01T00:10:00Z")
            summary = ingest_hospital_video(
                source_paths[2],
                config=config,
                camera_id="cam01",
                recorded_start="2026-01-01T00:20:00Z",
            )

        self.assertEqual(summary["queue_size"], 2)
        self.assertEqual(len(summary["evicted_video_ids"]), 1)

        metadata_files = sorted((dataset_root / "metadata").glob("*.json"))
        encoded_files = sorted((dataset_root / "encoded").glob("*.h265"))
        self.assertEqual(len(metadata_files), 2)
        self.assertEqual(len(encoded_files), 2)
        self.assertFalse((dataset_root / "metadata" / "cam01_20260101T000000Z.json").exists())

        moments = load_hospital_moments(dataset_root)
        self.assertEqual(len(moments), 2)
        self.assertTrue(all(moment.camera_id == "cam01" for moment in moments))

        with sqlite3.connect(str(hospital_catalog_path(dataset_root))) as connection:
            video_count = connection.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
            person_count = connection.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        self.assertEqual(video_count, 2)
        self.assertEqual(person_count, 2)

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_collect_moments_reads_hospital_metadata(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "hospital-collect"
        shutil.rmtree(case_root, ignore_errors=True)

        metadata_dir = case_root / "data" / "hospital" / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        encoded_dir = case_root / "data" / "hospital" / "encoded"
        encoded_dir.mkdir(parents=True, exist_ok=True)
        encoded_path = encoded_dir / "cam02_20260101T010000Z.h265"
        encoded_path.write_bytes(b"encoded")

        payload = {
            "schema_version": 1,
            "video": {
                "video_id": "cam02_20260101T010000Z",
                "camera_id": "cam02",
                "encoded_path": str(encoded_path),
            },
            "people": [
                {
                    "id": "cam02_20260101T010000Z:track:3",
                    "location": "cam02",
                    "split": "live",
                    "video_path": str(encoded_path),
                    "label_path": str(metadata_dir / "cam02_20260101T010000Z.json"),
                    "track_id": "3",
                    "fps": 25.0,
                    "start_frame": 10,
                    "end_frame": 30,
                    "start_second": 0.4,
                    "end_second": 1.24,
                    "captions": ["person"],
                    "representative_bbox": [11, 22, 44, 88],
                    "sample_frames": [10, 20, 30],
                    "text": "person in hospital surveillance video cam02",
                    "keywords": ["person", "hospital", "cam02"],
                    "camera_id": "cam02",
                    "metadata": {
                        "content_frames": [
                            {"frame_idx": 10, "second": 0.4, "actual_time": "2026-01-01T01:00:00Z", "bbox": [11, 22, 44, 88]}
                        ],
                        "description": "red shirt",
                    },
                    "attribute_evidence": {"top_color": "red", "top_type": "shirt"},
                    "scene_evidence": {},
                }
            ],
        }
        (metadata_dir / "cam02_20260101T010000Z.json").write_text(json.dumps(payload), encoding="utf-8")

        moments = collect_moments(case_root / "data" / "hospital", dataset_type="hospital")
        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0].camera_id, "cam02")
        self.assertEqual(moments[0].metadata["description"], "red shirt")
        self.assertEqual(moments[0].attribute_evidence["top_color"], "red")

        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
