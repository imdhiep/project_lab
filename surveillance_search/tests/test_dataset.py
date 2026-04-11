import json
import shutil
import unittest
from pathlib import Path

from surveillance_search.dataset import (
    build_moments_from_label,
    build_moments_from_personpath22_gt,
    build_moments_from_personpath22_json,
    collect_moments,
    iter_label_files,
)
from surveillance_search.video_tools import (
    attach_existing_visual_assets,
    build_frame_info,
    select_crop_path,
    select_frame_path,
    select_visual_path,
    summarize_captions,
)


class DatasetParsingTests(unittest.TestCase):
    def test_group_track_into_single_moment(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "case1"
        shutil.rmtree(case_root, ignore_errors=True)

        label_path = case_root / "amsterdam" / "test" / "label.json"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            [
                {
                    "caption": ["red sedan"],
                    "left": 10,
                    "top": 20,
                    "right": 30,
                    "bottom": 40,
                    "track_id": 7,
                }
            ],
            [
                {
                    "caption": ["red car"],
                    "left": 11,
                    "top": 21,
                    "right": 31,
                    "bottom": 41,
                    "track_id": 7,
                }
            ],
        ]
        label_path.write_text(json.dumps(payload), encoding="utf-8")

        moments = build_moments_from_label(label_path, fps=10.0, group_by_track=True)

        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0].start_frame, 0)
        self.assertEqual(moments[0].end_frame, 1)
        self.assertEqual(moments[0].captions, ["red car", "red sedan"])
        self.assertAlmostEqual(moments[0].end_second, 0.2)
        self.assertIn("red", moments[0].keywords)
        self.assertIsNone(select_visual_path(moments[0]))

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_build_moments_from_personpath22_json(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "personpath22-json"
        shutil.rmtree(case_root, ignore_errors=True)

        annotation_dir = case_root / "annotations"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        video_dir = case_root / "raw_data"
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / "cam_01.mp4").write_bytes(b"fake-video")
        annotation_path = annotation_dir / "anno_visible_2022.json"
        annotation_path.write_text(
            json.dumps(
                {
                    "videos": [
                        {
                            "id": 1,
                            "file_name": "cam_01.mp4",
                            "location": "station",
                            "split": "train",
                        }
                    ],
                    "images": [
                        {"id": 100, "video_id": 1, "frame_index": 0},
                        {"id": 101, "video_id": 1, "frame_index": 5},
                    ],
                    "annotations": [
                        {"id": 1, "image_id": 100, "track_id": 7, "bbox": [10, 20, 30, 40], "visibility": 0.95, "category_id": 1},
                        {"id": 2, "image_id": 101, "track_id": 7, "bbox": [12, 24, 30, 40], "visibility": 0.4, "category_id": 1},
                    ],
                    "categories": [{"id": 1, "name": "person"}],
                }
            ),
            encoding="utf-8",
        )

        moments = build_moments_from_personpath22_json(annotation_path, dataset_root=case_root, fps=10.0, group_by_track=True)

        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0].location, "station")
        self.assertEqual(moments[0].split, "train")
        self.assertEqual(moments[0].track_id, "7")
        self.assertEqual(moments[0].start_frame, 0)
        self.assertEqual(moments[0].end_frame, 5)
        self.assertIn("person", moments[0].keywords)
        self.assertIn("person", moments[0].text)
        self.assertEqual(Path(moments[0].video_path).name, "cam_01.mp4")

        filtered = collect_moments(
            case_root,
            fps=10.0,
            group_by_track=True,
            dataset_type="personpath22",
            locations=["station"],
            splits=["train"],
        )
        self.assertEqual(len(filtered), 1)

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_build_moments_from_personpath22_gt(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "personpath22-gt"
        shutil.rmtree(case_root, ignore_errors=True)

        gt_dir = case_root / "train" / "cam_02.mp4" / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        (case_root / "raw_data").mkdir(parents=True, exist_ok=True)
        (case_root / "raw_data" / "cam_02.mp4").write_bytes(b"fake-video")
        gt_path = gt_dir / "gt.txt"
        gt_path.write_text("1,3,10,20,30,40,1,1,1\n2,3,11,21,30,40,1,1,1\n", encoding="utf-8")

        moments = build_moments_from_personpath22_gt(gt_path, dataset_root=case_root, fps=10.0, group_by_track=True)

        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0].location, "cam_02")
        self.assertEqual(moments[0].split, "train")
        self.assertEqual(moments[0].track_id, "3")
        self.assertEqual(moments[0].start_frame, 0)
        self.assertEqual(moments[0].end_frame, 1)
        self.assertIn("person", moments[0].keywords)

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_build_moments_from_personpath22_per_video_json(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "personpath22-per-video-json"
        shutil.rmtree(case_root, ignore_errors=True)

        annotation_dir = case_root / "annotations" / "anno_visible_2022"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        video_dir = case_root / "raw_data"
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / "uid_vid_00000.mp4").write_bytes(b"fake-video")
        annotation_path = annotation_dir / "uid_vid_00000.mp4.json"
        annotation_path.write_text(
            json.dumps(
                {
                    "entities": [
                        {"bb": [10, 20, 30, 40], "id": 7, "blob": {"frame_idx": 0}, "confidence": 1.0},
                        {"bb": [12, 24, 30, 40], "id": 7, "blob": {"frame_idx": 5}, "confidence": 1.0},
                    ],
                    "metadata": {
                        "data_path": "uid_vid_00000.mp4",
                        "fps": 25.0,
                        "number_of_frames": 100,
                    },
                }
            ),
            encoding="utf-8",
        )

        discovered = list(iter_label_files(case_root, dataset_type="personpath22"))
        self.assertEqual(discovered, [annotation_path])

        moments = build_moments_from_personpath22_json(annotation_path, dataset_root=case_root, fps=10.0, group_by_track=True)

        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0].location, "uid_vid_00000")
        self.assertEqual(moments[0].split, "train")
        self.assertEqual(moments[0].track_id, "7")
        self.assertEqual(moments[0].start_frame, 0)
        self.assertEqual(moments[0].end_frame, 5)
        self.assertEqual(Path(moments[0].video_path).name, "uid_vid_00000.mp4")

        collected = collect_moments(case_root, dataset_type="personpath22")
        self.assertEqual(len(collected), 1)

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_attach_existing_visual_assets(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "case-assets"
        shutil.rmtree(case_root, ignore_errors=True)

        label_path = case_root / "amsterdam" / "test" / "label.json"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            json.dumps(
                [
                    [
                        {
                            "caption": ["red sedan"],
                            "left": 10,
                            "top": 20,
                            "right": 30,
                            "bottom": 40,
                            "track_id": 7,
                        }
                    ]
                ]
            ),
            encoding="utf-8",
        )

        moments = build_moments_from_label(label_path, fps=10.0, group_by_track=True)
        assets_dir = case_root / "assets" / "amsterdam_test_track_7"
        assets_dir.mkdir(parents=True, exist_ok=True)
        (assets_dir / "frame_000000.jpg").write_bytes(b"frame")
        (assets_dir / "crop_000000.jpg").write_bytes(b"crop")

        attach_existing_visual_assets(moments, case_root / "assets")

        self.assertTrue(moments[0].frame_paths)
        self.assertTrue(moments[0].crop_paths)
        self.assertEqual(select_frame_path(moments[0]), str(assets_dir / "frame_000000.jpg"))
        self.assertEqual(select_crop_path(moments[0]), str(assets_dir / "crop_000000.jpg"))
        self.assertEqual(select_visual_path(moments[0]), str(assets_dir / "frame_000000.jpg"))
        self.assertEqual(select_visual_path(moments[0], preference="crop"), str(assets_dir / "crop_000000.jpg"))

        frame_info = build_frame_info(moments[0])
        self.assertEqual(frame_info["preview_frame_idx"], 0)
        self.assertEqual(frame_info["location"], "amsterdam")
        self.assertEqual(summarize_captions(moments[0]), "red sedan")

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_collect_moments_applies_attribute_and_scene_enrichment(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "personpath22-enrichment"
        shutil.rmtree(case_root, ignore_errors=True)

        annotation_dir = case_root / "annotations" / "anno_visible_2022"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        video_dir = case_root / "raw_data"
        video_dir.mkdir(parents=True, exist_ok=True)
        enrichment_dir = case_root / "enrichment"
        enrichment_dir.mkdir(parents=True, exist_ok=True)

        (video_dir / "uid_vid_00000.mp4").write_bytes(b"fake-video")
        annotation_path = annotation_dir / "uid_vid_00000.mp4.json"
        annotation_path.write_text(
            json.dumps(
                {
                    "entities": [
                        {"bb": [10, 20, 30, 40], "id": 7, "blob": {"frame_idx": 0}, "confidence": 1.0},
                    ],
                    "metadata": {
                        "data_path": "uid_vid_00000.mp4",
                        "fps": 25.0,
                        "number_of_frames": 100,
                    },
                }
            ),
            encoding="utf-8",
        )
        (enrichment_dir / "person_attributes.json").write_text(
            json.dumps(
                [
                    {
                        "video_name": "uid_vid_00000.mp4",
                        "track_id": "7",
                        "attributes": {"top_color": "red", "top_type": "shirt", "backpack": True},
                    }
                ]
            ),
            encoding="utf-8",
        )
        (enrichment_dir / "scene_context.json").write_text(
            json.dumps(
                [
                    {
                        "video_name": "uid_vid_00000.mp4",
                        "track_id": "7",
                        "scene": {"near_road": True},
                    }
                ]
            ),
            encoding="utf-8",
        )

        moments = collect_moments(case_root, dataset_type="personpath22")
        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0].attribute_evidence["top_color"], "red")
        self.assertTrue(moments[0].scene_evidence["near_road"])
        self.assertIn("red", moments[0].keywords)
        self.assertIn("road", moments[0].keywords)

        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
