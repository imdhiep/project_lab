import json
import shutil
import unittest
from pathlib import Path

from lava_search.dataset import build_moments_from_label
from lava_search.video_tools import attach_existing_visual_assets, select_visual_path


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
        self.assertEqual(select_visual_path(moments[0]), str(assets_dir / "crop_000000.jpg"))

        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
