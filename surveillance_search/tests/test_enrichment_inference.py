import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from surveillance_search.enrichment_inference import infer_and_write_enrichment, infer_enrichment_payloads
from surveillance_search.models import Moment


class EnrichmentInferenceTests(unittest.TestCase):
    def test_infer_enrichment_payloads_builds_attribute_and_scene_records(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "enrichment-inference"
        shutil.rmtree(case_root, ignore_errors=True)
        case_root.mkdir(parents=True, exist_ok=True)

        crop_path = case_root / "crop.jpg"
        frame_path = case_root / "frame.jpg"
        crop_path.write_bytes(b"fake")
        frame_path.write_bytes(b"fake")

        moment = Moment(
            id="uid_vid_00000:train:track:7",
            location="uid_vid_00000",
            split="train",
            video_path=str(case_root / "uid_vid_00000.mp4"),
            label_path=str(case_root / "uid_vid_00000.mp4.json"),
            track_id="7",
            fps=24.0,
            start_frame=0,
            end_frame=12,
            start_second=0.0,
            end_second=0.5,
            captions=["person"],
            representative_bbox=[10, 20, 40, 80],
            sample_frames=[0, 6, 12],
            text="person",
            keywords=["person"],
            frame_paths=[str(frame_path)],
            crop_paths=[str(crop_path)],
        )

        def fake_encode_images_clip(paths, **kwargs):
            return np.asarray([[1.0, 0.0] for _ in paths], dtype="float32")

        def fake_encode_texts_clip(texts, **kwargs):
            if len(texts) == 5:
                return np.asarray(
                    [
                        [0.95, 0.0],
                        [0.2, 0.0],
                        [0.1, 0.0],
                        [0.1, 0.0],
                        [0.1, 0.0],
                    ],
                    dtype="float32",
                )
            joined = " ".join(texts).lower()
            if "backpack" in joined:
                return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
            if "hat" in joined:
                return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype="float32")
            if "without a bag" in joined:
                return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
            if "next to a road" in joined:
                return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
            raise AssertionError(f"Unexpected prompt set: {texts}")

        with (
            patch("surveillance_search.enrichment_inference.encode_images_clip", side_effect=fake_encode_images_clip),
            patch("surveillance_search.enrichment_inference.encode_texts_clip", side_effect=fake_encode_texts_clip),
            patch("surveillance_search.enrichment_inference._dominant_color_from_crop", return_value="red"),
        ):
            attribute_records, scene_records = infer_enrichment_payloads(
                [moment],
                clip_model_name="ViT-L-14",
                clip_pretrained="laion2b_s32b_b82k",
            )

        self.assertEqual(len(attribute_records), 1)
        self.assertEqual(attribute_records[0]["attributes"]["top_color"], "red")
        self.assertEqual(attribute_records[0]["attributes"]["top_type"], "shirt")
        self.assertTrue(attribute_records[0]["attributes"]["backpack"])
        self.assertTrue(attribute_records[0]["attributes"]["bag"])
        self.assertEqual(len(scene_records), 1)
        self.assertTrue(scene_records[0]["scene"]["near_road"])

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_infer_and_write_enrichment_writes_sidecars(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "enrichment-write"
        shutil.rmtree(case_root, ignore_errors=True)
        case_root.mkdir(parents=True, exist_ok=True)

        crop_path = case_root / "crop.jpg"
        frame_path = case_root / "frame.jpg"
        crop_path.write_bytes(b"fake")
        frame_path.write_bytes(b"fake")

        moment = Moment(
            id="m1",
            location="cam_01",
            split="train",
            video_path=str(case_root / "cam_01.mp4"),
            label_path=str(case_root / "label.json"),
            track_id="5",
            fps=10.0,
            start_frame=0,
            end_frame=1,
            start_second=0.0,
            end_second=0.2,
            captions=["person"],
            representative_bbox=[0, 0, 10, 10],
            sample_frames=[0],
            text="person",
            keywords=["person"],
            frame_paths=[str(frame_path)],
            crop_paths=[str(crop_path)],
        )

        with patch(
            "surveillance_search.enrichment_inference.infer_enrichment_payloads",
            return_value=(
                [{"moment_id": "m1", "video_name": "cam_01.mp4", "track_id": "5", "attributes": {"top_color": "red"}}],
                [{"moment_id": "m1", "video_name": "cam_01.mp4", "track_id": "5", "scene": {"near_road": True}}],
            ),
        ):
            summary = infer_and_write_enrichment(
                [moment],
                dataset_root=case_root,
                clip_model_name="ViT-L-14",
                clip_pretrained="laion2b_s32b_b82k",
            )

        self.assertEqual(summary["attribute_count"], 1)
        self.assertEqual(summary["scene_count"], 1)
        self.assertTrue((case_root / "enrichment" / "person_attributes.json").exists())
        self.assertTrue((case_root / "enrichment" / "scene_context.json").exists())

        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
