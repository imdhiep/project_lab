import shutil
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from surveillance_search.answer_selection import annotate_results_with_answer_selection


class AnswerSelectionTests(unittest.TestCase):
    def _make_image(self, path: Path, color: tuple[int, int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (48, 48), color=color).save(path)

    def test_attribute_query_prefers_crop_answer_asset(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests" / "answer-attribute"
        shutil.rmtree(temp_root, ignore_errors=True)
        temp_root.mkdir(parents=True, exist_ok=True)

        frame_path = temp_root / "frame_000010.jpg"
        crop_path = temp_root / "crop_000010.jpg"
        self._make_image(frame_path, (120, 120, 120))
        self._make_image(crop_path, (255, 0, 0))

        results = [
            {
                "frame_paths": [str(frame_path)],
                "crop_paths": [str(crop_path)],
                "frame_info": {"fps": 10.0, "start_second": 0.0, "end_second": 5.0},
                "start_second": 0.0,
                "end_second": 5.0,
                "query_requirements": {"attribute_terms": ["red", "shirt"], "scene_terms": [], "needs_near_road": False},
            }
        ]

        with mock.patch("surveillance_search.answer_selection.encode_images_clip", return_value=np.asarray([[0.2], [0.9]], dtype="float32")), mock.patch(
            "surveillance_search.answer_selection.encode_texts_clip",
            return_value=np.asarray([[1.0]], dtype="float32"),
        ):
            annotate_results_with_answer_selection(
                results=results,
                query_text="person in red shirt",
                search_mode="person",
                structured_query=None,
                clip_model_name="mock",
                clip_pretrained="mock",
                device="cpu",
                batch_size=4,
            )

        self.assertEqual(results[0]["answer_asset_type"], "crop")
        self.assertEqual(Path(results[0]["answer_visual_path"]).name, "crop_000010.jpg")
        self.assertEqual(results[0]["query_profile"]["primary_intent"], "appearance_match")
        self.assertIn("answer_window", results[0])

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_scene_query_prefers_frame_answer_asset(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests" / "answer-scene"
        shutil.rmtree(temp_root, ignore_errors=True)
        temp_root.mkdir(parents=True, exist_ok=True)

        frame_path = temp_root / "frame_000020.jpg"
        crop_path = temp_root / "crop_000020.jpg"
        self._make_image(frame_path, (180, 180, 180))
        self._make_image(crop_path, (20, 20, 20))

        results = [
            {
                "frame_paths": [str(frame_path)],
                "crop_paths": [str(crop_path)],
                "frame_info": {"fps": 10.0, "start_second": 0.0, "end_second": 6.0},
                "start_second": 0.0,
                "end_second": 6.0,
                "query_requirements": {"attribute_terms": [], "scene_terms": ["road"], "needs_near_road": True},
            }
        ]

        with mock.patch("surveillance_search.answer_selection.encode_images_clip", return_value=np.asarray([[0.9], [0.2]], dtype="float32")), mock.patch(
            "surveillance_search.answer_selection.encode_texts_clip",
            return_value=np.asarray([[1.0]], dtype="float32"),
        ):
            annotate_results_with_answer_selection(
                results=results,
                query_text="person near the road",
                search_mode="person",
                structured_query=None,
                clip_model_name="mock",
                clip_pretrained="mock",
                device="cpu",
                batch_size=4,
            )

        self.assertEqual(results[0]["answer_asset_type"], "frame")
        self.assertEqual(Path(results[0]["answer_visual_path"]).name, "frame_000020.jpg")
        self.assertEqual(results[0]["query_profile"]["primary_intent"], "scene_relation")

        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
