import unittest
from argparse import Namespace

from surveillance_search.cli import parse_weights, resolve_profile
from surveillance_search.indexer import (
    _ranked_indices_with_person_filter,
    _require_person_candidates,
    _compute_text_precision_scores,
    _expand_query_text,
    _format_result,
    _fuse_score_vectors,
    _person_signal,
)
from surveillance_search.models import Moment
from surveillance_search.vision import _is_cuda_oom


class CliHelperTests(unittest.TestCase):
    def test_parse_weights(self) -> None:
        weights = parse_weights("sparse=0.1,dense=0.4,clip_text=0.2,clip_image=0.3")
        self.assertEqual(
            weights,
            {
                "sparse": 0.1,
                "dense": 0.4,
                "clip_text": 0.2,
                "clip_image": 0.3,
            },
        )

    def test_resolve_profile_with_flags(self) -> None:
        profile = resolve_profile(
            "strongest",
            Namespace(no_sparse=False, no_dense=True, no_clip=False),
        )
        self.assertEqual(profile, (True, False, True))

    def test_person_query_expansion(self) -> None:
        expanded = _expand_query_text("person waiting near road", "person")
        self.assertIn("pedestrian", expanded)

    def test_person_signal(self) -> None:
        moment = Moment(
            id="m1",
            location="amsterdam",
            split="test",
            video_path="test.mp4",
            label_path="label.json",
            track_id="7",
            fps=30.0,
            start_frame=0,
            end_frame=10,
            start_second=0.0,
            end_second=0.33,
            captions=["pedestrian in white shirt"],
            representative_bbox=[0, 0, 10, 10],
            sample_frames=[0],
            text="pedestrian in white shirt",
            keywords=["pedestrian", "white", "shirt"],
        )
        self.assertGreater(_person_signal(moment), 0.0)

    def test_format_result_prefers_frame_output(self) -> None:
        moment = Moment(
            id="m2",
            location="amsterdam",
            split="test",
            video_path="test.mp4",
            label_path="label.json",
            track_id="8",
            fps=30.0,
            start_frame=12,
            end_frame=18,
            start_second=0.4,
            end_second=0.633,
            captions=["person in red shirt", "walking near road"],
            representative_bbox=[1, 2, 30, 40],
            sample_frames=[12, 15],
            text="person in red shirt",
            keywords=["person", "red", "shirt"],
            frame_paths=["/tmp/frame_000012.jpg"],
            crop_paths=["/tmp/crop_000012.jpg"],
        )

        payload = _format_result(
            moment=moment,
            score=0.9,
            score_breakdown={"clip_text": 0.5, "clip_image": 0.4},
            normalized_breakdown={"clip_text": 1.0, "clip_image": 0.8},
            weights={"clip_text": 0.5, "clip_image": 0.5},
            search_mode="person",
            expanded_query=None,
            person_match=1.0,
            attribute_match=1.0,
            scene_match=1.0,
        )

        self.assertEqual(payload["frame_path"], "/tmp/frame_000012.jpg")
        self.assertEqual(payload["crop_path"], "/tmp/crop_000012.jpg")
        self.assertEqual(payload["visual_path"], "/tmp/frame_000012.jpg")
        self.assertIn("person in red shirt", payload["caption_text"])
        self.assertEqual(payload["frame_info"]["preview_frame_idx"], 12)
        self.assertEqual(payload["frame_info"]["bbox"], [1, 2, 30, 40])
        self.assertEqual(payload["attribute_match"], 1.0)
        self.assertEqual(payload["scene_match"], 1.0)

    def test_rrf_fusion_prefers_consensus_result(self) -> None:
        combined, normalized, fusion_breakdown, weights = _fuse_score_vectors(
            {
                "sparse": [0.99, 0.75, 0.05],
                "dense": [0.05, 0.74, 0.99],
                "clip_text": [0.2, 0.95, 0.1],
            },
            None,
        )

        self.assertGreater(combined[1], combined[0])
        self.assertGreater(combined[1], combined[2])
        self.assertIn("sparse", normalized)
        self.assertIn("dense", fusion_breakdown)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_text_precision_scores_reward_literal_match(self) -> None:
        exact = Moment(
            id="m3",
            location="amsterdam",
            split="test",
            video_path="test.mp4",
            label_path="label.json",
            track_id="9",
            fps=30.0,
            start_frame=0,
            end_frame=10,
            start_second=0.0,
            end_second=0.33,
            captions=["red car turning left at intersection"],
            representative_bbox=[0, 0, 10, 10],
            sample_frames=[0],
            text="red car turning left at intersection",
            keywords=["red", "car", "turning", "left"],
        )
        vague = Moment(
            id="m4",
            location="amsterdam",
            split="test",
            video_path="test.mp4",
            label_path="label.json",
            track_id="10",
            fps=30.0,
            start_frame=0,
            end_frame=10,
            start_second=0.0,
            end_second=0.33,
            captions=["vehicle moving through junction"],
            representative_bbox=[0, 0, 10, 10],
            sample_frames=[0],
            text="vehicle moving through junction",
            keywords=["vehicle", "moving"],
        )

        scores = _compute_text_precision_scores("red car turning left", [vague, exact], "default")
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[1], 0.9)

    def test_person_filter_removes_non_person_results_when_possible(self) -> None:
        ranked = _ranked_indices_with_person_filter(
            [0.9, 0.8, 0.7],
            [0.0, 1.0, 0.65],
            top_k=2,
        )
        self.assertEqual(ranked.tolist(), [1, 2])

    def test_person_mode_rejects_datasets_without_person_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not contain person-labeled moments"):
            _require_person_candidates([0.0, 0.0, 0.0])

    def test_cuda_oom_detection(self) -> None:
        self.assertTrue(_is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")))
        self.assertFalse(_is_cuda_oom(RuntimeError("Some other runtime failure")))


if __name__ == "__main__":
    unittest.main()
