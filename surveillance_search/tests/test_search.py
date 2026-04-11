import shutil
import unittest
from pathlib import Path
from unittest import mock

from surveillance_search.indexer import build_search_bundle, search_index
from surveillance_search.models import Moment


class SearchTests(unittest.TestCase):
    def test_person_mode_returns_person_result_for_person_dataset(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "search-person"
        shutil.rmtree(case_root, ignore_errors=True)
        case_root.mkdir(parents=True, exist_ok=True)

        moments = [
            Moment(
                id="station:train:track:7",
                location="station",
                split="train",
                video_path=str(case_root / "cam_01.mp4"),
                label_path=str(case_root / "anno_visible_2022.json"),
                track_id="7",
                fps=10.0,
                start_frame=0,
                end_frame=3,
                start_second=0.0,
                end_second=0.4,
                captions=["person", "partially occluded person"],
                representative_bbox=[10, 20, 40, 80],
                sample_frames=[0, 1, 2],
                text="person pedestrian track in surveillance video cam_01. camera location station. tracked person 7.",
                keywords=["person", "pedestrian", "station", "track", "surveillance"],
            )
        ]

        build_search_bundle(
            moments=moments,
            output_dir=case_root,
            enable_sparse=True,
            enable_dense=False,
            enable_clip=False,
        )

        results = search_index(case_root, query_text="person near road", search_mode="person", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0]["person_match"], 0.0)
        self.assertIn("person", " ".join(results[0]["captions"]).lower())

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_person_mode_marks_generic_person_only_match_when_attributes_are_unsupported(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "search-person-generic-warning"
        shutil.rmtree(case_root, ignore_errors=True)
        case_root.mkdir(parents=True, exist_ok=True)

        moments = [
            Moment(
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
                text="person pedestrian track in surveillance video uid_vid_00000. tracked person 7.",
                keywords=["person", "pedestrian", "surveillance", "track"],
            )
        ]

        build_search_bundle(
            moments=moments,
            output_dir=case_root,
            enable_sparse=True,
            enable_dense=False,
            enable_clip=False,
        )

        results = search_index(
            case_root,
            query_text="person in red shirt near the road",
            search_mode="person",
            top_k=1,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["person_match"], 1.0)
        self.assertEqual(results[0]["precision_match"], 0.0)
        self.assertEqual(results[0]["match_quality"], "generic_person_only")
        self.assertIn("does not label clothing color", results[0]["result_warning"])

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_person_mode_can_fully_verify_attribute_and_scene_evidence(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "search-person-verified"
        shutil.rmtree(case_root, ignore_errors=True)
        case_root.mkdir(parents=True, exist_ok=True)

        moments = [
            Moment(
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
                text="person pedestrian track in surveillance video uid_vid_00000. tracked person 7. Evidence: red shirt. backpack. near road.",
                keywords=["person", "pedestrian", "surveillance", "track", "red", "shirt", "backpack", "road"],
                attribute_evidence={"top_color": "red", "top_type": "shirt", "backpack": True},
                scene_evidence={"near_road": True},
            )
        ]

        build_search_bundle(
            moments=moments,
            output_dir=case_root,
            enable_sparse=True,
            enable_dense=False,
            enable_clip=False,
        )

        results = search_index(
            case_root,
            query_text="person in red shirt near the road",
            search_mode="person",
            top_k=1,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["attribute_match"], 1.0)
        self.assertEqual(results[0]["scene_match"], 1.0)
        self.assertEqual(results[0]["match_quality"], "fully_verified")
        self.assertEqual(results[0]["verified_attributes"], ["red shirt", "backpack"])
        self.assertEqual(results[0]["verified_scene_relations"], ["near road"])
        self.assertNotIn("result_warning", results[0])

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_person_mode_rejects_non_person_dataset(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "search-non-person"
        shutil.rmtree(case_root, ignore_errors=True)
        case_root.mkdir(parents=True, exist_ok=True)

        moments = [
            Moment(
                id="amsterdam:test:track:1",
                location="amsterdam",
                split="test",
                video_path=str(case_root / "test.mp4"),
                label_path=str(case_root / "label.json"),
                track_id="1",
                fps=10.0,
                start_frame=0,
                end_frame=1,
                start_second=0.0,
                end_second=0.2,
                captions=["white van"],
                representative_bbox=[0, 0, 10, 10],
                sample_frames=[0],
                text="white van in traffic scene",
                keywords=["white", "van", "traffic"],
            )
        ]

        build_search_bundle(
            moments=moments,
            output_dir=case_root,
            enable_sparse=True,
            enable_dense=False,
            enable_clip=False,
        )

        with self.assertRaisesRegex(ValueError, "does not contain person-labeled moments"):
            search_index(case_root, query_text="person near road", search_mode="person", top_k=1)

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_qwen_parser_can_normalize_vietnamese_person_query(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "search-qwen-parser"
        shutil.rmtree(case_root, ignore_errors=True)
        case_root.mkdir(parents=True, exist_ok=True)

        moments = [
            Moment(
                id="uid_vid_00007:train:track:1",
                location="uid_vid_00007",
                split="train",
                video_path=str(case_root / "uid_vid_00007.mp4"),
                label_path=str(case_root / "uid_vid_00007.mp4.json"),
                track_id="1",
                fps=24.0,
                start_frame=0,
                end_frame=20,
                start_second=0.0,
                end_second=0.8,
                captions=["person"],
                representative_bbox=[10, 20, 40, 80],
                sample_frames=[0, 10, 20],
                text="person pedestrian track in surveillance video uid_vid_00007. Evidence: red sweater. near road.",
                keywords=["person", "pedestrian", "red", "sweater", "road"],
                attribute_evidence={"top_color": "red", "top_type": "sweater"},
                scene_evidence={"near_road": True},
            )
        ]

        build_search_bundle(
            moments=moments,
            output_dir=case_root,
            enable_sparse=True,
            enable_dense=False,
            enable_clip=False,
        )

        with mock.patch("surveillance_search.indexer.qwen_feature_available", return_value=True), mock.patch(
            "surveillance_search.indexer.parse_person_query_with_qwen",
            return_value={
                "must_be_person": True,
                "top_color": "red",
                "top_type": "sweater",
                "accessories": [],
                "near_road": True,
                "retrieval_terms": ["person", "red", "sweater", "near", "road"],
                "normalized_query": "person red sweater near road",
                "parser_source": "mock",
            },
        ):
            results = search_index(
                case_root,
                query_text="người mặc áo đỏ gần đường",
                search_mode="person",
                top_k=1,
                use_qwen_parser=True,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["attribute_match"], 1.0)
        self.assertEqual(results[0]["scene_match"], 1.0)
        self.assertTrue(results[0]["qwen_parser_used"])
        self.assertEqual(results[0]["structured_query"]["top_color"], "red")

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_qwen_reranker_can_reorder_top_candidates(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests"
        case_root = temp_root / "search-qwen-reranker"
        shutil.rmtree(case_root, ignore_errors=True)
        case_root.mkdir(parents=True, exist_ok=True)

        moments = [
            Moment(
                id="uid_vid_00002:train:track:3",
                location="uid_vid_00002",
                split="train",
                video_path=str(case_root / "uid_vid_00002.mp4"),
                label_path=str(case_root / "uid_vid_00002.mp4.json"),
                track_id="3",
                fps=24.0,
                start_frame=0,
                end_frame=20,
                start_second=0.0,
                end_second=0.8,
                captions=["person"],
                representative_bbox=[10, 20, 40, 80],
                sample_frames=[0, 10, 20],
                text="person pedestrian track in surveillance video",
                keywords=["person", "pedestrian", "surveillance"],
                attribute_evidence={"top_color": "black", "top_type": "coat"},
            ),
            Moment(
                id="uid_vid_00007:train:track:1",
                location="uid_vid_00007",
                split="train",
                video_path=str(case_root / "uid_vid_00007.mp4"),
                label_path=str(case_root / "uid_vid_00007.mp4.json"),
                track_id="1",
                fps=24.0,
                start_frame=0,
                end_frame=20,
                start_second=0.0,
                end_second=0.8,
                captions=["person"],
                representative_bbox=[10, 20, 40, 80],
                sample_frames=[0, 10, 20],
                text="person pedestrian track in surveillance video",
                keywords=["person", "pedestrian", "surveillance"],
                attribute_evidence={"top_color": "red", "top_type": "sweater"},
                scene_evidence={"near_road": True},
            ),
        ]

        build_search_bundle(
            moments=moments,
            output_dir=case_root,
            enable_sparse=True,
            enable_dense=False,
            enable_clip=False,
        )

        def _mock_rerank(*, moments, **kwargs):
            scores = []
            for moment in moments:
                scores.append(0.95 if moment.scene_evidence.get("near_road") else 0.05)
            return scores

        with mock.patch("surveillance_search.indexer.qwen_feature_available", return_value=True), mock.patch(
            "surveillance_search.indexer.rerank_moments_with_qwen",
            side_effect=_mock_rerank,
        ):
            results = search_index(
                case_root,
                query_text="person in red sweater near the road",
                search_mode="person",
                top_k=2,
                use_qwen_reranker=True,
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["track_id"], "1")
        self.assertTrue(results[0]["qwen_reranker_used"])
        self.assertGreater(results[0]["qwen_rerank_score"], results[1]["qwen_rerank_score"])

        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
