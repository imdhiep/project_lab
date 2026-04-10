import unittest
from argparse import Namespace

from lava_search.cli import parse_weights, resolve_profile
from lava_search.indexer import _expand_query_text, _person_signal
from lava_search.models import Moment


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


if __name__ == "__main__":
    unittest.main()
