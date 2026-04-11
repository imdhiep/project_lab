import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from surveillance_search.person_query_stack import ensure_person_query_stack_layout, write_stack_configs


class PersonQueryStackTests(unittest.TestCase):
    def test_write_stack_configs_creates_expected_files(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests" / "person-query-stack"
        shutil.rmtree(temp_root, ignore_errors=True)
        temp_root.mkdir(parents=True, exist_ok=True)

        with patch("surveillance_search.person_query_stack.project_root", return_value=temp_root):
            layout = ensure_person_query_stack_layout()
            paths = write_stack_configs()

        self.assertTrue(layout["pa100k_root"].exists())
        self.assertTrue(layout["bdd100k_root"].exists())
        self.assertTrue(paths["attribute_config"].exists())
        self.assertTrue(paths["scene_config"].exists())
        stack_payload = json.loads(paths["stack_config"].read_text(encoding="utf-8"))
        self.assertEqual(stack_payload["video_base_dataset"]["name"], "PersonPath22")
        self.assertEqual(stack_payload["attribute_supervision"]["name"], "PA-100K")
        self.assertEqual(stack_payload["scene_supervision"]["name"], "BDD100K")

        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
