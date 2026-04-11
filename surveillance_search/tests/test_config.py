import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from surveillance_search.config import default_artifacts_root, default_data_root


class ConfigTests(unittest.TestCase):
    def test_default_data_root_prefers_new_lava_layout(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests" / "config-lava-current"
        shutil.rmtree(temp_root, ignore_errors=True)

        (temp_root / "data" / "lava" / "amsterdam" / "test").mkdir(parents=True, exist_ok=True)
        with patch("surveillance_search.config.project_root", return_value=temp_root):
            self.assertEqual(default_data_root(), temp_root / "data" / "lava")

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_default_data_root_falls_back_to_flat_legacy_lava_layout(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests" / "config-lava-flat-legacy"
        shutil.rmtree(temp_root, ignore_errors=True)

        (temp_root / "data" / "amsterdam" / "test").mkdir(parents=True, exist_ok=True)
        with patch("surveillance_search.config.project_root", return_value=temp_root):
            self.assertEqual(default_data_root(), temp_root / "data")

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_default_data_root_falls_back_to_nested_legacy_lava_layout(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests" / "config-lava-nested-legacy"
        shutil.rmtree(temp_root, ignore_errors=True)

        (temp_root / "data" / "Lava_Dataset" / "amsterdam" / "test").mkdir(parents=True, exist_ok=True)
        with patch("surveillance_search.config.project_root", return_value=temp_root):
            self.assertEqual(default_data_root(), temp_root / "data" / "Lava_Dataset")

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_default_data_root_personpath22_prefers_project_data_folder(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests" / "config-personpath22-current"
        shutil.rmtree(temp_root, ignore_errors=True)

        (temp_root / "data" / "personpath22" / "annotations").mkdir(parents=True, exist_ok=True)
        with patch("surveillance_search.config.project_root", return_value=temp_root):
            self.assertEqual(default_data_root("personpath22"), temp_root / "data" / "personpath22")

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_default_data_root_defaults_to_new_lava_folder_when_empty(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests" / "config-empty"
        shutil.rmtree(temp_root, ignore_errors=True)
        temp_root.mkdir(parents=True, exist_ok=True)

        with patch("surveillance_search.config.project_root", return_value=temp_root):
            self.assertEqual(default_data_root(), temp_root / "data" / "lava")

        shutil.rmtree(temp_root, ignore_errors=True)

    def test_default_artifacts_root_prefers_data_artifacts(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp-tests" / "config-artifacts"
        shutil.rmtree(temp_root, ignore_errors=True)

        (temp_root / "data" / "artifacts").mkdir(parents=True, exist_ok=True)
        with patch("surveillance_search.config.project_root", return_value=temp_root):
            self.assertEqual(default_artifacts_root(), temp_root / "data" / "artifacts")

        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
