from pathlib import Path

DEFAULT_REPO_ID = "xiaoyu123hhh/Lava_Dataset"
DEFAULT_DATASET_TYPE = "lava"
DEFAULT_LOCATIONS = (
    "amsterdam",
    "caldot1",
    "caldot2",
    "jackson",
    "shibuya",
    "warsaw",
)
DEFAULT_SPLITS = ("train", "test")
DEFAULT_FPS = 30.0
DEFAULT_TOP_K = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_ASSETS_PER_MOMENT = 3
DEFAULT_SENTENCE_MODEL = "BAAI/bge-m3"
DEFAULT_CLIP_MODEL = "ViT-L-14"
DEFAULT_CLIP_PRETRAINED = "laion2b_s32b_b82k"
DEFAULT_PROFILE = "strongest"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _looks_like_lava_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / location).is_dir() for location in DEFAULT_LOCATIONS)


def _looks_like_personpath22_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "annotations").is_dir() or (path / "raw_data").is_dir():
        return True
    return any((path / split / "gt").is_dir() for split in ("train", "test", "val"))


def _legacy_personpath22_root() -> Path:
    try:
        return project_root().parents[1] / "datasets" / "personpath22"
    except IndexError:
        return project_root() / "datasets" / "personpath22"


def default_data_root(dataset_type: str = DEFAULT_DATASET_TYPE) -> Path:
    data_root = project_root() / "data"
    if dataset_type == "personpath22":
        preferred_root = data_root / "personpath22"
        legacy_external_root = _legacy_personpath22_root()
        if _looks_like_personpath22_root(preferred_root):
            return preferred_root
        if _looks_like_personpath22_root(data_root):
            return data_root
        if _looks_like_personpath22_root(legacy_external_root):
            return legacy_external_root
        return preferred_root

    preferred_root = data_root / "lava"
    legacy_root = data_root / "Lava_Dataset"

    # Prefer the new repo layout (`./data/lava/<location>/<split>/...`), but keep
    # compatibility with the older layouts (`./data/<location>/...` and
    # `./data/Lava_Dataset/...`).
    if _looks_like_lava_root(preferred_root):
        return preferred_root
    if _looks_like_lava_root(data_root):
        return data_root
    if _looks_like_lava_root(legacy_root):
        return legacy_root
    return preferred_root


def default_artifacts_root() -> Path:
    data_artifacts_root = project_root() / "data" / "artifacts"
    legacy_root = project_root() / "artifacts"
    if data_artifacts_root.exists() or not legacy_root.exists():
        return data_artifacts_root
    return legacy_root


def default_bundle_root() -> Path:
    return default_artifacts_root() / "full"


def default_visual_root() -> Path:
    return default_artifacts_root() / "visual_assets"
