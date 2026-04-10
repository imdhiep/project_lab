from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class VisualAsset:
    path: str
    asset_type: str
    frame_idx: int
    bbox: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "VisualAsset":
        return cls(**payload)


@dataclass
class Moment:
    id: str
    location: str
    split: str
    video_path: str
    label_path: str
    track_id: str
    fps: float
    start_frame: int
    end_frame: int
    start_second: float
    end_second: float
    captions: list[str]
    representative_bbox: list[int]
    sample_frames: list[int]
    text: str
    keywords: list[str] = field(default_factory=list)
    frame_paths: list[str] = field(default_factory=list)
    crop_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Moment":
        payload = dict(payload)
        payload.setdefault("fps", 30.0)
        payload.setdefault("keywords", [])
        payload.setdefault("frame_paths", [])
        payload.setdefault("crop_paths", [])
        return cls(**payload)
