from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from .config import DEFAULT_BATCH_SIZE, DEFAULT_CLIP_MODEL, DEFAULT_CLIP_PRETRAINED, DEFAULT_PROFILE, default_data_root, default_visual_root
from .hospital_ingest import DEFAULT_QUEUE_SIZE, HospitalIngestConfig, ingest_hospital_video
from .indexer import load_bundle, search_index
from .runtime import RuntimeConfig, load_runtime_manifest, rebuild_runtime_bundle
from .video_tools import extract_clip


class SearchRequest(BaseModel):
    query_text: str | None = None
    image_path: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    search_mode: str = Field(default="default", pattern="^(default|person)$")
    weights: dict[str, float] | None = None
    use_qwen_parser: bool = False
    use_qwen_reranker: bool = False
    qwen_rerank_limit: int = Field(default=20, ge=1, le=50)
    qwen_device: str | None = None

    @model_validator(mode="after")
    def validate_query(self):
        if not self.query_text and not self.image_path:
            raise ValueError("Provide `query_text` and/or `image_path`.")
        return self


class ClipRequest(SearchRequest):
    rank: int = Field(default=1, ge=1, le=50)
    duration: float = Field(default=4.0, gt=0.1, le=60.0)
    padding: float = Field(default=0.5, ge=0.0, le=10.0)
    output_path: str
    ffmpeg_bin: str = "ffmpeg"


class IngestVideoRequest(BaseModel):
    source_path: str
    camera_id: str | None = None
    start_time: str | None = None
    queue_size: int = Field(default=DEFAULT_QUEUE_SIZE, ge=1, le=10000)
    detection_fps: float = Field(default=1.0, gt=0.0, le=30.0)
    min_track_frames: int = Field(default=2, ge=1, le=1000)
    min_person_area: int = Field(default=1600, ge=1)
    enable_enrichment: bool = True
    profile: str = Field(default=DEFAULT_PROFILE, pattern="^(strongest|balanced|text-only|lite)$")
    clip_model: str = DEFAULT_CLIP_MODEL
    clip_pretrained: str = DEFAULT_CLIP_PRETRAINED
    device: str | None = None
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1, le=512)
    ffmpeg_bin: str = "ffmpeg"
    encoder: str = "hevc_nvenc"


def create_app(output_dir: str | Path, dataset_root: str | Path | None = None, assets_dir: str | Path | None = None) -> FastAPI:
    bundle_dir = Path(output_dir)
    hospital_root = Path(dataset_root) if dataset_root else default_data_root("hospital")
    hospital_assets_dir = Path(assets_dir) if assets_dir else default_visual_root()
    app = FastAPI(title="Surveillance Search API", version="1.0.0")

    @app.get("/")
    def root():
        bundle = load_bundle(bundle_dir)
        return {
            "service": "surveillance-search",
            "bundle_dir": str(bundle_dir),
            "artifacts": bundle.get("artifacts", {}),
        }

    @app.get("/health")
    def health():
        bundle = load_bundle(bundle_dir)
        return {"ok": True, "bundle_dir": str(bundle_dir), "artifacts": bundle.get("artifacts", {})}

    @app.get("/manifest")
    def manifest():
        return load_runtime_manifest(bundle_dir) or {}

    @app.post("/search")
    def search(payload: SearchRequest):
        try:
            return {
                "results": search_index(
                    bundle_dir,
                    query_text=payload.query_text,
                    query_image_path=payload.image_path,
                    top_k=payload.top_k,
                    weights=payload.weights,
                    search_mode=payload.search_mode,
                    use_qwen_parser=payload.use_qwen_parser,
                    use_qwen_reranker=payload.use_qwen_reranker,
                    qwen_rerank_limit=payload.qwen_rerank_limit,
                    qwen_device=payload.qwen_device,
                )
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/clip")
    def build_clip(payload: ClipRequest):
        try:
            results = search_index(
                bundle_dir,
                query_text=payload.query_text,
                query_image_path=payload.image_path,
                top_k=max(payload.rank, payload.top_k),
                weights=payload.weights,
                search_mode=payload.search_mode,
                use_qwen_parser=payload.use_qwen_parser,
                use_qwen_reranker=payload.use_qwen_reranker,
                qwen_rerank_limit=payload.qwen_rerank_limit,
                qwen_device=payload.qwen_device,
            )
            if len(results) < payload.rank:
                raise ValueError("Requested rank exceeds available results.")

            chosen = results[payload.rank - 1]
            output_path = Path(payload.output_path)
            answer_window = chosen.get("answer_window") or {}
            if answer_window:
                start_second = max(float(answer_window.get("start_second", chosen["start_second"])) - payload.padding, 0.0)
                duration_seconds = max(float(answer_window.get("duration_seconds", payload.duration)) + payload.padding, 0.1)
            else:
                start_second = max(chosen["start_second"] - payload.padding, 0.0)
                duration_seconds = payload.duration
            clip_path = extract_clip(
                video_path=Path(chosen["video_path"]),
                output_path=output_path,
                start_second=start_second,
                duration_seconds=duration_seconds,
                ffmpeg_bin=payload.ffmpeg_bin,
            )
            return {"clip_path": str(clip_path), "result": chosen}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/ingest/video")
    def ingest_video(payload: IngestVideoRequest):
        try:
            ingest_summary = ingest_hospital_video(
                source_path=Path(payload.source_path),
                config=HospitalIngestConfig(
                    dataset_root=hospital_root,
                    assets_dir=hospital_assets_dir,
                    ffmpeg_bin=payload.ffmpeg_bin,
                    preferred_encoder=payload.encoder,
                    queue_size=payload.queue_size,
                    detection_fps=payload.detection_fps,
                    min_track_frames=payload.min_track_frames,
                    min_person_area=payload.min_person_area,
                    enable_enrichment=payload.enable_enrichment,
                    clip_model_name=payload.clip_model,
                    clip_pretrained=payload.clip_pretrained,
                    device=payload.device,
                    batch_size=payload.batch_size,
                ),
                camera_id=payload.camera_id,
                recorded_start=payload.start_time,
            )
            enable_sparse = payload.profile in {"strongest", "balanced", "text-only", "lite"}
            enable_dense = payload.profile in {"strongest", "balanced"}
            enable_clip = payload.profile in {"strongest", "balanced"}
            manifest = rebuild_runtime_bundle(
                RuntimeConfig(
                    dataset_type="hospital",
                    dataset_root=hospital_root,
                    output_dir=bundle_dir,
                    assets_dir=hospital_assets_dir,
                    locations=None,
                    splits=None,
                    group_by_track=True,
                    fps=30.0,
                    enable_sparse=enable_sparse,
                    enable_dense=enable_dense,
                    enable_clip=enable_clip,
                    sentence_model_name="BAAI/bge-m3",
                    clip_model_name=payload.clip_model,
                    clip_pretrained=payload.clip_pretrained,
                    device=payload.device,
                    batch_size=payload.batch_size,
                    max_assets_per_moment=3,
                    crop_padding=0.08,
                    ffmpeg_bin=payload.ffmpeg_bin,
                    enable_enrichment=False,
                )
            )
            return {"ingest": ingest_summary, "manifest": manifest}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
