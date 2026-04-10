from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from .indexer import load_bundle, search_index
from .runtime import load_runtime_manifest
from .video_tools import extract_clip


class SearchRequest(BaseModel):
    query_text: str | None = None
    image_path: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    search_mode: str = Field(default="default", pattern="^(default|person)$")
    weights: dict[str, float] | None = None

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


def create_app(output_dir: str | Path) -> FastAPI:
    bundle_dir = Path(output_dir)
    app = FastAPI(title="LAVA Search API", version="1.0.0")

    @app.get("/")
    def root():
        bundle = load_bundle(bundle_dir)
        return {
            "service": "lava-search",
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
            )
            if len(results) < payload.rank:
                raise ValueError("Requested rank exceeds available results.")

            chosen = results[payload.rank - 1]
            output_path = Path(payload.output_path)
            clip_path = extract_clip(
                video_path=Path(chosen["video_path"]),
                output_path=output_path,
                start_second=max(chosen["start_second"] - payload.padding, 0.0),
                duration_seconds=payload.duration,
                ffmpeg_bin=payload.ffmpeg_bin,
            )
            return {"clip_path": str(clip_path), "result": chosen}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
