from __future__ import annotations

import json
import pickle
from pathlib import Path

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_PRETRAINED,
    DEFAULT_SENTENCE_MODEL,
)
from .models import Moment
from .video_tools import select_visual_path
from .vision import encode_images_clip, encode_texts_clip, encode_texts_sentence_transformer

DEFAULT_WEIGHTS = {
    "sparse": 0.15,
    "dense": 0.35,
    "clip_text": 0.25,
    "clip_image": 0.25,
}
PERSON_TERMS = frozenset(
    {
        "person",
        "persons",
        "pedestrian",
        "pedestrians",
        "man",
        "men",
        "woman",
        "women",
        "people",
        "human",
        "humans",
        "walker",
        "walkers",
    }
)
PERSON_QUERY_EXPANSION = "person pedestrian man woman people human walker"


def _np():
    import numpy as np

    return np


def _ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_moments(moments: list[Moment], output_dir: Path) -> Path:
    moments_path = output_dir / "moments.json"
    with moments_path.open("w", encoding="utf-8") as handle:
        json.dump([moment.to_dict() for moment in moments], handle, ensure_ascii=False, indent=2)
    return moments_path


def _load_moments(moments_path: Path) -> list[Moment]:
    with moments_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [Moment.from_dict(item) for item in payload]


def _save_bundle(output_dir: Path, payload: dict) -> Path:
    bundle_path = output_dir / "bundle.json"
    with bundle_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return bundle_path


def load_bundle(output_dir: Path) -> dict:
    bundle_path = output_dir / "bundle.json"
    with bundle_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_faiss(embeddings, index_path: Path) -> str | None:
    try:
        import faiss
    except ImportError:
        return None

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, str(index_path))
    return str(index_path)


def _artifact_enabled(bundle: dict, name: str) -> bool:
    return bool(bundle.get("artifacts", {}).get(name, {}).get("enabled"))


def _tokenize_query(text: str) -> set[str]:
    return {token.strip(" ,.!?;:()[]{}\"'").lower() for token in text.split() if token.strip()}


def _expand_query_text(query_text: str, search_mode: str) -> str:
    query_text = query_text.strip()
    if search_mode != "person" or not query_text:
        return query_text

    tokens = _tokenize_query(query_text)
    if tokens & PERSON_TERMS:
        return f"{query_text}. pedestrian person human"
    return f"{query_text}. {PERSON_QUERY_EXPANSION}"


def _person_signal(moment: Moment) -> float:
    terms = {token.lower() for token in moment.keywords}
    for caption in moment.captions:
        terms.update(_tokenize_query(caption))

    matches = len(terms & PERSON_TERMS)
    if matches == 0:
        return 0.0
    if matches == 1:
        return 0.65
    return 1.0


def build_search_bundle(
    moments: list[Moment],
    output_dir: Path,
    enable_sparse: bool = True,
    enable_dense: bool = True,
    enable_clip: bool = True,
    sentence_model_name: str = DEFAULT_SENTENCE_MODEL,
    clip_model_name: str = DEFAULT_CLIP_MODEL,
    clip_pretrained: str = DEFAULT_CLIP_PRETRAINED,
    device: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Path:
    if not moments:
        raise ValueError("No moments found. Download labels first or check dataset path.")

    np = _np()
    output_dir = _ensure_output_dir(output_dir)
    moments_path = _save_moments(moments, output_dir)
    texts = [moment.text for moment in moments]

    bundle = {
        "moments_path": str(moments_path),
        "artifact_dir": str(output_dir),
        "defaults": {
            "sentence_model_name": sentence_model_name,
            "clip_model_name": clip_model_name,
            "clip_pretrained": clip_pretrained,
            "device": device,
            "batch_size": batch_size,
        },
        "artifacts": {},
    }

    if enable_sparse:
        bundle["artifacts"]["sparse"] = _build_sparse_artifact(texts, output_dir)

    if enable_dense:
        bundle["artifacts"]["dense"] = _build_dense_artifact(
            texts=texts,
            output_dir=output_dir,
            model_name=sentence_model_name,
            device=device,
            batch_size=batch_size,
        )

    if enable_clip:
        visual_paths = [select_visual_path(moment) for moment in moments]
        valid_items = [(index, path) for index, path in enumerate(visual_paths) if path]
        if not valid_items:
            bundle["artifacts"]["clip"] = {
                "enabled": False,
                "reason": "No visual assets found. Run `prepare-assets` after downloading videos.",
            }
        else:
            selected_indices = np.array([item[0] for item in valid_items], dtype="int32")
            selected_paths = [item[1] for item in valid_items]
            bundle["artifacts"]["clip"] = _build_clip_artifact(
                visual_paths=selected_paths,
                moment_indices=selected_indices,
                output_dir=output_dir,
                model_name=clip_model_name,
                pretrained=clip_pretrained,
                device=device,
                batch_size=batch_size,
            )

    return _save_bundle(output_dir, bundle)


def _build_sparse_artifact(texts: list[str], output_dir: Path) -> dict:
    from scipy.sparse import save_npz
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    matrix = vectorizer.fit_transform(texts)

    vectorizer_path = output_dir / "sparse_vectorizer.pkl"
    matrix_path = output_dir / "sparse_matrix.npz"

    with vectorizer_path.open("wb") as handle:
        pickle.dump(vectorizer, handle)
    save_npz(matrix_path, matrix)

    return {
        "enabled": True,
        "vectorizer_path": str(vectorizer_path),
        "matrix_path": str(matrix_path),
    }


def _build_dense_artifact(
    texts: list[str],
    output_dir: Path,
    model_name: str,
    device: str | None,
    batch_size: int,
) -> dict:
    np = _np()
    embeddings = encode_texts_sentence_transformer(
        texts,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )

    embeddings_path = output_dir / "dense_embeddings.npy"
    index_path = output_dir / "dense.index"
    np.save(embeddings_path, embeddings)

    return {
        "enabled": True,
        "model_name": model_name,
        "embeddings_path": str(embeddings_path),
        "index_path": _write_faiss(embeddings, index_path),
    }


def _build_clip_artifact(
    visual_paths: list[str],
    moment_indices,
    output_dir: Path,
    model_name: str,
    pretrained: str,
    device: str | None,
    batch_size: int,
) -> dict:
    np = _np()
    embeddings = encode_images_clip(
        visual_paths,
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        batch_size=batch_size,
    )

    embeddings_path = output_dir / "clip_image_embeddings.npy"
    moment_indices_path = output_dir / "clip_moment_indices.npy"
    image_paths_path = output_dir / "clip_image_paths.json"
    index_path = output_dir / "clip.index"

    np.save(embeddings_path, embeddings)
    np.save(moment_indices_path, moment_indices)
    with image_paths_path.open("w", encoding="utf-8") as handle:
        json.dump(visual_paths, handle, ensure_ascii=False, indent=2)

    return {
        "enabled": True,
        "model_name": model_name,
        "pretrained": pretrained,
        "embeddings_path": str(embeddings_path),
        "moment_indices_path": str(moment_indices_path),
        "image_paths_path": str(image_paths_path),
        "index_path": _write_faiss(embeddings, index_path),
    }


def build_sparse_index(moments: list[Moment], output_dir: Path) -> Path:
    return build_search_bundle(
        moments,
        output_dir,
        enable_sparse=True,
        enable_dense=False,
        enable_clip=False,
    )


def build_dense_index(moments: list[Moment], output_dir: Path, model_name: str) -> Path:
    return build_search_bundle(
        moments,
        output_dir,
        enable_sparse=False,
        enable_dense=True,
        enable_clip=False,
        sentence_model_name=model_name,
    )


def _normalize_scores(scores):
    np = _np()
    scores = np.asarray(scores, dtype="float32")
    finite = np.isfinite(scores)
    normalized = np.zeros_like(scores)
    if not finite.any():
        return normalized

    valid = scores[finite]
    min_value = float(valid.min())
    max_value = float(valid.max())
    if abs(max_value - min_value) < 1e-12:
        normalized[finite] = 1.0 if max_value > 0 else 0.0
        return normalized

    normalized[finite] = (scores[finite] - min_value) / (max_value - min_value)
    return normalized


def _resolve_weights(active_keys: list[str], user_weights: dict[str, float] | None):
    resolved = {}
    total = 0.0
    for key in active_keys:
        value = DEFAULT_WEIGHTS.get(key, 1.0)
        if user_weights and key in user_weights:
            value = max(float(user_weights[key]), 0.0)
        resolved[key] = value
        total += value

    if total <= 0:
        uniform = 1.0 / max(len(active_keys), 1)
        return {key: uniform for key in active_keys}

    return {key: value / total for key, value in resolved.items()}


def _fuse_score_vectors(score_vectors: dict[str, object], weights: dict[str, float] | None):
    np = _np()
    active_keys = list(score_vectors.keys())
    resolved_weights = _resolve_weights(active_keys, weights)
    normalized = {key: _normalize_scores(score_vectors[key]) for key in active_keys}
    combined = np.zeros_like(next(iter(normalized.values())))
    for key in active_keys:
        combined += normalized[key] * resolved_weights[key]
    return combined, normalized, resolved_weights


def search_index(
    output_dir: Path,
    query_text: str | None = None,
    query_image_path: str | None = None,
    top_k: int = 5,
    weights: dict[str, float] | None = None,
    search_mode: str = "default",
) -> list[dict]:
    if not query_text and not query_image_path:
        raise ValueError("Provide `query_text` and/or `query_image_path`.")

    np = _np()
    bundle = load_bundle(output_dir)
    moments = _load_moments(Path(bundle["moments_path"]))
    score_vectors: dict[str, object] = {}
    effective_query_text = _expand_query_text(query_text, search_mode) if query_text else None

    if effective_query_text and _artifact_enabled(bundle, "sparse"):
        score_vectors["sparse"] = _search_sparse_scores(bundle, effective_query_text)

    if effective_query_text and _artifact_enabled(bundle, "dense"):
        score_vectors["dense"] = _search_dense_scores(bundle, effective_query_text)

    if effective_query_text and _artifact_enabled(bundle, "clip"):
        score_vectors["clip_text"] = _search_clip_text_scores(bundle, effective_query_text, len(moments))

    if query_image_path and _artifact_enabled(bundle, "clip"):
        score_vectors["clip_image"] = _search_clip_image_scores(bundle, query_image_path, len(moments))

    if not score_vectors:
        raise ValueError("No search branch is available for the provided query.")

    combined, normalized, resolved_weights = _fuse_score_vectors(score_vectors, weights)
    person_scores = None
    if search_mode == "person":
        person_scores = np.array([_person_signal(moment) for moment in moments], dtype="float32")
        combined = combined + (0.35 * person_scores)
    ranked_indices = np.argsort(-combined)[: min(top_k, len(moments))]

    results = []
    for index in ranked_indices:
        raw_breakdown = {key: float(score_vectors[key][index]) for key in score_vectors}
        norm_breakdown = {key: float(normalized[key][index]) for key in normalized}
        results.append(
            _format_result(
                moment=moments[int(index)],
                score=float(combined[index]),
                score_breakdown=raw_breakdown,
                normalized_breakdown=norm_breakdown,
                weights=resolved_weights,
                search_mode=search_mode,
                expanded_query=effective_query_text if effective_query_text != query_text else None,
                person_match=float(person_scores[index]) if person_scores is not None else None,
            )
        )
    return results


def _search_sparse_scores(bundle: dict, query_text: str):
    from scipy.sparse import load_npz
    from sklearn.metrics.pairwise import cosine_similarity

    artifact = bundle["artifacts"]["sparse"]
    with Path(artifact["vectorizer_path"]).open("rb") as handle:
        vectorizer = pickle.load(handle)
    matrix = load_npz(artifact["matrix_path"])
    query_vector = vectorizer.transform([query_text])
    return cosine_similarity(query_vector, matrix).ravel().astype("float32")


def _search_dense_scores(bundle: dict, query_text: str):
    np = _np()
    artifact = bundle["artifacts"]["dense"]
    embeddings = np.load(artifact["embeddings_path"])
    query_vector = encode_texts_sentence_transformer(
        [query_text],
        model_name=artifact["model_name"],
        device=bundle["defaults"].get("device"),
        batch_size=bundle["defaults"].get("batch_size", DEFAULT_BATCH_SIZE),
    )[0]
    return (embeddings @ query_vector).astype("float32")


def _search_clip_text_scores(bundle: dict, query_text: str, total_moments: int):
    artifact = bundle["artifacts"]["clip"]
    query_vector = encode_texts_clip(
        [query_text],
        model_name=artifact["model_name"],
        pretrained=artifact["pretrained"],
        device=bundle["defaults"].get("device"),
        batch_size=bundle["defaults"].get("batch_size", DEFAULT_BATCH_SIZE),
    )[0]
    return _score_clip_embeddings(artifact, query_vector, total_moments)


def _search_clip_image_scores(bundle: dict, image_path: str, total_moments: int):
    artifact = bundle["artifacts"]["clip"]
    query_vector = encode_images_clip(
        [image_path],
        model_name=artifact["model_name"],
        pretrained=artifact["pretrained"],
        device=bundle["defaults"].get("device"),
        batch_size=1,
    )[0]
    return _score_clip_embeddings(artifact, query_vector, total_moments)


def _score_clip_embeddings(artifact: dict, query_vector, total_moments: int):
    np = _np()
    embeddings = np.load(artifact["embeddings_path"])
    moment_indices = np.load(artifact["moment_indices_path"])
    partial_scores = embeddings @ query_vector
    full_scores = np.zeros(total_moments, dtype="float32")
    full_scores[moment_indices] = partial_scores.astype("float32")
    return full_scores


def _format_result(
    moment: Moment,
    score: float,
    score_breakdown: dict[str, float],
    normalized_breakdown: dict[str, float],
    weights: dict[str, float],
    search_mode: str,
    expanded_query: str | None,
    person_match: float | None,
) -> dict:
    payload = moment.to_dict()
    payload["score"] = round(score, 6)
    payload["score_breakdown"] = {key: round(value, 6) for key, value in score_breakdown.items()}
    payload["normalized_breakdown"] = {
        key: round(value, 6) for key, value in normalized_breakdown.items()
    }
    payload["weights"] = {key: round(value, 6) for key, value in weights.items()}
    payload["visual_path"] = select_visual_path(moment)
    payload["search_mode"] = search_mode
    if expanded_query:
        payload["expanded_query"] = expanded_query
    if person_match is not None:
        payload["person_match"] = round(person_match, 6)
    return payload
