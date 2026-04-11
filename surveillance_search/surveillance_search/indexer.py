from __future__ import annotations

import json
import pickle
from pathlib import Path

from .answer_selection import annotate_results_with_answer_selection
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_PRETRAINED,
    DEFAULT_SENTENCE_MODEL,
)
from .enrichment import evidence_terms, summarize_attribute_evidence, summarize_scene_evidence
from .models import Moment
from .qwen_integration import (
    parse_person_query_with_qwen,
    qwen_feature_available,
    rerank_moments_with_qwen,
)
from .video_tools import build_frame_info, select_crop_path, select_frame_path, select_visual_path, summarize_captions
from .vision import encode_images_clip, encode_texts_clip, encode_texts_sentence_transformer

DEFAULT_WEIGHTS = {
    "sparse": 0.3,
    "dense": 0.45,
    "clip_text": 0.1,
    "clip_image": 0.15,
}
DEFAULT_RRF_K = 60.0
TEXT_PRECISION_GATE_FLOOR = 0.35
TEXT_PRECISION_MIN_SPARSE_SCORE = 0.05
PERSON_MODE_FLOOR = 0.7
PERSON_ATTRIBUTE_GATE_FLOOR = 0.08
ATTRIBUTE_EVIDENCE_GATE_FLOOR = 0.2
SCENE_EVIDENCE_GATE_FLOOR = 0.2
QWEN_RERANK_BLEND = 0.45
QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "in",
        "near",
        "of",
        "on",
        "the",
        "to",
        "with",
    }
)
COLOR_TERMS = frozenset(
    {
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "white",
        "black",
        "gray",
        "grey",
        "brown",
        "pink",
        "purple",
    }
)
PERSON_ATTRIBUTE_TERMS = frozenset(
    {
        "shirt",
        "jacket",
        "coat",
        "hoodie",
        "sweater",
        "vest",
        "pants",
        "trousers",
        "shorts",
        "skirt",
        "dress",
        "backpack",
        "bag",
        "handbag",
        "hat",
        "cap",
        "helmet",
        "umbrella",
    }
)
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
GENERIC_PERSON_CAPTIONS = frozenset(
    {
        "person",
        "partially occluded person",
        "heavily occluded person",
    }
)


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


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _expand_query_text(query_text: str, search_mode: str) -> str:
    query_text = query_text.strip()
    if search_mode != "person" or not query_text:
        return query_text

    tokens = _tokenize_query(query_text)
    if tokens & PERSON_TERMS:
        return f"{query_text}. pedestrian person human"
    return f"{query_text}. {PERSON_QUERY_EXPANSION}"


def _person_signal(moment: Moment) -> float:
    terms = _moment_terms(moment)
    matches = len(terms & PERSON_TERMS)
    if matches == 0:
        return 0.0
    if matches == 1:
        return 0.65
    return 1.0


def _moment_terms(moment: Moment) -> set[str]:
    terms = _tokenize_query(moment.text)
    for keyword in moment.keywords:
        terms.update(_tokenize_query(keyword))
    for caption in moment.captions:
        terms.update(_tokenize_query(caption))
    terms.update(evidence_terms(moment.attribute_evidence, moment.scene_evidence))
    return terms


def _query_evidence_requirements(query_text: str, search_mode: str) -> dict[str, object]:
    if search_mode != "person" or not query_text:
        return {
            "attribute_terms": set(),
            "scene_terms": set(),
            "needs_near_road": False,
        }

    raw_tokens = _tokenize_query(query_text)
    attribute_terms = set()
    attribute_terms.update(raw_tokens & COLOR_TERMS)
    attribute_terms.update(raw_tokens & PERSON_ATTRIBUTE_TERMS)

    scene_terms = set()
    if "road" in raw_tokens or "street" in raw_tokens:
        scene_terms.add("road")

    return {
        "attribute_terms": attribute_terms,
        "scene_terms": scene_terms,
        "needs_near_road": bool(scene_terms) and any(token in raw_tokens for token in {"near", "beside", "by", "next"}),
    }


def _attribute_scene_match_scores(query_text: str, moments: list[Moment], search_mode: str):
    np = _np()
    requirements = _query_evidence_requirements(query_text, search_mode)
    attribute_terms = requirements["attribute_terms"]
    scene_terms = requirements["scene_terms"]
    attribute_scores = np.ones(len(moments), dtype="float32")
    scene_scores = np.ones(len(moments), dtype="float32")

    if not attribute_terms and not scene_terms:
        return attribute_scores, scene_scores, requirements

    if attribute_terms:
        attribute_scores = np.zeros(len(moments), dtype="float32")
    if scene_terms:
        scene_scores = np.zeros(len(moments), dtype="float32")

    for index, moment in enumerate(moments):
        evidence = evidence_terms(moment.attribute_evidence, moment.scene_evidence)
        if attribute_terms:
            attribute_scores[index] = len(attribute_terms & evidence) / len(attribute_terms)

        if scene_terms:
            near_road_available = "near" in evidence and "road" in evidence
            if requirements["needs_near_road"]:
                scene_scores[index] = 1.0 if near_road_available else 0.0
            else:
                scene_scores[index] = len(scene_terms & evidence) / len(scene_terms)

    return attribute_scores, scene_scores, requirements


def _apply_attribute_scene_gates(combined, attribute_scores, scene_scores, requirements: dict[str, object]):
    updated = combined
    if requirements.get("attribute_terms"):
        updated = updated * (
            ATTRIBUTE_EVIDENCE_GATE_FLOOR
            + ((1.0 - ATTRIBUTE_EVIDENCE_GATE_FLOOR) * attribute_scores)
        )
    if requirements.get("scene_terms"):
        updated = updated * (
            SCENE_EVIDENCE_GATE_FLOOR
            + ((1.0 - SCENE_EVIDENCE_GATE_FLOOR) * scene_scores)
        )
    return updated


def _query_terms_for_precision(query_text: str, search_mode: str) -> set[str]:
    query_terms = _tokenize_query(query_text)
    filtered = {token for token in query_terms if token not in QUERY_STOPWORDS}
    if search_mode == "person":
        filtered = {token for token in filtered if token not in PERSON_TERMS}
    return filtered or query_terms


def _compute_text_precision_scores(query_text: str, moments: list[Moment], search_mode: str):
    np = _np()
    query_terms = _query_terms_for_precision(query_text, search_mode)
    if not query_terms:
        return np.ones(len(moments), dtype="float32")

    normalized_query = _normalize_text(query_text)
    scores = np.zeros(len(moments), dtype="float32")
    for index, moment in enumerate(moments):
        moment_terms = _moment_terms(moment)
        overlap = len(query_terms & moment_terms) / len(query_terms)
        keyword_overlap = len(query_terms & {keyword.lower() for keyword in moment.keywords}) / len(query_terms)
        phrase_bonus = 0.0
        haystacks = [moment.text, *moment.captions]
        if normalized_query and any(normalized_query in _normalize_text(text) for text in haystacks if text):
            phrase_bonus = 1.0

        scores[index] = min(1.0, (0.7 * overlap) + (0.15 * keyword_overlap) + (0.15 * phrase_bonus))
    return scores


def _query_requests_person_attributes(query_text: str, search_mode: str) -> bool:
    if search_mode != "person" or not query_text:
        return False
    requirements = _query_evidence_requirements(query_text, search_mode)
    query_terms = _query_terms_for_precision(query_text, search_mode)
    return bool(query_terms or requirements["attribute_terms"] or requirements["scene_terms"])


def _is_generic_person_moment(moment: Moment) -> bool:
    normalized_captions = {_normalize_text(caption) for caption in moment.captions if caption}
    return bool(normalized_captions) and normalized_captions.issubset(GENERIC_PERSON_CAPTIONS)


def _apply_generic_person_attribute_gate(combined, moments: list[Moment], precision_scores, query_text: str, search_mode: str):
    if precision_scores is None or not _query_requests_person_attributes(query_text, search_mode):
        return combined

    np = _np()
    generic_mask = np.asarray(
        [1.0 if _is_generic_person_moment(moment) else 0.0 for moment in moments],
        dtype="float32",
    )
    if float(np.max(generic_mask)) <= 0.0:
        return combined

    unsupported_mask = generic_mask * (np.asarray(precision_scores, dtype="float32") <= 0.0)
    if float(np.max(unsupported_mask)) <= 0.0:
        return combined

    penalties = np.where(
        unsupported_mask > 0.0,
        PERSON_ATTRIBUTE_GATE_FLOOR,
        1.0,
    ).astype("float32")
    return np.asarray(combined, dtype="float32") * penalties


def _build_result_warning(
    moment: Moment,
    query_text: str | None,
    search_mode: str,
    person_match: float | None,
    precision_match: float | None,
    attribute_match: float | None,
    scene_match: float | None,
) -> str | None:
    if search_mode != "person" or not query_text:
        return None
    if person_match is None or person_match <= 0.0:
        return None
    if not _query_requests_person_attributes(query_text, search_mode):
        return None

    requirements = _query_evidence_requirements(query_text, search_mode)
    missing_chunks: list[str] = []
    if requirements["attribute_terms"] and (attribute_match is None or attribute_match <= 0.0):
        missing_chunks.append("appearance attributes")
    if requirements["scene_terms"] and (scene_match is None or scene_match <= 0.0):
        missing_chunks.append("scene relations")

    if not missing_chunks:
        return None

    if _is_generic_person_moment(moment):
        scope = " and ".join(missing_chunks)
        if missing_chunks == ["appearance attributes", "scene relations"]:
            scope = "label clothing color or scene relations like 'near the road'"
        elif missing_chunks == ["appearance attributes"]:
            scope = "label clothing color or carried-item attributes"
        elif missing_chunks == ["scene relations"]:
            scope = "verify scene relations like 'near the road'"
        return (
            "Generic person-only match: this dataset confirms a person track here, "
            f"but it does not {scope} for this result."
        )

    if precision_match is not None and precision_match > 0.0:
        return None
    return f"Partial verification: missing evidence for {' and '.join(missing_chunks)}."


def _match_quality(
    moment: Moment,
    query_text: str | None,
    search_mode: str,
    attribute_match: float | None,
    scene_match: float | None,
) -> str | None:
    if search_mode != "person" or not query_text:
        return None

    requirements = _query_evidence_requirements(query_text, search_mode)
    requested_attributes = bool(requirements["attribute_terms"])
    requested_scene = bool(requirements["scene_terms"])
    if not requested_attributes and not requested_scene:
        return None

    attr_ok = (not requested_attributes) or ((attribute_match or 0.0) >= 0.999)
    scene_ok = (not requested_scene) or ((scene_match or 0.0) >= 0.999)
    if attr_ok and scene_ok:
        return "fully_verified"

    attr_some = requested_attributes and ((attribute_match or 0.0) > 0.0)
    scene_some = requested_scene and ((scene_match or 0.0) > 0.0)
    if attr_some or scene_some:
        return "partially_verified"

    if _is_generic_person_moment(moment):
        return "generic_person_only"
    return "unverified"


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
        visual_paths = [select_visual_path(moment, preference="frame") for moment in moments]
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


def _reciprocal_rank_scores(scores, k: float = DEFAULT_RRF_K):
    np = _np()
    scores = np.asarray(scores, dtype="float32")
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(scores.shape[0], dtype="float32")
    ranks[order] = np.arange(1, scores.shape[0] + 1, dtype="float32")
    return 1.0 / (k + ranks)


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
    rank_scores = {key: _reciprocal_rank_scores(score_vectors[key]) for key in active_keys}
    combined = np.zeros_like(next(iter(rank_scores.values())))
    fusion_breakdown = {}
    for key in active_keys:
        contribution = rank_scores[key] * resolved_weights[key]
        fusion_breakdown[key] = contribution.astype("float32")
        combined += contribution
    return combined, normalized, fusion_breakdown, resolved_weights


def _should_apply_text_precision_gate(score_vectors: dict[str, object]) -> bool:
    sparse_scores = score_vectors.get("sparse")
    if sparse_scores is None:
        return False
    return float(_np().max(sparse_scores)) >= TEXT_PRECISION_MIN_SPARSE_SCORE


def _apply_text_precision_gate(combined, precision_scores):
    return combined * (TEXT_PRECISION_GATE_FLOOR + ((1.0 - TEXT_PRECISION_GATE_FLOOR) * precision_scores))


def _apply_person_mode_boost(combined, person_scores):
    return combined * (PERSON_MODE_FLOOR + ((1.0 - PERSON_MODE_FLOOR) * person_scores))


def _require_person_candidates(person_scores):
    np = _np()
    person_scores = np.asarray(person_scores, dtype="float32")
    if person_scores.size == 0 or float(np.max(person_scores)) <= 0.0:
        raise ValueError(
            "Person mode requested, but the loaded dataset does not contain person-labeled moments. "
            "Switch datasets or use the default search mode."
        )
    return person_scores


def _ranked_indices_with_person_filter(combined, person_scores, top_k: int):
    np = _np()
    combined = np.asarray(combined, dtype="float32")
    person_scores = np.asarray(person_scores, dtype="float32")
    ranked = np.argsort(-combined)
    positive_person = [int(index) for index in ranked if float(person_scores[int(index)]) > 0.0]
    if positive_person:
        return np.asarray(positive_person[:top_k], dtype="int32")
    return ranked[:top_k]


def _apply_qwen_rerank(
    combined,
    ranked_indices,
    rerank_scores: list[float],
):
    np = _np()
    ranked_indices = np.asarray(ranked_indices, dtype="int32")
    if ranked_indices.size == 0 or not rerank_scores:
        return ranked_indices, {}, {}

    rerank_vector = np.asarray(rerank_scores, dtype="float32")
    candidate_scores = np.asarray([combined[int(index)] for index in ranked_indices], dtype="float32")
    base_norm = _normalize_scores(candidate_scores)
    rerank_norm = _normalize_scores(rerank_vector)
    fused = ((1.0 - QWEN_RERANK_BLEND) * base_norm) + (QWEN_RERANK_BLEND * rerank_norm)
    order = np.argsort(-fused, kind="stable")
    reranked_indices = ranked_indices[order]
    rerank_lookup = {int(ranked_indices[i]): float(rerank_vector[i]) for i in range(len(ranked_indices))}
    fused_lookup = {int(ranked_indices[i]): float(fused[i]) for i in range(len(ranked_indices))}
    return reranked_indices, rerank_lookup, fused_lookup


def search_index(
    output_dir: Path,
    query_text: str | None = None,
    query_image_path: str | None = None,
    top_k: int = 5,
    weights: dict[str, float] | None = None,
    search_mode: str = "default",
    use_qwen_parser: bool = False,
    use_qwen_reranker: bool = False,
    qwen_rerank_limit: int = 20,
    qwen_device: str | None = None,
) -> list[dict]:
    if not query_text and not query_image_path:
        raise ValueError("Provide `query_text` and/or `query_image_path`.")

    np = _np()
    bundle = load_bundle(output_dir)
    moments = _load_moments(Path(bundle["moments_path"]))
    score_vectors: dict[str, object] = {}
    parser_payload = None
    query_text_for_matching = query_text
    qwen_parser_used = False
    qwen_reranker_used = False
    qwen_parser_error = None
    qwen_reranker_error = None
    if query_text and search_mode == "person" and use_qwen_parser and qwen_feature_available("parser"):
        try:
            parser_payload = parse_person_query_with_qwen(
                query_text,
                device=qwen_device,
            )
            normalized_query = str(parser_payload.get("normalized_query") or "").strip()
            if normalized_query:
                query_text_for_matching = normalized_query
            qwen_parser_used = True
        except Exception as exc:
            qwen_parser_error = str(exc)

    effective_query_text = _expand_query_text(query_text_for_matching, search_mode) if query_text_for_matching else None

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

    combined, normalized, fusion_breakdown, resolved_weights = _fuse_score_vectors(score_vectors, weights)
    precision_scores = None
    if query_text_for_matching:
        precision_scores = _compute_text_precision_scores(query_text_for_matching, moments, search_mode)
        if _should_apply_text_precision_gate(score_vectors):
            combined = _apply_text_precision_gate(combined, precision_scores)
        if search_mode == "person":
            combined = _apply_generic_person_attribute_gate(
                combined,
                moments,
                precision_scores,
                query_text_for_matching,
                search_mode,
            )
    attribute_scores = None
    scene_scores = None
    requirements = None
    person_scores = None
    if search_mode == "person":
        if query_text_for_matching:
            attribute_scores, scene_scores, requirements = _attribute_scene_match_scores(
                query_text_for_matching,
                moments,
                search_mode,
            )
            combined = _apply_attribute_scene_gates(
                combined,
                attribute_scores,
                scene_scores,
                requirements,
            )
        person_scores = _require_person_candidates(
            np.array([_person_signal(moment) for moment in moments], dtype="float32")
        )
        combined = _apply_person_mode_boost(combined, person_scores)
        ranked_indices = _ranked_indices_with_person_filter(
            combined,
            person_scores,
            min(top_k, len(moments)),
        )
    else:
        ranked_indices = np.argsort(-combined)[: min(top_k, len(moments))]

    rerank_score_lookup: dict[int, float] = {}
    rerank_fused_lookup: dict[int, float] = {}
    if (
        query_text_for_matching
        and use_qwen_reranker
        and qwen_feature_available("reranker")
        and len(ranked_indices) > 1
    ):
        rerank_limit = min(max(top_k * 4, top_k), max(qwen_rerank_limit, top_k), len(ranked_indices))
        rerank_indices = np.asarray(ranked_indices[:rerank_limit], dtype="int32")
        try:
            rerank_scores = rerank_moments_with_qwen(
                query_text=query_text_for_matching,
                moments=[moments[int(index)] for index in rerank_indices],
                device=qwen_device,
            )
            reranked_subset, rerank_score_lookup, rerank_fused_lookup = _apply_qwen_rerank(
                combined,
                rerank_indices,
                rerank_scores,
            )
            trailing = [int(index) for index in ranked_indices if int(index) not in set(int(item) for item in rerank_indices)]
            ranked_indices = np.asarray([*reranked_subset.tolist(), *trailing], dtype="int32")
            qwen_reranker_used = True
        except Exception as exc:
            qwen_reranker_error = str(exc)

    results = []
    for index in ranked_indices:
        raw_breakdown = {key: float(score_vectors[key][index]) for key in score_vectors}
        norm_breakdown = {key: float(normalized[key][index]) for key in normalized}
        fused_breakdown = {key: float(fusion_breakdown[key][index]) for key in fusion_breakdown}
        results.append(
            _format_result(
                moment=moments[int(index)],
                score=float(rerank_fused_lookup.get(int(index), combined[index])),
                score_breakdown=raw_breakdown,
                normalized_breakdown=norm_breakdown,
                weights=resolved_weights,
                search_mode=search_mode,
                expanded_query=effective_query_text if effective_query_text != query_text else None,
                person_match=float(person_scores[index]) if person_scores is not None else None,
                fusion_breakdown=fused_breakdown,
                precision_match=float(precision_scores[index]) if precision_scores is not None else None,
                query_text=query_text_for_matching,
                attribute_match=float(attribute_scores[index]) if attribute_scores is not None else None,
                scene_match=float(scene_scores[index]) if scene_scores is not None else None,
                structured_query=parser_payload,
                qwen_parser_used=qwen_parser_used,
                qwen_reranker_used=qwen_reranker_used,
                qwen_rerank_score=rerank_score_lookup.get(int(index)),
                qwen_parser_error=qwen_parser_error,
                qwen_reranker_error=qwen_reranker_error,
                query_requirements=requirements,
            )
        )
        if len(results) >= top_k:
            break
    clip_artifact = bundle.get("artifacts", {}).get("clip", {})
    clip_model_name = clip_artifact.get("model_name") or bundle["defaults"].get("clip_model_name", DEFAULT_CLIP_MODEL)
    clip_pretrained = clip_artifact.get("pretrained") or bundle["defaults"].get("clip_pretrained", DEFAULT_CLIP_PRETRAINED)
    annotate_results_with_answer_selection(
        results=results,
        query_text=query_text_for_matching,
        search_mode=search_mode,
        structured_query=parser_payload,
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
        device=bundle["defaults"].get("device"),
        batch_size=bundle["defaults"].get("batch_size", DEFAULT_BATCH_SIZE),
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
    fusion_breakdown: dict[str, float] | None = None,
    precision_match: float | None = None,
    query_text: str | None = None,
    attribute_match: float | None = None,
    scene_match: float | None = None,
    structured_query: dict[str, object] | None = None,
    qwen_parser_used: bool = False,
    qwen_reranker_used: bool = False,
    qwen_rerank_score: float | None = None,
    qwen_parser_error: str | None = None,
    qwen_reranker_error: str | None = None,
    query_requirements: dict[str, object] | None = None,
) -> dict:
    payload = moment.to_dict()
    frame_path = select_frame_path(moment)
    crop_path = select_crop_path(moment)
    visual_path = select_visual_path(moment, preference="frame")
    payload["score"] = round(score, 6)
    payload["score_breakdown"] = {key: round(value, 6) for key, value in score_breakdown.items()}
    payload["normalized_breakdown"] = {
        key: round(value, 6) for key, value in normalized_breakdown.items()
    }
    if fusion_breakdown:
        payload["fusion_breakdown"] = {
            key: round(value, 6) for key, value in fusion_breakdown.items()
        }
    payload["weights"] = {key: round(value, 6) for key, value in weights.items()}
    payload["visual_path"] = visual_path
    payload["frame_path"] = frame_path or visual_path
    payload["crop_path"] = crop_path
    payload["caption_text"] = summarize_captions(moment, limit=2)
    payload["frame_info"] = build_frame_info(moment)
    payload["search_mode"] = search_mode
    if expanded_query:
        payload["expanded_query"] = expanded_query
    if person_match is not None:
        payload["person_match"] = round(person_match, 6)
    if precision_match is not None:
        payload["precision_match"] = round(precision_match, 6)
    if attribute_match is not None:
        payload["attribute_match"] = round(attribute_match, 6)
    if scene_match is not None:
        payload["scene_match"] = round(scene_match, 6)
    if structured_query:
        payload["structured_query"] = structured_query
    if qwen_parser_used:
        payload["qwen_parser_used"] = True
    if qwen_reranker_used:
        payload["qwen_reranker_used"] = True
    if qwen_rerank_score is not None:
        payload["qwen_rerank_score"] = round(float(qwen_rerank_score), 6)
    if qwen_parser_error:
        payload["qwen_parser_error"] = qwen_parser_error
    if qwen_reranker_error:
        payload["qwen_reranker_error"] = qwen_reranker_error
    if query_requirements is not None:
        payload["query_requirements"] = {
            "attribute_terms": sorted(str(term) for term in query_requirements.get("attribute_terms", set())),
            "scene_terms": sorted(str(term) for term in query_requirements.get("scene_terms", set())),
            "needs_near_road": bool(query_requirements.get("needs_near_road", False)),
        }
    if moment.attribute_evidence:
        payload["attribute_evidence"] = moment.attribute_evidence
        payload["verified_attributes"] = summarize_attribute_evidence(moment.attribute_evidence)
    if moment.scene_evidence:
        payload["scene_evidence"] = moment.scene_evidence
        payload["verified_scene_relations"] = summarize_scene_evidence(moment.scene_evidence)
    result_warning = _build_result_warning(
        moment=moment,
        query_text=query_text,
        search_mode=search_mode,
        person_match=person_match,
        precision_match=precision_match,
        attribute_match=attribute_match,
        scene_match=scene_match,
    )
    match_quality = _match_quality(
        moment=moment,
        query_text=query_text,
        search_mode=search_mode,
        attribute_match=attribute_match,
        scene_match=scene_match,
    )
    if match_quality:
        payload["match_quality"] = match_quality
    if result_warning:
        payload["result_warning"] = result_warning
    return payload
