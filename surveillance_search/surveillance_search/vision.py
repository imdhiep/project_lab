from __future__ import annotations

from functools import lru_cache
import warnings
from pathlib import Path


def detect_device(requested: str | None = None) -> str:
    if requested:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def _clear_torch_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _warn_cpu_fallback(stage: str) -> None:
    warnings.warn(
        f"{stage} ran out of CUDA memory, retrying on CPU. "
        "Set an explicit --device cpu if you want predictable low-VRAM behavior.",
        RuntimeWarning,
        stacklevel=2,
    )


@lru_cache(maxsize=4)
def _load_sentence_transformer_cached(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def _normalize_rows(array):
    import numpy as np

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return array / norms


def encode_texts_sentence_transformer(
    texts: list[str],
    model_name: str,
    device: str | None = None,
    batch_size: int = 16,
):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for dense text embeddings."
        ) from exc

    resolved_device = detect_device(device)
    try:
        model = _load_sentence_transformer_cached(model_name, resolved_device)
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > batch_size,
        )
        return vectors.astype("float32")
    except Exception as exc:
        if resolved_device == "cpu" or not _is_cuda_oom(exc):
            raise
        _warn_cpu_fallback("SentenceTransformer encoding")
        _clear_torch_cuda_cache()
        model = _load_sentence_transformer_cached(model_name, "cpu")
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > batch_size,
        )
        return vectors.astype("float32")


@lru_cache(maxsize=4)
def _load_open_clip_cached(model_name: str, pretrained: str, resolved_device: str):
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "open-clip-torch is required for multimodal CLIP indexing."
        ) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=resolved_device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    return {
        "device": resolved_device,
        "model": model,
        "preprocess": preprocess,
        "tokenizer": tokenizer,
    }


def _load_open_clip(model_name: str, pretrained: str, device: str | None = None):
    resolved_device = detect_device(device)
    try:
        return _load_open_clip_cached(model_name, pretrained, resolved_device)
    except Exception as exc:
        if resolved_device == "cpu" or not _is_cuda_oom(exc):
            raise
        _warn_cpu_fallback("OpenCLIP loading")
        _clear_torch_cuda_cache()
        return _load_open_clip_cached(model_name, pretrained, "cpu")


def encode_texts_clip(
    texts: list[str],
    model_name: str,
    pretrained: str,
    device: str | None = None,
    batch_size: int = 16,
):
    import numpy as np

    bundle = _load_open_clip(model_name, pretrained, device=device)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    resolved_device = bundle["device"]

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for CLIP text encoding.") from exc

    try:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                tokens = tokenizer(batch).to(resolved_device)
                embeddings = model.encode_text(tokens).detach().cpu().numpy()
                outputs.append(embeddings)
        return _normalize_rows(np.vstack(outputs)).astype("float32")
    except Exception as exc:
        if resolved_device == "cpu" or not _is_cuda_oom(exc):
            raise
        _warn_cpu_fallback("OpenCLIP text encoding")
        _clear_torch_cuda_cache()
        return encode_texts_clip(
            texts,
            model_name=model_name,
            pretrained=pretrained,
            device="cpu",
            batch_size=batch_size,
        )


def encode_images_clip(
    image_paths: list[str | Path],
    model_name: str,
    pretrained: str,
    device: str | None = None,
    batch_size: int = 16,
):
    import numpy as np

    bundle = _load_open_clip(model_name, pretrained, device=device)
    model = bundle["model"]
    preprocess = bundle["preprocess"]
    resolved_device = bundle["device"]

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for CLIP image encoding.") from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for CLIP image encoding.") from exc

    try:
        tensors = []
        outputs = []
        with torch.no_grad():
            for image_path in image_paths:
                tensor = preprocess(Image.open(image_path).convert("RGB"))
                tensors.append(tensor)
                if len(tensors) >= batch_size:
                    batch = torch.stack(tensors).to(resolved_device)
                    embeddings = model.encode_image(batch).detach().cpu().numpy()
                    outputs.append(embeddings)
                    tensors = []

            if tensors:
                batch = torch.stack(tensors).to(resolved_device)
                embeddings = model.encode_image(batch).detach().cpu().numpy()
                outputs.append(embeddings)

        return _normalize_rows(np.vstack(outputs)).astype("float32")
    except Exception as exc:
        if resolved_device == "cpu" or not _is_cuda_oom(exc):
            raise
        _warn_cpu_fallback("OpenCLIP image encoding")
        _clear_torch_cuda_cache()
        return encode_images_clip(
            image_paths,
            model_name=model_name,
            pretrained=pretrained,
            device="cpu",
            batch_size=batch_size,
        )
