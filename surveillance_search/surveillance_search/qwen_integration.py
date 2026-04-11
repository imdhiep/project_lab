from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .config import project_root
from .models import Moment
from .vision import detect_device

DEFAULT_QWEN_QUERY_PARSER_DIRNAME = "Qwen3-4B-Instruct-2507"
DEFAULT_QWEN_EMBEDDING_DIRNAME = "Qwen3-Embedding-4B"
DEFAULT_QWEN_RERANKER_DIRNAME = "Qwen3-Reranker-4B"
DEFAULT_QWEN_RERANK_INSTRUCTION = (
    "Given a surveillance search query, retrieve the tracked person candidate that best matches "
    "the requested appearance attributes, carried items, and scene relations."
)

_COLOR_ALIASES = {
    "red": {"red", "reddish", "do", "đỏ"},
    "blue": {"blue", "xanhduong", "xanhdương", "xanh_duong", "xanh_dương"},
    "green": {"green", "xanhla", "xanh_lá", "xanhlaa", "xanh lá"},
    "yellow": {"yellow", "vang", "vàng"},
    "orange": {"orange", "cam"},
    "white": {"white", "trang", "trắng"},
    "black": {"black", "den", "đen"},
    "gray": {"gray", "grey", "xam", "xám"},
    "brown": {"brown", "nau", "nâu"},
    "pink": {"pink", "hong", "hồng"},
    "purple": {"purple", "tim", "tím"},
}
_TOP_TYPE_ALIASES = {
    "shirt": {"shirt", "t-shirt", "tee", "aothun", "áo thun", "ao thun", "somi", "sơ mi"},
    "jacket": {"jacket", "ao khoac", "áo khoác", "khoac", "khoác"},
    "coat": {"coat", "overcoat", "mangto", "măng tô"},
    "hoodie": {"hoodie", "ao hoodie", "áo hoodie", "hooded"},
    "sweater": {"sweater", "jumper", "len", "áo len"},
    "vest": {"vest", "ghi le", "ghi-lê", "ghile"},
    "pants": {"pants", "trousers", "quan", "quần"},
    "shorts": {"shorts", "quan short", "quần short"},
    "skirt": {"skirt", "vay", "váy"},
    "dress": {"dress", "dam", "đầm"},
}
_ACCESSORY_ALIASES = {
    "backpack": {"backpack", "balo", "ba lo"},
    "bag": {"bag", "handbag", "tui", "túi"},
    "hat": {"hat", "cap", "mu", "mũ", "non", "nón"},
    "helmet": {"helmet", "mu bao hiem", "mũ bảo hiểm", "bao hiem", "bảo hiểm"},
    "umbrella": {"umbrella", "o", "ô", "du"},
}
_NEAR_ALIASES = {"near", "nearby", "gần", "gan", "beside", "next", "by"}
_ROAD_ALIASES = {"road", "street", "duong", "đường", "pho", "phố"}


def _slug_tokens(text: str) -> set[str]:
    normalized = text.lower()
    raw = re.findall(r"[\wÀ-ỹ]+", normalized, flags=re.UNICODE)
    tokens: set[str] = set()
    for token in raw:
        compact = token.replace("_", "").replace("-", "")
        tokens.add(token)
        tokens.add(compact)
    return {token for token in tokens if token}


def local_models_root() -> Path:
    return project_root() / "models"


def discover_local_qwen_models() -> dict[str, str | None]:
    root = local_models_root()
    mapping = {
        "parser": root / DEFAULT_QWEN_QUERY_PARSER_DIRNAME,
        "embedding": root / DEFAULT_QWEN_EMBEDDING_DIRNAME,
        "reranker": root / DEFAULT_QWEN_RERANKER_DIRNAME,
    }
    return {
        name: str(path) if path.exists() and path.is_dir() else None
        for name, path in mapping.items()
    }


def local_qwen_embedding_model() -> str | None:
    return discover_local_qwen_models().get("embedding")


def qwen_feature_available(feature: str) -> bool:
    return bool(discover_local_qwen_models().get(feature))


def _canonical_value(tokens: set[str], aliases: dict[str, set[str]]) -> str | None:
    for canonical, candidates in aliases.items():
        if tokens & candidates:
            return canonical
    return None


def _canonical_multi_values(tokens: set[str], aliases: dict[str, set[str]]) -> list[str]:
    found = [canonical for canonical, candidates in aliases.items() if tokens & candidates]
    return sorted(set(found))


def _heuristic_parse_query(query_text: str) -> dict[str, object]:
    tokens = _slug_tokens(query_text)
    color = _canonical_value(tokens, _COLOR_ALIASES)
    top_type = _canonical_value(tokens, _TOP_TYPE_ALIASES)
    accessories = _canonical_multi_values(tokens, _ACCESSORY_ALIASES)
    near_road = bool(tokens & _NEAR_ALIASES) and bool(tokens & _ROAD_ALIASES)

    retrieval_terms = ["person"]
    if color:
        retrieval_terms.append(color)
    if top_type:
        retrieval_terms.append(top_type)
    retrieval_terms.extend(accessories)
    if near_road:
        retrieval_terms.extend(["near", "road"])

    normalized_query = " ".join(dict.fromkeys(retrieval_terms))
    return {
        "must_be_person": True,
        "top_color": color,
        "top_type": top_type,
        "accessories": accessories,
        "near_road": near_road,
        "retrieval_terms": list(dict.fromkeys(retrieval_terms)),
        "normalized_query": normalized_query or query_text.strip(),
        "parser_source": "heuristic",
    }


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


@lru_cache(maxsize=2)
def _load_qwen_instruct(model_path: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        for field in ("temperature", "top_p", "top_k"):
            if hasattr(model.generation_config, field):
                setattr(model.generation_config, field, None)
    model.eval()
    return tokenizer, model


def _extract_json_object(text: str) -> dict[str, object] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def parse_person_query_with_qwen(
    query_text: str,
    model_path: str | None = None,
    device: str | None = None,
) -> dict[str, object]:
    discovered = discover_local_qwen_models()
    resolved_model_path = model_path or discovered.get("parser")
    if not resolved_model_path:
        return _heuristic_parse_query(query_text)

    resolved_device = detect_device(device)
    try:
        import torch
    except ImportError:
        return _heuristic_parse_query(query_text)

    try:
        tokenizer, model = _load_qwen_instruct(resolved_model_path, resolved_device)
        prompt = (
            "You convert surveillance person-search requests into compact JSON.\n"
            "Return JSON only with these keys:\n"
            "{\n"
            '  "must_be_person": true,\n'
            '  "top_color": string or null,\n'
            '  "top_type": string or null,\n'
            '  "accessories": array of strings from [backpack, bag, hat, helmet, umbrella],\n'
            '  "near_road": boolean,\n'
            '  "retrieval_terms": array of short English keywords,\n'
            '  "normalized_query": short English retrieval string\n'
            "}\n"
            "Use only normalized English values. If something is not specified, use null or []."
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query_text},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        payload = _extract_json_object(decoded)
        if not isinstance(payload, dict):
            raise ValueError("Qwen parser did not return JSON.")
        heuristic = _heuristic_parse_query(query_text)
        result = {
            "must_be_person": bool(payload.get("must_be_person", True)),
            "top_color": payload.get("top_color") or heuristic.get("top_color"),
            "top_type": payload.get("top_type") or heuristic.get("top_type"),
            "accessories": sorted(
                set((payload.get("accessories") or []) + (heuristic.get("accessories") or []))
            ),
            "near_road": bool(payload.get("near_road", heuristic.get("near_road", False))),
            "retrieval_terms": payload.get("retrieval_terms") or heuristic.get("retrieval_terms") or ["person"],
            "normalized_query": payload.get("normalized_query") or heuristic.get("normalized_query") or query_text.strip(),
            "parser_source": "qwen3",
        }
        if "person" not in result["retrieval_terms"]:
            result["retrieval_terms"] = ["person", *result["retrieval_terms"]]
        if "person" not in result["normalized_query"].lower():
            result["normalized_query"] = f"person {result['normalized_query']}".strip()
        return result
    except Exception as exc:
        if resolved_device != "cpu" and _is_cuda_oom(exc):
            _clear_torch_cuda_cache()
            return parse_person_query_with_qwen(
                query_text=query_text,
                model_path=resolved_model_path,
                device="cpu",
            )
        return _heuristic_parse_query(query_text)


def _moment_to_rerank_document(moment: Moment) -> str:
    chunks = [
        f"Location: {moment.location}",
        f"Split: {moment.split}",
        f"Track: {moment.track_id}",
        f"Captions: {', '.join(moment.captions) if moment.captions else 'none'}",
        f"Keywords: {', '.join(moment.keywords) if moment.keywords else 'none'}",
        f"Text: {moment.text}",
    ]
    if moment.attribute_evidence:
        attr_chunks = []
        top_color = moment.attribute_evidence.get("top_color")
        top_type = moment.attribute_evidence.get("top_type")
        if top_color or top_type:
            attr_chunks.append(" ".join(part for part in (top_color, top_type) if part))
        for key in ("backpack", "bag", "hat", "helmet", "umbrella"):
            if moment.attribute_evidence.get(key):
                attr_chunks.append(key)
        if attr_chunks:
            chunks.append(f"Verified attributes: {', '.join(attr_chunks)}")
    if moment.scene_evidence and moment.scene_evidence.get("near_road"):
        chunks.append("Verified scene relation: near road")
    return "\n".join(chunks)


@lru_cache(maxsize=2)
def _load_qwen_reranker(model_path: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return tokenizer, model


def rerank_moments_with_qwen(
    query_text: str,
    moments: list[Moment],
    model_path: str | None = None,
    device: str | None = None,
    instruction: str = DEFAULT_QWEN_RERANK_INSTRUCTION,
) -> list[float]:
    discovered = discover_local_qwen_models()
    resolved_model_path = model_path or discovered.get("reranker")
    if not resolved_model_path or not moments:
        return []

    resolved_device = detect_device(device)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for Qwen reranking.") from exc

    try:
        tokenizer, model = _load_qwen_reranker(resolved_model_path, resolved_device)
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        max_length = 4096
        prefix = (
            "<|im_start|>system\n"
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

        pairs = [
            "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                instruction=instruction,
                query=query_text,
                doc=_moment_to_rerank_document(moment),
            )
            for moment in moments
        ]

        inputs = tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        )
        for index, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = prefix_tokens + input_ids + suffix_tokens
        padded = tokenizer.pad(inputs, padding=True, return_tensors="pt")
        padded = {key: value.to(model.device) for key, value in padded.items()}

        with torch.no_grad():
            batch_scores = model(**padded).logits[:, -1, :]
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            stacked = torch.stack([false_vector, true_vector], dim=1)
            stacked = torch.nn.functional.log_softmax(stacked, dim=1)
            return stacked[:, 1].exp().detach().cpu().tolist()
    except Exception as exc:
        if resolved_device != "cpu" and _is_cuda_oom(exc):
            _clear_torch_cuda_cache()
            return rerank_moments_with_qwen(
                query_text=query_text,
                moments=moments,
                model_path=resolved_model_path,
                device="cpu",
                instruction=instruction,
            )
        raise
