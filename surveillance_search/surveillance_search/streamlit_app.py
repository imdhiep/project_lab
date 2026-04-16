from __future__ import annotations

import html
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

try:
    from .config import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_CLIP_MODEL,
        DEFAULT_CLIP_PRETRAINED,
        DEFAULT_SENTENCE_MODEL,
        default_bundle_root,
        default_data_root,
        default_visual_root,
        project_root,
    )
    from .hospital_ingest import HospitalIngestConfig, bootstrap_nvidia_hospital_dataset, ingest_hospital_video
    from .indexer import load_bundle, search_index
    from .qwen_integration import qwen_feature_available
    from .runtime import RuntimeConfig, rebuild_runtime_bundle
except ImportError:
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    from surveillance_search.config import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_CLIP_MODEL,
        DEFAULT_CLIP_PRETRAINED,
        DEFAULT_SENTENCE_MODEL,
        default_bundle_root,
        default_data_root,
        default_visual_root,
        project_root,
    )
    from surveillance_search.hospital_ingest import (
        HospitalIngestConfig,
        bootstrap_nvidia_hospital_dataset,
        ingest_hospital_video,
    )
    from surveillance_search.indexer import load_bundle, search_index
    from surveillance_search.qwen_integration import qwen_feature_available
    from surveillance_search.runtime import RuntimeConfig, rebuild_runtime_bundle


DEFAULT_QUERY = "person in red shirt near the road"
DEFAULT_HOSPITAL_QUEUE_SIZE = 32


@st.cache_data(show_spinner=False)
def _load_bundle_stats(bundle_dir: str) -> dict:
    bundle = load_bundle(Path(bundle_dir))
    moments = json.loads(Path(bundle["moments_path"]).read_text(encoding="utf-8"))
    total = len(moments)
    with_visual = sum(1 for item in moments if item.get("frame_paths") or item.get("crop_paths"))
    return {
        "total_moments": total,
        "with_visual": with_visual,
        "visual_ratio": (with_visual / total) if total else 0.0,
    }


def _empty_bundle() -> dict:
    return {
        "artifacts": {
            "sparse": {"enabled": False},
            "dense": {"enabled": False},
            "clip": {"enabled": False},
        }
    }


def _empty_bundle_stats() -> dict:
    return {
        "total_moments": 0,
        "with_visual": 0,
        "visual_ratio": 0.0,
    }


def _load_bundle_state(bundle_dir: str) -> tuple[dict, dict, bool, str | None]:
    try:
        bundle = load_bundle(Path(bundle_dir))
        bundle_stats = _load_bundle_stats(bundle_dir)
        return bundle, bundle_stats, True, None
    except Exception as exc:
        return _empty_bundle(), _empty_bundle_stats(), False, str(exc)


def _default_nvidia_hospital_root() -> Path:
    return (
        project_root().parents[1]
        / "Multi-Camera-Person-Tracking-and-Re-Identification"
        / "data"
        / "NVIDIA_SmartSpaces"
        / "MTMC_Tracking_2025"
        / "val"
        / "Hospital_000"
    )


def _hospital_runtime_config(bundle_dir: str | Path) -> RuntimeConfig:
    return RuntimeConfig(
        dataset_type="hospital",
        dataset_root=default_data_root("hospital"),
        output_dir=Path(bundle_dir),
        assets_dir=default_visual_root(),
        locations=None,
        splits=None,
        group_by_track=True,
        fps=30.0,
        enable_sparse=True,
        enable_dense=True,
        enable_clip=True,
        sentence_model_name=DEFAULT_SENTENCE_MODEL,
        clip_model_name=DEFAULT_CLIP_MODEL,
        clip_pretrained=DEFAULT_CLIP_PRETRAINED,
        device=None,
        batch_size=DEFAULT_BATCH_SIZE,
        max_assets_per_moment=8,
        crop_padding=0.08,
        ffmpeg_bin="ffmpeg",
        enable_enrichment=False,
    )


def _hospital_ingest_config() -> HospitalIngestConfig:
    return HospitalIngestConfig(
        dataset_root=default_data_root("hospital"),
        assets_dir=default_visual_root(),
        ffmpeg_bin="ffmpeg",
        preferred_encoder="hevc_nvenc",
        queue_size=DEFAULT_HOSPITAL_QUEUE_SIZE,
        enable_enrichment=True,
        clip_model_name=DEFAULT_CLIP_MODEL,
        clip_pretrained=DEFAULT_CLIP_PRETRAINED,
        batch_size=DEFAULT_BATCH_SIZE,
        max_assets_per_moment=8,
        crop_padding=0.08,
        timeline_stride_seconds=1.0,
        max_timeline_segments=16,
    )


def _store_pipeline_notice(kind: str, message: str, payload: dict | None = None) -> None:
    st.session_state["pipeline_notice"] = {
        "kind": kind,
        "message": message,
        "payload": payload or {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _render_pipeline_notice() -> None:
    notice = st.session_state.get("pipeline_notice")
    if not notice:
        return
    kind = notice.get("kind", "info")
    message = str(notice.get("message", "")).strip()
    payload = notice.get("payload") or {}
    if kind == "success":
        st.success(message)
    elif kind == "error":
        st.error(message)
    else:
        st.info(message)
    if payload:
        with st.expander("Latest pipeline result", expanded=False):
            st.json(payload)


def _refresh_after_pipeline(bundle_dir: str, notice_kind: str, message: str, payload: dict | None = None) -> None:
    _load_bundle_stats.clear()
    st.session_state["bundle_dir"] = bundle_dir
    st.session_state["search_results"] = None
    st.session_state["last_updated"] = datetime.now(timezone.utc)
    _store_pipeline_notice(notice_kind, message, payload)
    st.rerun()


def _run_hospital_rebuild(bundle_dir: str) -> None:
    with st.spinner("Đang dựng lại metadata chi tiết cho 31 video NVIDIA Hospital..."):
        summary = bootstrap_nvidia_hospital_dataset(
            nvidia_root=_default_nvidia_hospital_root(),
            config=_hospital_ingest_config(),
            recorded_start="2026-01-01T00:00:00Z",
            limit=31,
        )
        manifest = rebuild_runtime_bundle(_hospital_runtime_config(bundle_dir))
    _refresh_after_pipeline(
        bundle_dir,
        "success",
        f"Đã rebuild {summary['processed_videos']} video NVIDIA Hospital và cập nhật bundle với {manifest['moment_count']} person-tracks.",
        {"bootstrap": summary, "manifest": manifest},
    )


def _save_uploaded_video(uploaded_file) -> Path:
    upload_dir = project_root() / ".streamlit-cache" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_path = upload_dir / f"{timestamp}_{Path(uploaded_file.name).stem}{suffix}"
    with target_path.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return target_path


def _run_hospital_add_video(bundle_dir: str, uploaded_video, camera_id: str, start_time: str) -> None:
    source_path = _save_uploaded_video(uploaded_video)
    with st.spinner("Đang ingest video mới, convert H.265, sinh metadata theo từng người, rồi rebuild bundle..."):
        ingest_summary = ingest_hospital_video(
            source_path=source_path,
            config=_hospital_ingest_config(),
            camera_id=camera_id.strip() or None,
            recorded_start=start_time.strip() or None,
        )
        manifest = rebuild_runtime_bundle(_hospital_runtime_config(bundle_dir))
    _refresh_after_pipeline(
        bundle_dir,
        "success",
        f"Đã add video mới `{ingest_summary['video_id']}` và cập nhật bundle với {manifest['moment_count']} person-tracks.",
        {"ingest": ingest_summary, "manifest": manifest},
    )


def _render_hospital_pipeline_panel(bundle_dir: str) -> None:
    st.markdown(
        """
        <div class="panel panel-strong" style="margin: 0.6rem 0 1rem;">
            <div class="section-title">Hospital pipeline controls</div>
            <p class="section-copy">
                Rebuild toàn bộ metadata cho 31 video NVIDIA Hospital, hoặc add đúng 1 video mới để pipeline chỉ xử lý video đó,
                convert sang H.265, sinh person metadata chi tiết, rồi cập nhật bundle query.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_pipeline_notice()

    rebuild_col, add_col = st.columns(2, gap="large")
    with rebuild_col:
        st.markdown(
            """
            <div class="panel">
                <div class="section-title">Rebuild 31 video nền</div>
                <p class="section-copy">
                    Dùng lại 31 camera NVIDIA Hospital có sẵn, convert sang H.265, sinh metadata per-person chi tiết, rồi rebuild bundle.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Run 31-video rebuild", type="primary", width="stretch"):
            try:
                _run_hospital_rebuild(bundle_dir)
            except Exception as exc:
                _store_pipeline_notice("error", f"Không thể rebuild 31 video: {exc}")
                st.rerun()

    with add_col:
        st.markdown(
            """
            <div class="panel">
                <div class="section-title">Add video mới</div>
                <p class="section-copy">
                    Chỉ ingest đúng video mới, tự convert sang H.265, detect người, sinh metadata chi tiết theo track, rồi cập nhật bundle.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("hospital_add_video_form", clear_on_submit=True):
            uploaded_video = st.file_uploader(
                "Video mới",
                type=["mp4", "mov", "avi", "mkv", "h265", "hevc"],
                help="Có thể upload video gốc; pipeline sẽ tự convert sang H.265 để lưu trong queue.",
            )
            camera_id = st.text_input(
                "Camera ID",
                value="",
                help="Để trống nếu muốn dùng tên file làm camera ID.",
            )
            start_time = st.text_input(
                "Recorded start UTC",
                value="",
                placeholder="2026-01-01T00:10:00Z",
                help="Không bắt buộc. Nếu để trống, hệ thống sẽ dùng thời gian file hiện có.",
            )
            submit_add = st.form_submit_button("Add new video", type="primary", width="stretch")

        if submit_add:
            if uploaded_video is None:
                st.warning("Chọn một video trước khi bấm add.")
            else:
                try:
                    _run_hospital_add_video(bundle_dir, uploaded_video, camera_id, start_time)
                except Exception as exc:
                    _store_pipeline_notice("error", f"Không thể add video mới: {exc}")
                    st.rerun()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --bg: #f5f3ee;
            --surface: rgba(255, 255, 255, 0.82);
            --surface-strong: rgba(255, 255, 255, 0.96);
            --ink: #162126;
            --muted: #607078;
            --line: rgba(22, 33, 38, 0.12);
            --accent: #0f766e;
            --accent-soft: rgba(15, 118, 110, 0.10);
            --warn: #a16207;
            --warn-soft: rgba(161, 98, 7, 0.10);
            --shadow: 0 18px 48px rgba(17, 24, 39, 0.08);
            --radius: 24px;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 30%),
                radial-gradient(circle at top right, rgba(20, 83, 45, 0.06), transparent 24%),
                linear-gradient(180deg, #f7f6f1 0%, #f1eee7 100%);
            color: var(--ink);
        }

        html, body, [class*="css"]  {
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        }

        h1, h2, h3, h4 {
            font-family: "Manrope", "IBM Plex Sans", sans-serif;
            letter-spacing: -0.02em;
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: rgba(251, 250, 246, 0.96);
            border-right: 1px solid rgba(22, 33, 38, 0.08);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label {
            color: var(--ink);
        }

        [data-testid="stTextInputRootElement"] input,
        [data-testid="stFileUploader"] section,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        [data-testid="stSlider"] {
            border-radius: 18px !important;
        }

        [data-testid="stButton"] button,
        [data-testid="stFormSubmitButton"] button {
            border-radius: 999px !important;
            border: 0 !important;
            background: linear-gradient(135deg, #0f766e 0%, #115e59 100%) !important;
            color: white !important;
            font-weight: 700 !important;
            padding: 0.8rem 1.2rem !important;
            box-shadow: 0 10px 24px rgba(15, 118, 110, 0.22);
        }

        [data-testid="stButton"] button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            filter: brightness(1.03);
        }

        [data-testid="stTabs"] button {
            font-weight: 600;
        }

        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow);
        }

        [data-testid="stAlert"] {
            border-radius: 18px;
        }

        .hero-shell {
            background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,255,255,0.78));
            border: 1px solid rgba(22, 33, 38, 0.08);
            border-radius: 28px;
            padding: 1.6rem 1.7rem;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
            backdrop-filter: blur(10px);
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.74rem;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 0.55rem;
        }

        .hero-title {
            font-family: "Manrope", sans-serif;
            font-size: 3rem;
            line-height: 1.02;
            font-weight: 800;
            margin: 0;
            color: var(--ink);
        }

        .hero-copy {
            margin-top: 0.85rem;
            font-size: 1rem;
            line-height: 1.65;
            color: var(--muted);
            max-width: 60rem;
        }

        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-weight: 700;
            font-size: 0.84rem;
        }

        .status-chip.warn {
            background: var(--warn-soft);
            color: var(--warn);
        }

        .panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 1.2rem 1.25rem;
            box-shadow: var(--shadow);
        }

        .panel-strong {
            background: var(--surface-strong);
        }

        .section-title {
            font-family: "Manrope", sans-serif;
            font-size: 1.18rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .section-copy {
            color: var(--muted);
            line-height: 1.6;
            margin-bottom: 0;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 0.9rem;
        }

        .mini-stat {
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(22, 33, 38, 0.08);
            border-radius: 18px;
            padding: 0.9rem 1rem;
        }

        .mini-stat-label {
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 0.2rem;
        }

        .mini-stat-value {
            color: var(--ink);
            font-size: 1.02rem;
            font-weight: 700;
        }

        .result-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow);
        }

        .result-rank {
            color: var(--accent);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .empty-state {
            background: rgba(255,255,255,0.72);
            border: 1px dashed rgba(22, 33, 38, 0.18);
            border-radius: 28px;
            padding: 2.6rem 2rem;
            text-align: center;
        }

        .empty-title {
            font-family: "Manrope", sans-serif;
            font-size: 1.55rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .empty-copy {
            color: var(--muted);
            max-width: 42rem;
            margin: 0 auto;
            line-height: 1.7;
        }

        .footer-note {
            color: var(--muted);
            font-size: 0.88rem;
            text-align: right;
            padding: 0.8rem 0 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _bundle_chip(bundle: dict) -> str:
    artifacts = bundle.get("artifacts", {})
    if artifacts.get("clip", {}).get("enabled"):
        return '<span class="status-chip">Full multimodal bundle ready</span>'
    return '<span class="status-chip warn">Text-first bundle active</span>'


def _render_header(bundle: dict) -> None:
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="eyebrow">Surveillance Retrieval</div>
            <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;flex-wrap:wrap;">
                <div>
                    <h1 class="hero-title">Find the right track without second-guessing the interface.</h1>
                    <p class="hero-copy">
                        Search camera footage with text, an image, or both. The layout is tuned to show the most
                        important answer first, then let you drill into evidence and ranking detail only when you need it.
                    </p>
                </div>
                <div>{_bundle_chip(bundle)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(bundle_dir: str, bundle: dict, bundle_stats: dict, bundle_ready: bool) -> tuple[str, int, str, bool, bool]:
    with st.sidebar:
        st.markdown("### Search Controls")
        st.caption("Keep the sidebar for setup and filters so the main canvas can focus on evidence.")
        bundle_dir = st.text_input("Bundle directory", bundle_dir)
        search_mode = st.selectbox(
            "Search mode",
            options=["default", "person"],
            index=1,
            help="Use person mode when the query is about people or pedestrian tracks.",
        )
        top_k = st.slider("Results to rank", min_value=3, max_value=20, value=5)

        with st.expander("Bundle status", expanded=True):
            artifacts = bundle.get("artifacts", {})
            if bundle_ready:
                st.success("Bundle loaded and query-ready.")
            else:
                st.warning("Bundle chưa sẵn sàng. Bạn có thể rebuild 31 video hoặc add video mới ở panel chính.")
            status_cols = st.columns(3)
            status_cols[0].metric("Sparse", "On" if artifacts.get("sparse", {}).get("enabled") else "Off")
            status_cols[1].metric("Dense", "On" if artifacts.get("dense", {}).get("enabled") else "Off")
            status_cols[2].metric("CLIP", "On" if artifacts.get("clip", {}).get("enabled") else "Off")
            coverage_cols = st.columns(2)
            coverage_cols[0].metric("Indexed tracks", f"{bundle_stats['total_moments']:,}")
            coverage_cols[1].metric("Preview coverage", f"{bundle_stats['visual_ratio'] * 100:.1f}%")
            if artifacts.get("dense", {}).get("model_name"):
                st.caption(f"Dense model: `{artifacts['dense']['model_name']}`")
            if artifacts.get("clip", {}).get("enabled"):
                st.caption(
                    "Visual search is active, but preview coverage depends on extracted assets. Some ranked tracks may still have no frame preview."
                )
            else:
                st.caption(
                    "Visual search is off in this bundle. Results will lean on sparse and dense text evidence."
                )

        with st.expander("Advanced guidance", expanded=False):
            qwen_parser_available = qwen_feature_available("parser")
            qwen_reranker_available = qwen_feature_available("reranker")
            use_qwen_parser = st.checkbox(
                "Use local Qwen query parser",
                value=qwen_parser_available,
                disabled=not qwen_parser_available,
                help="Best for Vietnamese or attribute-heavy person queries. The parser rewrites the query into structured surveillance constraints.",
            )
            use_qwen_reranker = st.checkbox(
                "Use local Qwen reranker",
                value=qwen_reranker_available,
                disabled=not qwen_reranker_available,
                help="Slower, but usually better for final ranking quality on person queries because it judges the top candidates directly.",
            )
            st.markdown(
                """
                - Use short, concrete language for the most reliable matches.
                - `person` mode is best for people-centric retrieval.
                - If the dataset only labels generic people, the app will warn when attributes like clothing color are unsupported.
                """
            )
            if qwen_parser_available or qwen_reranker_available:
                st.caption(
                    "Qwen local stack is available. For accuracy-first Vietnamese search, keep both options enabled."
                )
            else:
                st.caption(
                    "No local Qwen models were found under `./models`, so the app will stay on the faster retrieval-only path."
                )

    return bundle_dir, top_k, search_mode, use_qwen_parser, use_qwen_reranker


def _hero_metric(label: str, value: str, hint: str) -> None:
    st.markdown(
        f"""
        <div class="mini-stat">
            <div class="mini-stat-label">{html.escape(label)}</div>
            <div class="mini-stat-value">{html.escape(value)}</div>
            <div class="mini-stat-label">{html.escape(hint)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.0f}%"


def _query_requested(item: dict, kind: str) -> bool:
    requirements = item.get("query_requirements") or {}
    if kind == "attribute":
        return bool(requirements.get("attribute_terms"))
    if kind == "scene":
        return bool(requirements.get("scene_terms"))
    return False


def _format_evidence_metric(item: dict, kind: str) -> str:
    if kind == "person":
        return _format_percent(item.get("person_match"))
    if not _query_requested(item, kind):
        return "Not asked"
    field = "attribute_match" if kind == "attribute" else "scene_match"
    return _format_percent(item.get(field))


def _render_search_form(bundle: dict) -> tuple[str, str | None, bool]:
    st.markdown(
        """
        <div class="panel panel-strong">
            <div class="section-title">Start with one clear search prompt</div>
            <p class="section-copy">
                Describe the track you want to find, then optionally add a reference image. The interface will show the top match first and keep ranking diagnostics below.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("search_form", clear_on_submit=False):
        query_text = st.text_input(
            "Describe what you want to find",
            value=st.session_state.get("query_text", DEFAULT_QUERY),
            placeholder="person in red shirt near the road",
            help="Keep the wording natural and specific, but remember the dataset may not label every visual attribute.",
        )
        uploader_col, submit_col = st.columns([1.7, 0.7])
        with uploader_col:
            uploaded = st.file_uploader(
                "Optional reference image",
                type=["jpg", "jpeg", "png"],
                help="Use this when you want the ranking to incorporate a visual cue.",
            )
        with submit_col:
            st.caption("Primary action")
            run = st.form_submit_button("Run Search", type="primary", width="stretch")

    image_path = None
    if uploaded is not None:
        temp_dir = project_root() / ".streamlit-cache"
        temp_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(uploaded.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=suffix, delete=False) as handle:
            handle.write(uploaded.read())
            image_path = handle.name

    return query_text, image_path, run


def _render_empty_state(
    title: str = "Search results will appear here.",
    message: str = (
        "Start with a short query such as “person crossing the street” or “white van near the curb”. "
        "The app will surface one hero result first, then show ranking detail and diagnostics only if you want to inspect them."
    ),
) -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-title">{title}</div>
            <p class="empty-copy">{message}</p>
        </div>
        """.format(title=html.escape(title), message=html.escape(message)),
        unsafe_allow_html=True,
    )


def _render_media(path: str | None, title: str, empty_message: str) -> None:
    if path and Path(path).exists():
        st.image(path, caption=title, width="stretch")
    else:
        st.markdown(
            f"""
            <div class="panel" style="min-height: 220px; display:flex; align-items:center; justify-content:center;">
                <div style="text-align:center;">
                    <div class="section-title" style="font-size:1rem;">{html.escape(title)}</div>
                    <p class="section-copy">{html.escape(empty_message)}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _answer_window_text(item: dict) -> str | None:
    window = item.get("answer_window") or {}
    if not window:
        return None
    return (
        f"Answer window: `{window.get('start_second', 0):.2f}s` to "
        f"`{window.get('end_second', 0):.2f}s`"
    )


def _result_summary_text(item: dict) -> str:
    frame_info = item.get("frame_info", {})
    location = frame_info.get("location") or item.get("location") or "unknown"
    split = frame_info.get("split") or item.get("split") or "unknown"
    track_id = frame_info.get("track_id") or item.get("track_id") or "unknown"
    start_second = frame_info.get("start_second", item.get("start_second", 0))
    end_second = frame_info.get("end_second", item.get("end_second", 0))
    return (
        f"Track `{track_id}` in `{location}` / `{split}` spanning "
        f"`{start_second:.2f}s` to `{end_second:.2f}s`."
    )


def _result_has_preview(item: dict) -> bool:
    for key in ("answer_visual_path", "answer_context_frame_path", "frame_path", "visual_path", "crop_path"):
        value = item.get(key)
        if value and Path(value).exists():
            return True
    return False


def _is_generic_only_result(item: dict) -> bool:
    return item.get("match_quality") == "generic_person_only"


def _evidence_list_text(values: list[str] | None, empty: str = "None verified") -> str:
    if not values:
        return empty
    return ", ".join(str(value) for value in values)


def _derive_search_assessment(results: list[dict], search_mode: str) -> dict:
    generic_only = bool(results) and all(_is_generic_only_result(item) for item in results)
    no_preview_in_ranked = bool(results) and not any(_result_has_preview(item) for item in results)

    if search_mode == "person" and generic_only:
        title = "No exact attribute-level match found"
        summary = (
            "The system found person tracks, but this dataset does not contain evidence for the requested attributes. "
            "The ranked results below are fallback person-only matches, not verified matches for clothing color or road proximity."
        )
        tone = "warning"
    else:
        title = "Best available match"
        summary = (
            "The hero result is the strongest ranked track under the current bundle, query, and search mode."
        )
        tone = "info"

    if no_preview_in_ranked:
        summary = (
            f"{summary} This query's current top results also have no extracted previews, so visual evidence is limited in the UI."
        )

    return {
        "title": title,
        "summary": summary,
        "tone": tone,
        "generic_only": generic_only,
        "no_preview_in_ranked": no_preview_in_ranked,
    }


def _render_top_match(item: dict, query_text: str, assessment: dict) -> None:
    left, right = st.columns([1.45, 0.95], gap="large")

    with left:
        st.markdown(
            """
            <div class="panel panel-strong">
                <div class="section-title">Top match</div>
                <p class="section-copy">The strongest ranked result is shown first so you can judge relevance without scanning the whole page.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if assessment["tone"] == "warning":
            st.warning(assessment["summary"])
        else:
            st.info(assessment["summary"])
        _render_media(
            item.get("answer_visual_path") or item.get("frame_path") or item.get("visual_path"),
            "Best answer frame",
            "No frame preview is available for this top-ranked track in the current bundle.",
        )
        if item.get("answer_context_frame_path"):
            st.image(item["answer_context_frame_path"], caption="Context frame", width="stretch")
        crop_path = item.get("crop_path")
        if crop_path and Path(crop_path).exists() and crop_path != item.get("answer_visual_path"):
            st.image(crop_path, caption="Track crop", width="stretch")

    with right:
        st.markdown(
            f"""
            <div class="panel panel-strong">
                <div class="eyebrow">Query summary</div>
                <div class="section-title" style="font-size:1.6rem;">{html.escape(item.get('caption_text', 'Top result'))}</div>
                <p class="section-copy">{html.escape(_result_summary_text(item))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if item.get("result_warning"):
            st.warning(item["result_warning"])
        answer_window = _answer_window_text(item)
        if answer_window:
            st.caption(answer_window)

        stats_one, stats_two = st.columns(2, gap="small")
        with stats_one:
            quality_map = {
                "generic_person_only": "Generic only",
                "partially_verified": "Partial evidence",
                "fully_verified": "Fully verified",
                "unverified": "Unverified",
            }
            quality = quality_map.get(item.get("match_quality"), "Evidence-backed")
            _hero_metric("Match quality", quality, "What kind of answer this result represents.")
            _hero_metric("Person evidence", _format_evidence_metric(item, "person"), "How strongly the result looks like a person track.")
        with stats_two:
            _hero_metric("Attribute evidence", _format_evidence_metric(item, "attribute"), "How much of the requested appearance evidence is verified.")
            _hero_metric("Scene evidence", _format_evidence_metric(item, "scene"), "How much of the requested scene relation evidence is verified.")

        st.caption(f"Verified attributes: {_evidence_list_text(item.get('verified_attributes'))}")
        st.caption(f"Scene relations: {_evidence_list_text(item.get('verified_scene_relations'))}")
        if item.get("query_profile"):
            st.caption(f"Query intent: {item['query_profile'].get('primary_intent', 'unknown')}")
        if item.get("structured_query"):
            st.caption(f"Qwen parsed query: {json.dumps(item['structured_query'], ensure_ascii=False)}")
        if item.get("qwen_rerank_used"):
            st.caption(f"Qwen rerank score: {item.get('qwen_rerank_score', 0.0):.3f}")

        st.markdown(
            f"""
            <div class="panel" style="margin-top:0.85rem;">
                <div class="section-title" style="font-size:1rem;">Query</div>
                <p class="section-copy">{html.escape(query_text or 'Image-only search')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Text coverage: {_format_percent(item.get('precision_match'))} • Search mode: {item.get('search_mode', 'default').title()}")
        if item.get("expanded_query"):
            st.caption(f"Expanded internally as: {item['expanded_query']}")


def _render_ranked_result(item: dict, rank: int) -> None:
    label = item.get("caption_text", "Result")
    meta = _result_summary_text(item)
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="result-rank">Rank {rank}</div>
            <div class="section-title">{html.escape(label)}</div>
            <p class="section-copy">{html.escape(meta)}</p>
            """,
            unsafe_allow_html=True,
        )
        quick_cols = st.columns(4)
        quick_cols[0].metric("Rank score", f"{item['score']:.4f}")
        quick_cols[1].metric("Person", _format_evidence_metric(item, "person"))
        quick_cols[2].metric("Attributes", _format_evidence_metric(item, "attribute"))
        quick_cols[3].metric("Scene", _format_evidence_metric(item, "scene"))
        answer_window = _answer_window_text(item)
        if answer_window:
            st.caption(answer_window)
        if item.get("result_warning"):
            st.caption(item["result_warning"])
        st.caption(
            f"Track `{item.get('track_id', 'N/A')}` • Verified attributes: {_evidence_list_text(item.get('verified_attributes'))} • Scene: {_evidence_list_text(item.get('verified_scene_relations'))}"
        )
        with st.expander("See details", expanded=False):
            media_col, detail_col = st.columns([1.1, 0.9], gap="large")
            with media_col:
                _render_media(
                    item.get("answer_visual_path") or item.get("frame_path") or item.get("visual_path"),
                    "Best answer frame",
                    "This result does not have a frame preview in the current bundle.",
                )
            with detail_col:
                st.json(
                    {
                        "frame_info": item.get("frame_info", {}),
                        "captions": item.get("captions", []),
                        "video_path": item.get("video_path"),
                        "score_breakdown": item.get("score_breakdown", {}),
                        "normalized_breakdown": item.get("normalized_breakdown", {}),
                        "fusion_breakdown": item.get("fusion_breakdown", {}),
                        "precision_match": item.get("precision_match"),
                        "person_match": item.get("person_match"),
                        "attribute_match": item.get("attribute_match"),
                        "scene_match": item.get("scene_match"),
                        "verified_attributes": item.get("verified_attributes"),
                        "verified_scene_relations": item.get("verified_scene_relations"),
                        "query_profile": item.get("query_profile"),
                        "answer_visual_path": item.get("answer_visual_path"),
                        "answer_asset_type": item.get("answer_asset_type"),
                        "answer_frame_idx": item.get("answer_frame_idx"),
                        "answer_second": item.get("answer_second"),
                        "answer_window": item.get("answer_window"),
                        "answer_selection": item.get("answer_selection"),
                        "query_requirements": item.get("query_requirements"),
                        "attribute_evidence": item.get("attribute_evidence"),
                        "scene_evidence": item.get("scene_evidence"),
                        "match_quality": item.get("match_quality"),
                    }
                )


def _render_overview(results: list[dict], query_text: str, search_mode: str) -> None:
    top = results[0]
    assessment = _derive_search_assessment(results, search_mode)
    metrics = st.columns(4)
    metrics[0].metric("Results shown", str(len(results)))
    metrics[1].metric("Result type", "Generic only" if assessment["generic_only"] else "Best match")
    metrics[2].metric("Best person evidence", _format_evidence_metric(top, "person"))
    if _query_requested(top, "attribute") or _query_requested(top, "scene"):
        metrics[3].metric("Verified evidence", _format_percent(max(top.get("attribute_match") or 0.0, top.get("scene_match") or 0.0)))
    else:
        metrics[3].metric("Verified evidence", "Not asked")

    if search_mode == "person":
        st.caption(
            "Person mode prioritizes people tracks first, then separates text coverage from verified appearance and scene evidence."
        )

    _render_top_match(top, query_text, assessment)


def _render_diagnostics(results: list[dict], bundle: dict) -> None:
    top = results[0]
    diag_left, diag_right = st.columns([1.0, 1.0], gap="large")
    with diag_left:
        st.markdown(
            """
            <div class="panel">
                <div class="section-title">Bundle capabilities</div>
                <p class="section-copy">A quick read on which retrieval branches are active in the loaded bundle.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.json(bundle.get("artifacts", {}))

    with diag_right:
        st.markdown(
            """
            <div class="panel">
                <div class="section-title">Top-match diagnostics</div>
                <p class="section-copy">Use this when you want to inspect how the ranking was composed.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.json(
            {
                "score_breakdown": top.get("score_breakdown", {}),
                "normalized_breakdown": top.get("normalized_breakdown", {}),
                "fusion_breakdown": top.get("fusion_breakdown", {}),
                "weights": top.get("weights", {}),
                "expanded_query": top.get("expanded_query"),
                "match_quality": top.get("match_quality"),
                "attribute_match": top.get("attribute_match"),
                "scene_match": top.get("scene_match"),
                "verified_attributes": top.get("verified_attributes"),
                "verified_scene_relations": top.get("verified_scene_relations"),
                "structured_query": top.get("structured_query"),
                "qwen_parser_used": top.get("qwen_parser_used"),
                "qwen_reranker_used": top.get("qwen_reranker_used"),
                "qwen_rerank_score": top.get("qwen_rerank_score"),
                "qwen_parser_error": top.get("qwen_parser_error"),
                "qwen_reranker_error": top.get("qwen_reranker_error"),
                "query_requirements": top.get("query_requirements"),
                "query_profile": top.get("query_profile"),
                "answer_visual_path": top.get("answer_visual_path"),
                "answer_asset_type": top.get("answer_asset_type"),
                "answer_frame_idx": top.get("answer_frame_idx"),
                "answer_second": top.get("answer_second"),
                "answer_window": top.get("answer_window"),
                "answer_selection": top.get("answer_selection"),
            }
        )


def _run_search(
    bundle_dir: str,
    query_text: str,
    image_path: str | None,
    top_k: int,
    search_mode: str,
    use_qwen_parser: bool,
    use_qwen_reranker: bool,
) -> None:
    with st.spinner("Ranking the most relevant tracks..."):
        results = search_index(
            Path(bundle_dir),
            query_text=query_text or None,
            query_image_path=image_path,
            top_k=top_k,
            search_mode=search_mode,
            use_qwen_parser=use_qwen_parser,
            use_qwen_reranker=use_qwen_reranker,
        )
    st.session_state["search_results"] = results
    st.session_state["query_text"] = query_text
    st.session_state["search_mode"] = search_mode
    st.session_state["last_updated"] = datetime.now(timezone.utc)


def main() -> None:
    st.set_page_config(page_title="Surveillance Search", layout="wide")
    _inject_styles()

    st.session_state.setdefault("bundle_dir", str(default_bundle_root()))
    st.session_state.setdefault("search_results", None)
    st.session_state.setdefault("query_text", DEFAULT_QUERY)
    st.session_state.setdefault("search_mode", "person")
    st.session_state.setdefault("last_updated", None)

    bundle_dir = st.session_state["bundle_dir"]
    bundle, bundle_stats, bundle_ready, bundle_error = _load_bundle_state(bundle_dir)

    sidebar_bundle_dir, top_k, search_mode, use_qwen_parser, use_qwen_reranker = _render_sidebar(
        bundle_dir,
        bundle,
        bundle_stats,
        bundle_ready,
    )
    if sidebar_bundle_dir != bundle_dir:
        bundle_dir = sidebar_bundle_dir
        st.session_state["bundle_dir"] = bundle_dir
        bundle, bundle_stats, bundle_ready, bundle_error = _load_bundle_state(bundle_dir)

    _render_header(bundle)
    _render_hospital_pipeline_panel(bundle_dir)

    if not bundle_ready:
        if bundle_error:
            st.info(f"Bundle hiện chưa load được từ `{bundle_dir}`: {bundle_error}")
        _render_empty_state(
            title="Bundle chưa sẵn sàng để query.",
            message=(
                "Bấm `Run 31-video rebuild` để dựng sẵn dữ liệu cho 31 video NVIDIA Hospital, "
                "hoặc upload 1 video mới ở panel trên để ingest incremental rồi build bundle."
            ),
        )
        return

    query_text, image_path, run = _render_search_form(bundle)

    if run:
        if not query_text and not image_path:
            st.warning("Nhập mô tả hoặc thêm ảnh tham chiếu để bắt đầu.")
        else:
            try:
                _run_search(
                    bundle_dir,
                    query_text,
                    image_path,
                    top_k,
                    search_mode,
                    use_qwen_parser,
                    use_qwen_reranker,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                bundle, bundle_stats, bundle_ready, bundle_error = _load_bundle_state(bundle_dir)

    results = st.session_state.get("search_results")
    if not results:
        _render_empty_state()
    else:
        tabs = st.tabs(["Overview", "Ranked Results", "Diagnostics"])
        with tabs[0]:
            _render_overview(results, st.session_state.get("query_text", ""), search_mode)
        with tabs[1]:
            for rank, item in enumerate(results, start=1):
                _render_ranked_result(item, rank)
        with tabs[2]:
            _render_diagnostics(results, bundle)

    updated_at = st.session_state.get("last_updated")
    if updated_at is not None:
        timestamp = updated_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        st.markdown(
            f'<div class="footer-note">Last search updated {html.escape(timestamp)}</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
