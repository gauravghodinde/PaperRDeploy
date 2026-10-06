"""
app.py — Read · See · Synthesize
==================================
Unified Streamlit application that runs all three pipeline stages:

  Stage 1 (Read)      : PDF upload → title, abstract, supp/comp extraction
  Stage 2 (See)       : ML classification of graphical-abstract evidence
  Stage 3 (Synthesize): Multi-LLM related work generation (Gemini, Llama, GPT, DeepSeek, Ensemble)

Run
---
  cd d:/phdd/final/relative_work_generation
  streamlit run app.py
"""

import os
import sys
import json
import tempfile
import time
from pathlib import Path
from io import BytesIO

import streamlit as st

# ── PATH SETUP ─────────────────────────────────────────────────────────────
_APP_DIR      = Path(__file__).parent.resolve()          # …/relative_work_generation
_PROJECT_ROOT = _APP_DIR.parent                          # …/final
_CLASS_DIR    = _PROJECT_ROOT / "classification"

# Make sure both source trees are importable
for p in [str(_APP_DIR), str(_CLASS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── PAGE CONFIG (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Read · See · Synthesize",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');

:root {
    --bg-dark:      #0a0e1a;
    --bg-card:      #111827;
    --bg-elevated:  #1a2235;
    --accent-blue:  #3b82f6;
    --accent-purple:#8b5cf6;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-red:   #ef4444;
    --text-primary: #f1f5f9;
    --text-muted:   #94a3b8;
    --border:       rgba(59,130,246,0.15);
    --glow-blue:    rgba(59,130,246,0.25);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-dark);
    color: var(--text-primary);
}

/* Main header */
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    text-align: center;
    color: var(--text-muted);
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* Stage badge */
.stage-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.9rem;
    border-radius: 2rem;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.stage-read    { background: rgba(59,130,246,0.15);  color: #60a5fa; border: 1px solid rgba(59,130,246,0.3); }
.stage-see     { background: rgba(139,92,246,0.15);  color: #a78bfa; border: 1px solid rgba(139,92,246,0.3); }
.stage-synth   { background: rgba(16,185,129,0.15);  color: #34d399; border: 1px solid rgba(16,185,129,0.3); }

/* Cards */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.card-elevated {
    background: var(--bg-elevated);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 0 32px rgba(139,92,246,0.08);
}

/* Model tab pills */
.model-pill {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 1rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.4rem;
}
.pill-gemini  { background: rgba(59,130,246,0.15);  color: #60a5fa; }
.pill-llama   { background: rgba(249,115,22,0.15);  color: #fb923c; }
.pill-gpt     { background: rgba(16,185,129,0.15);  color: #34d399; }
.pill-deepseek{ background: rgba(168,85,247,0.15);  color: #c084fc; }
.pill-ensemble{ background: rgba(245,158,11,0.15);  color: #fbbf24; }

/* Metric card */
.metric-box {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value { font-size: 1.6rem; font-weight: 700; color: var(--accent-blue); }
.metric-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }

/* Status pill */
.status-supp { color: #60a5fa; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.2);
               border-radius: 8px; padding: 0.2rem 0.6rem; font-size: 0.75rem; font-weight: 600; }
.status-comp { color: #34d399; background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.2);
               border-radius: 8px; padding: 0.2rem 0.6rem; font-size: 0.75rem; font-weight: 600; }

/* Streamlit widget overrides */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px var(--glow-blue) !important;
}
.stTextArea textarea, .stTextInput input {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
.stSelectbox select, div[data-baseweb="select"] {
    background: var(--bg-elevated) !important;
    border-color: var(--border) !important;
}
.stTab [data-baseweb="tab"] {
    color: var(--text-muted) !important;
}
.stTab [aria-selected="true"] {
    color: var(--accent-blue) !important;
}
hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE INIT ───────────────────────────────────────────────────────
defaults = {
    "stage1_done":   False,
    "stage2_done":   False,
    "stage3_done":   False,
    "paper_info":    {},
    "see_output":    None,
    "pipeline_out":  None,
    "pipeline_obj":  None,
    "tmp_pdf_path":  None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── HELPERS ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_pipeline():
    """Load the main pipeline (cached across reruns)."""
    try:
        from src.pipeline import RelatedWorkPipeline
        return RelatedWorkPipeline(verbose=False)
    except Exception as e:
        return None

@st.cache_resource(show_spinner=False)
def _load_see_stage():
    """Load SeeStage classifier (cached)."""
    try:
        from src.see_stage import SeeStage
        return SeeStage(auto_load=True)
    except Exception as e:
        return None

def _save_tmp_pdf(uploaded_file) -> str:
    """Save uploaded PDF to a temp file and return the path."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

def _show_card(content: str, title: str = "", color: str = "blue"):
    border_map = {"blue": "#3b82f6", "purple": "#8b5cf6", "green": "#10b981", "amber": "#f59e0b"}
    border = border_map.get(color, "#3b82f6")
    title_html = f"<div style='font-size:0.75rem;color:{border};font-weight:600;letter-spacing:.05em;text-transform:uppercase;margin-bottom:0.5rem'>{title}</div>" if title else ""
    st.markdown(f"""
    <div style='background:#111827;border:1px solid {border}30;border-left:3px solid {border};
                border-radius:10px;padding:1.2rem 1.4rem;margin-bottom:0.8rem;'>
        {title_html}
        <div style='color:#e2e8f0;font-size:0.92rem;line-height:1.7;white-space:pre-wrap;'>{content}</div>
    </div>
    """, unsafe_allow_html=True)

def _metric_row(metrics: list):
    """Display a row of metric boxes. Each item: (label, value, color?)."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{value}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**🔑 API Keys**")
    google_key = st.text_input("Google API Key", type="password",
                               value=os.environ.get("GOOGLE_API_KEY", ""),
                               help="Required for Gemini models")
    groq_key   = st.text_input("Groq API Key",   type="password",
                               value=os.environ.get("GROQ_API_KEY", ""),
                               help="Required for Llama & GPT-OSS-120b (free tier)")
    deepseek_key = st.text_input("DeepSeek API Key", type="password",
                                 value=os.environ.get("DEEPSEEK_API_KEY", ""),
                                 help="Optional – for DeepSeek model")
    s2_key     = st.text_input("Semantic Scholar Key", type="password",
                               value=os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
                               help="Optional but recommended")

    if st.button("💾 Save Keys to Environment"):
        from src.utils.config import settings
        if google_key:   
            os.environ["GOOGLE_API_KEY"] = google_key
            settings.GOOGLE_API_KEY = google_key
        if groq_key:     
            os.environ["GROQ_API_KEY"]   = groq_key
            settings.GROQ_API_KEY = groq_key
        if deepseek_key: 
            os.environ["DEEPSEEK_API_KEY"] = deepseek_key
            settings.DEEPSEEK_API_KEY = deepseek_key
        if s2_key:       
            os.environ["SEMANTIC_SCHOLAR_API_KEY"] = s2_key
            settings.SEMANTIC_SCHOLAR_API_KEY = s2_key
        st.success("Keys saved to session environment ✅")

    st.markdown("---")
    st.markdown("**📊 Pipeline Settings**")
    num_papers  = st.slider("Related papers to retrieve", 5, 30, 15, 5)
    year_from   = st.number_input("Earliest publication year", min_value=2000,
                                  max_value=2025, value=2020, step=1)

    st.markdown("---")
    st.markdown("**🤖 See Stage**")
    verify_labels = st.toggle("Re-classify with ML model", value=True,
                              help="Use the trained RF classifier to verify LLM-assigned labels")

    st.markdown("---")
    st.markdown("""
    <div style='color:#64748b;font-size:0.78rem;line-height:1.6'>
    <b>Read · See · Synthesize</b><br>
    PhD Research Pipeline<br>
    Multi-LLM Related Work Generation<br><br>
    Stage 1: PDF → Title / Abstract / Evidence<br>
    Stage 2: ML-classify Supp / Comp units<br>
    Stage 3: Gemini + Llama + GPT + DeepSeek → Ensemble
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("<div class='hero-title'>Read · See · Synthesize</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-subtitle'>Multi-LLM Related Work Generation with Graphical Abstract Intelligence</div>",
    unsafe_allow_html=True
)

# Pipeline status bar
col1, col2, col3 = st.columns(3)
with col1:
    color1 = "#3b82f6" if st.session_state.stage1_done else "#374151"
    st.markdown(f"<div style='text-align:center;padding:0.5rem;border-radius:8px;background:{color1}20;border:1px solid {color1}40;color:{color1};font-weight:600'>📖 Stage 1: READ {'✓' if st.session_state.stage1_done else '○'}</div>", unsafe_allow_html=True)
with col2:
    color2 = "#8b5cf6" if st.session_state.stage2_done else "#374151"
    st.markdown(f"<div style='text-align:center;padding:0.5rem;border-radius:8px;background:{color2}20;border:1px solid {color2}40;color:{color2};font-weight:600'>👁️ Stage 2: SEE {'✓' if st.session_state.stage2_done else '○'}</div>", unsafe_allow_html=True)
with col3:
    color3 = "#10b981" if st.session_state.stage3_done else "#374151"
    st.markdown(f"<div style='text-align:center;padding:0.5rem;border-radius:8px;background:{color3}20;border:1px solid {color3}40;color:{color3};font-weight:600'>⚡ Stage 3: SYNTHESIZE {'✓' if st.session_state.stage3_done else '○'}</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 : READ
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📖 STAGE 1 — Read: Upload & Extract", expanded=not st.session_state.stage1_done):
    st.markdown("<div class='stage-badge stage-read'>📖 Stage 1 · Read</div>", unsafe_allow_html=True)
    st.markdown("Upload your target paper PDF. The system will automatically extract the title, abstract, and graphical abstract evidence.")

    uploaded_pdf = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"], key="pdf_upload")

    st.markdown("**Or provide graphical abstract evidence manually:**")
    c1, c2 = st.columns(2)
    with c1:
        supp_override = st.text_area(
            "Supplementary Info (optional override)",
            height=140,
            placeholder="• Hyperparameter settings: lr=1e-4, batch=32\n• Model architecture: 12-layer transformer\n• Dataset: ACL Anthology, 2.7M papers",
            key="supp_override"
        )
    with c2:
        comp_override = st.text_area(
            "Complementary Info (optional override)",
            height=140,
            placeholder="• First system to integrate graphical abstracts into related work generation\n• Introduces supp/comp classification as a novel NLP sub-task\n• Real-world deployment at IEEE submission portal",
            key="comp_override"
        )

    if st.button("🚀 Run Stage 1: Extract from PDF", key="btn_stage1"):
        if uploaded_pdf is None:
            st.error("Please upload a PDF file first.")
        else:
            with st.spinner("Extracting from PDF…"):
                try:
                    # Save to temp file
                    tmp_path = _save_tmp_pdf(uploaded_pdf)
                    st.session_state.tmp_pdf_path = tmp_path

                    pipeline = _load_pipeline()
                    if pipeline is None:
                        st.error("Pipeline failed to load. Check API keys and requirements.")
                    else:
                        result = pipeline.pdf_extractor.extract_all(
                            tmp_path,
                            supplementary_override=supp_override.strip() or None,
                            complementary_override=comp_override.strip() or None,
                        )
                        if "error" in result:
                            st.error(f"Extraction failed: {result['error']}")
                        else:
                            st.session_state.paper_info   = result
                            st.session_state.stage1_done  = True
                            st.session_state.stage2_done  = False
                            st.session_state.stage3_done  = False
                            st.session_state.see_output   = None
                            st.session_state.pipeline_out = None
                            st.success("✅ Stage 1 complete!")
                            st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# Stage 1 result preview
if st.session_state.stage1_done:
    pi = st.session_state.paper_info
    with st.container():
        st.markdown("### 📄 Extracted Paper Content")
        _metric_row([
            ("Title Length",  f"{len(pi.get('title',''))} chars"),
            ("Abstract",      f"{len(pi.get('abstract',''))} chars"),
            ("Supplementary", f"{len(pi.get('supplementary_info','') or '')} chars"),
            ("Complementary", f"{len(pi.get('complementary_info','') or '')} chars"),
        ])
        if pi.get("title"):
            _show_card(pi["title"], "Title", "blue")
        if pi.get("abstract"):
            _show_card(pi["abstract"][:600] + ("…" if len(pi.get("abstract","")) > 600 else ""), "Abstract", "purple")
        c1, c2 = st.columns(2)
        with c1:
            if pi.get("supplementary_info"):
                _show_card(pi["supplementary_info"][:500] + "…", "Supplementary Info", "blue")
        with c2:
            if pi.get("complementary_info"):
                _show_card(pi["complementary_info"][:500] + "…", "Complementary Info", "green")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 : SEE
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("👁️ STAGE 2 — See: Classify Evidence", expanded=st.session_state.stage1_done and not st.session_state.stage2_done):
    st.markdown("<div class='stage-badge stage-see'>👁️ Stage 2 · See</div>", unsafe_allow_html=True)
    st.markdown("The ML classifier (Random Forest, 96% accuracy) re-classifies each information unit extracted from the graphical abstract.")

    if not st.session_state.stage1_done:
        st.info("Complete Stage 1 first.")
    else:
        if st.button("🔍 Run Stage 2: Classify Evidence", key="btn_stage2"):
            with st.spinner("Classifying evidence units with ML model…"):
                try:
                    see = _load_see_stage()
                    pi  = st.session_state.paper_info
                    if see is not None:
                        output = see.process(
                            abstract          = pi.get("abstract", ""),
                            supplementary_raw = pi.get("supplementary_info", "") or "",
                            complementary_raw = pi.get("complementary_info", "") or "",
                            verify_labels     = verify_labels,
                        )
                        st.session_state.see_output  = output
                        st.session_state.stage2_done = True
                        # Propagate classified text back to paper_info
                        pi["supplementary_info"] = output.supplementary_text or pi.get("supplementary_info", "")
                        pi["complementary_info"] = output.complementary_text or pi.get("complementary_info", "")
                        st.session_state.paper_info = pi
                        st.success("✅ Stage 2 complete!")
                        st.rerun()
                    else:
                        st.warning("SeeStage not available. Skipping to Stage 3.")
                        st.session_state.stage2_done = True
                        st.rerun()
                except Exception as e:
                    st.error(f"Stage 2 error: {e}")

# Stage 2 results
if st.session_state.stage2_done and st.session_state.see_output is not None:
    out = st.session_state.see_output
    st.markdown("### 🔬 Classification Results")
    _metric_row([
        ("Total Units",      str(out.total_units)),
        ("Classified",       str(out.classified_units)),
        ("Supplementary",    str(len(out.supplementary))),
        ("Complementary",    str(len(out.complementary))),
        ("Accuracy Rate",    f"{out.classification_rate:.0%}"),
    ])
    st.markdown(f"**Model:** `{out.model_used}`")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🔵 Supplementary Units** (supporting/validating)")
        for u in out.supplementary:
            conf_bar = "▓" * int(u.confidence * 10) + "░" * (10 - int(u.confidence * 10))
            st.markdown(f"""
            <div style='background:#1e2d45;border-left:3px solid #3b82f6;border-radius:6px;padding:0.6rem 0.9rem;margin-bottom:0.4rem;font-size:0.85rem'>
                <span class='status-supp'>SUPPLEMENTARY</span>
                <span style='float:right;color:#64748b;font-size:0.7rem'>{u.confidence:.0%} {conf_bar}</span>
                <br><span style='color:#e2e8f0'>{u.text}</span>
            </div>
            """, unsafe_allow_html=True)
    with c2:
        st.markdown("**🟢 Complementary Units** (additive/novel)")
        for u in out.complementary:
            conf_bar = "▓" * int(u.confidence * 10) + "░" * (10 - int(u.confidence * 10))
            st.markdown(f"""
            <div style='background:#142a1e;border-left:3px solid #10b981;border-radius:6px;padding:0.6rem 0.9rem;margin-bottom:0.4rem;font-size:0.85rem'>
                <span class='status-comp'>COMPLEMENTARY</span>
                <span style='float:right;color:#64748b;font-size:0.7rem'>{u.confidence:.0%} {conf_bar}</span>
                <br><span style='color:#e2e8f0'>{u.text}</span>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 : SYNTHESIZE
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("⚡ STAGE 3 — Synthesize: Generate Related Work", expanded=st.session_state.stage2_done and not st.session_state.stage3_done):
    st.markdown("<div class='stage-badge stage-synth'>⚡ Stage 3 · Synthesize</div>", unsafe_allow_html=True)
    st.markdown("Four LLMs generate independent related work sections. A consensus-based ensemble creates the final synthesis.")

    if not st.session_state.stage2_done:
        st.info("Complete Stages 1 & 2 first.")
    else:
        if st.button("⚡ Run Stage 3: Generate Related Work", key="btn_stage3"):
            if not (google_key or groq_key):
                st.error("Please provide at least one API key (Google or Groq) in the sidebar.")
            else:
                # Set keys to environment and settings
                from src.utils.config import settings
                if google_key:   
                    os.environ["GOOGLE_API_KEY"] = google_key
                    settings.GOOGLE_API_KEY = google_key
                if groq_key:     
                    os.environ["GROQ_API_KEY"]   = groq_key
                    settings.GROQ_API_KEY = groq_key
                if deepseek_key: 
                    os.environ["DEEPSEEK_API_KEY"] = deepseek_key
                    settings.DEEPSEEK_API_KEY = deepseek_key
                if s2_key:       
                    os.environ["SEMANTIC_SCHOLAR_API_KEY"] = s2_key
                    settings.SEMANTIC_SCHOLAR_API_KEY = s2_key

                progress = st.progress(0, text="Initialising pipeline…")
                status   = st.empty()

                try:
                    pipeline = _load_pipeline()
                    if pipeline is None:
                        st.error("Pipeline failed to load. Check API keys and requirements.")
                    else:
                        pi = st.session_state.paper_info
                        progress.progress(10, "🔑 Extracting keywords…")
                        status.info("Generating search keywords from paper content…")

                        keywords = pipeline.keyword_extractor.extract_keywords(
                            title=pi.get("title", ""),
                            abstract=pi.get("abstract", ""),
                            supplementary=pi.get("supplementary_info"),
                            use_llm=True
                        )

                        progress.progress(25, "🔍 Searching Semantic Scholar…")
                        status.info(f"Searching for {num_papers} related papers…")
                        search_query = " ".join(keywords[:5])
                        papers = pipeline.ss_client.get_top_papers(
                            query=search_query,
                            top_n=num_papers,
                            year_from=year_from
                        )

                        progress.progress(40, "📝 Summarising papers…")
                        status.info(f"Summarising {len(papers)} papers…")
                        summaries = pipeline.summary_chain.summarize_papers(papers, show_progress=False)

                        progress.progress(55, "🔗 Creating compressive summaries…")
                        status.info("Creating compressive summaries…")
                        comp_summaries = pipeline.compressive_chain.create_compressive_summaries(
                            papers=papers, summaries=summaries, show_progress=False
                        )

                        progress.progress(70, "🤖 Running multi-LLM synthesis…")
                        status.info("Generating related work with 4 models + ensemble…")
                        result = pipeline.related_work_chain.generate_with_references(
                            target_title=pi.get("title", ""),
                            target_abstract=pi.get("abstract", ""),
                            compressive_summaries=comp_summaries,
                            papers=papers
                        )

                        progress.progress(100, "✅ Done!")
                        status.empty()

                        # Build final output dict
                        final_output = {
                            "target_paper": pi,
                            "keywords":     keywords,
                            "related_papers": [
                                {"title": p.title, "authors": p.citation_key,
                                 "year": p.year, "citations": p.citation_count}
                                for p in papers
                            ],
                            "model_results": result,
                        }
                        st.session_state.pipeline_out = final_output
                        st.session_state.stage3_done  = True
                        st.success("✅ Stage 3 complete!")
                        st.rerun()

                except Exception as e:
                    progress.empty()
                    status.empty()
                    st.error(f"Stage 3 error: {e}")
                    st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.stage3_done and st.session_state.pipeline_out:
    out = st.session_state.pipeline_out
    results = out.get("model_results", {})
    keywords = out.get("keywords", [])
    rel_papers = out.get("related_papers", [])

    st.markdown("---")
    st.markdown("## 📊 Pipeline Results")

    # Summary metrics
    _metric_row([
        ("Keywords",        str(len(keywords))),
        ("Papers Retrieved",str(len(rel_papers))),
        ("Models Used",     str(sum(1 for m in results.values() if m and m.get("related_work")))),
        ("Ensemble",        "✓" if results.get("ensemble", {}).get("related_work") else "✗"),
    ])

    # Keywords chips
    if keywords:
        chips = " ".join([
            f"<span style='background:#1e3a5f;color:#60a5fa;border-radius:6px;padding:0.2rem 0.6rem;margin:0.2rem;display:inline-block;font-size:0.8rem'>{k}</span>"
            for k in keywords
        ])
        st.markdown(f"**Search Keywords:** {chips}", unsafe_allow_html=True)

    # Model tabs
    model_info = {
        "ensemble": ("🌟 Ensemble",        "amber"),
        "gemini":   ("🔵 Gemini",          "blue"),
        "llama":    ("🟠 LLaMA-3.3-70B",   "amber"),
        "openai":   ("⚪ GPT-OSS-120B",     "green"),
        "deepseek": ("🔷 DeepSeek",         "purple"),
    }
    available_tabs = [k for k in model_info if results.get(k, {}) and results[k].get("related_work")]

    if available_tabs:
        tab_labels = [model_info[k][0] for k in available_tabs]
        tabs       = st.tabs(tab_labels)
        for tab, model_key in zip(tabs, available_tabs):
            with tab:
                model_result = results[model_key]
                rw = model_result.get("related_work", "")
                refs = model_result.get("references", "")
                word_count = len(rw.split()) if rw else 0

                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**Word count:** {word_count}")
                with col_b:
                    st.download_button(
                        f"⬇ Download",
                        data=f"RELATED WORK — {model_info[model_key][0]}\n{'='*60}\n\n{rw}\n\nREFERENCES\n{'='*60}\n\n{refs}",
                        file_name=f"related_work_{model_key}.txt",
                        mime="text/plain",
                        key=f"dl_{model_key}"
                    )

                st.markdown(f"""
                <div style='background:#111827;border:1px solid #1e3a5f;border-radius:10px;
                            padding:1.4rem;line-height:1.8;color:#e2e8f0;font-size:0.95rem;
                            white-space:pre-wrap;'>{rw}</div>
                """, unsafe_allow_html=True)

                if refs:
                    with st.expander("📚 References"):
                        st.markdown(f"<pre style='color:#94a3b8;font-size:0.82rem'>{refs}</pre>", unsafe_allow_html=True)

    # Related papers table
    if rel_papers:
        st.markdown("### 🗂️ Retrieved Related Papers")
        df_papers = [{
            "Title":     p["title"][:80] + ("…" if len(p["title"]) > 80 else ""),
            "Year":      p.get("year", "N/A"),
            "Citations": p.get("citations", 0),
            "Authors":   p.get("authors", "")[:50],
        } for p in rel_papers]
        st.dataframe(df_papers, use_container_width=True, hide_index=True)

    # Download full JSON
    st.markdown("---")
    st.download_button(
        "⬇ Download Full Pipeline Output (JSON)",
        data=json.dumps(out, indent=2, default=str, ensure_ascii=False),
        file_name="pipeline_output.json",
        mime="application/json"
    )

    # Reset button
    if st.button("🔄 Start Over", key="btn_reset"):
        for key in list(defaults.keys()):
            st.session_state[key] = defaults[key]
        st.rerun()
