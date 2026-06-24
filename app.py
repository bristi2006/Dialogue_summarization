import json
from html import escape

import streamlit as st
import streamlit.components.v1 as components
from transformers import pipeline


st.set_page_config(
    page_title="Dialogue Summarizer",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def icon_svg(name: str) -> str:
    icons = {
        "dialogue": "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z'/></svg>",
        "upload": "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/><path d='M7 10l5-5 5 5'/><path d='M12 5v12'/></svg>",
        "summary": "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M4 6h16'/><path d='M4 12h10'/><path d='M4 18h16'/><path d='M18 10l2 2-2 2'/></svg>",
        "stats": "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M4 20V10'/><path d='M10 20V4'/><path d='M16 20v-8'/><path d='M22 20H2'/></svg>",
        "copy": "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><rect x='9' y='9' width='13' height='13' rx='2'/><path d='M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1'/></svg>",
        "sparks": "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z'/><path d='m5 3 1 2.5L8.5 6 6 7 5 9.5 4 7 1.5 6 4 5.5z'/><path d='m19 17 1 2.5 2.5.5-2.5 1-1 2.5-1-2.5-2.5-1 2.5-1z'/></svg>"
    }
    return f"<span class='inline-icon icon-{name}'>{icons[name]}</span>"


def copy_icon_svg() -> str:
    return (
        "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' "
        "stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
        "<rect x='9' y='9' width='13' height='13' rx='2'/><path d='M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1'/></svg>"
    )


custom_css = """
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #0F1117;
        color: #E6EDF3;
    }

    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        transition: max-width 0.3s ease;
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: #E6EDF3 !important;
    }

    .hero-section {
        text-align: center;
        padding: 24px 20px;
        margin-bottom: 20px;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 999px;
        border: 1px solid #30363D;
        background: rgba(22, 27, 34, 0.8);
        color: #8B949E;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    
    .hero-badge .inline-icon {
        color: #8BB6E8;
    }

    .hero-title {
        font-size: 42px;
        line-height: 1.1;
        font-weight: 700;
        background: linear-gradient(135deg, #8BB6E8 0%, #E6EDF3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }
    
    .hero-title .inline-icon {
        width: 38px;
        height: 38px;
        color: #8BB6E8;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #8B949E;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Container Card Styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, #1C2128 0%, #161B22 100%) !important;
        border: 1px solid #30363D !important;
        border-radius: 16px !important;
        padding: 24px !important;
        position: relative !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #4F7CAC !important;
        box-shadow: 0 12px 32px rgba(79, 124, 172, 0.12) !important;
        transform: translateY(-2px) !important;
    }

    .input-title,
    .stats-title,
    .summary-title {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #E6EDF3;
        font-weight: 600;
        font-size: 18px;
        letter-spacing: -0.25px;
        margin-bottom: 12px;
    }

    .inline-icon {
        width: 18px;
        height: 18px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 auto;
        color: #8BB6E8;
    }

    .inline-icon svg {
        width: 100%;
        height: 100%;
    }

    .summary-body {
        color: #E6EDF3;
        font-size: 15px;
        line-height: 1.75;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* Stats Styling */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
        margin-top: 8px;
    }

    .stat-box {
        background-color: #0F1117;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #30363D;
        transition: all 0.2s ease;
    }

    .stat-box:hover {
        border-color: #4F7CAC;
        transform: translateY(-1px);
    }

    .stat-label {
        font-size: 11px;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    .stat-value {
        font-size: 26px;
        font-weight: 700;
        color: #8BB6E8;
        margin-top: 6px;
    }

    /* Textarea Styling */
    textarea {
        background-color: #0F1117 !important;
        border: 1px solid #30363D !important;
        color: #E6EDF3 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important;
        font-size: 14px !important;
    }

    textarea:focus {
        border-color: #4F7CAC !important;
        box-shadow: 0 0 0 3px rgba(79, 124, 172, 0.15) !important;
        outline: none !important;
    }

    /* File Uploader Styling */
    [data-testid="stFileUploader"] {
        background-color: #0F1117 !important;
        border: 1px dashed #30363D !important;
        border-radius: 12px !important;
        padding: 12px !important;
        transition: all 0.2s ease !important;
        margin-bottom: 12px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #4F7CAC !important;
    }

    /* Button Styling */
    div[data-testid="stButton"] button {
        background-color: #4F7CAC !important;
        color: #E6EDF3 !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        height: 48px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 10px rgba(79, 124, 172, 0.18) !important;
    }

    div[data-testid="stButton"] button:hover {
        background-color: #5A88C0 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(79, 124, 172, 0.24) !important;
        border-color: transparent !important;
    }

    /* Position the copy button iframe */
    div:has(iframe[title="copy_button"]) {
        position: absolute !important;
        top: 20px !important;
        right: 20px !important;
        z-index: 99 !important;
        width: 120px !important;
        height: 32px !important;
    }

    /* Reduce vertical block gaps */
    div[data-testid="stVerticalBlock"] > div {
        margin-bottom: 12px !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: #161B22;
    }

    ::-webkit-scrollbar-thumb {
        background: #30363D;
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #4F7CAC;
    }

    @media (max-width: 768px) {
        .hero-title {
            font-size: 32px;
        }

        .hero-subtitle {
            font-size: 14px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 16px !important;
        }
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="Bristi2006/Dialogue_Summarizer",
        tokenizer="Bristi2006/Dialogue_Summarizer",
    )


if "dialogue" not in st.session_state:
    st.session_state.dialogue = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "validation_error" not in st.session_state:
    st.session_state.validation_error = ""
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

summarizer = load_model()


def calculate_stats(input_text, summary_text):
    input_words = len(input_text.split())
    summary_words = len(summary_text.split())
    ratio = round((1 - (summary_words / input_words)) * 100, 1) if input_words > 0 else 0
    return input_words, summary_words, ratio


def render_input_panel():
    with st.container(border=True):
        st.markdown(f'<div class="input-title">{icon_svg("upload")} Upload Dialogue <span style="color:#8B949E;font-weight:500;font-size:13px;">(optional)</span></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["txt"], label_visibility="collapsed", key="file_uploader")

        # Load file content if uploaded
        if uploaded_file is not None:
            file_name = uploaded_file.name
            if st.session_state.last_uploaded_file != file_name:
                try:
                    uploaded_file.seek(0)
                    file_content = uploaded_file.read().decode("utf-8")
                    if file_content.strip():
                        st.session_state.dialogue = file_content
                        st.session_state.last_uploaded_file = file_name
                        st.session_state.validation_error = ""
                except Exception as e:
                    st.session_state.validation_error = f"Error reading file: {e}"
            
            # Render file status message only when file is uploaded
            if st.session_state.last_uploaded_file == file_name:
                st.success(f"File '{file_name}' loaded successfully.")
        else:
            st.session_state.last_uploaded_file = None

        st.markdown(f'<div class="input-title" style="margin-top: 18px;">{icon_svg("dialogue")} Enter Your Dialogue</div>', unsafe_allow_html=True)
        st.text_area(
            "",
            key="dialogue",
            height=300,
            placeholder="A: Hi\nB: Hello, how are you?\nA: I'm doing great! Working on an AI project...\nB: That sounds interesting!",
            label_visibility="collapsed",
        )

        if st.session_state.validation_error:
            st.error(st.session_state.validation_error)

        if st.button("Generate Summary", use_container_width=True, key="generate_btn"):
            active_dialogue = ""
            if uploaded_file is not None:
                try:
                    uploaded_file.seek(0)
                    active_dialogue = uploaded_file.read().decode("utf-8").strip()
                except Exception as e:
                    st.session_state.validation_error = f"Error reading file: {e}"
            else:
                active_dialogue = st.session_state.dialogue.strip()

            if not active_dialogue:
                st.session_state.validation_error = "Please enter a dialogue or upload a text file."
                st.session_state.summary = ""
                st.rerun()
            else:
                st.session_state.validation_error = ""
                with st.spinner("Analyzing dialogue..."):
                    try:
                        result = summarizer(
                            active_dialogue,
                            max_length=60,
                            min_length=10,
                            do_sample=False,
                        )
                        st.session_state.summary = result[0]["summary_text"]
                        st.session_state.dialogue = active_dialogue
                        st.rerun()
                    except Exception as e:
                        st.session_state.validation_error = f"Error summarizing dialogue: {e}"
                        st.rerun()


def render_output_panel():
    summary_text = st.session_state.summary
    if not summary_text:
        return

    # Render Summary card
    with st.container(border=True):
        # 1. Render copy button iframe
        summary_attr = escape(summary_text, quote=True)
        components.html(
            f"""
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    overflow: hidden;
                    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                }}

                .copy-wrap {{
                    display: flex;
                    justify-content: flex-end;
                    align-items: center;
                    height: 100%;
                    gap: 6px;
                }}

                .copy-btn {{
                    width: 28px;
                    height: 28px;
                    border-radius: 6px;
                    border: 1px solid #30363D;
                    background: rgba(22, 27, 34, 0.92);
                    color: #8B949E;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    padding: 0;
                }}

                .copy-btn:hover {{
                    border-color: #4F7CAC;
                    color: #E6EDF3;
                    transform: translateY(-1px);
                    box-shadow: 0 4px 12px rgba(79, 124, 172, 0.15);
                }}

                .copy-btn svg {{
                    width: 14px;
                    height: 14px;
                }}

                .copied-msg {{
                    font-size: 11px;
                    color: #4F7CAC;
                    font-weight: 600;
                    opacity: 0;
                    transition: opacity 0.2s ease;
                    white-space: nowrap;
                }}
            </style>
            <script>
                function copySummary(button) {{
                    const text = button.dataset.summary || '';
                    const feedback = button.previousElementSibling;
                    const showCopied = () => {{
                        if (feedback) {{
                            feedback.style.opacity = '1';
                            setTimeout(() => feedback.style.opacity = '0', 1200);
                        }}
                    }};

                    if (navigator.clipboard && navigator.clipboard.writeText) {{
                        navigator.clipboard.writeText(text).then(showCopied).catch(() => {{
                            const temp = document.createElement('textarea');
                            temp.value = text;
                            temp.style.position = 'fixed';
                            temp.style.opacity = '0';
                            document.body.appendChild(temp);
                            temp.focus();
                            temp.select();
                            document.execCommand('copy');
                            document.body.removeChild(temp);
                            showCopied();
                        }});
                        return;
                    }}

                    const temp = document.createElement('textarea');
                    temp.value = text;
                    temp.style.position = 'fixed';
                    temp.style.opacity = '0';
                    document.body.appendChild(temp);
                    temp.focus();
                    temp.select();
                    document.execCommand('copy');
                    document.body.removeChild(temp);
                    showCopied();
                }}
            </script>
            <div class="copy-wrap">
                <span class="copied-msg">Copied!</span>
                <button class="copy-btn" type="button" title="Copy summary" aria-label="Copy summary" data-summary="{summary_attr}" onclick="copySummary(this)">
                    {copy_icon_svg()}
                </button>
            </div>
            """,
            title="copy_button",
            height=32,
            scrolling=False,
        )

        # 2. Render summary content
        st.markdown(f'<div class="summary-title">{icon_svg("summary")} Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-body" style="margin-top: 12px;">{escape(summary_text)}</div>', unsafe_allow_html=True)

    # Render Statistics Card
    input_words, summary_words, compression = calculate_stats(st.session_state.dialogue, summary_text)

    with st.container(border=True):
        st.markdown(f'<div class="stats-title">{icon_svg("stats")} Statistics</div>', unsafe_allow_html=True)

        stat_col1, stat_col2, stat_col3 = st.columns(3, gap="small")
        with stat_col1:
            st.markdown(
                f'''
                <div class="stat-box">
                    <div class="stat-label">Input Words</div>
                    <div class="stat-value">{input_words}</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )
        with stat_col2:
            st.markdown(
                f'''
                <div class="stat-box">
                    <div class="stat-label">Summary Words</div>
                    <div class="stat-value">{summary_words}</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )
        with stat_col3:
            st.markdown(
                f'''
                <div class="stat-box">
                    <div class="stat-label">Compression</div>
                    <div class="stat-value">{compression}%</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )


# Main App Layout
st.markdown(
    f"""
    <div class="hero-section">
        <div class="hero-badge">{icon_svg("sparks")} AI Dialogue Summarizer</div>
        <div class="hero-title">{icon_svg("dialogue")} Dialogue Summarizer</div>
        <div class="hero-subtitle">Transform conversations into concise, meaningful summaries powered by advanced AI</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.summary:
    # 2-column layout when summary exists
    st.markdown("<style>.main .block-container { max-width: 1280px; }</style>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        render_input_panel()
    with col2:
        render_output_panel()
else:
    # Centered single-column layout when no summary exists
    st.markdown("<style>.main .block-container { max-width: 800px; }</style>", unsafe_allow_html=True)
    render_input_panel()
