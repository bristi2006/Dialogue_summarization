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
    }
    return f"<span class='inline-icon icon-{name}'>{icons[name]}</span>"


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
        max-width: 1280px;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: #E6EDF3 !important;
    }

    .hero-section {
        text-align: center;
        padding: 22px 20px 14px;
        margin-bottom: 16px;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 999px;
        border: 1px solid #30363D;
        background: rgba(22, 27, 34, 0.8);
        color: #8B949E;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .hero-title {
        font-size: 48px;
        line-height: 1.05;
        font-weight: 700;
        background: linear-gradient(135deg, #8BB6E8 0%, #E6EDF3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        margin-bottom: 10px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #8B949E;
        max-width: 760px;
        margin: 0 auto;
    }

    .input-section,
    .result-card,
    .stat-box {
        border: 1px solid #30363D;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    }

    .input-section {
        background-color: #1C2128;
        border-radius: 16px;
        padding: 18px;
        margin: 12px 0 16px;
        transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
    }

    .input-section:focus-within {
        border-color: #4F7CAC;
        box-shadow: 0 12px 30px rgba(79, 124, 172, 0.14);
        transform: translateY(-1px);
    }

    .result-card {
        background: linear-gradient(135deg, rgba(28, 33, 40, 0.95) 0%, rgba(22, 27, 34, 0.95) 100%);
        border-radius: 16px;
        padding: 18px;
        margin: 12px 0 20px;
    }

    .summary-shell {
        border-radius: 14px;
        border: 1px solid rgba(79, 124, 172, 0.18);
        background: linear-gradient(180deg, rgba(15, 17, 23, 0.28), rgba(28, 33, 40, 0.92));
        padding: 16px;
    }

    .summary-head,
    .input-title,
    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #E6EDF3;
        font-weight: 600;
        letter-spacing: -0.25px;
    }

    .summary-head {
        justify-content: space-between;
        margin-bottom: 12px;
    }

    .summary-title,
    .input-title,
    .section-title {
        font-size: 18px;
    }

    .summary-title {
        display: flex;
        align-items: center;
        gap: 10px;
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

    .copy-icon-btn {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        border: 1px solid #30363D;
        background: rgba(22, 27, 34, 0.92);
        color: #8B949E;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s ease;
        flex: 0 0 auto;
    }

    .copy-icon-btn:hover {
        border-color: #4F7CAC;
        color: #E6EDF3;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(79, 124, 172, 0.12);
    }

    .copy-icon-btn svg {
        width: 16px;
        height: 16px;
    }

    .summary-body {
        color: #E6EDF3;
        font-size: 15px;
        line-height: 1.75;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .stats-title {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 8px 0 14px;
        color: #E6EDF3;
        font-size: 18px;
        font-weight: 600;
        letter-spacing: -0.25px;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
    }

    .stat-box {
        background-color: #161B22;
        border-radius: 14px;
        padding: 18px 16px;
        text-align: center;
        transition: all 0.2s ease;
    }

    .stat-box:hover {
        border-color: #4F7CAC;
        transform: translateY(-1px);
    }

    .stat-label {
        font-size: 12px;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    .stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #8BB6E8;
        margin-top: 8px;
    }

    textarea {
        background-color: #1C2128 !important;
        border: 1px solid #30363D !important;
        color: #E6EDF3 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        font-family: 'Menlo', 'Monaco', monospace !important;
        transition: all 0.2s ease !important;
    }

    textarea:focus {
        border-color: #4F7CAC !important;
        box-shadow: 0 0 0 3px rgba(79, 124, 172, 0.1) !important;
        outline: none !important;
    }

    button {
        background-color: #4F7CAC !important;
        color: #E6EDF3 !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 10px rgba(79, 124, 172, 0.18) !important;
    }

    button:hover {
        background-color: #5A88C0 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(79, 124, 172, 0.24) !important;
    }

    .stAlert {
        border-radius: 12px !important;
        border: 1px solid #30363D !important;
    }

    [data-testid="stAlert"] {
        background-color: #1C2128 !important;
        border-radius: 12px !important;
    }

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #161B22;
    }

    ::-webkit-scrollbar-thumb {
        background: #30363D;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #4F7CAC;
    }

    @media (max-width: 768px) {
        .hero-title {
            font-size: 36px;
        }

        .hero-subtitle {
            font-size: 16px;
        }

        .hero-section {
            padding-top: 14px;
        }

        .input-section,
        .result-card {
            padding: 14px;
        }

        .summary-head {
            align-items: flex-start;
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
if "uploaded_dialogue" not in st.session_state:
    st.session_state.uploaded_dialogue = ""


summarizer = load_model()


def calculate_stats(input_text, summary_text):
    input_words = len(input_text.split())
    summary_words = len(summary_text.split())
    ratio = round((1 - (summary_words / input_words)) * 100, 1) if input_words > 0 else 0
    return input_words, summary_words, ratio


def render_summary_card(summary_text: str) -> None:
    summary_attr = escape(summary_text, quote=True)
    st.markdown('<div class="summary-shell">', unsafe_allow_html=True)
    head_col, button_col = st.columns([12, 1], gap="small")
    with head_col:
        st.markdown(f'<div class="summary-title">{icon_svg("summary")} Summary</div>', unsafe_allow_html=True)
    with button_col:
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
                }}

                .copy-btn {{
                    width: 34px;
                    height: 34px;
                    border-radius: 10px;
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
                    box-shadow: 0 6px 16px rgba(79, 124, 172, 0.12);
                }}

                .copy-btn svg {{
                    width: 16px;
                    height: 16px;
                }}
            </style>
            <script>
                function copySummary(button) {{
                    const text = button.dataset.summary || '';
                    const feedback = button.nextElementSibling;
                    const showCopied = () => {{
                        if (feedback) {{
                            feedback.textContent = 'Copied';
                            feedback.style.opacity = '1';
                            setTimeout(() => feedback.style.opacity = '0', 900);
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
                <button class="copy-btn" type="button" title="Copy summary" aria-label="Copy summary" data-summary="{summary_attr}" onclick="copySummary(this)">
                    {icon_svg('copy')}
                </button>
                <span style="margin-left:8px;font-size:12px;color:#8B949E;opacity:0;transition:opacity .2s ease;">Copied</span>
            </div>
            """,
            height=44,
            scrolling=False,
        )
    st.markdown(f'<div class="summary-body">{escape(summary_text)}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown(
    f"""
    <div class="hero-section">
        <div class="hero-badge">AI Dialogue Summarizer</div>
        <div class="hero-title">Dialogue Summarizer</div>
        <div class="hero-subtitle">Transform conversations into concise, meaningful summaries powered by advanced AI</div>
    </div>
    """,
    unsafe_allow_html=True,
)


col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown(f'<div class="input-title">{icon_svg("upload")} Upload Dialogue <span style="color:#8B949E;font-weight:500;">(optional)</span></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["txt"], label_visibility="collapsed", key="file_uploader")

    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read().decode("utf-8")
            if not file_content.strip():
                st.error("The uploaded file is empty. Please upload a file with content.")
            else:
                st.session_state.uploaded_dialogue = file_content
                st.session_state.dialogue = file_content
                st.success(f"File '{uploaded_file.name}' loaded successfully.")
        except UnicodeDecodeError:
            st.error("Could not decode the file. Please ensure it's a valid UTF-8 text file.")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown(f'<div class="input-title">{icon_svg("dialogue")} Enter Your Dialogue</div>', unsafe_allow_html=True)
    dialogue = st.text_area(
        "",
        value=st.session_state.dialogue,
        height=300,
        placeholder="A: Hi\nB: Hello, how are you?\nA: I'm doing great! Working on an AI project...\nB: That sounds interesting!",
        label_visibility="collapsed",
        key="dialogue_input",
    )
    st.session_state.dialogue = dialogue
    st.markdown("</div>", unsafe_allow_html=True)

    left_spacer, action_col, right_spacer = st.columns([1, 2, 1], gap="small")
    with action_col:
        if st.button("Generate Summary", use_container_width=True, key="generate_btn"):
            active_dialogue = dialogue.strip() or st.session_state.uploaded_dialogue.strip()
            if active_dialogue:
                with st.spinner("Analyzing dialogue..."):
                    result = summarizer(
                        active_dialogue,
                        max_length=60,
                        min_length=10,
                        do_sample=False,
                    )
                    st.session_state.summary = result[0]["summary_text"]
                    st.session_state.dialogue = active_dialogue
            else:
                st.warning("Please enter a dialogue or upload a text file.")

with col2:
    if st.session_state.summary:
        render_summary_card(st.session_state.summary)

        input_words, summary_words, compression = calculate_stats(st.session_state.dialogue, st.session_state.summary)
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
    else:
        st.markdown(
            '''
            <div class="result-card" style="text-align: center; padding: 60px 20px; opacity: 0.72;">
                <div style="font-size: 16px; color: #8B949E; line-height: 1.7;">Enter a dialogue or upload a text file, then click Generate Summary to see the result here.</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
