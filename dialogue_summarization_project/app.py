import streamlit as st
from transformers import pipeline
import re

# Configure page
st.set_page_config(
    page_title="Dialogue Summarizer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern SaaS styling
custom_css = """
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    /* Main background and text colors */
    [data-testid="stAppViewContainer"] {
        background-color: #0F1117;
        color: #E6EDF3;
    }

    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #E6EDF3 !important;
        font-weight: 600;
        letter-spacing: -0.5px;
    }

    p, label {
        color: #E6EDF3;
    }

    /* Text area styling */
    textarea {
        background-color: #1C2128 !important;
        border: 1px solid #30363D !important;
        color: #E6EDF3 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        font-family: 'Menlo', 'Monaco', monospace !important;
        transition: all 0.3s ease !important;
    }

    textarea:focus {
        border-color: #4F7CAC !important;
        box-shadow: 0 0 0 3px rgba(79, 124, 172, 0.1) !important;
        outline: none !important;
    }

    /* Button styling */
    button {
        background-color: #4F7CAC !important;
        color: #E6EDF3 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
        box-shadow: 0 2px 8px rgba(79, 124, 172, 0.2) !important;
    }

    button:hover {
        background-color: #5A88C0 !important;
        box-shadow: 0 4px 16px rgba(79, 124, 172, 0.3) !important;
        transform: translateY(-2px) !important;
    }

    button:active {
        transform: translateY(0) !important;
    }

    /* Input fields */
    input {
        background-color: #1C2128 !important;
        border: 1px solid #30363D !important;
        color: #E6EDF3 !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
        transition: all 0.3s ease !important;
    }

    input:focus {
        border-color: #4F7CAC !important;
        box-shadow: 0 0 0 3px rgba(79, 124, 172, 0.1) !important;
        outline: none !important;
    }

    /* Card styling */
    .card {
        background-color: #1C2128;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    .card:hover {
        border-color: #4F7CAC;
        box-shadow: 0 8px 24px rgba(79, 124, 172, 0.15);
    }

    .hero-section {
        text-align: center;
        padding: 40px 20px;
        margin-bottom: 40px;
    }

    .hero-title {
        font-size: 48px;
        font-weight: 700;
        background: linear-gradient(135deg, #4F7CAC 0%, #8B949E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 12px;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #8B949E;
        margin-bottom: 30px;
        font-weight: 400;
    }

    .input-section {
        background-color: #1C2128;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 28px;
        margin: 24px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    .input-section:focus-within {
        border-color: #4F7CAC;
        box-shadow: 0 8px 24px rgba(79, 124, 172, 0.15);
    }

    .result-card {
        background: linear-gradient(135deg, rgba(79, 124, 172, 0.1) 0%, rgba(139, 148, 158, 0.05) 100%);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 28px;
        margin: 24px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    .success-card {
        background: linear-gradient(135deg, rgba(29, 78, 47, 0.15) 0%, rgba(23, 60, 36, 0.1) 100%);
        border: 1px solid rgba(29, 78, 47, 0.3);
        border-radius: 12px;
        padding: 20px;
        color: #7EE787;
        margin: 12px 0;
    }

    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 20px 0;
    }

    .stat-box {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .stat-box:hover {
        border-color: #4F7CAC;
        background-color: #1C2128;
    }

    .stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #4F7CAC;
        margin: 8px 0;
    }

    .stat-label {
        font-size: 12px;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    .copy-button {
        background-color: #30363D !important;
        border: 1px solid #30363D !important;
        color: #E6EDF3 !important;
        padding: 8px 16px !important;
        font-size: 12px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }

    .copy-button:hover {
        background-color: #4F7CAC !important;
        border-color: #4F7CAC !important;
    }

    .file-upload-section {
        margin-bottom: 24px;
    }

    .upload-label {
        color: #8B949E;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 8px;
        display: block;
    }

    /* Spinner animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(79, 124, 172, 0.3);
        border-top-color: #4F7CAC;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 10px;
    }

    /* Remove default Streamlit styling */
    .stTextArea label {
        color: #E6EDF3 !important;
        font-weight: 500 !important;
    }

    .stFileUploader {
        color: #E6EDF3 !important;
    }

    /* Success/Error/Warning messages */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid #30363D !important;
    }

    [data-testid="stAlert"] {
        background-color: #1C2128 !important;
        border-radius: 12px !important;
    }

    /* Scrollbar styling */
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

    /* Responsive design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 36px;
        }

        .hero-subtitle {
            font-size: 16px;
        }

        .stats-container {
            grid-template-columns: 1fr;
        }

        .card {
            padding: 16px;
        }
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Load model once
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="Bristi2006/Dialogue_Summarizer",
        tokenizer="Bristi2006/Dialogue_Summarizer"
    )

# Initialize session state
if 'dialogue' not in st.session_state:
    st.session_state.dialogue = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'generating' not in st.session_state:
    st.session_state.generating = False

summarizer = load_model()

# Calculate word count and compression ratio
def calculate_stats(input_text, summary_text):
    input_words = len(input_text.split())
    summary_words = len(summary_text.split())
    ratio = round((1 - (summary_words / input_words)) * 100, 1) if input_words > 0 else 0
    return input_words, summary_words, ratio

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">✨ Dialogue Summarizer</div>
    <div class="hero-subtitle">Transform conversations into concise, meaningful summaries powered by advanced AI</div>
</div>
""", unsafe_allow_html=True)

# Main container
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # File upload section
    st.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
    st.markdown('<label class="upload-label">📁 Upload Dialogue (Optional)</label>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["txt"], label_visibility="collapsed", key="file_uploader")
    
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read().decode('utf-8')
            
            if not file_content.strip():
                st.error("❌ The uploaded file is empty. Please upload a file with content.")
            else:
                st.session_state.dialogue = file_content
                st.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
        except UnicodeDecodeError:
            st.error("❌ Could not decode the file. Please ensure it's a valid UTF-8 text file.")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Dialogue input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    dialogue = st.text_area(
        "📝 Enter Your Dialogue",
        value=st.session_state.dialogue,
        height=300,
        placeholder="A: Hi\nB: Hello, how are you?\nA: I'm doing great! Working on an AI project...\nB: That sounds interesting!",
        label_visibility="visible",
        key="dialogue_input"
    )
    
    st.session_state.dialogue = dialogue
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action button
    col_btn1, col_btn2, col_btn3 = st.columns(3, gap="small")
    
    with col_btn2:
        if st.button("🚀 Generate Summary", use_container_width=True, key="generate_btn"):
            if dialogue.strip():
                st.session_state.generating = True
                
                # Show loading state
                with st.spinner("✨ Analyzing dialogue..."):
                    result = summarizer(
                        dialogue,
                        max_length=60,
                        min_length=10,
                        do_sample=False
                    )
                    st.session_state.summary = result[0]["summary_text"]
                
                st.session_state.generating = False
            else:
                st.warning("⚠️ Please enter a dialogue to summarize.")

with col2:
    if st.session_state.summary:
        # Result card
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Summary")
        
        # Copy to clipboard button
        col_copy1, col_copy2 = st.columns([1, 4])
        with col_copy1:
            if st.button("📋 Copy", key="copy_btn", use_container_width=True):
                st.toast("✅ Copied to clipboard!", icon="✅")
        
        st.markdown(f'<div class="success-card"><p style="margin: 0; font-size: 16px; line-height: 1.6;">{st.session_state.summary}</p></div>', unsafe_allow_html=True)
        
        # Statistics section
        input_words, summary_words, compression = calculate_stats(dialogue, st.session_state.summary)
        
        st.markdown("### 📊 Statistics")
        
        stat_col1, stat_col2, stat_col3 = st.columns(3, gap="small")
        
        with stat_col1:
            st.markdown(f'''
            <div class="stat-box">
                <div class="stat-label">Input Words</div>
                <div class="stat-value">{input_words}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with stat_col2:
            st.markdown(f'''
            <div class="stat-box">
                <div class="stat-label">Summary Words</div>
                <div class="stat-value">{summary_words}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with stat_col3:
            st.markdown(f'''
            <div class="stat-box">
                <div class="stat-label">Compression</div>
                <div class="stat-value">{compression}%</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Placeholder for empty state
        st.markdown('''
        <div class="result-card" style="text-align: center; opacity: 0.6; padding: 60px 20px;">
            <p style="font-size: 18px; color: #8B949E;">👈 Enter a dialogue and click "Generate Summary" to see results here</p>
        </div>
        ''', unsafe_allow_html=True)