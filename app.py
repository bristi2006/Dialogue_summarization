import streamlit as st
from transformers import pipeline

# Load model once
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="Bristi2006/Dialogue_Summarizer",
        tokenizer="Bristi2006/Dialogue_Summarizer"
    )

# Initialize session state for dialogue
if 'dialogue' not in st.session_state:
    st.session_state.dialogue = ""

summarizer = load_model()

st.title("Dialogue Summarization App")

st.write("Enter a dialogue and generate a summary.")

# File uploader for .txt files
uploaded_file = st.file_uploader("Upload a .txt file (optional)", type=["txt"])

# Handle file upload
if uploaded_file is not None:
    try:
        # Read file content with UTF-8 encoding
        file_content = uploaded_file.read().decode('utf-8')
        
        # Check for empty file
        if not file_content.strip():
            st.error("❌ Error: The uploaded file is empty. Please upload a file with content.")
        else:
            # Update session state with file content
            st.session_state.dialogue = file_content
            st.success(f"✓ File '{uploaded_file.name}' loaded successfully!")
    except UnicodeDecodeError:
        st.error("❌ Error: Could not decode the file. Please ensure it's a valid UTF-8 text file.")
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

dialogue = st.text_area(
    "Dialogue",
    value=st.session_state.dialogue,
    height=250,
    placeholder="A: Hi\nB: Hello..."
)

if st.button("Generate Summary"):

    if dialogue.strip():

        result = summarizer(
            dialogue,
            max_length=60,
            min_length=10,
            do_sample=False
        )

        st.subheader("Summary")
        st.success(result[0]["summary_text"])

    else:
        st.warning("Please enter a dialogue.")
