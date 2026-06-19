import streamlit as st
from transformers import pipeline

# Load model once
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="models/transformer",
        tokenizer="models/transformer"
    )

summarizer = load_model()

st.title("Dialogue Summarization App")

st.write("Enter a dialogue and generate a summary.")

dialogue = st.text_area(
    "Dialogue",
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