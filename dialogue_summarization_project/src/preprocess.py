"""
Preprocessing helper functions for the Dialogue Summarization project.

This file is intentionally written in a beginner-friendly style.
Each function does one small job so it is easy to test and reuse.
"""

import html
import re
import string
from typing import Dict, List

import nltk
from nltk.tokenize import word_tokenize


EMOTICON_MAP = {
    ":)": " <HAPPY> ",
    ":-)": " <HAPPY> ",
    ":D": " <HAPPY> ",
    ":-D": " <HAPPY> ",
    ":(": " <SAD> ",
    ":-(": " <SAD> ",
    ";)": " <WINK> ",
    ";-)": " <WINK> ",
    ":/": " <UNSURE> ",
    ":-/": " <UNSURE> ",
}


def clean_text(text: str) -> str:
    """
    Clean a dialogue or summary string.

    Args:
        text: The original text.

    Returns:
        A cleaned version of the text.
    """
    # Handle missing values safely.
    if text is None:
        return ""

    # Convert to string in case a non-string value is passed by mistake.
    text = str(text)

    # Convert HTML entities like &amp; back to normal characters.
    text = html.unescape(text)

    # Remove HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace common emoticons with readable tokens before lowercasing.
    for emoticon, token in EMOTICON_MAP.items():
        text = text.replace(emoticon, token)

    # Convert all regular text to lowercase so "Hello" and "hello" are treated equally.
    text = text.lower()

    # Replace new lines and tabs with a space.
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    # Remove website links.
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep emoticon tokens readable after lowercasing.
    text = re.sub(r"<happy>", "<HAPPY>", text)
    text = re.sub(r"<sad>", "<SAD>", text)
    text = re.sub(r"<wink>", "<WINK>", text)
    text = re.sub(r"<unsure>", "<UNSURE>", text)

    # Remove extra spaces.
    text = re.sub(r"\s+", " ", text)

    # Remove spaces from the beginning and end.
    text = text.strip()

    return text


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation marks from text.

    Example:
        "hello, world!" becomes "hello world"
    """
    if text is None:
        return ""

    # str.maketrans creates a mapping that removes punctuation characters.
    translator = str.maketrans("", "", string.punctuation)
    return str(text).translate(translator)


def tokenize_text(text: str) -> List[str]:
    """
    Split text into word tokens using NLTK word_tokenize.

    This is a basic tokenizer for learning purposes.
    Later phases can use advanced tokenizers from Hugging Face Transformers.
    """
    nltk.download("punkt", quiet=True)
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass

    cleaned_text = clean_text(text)
    cleaned_text = remove_punctuation(cleaned_text)
    tokens = word_tokenize(cleaned_text)

    return tokens


def count_words(text: str) -> int:
    """
    Count the number of words in a text.
    """
    return len(tokenize_text(text))


def count_characters(text: str) -> int:
    """
    Count the number of characters in a text.
    """
    if text is None:
        return 0
    return len(str(text))


def preprocess_example(dialogue: str, summary: str) -> Dict[str, object]:
    """
    Preprocess one dialogue-summary pair.

    Args:
        dialogue: Conversation text from the dataset.
        summary: Human-written summary from the dataset.

    Returns:
        A dictionary containing cleaned text, tokens, and basic lengths.
    """
    clean_dialogue = clean_text(dialogue)
    clean_summary = clean_text(summary)

    dialogue_tokens = tokenize_text(clean_dialogue)
    summary_tokens = tokenize_text(clean_summary)

    return {
        "clean_dialogue": clean_dialogue,
        "clean_summary": clean_summary,
        "dialogue_tokens": dialogue_tokens,
        "summary_tokens": summary_tokens,
        "dialogue_word_count": len(dialogue_tokens),
        "summary_word_count": len(summary_tokens),
    }
