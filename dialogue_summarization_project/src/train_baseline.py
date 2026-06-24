"""
Classical NLP baseline for dialogue summarization.

This file implements a simple TF-IDF based extractive summarizer.

Extractive summarization means:
    We select important sentences from the original dialogue.

This is different from abstractive summarization, where a model writes
new sentences in its own words.
"""

from typing import Dict, List, Tuple

import nltk
import pandas as pd
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import clean_text


def download_nltk_data() -> None:
    """
    Download the NLTK tokenizer data needed for sentence splitting.

    The downloads are small. If the data already exists, NLTK will skip it.
    """
    nltk.download("punkt", quiet=True)

    # Newer NLTK versions may also need punkt_tab.
    # The try/except keeps the code beginner-friendly and Colab-safe.
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass


def split_dialogue_into_sentences(dialogue: str) -> List[str]:
    """
    Split one dialogue into sentences.

    SAMSum dialogues often contain speaker turns separated by new lines.
    We first split by line, then split each line into sentences.

    Args:
        dialogue: A conversation from the SAMSum dataset.

    Returns:
        A list of sentence-like text pieces.
    """
    if dialogue is None:
        return []

    all_sentences = []

    # Split the dialogue by speaker lines.
    dialogue_lines = str(dialogue).split("\n")

    for line in dialogue_lines:
        line = line.strip()

        # Skip empty lines.
        if not line:
            continue

        # Use NLTK to split each line into sentences.
        line_sentences = sent_tokenize(line)

        for sentence in line_sentences:
            sentence = sentence.strip()
            if sentence:
                all_sentences.append(sentence)

    return all_sentences


def score_sentences_with_tfidf(sentences: List[str]) -> List[Tuple[int, str, float]]:
    """
    Score sentences using TF-IDF.

    TF-IDF gives higher weight to words that are important in a document.
    Here, each sentence is treated like a small document.

    Args:
        sentences: List of sentences from one dialogue.

    Returns:
        A list of tuples:
        (sentence_index, sentence_text, score)
    """
    if not sentences:
        return []

    # TF-IDF converts text into numerical features.
    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # This can happen if all sentences contain only stopwords.
        return [(index, sentence, 0.0) for index, sentence in enumerate(sentences)]

    sentence_scores = []

    for index, sentence in enumerate(sentences):
        # Sum TF-IDF values for all words in the sentence.
        # A higher sum means the sentence has more important words.
        score = tfidf_matrix[index].sum()
        sentence_scores.append((index, sentence, float(score)))

    return sentence_scores


def generate_extractive_summary(dialogue: str, num_sentences: int = 3) -> str:
    """
    Generate an extractive summary for one dialogue.

    Steps:
        1. Split dialogue into sentences.
        2. Score each sentence using TF-IDF.
        3. Select the top-scoring sentences.
        4. Put selected sentences back in original order.

    Args:
        dialogue: Conversation text.
        num_sentences: Number of sentences to include in the summary.

    Returns:
        Generated extractive summary.
    """
    cleaned_dialogue = clean_text(dialogue)
    sentences = split_dialogue_into_sentences(cleaned_dialogue)

    if not sentences:
        return ""

    # If the dialogue has fewer sentences than requested, return all of them.
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    scored_sentences = score_sentences_with_tfidf(sentences)

    # Sort sentences by score from highest to lowest.
    top_sentences = sorted(
        scored_sentences,
        key=lambda item: item[2],
        reverse=True,
    )[:num_sentences]

    # extras

    print("num_sentences =", num_sentences)
    print("len(top_sentences) =", len(top_sentences))

    for idx, sentence, score in top_sentences:
        print("\nINDEX:", idx)
        print("SCORE:", score)
        print("TEXT:", sentence)

    # extras end

    # Sort selected sentences by original position so the summary reads naturally.
    top_sentences_in_order = sorted(top_sentences, key=lambda item: item[0])

    summary = "\n".join(sentence for _, sentence, _ in top_sentences_in_order)

    return summary


def calculate_rouge_scores(reference_summary: str, generated_summary: str) -> Dict[str, float]:
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores for one prediction.

    ROUGE compares the generated summary with the human-written summary.
    We use F1 scores because they balance precision and recall.
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    scores = scorer.score(reference_summary, generated_summary)

    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def create_predictions_dataframe(
    dataframe: pd.DataFrame,
    num_sentences: int = 3,
    max_samples: int = 100,
) -> pd.DataFrame:
    """
    Generate summaries and ROUGE scores for a small dataset sample.

    Args:
        dataframe: DataFrame with dialogue and summary columns.
        num_sentences: Number of extracted sentences per generated summary.
        max_samples: Number of examples to evaluate.

    Returns:
        DataFrame containing dialogues, reference summaries, predictions, and scores.
    """
    rows = []

    # Use a smaller sample so the baseline runs quickly for beginners.
    sample_df = dataframe.head(max_samples).copy()

    for _, row in sample_df.iterrows():
        dialogue = row["dialogue"]
        reference_summary = row["summary"]

        generated_summary = generate_extractive_summary(
            dialogue=dialogue,
            num_sentences=num_sentences,
        )

        rouge_scores = calculate_rouge_scores(
            reference_summary=reference_summary,
            generated_summary=generated_summary,
        )

        rows.append(
            {
                "id": row.get("id", ""),
                "dialogue": dialogue,
                "reference_summary": reference_summary,
                "generated_summary": generated_summary,
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"],
            }
        )

    return pd.DataFrame(rows)


def create_rouge_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average ROUGE scores across all evaluated examples.
    """
    return pd.DataFrame(
        [
            {
                "model_name": "TF-IDF Extractive Baseline",
                "rouge1": predictions_df["rouge1"].mean(),
                "rouge2": predictions_df["rouge2"].mean(),
                "rougeL": predictions_df["rougeL"].mean(),
            }
        ]
    )


def create_comparison_table(rouge_df: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    """
    Create a simple comparison table for project reporting.

    In later phases, you can add transformer models to this table.
    """
    comparison_df = rouge_df.copy()
    comparison_df["model_type"] = "Classical NLP"
    comparison_df["method"] = "TF-IDF sentence scoring"
    comparison_df["evaluated_samples"] = max_samples

    return comparison_df[
        [
            "model_name",
            "model_type",
            "method",
            "evaluated_samples",
            "rouge1",
            "rouge2",
            "rougeL",
        ]
    ]


def run_baseline(
    output_dir: str = "results",
    dataset_split: str = "test",
    max_samples: int = 100,
    num_sentences: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full TF-IDF baseline pipeline.

    This function:
        1. Loads the SAMSum dataset.
        2. Generates extractive summaries.
        3. Calculates ROUGE scores.
        4. Saves result CSV files.

    Returns:
        predictions_df, rouge_df, comparison_df
    """
    download_nltk_data()

    dataset = load_dataset("knkarthick/samsum")
    dataframe = dataset[dataset_split].to_pandas()

    predictions_df = create_predictions_dataframe(
        dataframe=dataframe,
        num_sentences=num_sentences,
        max_samples=max_samples,
    )

    rouge_df = create_rouge_summary(predictions_df)
    comparison_df = create_comparison_table(
        rouge_df=rouge_df,
        max_samples=len(predictions_df),
    )

    # Create the output folder if it does not exist.
    import os

    os.makedirs(output_dir, exist_ok=True)

    predictions_df.to_csv(f"{output_dir}/sample_predictions.csv", index=False)
    rouge_df.to_csv(f"{output_dir}/rouge_scores.csv", index=False)
    comparison_df.to_csv(f"{output_dir}/comparison_table.csv", index=False)

    return predictions_df, rouge_df, comparison_df


if __name__ == "__main__":
    # Running this file directly will execute the baseline.
    # Example:
    #     python src/train_baseline.py
    run_baseline()
