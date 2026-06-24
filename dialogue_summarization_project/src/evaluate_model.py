"""
Evaluation helpers for the transformer summarization model.

This file calculates ROUGE scores and saves sample predictions.

Note:
    This project file is named evaluate.py because Phase 3 asks for it.
    Hugging Face also has a package named evaluate. To avoid confusion,
    the function below loads the Hugging Face package carefully.
"""

import importlib
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_huggingface_rouge_metric():
    """
    Load the Hugging Face ROUGE metric.

    Because this file is also named evaluate.py, Python may try to import
    this file instead of the Hugging Face evaluate package. This helper
    temporarily removes the local src folder from the import path.
    """
    current_module = sys.modules.get("evaluate")
    src_folder = Path(__file__).resolve().parent
    original_sys_path = sys.path.copy()

    try:
        # Remove this local module name while importing the external package.
        sys.modules.pop("evaluate", None)

        # Remove the src folder from sys.path for this import only.
        cleaned_path = []
        for path in sys.path:
            if not path:
                cleaned_path.append(path)
                continue

            try:
                if Path(path).resolve() == src_folder:
                    continue
            except OSError:
                pass

            cleaned_path.append(path)

        sys.path = cleaned_path

        # Import Hugging Face evaluate package.
        huggingface_evaluate = importlib.import_module("evaluate")
        rouge_metric = huggingface_evaluate.load("rouge")

    finally:
        # Restore the import path and local module registration.
        sys.path = original_sys_path
        if current_module is not None:
            sys.modules["evaluate"] = current_module

    return rouge_metric


def postprocess_text(predictions: List[str], references: List[str]):
    """
    Clean decoded predictions and references before ROUGE scoring.

    ROUGE works best when each summary is a normal cleaned string.
    """
    predictions = [prediction.strip() for prediction in predictions]
    references = [reference.strip() for reference in references]

    return predictions, references


def compute_rouge_metrics(eval_prediction, tokenizer) -> Dict[str, float]:
    """
    Compute ROUGE scores during Trainer evaluation.

    Args:
        eval_prediction: Predictions and labels returned by the Trainer.
        tokenizer: Tokenizer used to decode token IDs back into text.

    Returns:
        Dictionary with ROUGE scores.
    """
    predictions, labels = eval_prediction

    # Some models return predictions inside a tuple.
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # In normal seq2seq generation, predictions should be token IDs.
    # In some Transformers/version combinations, predictions may come back
    # as logits with shape: batch_size x sequence_length x vocabulary_size.
    # If that happens, we choose the highest-scoring token at each position.
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    # Make sure predictions are integer token IDs.
    predictions = predictions.astype(np.int64)
    labels = labels.astype(np.int64)

    # Replace invalid prediction IDs before decoding.
    # This avoids tokenizer overflow errors in some Colab environments.
    vocabulary_size = len(tokenizer)
    predictions = np.where(
        (predictions >= 0) & (predictions < vocabulary_size),
        predictions,
        tokenizer.pad_token_id,
    )

    # Replace -100 labels because tokenizer cannot decode -100.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Also protect labels from any unexpected invalid IDs.
    labels = np.where(
        (labels >= 0) & (labels < vocabulary_size),
        labels,
        tokenizer.pad_token_id,
    )

    # Decode token IDs into readable text.
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_predictions, decoded_labels = postprocess_text(
        decoded_predictions,
        decoded_labels,
    )

    rouge_metric = load_huggingface_rouge_metric()
    result = rouge_metric.compute(
        predictions=decoded_predictions,
        references=decoded_labels,
        use_stemmer=True,
    )

    # Convert scores to percentages for easier reading.
    return {key: round(value * 100, 4) for key, value in result.items()}


def save_metrics_csv(metrics: Dict[str, float], output_path: str) -> pd.DataFrame:
    """
    Save evaluation metrics to a CSV file.
    """
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path, index=False)
    return metrics_df


def save_sample_predictions(
    dialogues: List[str],
    references: List[str],
    predictions: List[str],
    output_path: str,
) -> pd.DataFrame:
    """
    Save sample model predictions to a CSV file.
    """
    predictions_df = pd.DataFrame(
        {
            "dialogue": dialogues,
            "reference_summary": references,
            "generated_summary": predictions,
        }
    )
    predictions_df.to_csv(output_path, index=False)
    return predictions_df
