"""
Week 3: Fine-tune a transformer model for SAMSum dialogue summarization.

Roadmap outputs created by this file:
    1. Fine-tuned transformer checkpoint in models/transformer/
    2. Epoch checkpoints in models/transformer-checkpoints/
    3. Training + validation loss curve in results/training_curve.png
    4. Transformer ROUGE scores on the test set in results/transformer_metrics.csv
    5. Baseline vs Transformer comparison table in results/comparison_table.csv
    6. Sample generated summaries in results/transformer_sample_predictions.csv

Model used:
    facebook/bart-base

Basic question: Why BART?
    BART is an encoder-decoder transformer, which is the correct model family for
    abstractive summarization. The encoder reads the dialogue and the decoder
    writes a new summary. We use facebook/bart-base instead of bart-large-cnn
    because bart-base is smaller, faster, and easier to fine-tune on Colab. We
    do not use T5 or PEGASUS here only to keep Week 3 focused on one clear model;
    both are valid alternatives, but changing models would make comparison and
    debugging harder for a beginner project.
"""

from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, load_dataset
from rouge_score import rouge_scorer
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from evaluate_model import compute_rouge_metrics, save_metrics_csv


# Main model for Week 3.
# Basic question: Why not train a model from scratch?
# A pretrained model already understands English grammar and common text patterns.
# Fine-tuning teaches it the specific SAMSum task with much less data and time.
MODEL_NAME = "facebook/bart-base"

# Dialogue inputs are allowed to be longer than summaries.
# The roadmap asks for 512 input tokens and 128 summary tokens.
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

# Week 3 requires at least 3 epochs. You can increase to 4 or 5 if GPU time allows.
NUM_TRAIN_EPOCHS = 3

# These sample limits keep training realistic for a student/Colab setup.
# Set these to None if you want to train on the full SAMSum split.
MAX_TRAIN_SAMPLES = None
MAX_VALIDATION_SAMPLES = None
MAX_TEST_SAMPLES = None
MAX_SAMPLE_SUMMARIES = 10


def get_device_name() -> str:
    """
    Return GPU name if available, otherwise return CPU.
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def load_samsum_dataset() -> DatasetDict:
    """
    Load the SAMSum dataset from Hugging Face.
    """
    return load_dataset("knkarthick/samsum")


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """
    Load the BART tokenizer and model.

    The tokenizer converts text into token IDs.
    The model learns to convert dialogue token IDs into summary token IDs.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


def select_optional_subset(dataset_split, max_samples):
    """
    Select a smaller split only when max_samples is provided.

    Basic question: Why use small subsets sometimes?
        Transformer training can be slow on CPU or free Colab GPUs. A subset is
        useful for testing that the pipeline works before spending hours on the
        full training run.
    """
    if max_samples is None:
        return dataset_split

    return dataset_split.select(range(min(max_samples, len(dataset_split))))


def preprocess_function(examples: Dict[str, list], tokenizer) -> Dict[str, list]:
    """
    Tokenize dialogues and summaries.

    Args:
        examples: Batch of dataset rows.
        tokenizer: Hugging Face tokenizer.

    Returns:
        Tokenized inputs and labels.
    """
    # NOTE: We add "summarize:" prefix during preprocessing.
    # This helps the model understand the task. During inference in src/inference.py,
    # we must add the same prefix to maintain consistency.
    # This is a deliberate design choice for this project.
    inputs = ["summarize: " + dialogue for dialogue in examples["dialogue"]]
    targets = examples["summary"]

    # Tokenize dialogue text. Truncation is needed because transformer models
    # have a maximum context length and cannot accept unlimited text.
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    # Tokenize summary text as labels.
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


class SAMSumDataset(Dataset):
    """
    PyTorch Dataset wrapper for tokenized SAMSum examples.

    Basic question: Why create a Dataset class?
        A Dataset tells PyTorch/Trainer how to fetch one training example at a
        time. Hugging Face datasets can already do this, but the roadmap asks us
        to create a dataset class, so this wrapper makes that step explicit.
    """

    def __init__(self, tokenized_split):
        self.tokenized_split = tokenized_split

    def __len__(self) -> int:
        return len(self.tokenized_split)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.tokenized_split[index]


def prepare_tokenized_dataset(dataset: DatasetDict, tokenizer) -> DatasetDict:
    """
    Preprocess train, validation, and test splits.

    If MAX_*_SAMPLES is set, we select small subsets so the script can run faster.
    """
    small_dataset = DatasetDict(
        {
            "train": select_optional_subset(dataset["train"], MAX_TRAIN_SAMPLES),
            "validation": select_optional_subset(dataset["validation"], MAX_VALIDATION_SAMPLES),
            "test": select_optional_subset(dataset["test"], MAX_TEST_SAMPLES),
        }
    )

    tokenized_dataset = small_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=small_dataset["train"].column_names,
    )

    return small_dataset, tokenized_dataset


def create_training_arguments(output_dir: str = "models/transformer-checkpoints"):
    """
    Create Trainer settings.

    Important settings:
        evaluation_strategy='epoch' evaluates after each epoch.
        save_strategy='epoch' saves checkpoints after each epoch.
        predict_with_generate=True lets Trainer generate summaries for ROUGE.
    """
    common_args = {
        "output_dir": output_dir,
        "save_strategy": "epoch",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "weight_decay": 0.01,
        "save_total_limit": 2,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "predict_with_generate": True,
        "generation_max_length": MAX_TARGET_LENGTH,
        "logging_steps": 25,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
    }

    try:
        # Newer Transformers versions use eval_strategy.
        return Seq2SeqTrainingArguments(
            eval_strategy="epoch",
            **common_args,
        )
    except TypeError:
        # Some older Transformers versions use evaluation_strategy.
        return Seq2SeqTrainingArguments(
            evaluation_strategy="epoch",
            **common_args,
        )


def create_trainer(tokenizer, model, tokenized_dataset: DatasetDict, output_dir: str):
    """
    Create the Hugging Face Seq2SeqTrainer.
    """
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    training_args = create_training_arguments(output_dir=output_dir)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": SAMSumDataset(tokenized_dataset["train"]),
        "eval_dataset": SAMSumDataset(tokenized_dataset["validation"]),
        "data_collator": data_collator,
        "compute_metrics": partial(compute_rouge_metrics, tokenizer=tokenizer),
    }

    try:
        # Newer Transformers versions use processing_class instead of tokenizer.
        trainer = Seq2SeqTrainer(
            processing_class=tokenizer,
            **trainer_kwargs,
        )
    except TypeError:
        # Older Transformers versions still use tokenizer.
        trainer = Seq2SeqTrainer(
            tokenizer=tokenizer,
            **trainer_kwargs,
        )

    return trainer


def plot_training_curve(trainer, output_path: str = "results/training_curve.png") -> None:
    """
    Plot and save the training and validation loss curve.
    """
    log_history = trainer.state.log_history

    train_logs = [
        log for log in log_history
        if "loss" in log and "epoch" in log
    ]
    validation_logs = [
        log for log in log_history
        if "eval_loss" in log and "epoch" in log
    ]

    if not train_logs and not validation_logs:
        print("No loss values found, so no curve was created.")
        return

    plt.figure(figsize=(8, 5))

    if train_logs:
        train_epochs = [log["epoch"] for log in train_logs]
        train_losses = [log["loss"] for log in train_logs]
        plt.plot(train_epochs, train_losses, marker="o", label="Training loss")

    if validation_logs:
        validation_epochs = [log["epoch"] for log in validation_logs]
        validation_losses = [log["eval_loss"] for log in validation_logs]
        plt.plot(validation_epochs, validation_losses, marker="s", label="Validation loss")

    plt.title("Training and Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def decode_generated_summaries(predictions, tokenizer) -> List[str]:
    """
    Decode generated token IDs into readable summaries.
    """
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = predictions.astype(np.int64)
    vocabulary_size = len(tokenizer)
    predictions = np.where(
        (predictions >= 0) & (predictions < vocabulary_size),
        predictions,
        tokenizer.pad_token_id,
    )

    return tokenizer.batch_decode(predictions, skip_special_tokens=True)


def calculate_single_rouge(reference_summary: str, generated_summary: str) -> Dict[str, float]:
    """
    Calculate ROUGE scores for one generated summary.
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


def create_prediction_rows(test_dataset, generated_summaries: List[str]) -> pd.DataFrame:
    """
    Create test-set prediction rows with per-example ROUGE scores.
    """
    rows = []

    for example, generated_summary in zip(test_dataset, generated_summaries):
        reference_summary = example["summary"]
        rouge_scores = calculate_single_rouge(reference_summary, generated_summary)

        rows.append(
            {
                "id": example.get("id", ""),
                "dialogue": example["dialogue"],
                "reference_summary": reference_summary,
                "generated_summary": generated_summary,
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"],
            }
        )

    return pd.DataFrame(rows)


def evaluate_on_test_set(
    trainer,
    tokenizer,
    original_test_dataset,
    tokenized_test_dataset,
    metrics_output_path: str = "results/transformer_metrics.csv",
    predictions_output_path: str = "results/transformer_test_predictions.csv",
    sample_output_path: str = "results/transformer_sample_predictions.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate summaries on the test set and save ROUGE metrics.

    Basic question: Why use the test set here?
        The validation set is used during training to monitor progress. The test
        set is kept for final reporting so the result is a fair comparison with
        the Week 2 baseline.
    
    CRITICAL: This function now uses the inference pipeline approach instead of trainer.predict()
    to ensure consistency with manual inference. The model expects "summarize:" prefix in inputs,
    which the trainer.predict() was not providing, causing poor summaries.
    """
    from transformers import pipeline as hf_pipeline
    
    # Create a summarization pipeline using the trained model
    device = 0 if torch.cuda.is_available() else -1
    summarizer = hf_pipeline(
        task="summarization",
        model=trainer.model,
        tokenizer=tokenizer,
        device=device,
    )
    
    # Generate summaries using the same approach as manual inference
    # This ensures the "summarize:" prefix is added correctly
    generated_summaries = []
    for dialogue in original_test_dataset["dialogue"]:
        # Add the task prefix that the model was trained with
        input_text = "summarize: " + dialogue
        
        result = summarizer(
            input_text,
            max_length=MAX_TARGET_LENGTH,
            min_length=10,
            num_beams=8,  # Phase 4: Beam search for better quality
            do_sample=False,
            early_stopping=True,
        )
        summary = result[0]["summary_text"]
        
        # CRITICAL: Remove the "summarize:" prefix if the model outputs it
        # The model should only generate the summary, not the prefix
        if summary.startswith("summarize:"):
            summary = summary[len("summarize:"):].strip()
        
        generated_summaries.append(summary)
    
    predictions_df = create_prediction_rows(
        test_dataset=original_test_dataset,
        generated_summaries=generated_summaries,
    )
    predictions_df.to_csv(predictions_output_path, index=False)

    sample_df = predictions_df.head(MAX_SAMPLE_SUMMARIES).copy()
    sample_df.to_csv(sample_output_path, index=False)

    # Calculate ROUGE metrics
    rouge_scores = []
    for ref_summary, gen_summary in zip(
        original_test_dataset["summary"],
        generated_summaries
    ):
        score = calculate_single_rouge(ref_summary, gen_summary)
        rouge_scores.append(score)
    
    avg_rouge1 = np.mean([s["rouge1"] for s in rouge_scores]) * 100
    avg_rouge2 = np.mean([s["rouge2"] for s in rouge_scores]) * 100
    avg_rougeL = np.mean([s["rougeL"] for s in rouge_scores]) * 100
    
    metrics = {
        "test_rouge1": avg_rouge1,
        "test_rouge2": avg_rouge2,
        "test_rougeL": avg_rougeL,
        "model_name": "BART Fine-tuned Transformer",
        "evaluated_samples": len(predictions_df),
    }
    metrics_df = save_metrics_csv(
        metrics=metrics,
        output_path=metrics_output_path,
    )

    return metrics_df, predictions_df, sample_df


def create_comparison_table(
    transformer_metrics_df: pd.DataFrame,
    baseline_table_path: str = "results/comparison_table.csv",
    output_path: str = "results/comparison_table.csv",
) -> pd.DataFrame:
    """
    Create the Week 2 baseline vs Week 3 transformer comparison table.
    """
    baseline_columns = [
        "model_name",
        "model_type",
        "method",
        "evaluated_samples",
        "rouge1",
        "rouge2",
        "rougeL",
    ]

    if Path(baseline_table_path).exists():
        baseline_df = pd.read_csv(baseline_table_path)
    else:
        baseline_df = pd.DataFrame(columns=baseline_columns)

    # Avoid adding a duplicate transformer row when the script is run again.
    if not baseline_df.empty and "model_name" in baseline_df.columns:
        baseline_df = baseline_df[
            baseline_df["model_name"] != "BART Fine-tuned Transformer"
        ]

    metrics = transformer_metrics_df.iloc[0].to_dict()
    transformer_row = {
        "model_name": "BART Fine-tuned Transformer",
        "model_type": "Transformer",
        "method": "Fine-tuned facebook/bart-base",
        "evaluated_samples": metrics.get("evaluated_samples", MAX_TEST_SAMPLES or "full test set"),
        "rouge1": metrics.get("test_rouge1", metrics.get("eval_rouge1")),
        "rouge2": metrics.get("test_rouge2", metrics.get("eval_rouge2")),
        "rougeL": metrics.get("test_rougeL", metrics.get("eval_rougeL")),
    }

    comparison_df = pd.concat(
        [
            baseline_df[baseline_columns],
            pd.DataFrame([transformer_row]),
        ],
        ignore_index=True,
    )
    comparison_df.to_csv(output_path, index=False)

    return comparison_df


def train_transformer_model(
    model_output_dir: str = "models/transformer",
    checkpoint_dir: str = "models/transformer-checkpoints",
    results_dir: str = "results",
) -> Tuple[Seq2SeqTrainer, pd.DataFrame, pd.DataFrame]:
    """
    Run the full Phase 3 training pipeline.

    This function:
        1. Loads SAMSum.
        2. Loads facebook/bart-base.
        3. Tokenizes data.
        4. Creates a PyTorch Dataset wrapper.
        5. Trains with Trainer API for at least 3 epochs.
        6. Saves the trained model/checkpoints.
        7. Saves loss curve, ROUGE metrics, samples, and comparison table.
    """
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print("Device:", get_device_name())

    dataset = load_samsum_dataset()
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME)
    small_dataset, tokenized_dataset = prepare_tokenized_dataset(dataset, tokenizer)

    trainer = create_trainer(
        tokenizer=tokenizer,
        model=model,
        tokenized_dataset=tokenized_dataset,
        output_dir=checkpoint_dir,
    )

    trainer.train()

    # Evaluate on validation set after training.
    # This is mainly for checking whether the model learned during training.
    validation_metrics = trainer.evaluate()
    save_metrics_csv(
        metrics=validation_metrics,
        output_path=f"{results_dir}/transformer_validation_metrics.csv",
    )

    # Save final model and tokenizer.
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    plot_training_curve(
        trainer=trainer,
        output_path=f"{results_dir}/training_curve.png",
    )

    metrics_df, predictions_df, sample_df = evaluate_on_test_set(
        trainer=trainer,
        tokenizer=tokenizer,
        original_test_dataset=small_dataset["test"],
        tokenized_test_dataset=tokenized_dataset["test"],
        metrics_output_path=f"{results_dir}/transformer_metrics.csv",
        predictions_output_path=f"{results_dir}/transformer_test_predictions.csv",
        sample_output_path=f"{results_dir}/transformer_sample_predictions.csv",
    )

    create_comparison_table(
        transformer_metrics_df=metrics_df,
        baseline_table_path=f"{results_dir}/comparison_table.csv",
        output_path=f"{results_dir}/comparison_table.csv",
    )

    return trainer, metrics_df, sample_df


if __name__ == "__main__":
    train_transformer_model()
