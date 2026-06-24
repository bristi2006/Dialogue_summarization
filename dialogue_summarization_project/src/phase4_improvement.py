"""
Week 4: Improvement and ablation experiment.

Improvement used in this file:
    Compare multiple beam search configurations for decoding.

Greedy decoding (num_beams=1) chooses the best next word one step at a time.
Beam search (num_beams > 1) keeps several possible summaries at each step.
This experiment tests different beam widths to find the best configuration.
Beam search is often better for summarization because it checks more than one
possible sentence path, and different beam sizes offer different quality/speed tradeoffs.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    # The CSV and markdown report are the most important Phase 4 outputs.
    # If matplotlib is not installed, we skip only the optional chart.
    plt = None


# Use the model saved by Week 3 training.
MODEL_PATH = "models/transformer"

# Keep this small so the experiment is quick and beginner-friendly.
MAX_EXAMPLES = 200

# These values control summary length during generation.
MAX_SUMMARY_LENGTH = 80
MIN_SUMMARY_LENGTH = 10

# Beam sizes to test in the ablation study.
BEAM_SIZES = [1, 2, 4, 6, 8]

# Number of worst-performing examples to analyze in error analysis.
ERROR_ANALYSIS_SAMPLES = 15


def get_device() -> torch.device:
    """
    Choose GPU if available, otherwise use CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_saved_model(model_path: str = MODEL_PATH):
    """
    Load the fine-tuned model and tokenizer saved in Week 3.

    The tokenizer converts text into numbers.
    The model converts dialogue numbers into summary numbers.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Saved model folder not found: {model_path}. "
            "Run Week 3 training first with: python src/train_transformer.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    device = get_device()
    model.to(device)
    model.eval()

    return tokenizer, model, device


def load_test_examples(max_examples: int = MAX_EXAMPLES) -> pd.DataFrame:
    """
    Load a small part of the SAMSum test set.

    We use a small sample because this is an ablation experiment, not another
    full training run.
    """
    dataset = load_dataset("knkarthick/samsum", split="test")
    dataset = dataset.select(range(min(max_examples, len(dataset))))

    return pd.DataFrame(dataset)


def generate_one_summary(
    dialogue: str,
    tokenizer,
    model,
    device: torch.device,
    num_beams: int,
) -> str:
    """
    Generate one summary with either greedy decoding or beam search.

    If num_beams is 1, Hugging Face uses greedy decoding.
    If num_beams is larger than 1, Hugging Face uses beam search.
    """
    # Add the same task prefix used during Week 3 preprocessing.
    input_text = "summarize: " + dialogue

    # Convert the dialogue text into token IDs for the model.
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}

    # torch.no_grad() saves memory because we are only generating, not training.
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=MAX_SUMMARY_LENGTH,
            min_length=MIN_SUMMARY_LENGTH,
            num_beams=num_beams,
            do_sample=False,
            early_stopping=True,
        )

    # Convert token IDs back into normal readable text.
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def generate_summaries(
    examples_df: pd.DataFrame,
    tokenizer,
    model,
    device: torch.device,
    num_beams: int,
) -> List[str]:
    """
    Generate summaries for all selected examples.
    """
    summaries = []

    for dialogue in examples_df["dialogue"]:
        summary = generate_one_summary(
            dialogue=dialogue,
            tokenizer=tokenizer,
            model=model,
            device=device,
            num_beams=num_beams,
        )
        summaries.append(summary)

    return summaries


def calculate_rouge_scores(reference: str, prediction: str) -> Dict[str, float]:
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-L for one generated summary.

    ROUGE compares overlap between the generated summary and the human summary.
    Higher ROUGE usually means the generated summary is closer to the reference.
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )
    scores = scorer.score(reference, prediction)

    return {
        "rouge1": scores["rouge1"].fmeasure * 100,
        "rouge2": scores["rouge2"].fmeasure * 100,
        "rougeL": scores["rougeL"].fmeasure * 100,
    }


def build_prediction_table(
    examples_df: pd.DataFrame,
    beam_summaries_dict: Dict[int, List[str]],
) -> pd.DataFrame:
    """
    Create a row-by-row table for multiple beam sizes.

    beam_summaries_dict: {beam_size: [summaries for all examples]}

    For each example and beam size, we calculate ROUGE scores.
    The table stores generated summaries and ROUGE scores for each beam size.
    """
    rows = []

    for index, example in examples_df.iterrows():
        reference_summary = example["summary"]
        dialogue = example["dialogue"]
        row_dict = {
            "id": example.get("id", ""),
            "dialogue": dialogue,
            "reference_summary": reference_summary,
        }

        # Calculate ROUGE scores for each beam size.
        for beam_size in sorted(beam_summaries_dict.keys()):
            prediction = beam_summaries_dict[beam_size][index]
            scores = calculate_rouge_scores(reference_summary, prediction)

            row_dict[f"beam{beam_size}_summary"] = prediction
            row_dict[f"beam{beam_size}_rouge1"] = round(scores["rouge1"], 4)
            row_dict[f"beam{beam_size}_rouge2"] = round(scores["rouge2"], 4)
            row_dict[f"beam{beam_size}_rougeL"] = round(scores["rougeL"], 4)

        rows.append(row_dict)

    return pd.DataFrame(rows)


def build_ablation_table(predictions_df: pd.DataFrame, beam_sizes: List[int]) -> pd.DataFrame:
    """
    Create the final ablation table with average ROUGE scores for each beam size.

    One row per beam size, showing average ROUGE-1, ROUGE-2, and ROUGE-L.
    """
    rows = []

    for beam_size in beam_sizes:
        # Get column names for this beam size.
        rouge1_col = f"beam{beam_size}_rouge1"
        rouge2_col = f"beam{beam_size}_rouge2"
        rougeL_col = f"beam{beam_size}_rougeL"

        # Determine experiment name based on beam size.
        if beam_size == 1:
            experiment_name = "Greedy decoding"
        else:
            experiment_name = f"Beam search (size {beam_size})"

        rows.append(
            {
                "Experiment": experiment_name,
                "Num Beams": beam_size,
                "ROUGE-1": round(predictions_df[rouge1_col].mean(), 4),
                "ROUGE-2": round(predictions_df[rouge2_col].mean(), 4),
                "ROUGE-L": round(predictions_df[rougeL_col].mean(), 4),
            }
        )

    return pd.DataFrame(rows)


def choose_winner(ablation_df: pd.DataFrame) -> tuple:
    """
    Find the beam size with the best ROUGE-L score.

    Returns a tuple: (experiment_name, num_beams, rouge_l_score)

    ROUGE-L is useful here because it rewards longer matching word sequences,
    which often matters for summary fluency.
    """
    best_row = ablation_df.sort_values("ROUGE-L", ascending=False).iloc[0]
    return (
        best_row["Experiment"],
        best_row["Num Beams"],
        best_row["ROUGE-L"],
    )


def label_error_category(
    dialogue: str,
    reference: str,
    prediction: str,
    rouge_l: float,
) -> str:
    """
    Assign an error category to one summary using rule-based heuristics.

    This is beginner-friendly and helps identify common failure patterns.

    Categories:
    - too_short: Generated summary has < 5 words.
    - too_long: Generated summary is > 2x the reference length (min 30 words).
    - missing_information: ROUGE-L < 15 (summary is dissimilar to reference).
    - hallucination: >30% of prediction words don't appear in dialogue or reference.
    - wrong_names: Capitalized names in reference don't appear in prediction.
    - acceptable: None of the above issues detected.
    """
    prediction_words = prediction.split()
    reference_words = reference.split()
    dialogue_words = dialogue.split()

    # Rule 1: Too short.
    if len(prediction_words) < 5:
        return "too_short"

    # Rule 2: Too long.
    if len(prediction_words) > max(30, len(reference_words) * 2):
        return "too_long"

    # Rule 3: Missing information (low ROUGE-L indicates dissimilarity).
    if rouge_l < 15:
        return "missing_information"

    # Rule 4: Hallucination - check if many words are not in dialogue or reference.
    dialogue_lower = set(w.lower() for w in dialogue_words)
    reference_lower = set(w.lower() for w in reference_words)
    combined_vocab = dialogue_lower | reference_lower

    hallucinated_words = [w for w in prediction_words if w.lower() not in combined_vocab]
    hallucination_rate = len(hallucinated_words) / len(prediction_words) if prediction_words else 0

    if hallucination_rate > 0.3:  # More than 30% hallucinated.
        return "hallucination"

    # Rule 5: Wrong names - capitalized words in reference missing from prediction.
    # Simple heuristic: a capitalized word is likely a proper noun (name).
    capitalized_in_ref = [
        w for w in reference_words
        if w and w[0].isupper() and w not in ["I"]  # Exclude "I" to avoid false positives.
    ]
    prediction_lower = set(w.lower() for w in prediction_words)
    missing_names = [w for w in capitalized_in_ref if w.lower() not in prediction_lower]

    if missing_names:
        return "wrong_names"

    # Rule 6: Acceptable if none of the above.
    return "acceptable"


def build_error_analysis(predictions_df: pd.DataFrame, beam_size: int = 1) -> pd.DataFrame:
    """
    Select the worst-performing examples for error analysis.

    Finds the ERROR_ANALYSIS_SAMPLES (default 15) examples with the lowest ROUGE-L
    scores. These are the model's biggest failures, which are more useful to study
    than random examples.

    beam_size: Which beam size's summaries to analyze (default 1 for greedy).
    """
    # Get the ROUGE-L column for the specified beam size.
    rouge_col = f"beam{beam_size}_rougeL"
    summary_col = f"beam{beam_size}_summary"

    # Sort by ROUGE-L ascending to find the worst-performing examples.
    worst_examples = predictions_df.nsmallest(ERROR_ANALYSIS_SAMPLES, rouge_col)

    rows = []
    for _, row in worst_examples.iterrows():
        category = label_error_category(
            dialogue=row["dialogue"],
            reference=row["reference_summary"],
            prediction=row[summary_col],
            rouge_l=row[rouge_col],
        )
        rows.append(
            {
                "id": row["id"],
                "dialogue": row["dialogue"],
                "reference_summary": row["reference_summary"],
                "generated_summary": row[summary_col],
                "rouge_l": row[rouge_col],
                "error_category": category,
            }
        )

    return pd.DataFrame(rows)


def save_error_category_chart(error_df: pd.DataFrame, output_path: str) -> None:
    """
    Save a small bar chart showing error category counts.
    """
    if plt is None:
        print("matplotlib is not installed, so the error category chart was skipped.")
        return

    category_counts = error_df["error_category"].value_counts()

    plt.figure(figsize=(7, 4))
    category_counts.plot(kind="bar", color="#2f7f73")
    plt.title("Phase 4 Error Category Distribution")
    plt.xlabel("Error Category")
    plt.ylabel("Number of Examples")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """
    Convert a DataFrame into a markdown table without extra packages.

    Pandas has a to_markdown() method, but it often needs the optional tabulate
    package. This helper keeps the project requirements simple.
    """
    columns = list(dataframe.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []

    for _, row in dataframe.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator] + rows)


def write_markdown_report(
    ablation_df: pd.DataFrame,
    error_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Write a comprehensive markdown report of the Phase 4 experiment.

    Includes:
    - Explanation of beam search ablation study
    - Ablation results table
    - Best configuration identified
    - Error analysis findings
    - Conclusions
    """
    best_exp, best_beam, best_rouge = choose_winner(ablation_df)
    ablation_markdown = dataframe_to_markdown(ablation_df)

    # Prepare error analysis findings.
    error_category_counts = error_df["error_category"].value_counts()
    most_common_category = error_category_counts.idxmax() if len(error_category_counts) > 0 else "None"
    most_common_count = error_category_counts.max() if len(error_category_counts) > 0 else 0

    error_summary_lines = []
    for category in error_category_counts.index:
        count = error_category_counts[category]
        error_summary_lines.append(f"- **{category}**: {count} examples")

    error_summary = "\n".join(error_summary_lines) if error_summary_lines else "No error data available."

    report_text = f"""# Week 4: Improvement and Ablation Experiment

## Experiment Overview

This phase upgrades the Week 3 transformer model by systematically testing different decoding configurations.
No model retraining is performed—the same fine-tuned BART model is used throughout.

## Beam Search Ablation Study

### Why Test Multiple Beam Sizes?

Beam search is a decoding method that keeps multiple possible summary hypotheses at each generation step,
then selects the best complete summary. Different beam widths offer different quality-speed tradeoffs:

- **Num Beams = 1**: Greedy decoding. Fastest, but may miss better summaries.
- **Num Beams = 2-8**: Progressively wider search. Slower, but may find better summaries.

This study tests beam sizes {min(ablation_df['Num Beams'])} through {max(ablation_df['Num Beams'])}
to find the optimal balance between quality and computational cost.

## Ablation Results

{ablation_markdown}

### Best Configuration

- **Best Method**: {best_exp}
- **Beam Size**: {best_beam}
- **ROUGE-L Score**: {best_rouge}

The results show how beam width impacts summary quality on the SAMSum test set.

## Error Analysis Findings

This section analyzes the {len(error_df)} worst-performing examples (lowest ROUGE-L scores)
to understand common failure modes.

### Error Category Distribution

{error_summary}

### Most Common Error

The most frequent error category is **{most_common_category}** ({most_common_count} examples),
indicating the model's primary weakness.

## Discussion

### Key Observations

1. **Beam Size Impact**: Increasing beam size generally provides diminishing returns beyond a certain point.
2. **Error Patterns**: The error analysis reveals which failure modes are most prevalent.
3. **Quality-Speed Tradeoff**: Choose beam size based on application requirements (latency vs. quality).

### Why This Matters

This ablation study helps identify:
- Which decoding configuration produces the best summaries
- Common failure modes to address in future improvements
- The practical limits of beam search for this task

## Conclusion

The multi-beam ablation study provides systematic evidence about decoding strategies.
The identified best configuration (beam size {best_beam}) can be used as the standard
for future development, and the error analysis guides which problems to tackle next.

---

*Generated automatically by Phase 4 improvement experiment.*
"""

    Path(output_path).write_text(report_text, encoding="utf-8")


def run_phase4_experiment(
    model_path: str = MODEL_PATH,
    results_dir: str = "results",
    max_examples: int = MAX_EXAMPLES,
    beam_sizes: List[int] = BEAM_SIZES,
) -> None:
    """
    Run the full Week 4 improvement experiment with multi-beam ablation study.

    Outputs:
        results/phase4_predictions.csv      - Predictions for all beam sizes
        results/phase4_ablation_results.csv - Summary ROUGE scores per beam size
        results/phase4_error_analysis.csv   - Worst-performing examples and error categories
        results/phase4_error_categories.png - Chart of error category distribution
        results/phase4_report.md            - Comprehensive analysis report
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print("Loading saved transformer model...")
    tokenizer, model, device = load_saved_model(model_path)

    print("Loading test examples...")
    examples_df = load_test_examples(max_examples=max_examples)

    # Generate summaries for each beam size.
    print(f"Generating summaries for beam sizes: {beam_sizes}")
    beam_summaries_dict = {}

    for beam_size in beam_sizes:
        print(f"  - Generating summaries with num_beams={beam_size}...")
        beam_summaries_dict[beam_size] = generate_summaries(
            examples_df=examples_df,
            tokenizer=tokenizer,
            model=model,
            device=device,
            num_beams=beam_size,
        )

    # Build prediction table with all beam sizes.
    print("Building prediction table...")
    predictions_df = build_prediction_table(
        examples_df=examples_df,
        beam_summaries_dict=beam_summaries_dict,
    )
    predictions_df.to_csv(f"{results_dir}/phase4_predictions.csv", index=False)
    print(f"  - Saved: {results_dir}/phase4_predictions.csv")

    # Build ablation results table.
    print("Building ablation table...")
    ablation_df = build_ablation_table(predictions_df, beam_sizes=beam_sizes)
    ablation_df.to_csv(f"{results_dir}/phase4_ablation_results.csv", index=False)
    print(f"  - Saved: {results_dir}/phase4_ablation_results.csv")

    # Perform error analysis on the best beam size (greedy decoding, num_beams=1).
    print("Performing error analysis on greedy decoding (beam size 1)...")
    error_df = build_error_analysis(predictions_df, beam_size=1)
    error_df.to_csv(f"{results_dir}/phase4_error_analysis.csv", index=False)
    print(f"  - Saved: {results_dir}/phase4_error_analysis.csv")

    # Save error category chart.
    print("Generating error category chart...")
    save_error_category_chart(
        error_df=error_df,
        output_path=f"{results_dir}/phase4_error_categories.png",
    )
    print(f"  - Saved: {results_dir}/phase4_error_categories.png")

    # Write comprehensive markdown report.
    print("Writing markdown report...")
    write_markdown_report(
        ablation_df=ablation_df,
        error_df=error_df,
        output_path=f"{results_dir}/phase4_report.md",
    )
    print(f"  - Saved: {results_dir}/phase4_report.md")

    # Print summary results.
    print("\n" + "=" * 60)
    print("PHASE 4 EXPERIMENT COMPLETE")
    print("=" * 60)
    best_exp, best_beam, best_rouge = choose_winner(ablation_df)
    print(f"Best Configuration: {best_exp} (beam_size={best_beam})")
    print(f"Best ROUGE-L Score: {best_rouge}")
    print("\nAblation Results Summary:")
    print(ablation_df.to_string(index=False))
    print("\nError Category Distribution:")
    print(error_df["error_category"].value_counts().to_string())
    print("=" * 60)


if __name__ == "__main__":
    run_phase4_experiment()
