"""
Inference helpers for the transformer summarization model.

Inference means using a trained model to generate summaries for new dialogues.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def load_summarization_pipeline(model_path: str = "models/transformer"):
    """
    Load a saved summarization model and tokenizer.

    Args:
        model_path: Folder containing the saved model and tokenizer.

    Returns:
        A Hugging Face summarization pipeline.
    """
    # Use GPU if available, otherwise use CPU.
    device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    summarizer = pipeline(
        task="summarization",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    return summarizer


def generate_summary(
    dialogue: str,
    summarizer,
    max_length: int = 80,
    min_length: int = 10,
    num_beams: int = 8,  # Phase 4: Beam search with 8 beams produces better summaries
) -> str:
    """
    Generate a summary for one dialogue.

    Args:
        dialogue: Conversation text.
        summarizer: Hugging Face summarization pipeline.
        max_length: Maximum summary length.
        min_length: Minimum summary length.
        num_beams: Number of beams for beam search (default 8 for better quality).

    Returns:
        Generated summary as a string.
    """
    # CRITICAL: Add the same task prefix used during training in src/train_transformer.py
    # During training, model learns: "summarize: " + dialogue → summary
    # During inference, must provide same prefix for correct generation
    input_text = "summarize: " + dialogue

    result = summarizer(
        input_text,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        do_sample=False,
        early_stopping=True,  # Stop early when model produces [END] token
    )

    summary = result[0]["summary_text"]
    
    # Remove the "summarize:" prefix if the model outputs it
    # The model should only generate the summary, not the prefix
    if summary.startswith("summarize:"):
        summary = summary[len("summarize:"):].strip()
    
    return summary


if __name__ == "__main__":
    # Example usage after training:
    #     python src/inference.py
    example_dialogue = (
        "A: Are you coming to the meeting today?\n"
        "B: Yes, I will be there at 3 PM.\n"
        "A: Great, please bring the project report."
    )

    summarization_pipeline = load_summarization_pipeline("models/transformer")
    summary = generate_summary(example_dialogue, summarization_pipeline)

    print("Generated Summary:")
    print(summary)
