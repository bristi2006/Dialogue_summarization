 # dialogue_summarization_project

Beginner-friendly NLP internship project for dialogue summarization using the SAMSum dataset.

This repository currently contains work up to **Week 4**:

- Week 1: dataset loading, exploration, cleaning, tokenization, and visual analysis
- Week 2: TF-IDF extractive baseline model
- Week 3: BART transformer fine-tuning pipeline for SAMSum summarization
- Week 4: improvement and ablation experiment comparing greedy decoding with beam search decoding

## Project Overview

Dialogue summarization means converting a conversation into a short summary.

Example:

- Dialogue: A chat between two or more people.
- Summary: A short description of the important points in that chat.

Week 1 focuses on understanding the dataset and preparing reusable preprocessing
code. Week 2 adds a simple baseline model, and Week 3 fine-tunes a pretrained
transformer model. Week 4 improves the generation step and records an ablation
study.

## Dataset

This project uses the SAMSum dataset from Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("knkarthick/samsum")
```

The dataset contains conversations and human-written summaries.

Main columns:

- `id`: Unique sample ID
- `dialogue`: Conversation text
- `summary`: Human-written summary

Dataset splits:

- `train`: Used for training in later phases
- `validation`: Used for checking model performance during training
- `test`: Used for final evaluation

## Folder Structure

```text
dialogue_summarization_project/
|
|-- data_analysis.ipynb
|-- baseline_model.ipynb
|-- transformer_model.ipynb
|-- requirements.txt
|-- README.md
|
|-- src/
|   |-- preprocess.py
|   |-- train_baseline.py
|   |-- train_transformer.py
|   |-- evaluate.py
|   |-- inference.py
|   |-- phase4_improvement.py
|   |-- utils.py
|
|-- results/
|-- models/
|-- report/
|-- demo/
```

## Installation

### Option 1: Google Colab

Open `data_analysis.ipynb` in Google Colab and run the first installation cell:

```python
!pip install -r requirements.txt
```

If you upload only the notebook to Colab, install the libraries directly.
Keep the Transformers version pinned because newer versions may not include the
same summarization pipeline task name:

```python
!pip install datasets pandas numpy matplotlib seaborn nltk transformers==4.52.4 torch evaluate accelerate sentencepiece rouge-score
```

### Option 2: Local Computer

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
data_analysis.ipynb
```

## How to Run the Notebook

1. Open `data_analysis.ipynb`.
2. Run cells from top to bottom.
3. The notebook will download the SAMSum dataset.
4. It will show dataset overview, missing values, length distributions, common words, tokenization examples, and cleaning examples.
5. Plots will be saved inside the `results/` folder.

## Week 3 Transformer Training

The Week 3 roadmap is implemented in:

```text
transformer_model.ipynb
src/train_transformer.py
```

Run the transformer training script from the project folder:

```bash
python src/train_transformer.py
```

The script uses `facebook/bart-base` because BART is an encoder-decoder
transformer designed for text generation tasks like abstractive summarization.
`bart-base` is smaller and easier to fine-tune than `bart-large-cnn`, while T5
and PEGASUS are left as valid future alternatives so the Week 3 comparison stays
focused on one model.

Expected Week 3 outputs:

- Fine-tuned model in `models/transformer/`
- Epoch checkpoints in `models/transformer-checkpoints/`
- Training and validation loss curve in `results/training_curve.png`
- Transformer test ROUGE scores in `results/transformer_metrics.csv`
- Full transformer test predictions in `results/transformer_test_predictions.csv`
- 10 sample generated summaries in `results/transformer_sample_predictions.csv`
- Baseline vs Transformer table in `results/comparison_table.csv`

## Week 4 Improvement and Ablation Experiment

The Week 4 phase is implemented in:

```text
src/phase4_improvement.py
```

Run this script after the Week 3 transformer model has been trained and saved:

```bash
python src/phase4_improvement.py
```

Improvement technique used:

- Greedy decoding: chooses the best next token at each step.
- Beam search decoding: keeps several possible summary paths and chooses the best final summary.

This is a simple improvement because it does not retrain the model. It loads the
saved model from `models/transformer/` and changes only the decoding strategy.

Expected Week 4 outputs:

- Ablation table: `results/phase4_ablation_results.csv`
- Greedy vs beam predictions: `results/phase4_predictions.csv`
- Error analysis examples: `results/phase4_error_analysis.csv`
- Error category chart: `results/phase4_error_categories.png`
- Markdown explanation report: `results/phase4_report.md`

## Earlier Outputs

The data analysis notebook saves plots such as:

- Dialogue length distribution
- Summary length distribution
- Most common words in dialogues
- Most common words in summaries

These files are saved in:

```text
results/
```

The Week 2 baseline saves:

- `results/sample_predictions.csv`
- `results/rouge_scores.csv`
- `results/comparison_table.csv`

## Next Phases

Possible future phases:

- Week 5: Build a simple demo app
