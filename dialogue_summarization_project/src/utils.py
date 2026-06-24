"""
Utility functions for data analysis and plotting.

These helpers keep the notebook clean and make repeated tasks easier.
"""

from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocess import tokenize_text


def create_results_folder(folder_path: str = "results") -> Path:
    """
    Create the results folder if it does not already exist.

    Args:
        folder_path: Location where plots and outputs will be saved.

    Returns:
        A Path object pointing to the results folder.
    """
    results_path = Path(folder_path)
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path


def dataset_split_sizes(dataset) -> pd.DataFrame:
    """
    Count the number of samples in each dataset split.

    Args:
        dataset: Hugging Face DatasetDict object.

    Returns:
        A DataFrame with split names and sample counts.
    """
    rows = []

    for split_name in dataset.keys():
        rows.append(
            {
                "split": split_name,
                "number_of_samples": len(dataset[split_name]),
            }
        )

    return pd.DataFrame(rows)


def get_missing_value_report(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Count missing values for each column in a DataFrame.
    """
    missing_counts = dataframe.isnull().sum()

    return pd.DataFrame(
        {
            "column": missing_counts.index,
            "missing_values": missing_counts.values,
        }
    )


def get_most_common_words(texts: Iterable[str], top_n: int = 20) -> List[Tuple[str, int]]:
    """
    Find the most common words in a collection of texts.

    Args:
        texts: A list or column of text values.
        top_n: Number of most common words to return.

    Returns:
        A list of tuples like [("word", count), ...].
    """
    all_words = []

    for text in texts:
        tokens = tokenize_text(text)
        all_words.extend(tokens)

    word_counter = Counter(all_words)
    return word_counter.most_common(top_n)


def save_histogram(
    dataframe: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    output_path: str,
    bins: int = 40,
) -> None:
    """
    Create and save a histogram.

    Args:
        dataframe: DataFrame containing the data.
        column: Column to plot.
        title: Plot title.
        xlabel: Label for the x-axis.
        output_path: Where the plot image will be saved.
        bins: Number of histogram bins.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=dataframe, x=column, bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
