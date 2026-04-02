import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.config import LENGTH_BINS, LENGTH_LABELS, RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


# SST-2 (Sentiment)
def load_sst2(split: str = "train") -> pd.DataFrame:
    """
    Loads SST-2 dataset from Hugging Face and returns a pandas DataFrame.
    """
    dataset = load_dataset("glue", "sst2", split=split)

    df = pd.DataFrame({
        "sentence": dataset["sentence"],
        "label": dataset["label"]
    })

    return df


# Sentence length task
def get_sentence_length_labels(sentences: List[str]) -> List[int]:
    """
    Convert sentences into length bucket labels.
    """
    lengths = [len(s.split()) for s in sentences]

    labels = []
    for l in lengths:
        for i in range(len(LENGTH_BINS) - 1):
            if LENGTH_BINS[i] <= l < LENGTH_BINS[i + 1]:
                labels.append(i)
                break

    return labels


def build_length_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a length-based label column to the dataset.
    """
    df = df.copy()
    df["label"] = get_sentence_length_labels(df["sentence"].tolist())
    return df


# Train / Val / Test split
def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into train/val/test.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=df["label"]
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=RANDOM_SEED,
        stratify=train_df["label"]
    )

    return train_df, val_df, test_df




def preview_dataset(df: pd.DataFrame, n: int = 5):
    print(df.head(n))
    print("\nLabel distribution:")
    print(df["label"].value_counts())