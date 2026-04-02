import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.config import METRICS_DIR, PLOTS_DIR, RANDOM_SEED, ensure_dirs


def load_embeddings(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def train_layer_probe(X_train, y_train, X_val, y_val):
    """
    Train a simple logistic regression probe.
    """
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_SEED,
            solver="lbfgs"
        )
    )
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred, average="macro")

    return clf, acc, f1


def evaluate_layerwise_embeddings(embeddings, labels):
    """
    Train a separate probe for each layer.
    embeddings: [num_samples, num_layers, hidden_dim]
    """
    num_samples, num_layers, hidden_dim = embeddings.shape
    results = []

    # Make one split for all layers so comparison is fair
    idx = np.arange(num_samples)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=labels[train_idx]
    )

    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    for layer in range(num_layers):
        X = embeddings[:, layer, :]

        X_train = X[train_idx]
        X_val = X[val_idx]
        X_test = X[test_idx]

        clf, val_acc, val_f1 = train_layer_probe(X_train, y_train, X_val, y_val)

        test_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average="macro")

        results.append({
            "layer": layer,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
        })

        print(
            f"Layer {layer:02d} | "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
            f"test_acc={test_acc:.4f} test_f1={test_f1:.4f}"
        )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="results/embeddings/sst2_layer_embeddings.pkl"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="sst2"
    )
    args = parser.parse_args()

    ensure_dirs()

    embeddings_path = Path(args.embeddings_path)
    data = load_embeddings(embeddings_path)

    embeddings = np.array(data["embeddings"])
    labels = np.array(data["labels"])

    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded labels: {labels.shape}")

    results_df = evaluate_layerwise_embeddings(embeddings, labels)

    metrics_path = METRICS_DIR / f"{args.task_name}_layerwise_metrics.csv"
    results_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to: {metrics_path}")

    json_path = METRICS_DIR / f"{args.task_name}_layerwise_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results_df.to_dict(orient="records"), f, indent=2)
    print(f"Saved json to: {json_path}")


if __name__ == "__main__":
    main()