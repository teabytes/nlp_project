import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PLOTS_DIR, ensure_dirs


def plot_layerwise_metrics(metrics_path: Path, task_name: str):
    df = pd.read_csv(metrics_path)

    plt.figure(figsize=(10, 6))
    plt.plot(df["layer"], df["test_accuracy"], marker="o", label="Test Accuracy")
    plt.plot(df["layer"], df["test_macro_f1"], marker="o", label="Test Macro F1")

    best_row = df.loc[df["test_accuracy"].idxmax()]
    best_layer = int(best_row["layer"])
    best_acc = best_row["test_accuracy"]

    plt.axvline(best_layer, linestyle="--", alpha=0.5)
    plt.title(f"Layer-wise Probe Performance on {task_name.upper()}")
    plt.xlabel("Transformer Layer")
    plt.ylabel("Score")
    plt.xticks(df["layer"])
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.text(
        best_layer,
        best_acc,
        f"best layer {best_layer}",
        fontsize=9,
        ha="left",
        va="bottom"
    )

    out_path = PLOTS_DIR / f"{task_name}_layerwise_plot.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="results/metrics/sst2_layerwise_metrics.csv"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="sst2"
    )
    args = parser.parse_args()

    ensure_dirs()

    metrics_path = Path(args.metrics_path)
    plot_layerwise_metrics(metrics_path, args.task_name)


if __name__ == "__main__":
    main()