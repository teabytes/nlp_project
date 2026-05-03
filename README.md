## NLP Layer Probing

This repository contains code for probing which BERT layers encode different sentence properties. It supports three tasks: SST-2 sentiment classification, sentence length classification, and heuristic tense classification.

## Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run downloads the GLUE SST-2 dataset and the `bert-base-uncased` model from Hugging Face, so an internet connection is required.

## Device/System Used

The code was run and checked on:

- macOS Darwin 25.3.0
- Apple Silicon ARM64
- Python 3.12.7
- CPU execution by default

The embedding script automatically uses CUDA if a CUDA GPU is available through PyTorch.

## Running the Code

Create the output directories:

```bash
python -c "from src.config import ensure_dirs; ensure_dirs()"
```

Run SST-2 sentiment probing:

```bash
python -m src.extract_embeddings --task sst2 --max_samples 2000 --batch_size 16 --max_length 128 --use_cls_token
python -m src.train_probe --embeddings_path results/embeddings/sst2_layer_embeddings.pkl --task_name sst2
python -m src.plot_results --metrics_path results/metrics/sst2_layerwise_metrics.csv --task_name sst2
```

Run sentence length probing:

```bash
python -m src.extract_embeddings --task length --max_samples 2000 --batch_size 16 --max_length 128 --use_cls_token
python -m src.train_probe --embeddings_path results/embeddings/length_layer_embeddings.pkl --task_name length
python -m src.plot_results --metrics_path results/metrics/length_layerwise_metrics.csv --task_name length
```

Run tense probing:

```bash
python -m src.extract_embeddings --task tense --max_samples 2000 --batch_size 16 --max_length 128 --use_cls_token
python -m src.train_probe --embeddings_path results/embeddings/tense_layer_embeddings.pkl --task_name tense
python -m src.plot_results --metrics_path results/metrics/tense_layerwise_metrics.csv --task_name tense
```

## How Results Are Generated

For each task, `src/extract_embeddings.py` loads SST-2 sentences, creates the task labels, runs frozen `bert-base-uncased`, and saves layer-wise embeddings to `results/embeddings/<task>_layer_embeddings.pkl`.

Then `src/train_probe.py` trains one logistic regression probe per BERT layer using fixed random splits. It saves metrics to:

```text
results/metrics/<task>_layerwise_metrics.csv
results/metrics/<task>_layerwise_metrics.json
```

Finally, `src/plot_results.py` reads the metrics CSV and saves the layer-wise accuracy/F1 plot to:

```text
results/plots/<task>_layerwise_plot.png
```

The `data/` and `results/` folders are generated when the scripts run.
