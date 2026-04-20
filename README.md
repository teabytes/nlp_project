## NLP Layer Probing

This project studies how linguistic information is distributed across transformer layers using a layer-wise probing framework. We freeze a pretrained transformer encoder, extract hidden-state embeddings from each layer, and train lightweight classifiers on top of those embeddings to test whether specific linguistic properties are linearly recoverable.

The current implementation includes two completed probing tasks:

- **SST-2 sentiment classification** as a semantic task
- **Sentence length classification** as a surface-level task

The project is designed to support future expansion to additional probing tasks such as tense classification.

## What the project does

The pipeline works as follows:

1. Load a dataset of sentences and labels.
2. Pass each sentence through a frozen pretrained transformer.
3. Extract hidden states from every layer.
4. Pool each layer into a fixed-size sentence embedding.
5. Train a simple probe model for each layer.
6. Compare performance across layers using accuracy and macro F1.
7. Save metrics and plots for analysis.

The main goal is interpretability: to see which layers encode surface-level versus semantic information.

## Repository structure

```text
nlp_project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── results/
│   ├── embeddings/
│   ├── metrics/
│   └── plots/
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_utils.py
    ├── extract_embeddings.py
    ├── train_probe.py
    └── plot_results.py
```

## File guide

### `requirements.txt`
Lists the Python packages needed to run the project.

### `.gitignore`
Prevents temporary files, virtual environments, and generated outputs from being committed.

### `src/config.py`
Central configuration file for paths, model name, task names, and output locations.

### `src/data_utils.py`
Loads SST-2 and builds the sentence-length dataset. Also contains helper functions for splitting and previewing data.

### `src/extract_embeddings.py`
Loads a pretrained transformer, freezes it, extracts hidden states from every layer, and saves the layer-wise sentence embeddings.

### `src/train_probe.py`
Trains a lightweight logistic regression probe for each layer and saves layer-wise evaluation metrics.

### `src/plot_results.py`
Creates line plots showing probe performance across transformer layers.

### `data/`
Stores raw or processed datasets if needed in future experiments.

### `results/embeddings/`
Stores saved hidden-state embeddings produced by the extraction step.

### `results/metrics/`
Stores CSV and JSON files with layer-wise performance metrics.

### `results/plots/`
Stores generated figures for the report.

## Current completed experiments

### SST-2 sentiment probing
This task measures how well each transformer layer supports sentiment classification. In the current results, performance improves across layers and peaks in the final layer, suggesting that higher layers encode more abstract semantic information.

### Sentence length probing
This task measures how well each layer captures surface-level sentence length. In the current results, performance is highest in the early layers and declines in deeper layers, suggesting that surface-form information is emphasized early in the network.

## How to run the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Create output directories:

```bash
python -c "from src.config import ensure_dirs; ensure_dirs(); print('dirs ready')"
```

Extract embeddings for SST-2:

```bash
python -m src.extract_embeddings --task sst2 --max_samples 2000 --batch_size 16 --max_length 128 --use_cls_token
```

Train probes for SST-2:

```bash
python -m src.train_probe --embeddings_path results/embeddings/sst2_layer_embeddings.pkl --task_name sst2
```

Plot SST-2 results:

```bash
python -m src.plot_results --metrics_path results/metrics/sst2_layerwise_metrics.csv --task_name sst2
```

Extract embeddings for length:

```bash
python -m src.extract_embeddings --task length --max_samples 2000 --batch_size 16 --max_length 128 --use_cls_token
```

Train probes for length:

```bash
python -m src.train_probe --embeddings_path results/embeddings/length_layer_embeddings.pkl --task_name length
```

Plot length results:

```bash
python -m src.plot_results --metrics_path results/metrics/length_layerwise_metrics.csv --task_name length
```
Tense:
Extract embeddings:
```bash
python -m src.extract_embeddings --task tense --max_samples 2000 --batch_size 16 --max_length 128 --use_cls_token
```
Train probes:
```bash
python -m src.train_probe --embeddings_path results/embeddings/tense_layer_embeddings.pkl --task_name tense
```
Plot results:
```bash
python -m src.plot_results --metrics_path results/metrics/tense_layerwise_metrics.csv --task_name tense
```

## Expected findings

We expect surface-level tasks such as sentence length to be captured in earlier layers, while semantic tasks such as sentiment should become more separable in deeper layers. The current results already support this trend.

## Possible extensions

- Add a tense classification task
- Compare CLS pooling vs mean pooling
- Try a different pretrained encoder
- Add a more compact probe such as ridge regression or a shallow MLP
- Analyze intermediate-layer trends in more detail
- Add a combined summary figure for the report

## Notes

This project is meant for interpretability, not state-of-the-art accuracy. The probes are intentionally simple so that differences across layers are easy to interpret.
