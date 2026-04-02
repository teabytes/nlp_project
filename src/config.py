"""
This file gives the whole project one central place for:
paths
model name
task names
layer settings
output locations
"""

from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Results directories
RESULTS_DIR = ROOT_DIR / "results"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

# Default transformer model
MODEL_NAME = "bert-base-uncased"

# Tasks
TASKS = ["sst2", "length", "tense"]

# Layer settings
USE_CLS_TOKEN = True
USE_MEAN_POOLING = False

RANDOM_SEED = 42

# Probe settings
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Sentence length buckets
LENGTH_BINS = [0, 8, 15, 10**9]
LENGTH_LABELS = ["short", "medium", "long"]

# Tense labels
TENSE_LABELS = ["present", "past"]

# Output file names
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "layer_embeddings.pkl"
METRICS_FILE = METRICS_DIR / "probe_metrics.json"

# Ensure output directories exist
def ensure_dirs():
    for path in [RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, METRICS_DIR, PLOTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)