import argparse
import os
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from src.config import (
    MODEL_NAME,
    RANDOM_SEED,
    EMBEDDINGS_DIR,
    ensure_dirs,
)
from src.data_utils import load_sst2, build_length_dataset


def set_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SentenceDataset(Dataset):
    def __init__(self, sentences: List[str], labels: List[int]):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "sentence": self.sentences[idx],
            "label": int(self.labels[idx]),
        }


def collate_batch(batch, tokenizer, max_length: int = 128):
    sentences = [x["sentence"] for x in batch]
    labels = [x["label"] for x in batch]

    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    encoded["labels"] = torch.tensor(labels, dtype=torch.long)
    encoded["sentences"] = sentences
    return encoded


@torch.no_grad()
def extract_layer_embeddings(
    model,
    tokenizer,
    sentences: List[str],
    labels: List[int],
    batch_size: int = 16,
    max_length: int = 128,
    use_cls_token: bool = True,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Extract embeddings from every transformer layer.

    Returns a dictionary with:
      - sentences
      - labels
      - embeddings: list of shape [num_samples, num_layers, hidden_dim]
    """
    dataset = SentenceDataset(sentences, labels)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, max_length=max_length),
    )

    all_sentence_embeddings = []
    all_labels = []
    all_texts = []

    model.eval()

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states
        # hidden_states is a tuple:
        # hidden_states[0] = embeddings output
        # hidden_states[1:] = each transformer layer output

        batch_layer_vectors = []

        for layer_state in hidden_states:
            if use_cls_token:
                # shape: [batch, hidden_dim]
                layer_vec = layer_state[:, 0, :]
            else:
                mask = attention_mask.unsqueeze(-1).float()
                summed = (layer_state * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-9)
                layer_vec = summed / denom

            batch_layer_vectors.append(layer_vec.cpu().numpy())

        # Shape: [num_layers+1, batch_size, hidden_dim]
        batch_layer_vectors = np.stack(batch_layer_vectors, axis=1)

        all_sentence_embeddings.append(batch_layer_vectors)
        all_labels.extend(batch["labels"].cpu().numpy().tolist())
        all_texts.extend(batch["sentences"])

    # Shape: [num_samples, num_layers+1, hidden_dim]
    all_sentence_embeddings = np.concatenate(all_sentence_embeddings, axis=0)

    return {
        "sentences": all_texts,
        "labels": np.array(all_labels),
        "embeddings": all_sentence_embeddings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sst2", choices=["sst2", "length"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--use_cls_token", action="store_true")
    parser.add_argument("--use_mean_pooling", action="store_true")

    args = parser.parse_args()

    ensure_dirs()
    set_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    
    # Load dataset based on task
    if args.task == "sst2":
        df = load_sst2(split="train")

    elif args.task == "length":
        df = load_sst2(split="train")  # reuse SST-2 text
        df = build_length_dataset(df)

    else:
        raise ValueError(f"Unsupported task: {args.task}")  

    if args.max_samples is not None and args.max_samples > 0:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)

    sentences = df["sentence"].tolist()
    labels = df["label"].tolist()

    use_cls_token = True
    if args.use_mean_pooling:
        use_cls_token = False
    elif args.use_cls_token:
        use_cls_token = True

    result = extract_layer_embeddings(
        model=model,
        tokenizer=tokenizer,
        sentences=sentences,
        labels=labels,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_cls_token=use_cls_token,
        device=device,
    )

    out_path = EMBEDDINGS_DIR / f"{args.task}_layer_embeddings.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Saved embeddings to: {out_path}")
    print(f"Embeddings shape: {result['embeddings'].shape}")
    print(f"Labels shape: {result['labels'].shape}")


if __name__ == "__main__":
    main()