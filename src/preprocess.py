from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizer


# ---------------------------
# Label normalization
# ---------------------------
def normalize_label(label):
    if pd.isna(label):
        return None

    label = str(label).strip().lower()

    # unify variants
    if label == "new initiatives or programs":
        return "new initiatives & programs"
    if label == "partnerships & alliances":
        return "alliance & partnership"

    return label


# ---------------------------
# Convert row → multi-hot vector
# ---------------------------
def build_label_matrix(df, label_columns, label_list):
    df_labels = df[label_columns].fillna("")

    all_labels = []
    for _, row in df_labels.iterrows():
        labels = [normalize_label(v) for v in row if v != ""]
        labels = [l for l in labels if l is not None]
        all_labels.append(labels)

    # Create multi-hot encoding
    label_to_idx = {label: i for i, label in enumerate(label_list)}

    multi_hot = []
    for labels in all_labels:
        vec = np.zeros(len(label_list), dtype=np.float32)
        for l in labels:
            if l in label_to_idx:
                vec[label_to_idx[l]] = 1.0
        multi_hot.append(vec.tolist())

    df["labels"] = multi_hot
    return df


# ---------------------------
# Main preprocessing
# ---------------------------
def preprocess_and_save(cfg):
    ROOT = Path(__file__).resolve().parent.parent

    # Load data
    train_df = pd.read_csv(ROOT / "data/raw/train.csv")
    test_df = pd.read_csv(ROOT / "data/raw/test.csv")

    # Fill text columns
    for df in [train_df, test_df]:
        df["Title"] = df["Title"].fillna("").astype(str)
        df["Content"] = df["Content"].fillna("").astype(str)
        df["Target Organization"] = df["Target Organization"].fillna("").astype(str)

        df["Text"] = (
            "Title: " + df["Title"]
            + " Content: " + df["Content"]
            + " Target Organization: " + df["Target Organization"]
        )

    # Split train → train/eval
    train_df, eval_df = train_test_split(
        train_df,
        test_size=cfg["data"]["eval_split"],
        random_state=42
    )

    # Identify label columns
    non_label_cols = ['Title', 'Content', 'Target Organization', 'Text']
    label_columns = [col for col in train_df.columns if col not in non_label_cols]

    # ---------------------------
    # Build global label list (sorted!)
    # ---------------------------
    all_labels_raw = pd.concat([
        train_df[label_columns],
        test_df[label_columns],
        eval_df[label_columns]
    ])

    unique_labels = set()

    for col in label_columns:
        unique_labels.update(
            normalize_label(v) for v in all_labels_raw[col].dropna().unique()
        )

    unique_labels.discard(None)

    # sorted for consistency
    label_list = sorted(unique_labels)

    print(f"\nFinal label set ({len(label_list)}):")
    print(label_list)

    # ---------------------------
    # Build multi-hot labels
    # ---------------------------
    train_df = build_label_matrix(train_df, label_columns, label_list)
    test_df = build_label_matrix(test_df, label_columns, label_list)
    eval_df = build_label_matrix(eval_df, label_columns, label_list)

    # ---------------------------
    # Convert to HF datasets
    # ---------------------------
    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False),
        "eval": Dataset.from_pandas(eval_df, preserve_index=False),
    })

    # ---------------------------
    # Tokenization
    # ---------------------------
    tokenizer = DistilBertTokenizer.from_pretrained(cfg["model"]["name"])

    def preprocess_function(examples):
        return tokenizer(
            examples[cfg["data"]["text_column"]],
            truncation=True,
            padding="max_length",
            max_length=cfg["model"]["max_length"],
        )

    dataset_dict = dataset_dict.map(preprocess_function, batched=True)

    # ---------------------------
    # Keep ONLY required columns
    # ---------------------------
    keep_cols = ["input_ids", "attention_mask", "labels"]

    dataset_dict = dataset_dict.remove_columns([
        col for col in dataset_dict["train"].column_names if col not in keep_cols
    ])

    # ---------------------------
    # Save
    # ---------------------------
    processed_path = (ROOT / cfg["data"]["processed_path"]).resolve()
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_dict.save_to_disk(str(processed_path))

    print("\n Saved processed dataset")
    print(dataset_dict)


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "config/config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    preprocess_and_save(cfg)