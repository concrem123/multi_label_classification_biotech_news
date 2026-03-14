import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizer
import numpy as np
import random

def process_labels(df, labels_list):
    labels = pd.Series(labels_list, index=df.index)
    dummies = (
        labels.explode()
        .dropna()
        .pipe(pd.get_dummies)
        .groupby(level=0)
        .max()
        .astype(int)
    )
    df[dummies.columns] = dummies
    return df

def preprocess_and_save(cfg):
    train_df = pd.read_csv(
        os.path.join(os.getcwd(), "data", "raw", "train.csv")
    )

    test_df = pd.read_csv(
        os.path.join(os.getcwd(), "data", "raw", "test.csv")
    )

    all_data = pd.concat([train_df, test_df], ignore_index=True)

    for df in [train_df, test_df]:
        df["Title"] = df["Title"].fillna("").astype(str)
        df["Content"] = df["Content"].fillna("").astype(str)
        df["Target Organization"] = df["Target Organization"].fillna("").astype(str)

        df["Text"] = (
            "Title: " + df["Title"]
            + " Content: " + df["Content"]
            + " Target Organization: " + df["Target Organization"]
        )

    eval_split = cfg['data']['eval_split']
    train_df, eval_df = train_test_split(
        train_df,
        test_size=eval_split,
        random_state=42
    )
    non_label_cols = ['Title','Content','Target Organization','Text']

    label_columns = [col for col in train_df.columns if col not in non_label_cols]

    # Create a new DataFrame containing only the selected label columns
    df_labels_train = train_df[label_columns]
    df_labels_test = test_df[label_columns]
    df_eval = eval_df[label_columns]

    # Convert the label columns to lists for each row
    labels_list_train = df_labels_train.values.tolist()
    labels_list_test = df_labels_test.values.tolist()
    labels_list_eval = df_eval.values.tolist()

    unique_list = []
    for i, label in enumerate(label_columns):
        unique_list.append(all_data[label].unique())

    unique_values = list(set(list for sublist in unique_list for list in sublist))

    unique_values.remove(np.nan)

    for name in unique_values:
        train_df[name] = 0
        test_df[name] = 0
        eval_df[name] = 0

    train_df = process_labels(train_df, labels_list_train)
    test_df = process_labels(test_df, labels_list_test)
    eval_df = process_labels(eval_df, labels_list_eval)

    for df in [train_df, test_df, eval_df]:
        df[unique_values] = df[unique_values].astype("float32")
        df["labels"] = df[unique_values].values.tolist()
        


    dataset_dict = DatasetDict({
    "train": Dataset.from_pandas(train_df, preserve_index=False),
    "test": Dataset.from_pandas(test_df, preserve_index=False),
    "eval": Dataset.from_pandas(eval_df, preserve_index=False),
    })
    
    tokenizer = DistilBertTokenizer.from_pretrained(
        cfg["model"]["name"],
          do_lower_case=True,
          problem_type="multi_label_classification")
    

    # Tokenization function
    def preprocess_function(examples):
        return tokenizer(
            examples[cfg["data"]["text_column"]],
            truncation=True,
            padding="max_length",
            max_length=cfg["model"]["max_length"],
        )

    # Apply tokenization to all splits
    encoded_with_text = dataset_dict.map(preprocess_function, batched=True, desc="Tokenizing")

    # Resolve project root and dataset paths
    ROOT = Path(__file__).resolve().parent.parent
    processed_path = (ROOT / cfg["data"]["processed_path"]).resolve()
    processed_debug_path = (ROOT / cfg["data"]["processed_path_debug"]).resolve()

    # Create directories if they don't exist
    processed_debug_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # Save dataset with text (for debugging)
    encoded_with_text.save_to_disk(str(processed_debug_path))

    # Remove text column for training 
    encoded_dataset = encoded_with_text.remove_columns([cfg["data"]["text_column"],'Title', 'Content', 'Target Organization',*label_columns])
    encoded_dataset.save_to_disk(str(processed_path))


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "config/config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    preprocess_and_save(cfg)