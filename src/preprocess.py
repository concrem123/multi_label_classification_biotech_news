import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizer

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
    )

    non_label_cols = ['Title','Content','Target Organization','Text']

    label_columns = [col for col in train_df.columns if col not in non_label_cols]

    # Create a new DataFrame containing only the selected label columns
    df_labels_train = train_df[label_columns]
    df_labels_test = test_df[label_columns]

    unique_list = []
    for i, label in enumerate(label_columns):
        unique_list.append(all_data[label].unique())

    unique_values = list(set(list for sublist in unique_list for list in sublist))

    mapping_values = {}
    for i in range(len(unique_values)):
        mapping_values.update({unique_values[i]: i})

    for label in label_columns:
        train_df[label] = train_df[label].map(mapping_values)
        test_df[label] = test_df[label].map(mapping_values)
        eval_df[label] = eval_df[label].map(mapping_values)

    dataset_dict = DatasetDict({
    "train": Dataset.from_pandas(train_df, preserve_index=False),
    "test": Dataset.from_pandas(test_df, preserve_index=False),
    "eval": Dataset.from_pandas(eval_df, preserve_index=False),
    })
    
    tokenizer = DistilBertTokenizer.from_pretrained(cfg["model"]["name"], do_lower_case=True)
    

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
    encoded_dataset = encoded_with_text.remove_columns([cfg["data"]["text_column"]],'Title', 'Content', 'Target Organization')
    encoded_dataset.save_to_disk(str(processed_path))


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "config/config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    preprocess_and_save(cfg)