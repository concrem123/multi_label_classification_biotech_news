import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path
from datasets import load_dataset, DatasetDict

def preprocess_and_save(cfg):
    train_df = pd.read_csv(
        os.path.join(os.getcwd(), "data", "raw", "train.csv")
    )

    test_df = pd.read_csv(
        os.path.join(os.getcwd(), "data", "raw", "test.csv")
    )

    all_data = pd.concat([train_df, test_df], ignore_index=True)

    train_df['Text'] = "Title: "+ train_df['Title'] + "Content: " +train_df['Content'] + "Target Organization: " + train_df['Target Organization']
    test_df['Text'] = "Title: "+ test_df['Title'] + "Content: " +test_df['Content'] + "Target Organization: " + test_df['Target Organization']

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

    # Convert the label columns to lists for each row
    labels_list_train = df_labels_train.values.tolist()
    labels_list_test = df_labels_test.values.tolist()

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
        "train": train_df,
        "test": test_df,
        "eval": eval_df})

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "configs/configs.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    preprocess_and_save(cfg)