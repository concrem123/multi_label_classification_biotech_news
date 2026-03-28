from pathlib import Path
from datasets import load_from_disk
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
config_path = ROOT / "config/config.yaml"

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)


# Resolve processed dataset path relative to project root
ROOT = Path(__file__).resolve().parent.parent
processed_path = (ROOT / cfg["data"]["processed_path"]).resolve()  
dataset = load_from_disk(str(processed_path))
dataset.set_format(
type="torch",
columns=["input_ids", "attention_mask", "labels"]
)
tokenized_train_dataset = dataset["train"]
tokenized_eval_dataset = dataset["eval"]


print (tokenized_train_dataset[0])