import numpy as np
from sklearn.metrics import classification_report
from datasets import load_from_disk
from transformers import Trainer, DistilBertForSequenceClassification
from pathlib import Path
import yaml
from transformers import TrainingArguments
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch


def evaluate(cfg):

    # Resolve processed dataset path relative to project root
    ROOT = Path(__file__).resolve().parent.parent
    processed_path = (ROOT / cfg["data"]["processed_path"]).resolve()

    dataset = load_from_disk(str(processed_path))

    tokenized_test_dataset = dataset["test"]

    # Resolve output_dir relative to project root
    output_dir = (ROOT / cfg["training"]["bert_peft_trainer"]).resolve()

    base_model = DistilBertForSequenceClassification.from_pretrained(
    cfg["model"]["name"],
    num_labels=cfg["model"]["num_labels"],
    problem_type="multi_label_classification",
    )

    # Load the final model checkpoint (from QLoRA training)
    adapter_path = str(output_dir / "checkpoint-468")
    
    # Load the PEFT model and merge it with the base model
    model = PeftModel.from_pretrained(base_model, adapter_path)

    merged_model = model.merge_and_unload()

    training_args = TrainingArguments(
        output_dir=str(output_dir / "eval"),
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        fp16=False,
        )
    
    
    trainer = Trainer(
        model=merged_model,
        args=training_args,
    )


    # Predictions
    predictions = trainer.predict(tokenized_test_dataset)

    logits = predictions.predictions
    labels = predictions.label_ids

    #  Multi-label conversion 
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)

    # class names
    class_names = cfg["model"].get("class_names", None)

    print(
        classification_report(
            labels,
            preds,
            target_names=class_names,
            zero_division=0
        )
    )

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "config/config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg)