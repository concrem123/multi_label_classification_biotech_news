from pathlib import Path
import yaml 
from peft import LoraConfig,LoftQConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DebertaV2ForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_from_disk
from torch.nn import BCEWithLogitsLoss
import numpy as np
import torch

def compute_class_weights(dataset):
    """
    Computes pos_weight for BCEWithLogitsLoss
    pos_weight = (num_negative / num_positive) per class
    """
    labels = np.array(dataset["labels"])  # shape: (N, num_labels)

    pos_counts = labels.sum(axis=0)              # positives per class
    neg_counts = labels.shape[0] - pos_counts    # negatives per class

    # Avoid division by zero
    pos_counts = np.clip(pos_counts, 1, None)

    pos_weight = neg_counts / pos_counts

    return torch.tensor(pos_weight, dtype=torch.float32)

def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs > 0.3).astype(int)

    return {
        "eval_f1_micro": f1_score(labels, predictions, average="micro"),
        "eval_f1_macro": f1_score(labels, predictions, average="macro")
    }

def train(cfg):

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

    model = DebertaV2ForSequenceClassification.from_pretrained(
    cfg["model"]["name"],
    num_labels=cfg["model"]["num_labels"],
    problem_type="multi_label_classification",
    )

    class_weights = compute_class_weights(tokenized_train_dataset)

    model.classifier.loss_fct = BCEWithLogitsLoss(pos_weight=class_weights)

    loftq_config = LoftQConfig(
        loftq_bits=cfg["lora"]["loftq_bits"],
    )

    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha= cfg["lora"]["alpha"],
        bias=cfg["lora"]["bias"],
        target_modules=cfg["lora"]["target_modules"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        task_type=cfg["lora"]["task_type"],
        inference_mode=cfg["lora"]["inference_mode"],
    )

    lora_model = get_peft_model(model, lora_config)

    lora_model.print_trainable_parameters()

    # Resolve output_dir relative to project root
    output_dir = (ROOT / cfg["training"]["bert_peft_trainer"]).resolve()

    training_args = TrainingArguments(
        output_dir=str(output_dir),

       # evaluation & logging
        evaluation_strategy=cfg["training"].get("evaluation_strategy", "epoch"),
        logging_strategy="steps",
        logging_steps=cfg["training"].get("logging_steps", 100),

        # training parameters
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        num_train_epochs=cfg["training"]["epochs"],
        learning_rate=float(cfg["training"]["learning_rate"]),

        # model selection
        load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
        metric_for_best_model=cfg["training"].get("metric_for_best_model"), 
        greater_is_better=True,

        # output & logging
        logging_dir=str(output_dir / "logs"),
        report_to="tensorboard",

        # save strategy
        save_strategy=cfg["training"].get("save_strategy", "epoch"),
        save_total_limit=cfg["training"].get("save_total_limit", 5),

        # GPU friendly
        fp16= False,
    )

    bert_peft_trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_train_dataset, # training dataset requires column input_ids
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
        )
    bert_peft_trainer.train()
    bert_peft_trainer.save_model(str(output_dir / "final"))

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "config/config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)
