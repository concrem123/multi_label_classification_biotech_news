import yaml
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from peft import PeftModel

def predict_sentiment(text, cfg):
    ROOT = Path(__file__).resolve().parent.parent
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
    merged_model.eval()    

    tokenizer = DistilBertTokenizer.from_pretrained(
        cfg["model"]["name"],
          do_lower_case=True,
          problem_type="multi_label_classification")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=cfg["model"]["max_length"])

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    #  Multi-label conversion 
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)

    print(preds)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "configs/distilbert.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
