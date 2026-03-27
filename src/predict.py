from model import predict_sentiment
import yaml
from pathlib import Path    


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "config/config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    sample_text = """
    
    hbibihihhjkbkhjbkubkjbkjb 
    
    """

    predict_sentiment(sample_text, cfg)