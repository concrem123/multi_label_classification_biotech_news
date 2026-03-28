from model import predict_sentiment
import yaml
from pathlib import Path    


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "config/config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    sample_text = """
    
    'Title: Funeral pre-payment plan provider Safe Hands collapses leaving services at risk Content: A provider of pre-paid funeral plans has collapsed, placing in doubt the arrangements made by 46,000 customers in advance of their deaths and whether they will get their money back. Wakefield-based Safe Hands had been in discussions with the Financial Conduct Authority (FCA) before calling in administrators as the watchdog is due to oversee the currently unregulated funeral plan market from July. However, the firm stopped taking new orders and withdrew its application for FCA authorisation last month. The Joint administrators from FRP Advisory said their appointment "was made by the directors of the company, after a period of severe financial challenge, which has left the business unsustainable in its current form". "The company has ceased to trade insofar as it will not be accepting any new customers but will be assisting current plan holders with contingency funeral planning services with the assistance of a third-party provider." It said that Dignity, the UK listed funeral provider, had agreed to "temporarily provide existing customers with funeral planning services for a period of 14 days". The administrators added that there were insufficient funds to enable refunds to be issued to all plan holders in full. But they expressed hope a buyer could be found. Customer money was held in a trust fund established by Safe Hands. But the administrators told plan holders on Thursday: "Customer instalments appear to have been used by the company to acquire investments, subject to deductions for the costs of administering the funeral plans. "Investments are understood by the Administrators to comprise a combination of equities, bonds, cash, real estate and loans.  "The legal structure of these investments appears to be complicated and needs to be investigated to understand which of these investments can be realised for the benefit of plan holders. "A dedicated UK based customer service team has been established... to assist you with any questions you may have in respect of the administration, the defined role of Dignity, the impact of the administration on your plan and any potential return of the money you have invested," the statement added. "You will receive written communication in the post within the next seven days." Target Organization: FCA'    
    
    """

    pred_labels = predict_sentiment(sample_text, cfg)

