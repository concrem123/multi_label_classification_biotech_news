Setup

For Mac

'''bash
python3.11.5 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
'''

For windows 
'''bash
python3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
'''

# multi_label_classification_biotech_news
This model uses a fine tuned pubmed bert model to label biotech news. Each biotech news article can have multiple labels

The below are the original 30 labels from the dataset

| ID | Label                            |
| -- | -------------------------------- |
| 1  | Event Organization               |
| 2  | Other                            |
| 3  | Product Launching & Presentation |
| 4  | Patent Publication               |
| 5  | Investment in Public Company     |
| 6  | Foundation                       |
| 7  | Expanding Industry               |
| 8  | Department Establishment         |
| 9  | Closing                          |
| 10 | Subsidiary Establishment         |
| 11 | Executive Statement              |
| 12 | IPO Exit                         |
| 13 | Service & Product Providing      |
| 14 | Hiring                           |
| 15 | Partnerships & Alliances         |
| 16 | Expanding Geography              |
| 17 | Product Updates                  |
| 18 | Executive Appointment            |
| 19 | M&A                              |
| 20 | New Initiatives & Programs       |
| 21 | Company Description              |
| 22 | Support & Philanthropy           |
| 23 | New Initiatives or Programs      |
| 24 | Funding Round                    |
| 25 | Clinical Trial Sponsorship       |
| 26 | Alliance & Partnership           |
| 27 | Participation in an Event        |
| 28 | Regulatory Approval              |
| 29 | Article Publication              |
