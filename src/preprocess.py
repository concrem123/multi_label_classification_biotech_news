import pandas as pd
import numpy as np
import torch

from datasets import load_dataset
    
dataset = load_dataset('knowledgator/events_classification_biotech') 

print(dataset)  