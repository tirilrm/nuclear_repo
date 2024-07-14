from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
import torch
import torch.nn as nn

class DistilBertRE(nn.Module):

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
        #self.BiLSTM =         