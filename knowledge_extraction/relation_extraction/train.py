from transformers import AutoTokenizer, DistilBertModel, pipeline
from datasets import load_dataset
import torch
import torch.nn as nn

class DistilBertRE(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.model_name = 'dslim/distilbert-NER'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.ner = pipeline('ner', model=self.model_name, tokenizer=self.tokenizer)

        self.lstm = nn.LSTM(self.distilbert.config.hidden_size, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
    
    def get_word_embeddings():
        pass