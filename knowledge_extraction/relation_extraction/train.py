import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import pipeline, AutoTokenizer

import json
import pandas as pd
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'dslim/distilbert-NER'

# Hyperparameters
input_size = 512
num_layers = 4          # may require tuning
hidden_size = 256       # may require tuning
num_classes = 97        # 96 different relations plus '0' for no relation
learning_rate = 0.001   # may require tuning
batch_size = 64
num_epochs = 5

'''
NOTES
- hidden_size must be multiplied by 2 since it's bidirectional; one layer going forward and one going backward
'''

class RelationExtractorBRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, model_name):
        super(RelationExtractorBRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_name = model_name
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True) #check if batch_first=True is required
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.ner_pipeline = pipeline('ner', model=self.model_name, tokenizer=self.tokenizer)

    def forward(self, x):

        '''
        x is the training data of input size 512
        we need to prepare (and PAD) the DocRED training data for this purpose
        '''
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out
    
    # DocRED helper functions
    def get_data(self, get_distant=False):
        docred_data = load_dataset('docred', trust_remote_code=True)
        train_annotated = pd.DataFrame(docred_data['train_annontated'])
        train_distant = None
        if get_distant:
            train_distant = pd.DataFrame(docred_data['train_distant'])
        test = pd.DataFrame(docred_data['test'])
        validation = pd.DataFrame(docred_data['validation'])

        return train_annotated, train_distant, test, validation
    
    def get_info(self, instance):
        sents_raw = instance['sents']
        sents = [' '.join(sublist) for sublist in sents_raw]
        vertexSet = instance['vertexSet']
        labels = instance['labels']

        return sents, vertexSet, labels
    
    def make_triplets(self):
        pass
    
    def merge_BERT_result(self, entities, name):
        if self.model_name not in ['dslim/bert-base-NER', 'dslim/distilbert-NER']:
            raise ValueError('NER model not compatible.')
        
        merged_entities = []
        current = None
    
    def identify_entities(self, data, length = -1):
        if length == -1:
            length = len(data)
        