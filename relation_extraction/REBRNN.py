import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, DistilBertModel

from difflib import SequenceMatcher
import pickle
import json
import pandas as pd
import numpy as np
import Levenshtein
import re
import time
import os

'''
NOTES
- hidden_size must be multiplied by 2 since it's bidirectional; one layer going forward and one going backward
'''

'''
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model for NER 
model_name = 'dslim/distilbert-NER'

# Custom keywords
try:
    with open('keyword_matching/directory.pkl', 'rb') as file:
        keywords = pickle.load(file)
except FileNotFoundError or FileExistsError:
    with open('directory.pkl', 'rb') as file:
        keywords = pickle.load(file)

# Hyperparameters
input_size = 512
num_layers = 4          # may require tuning
hidden_size = 256       # may require tuning
num_classes = 97        # 96 different relations plus '0' for no relation
learning_rate = 0.001   # may require tuning
batch_size = 64
num_epochs = 5'''

class RelationExtractorBRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, pair_embedding_width, pair_embedding_length, model_name, device, threshold=0.7):
        super(RelationExtractorBRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_name = model_name
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True) #check if batch_first=True is required
        self.fc_lstm = nn.Linear(hidden_size * 2, hidden_size)

        self.pair_transform = nn.Linear(pair_embedding_width, hidden_size)
        self.fc_combined = nn.Linear(hidden_size * 3, num_classes)
        
        self.device = device
        self.threshold = threshold


    def forward(self, text_embeddings, pair_embeddings):
        '''
        text_embeddings.shape: [32, 512, 768]
        pair_embeddings.shape: [32, 3000, 1540]

        Pipeline:
        1) Run text embedding through the bidirectional lstm
        2) Run pairs through a linear layer
        3) Concatenate the weights
        4) Run everything again through a last layer

        forward method returns tensor of shape [32, 3000, 97]
        '''
        
        batch_size = text_embeddings.size(0)
        h0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_size).to(self.device)

        # 1. Run text embedding through lstm
        lstm_out, _ = self.lstm(text_embeddings, (h0, c0))
        lstm_out = lstm_out[:, -1, :] # last hidden state of the lstm output

        # 2. Connect pairs
        pair_transformed = self.pair_transform(pair_embeddings)

        # 3. Combine text and pairs (text is repeated for each pair)
        lstm_out_expanded = lstm_out.unsqueeze(1).expand(-1, pair_transformed.size(1), -1)
        combined = torch.cat((lstm_out_expanded, pair_transformed), dim=2)

        # 4. Join everything through a fc layer
        combined = combined.view(-1, combined.size(-1))
        output = self.fc_combined(combined)
        output = output.view(batch_size, -1, self.num_classes)

        # 5. Return predicted relation for each pair
        output = torch.softmax(output, dim=-1)
        pred_classes = torch.argmax(output, dim=-1)
        pred_classes = pred_classes.view(batch_size, -1)

        print(pred_classes.shape)

        return pred_classes