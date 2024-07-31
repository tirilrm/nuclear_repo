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

    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_size, model_name, device, threshold=0.7):
        super(RelationExtractorBRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.model_name = model_name
        
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True) #check if batch_first=True is required
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
        self.device = device
        self.threshold = threshold

    def forward(self, x):

        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out