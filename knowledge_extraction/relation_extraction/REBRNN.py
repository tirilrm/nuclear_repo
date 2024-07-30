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

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def forward(self, x):

        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out
    
    #def set_data_loaders(self, train_preprocessed, val_preprocessed, test_preprocessed):

        train_dataset = RelationExtractionDataset(train_preprocessed)
        val_dataset = RelationExtractionDataset(val_preprocessed)
        test_dataset = RelationExtractionDataset(test_preprocessed)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        pass

    

class CustomLoss(nn.Module):
    def __init__(self, threshold=0.8):
        super(CustomLoss, self).__init__()
        self.threshold = threshold
        self.cross_entropy = nn.CrossEntropyLoss()

    def similarity(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def compute_instance_loss(self, entity_pairs, predictions, triplets):
        instance_loss = 0
        output = []

        for i in range(len(predictions)):
            prediction = predictions[i]
            pred_head, pred_tail = entity_pairs[i]
            
            for triplet in triplets:
                best_similarity = 0
                best_relation = '0'

                gold_heads, gold_relation, gold_tails = triplet
                head_similarity = max(self.similarity(pred_head, gold_head) for gold_head in gold_heads)
                tail_similarity = max(self.similarity(pred_tail, gold_tail) for gold_tail in gold_tails)

                if head_similarity > self.threshold and tail_similarity > self.threshold:
                    avg_similarity = (head_similarity + tail_similarity) / 2

                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_relation = gold_relation
                
            output.append((prediction, best_relation))

        for pred, best_relation in output:
            target = torch.tensor(int(best_relation), dtype=torch.long, device=predictions.device)
            loss = self.cross_entropy(pred.unsqueeze(0), target.unsqueeze(0))
            instance_loss += loss

        return instance_loss / len(output)
    
    def forward(self, batch_entity_pairs, batch_predictions, batch_triplets):
        batch_loss = 0
        for entity_pairs, predicitons, triplets in zip(batch_entity_pairs, batch_predictions, batch_triplets):
            instance_loss = self.compute_instance_loss(entity_pairs, predicitons, triplets)
            batch_loss += instance_loss
        
        return batch_loss / len(batch_predictions)