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