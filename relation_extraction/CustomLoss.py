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
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def similarity(self, a, b):
        return self.cosine_similarity(a, b).item()

    def compute_instance_loss(self, pair_embeddings, predictions, triplet_embeddings):
        '''
        For pred in predictions:
        Find the 'best matching pair' among triplets
        If match > threshold check if predicted relation == gold relation
        Else: check if predicted relation == '0'
        '''
        instance_loss = 0

        EMBEDDING_DIM = 768

        for i in range(len(predictions)):
            prediction = predictions[i] # Tensor of shape [1, num_classes]
            pair_embedding = pair_embeddings[i]
            pred_head = pair_embedding[:EMBEDDING_DIM] # e1_emb
            pred_tail = pair_embedding[EMBEDDING_DIM:2*EMBEDDING_DIM] # e2_emb
            
            best_similarity = 0
            best_triplet = None
            
            for triplet in triplet_embeddings:
                gold_head = triplet[:EMBEDDING_DIM]
                gold_tail = triplet[EMBEDDING_DIM+1:2*EMBEDDING_DIM+1]

                head_similarity = self.similarity(pred_head, gold_head)
                tail_similarity = self.similarity(pred_tail, gold_tail)

                if head_similarity > self.threshold and tail_similarity > self.threshold:
                    avg_similarity = (head_similarity + tail_similarity) / 2

                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_triplet = triplet
            
            if best_triplet is not None:
                gold_relation = best_triplet[EMBEDDING_DIM:EMBEDDING_DIM+1]
                target = torch.tensor(int(gold_relation), dtype=torch.long, device=predictions.device)
            else:
                target = torch.tensor(0, dtype=torch.long, device=predictions.device) # Default relation is '0'
            
            prediction = prediction.unsqueeze(0)
            loss = self.cross_entropy(prediction, target.unsqueeze(0))
            instance_loss += loss
            
        avg_instance_loss = instance_loss / len(predictions)
        return avg_instance_loss
    
    def forward(self, batch_entity_pairs, batch_predictions, batch_triplets):
        batch_loss = 0
        for entity_pairs, predicitons, triplets in zip(batch_entity_pairs, batch_predictions, batch_triplets):
            instance_loss = self.compute_instance_loss(entity_pairs, predicitons, triplets)
            batch_loss += instance_loss
        
        avg_loss = batch_loss / len(batch_predictions)
        return avg_loss