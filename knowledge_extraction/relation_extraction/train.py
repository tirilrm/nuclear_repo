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

# Model for NER 
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
    
    #################################
    #### DocRED helper functions ####
    #################################

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
    
    def make_triplets(self, vertexSet, labels):
        '''
        Constructs triplets from vertex set and labels.

        Args:
            vertexSet (list): List of entities with their names and types.
            labels (dict): Dictionary containing 'head', 'tail', 'relation_id', and 'relation_text'.

        Returns:
            list: List of triplets of format [[head(s)], [relation_id, relation_text], [tail(s)]].
                - `head` and `tail` contain lists of one or more entities (len > 1 in case of synonyms, e.g., Swedish and Sweden).
                - `relation` contains relation_id and corresponding relation_text.
        '''

        names = []
        types = []
        triplets = []

        head = labels['head'] # contains vertexSet indices of heads
        tail = labels['tail'] # contains vertexSet indices of tails
        relation_ids = labels['relation_id']
        relation_texts = labels['relation_text']

        if not len(head) == len(tail) == len(relation_texts) == len(relation_ids):
            raise ValueError("Labels are not unform length")

        # Get names and types from vertexSet
        for entities in vertexSet:
            sub_names = [entity['name'] for entity in entities]
            sub_types = [entity['type'] for entity in entities]
            names.append(sub_names)
            types.append(sub_types)

        # Construct triplets
        for i in range(len(head)):
            head_index = head[i]
            tail_index = tail[i]
            relation_id = relation_ids[i]
            relation_text = relation_texts[i]

            head_entities = names[head_index]
            tail_entities = names[tail_index]
            relation = [relation_id, relation_text]
            triplets.append([head_entities, relation, tail_entities])
        
        return triplets
    
    #######################################
    #### (Distil)BERT helper functions ####
    #######################################

    def merge_result(self, entities):
        '''
        Merges consecutive entities output from the NER model into single entities.

        This function combines consecutive sub-word tokens or consecutive words 
        that belong to the same named entity type into a single entity. The function 
        works specifically with models 'dslim/bert-base-NER' and 'dslim/distilbert-NER'.

        Args:
            entities (list): List of entity dictionaries. Each dictionary contains 
                            'word', 'start', 'end', 'entity', and 'score' keys.

        Returns:
            list: List of merged entity dictionaries.
        '''
        merged_entities = []
        current = None

        if self.model_name not in ['dslim/bert-base-NER', 'dslim/distilbert-NER']:
            raise ValueError('NER model not compatible.')

        for entity in entities:
            if current == None:
                current = entity
            else:
                if entity['word'].startswith('##') and entity['start'] == current['end']:
                    current['word'] += entity['word'][2:]
                    current['end'] = entity['end']
                    current['score'] = min(current['score'], entity['score'])
                elif entity['start'] == current['end'] and entity['entity'][2:] == current['entity'][2:]:
                    current['word'] += entity['word']
                    current['end'] = entity['end']
                    current['score'] = min(current['score'], entity['score'])
                elif entity['start'] + 1 == current['end'] and entity['entity'][2:] == current['entity'][2:]:
                    current['word'] += ' ' + entity['word']
                    current['end'] = entity['end']
                    current['score'] = min(current['score'], entity['score'])
                else:
                    merged_entities.append(current)
                    current = entity
        if current is not None:
            merged_entities.append(current)
        
        return merged_entities

    def combine_entities(self, entities):
        '''
        Combines entities that are split across multiple tokens, handling cases where entities span multiple tokens 
        or are separated by hyphens.

        Args:
            entities (list): List of entity dictionaries, where each dictionary contains 'entity', 'score', 'start', 'end', and 'word'.

        Returns:
            list: List of combined entity dictionaries, each containing a merged entity with its type, score, start, end, and word.

        The function processes entities by iterating through them and merging tokens that are part of the same entity.
        It considers entities that start with 'B-' (beginning) and merges subsequent tokens marked with 'I-' (inside).
        It also handles hyphens by merging tokens before and after the hyphen as part of the same entity.
        The combined entity retains the minimum score, earliest start position, and latest end position from the merged tokens.
        '''
        
        combined_entities = []
        i = 0
        while i < len(entities):
            current_entity = entities[i]
            if current_entity['entity'].startswith('B-'):
                entity_type = current_entity['entity'][2:]
                combined_entity = {
                    'entity': entity_type,
                    'score': current_entity['score'],
                    'start': current_entity['start'],
                    'end': current_entity['end'],
                    'word': current_entity['word']
                }
                j = i + 1
                while j < len(entities):
                    if entities[j]['word'] == '-':
                        combined_entity['word'] += entities[j]['word']
                        combined_entity['end'] = entities[j]['end']
                        combined_entity['score'] = min(combined_entity['score'], entities[j]['score'])
                        j += 1
                    elif entities[j]['entity'] == f'I-{entity_type}' and (entities[j]['start'] == combined_entity['end'] + 1):
                        combined_entity['word'] += ' ' + entities[j]['word']
                        combined_entity['end'] = entities[j]['end']
                        combined_entity['score'] = min(combined_entity['score'], entities[j]['score'])
                        j += 1
                    elif (entities[j-1]['word'] == '-'):
                        combined_entity['word'] += entities[j]['word']
                        combined_entity['end'] = entities[j]['end']
                        combined_entity['score'] = min(combined_entity['score'], entities[j]['score'])
                        j += 1
                    else:
                        break
                combined_entities.append(combined_entity)
                i = j
            else:
                i += 1
        return combined_entities
    
    def extract_entities(self, sents):
        '''
        Extracts and combines entities from a list of sentences using the NER pipeline and merging functions.

        Args:
            sents (list): List of sentences (strings) to process for named entity recognition.

        Returns:
            list: List of combined entity dictionaries, each containing an entity type, score, start, end, and word.
        '''
        entities = []
        for sent in sents:
            ner_result = self.ner_pipeline(sent)
            merged_result = self.merge_result(ner_result)
            joined_result = self.combine_entities(merged_result)
            entities.extend(joined_result)
        
        return entities
    
    def tag_sents(self, sents, entities):
        '''
        Tags sentences with B-<entity type> at the start and E-<entity type> at the end of each identified entity.

        Args:
            sents (list): List of sentences (strings) to process for named entity tagging.
            entities (list): List of lists containing entities for each sentence.

        Returns:
            list: List of tagged sentences.
        '''
        tagged_sents = []
        for i, sent in enumerate(sents):
            tagged_sent = sent
            offset = 0
            for entity in entities[i]:
                start = entity['start'] + offset
                end = entity['end'] + offset
                word = entity['word']
                b_tag = '<B-' + entity['entity'] + '> '
                e_tag = ' <E-' + entity['entity'] + '>'
                tagged_sent = tagged_sent[:start] + b_tag + word + e_tag + tagged_sent[end:]
                offset += len(b_tag) + len(e_tag)
            tagged_sents.append(tagged_sent)

        return tagged_sents

model = RelationExtractorBRNN(input_size, hidden_size, num_layers, num_classes, model_name)