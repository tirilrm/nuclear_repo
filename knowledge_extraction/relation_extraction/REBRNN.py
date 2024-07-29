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

relation_mapping = {
    'P6': 1, 'P17': 2, 'P19': 3, 'P20': 4, 'P22': 5, 'P25': 6, 'P26': 7, 'P27': 8, 'P30': 9, 'P31': 10,
    'P35': 11, 'P36': 12, 'P37': 13, 'P39': 14, 'P40': 15, 'P50': 16, 'P54': 17, 'P57': 18, 'P58': 19,
    'P69': 20, 'P86': 21, 'P102': 22, 'P108': 23, 'P112': 24, 'P118': 25, 'P123': 26, 'P127': 27, 'P131': 28,
    'P136': 29, 'P137': 30, 'P140': 31, 'P150': 32, 'P155': 33, 'P156': 34, 'P159': 35, 'P161': 36, 'P162': 37,
    'P166': 38, 'P170': 39, 'P171': 40, 'P172': 41, 'P175': 42, 'P176': 43, 'P178': 44, 'P179': 45, 'P190': 46,
    'P194': 47, 'P205': 48, 'P206': 49, 'P241': 50, 'P264': 51, 'P272': 52, 'P276': 53, 'P279': 54, 'P355': 55,
    'P361': 56, 'P364': 57, 'P400': 58, 'P403': 59, 'P449': 60, 'P463': 61, 'P488': 62, 'P495': 63, 'P527': 64,
    'P551': 65, 'P569': 66, 'P570': 67, 'P571': 68, 'P576': 69, 'P577': 70, 'P580': 71, 'P582': 72, 'P585': 73,
    'P607': 74, 'P674': 75, 'P676': 76, 'P706': 77, 'P710': 78, 'P737': 79, 'P740': 80, 'P749': 81, 'P800': 82,
    'P807': 83, 'P840': 84, 'P937': 85, 'P1001': 86, 'P1056': 87, 'P1198': 88, 'P1336': 89, 'P1344': 90, 'P1365': 91,
    'P1366': 92, 'P1376': 93, 'P1412': 94, 'P1441': 95, 'P3373': 96,
    'no_relation': 0 
}

def custom_collate_fn(batch):
    embeddings = torch.stack([item['embeddings'].squeeze(0) for item in batch], dim=0)
    entity_pairs = [item['entity_pairs'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    return {
        'embeddings': embeddings,
        'entity_pairs': entity_pairs,
        'labels': labels
    }

class RelationExtractorBRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_size, model_name, device, custom_keywords=None, threshold=0.7):
        super(RelationExtractorBRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.model_name = model_name
        
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True) #check if batch_first=True is required
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.distilbert = DistilBertModel.from_pretrained(self.model_name)
        self.ner_pipeline = pipeline('ner', model=self.model_name, tokenizer=self.tokenizer)

        self.device = device

        self.custom_keywords = custom_keywords
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
    
    def set_data_loaders(self, train_preprocessed, val_preprocessed, test_preprocessed):

        train_dataset = RelationExtractionDataset(train_preprocessed)
        val_dataset = RelationExtractionDataset(val_preprocessed)
        test_dataset = RelationExtractionDataset(test_preprocessed)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    def tokenize_input(self, context):
        inputs = self.tokenizer(
            context,
            padding='max_length',
            truncation=True,
            max_length=self.input_size,
            return_tensors='pt'
        )

        return inputs
    
    def preprocessor(self, docred_data, length=-1):
        '''
        what we want for training: 
        - tagged sents
        - entity pairs
        - gold triplets

        get the DocRED data using load_data():
        for instance in docred_instances:
            a1) get entities using extract_entities()
            a2) make entity pairs using make_pairs()
            b1) tag sents using tag_sents()
            c1) make triplets using make_triplets()
        
        put in data loader?
        return pairs, sents, triplets for train/test/validation
        '''
        start = time.time()
        if length < 0:
            length = len(docred_data)

        data = {}
        count = 0
        for i in range(length):
            _data = {}
            sents, vertexSet, labels = self.get_info(docred_data.iloc[i])
            entities = self.extract_entities(sents)
            entity_pairs = self.make_pairs(entities)
            _, tagged_text = self.tag_sents(sents, entities)
            triplets = self.make_triplets(vertexSet, labels)
            inputs = self.tokenize_input(tagged_text)
            
            with torch.no_grad():
                embeddings = self.distilbert(**inputs).last_hidden_state.numpy()
            
            _data['entity_pairs'] = entity_pairs
            _data['sents'] = tagged_text
            _data['inputs'] = inputs
            _data['embeddings'] = embeddings.tolist()
            _data['triplets'] = triplets
            data[i] = _data

            padding = inputs['attention_mask'][0].tolist().count(0)
            if padding == 0:
                count += 1

        end = time.time()
        print(f'Preprocessing took {(end-start)/60:.2f} minutes. {count} instances ({(count/length)*100:.2f}%) exceeded max length.')
        
        return data

    ##############################################
    #### Data pre-processing helper functions ####
    ##############################################

    def are_similar(self, e1, e2):
        ratio = Levenshtein.ratio(e1[0], e2[0])
        if ratio > self.threshold and e1[1] == e2[1]:
            return True
        return False

    def make_pairs(self, entities):
        try:
            entities_flattened = [[item['word'], item['entity']] for entity in entities for item in entity]
        except TypeError:
            print('Typeerror: entity list is flat')
            entities_flattened = [[item['word'], item['entity']] for item in entities]
        length = len(entities_flattened)
        pairs = set()
        for i in range(length):
            for j in range(length):
                if i != j:
                    e1 = entities_flattened[i]
                    e2 = entities_flattened[j]
                    pair = (tuple(e1), tuple(e2))
                    if not self.are_similar(e1, e2):
                        pairs.add(pair)
                    else:
                        #print(pair)
                        pass
        return list(pairs)
    
    #################################
    #### DocRED helper functions ####
    #################################

    def load_data(self, get_distant=False):
        docred_data = load_dataset('docred', trust_remote_code=True)
        train_annotated = pd.DataFrame(docred_data['train_annotated'])
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
            relation_num_id = relation_mapping.get(relation_id, 0)
            triplets.append([head_entities, relation_num_id, tail_entities, [relation_id, relation_text]])
        
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
        Extracts and combines entities from a list of sentences using 
         1) NER pipeline and merging functions;
         2) based on keyword watch if keywords given

        Args:
            sents (list): List of sentences (strings) to process for named entity recognition.

        Returns:
            list: List of combined entity dictionaries, each containing an entity type, score, start, end, and word.
        '''
        all_entities = []

        for sent in sents:
            entities = []

            # NER entities
            ner_result = self.ner_pipeline(sent)
            merged_result = self.merge_result(ner_result)
            joined_result = self.combine_entities(merged_result)
            for result in joined_result:
                entity = {
                    'word': result['word'],
                    'entity': result['entity'],
                    'start': result['start'],
                    'end': result['end']
                }
                entities.append(entity)
        
            # Keyword entities
            if self.custom_keywords:
                for label, terms in self.custom_keywords.items():
                    # Precompile the regex patterns
                    term_patterns = [re.compile(re.escape(term)) for term in terms]
                    for pattern in term_patterns:
                        for match in pattern.finditer(sent):
                            entity = {
                                'word': match.group(),
                                'entity': label,
                                'start': match.start(),
                                'end': match.end()
                            }
                            entities.append(entity)
            all_entities.append(entities)
        
        return all_entities
    
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
        tagged_text = " [SEP] ".join(tagged_sents)

        return tagged_sents, tagged_text

class RelationExtractionDataset(Dataset):
    def __init__(self, preprocessed_data):
        self.data = preprocessed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        embeddings = torch.tensor(item['embeddings'], dtype=torch.float32) 
        labels = torch.tensor([triplet[1] for triplet in item['triplets']], dtype=torch.long)
        entity_pairs = item['entity_pairs']
        tagged_sents = item['sents']

        return {
            'embeddings': embeddings,
            'labels': labels,
            'entity_pairs': entity_pairs,
            'tagged_sents': tagged_sents
        }

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