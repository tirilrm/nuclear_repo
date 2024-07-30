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


class CustomDocREDDataset(Dataset):
    def __init__(self, dataset, input_size, batch_size, model_name, custom_keywords, device, shuffle=True, threshold=0.8, length=-1):
        self.input_size = input_size
        self.batch_size = batch_size
        self.model_name = model_name
        self.custom_keywords = custom_keywords
        self.device = device
        self.threshold = threshold
        self.length = length
        
        # Load the models for tokenization and NER
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.distilbert = DistilBertModel.from_pretrained(self.model_name)
        self.ner_pipeline = pipeline('ner', model=self.model_name, tokenizer=self.tokenizer)

        # Define an embedding layer for entity types
        self.entity_type_to_id = {'O': 0, 'PER': 1, 'ORG': 2, 'LOC': 3, 'MISC': 4, 'FUEL': 5, 'FUEL_CYCLE': 6, 'SMR_DESIGN': 7, 'REACTOR': 8, 'SMR': 9, 'POLITICAL': 10} 
        self.entity_type_embedding = nn.Embedding(num_embeddings=len(self.entity_type_to_id), embedding_dim=self.distilbert.config.hidden_size)

        # Relation mapping for output layer
        self.relation_mapping = {
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

        # Get the data via the transformers library
        if dataset not in ['train_annotated', 'train_distant', 'test', 'validation']:
            raise ValueError('Data set does not exist')
        
        data = load_dataset('docred', trust_remote_code=True)
        self.raw_data = pd.DataFrame(data[dataset])
        self.preprocessed_data = self.preprocessor(self.raw_data, length=self.length)
        #self.data_loader = DataLoader(self.preprocessed_data, batch_size=self.batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)
    
    def __len__(self):
        return len(self.preprocessed_data)
    
    def __getitem__(self, idx):
        item = self.preprocessed_data[idx]
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

    #def tokenize_input(self, context):
        if isinstance(context, str):
            context = [context]

        inputs = self.tokenizer(
            context,
            padding='max_length',
            truncation=True,
            max_length=self.input_size,
            return_tensors='pt'
        )

        return inputs
    
    def preprocessor(self, docred_data, length):
        '''
        what we want for training: 
        - embeded sents
        - embeded entitiy pairs
            - embeded entity type layer
        - embeded triplets
        '''
        print('Starting preprocessing...')
        start = time.time()
        if length < 0:
            length = len(docred_data)

        data = []
        for i in range(length):
            if i % 50 == 0:
                print(f"{(i/length)*100:.2f}% finished")
            

            # Create entity pairs and triplets
            sents, vertexSet, labels = self.get_info(docred_data.iloc[i])
            entities = self.extract_entities(sents)
            indexed_entities = self.get_entity_positions(sents, entities)
            indexed_pairs = self.make_pairs(indexed_entities)
            indexed_triplets = self.make_triplets(vertexSet, labels)

            # Get the embeded data
            text_emb, pair_emb, triplet_emb = self.embed_data(sents, indexed_pairs, indexed_triplets)

            data.append({
                'text_embeddings': text_emb,
                'pair_embeddings': pair_emb,
                'triplet_embeddings': triplet_emb
            })

        end = time.time()
        print(f'Preprocessing took {(end-start)/60:.2f} minutes.')
        
        return data
    
    def embed_data(self, sents, indexed_pairs, indexed_triplets):
        full_text = ' '.join(sents)
        inputs = self.tokenizer(
            full_text, 
            return_tensors='pt', 
            padding='max_length', 
            max_length=self.input_size,
            truncation=True
        )

        with torch.no_grad():
            outputs = self.distilbert(**inputs)
            text_embeddings = outputs.last_hidden_state[0].cpu().numpy()

            sentence_offsets = [0]
            for sent in sents:
                sentence_offsets.append(sentence_offsets[-1] + len(sent.split(' ')))
            
            pair_embeddings = []
            for pair in indexed_pairs:
                e1, e2 = pair
                e1_start = sentence_offsets[e1['s_id']] + e1['pos'][0]
                e1_end = sentence_offsets[e1['s_id']] + e1['pos'][1]
                e2_start = sentence_offsets[e2['s_id']] + e2['pos'][0]
                e2_end = sentence_offsets[e2['s_id']] + e2['pos'][1]

                if e1_start >= e1_end or e2_start >= e2_end:
                    print(f"Skipping pair with invalid indices: ({e1_start}, {e1_end}), ({e2_start}, {e2_end})")
                    continue
                
                e1_slice = text_embeddings[e1_start:e1_end]
                e2_slice = text_embeddings[e2_start:e2_end]

                if e1_slice.size == 0 or e2_slice.size == 0:
                    print(f"Empty slice detected: e1_slice size = {e1_slice.size}, e2_slice size = {e2_slice.size}")
                    continue

                e1_emb = np.mean(e1_slice, axis=0)
                e2_emb = np.mean(e2_slice, axis=0)

                entity_type_emb = self.get_entity_type_embedding(e1['entity'], e2['entity']).cpu().numpy()
                pair_emb = np.concatenate((e1_emb, e2_emb, entity_type_emb[0], entity_type_emb[1]), axis=0)
                pair_embeddings.append(pair_emb)
            
            triplet_embeddings = []
            for triplet in indexed_triplets:
                head_embs = [np.mean(text_embeddings[sentence_offsets[h['s_id']] + h['pos'][0]:sentence_offsets[h['s_id']] + h['pos'][1]], axis=0) for h in triplet['head']]
                tail_embs = [np.mean(text_embeddings[sentence_offsets[t['s_id']] + t['pos'][0]:sentence_offsets[t['s_id']] + t['pos'][1]], axis=0) for t in triplet['tail']]
                head_embedding = np.concatenate(head_embs, axis=0)
                tail_embedding = np.concatenate(tail_embs, axis=0)
                relation_id = triplet['relation']['id'] # keep relation id as is
                triplet_embeddings.append((head_embedding, relation_id, tail_embedding))
        
        return text_embeddings, pair_embeddings, triplet_embeddings
    
    def get_entity_type_embedding(self, e1_type, e2_type):
        e1_type_id = torch.tensor([self.entity_type_to_id[e1_type]], dtype=torch.long)
        e2_type_id = torch.tensor([self.entity_type_to_id[e2_type]], dtype=torch.long)

        e1_type_emb = self.entity_type_embedding(e1_type_id).squeeze(0)
        e2_type_emb = self.entity_type_embedding(e2_type_id).squeeze(0)

        entity_type_emb = torch.stack((e1_type_emb, e2_type_emb), dim=1)
        
        return entity_type_emb


    def get_info(self, instance):
        sents_raw = instance['sents']
        sents = [' '.join(sublist) for sublist in sents_raw]
        vertexSet = instance['vertexSet']
        labels = instance['labels']

        return sents, vertexSet, labels

    #def load_data(self, get_distant=False):
        pass
        docred_data = load_dataset('docred', trust_remote_code=True)
        train_annotated = pd.DataFrame(docred_data['train_annotated'])
        train_distant = None
        if get_distant:
            train_distant = pd.DataFrame(docred_data['train_distant'])
        test = pd.DataFrame(docred_data['test'])
        validation = pd.DataFrame(docred_data['validation'])

        return train_annotated, train_distant, test, validation
   
    ###########################################
    #### Triplet Creation Helper Functions ####
    ###########################################

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
        triplets = []

        head = labels['head'] # contains vertexSet indices of heads
        tail = labels['tail'] # contains vertexSet indices of tails
        relation_ids = labels['relation_id']
        relation_texts = labels['relation_text']

        if not len(head) == len(tail) == len(relation_texts) == len(relation_ids):
            raise ValueError("Labels are not unform length")

        # Construct triplets
        for i in range(len(head)):
            head_index = head[i]
            tail_index = tail[i]
            relation_id = relation_ids[i]
            relation_text = relation_texts[i]

            head_entities = vertexSet[head_index]
            tail_entities = vertexSet[tail_index]
            relation_num_id = self.relation_mapping.get(relation_id, 0)

            head_indices = [{'s_id': entity['sent_id'], 'pos': entity['pos']} for entity in head_entities]
            tail_indices = [{'s_id': entity['sent_id'], 'pos': entity['pos']} for entity in tail_entities]

            triplets.append({
                'head': head_indices,
                'relation': {'id': relation_num_id, 'text': relation_text},
                'tail': tail_indices
            })
        return triplets
    
    ###########################################
    #### NER Entity Pairs Helper Functions ####
    ###########################################

    def are_similar(self, e1, e2):
        ratio = Levenshtein.ratio(e1['word'], e2['word'])
        return ratio > self.threshold and e1['entity'] == e2['entity']

    def make_pairs(self, indexed_entities):
        length = len(indexed_entities)
        pairs = set()
        for i in range(length):
            for j in range(length):
                if i != j:
                    e1 = indexed_entities[i]
                    e2 = indexed_entities[j]
                    if not self.are_similar(e1, e2):
                        pair = (
                            (e1['s_id'], tuple(e1['pos']), e1['entity']),
                            (e2['s_id'], tuple(e2['pos']), e2['entity'])
                        )
                        pairs.add(pair)
        pairs_list = [
            (
                {'s_id': pair[0][0], 'pos': list(pair[0][1]), 'entity': pair[0][2]},
                {'s_id': pair[1][0], 'pos': list(pair[1][1]), 'entity': pair[1][2]}
            ) for pair in pairs
        ]
        return pairs_list

    def char_to_word_positions(self, sent, start, end):
        words = sent.split()
        current_char_index = 0
        start_word_index = -1
        end_word_index = -1

        for i, word in enumerate(words):
            word_length = len(word)
            
            # Adjust for leading spaces
            while current_char_index < len(sent) and sent[current_char_index] == ' ':
                current_char_index += 1

            if start_word_index == -1 and current_char_index <= start < current_char_index + word_length:
                start_word_index = i
            
            if current_char_index < end <= current_char_index + word_length:
                end_word_index = i + 1
            
            current_char_index += word_length + 1
            
            if start_word_index != -1 and end_word_index != -1:
                break

        if end_word_index == -1:
            end_word_index = len(words)
        
        if start_word_index == -1 or end_word_index == -1:
            print(f"Sentence: {sent}")
            print(f"Start index: {start}, End index: {end}")
            print(f"Computed start_word_index: {start_word_index}, end_word_index: {end_word_index}")

        return [start_word_index, end_word_index]

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
            if current is None:
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
                    elif entities[j-1]['word'] == '-':
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

        for i, sent in enumerate(sents):
            entities = []

            # NER entities
            ner_result = self.ner_pipeline(sent)
            merged_result = self.merge_result(ner_result)
            joined_result = self.combine_entities(merged_result)

            for result in joined_result:
                entity = {
                    's_id': i,
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
                                's_id': i,
                                'word': match.group(),
                                'entity': label,
                                'start': match.start(),
                                'end': match.end()
                            }
                            entities.append(entity)
            all_entities.extend(entities)
        
        return all_entities
    
    def get_entity_positions(self, sents, entities):
        entity_indices = []

        for entity in entities:
            s_id = entity['s_id']
            start = entity['start']
            end = entity['end']
            entity_indices.append({
                's_id': s_id,
                'pos': self.char_to_word_positions(sents[s_id], start, end),
                'word': entity['word'],
                'entity': entity['entity']
            })
        
        return entity_indices
    
    #def tag_sents(self, sents, entities):
        '''
        Tags sentences with B-<entity type> at the start and E-<entity type> at the end of each identified entity.

        Args:
            sents (list): List of sentences (strings) to process for named entity tagging.
            entities (list): List of lists containing entities for each sentence.

        Returns:
            list: List of tagged sentences.
            string: Tagged sentences joined with a [SEP] marker
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