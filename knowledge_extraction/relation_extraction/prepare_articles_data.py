from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import load_dataset
from torch import nn
import os
import json
import torch
import pickle
import re
import pandas as pd

from _RE import merge_result, combine_entities, join_text

model_name = 'dslim/distilbert-NER'
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline('ner', model=ner_model, tokenizer=tokenizer)

def identify_entities(paragraphs, custom_keywords, model_name = 'dslim/distilbert-NER'):
    '''
    Finds all entities in a given paraph using 
    1) The pretrained and finetuned DistilBERT extractor, and
    2) The keyword dictionary

    Returns a list of entities where each entity is indicated by:
    - 'word': the name of the entity (e.g. 'Barack Obama')
    - 'entity': the type of entity (for BERT-based models: one of ORG, LOC, PER, MISC)
    - start and end character index of the word
    '''
    ner_entities = []
    keyword_entities = []
    for paragraph in paragraphs:
        raw = ner_pipeline(paragraph)
        raw_merged = merge_result(raw, model_name)
        ner_results = combine_entities(raw_merged)
        
        # Add standard NER results
        for result in ner_results:
            entity = {
                'word': result['word'],
                'entity': result['entity'],
                'start': result['start'],
                'end': result['end']
            }
            ner_entities.append(entity)
        
        # Add custom keyword results
        for label, terms in custom_keywords.items():
            for term in terms:
                for match in re.finditer(term, paragraph):
                    entity = {
                        'word': term,
                        'entity': label,
                        'start': match.start(),
                        'end': match.end()
                    }
                    keyword_entities.append(entity)

    result = {
        'Context': paragraphs ,
        'NER': ner_entities, 
        'Keywords': keyword_entities
    }
    return result

def create_entity_pairs(entities):
    '''
    `entities` contains separate lists of NER-identified entities and keyword-matched entities
    Returns a set of heterogeneous entity pairs for form (('name', 'entity'), ('name', 'entity))
    For N entities, `create_entity_pairs` returns a list of maximum length N^2-N
    '''
    joined = []
    joined.extend(entities['NER'])
    joined.extend(entities['Keywords'])
    pairs = set()
    
    for i, entity1 in enumerate(joined):
        for j, entity2 in enumerate(joined):
            if i != j:
                # Use a sorted tuple to avoid duplicates where order does not matter
                pair = tuple(sorted((tuple([entity1['word'], entity1['entity']]), tuple([entity2['word'], entity2['entity']]))))
                pairs.add(pair)
                
    return list(pairs)

def move_to_root():
    current_directory = os.path.abspath(os.path.dirname(__file__))
    two_layers_up = os.path.dirname(os.path.dirname(current_directory))
    os.chdir(two_layers_up)

def get_articles():
    move_to_root()
    file = open('scraping/articles/filtered_articles.json', 'r')
    articles = file.read()
    articles = json.loads(articles)
    return articles

def get_keywords():
    move_to_root()
    with open('keyword_matching/directory.pkl', 'rb') as file:
        keywords = pickle.load(file)
    return keywords

custom_keywords = get_keywords()

def make_articles_data():
    articles = get_articles()
    context_and_pairs = []
    length = len(articles)
    print(length)
    for i, article in enumerate(articles):
        print(f"{(i/length)*100:.3f}%")
        elems = {}

        context = article['text']
        entities = identify_entities(context, custom_keywords)
        pairs = create_entity_pairs(entities)

        elems['context'] = article['text']
        elems['pairs'] = pairs
        context_and_pairs.append(elems)

    move_to_root()
    with open('knowledge_extraction/relation_extraction/data/context_and_pairs.pkl', 'wb') as file:
        pickle.dump(context_and_pairs, file)

#data = load_dataset('docred', trust_remote_code=True)
#make_articles_data()