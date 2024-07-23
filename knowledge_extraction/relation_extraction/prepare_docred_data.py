from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import load_dataset
import pandas as pd
from _RE import join_text
import pickle

model_name = 'dslim/distilbert-NER'
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
dataset = load_dataset('docred', trust_remote_code=True)

from prepare_articles_data import identify_entities, create_entity_pairs, get_keywords, move_to_root

ner_pipeline = pipeline('ner', model=ner_model, tokenizer=tokenizer)

def make_paragraphs(sents):
    paragraphs = []
    for word_list in sents:
        paragraph = ' '.join(word_list)
        paragraphs.append(paragraph)
    return paragraphs

datasets = [
    'validation',
    'test',
    'train_annotated',
    'train_distant'
]

custom_keywords = get_keywords()
train_test_data = {}

for datatype in datasets:
    print('Now doing:', datatype)
    ds = pd.DataFrame(dataset[datatype])
    context_and_pairs = []
    length = len(ds)

    for i in range(length):
        print(f"{(i/length)*100:.3f}%")
        elems = {}
        sents = ds.iloc[i]['sents']
        paragraphs = make_paragraphs(sents)
        entities = identify_entities(paragraphs, custom_keywords)
        pairs = create_entity_pairs(entities)
        elems['context'] = join_text(sents, fancy=False)
        elems['pairs'] = pairs
        context_and_pairs.append(elems)
    
    train_test_data[datatype] = context_and_pairs

for e in context_and_pairs:
    print(e['context'])
    for p in e['pairs']:
        print(p)

move_to_root()
with open('knowledge_extraction/relation_extraction/data/docred_context_and_pairs.pkl', 'wb') as file:
    pickle.dump(context_and_pairs, file)